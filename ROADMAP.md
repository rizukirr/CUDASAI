# Roadmap: tiny LLM inference in CUDA

Target: run a GPT-2 small (124M parameters, 12 layers, 768 wide, 12 heads, 1024 context) decoder in fp32 on the RTX 4050 and generate text from a prompt. GPT-2 small is the right first target because its weights are public, it fits in 500 MB, and llm.c ships a plain C reference implementation you can diff against layer by layer.

Every function below follows the same contract: a C-style function over `Cds` arrays, a CPU reference implementation of the same math, and one check that compares the two on small shapes. Build the check before the kernel is fast. A kernel that is fast and wrong is worth nothing.

The stages are in dependency order. Finish a stage, including its checks, before starting the next.

## Stage 0: what exists today

- `cds_new(bytes)`: a root that owns a device arena for data and a host arena for metadata, and holds no data itself.
- `cds_array(root, shape, dim, type, data)`: an array created inside a root, released with it.
- `cds_array_rand(root, shape, dim, type)`: an array inside a root filled with uniform random values, for weights and biases.
- `cds_to_host`, `cds_free`, `cds_count`, `cds_data_at`.
- `cds_linear(x, W, b, y)`: `y = x · W^T + b` through cuBLASLt. Allocates nothing. `main.cu` shows the two-root pattern: parameters in one root, activations in another.

## Stage 1: memory and plumbing

| Function | Does | Check |
|---|---|---|
| `cds_checkpoint(root)`, `cds_restore(root, cp)` | Wrap `arena_checkpoint` and `arena_restore` on both arenas of the activation root. Called once per generated token so activations are reused instead of growing. | `cds_array` after restore returns the same device address. |
| `cds_view(root, src, shape, dim)` | A new `Cds` over existing device memory with a different shape. No copy. Needed for splitting heads and flattening batch dims. | Same `data` pointer, new strides. |
| `cds_copy(dst, src)` | Device to device copy of `capacity` bytes. | Round trip through host matches. |
| `cds_fill(dst, value)` | Fill with a scalar. One thread per element, the first kernel you write yourself. | Host readback. |

Also in this stage: a `CdsContext` holding the cuBLASLt handle, a cuBLASLt workspace of 32 MB, and a CUDA stream. Every kernel launch and every cuBLASLt call takes the stream. Without it you cannot overlap anything later, and retrofitting a stream argument into forty functions is painful.

## Stage 2: elementwise and reduction kernels

These are the kernels where you learn CUDA. Each is one or two kernels. Write the naive one-thread-per-element version first, then the shared-memory reduction for the row-wise ones.

| Function | Does | Shape | Check |
|---|---|---|---|
| `cds_add(a, b, out)` | `out = a + b`. Used for residual connections. | same shape | CPU loop. |
| `cds_scale(a, s, out)` | `out = a * s`. | same shape | CPU loop. |
| `cds_gelu(a, out)` | GELU with the tanh approximation GPT-2 uses. | same shape | CPU loop, tolerance 1e-5. |
| `cds_softmax_rows(a, out)` | Numerically stable softmax over the last dimension: subtract the row max, exp, divide by the row sum. One block per row, warp shuffle for the reductions. | `[rows, cols]` | Rows sum to 1, matches CPU to 1e-5. |
| `cds_layernorm(x, gamma, beta, out, eps)` | Per-row mean and variance, then `gamma * (x - mean) / sqrt(var + eps) + beta`. GPT-2 uses LayerNorm. Llama uses RMSNorm, which drops the mean. | `[rows, d]` | CPU loop, tolerance 1e-4. |
| `cds_argmax_rows(a, out_indices)` | Index of the max in each row. The greedy sampler. | `[rows, vocab]` to `[rows]` | CPU loop. |

The softmax and layernorm reductions are where most people get their first wrong answer from a race condition. Run them under `compute-sanitizer --tool racecheck`.

## Stage 3: matmul variants

| Function | Does | Check |
|---|---|---|
| `cds_matmul(a, b, out, transpose_a, transpose_b)` | General `[M,K] · [K,N]` through cuBLASLt with transpose flags. `cds_linear` becomes a thin wrapper. | CPU triple loop on 5x7x3. |
| `cds_matmul_batched(a, b, out, ...)` | The same, batched over a leading dimension, using the cuBLASLt strided batch attributes. Attention needs `Q · K^T` and `P · V` once per head. | Loop of single matmuls gives the same result. |
| `cds_transpose_heads(x, out, n_heads)` | `[seq, n_heads * head_dim]` to `[n_heads, seq, head_dim]` and back. A permute kernel. | Round trip is the identity. |

Write your own naive and tiled matmul kernels in this stage too, even though cuBLASLt will be the one you ship. Benchmark all three on 768x768x1024. Knowing the gap between your tiled kernel and cuBLASLt is the point of the exercise.

## Stage 4: embeddings and positions

| Function | Does | Check |
|---|---|---|
| `cds_embed(table, token_ids, out)` | Gather rows: `out[i] = table[ids[i]]`. One thread per output element. | CPU gather. |
| `cds_pos_embed_add(x, pos_table, start, out)` | Add the learned position vector for positions `start .. start + seq`. GPT-2 uses learned positions. Llama uses RoPE, a rotation applied to Q and K inside attention, which you would add here if you switch models. | CPU loop. |

The output projection to logits reuses `cds_matmul` with the embedding table transposed. GPT-2 ties those weights.

## Stage 5: attention

Build it in two forms. The full-sequence form first, because it is simpler to check. The cached form second, because inference needs it.

| Function | Does | Check |
|---|---|---|
| `cds_attention(q, k, v, out, n_heads, causal)` | Per head: `scores = q · k^T / sqrt(head_dim)`, mask positions after the diagonal with negative infinity, softmax rows, multiply by `v`, merge heads. Composed from stages 2 and 3. | CPU reference on 2 heads, seq 4, head_dim 3. |
| `cds_kv_cache_new(root, n_layers, max_seq, ...)` | Preallocated `[n_layers, max_seq, d_model]` for K and for V. Lives in the parameter root. | Allocation only. |
| `cds_kv_cache_append(cache, layer, k, v, pos)` | Copy this step's K and V rows into the cache at `pos`. | Readback. |
| `cds_attention_cached(q, cache, layer, pos, out, n_heads)` | Attention for one new token against positions `0 .. pos`. No causal mask needed because the cache holds only the past. | Equals the last row of the full-sequence form. |

## Stage 6: the transformer block

| Function | Does | Check |
|---|---|---|
| `cds_mlp(x, W1, b1, W2, b2, out)` | `linear, gelu, linear`, hidden size `4 * d_model`. | Composition of checked parts. |
| `cds_block_forward(x, params, cache, pos, out)` | `x + attn(ln1(x))` then `+ mlp(ln2(x))`, with the residual adds. | Diff against the llm.c reference for one layer. |

A `CdsBlockParams` struct holds the twelve parameter arrays per layer, all views into one weight blob in the parameter root.

## Stage 7: the model

| Function | Does | Check |
|---|---|---|
| `cds_gpt2_load(path, root)` | Read the llm.c checkpoint format (`gpt2_124M.bin`): a header, then every parameter in a fixed order as fp32. One `cudaMemcpy` of the blob, then views. | Parameter count equals 124,439,808. |
| `cds_gpt2_forward(model, tokens, n, pos, logits)` | Embed, add positions, run every block, final layernorm, project to vocab. Returns `[n, vocab]` logits. | Logits for a fixed prompt match llm.c to 1e-3. |

Matching llm.c on the full forward pass is the milestone. Everything before it is checked in isolation, and this is the first time the composition is checked end to end.

## Stage 8: tokenizer and sampling

| Function | Does | Check |
|---|---|---|
| `cds_tokenizer_load(path)` | Read `gpt2_tokenizer.bin` from llm.c, a vocab of 50,257 byte strings. | Token 0 decodes to `!`. |
| `cds_tokenizer_decode(tok, id)` | Id to bytes. Needed first because the first demo can start from a known token id. | Known ids. |
| `cds_tokenizer_encode(tok, text, ids)` | Byte-pair encoding of a prompt. The hardest pure-CPU piece in the project. Do it last. | Round trip through decode. |
| `cds_sample_greedy(logits)` | `cds_argmax_rows` on the last row. | CPU. |
| `cds_sample_topk(logits, k, temperature, rng)` | Scale by temperature, keep the top k, softmax, draw. Copy the top-k candidates to the host and draw there, since k is tiny. | Histogram over many draws matches the softmax. |

## Stage 9: the generation loop

```
prefill:  logits = forward(prompt_tokens, n, pos=0)      fills the KV cache
decode:   repeat max_new_tokens times:
            next = sample(logits)
            print(decode(next))
            restore(activation checkpoint)
            logits = forward([next], 1, pos)
            pos += 1
```

Measure tokens per second. On an RTX 4050 in fp32 with a 124M model you should see well over 100 tokens per second once the decode step launches nothing but kernels and one `cudaMemcpy` of the sampled token.

## Stage 10: after it works

In order of payoff:

1. fp16 or bf16 weights and activations through cuBLASLt. Halves memory traffic, which is the whole cost of decode.
2. Fuse layernorm into the following matmul's prologue, and residual add into the epilogue. cuBLASLt epilogues do some of this for free.
3. CUDA graphs for the decode step, which removes launch overhead for a step that is dozens of tiny kernels.
4. A fused attention kernel that never materialises the score matrix. Flash attention is this idea taken to the end.

## Reference material

- llm.c, `train_gpt2.c` and `train_gpt2.cu`: the CPU reference for every layer and the checkpoint and tokenizer formats this roadmap assumes.
- The GPT-2 paper and the nanoGPT `model.py` for the exact layer order and the GELU variant.
- cuBLASLt documentation, the `cublasLtMatmul` page and the epilogue enum, for stages 3 and 10.
