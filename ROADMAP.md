# Roadmap: tiny LLM inference in CUDA

Target: run a GPT-2 small (124M parameters, 12 layers, 768 wide, 12 heads, 1024 context) decoder in fp32 on the RTX 4050 and generate text from a prompt. GPT-2 small is the right first target because its weights are public, it fits in 500 MB, and llm.c ships a plain C reference implementation you can diff against layer by layer.

Every function below follows the same contract: a C-style function over `CdsArray` values owned by a `Cds` root, a CPU reference implementation of the same math, and one check that compares the two on small shapes. Build the check before the kernel is fast. A kernel that is fast and wrong is worth nothing.

The stages are in dependency order. Finish a stage, including its checks, before starting the next. Tick a box when the function and its check both pass.

## Stage 0: what exists today

- [x] `cds_new(bytes)`: a `Cds` root that owns a device arena for array data and a host arena for `CdsArray` structs.
- [x] `cds_array(root, shape, dim, type, data)`: a `CdsArray` created inside a root, released with it.
- [x] `cds_array_rand(root, shape, dim, type)`: an array inside a root filled with uniform random values, for weights and biases.
- [x] `cds_view(root, src, offset, shape, dim)`: a `CdsArray` over a window of another array's device memory, no copy. For reshapes, head splits and slicing parameters out of one weight blob.
- [x] `cds_copy(dst, src)`: device to device copy between arrays of equal count and type.
- [x] `cds_fill(dst, value)`: set every element to a scalar. The first hand-written kernel, in `src/internal/_kernel.h`.
- [x] `cds_checkpoint(root)`, `cds_restore(root, cp)`: rewind a root to a saved position so activations are reused per token instead of growing.
- [x] `cds_to_host`, `cds_free`, `cds_to_meta`.
- [x] `cds_linear(x, W, b, y)`: `y = x · W^T + b` through cuBLASLt. Allocates nothing. `main.cu` shows the two-root pattern: parameters in one root, activations in another.

## Stage 1: memory and plumbing

- [x] `CdsContext`: holds the cuBLASLt handle, a cuBLASLt workspace of 32 MB, and a CUDA stream. Every kernel launch and every cuBLASLt call takes the stream. Without it you cannot overlap anything later, and retrofitting a stream argument into forty functions is painful.

## Stage 2: elementwise and reduction kernels

These are the kernels where you learn CUDA. Each is one or two kernels. Write the naive one-thread-per-element version first, then the shared-memory reduction for the row-wise ones.

- [x] `cds_add(a, b, out)`: `out = a + b`. Used for residual connections. Same shape. Check: CPU loop.
- [x] `cds_scale(a, s, out)`: `out = a * s`. Same shape. Check: CPU loop.
- [x] `cds_gelu(a, out)`: GELU with the tanh approximation GPT-2 uses. Same shape. Check: CPU loop, tolerance 1e-5.
- [x] `cds_softmax_rows(a, out)`: numerically stable softmax over the last dimension: subtract the row max, exp, divide by the row sum. One block per row, warp shuffle for the reductions. Shape `[rows, cols]`. Check: rows sum to 1, matches CPU to 1e-5.
- [x] `cds_layernorm(x, gamma, beta, out, eps)`: per-row mean and variance, then `gamma * (x - mean) / sqrt(var + eps) + beta`. GPT-2 uses LayerNorm. Llama uses RMSNorm, which drops the mean. Shape `[rows, d]`. Check: CPU loop, tolerance 1e-4.
- [ ] `cds_argmax_rows(a, out_indices)`: index of the max in each row. The greedy sampler. Shape `[rows, vocab]` to `[rows]`. Check: CPU loop.

The softmax and layernorm reductions are where most people get their first wrong answer from a race condition. Run them under `compute-sanitizer --tool racecheck`.

## Stage 3: matmul variants

- [ ] `cds_matmul(a, b, out, transpose_a, transpose_b)`: general `[M,K] · [K,N]` through cuBLASLt with transpose flags. `cds_linear` becomes a thin wrapper. Check: CPU triple loop on 5x7x3.
- [ ] `cds_matmul_batched(a, b, out, ...)`: the same, batched over a leading dimension, using the cuBLASLt strided batch attributes. Attention needs `Q · K^T` and `P · V` once per head. Check: loop of single matmuls gives the same result.
- [ ] `cds_transpose_heads(x, out, n_heads)`: `[seq, n_heads * head_dim]` to `[n_heads, seq, head_dim]` and back. A permute kernel. Check: round trip is the identity.
- [ ] Naive and tiled matmul kernels of your own, even though cuBLASLt will be the one you ship. Benchmark all three on 768x768x1024. Knowing the gap between your tiled kernel and cuBLASLt is the point of the exercise.

## Stage 4: embeddings and positions

- [ ] `cds_embed(table, token_ids, out)`: gather rows: `out[i] = table[ids[i]]`. One thread per output element. Check: CPU gather.
- [ ] `cds_pos_embed_add(x, pos_table, start, out)`: add the learned position vector for positions `start .. start + seq`. GPT-2 uses learned positions. Llama uses RoPE, a rotation applied to Q and K inside attention, which you would add here if you switch models. Check: CPU loop.

The output projection to logits reuses `cds_matmul` with the embedding table transposed. GPT-2 ties those weights.

## Stage 5: attention

Build it in two forms. The full-sequence form first, because it is simpler to check. The cached form second, because inference needs it.

- [ ] `cds_attention(q, k, v, out, n_heads, causal)`: per head: `scores = q · k^T / sqrt(head_dim)`, mask positions after the diagonal with negative infinity, softmax rows, multiply by `v`, merge heads. Composed from stages 2 and 3. Check: CPU reference on 2 heads, seq 4, head_dim 3.
- [ ] `cds_kv_cache_new(root, n_layers, max_seq, ...)`: preallocated `[n_layers, max_seq, d_model]` for K and for V. Lives in the parameter root. Check: allocation only.
- [ ] `cds_kv_cache_append(cache, layer, k, v, pos)`: copy this step's K and V rows into the cache at `pos`. Check: readback.
- [ ] `cds_attention_cached(q, cache, layer, pos, out, n_heads)`: attention for one new token against positions `0 .. pos`. No causal mask needed because the cache holds only the past. Check: equals the last row of the full-sequence form.

## Stage 6: the transformer block

- [ ] `cds_mlp(x, W1, b1, W2, b2, out)`: `linear, gelu, linear`, hidden size `4 * d_model`. Check: composition of checked parts.
- [ ] `cds_block_forward(x, params, cache, pos, out)`: `x + attn(ln1(x))` then `+ mlp(ln2(x))`, with the residual adds. Check: diff against the llm.c reference for one layer.
- [ ] `CdsBlockParams`: a struct holding the twelve parameter arrays per layer, all views into one weight blob in the parameter root.

## Stage 7: the model

- [ ] `cds_gpt2_load(path, root)`: read the llm.c checkpoint format (`gpt2_124M.bin`): a header, then every parameter in a fixed order as fp32. One `cudaMemcpy` of the blob, then views. Check: parameter count equals 124,439,808.
- [ ] `cds_gpt2_forward(model, tokens, n, pos, logits)`: embed, add positions, run every block, final layernorm, project to vocab. Returns `[n, vocab]` logits. Check: logits for a fixed prompt match llm.c to 1e-3.

Matching llm.c on the full forward pass is the milestone. Everything before it is checked in isolation, and this is the first time the composition is checked end to end.

## Stage 8: tokenizer and sampling

- [ ] `cds_tokenizer_load(path)`: read `gpt2_tokenizer.bin` from llm.c, a vocab of 50,257 byte strings. Check: token 0 decodes to `!`.
- [ ] `cds_tokenizer_decode(tok, id)`: id to bytes. Needed first because the first demo can start from a known token id. Check: known ids.
- [ ] `cds_tokenizer_encode(tok, text, ids)`: byte-pair encoding of a prompt. The hardest pure-CPU piece in the project. Do it last. Check: round trip through decode.
- [ ] `cds_sample_greedy(logits)`: `cds_argmax_rows` on the last row. Check: CPU.
- [ ] `cds_sample_topk(logits, k, temperature, rng)`: scale by temperature, keep the top k, softmax, draw. Copy the top-k candidates to the host and draw there, since k is tiny. Check: histogram over many draws matches the softmax.

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

- [ ] Prefill and decode loop as above, printing tokens as they are produced.
- [ ] Measure tokens per second. On an RTX 4050 in fp32 with a 124M model you should see well over 100 tokens per second once the decode step launches nothing but kernels and one `cudaMemcpy` of the sampled token.

## Stage 10: after it works

In order of payoff:

- [ ] fp16 or bf16 weights and activations through cuBLASLt. Halves memory traffic, which is the whole cost of decode.
- [ ] Fuse layernorm into the following matmul's prologue, and residual add into the epilogue. cuBLASLt epilogues do some of this for free.
- [ ] CUDA graphs for the decode step, which removes launch overhead for a step that is dozens of tiny kernels.
- [ ] A fused attention kernel that never materialises the score matrix. Flash attention is this idea taken to the end.

## Reference material

- llm.c, `train_gpt2.c` and `train_gpt2.cu`: the CPU reference for every layer and the checkpoint and tokenizer formats this roadmap assumes.
- The GPT-2 paper and the nanoGPT `model.py` for the exact layer order and the GELU variant.
- cuBLASLt documentation, the `cublasLtMatmul` page and the epilogue enum, for stages 3 and 10.
