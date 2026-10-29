#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <cstdio>
#include <cstdlib>

int main(void) {
  // One context for the whole program: handle, workspace and stream.
  CdsContext *ctx = cds_context_new(32 << 20);
  if (!ctx) {
    fprintf(stderr, "cds_context_new failed\n");
    return 1;
  }

  // One root for parameters, one for activations. Only these two are freed.
  Cds *params = cds_new(1 << 20);
  Cds *acts = cds_new(1 << 20);
  if (!params || !acts) {
    fprintf(stderr, "cds_new failed\n");
    cds_free(acts);
    cds_free(params);
    cds_context_free(ctx);
    return 1;
  }

  // Two GELU hidden layers, each followed by LayerNorm, then a linear output
  // projection.
  constexpr size_t batch = 4, hidden_layers = 2;
  constexpr size_t widths[] = {3, 4, 4, 2};
  constexpr double eps = 1e-5;
  CdsArray *weights[hidden_layers + 1]{};
  CdsArray *biases[hidden_layers + 1]{};
  CdsArray *gammas[hidden_layers]{};
  CdsArray *betas[hidden_layers]{};
  srand(42);

  for (size_t layer = 0; layer <= hidden_layers; ++layer) {
    const size_t w_shape[] = {widths[layer + 1], widths[layer]};
    const size_t b_shape[] = {widths[layer + 1]};
    weights[layer] = cds_array_rand(params, w_shape, 2, CDS_TYPE_FLOAT32);
    biases[layer] = cds_array_rand(params, b_shape, 1, CDS_TYPE_FLOAT32);
    if (!weights[layer] || !biases[layer]) {
      fprintf(stderr, "cds_array_rand failed\n");
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }

    printf("layer %zu weights and bias:\n", layer + 1);
    if (!print_array(weights[layer]) || !print_array(biases[layer])) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
  }

  // LayerNorm starts as the identity transform: gamma = 1, beta = 0. A null
  // data pointer zero-fills, so only gamma needs cds_fill.
  const float one = 1.0f;
  for (size_t layer = 0; layer < hidden_layers; ++layer) {
    const size_t shape[] = {widths[layer + 1]};
    gammas[layer] = cds_array(params, shape, 1, CDS_TYPE_FLOAT32, nullptr);
    betas[layer] = cds_array(params, shape, 1, CDS_TYPE_FLOAT32, nullptr);
    if (!gammas[layer] || !betas[layer] ||
        cds_fill(ctx, gammas[layer], &one) != CDS_OK) {
      fprintf(stderr, "layernorm parameters failed\n");
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
  }

  // Each batch runs the whole network before its activations are reclaimed.
  const size_t x_shape[] = {batch, widths[0]};
  const size_t y_shape[] = {batch, widths[hidden_layers + 1]};
  CdsCheckpoint cp = cds_checkpoint(acts);

  for (int step = 0; step < 3; step++) {
    float x_host[batch][widths[0]];
    for (size_t i = 0; i < batch; i++)
      for (size_t j = 0; j < widths[0]; j++)
        x_host[i][j] = (float)(step * 100 + i * widths[0] + j);

    CdsArray *x = cds_array(acts, x_shape, 2, CDS_TYPE_FLOAT32, x_host);
    CdsArray *logits = cds_array(acts, y_shape, 2, CDS_TYPE_FLOAT32, nullptr);
    CdsArray *y = cds_array(acts, y_shape, 2, CDS_TYPE_FLOAT32, nullptr);
    if (!x || !logits || !y) {
      fprintf(stderr, "cds_array failed\n");
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }

    CdsArray *h = x;
    for (size_t layer = 0; layer < hidden_layers; ++layer) {
      const size_t shape[] = {batch, widths[layer + 1]};
      CdsArray *next = cds_array(acts, shape, 2, CDS_TYPE_FLOAT32, nullptr);
      if (!next) {
        fprintf(stderr, "cds_array failed\n");
        cds_free(acts);
        cds_free(params);
        cds_context_free(ctx);
        return 1;
      }
      cds_linear(ctx, h, weights[layer], biases[layer], CDS_ACTIVATION_GELU,
                 next);
      // In place: x and y may be the same array.
      cds_layernorm(ctx, next, gammas[layer], betas[layer], eps, next);
      h = next;
    }

    // Final logits and probabilities are computed after the hidden-layer loop.
    cds_linear(ctx, h, weights[hidden_layers], biases[hidden_layers],
               CDS_ACTIVATION_NONE, logits);
    cds_softmax(ctx, logits, y);

    printf("\nbatch %d\nx\n", step);
    if (!print_array(x)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
    puts("h = layernorm(gelu(linear(...))), last hidden layer");
    if (!print_array(h)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
    puts("logits (linear output without GELU)");
    if (!print_array(logits)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
    puts("y = softmax(logits)");
    if (!print_array(y)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }

    // Reclaim this batch only after consuming its output probabilities.
    cds_restore(acts, cp);
  }

  cds_free(acts);
  cds_free(params);
  cds_context_free(ctx);
  return 0;
}
