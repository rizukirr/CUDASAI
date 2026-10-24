#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

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
    return 1;
  }

  // Parameters: a 3 to 2 linear layer with random weights and bias, created
  // once. The seed makes every run produce the same model.
  const size_t batch = 4, in_features = 3, out_features = 2;
  srand(42);
  size_t w_shape[] = {out_features, in_features};
  size_t b_shape[] = {out_features};
  CdsArray *W = cds_array_rand(params, w_shape, 2, CDS_FLOAT32);
  CdsArray *b = cds_array_rand(params, b_shape, 1, CDS_FLOAT32);
  if (!W || !b) {
    fprintf(stderr, "cds_array_rand failed\n");
    return 1;
  }

  puts("W");
  if (!print_array(W)) {
    cds_free(acts);
    cds_free(params);
    cds_context_free(ctx);
    return 1;
  }
  puts("b");
  if (!print_array(b)) {
    cds_free(acts);
    cds_free(params);
    cds_context_free(ctx);
    return 1;
  }

  // Feed three batches through the same layer, the way a generation loop
  // feeds one token after another. Every batch's activations are created
  // after the checkpoint and reclaimed by the restore, so the activation
  // root never grows past what one batch needs.
  size_t x_shape[] = {batch, in_features};
  size_t y_shape[] = {batch, out_features};
  CdsCheckpoint cp = cds_checkpoint(acts);

  for (int step = 0; step < 3; step++) {
    float x_host[4][3];
    for (size_t i = 0; i < batch; i++)
      for (size_t j = 0; j < in_features; j++)
        x_host[i][j] = (float)(step * 100 + i * in_features + j);

    CdsArray *x = cds_array(acts, x_shape, 2, CDS_FLOAT32, x_host);
    CdsArray *y = cds_array(acts, y_shape, 2, CDS_FLOAT32, nullptr);
    if (!x || !y) {
      fprintf(stderr, "cds_array failed\n");
      return 1;
    }

    cds_linear(ctx, x, W, b, y);

    printf("\nstep %d\nx\n", step);
    if (!print_array(x)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }
    puts("y = x W^T + b");
    if (!print_array(y)) {
      cds_free(acts);
      cds_free(params);
      cds_context_free(ctx);
      return 1;
    }

    // x and y are gone after this. The next step reuses the same bytes.
    cds_restore(acts, cp);
  }

  cds_free(acts);
  cds_free(params);
  cds_context_free(ctx);
  return 0;
}
