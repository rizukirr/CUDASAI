#include "array.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

static void print_matrix(const char *name, const float *m, size_t rows,
                         size_t cols) {
  printf("%s [%zu x %zu]\n", name, rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++)
      printf("%8.4f ", m[i * cols + j]);
    printf("\n");
  }
}

int main(void) {
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

  float w_host[2][3], b_host[2];
  cds_to_host(W, w_host);
  cds_to_host(b, b_host);
  print_matrix("W", &w_host[0][0], out_features, in_features);
  print_matrix("b", b_host, 1, out_features);

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

    cds_linear(x, W, b, y);

    float y_host[4][2];
    cds_to_host(y, y_host);

    printf("\nstep %d, x data at %p\n", step, cds_data_at(x, 0));
    print_matrix("x", &x_host[0][0], batch, in_features);
    print_matrix("y = x W^T + b", &y_host[0][0], batch, out_features);

    // x and y are gone after this. The next step reuses the same bytes.
    cds_restore(acts, cp);
  }

  cds_free(acts);
  cds_free(params);
  return 0;
}
