#include "array.h"
#include "tensor.h"
#include <stdio.h>

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

  // Parameters: a 3 to 2 linear layer, loaded once. Row n of W is the weight
  // vector of output n.
  const size_t batch = 4, in_features = 3, out_features = 2;
  float w_host[2][3] = {
      {1, 0, -1},
      {0.5f, 0.5f, 0.5f},
  };
  float b_host[2] = {10, -10};
  size_t w_shape[] = {out_features, in_features};
  size_t b_shape[] = {out_features};
  CdsArray *W = cds_array(params, w_shape, 2, CDS_FLOAT32, w_host);
  CdsArray *b = cds_array(params, b_shape, 1, CDS_FLOAT32, b_host);

  // Activations: one batch of 4 samples.
  float x_host[4][3] = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
      {10, 11, 12},
  };
  size_t x_shape[] = {batch, in_features};
  size_t y_shape[] = {batch, out_features};
  CdsArray *x = cds_array(acts, x_shape, 2, CDS_FLOAT32, x_host);
  CdsArray *y = cds_array(acts, y_shape, 2, CDS_FLOAT32, nullptr);

  if (!W || !b || !x || !y) {
    fprintf(stderr, "cds_array failed\n");
    return 1;
  }

  cds_linear(x, W, b, y);

  float y_host[4][2];
  cds_to_host(y, y_host);

  print_matrix("x", &x_host[0][0], batch, in_features);
  print_matrix("W", &w_host[0][0], out_features, in_features);
  print_matrix("y = x W^T + b", &y_host[0][0], batch, out_features);

  // W, b, x and y are owned by their roots and go away with them.
  cds_free(acts);
  cds_free(params);
  return 0;
}
