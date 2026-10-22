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
  // A batch of 4 samples with 3 features each, mapped to 2 outputs.
  const size_t batch = 4, in_features = 3, out_features = 2;
  float x_host[4][3] = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
      {10, 11, 12},
  };
  // Row n of W is the weight vector of output n.
  float w_host[2][3] = {
      {1, 0, -1},
      {0.5f, 0.5f, 0.5f},
  };
  float b_host[2] = {10, -10};

  // Parameters: created once, reused for every batch.
  size_t w_shape[] = {out_features, in_features};
  size_t b_shape[] = {out_features};
  Cds *W = cds_new(w_shape, 2, CDS_FLOAT32, w_host);
  Cds *b = cds_new(b_shape, 1, CDS_FLOAT32, b_host);

  // Activations: created per batch.
  size_t x_shape[] = {batch, in_features};
  size_t y_shape[] = {batch, out_features};
  Cds *x = cds_new(x_shape, 2, CDS_FLOAT32, x_host);
  Cds *y = cds_new(y_shape, 2, CDS_FLOAT32, nullptr);

  if (!W || !b || !x || !y) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  cds_linear(x, W, b, y);

  float y_host[4][2];
  cds_to_host(y, y_host);

  print_matrix("x", &x_host[0][0], batch, in_features);
  print_matrix("W", &w_host[0][0], out_features, in_features);
  print_matrix("y = x W^T + b", &y_host[0][0], batch, out_features);

  cds_free(y);
  cds_free(x);
  cds_free(b);
  cds_free(W);
  return 0;
}
