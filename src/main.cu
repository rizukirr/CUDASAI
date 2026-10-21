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
  // A batch of 4 samples with 3 features each.
  const size_t batch = 4, in_features = 3, out_features = 2;
  float input[4][3] = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
      {10, 11, 12},
  };

  size_t shape[] = {batch, in_features};
  Cds *x = cds_new(shape, 2, CDS_FLOAT32, input);
  if (!x) {
    fprintf(stderr, "cds_new failed\n");
    return 1;
  }

  // y = x · W^T + b, with W and b drawn at random inside cds_linear.
  Cds *y = cds_linear(x, out_features, true);
  if (!y) {
    fprintf(stderr, "cds_linear failed\n");
    cds_free(x);
    return 1;
  }

  float output[4][2];
  cds_to_host(y, output);

  print_matrix("input", &input[0][0], batch, in_features);
  print_matrix("output", &output[0][0], batch, out_features);

  // y is owned by x, so one free releases both.
  cds_free(x);
  return 0;
}
