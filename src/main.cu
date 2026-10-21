#include "array.h"
#include <stdio.h>

__global__ void print_data(const size_t count, Cds *a) {
  for (size_t i = 0; i < count; i++) {
    printf("%d\n", *(int *)cds_data_at(a, i));
  }
}

int main(void) {
  size_t shape[] = {10, 10};

  int data[10][10] = {};
  Cds *a = cds_new(shape, 2, CDS_INT32, data);
  size_t count = cds_count(a);
  const unsigned int threads = 256;
  size_t blocks = count / threads + (count % threads != 0);

  print_data<<<blocks, threads>>>(count, a);

  cds_free(a);
  return 0;
}
