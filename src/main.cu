#include "array.h"
#include <stdio.h>

int main(void) {
  size_t shape[] = {10, 10};

  Cds *a = cds_new(shape, 2, CDS_FLOAT32, nullptr);
  if (!a) {
    fprintf(stderr, "cds_new failed\n");
    return 1;
  }

  float host[10][10];
  cds_to_host(a, host);
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++)
      printf("%g ", host[i][j]);
    printf("\n");
  }

  cds_free(a);
  return 0;
}
