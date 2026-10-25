#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <cuda_runtime.h>

int main(void) {
  printf("op: A * b\n");
  CdsContext *ctx = cds_context_new(32 << 20);
  if (!ctx) {
    fprintf(stderr, "cds_context_new failed\n");
    return 1;
  }

  Cds *cds = cds_new(1 << 20);
  if (!cds) {
    cds_context_free(ctx);
    return 1;
  }

  size_t shape[2] = {2, 2};
  float a_data[4] = {1, 2, 2, 1};
  CdsArray *a = cds_array(cds, shape, 2, CDS_FLOAT32, a_data);
  printf("b: 2\n");
  printf("a: ");
  if (!print_array(a)) {
    cds_free(cds);
    cds_context_free(ctx);
    return 1;
  }
  CdsArray *dest = cds_array(cds, shape, 2, CDS_FLOAT32, nullptr);
  cds_scale<float>(ctx, a, 2.0f, dest);

  printf("result:\n");
  if (!print_array(dest)) {
    cds_free(cds);
    cds_context_free(ctx);
    return 1;
  }

  cds_free(cds);
  cds_context_free(ctx);
}
