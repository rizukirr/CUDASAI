#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <cstdio>

int main() {
  CdsContext *ctx = cds_context_new(32 << 20);
  if (!ctx)
    return 1;

  Cds *cds = cds_new(1 << 20);
  if (!cds) {
    cds_context_free(ctx);
    return 1;
  }

  float data[] = {0.55f, 0.45, 0.69, 0.31, 0.73, 0.27, 0.70, 0.30};
  size_t shape[] = {4, 2};
  CdsArray *x = cds_array(cds, shape, 2, CDS_TYPE_FLOAT32, data);
  if (!x) {
    cds_context_free(ctx);
    cds_free(cds);
    return 1;
  }

  printf("before: \n");
  print_array(x);

  size_t dest_shape[] = {4};
  CdsArray *dest = cds_array(cds, dest_shape, 1, CDS_TYPE_INT32, nullptr);
  if (!dest) {
    cds_context_free(ctx);
    cds_free(cds);
    return 1;
  }

  if (cds_argmax(ctx, x, dest) != CDS_OK) {
    cds_context_free(ctx);
    cds_free(cds);
    return 1;
  }

  printf("after:\n");
  print_array(dest);

  cds_context_free(ctx);
  cds_free(cds);
  return 0;
}
