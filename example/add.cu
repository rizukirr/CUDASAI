#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <cstdio>

int main(void) {
  CdsContext *ctx = cds_context_new(32 << 20);
  if (!ctx) {
    fprintf(stderr, "cds_context_new failed\n");
    return 1;
  }

  Cds *cds = cds_new(1 << 20);
  size_t a_shape[] = {2, 2};
  float a_data[2][2] = {{1.0f, 1.0f}, {2.0f, 2.0f}};

  size_t b_shape[] = {2, 2};
  float b_data[2][2] = {{1.0f, 1.0f}, {2.0f, 2.0f}};

  CdsArray *a = cds_array(cds, a_shape, 2, CDS_FLOAT32, a_data);
  CdsArray *b = cds_array(cds, b_shape, 2, CDS_FLOAT32, b_data);
  CdsArray *dest = cds_array(cds, a_shape, 2, CDS_FLOAT32, nullptr);

  cds_add(ctx, a, b, dest);

  if (!print_array(dest)) {
    cds_free(cds);
    cds_context_free(ctx);
    return 1;
  }

  cds_free(cds);
  cds_context_free(ctx);
}
