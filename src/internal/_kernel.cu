#define ARENA_USE_CUDA
#include "_kernel.h"
#include <assert.h>

void kernel_fill(CdsArray *dst, const void *value) {
  assert(dst != nullptr);
  assert(value != nullptr);

  const unsigned blocks = cds_blocks_for(dst->count);
  switch (dst->type) {
  case CDS_FLOAT32:
    fill_kernel<<<blocks, CDS_THREAD_DEFAULT>>>(
        (float *)dst->data, dst->count, *((const float *)value));
    break;
  case CDS_FLOAT64:
    fill_kernel<<<blocks, CDS_THREAD_DEFAULT>>>(
        (double *)dst->data, dst->count, *((const double *)value));
    break;
  }

  CUDA_CHECK(cudaGetLastError());
}
