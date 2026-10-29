#ifndef CDS_INTERNAL_KERNEL_H
#define CDS_INTERNAL_KERNEL_H

#include "_error.h"
#include "array.h"
#include "kernel.h"
#include <cublasLt.h>

constexpr unsigned CDS_THREAD_DEFAULT = 256;

struct CdsContext {
  cublasLtHandle_t lt;
  void *workspace;
  size_t workspace_size;
  cudaStream_t stream;
};

// Blocks needed so that `count` elements each get one thread.
inline unsigned cds_blocks_for(size_t count) {
  return (unsigned)((count + CDS_THREAD_DEFAULT - 1) / CDS_THREAD_DEFAULT);
}

// Kernel templates stay in this header so every file that launches one can
// instantiate it for its own type. Each kernel is one thread per element
// unless its comment says otherwise.

template <typename T>
__global__ void fill_kernel(T *data, size_t count, T value) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < count)
    data[i] = value;
}

template <typename T>
__global__ void cds_softmax_lt(const T *x, T *y, size_t cols);

// Host launchers. Each one picks the instantiation for the array's dtype,
// launches on the context's stream and checks the launch.

CdsError cds_matmul_lt(const CdsContext *ctx, CdsActivation activation,
                       const CdsArray *x, const CdsArray *w,
                       const CdsArray *bias, CdsArray *y);

CdsError cds_add_lt(const CdsContext *ctx, const CdsArray *a, const CdsArray *b,
                    CdsArray *dest);

template <typename T>
CdsError cds_scale_lt(const CdsContext *ctx, const CdsArray *a, const T b,
                      CdsArray *dest);

CdsError kernel_fill(const CdsContext *ctx, CdsArray *dst, const void *value);

#endif // CDS_INTERNAL_KERNEL_H
