#ifndef CDS_INTERNAL_KERNEL_H
#define CDS_INTERNAL_KERNEL_H

#include "_array.h"

constexpr unsigned CDS_THREAD_DEFAULT = 256;

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

// Host launchers. Each one picks the instantiation for the array's dtype,
// launches on the default stream and checks the launch.

void kernel_fill(CdsArray *dst, const void *value);

#endif // CDS_INTERNAL_KERNEL_H
