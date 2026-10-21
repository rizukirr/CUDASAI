#ifndef TEST_CUDA_RUNTIME_API_H
#define TEST_CUDA_RUNTIME_API_H

#include <stddef.h>

enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation };
enum { cudaMemAttachGlobal = 1 };
cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void *ptr);

#endif
