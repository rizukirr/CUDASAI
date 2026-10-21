#ifndef TEST_CUDA_RUNTIME_API_H
#define TEST_CUDA_RUNTIME_API_H

#include <stddef.h>

enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation };
cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);

#endif
