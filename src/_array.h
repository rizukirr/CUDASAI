#ifndef CDS_INTERNAL_H
#define CDS_INTERNAL_H

#include "arena.h"
#include "array.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t cds_err_ = (call);                                             \
    if (cds_err_ != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(cds_err_));                                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

constexpr size_t CDS_MAX_DIM = 8;

struct Cds {
  size_t dim;
  size_t elem_size;
  size_t count;
  size_t capacity;
  CdsType type;
  size_t shape[CDS_MAX_DIM];
  size_t strides[CDS_MAX_DIM];
  void *data;   // device pointer
  Arena *arena; // device memory for data
  Arena *host;  // host memory for Cds structs derived from this array
  bool owner;   // true only for the root array created by cds_new
};

// Indexed by CdsType, so the order must match the enum.
static constexpr size_t cds_alignof[] = {
    alignof(float),
    alignof(double),
};

void random_fill_float32(float *data, size_t count, float low, float high);

void random_fill_float64(double *data, size_t count, double low, double high);

/**
 * Fill dim, type, elem_size, count, capacity, shape and strides.
 * Returns false when the byte size overflows.
 */
[[nodiscard("check the returned value for overflow")]]
bool cds_init_meta(Cds *self, const size_t *shape, size_t dim, CdsType type);

/**
 * Allocate the struct from the root's host arena and the data from its device
 * arena, leaving the data uninitialised. Nothing here needs freeing: both go
 * away with the root.
 */
[[nodiscard]]
Cds *cds_new_derived(const Cds *root, const size_t *shape, size_t dim,
                     CdsType type);

#endif // CDS_INTERNAL_H
