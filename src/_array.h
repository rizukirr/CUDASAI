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
  void *data; // device pointer
  Arena *arena;
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
 * Allocate from `arena` and fill the data with uniform random numbers in
 * [-1/sqrt(fan_in), 1/sqrt(fan_in)] where fan_in is the last dimension.
 */
[[nodiscard]]
Cds *cds_new_rand(Arena *arena, const size_t *shape, size_t dim, CdsType type);

/**
 * Allocate from `arena` and zero-fill the data.
 */
[[nodiscard]]
Cds *cds_new_tmp(Arena *arena, const size_t *shape, size_t dim, CdsType type);

#endif // CDS_INTERNAL_H
