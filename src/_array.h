#ifndef CDS_INTERNAL_H
#define CDS_INTERNAL_H

#include "arena.h"
#include "array.h"

constexpr size_t CDS_MAX_DIM = 8;

struct Cds {
  size_t dim;
  size_t elem_size;
  size_t count;
  size_t capacity;
  CdsType type;
  size_t shape[CDS_MAX_DIM];
  size_t strides[CDS_MAX_DIM];
  void *data;
  Arena *arena;
};

static constexpr size_t cds_alignof[] = {
    alignof(int8_t),  alignof(int16_t),  alignof(int32_t),  alignof(int64_t),
    alignof(uint8_t), alignof(uint16_t), alignof(uint32_t), alignof(uint64_t),
    alignof(float),   alignof(double),
};

void random_fill_float32(float *data, size_t count, double low, double high);

void random_fill_float64(float *data, size_t count, double low, double high);

[[nodiscard]]
size_t cds_count(const size_t *shape, const size_t dim);

[[nodiscard("check the returned value for overflow")]]
bool cds_strides(const size_t *shape, const size_t dim, const size_t elem_size,
                 size_t *strides);

/**
 * Copy the metadata from src and fill the data with random numbers.
 */
[[nodiscard]]
Cds *cds_new_rand(Arena *arena, const size_t *shape, size_t dim, CdsType type);

[[nodiscard]]
Cds *cds_new_tmp(Arena *arena, const size_t *shape, size_t dim, CdsType type);

#endif // CDS_INTERNAL_H
