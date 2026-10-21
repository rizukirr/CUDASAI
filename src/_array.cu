#define ARENA_USE_CUDA
#define ARENA_IMPLEMENTATION
#include "_array.h"
#include "arena.h"
#include "array.h"
#include <string.h>

void random_fill_float32(float *data, size_t count, float low, float high) {

  for (size_t i = 0; i < count; i++) {
    data[i] = low + (rand() / ((float)RAND_MAX + 1.0)) * (high - low);
  }
}

void random_fill_float64(double *data, size_t count, double low, double high) {

  for (size_t i = 0; i < count; i++) {
    data[i] = low + (rand() / ((double)RAND_MAX + 1.0)) * (high - low);
  }
}

size_t cds_count(const size_t *shape, const size_t dim) {
  size_t count = 1;
  for (size_t i = 0; i < dim; i++) {
    count *= shape[i];
  }
  return count;
}

bool cds_strides(const size_t *shape, const size_t dim, const size_t elem_size,
                 size_t *strides) {
  strides[dim - 1] = elem_size;
  for (size_t i = dim - 1; i > 0; i--) {
    if (__builtin_mul_overflow(strides[i], shape[i], &strides[i - 1])) {
      return false;
    }
  }

  return true;
}

Cds *cds_new_rand(Arena *arena, const size_t *shape, size_t dim, CdsType type) {
  assert(arena != nullptr);
  assert(shape != nullptr);
  assert(dim > 0);
  assert(dim <= CDS_MAX_DIM);
  assert(type != CDS_FLOAT32 && type != CDS_FLOAT64);

  Cds *self = (Cds *)arena_alloc(arena, sizeof(Cds), alignof(Cds));
  self->dim = dim;
  self->type = type;
  self->elem_size = cds_type_size[type];
  self->count = cds_count(shape, dim);
  self->capacity = self->count * self->elem_size;
  memcpy(self->shape, shape, sizeof(size_t) * dim);
  if (!cds_strides(shape, dim, self->elem_size, self->strides)) {
    return nullptr;
  }

  self->arena = arena;
  self->data = arena_alloc(arena, self->capacity, cds_alignof[type]);
  if (!self->data) {
    return nullptr;
  }

  double bound = 1.0 / sqrt(self->shape[self->dim - 1]);
  switch (self->type) {
  case CDS_FLOAT32:
    random_fill_float32(static_cast<float *>(self->data), self->count, -bound,
                        bound);
    break;
  case CDS_FLOAT64:
    random_fill_float64(static_cast<double *>(self->data), self->count, -bound,
                        bound);
    break;
  default:
    return nullptr;
  }

  return self;
}

Cds *cds_new_tmp(Arena *arena, const size_t *shape, size_t dim, CdsType type) {
  assert(arena != nullptr);
  assert(shape != nullptr);
  assert(dim > 0);
  assert(dim <= CDS_MAX_DIM);
  assert(type != CDS_FLOAT32 && type != CDS_FLOAT64);

  Cds *self = (Cds *)arena_alloc(arena, sizeof(Cds), alignof(Cds));
  self->dim = dim;
  self->type = type;
  self->elem_size = cds_type_size[type];
  self->count = cds_count(shape, dim);
  self->capacity = self->count * self->elem_size;
  memcpy(self->shape, shape, sizeof(size_t) * dim);
  if (!cds_strides(shape, dim, self->elem_size, self->strides)) {
    return nullptr;
  }

  self->arena = arena;
  self->data = arena_alloc(arena, self->capacity, cds_alignof[type]);
  if (!self->data) {
    return nullptr;
  }
  memset(self->data, 0, self->capacity);

  return self;
}
