#define ARENA_USE_CUDA
#define ARENA_IMPLEMENTATION
#include "_array.h"
#include "arena.h"
#include "array.h"
#include <cuda_runtime.h>

Cds *cds_new(const size_t *shape, size_t dim, CdsType type, const void *data) {
  assert(shape != nullptr);
  assert(dim > 0);
  assert(dim <= CDS_MAX_DIM);

  Cds *self = (Cds *)ARENA_MALLOC(sizeof(Cds));
  if (!self)
    return nullptr;

  self->dim = dim;
  self->type = type;
  self->elem_size = cds_type_size[type];
  self->count = cds_count(shape, dim);
  self->capacity = self->count * self->elem_size;
  memcpy(self->shape, shape, sizeof(size_t) * dim);
  if (!cds_strides(shape, dim, self->elem_size, self->strides)) {
    ARENA_FREE(self);
    return nullptr;
  }

  self->arena = arena_create(self->capacity * 2);
  if (!self->arena) {
    ARENA_FREE(self);
    return nullptr;
  }

  self->data = arena_alloc(self->arena, self->capacity, cds_alignof[type]);
  if (!self->data) {
    arena_free(self->arena);
    ARENA_FREE(self);
    return nullptr;
  }

  if (data == nullptr)
    memset(self->data, 0, self->capacity);
  else
    memcpy(self->data, data, self->capacity);

  return self;
}

__host__ __device__ size_t cds_count(Cds *self) { return self->count; }

__host__ __device__ void *cds_data_at(Cds *self, size_t index) {
  return (unsigned char *)self->data + self->strides[0] * index;
}

void cds_free(Cds *self) {
  if (self == nullptr)
    return;

  arena_free(self->arena);
  ARENA_FREE(self);
}
