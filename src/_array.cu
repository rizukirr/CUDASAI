#define ARENA_USE_CUDA
#include "_array.h"
#include <assert.h>
#include <math.h>
#include <string.h>

bool cds_init_meta(CdsArray *self, const size_t *shape, size_t dim,
                   CdsType type) {
  assert(self != nullptr);
  assert(shape != nullptr);
  assert(dim > 0);
  assert(dim <= CDS_MAX_DIM);
  assert(type == CDS_FLOAT32 || type == CDS_FLOAT64);

  self->dim = dim;
  self->type = type;
  self->elem_size = cds_type_size[type];
  memcpy(self->shape, shape, sizeof(size_t) * dim);

  // Strides in bytes, innermost first, so the last stride is the element size
  // and strides[0] * shape[0] is the total byte size.
  self->strides[dim - 1] = self->elem_size;
  for (size_t i = dim - 1; i > 0; i--) {
    if (__builtin_mul_overflow(self->strides[i], shape[i],
                               &self->strides[i - 1]))
      return false;
  }
  if (__builtin_mul_overflow(self->strides[0], shape[0], &self->capacity))
    return false;

  self->count = self->capacity / self->elem_size;
  self->data = nullptr;
  return true;
}

CdsArray *cds_new_struct(const Cds *root, const size_t *shape, size_t dim,
                         CdsType type) {
  assert(root != nullptr);

  CdsArray *self =
      (CdsArray *)arena_alloc(root->host, sizeof(CdsArray), alignof(CdsArray));
  if (!self)
    return nullptr;

  if (!cds_init_meta(self, shape, dim, type))
    return nullptr;
  return self;
}

CdsArray *cds_new_derived(const Cds *root, const size_t *shape, size_t dim,
                          CdsType type) {
  CdsArray *self = cds_new_struct(root, shape, dim, type);
  if (!self)
    return nullptr;

  self->data = arena_alloc(root->arena, self->capacity, cds_alignof[type]);
  if (!self->data)
    return nullptr;
  return self;
}
