#define ARENA_USE_CUDA
#include "_array.h"
#include <assert.h>
#include <math.h>
#include <string.h>

void random_fill_float32(float *data, size_t count, float low, float high) {
  for (size_t i = 0; i < count; i++) {
    data[i] = low + (rand() / ((float)RAND_MAX + 1.0f)) * (high - low);
  }
}

void random_fill_float64(double *data, size_t count, double low, double high) {
  for (size_t i = 0; i < count; i++) {
    data[i] = low + (rand() / ((double)RAND_MAX + 1.0)) * (high - low);
  }
}

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

CdsArray *cds_new_derived(const Cds *root, const size_t *shape, size_t dim,
                          CdsType type) {
  assert(root != nullptr);

  CdsArray *self =
      (CdsArray *)arena_alloc(root->host, sizeof(CdsArray), alignof(CdsArray));
  if (!self)
    return nullptr;

  if (!cds_init_meta(self, shape, dim, type))
    return nullptr;

  self->data = arena_alloc(root->arena, self->capacity, cds_alignof[type]);
  if (!self->data)
    return nullptr;
  return self;
}
