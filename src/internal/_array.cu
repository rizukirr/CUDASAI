#define ARENA_USE_CUDA
#include "_array.h"
#include "array.h"
#include <assert.h>
#include <math.h>
#include <string.h>

bool cds_init_meta(CdsArray *self, const size_t *shape, size_t dim,
                   CdsType type) {
  assert(self != nullptr);
  if (CDS_REQUIRE(shape != nullptr) || CDS_REQUIRE(dim > 0) ||
      CDS_REQUIRE(dim <= CDS_MAX_DIM) ||
      CDS_REQUIRE(type == CDS_TYPE_FLOAT32 || type == CDS_TYPE_FLOAT64))
    return false;

  self->dim = dim;
  self->type = type;
  self->elem_size = cds_type_size[type];
  memcpy(self->shape, shape, sizeof(size_t) * dim);

  // Strides in bytes, innermost first, so the last stride is the element size
  // and strides[0] * shape[0] is the total byte size.
  self->strides[dim - 1] = self->elem_size;
  for (size_t i = dim - 1; i > 0; i--) {
    if (__builtin_mul_overflow(self->strides[i], shape[i],
                               &self->strides[i - 1])) {
      CDS_FAIL(CDS_ERR_INVALID, "array byte size overflows size_t");
      return false;
    }
  }
  if (__builtin_mul_overflow(self->strides[0], shape[0], &self->capacity)) {
    CDS_FAIL(CDS_ERR_INVALID, "array byte size overflows size_t");
    return false;
  }

  self->count = self->capacity / self->elem_size;
  self->data = nullptr;
  return true;
}

CdsArray *cds_new_struct(const Cds *root, const size_t *shape, size_t dim,
                         CdsType type) {
  if (CDS_REQUIRE(root != nullptr))
    return nullptr;

  CdsArray *self =
      (CdsArray *)arena_alloc(root->host, sizeof(CdsArray), alignof(CdsArray));
  if (!self) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "host arena allocation failed");
    return nullptr;
  }

  if (!cds_init_meta(self, shape, dim, type))
    return nullptr;
  return self;
}

CdsArray *cds_new_derived(const Cds *root, const size_t *shape, size_t dim,
                          CdsType type) {
  CdsArray *self = cds_new_struct(root, shape, dim, type);
  if (!self)
    return nullptr;

  // The arena refuses zero-byte allocations.
  if (CDS_REQUIRE(self->capacity > 0))
    return nullptr;

  self->data = arena_alloc(root->arena, self->capacity, cds_alignof[type]);
  if (!self->data) {
    // The arena's cudaMalloc failure is still pending: clear it so the next
    // launch check does not report it as its own.
    (void)cudaGetLastError();
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "device allocation of %zu bytes failed",
             self->capacity);
    return nullptr;
  }
  return self;
}
