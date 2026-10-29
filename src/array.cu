#define ARENA_USE_CUDA
#define ARENA_IMPLEMENTATION
#include "_array.h"
#include <assert.h>
#include <math.h>
#include <string.h>

Cds *cds_new(size_t device_bytes) {
  if (CDS_REQUIRE(device_bytes > 0))
    return nullptr;

  Cds *self = (Cds *)ARENA_MALLOC(sizeof(Cds));
  if (!self) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "host allocation of Cds failed");
    return nullptr;
  }

  self->arena = arena_create(device_bytes);
  if (!self->arena) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "device arena creation failed");
    ARENA_FREE(self);
    return nullptr;
  }

  // Structs of arrays created inside this root live here and die with it.
  self->host = arena_create_with(16 * sizeof(CdsArray), arena_host_malloc,
                                 arena_host_free);
  if (!self->host) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "host arena creation failed");
    arena_free(self->arena);
    ARENA_FREE(self);
    return nullptr;
  }

  return self;
}

CdsArray *cds_array(Cds *root, const size_t *shape, size_t dim, CdsType type,
                    const void *data) {
  CdsArray *self = cds_new_derived(root, shape, dim, type);
  if (!self)
    return nullptr;

  CdsError err =
      data == nullptr
          ? CDS_CUDA(cudaMemset(self->data, 0, self->capacity))
          : CDS_CUDA(cudaMemcpy(self->data, data, self->capacity,
                                cudaMemcpyHostToDevice));
  return err == CDS_OK ? self : nullptr;
}

CdsArray *cds_array_rand(Cds *root, const size_t *shape, size_t dim,
                         CdsType type) {
  CdsArray *self = cds_new_derived(root, shape, dim, type);
  if (!self)
    return nullptr;

  void *staging = ARENA_MALLOC(self->capacity);
  if (!staging) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "host staging allocation of %zu bytes failed",
             self->capacity);
    return nullptr;
  }

  double bound = 1.0 / sqrt((double)self->shape[self->dim - 1]);
  switch (self->type) {
  case CDS_TYPE_FLOAT32:
    random_fill((float *)staging, self->count, (float)-bound, (float)bound);
    break;
  case CDS_TYPE_FLOAT64:
    random_fill((double *)staging, self->count, -bound, bound);
    break;
  }

  CdsError err = CDS_CUDA(
      cudaMemcpy(self->data, staging, self->capacity, cudaMemcpyHostToDevice));
  ARENA_FREE(staging);
  return err == CDS_OK ? self : nullptr;
}

CdsArray *cds_view(Cds *root, const CdsArray *src, size_t offset,
                   const size_t *shape, size_t dim) {
  if (CDS_REQUIRE(src != nullptr))
    return nullptr;

  CdsArray *self = cds_new_struct(root, shape, dim, src->type);
  if (!self)
    return nullptr;

  if (CDS_REQUIRE(offset <= src->count &&
                  self->count <= src->count - offset))
    return nullptr;
  self->data = (unsigned char *)src->data + src->elem_size * offset;
  return self;
}

CdsError cds_copy(CdsArray *dst, const CdsArray *src) {
  CDS_TRY(CDS_REQUIRE(dst != nullptr && src != nullptr));
  CDS_TRY(CDS_REQUIRE(dst->count == src->count));
  CDS_TRY(CDS_REQUIRE(dst->type == src->type));

  const unsigned char *d = (const unsigned char *)dst->data;
  const unsigned char *s = (const unsigned char *)src->data;
  CDS_TRY(CDS_REQUIRE(d + dst->capacity <= s || s + src->capacity <= d));

  return CDS_CUDA(cudaMemcpy(dst->data, src->data, dst->capacity,
                             cudaMemcpyDeviceToDevice));
}

CdsCheckpoint cds_checkpoint(const Cds *root) {
  assert(root != nullptr);
  CdsCheckpoint cp = {arena_checkpoint(root->arena),
                      arena_checkpoint(root->host)};
  return cp;
}

CdsError cds_restore(Cds *root, CdsCheckpoint cp) {
  CDS_TRY(CDS_REQUIRE(root != nullptr));
  // Kernels still reading rewound memory must finish before it is reused.
  CDS_TRY(CDS_CUDA(cudaDeviceSynchronize()));
  arena_restore(root->arena, cp.device);
  arena_restore(root->host, cp.host);
  return CDS_OK;
}

CdsError cds_to_host(const CdsArray *self, void *dst) {
  CDS_TRY(CDS_REQUIRE(self != nullptr));
  CDS_TRY(CDS_REQUIRE(dst != nullptr));
  return CDS_CUDA(
      cudaMemcpy(dst, self->data, self->capacity, cudaMemcpyDeviceToHost));
}

CdsError cds_to_meta(const CdsArray *self, CdsArrayMeta *meta) {
  CDS_TRY(CDS_REQUIRE(self != nullptr));
  CDS_TRY(CDS_REQUIRE(meta != nullptr));
  meta->count = self->count;
  meta->dim = self->dim;
  meta->capacity = self->capacity;
  meta->elem_size = self->elem_size;
  meta->data = self->data;
  memcpy(meta->shape, self->shape, sizeof(size_t) * self->dim);
  return CDS_OK;
}

void cds_free(Cds *root) {
  if (root == nullptr)
    return;

  // Kernels still reading this arena must finish before its memory goes away.
  // A failed sync is recorded, and the memory is released regardless.
  (void)CDS_CUDA(cudaDeviceSynchronize());
  arena_free(root->arena);
  arena_free(root->host);
  ARENA_FREE(root);
}
