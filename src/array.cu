#define ARENA_USE_CUDA
#define ARENA_IMPLEMENTATION
#include "_array.h"
#include <assert.h>
#include <math.h>

Cds *cds_new(size_t device_bytes) {
  Cds *self = (Cds *)ARENA_MALLOC(sizeof(Cds));
  if (!self)
    return nullptr;

  self->arena = arena_create(device_bytes);
  if (!self->arena) {
    ARENA_FREE(self);
    return nullptr;
  }

  // Structs of arrays created inside this root live here and die with it.
  self->host = arena_create_with(16 * sizeof(CdsArray), arena_host_malloc,
                                 arena_host_free);
  if (!self->host) {
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

  if (data == nullptr)
    CUDA_CHECK(cudaMemset(self->data, 0, self->capacity));
  else
    CUDA_CHECK(
        cudaMemcpy(self->data, data, self->capacity, cudaMemcpyHostToDevice));

  return self;
}

CdsArray *cds_array_rand(Cds *root, const size_t *shape, size_t dim,
                         CdsType type) {
  CdsArray *self = cds_new_derived(root, shape, dim, type);
  if (!self)
    return nullptr;

  void *staging = ARENA_MALLOC(self->capacity);
  if (!staging)
    return nullptr;

  double bound = 1.0 / sqrt((double)self->shape[self->dim - 1]);
  switch (self->type) {
  case CDS_FLOAT32:
    random_fill((float *)staging, self->count, (float)-bound, (float)bound);
    break;
  case CDS_FLOAT64:
    random_fill((double *)staging, self->count, -bound, bound);
    break;
  }

  CUDA_CHECK(
      cudaMemcpy(self->data, staging, self->capacity, cudaMemcpyHostToDevice));
  ARENA_FREE(staging);
  return self;
}

CdsArray *cds_view(Cds *root, const CdsArray *src, size_t offset,
                   const size_t *shape, size_t dim) {
  assert(src != nullptr);

  CdsArray *self = cds_new_struct(root, shape, dim, src->type);
  if (!self)
    return nullptr;

  assert(offset + self->count <= src->count && "view runs past the source");
  self->data = cds_data_at(src, offset);
  return self;
}

void cds_copy(CdsArray *dst, const CdsArray *src) {
  assert(dst != nullptr);
  assert(src != nullptr);
  assert(dst->count == src->count && "copy count mismatch");
  assert(dst->type == src->type && "copy type mismatch");

  const unsigned char *d = (const unsigned char *)dst->data;
  const unsigned char *s = (const unsigned char *)src->data;
  assert((d + dst->capacity <= s || s + src->capacity <= d) &&
         "copy ranges overlap");

  CUDA_CHECK(cudaMemcpy(dst->data, src->data, dst->capacity,
                        cudaMemcpyDeviceToDevice));
}

CdsCheckpoint cds_checkpoint(const Cds *root) {
  assert(root != nullptr);
  CdsCheckpoint cp = {arena_checkpoint(root->arena),
                      arena_checkpoint(root->host)};
  return cp;
}

void cds_restore(Cds *root, CdsCheckpoint cp) {
  assert(root != nullptr);
  // Kernels still reading rewound memory must finish before it is reused.
  CUDA_CHECK(cudaDeviceSynchronize());
  arena_restore(root->arena, cp.device);
  arena_restore(root->host, cp.host);
}

__host__ __device__ size_t cds_count(const CdsArray *self) {
  return self->count;
}

__host__ __device__ void *cds_data_at(const CdsArray *self, size_t index) {
  return (unsigned char *)self->data + self->elem_size * index;
}

void cds_to_host(const CdsArray *self, void *dst) {
  assert(self != nullptr);
  assert(dst != nullptr);
  CUDA_CHECK(
      cudaMemcpy(dst, self->data, self->capacity, cudaMemcpyDeviceToHost));
}

void cds_free(Cds *root) {
  if (root == nullptr)
    return;

  // Kernels still reading this arena must finish before its memory goes away.
  CUDA_CHECK(cudaDeviceSynchronize());
  arena_free(root->arena);
  arena_free(root->host);
  ARENA_FREE(root);
}
