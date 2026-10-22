#define ARENA_USE_CUDA
#define ARENA_IMPLEMENTATION
#include "_array.h"
#include <assert.h>

Cds *cds_new(size_t device_bytes) {
  Cds *self = (Cds *)calloc(1, sizeof(Cds));
  if (!self)
    return nullptr;

  self->arena = arena_create(device_bytes);
  if (!self->arena) {
    free(self);
    return nullptr;
  }

  // Structs of arrays created inside this root live here and die with it.
  self->host = arena_create_with(16 * sizeof(Cds), arena_host_malloc,
                                 arena_host_free);
  if (!self->host) {
    arena_free(self->arena);
    free(self);
    return nullptr;
  }

  self->owner = true;
  return self;
}

Cds *cds_array(Cds *root, const size_t *shape, size_t dim, CdsType type,
               const void *data) {
  Cds *self = cds_new_derived(root, shape, dim, type);
  if (!self)
    return nullptr;

  if (data == nullptr)
    CUDA_CHECK(cudaMemset(self->data, 0, self->capacity));
  else
    CUDA_CHECK(
        cudaMemcpy(self->data, data, self->capacity, cudaMemcpyHostToDevice));

  return self;
}

size_t cds_count(const Cds *self) { return self->count; }

__host__ __device__ void *cds_data_at(const Cds *self, size_t index) {
  return (unsigned char *)self->data + self->elem_size * index;
}

void cds_to_host(const Cds *self, void *dst) {
  assert(self != nullptr);
  assert(self->data != nullptr && "a root holds no data");
  assert(dst != nullptr);
  CUDA_CHECK(
      cudaMemcpy(dst, self->data, self->capacity, cudaMemcpyDeviceToHost));
}

void cds_free(Cds *self) {
  if (self == nullptr || !self->owner)
    return;

  // Kernels still reading this arena must finish before its memory goes away.
  CUDA_CHECK(cudaDeviceSynchronize());
  arena_free(self->arena);
  arena_free(self->host);
  free(self);
}
