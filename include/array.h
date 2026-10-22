#ifndef CDS_ARRAY_H
#define CDS_ARRAY_H

#include "export.h"
#include <stddef.h>

typedef struct Cds Cds;           // a root: owns device memory
typedef struct CdsArray CdsArray; // an array living inside a root

typedef enum {
  CDS_FLOAT32,
  CDS_FLOAT64,
} CdsType;

// Indexed by CdsType, so the order must match the enum.
static constexpr size_t cds_type_size[] = {
    sizeof(float),
    sizeof(double),
};

/**
 * Create a root: an owner of device memory. Arrays are created inside it with
 * cds_array and all die together when the root is freed. `device_bytes` is
 * the arena block size, not a limit: the root grows by another block when an
 * array does not fit.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API Cds *cds_new(size_t device_bytes);

/**
 * Create an array inside `root`. `data`, when non-null, points to
 * count * sizeof(type) bytes of host memory that are copied to the device.
 * A null `data` zero-fills the array. The array is released by cds_free on
 * the root, never on its own.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API CdsArray *cds_array(Cds *root, const size_t *shape, size_t dim,
                       CdsType type, const void *data);

/**
 * Create an array inside `root` filled with uniform random values in
 * [-1/sqrt(fan_in), 1/sqrt(fan_in)], where fan_in is the last dimension of
 * `shape`. This is the PyTorch default initialisation for linear weights.
 * Draws from rand(), so call srand() first for a reproducible model.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API CdsArray *cds_array_rand(Cds *root, const size_t *shape, size_t dim,
                                 CdsType type);

[[nodiscard]]
CDS_API size_t cds_count(const CdsArray *self);

/**
 * Device address of element `index` in row-major order. The returned pointer
 * must only be dereferenced in device code.
 */
[[nodiscard]]
CDS_API __host__ __device__ void *cds_data_at(const CdsArray *self,
                                              size_t index);

/**
 * Copy the whole array from the device into `dst`, which must hold
 * cds_count(self) * sizeof(type) bytes. Blocks until the copy completes.
 */
CDS_API void cds_to_host(const CdsArray *self, void *dst);

/**
 * Release a root together with every array created inside it. Arrays have no
 * free of their own.
 */
CDS_API void cds_free(Cds *root);

#endif // CDS_ARRAY_H
