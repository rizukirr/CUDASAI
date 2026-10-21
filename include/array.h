#ifndef CDS_ARRAY_H
#define CDS_ARRAY_H

#include "export.h"
#include <stddef.h>

typedef struct Cds Cds;

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
 * Create an array on the device. `data`, when non-null, points to
 * count * sizeof(type) bytes of host memory that are copied to the device.
 * A null `data` zero-fills the array.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API Cds *cds_new(const size_t *shape, size_t dim, CdsType type,
                     const void *data);

[[nodiscard]]
CDS_API size_t cds_count(const Cds *self);

/**
 * Device address of element `index` in row-major order. The returned pointer
 * must only be dereferenced in device code.
 */
[[nodiscard]]
CDS_API __host__ __device__ void *cds_data_at(const Cds *self, size_t index);

/**
 * Copy the whole array from the device into `dst`, which must hold
 * cds_count(self) * sizeof(type) bytes. Blocks until the copy completes.
 */
CDS_API void cds_to_host(const Cds *self, void *dst);

CDS_API void cds_free(Cds *self);

#endif // CDS_ARRAY_H
