#ifndef CDS_ARRAY_H
#define CDS_ARRAY_H

#include "export.h"

typedef struct Cds Cds;

typedef enum {
  CDS_FLOAT32,
  CDS_FLOAT64,
} CdsType;

// Order is matters according to CdsType enum order
static constexpr size_t cds_type_size[] = {
    sizeof(float),
    sizeof(double),
};

[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API Cds *cds_new(const size_t *shape, size_t dim, CdsType type,
                     const void *data);

[[nodiscard]]
CDS_API __host__ __device__ size_t cds_count(Cds *self);

[[nodiscard]]
CDS_API __host__ __device__ void *cds_data_at(Cds *self, size_t index);

CDS_API void cds_free(Cds *self);

#endif // CUDASAI_ARRAY_H
