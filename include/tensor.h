#ifndef CDS_TENSOR_H
#define CDS_TENSOR_H

#include "array.h"
#include "export.h"

CDS_API Cds *cds_linear(Cds *data, size_t out_features, bool use_bias);

#endif // CDS_TENSOR_H
