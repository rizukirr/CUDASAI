#ifndef CDS_TENSOR_H
#define CDS_TENSOR_H

#include "array.h"
#include "export.h"

/**
 * y = x · W^T + b, row-major, through cuBLASLt.
 *
 * x is [..., in_features], W is [out_features, in_features], b is
 * [out_features] or null, and y is [..., out_features] with the same leading
 * dimensions as x. All four share one dtype. Nothing is allocated: the caller
 * owns every array, so W and b can live for the whole program while x and y
 * are created per batch.
 */
CDS_API void cds_linear(const Cds *x, const Cds *W, const Cds *b, Cds *y);

#endif // CDS_TENSOR_H
