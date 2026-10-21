#include "_array.h"
#include "array.h"

#include "tensor.h"
#include <assert.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <math.h>

Cds *cds_linear(Cds *data, size_t out_features, bool use_bias) {
  assert(out_features > 0);
  assert(data != nullptr);

  size_t w_shape[] = {out_features, data->shape[data->dim - 1]};
  Cds *weights = cds_new_rand(data->arena, w_shape, 2, data->type);

  size_t b_shape[] = {out_features};
  Cds *bias = cds_new_rand(data->arena, b_shape, 1, data->type);

  size_t output_shape[CDS_MAX_DIM];
  memcpy(output_shape, data->shape, sizeof(size_t) * data->dim);
  output_shape[data->dim - 1] = out_features;

  Cds *output = cds_new_tmp(data->arena, output_shape, data->dim, data->type);

  return output;
}
