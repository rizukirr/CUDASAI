#define ARENA_USE_CUDA
#include "_array.h"
#include "tensor.h"
#include <assert.h>
#include <cublasLt.h>
#include <string.h>

Cds *cds_linear(Cds *data, size_t out_features, bool use_bias) {
  assert(out_features > 0);
  assert(data != nullptr);

  size_t w_shape[] = {out_features, data->shape[data->dim - 1]};
  Cds *weights = cds_new_rand(data->arena, w_shape, 2, data->type);
  if (!weights)
    return nullptr;

  Cds *bias = nullptr;
  if (use_bias) {
    size_t b_shape[] = {out_features};
    bias = cds_new_rand(data->arena, b_shape, 1, data->type);
    if (!bias)
      return nullptr;
  }

  size_t output_shape[CDS_MAX_DIM];
  memcpy(output_shape, data->shape, sizeof(size_t) * data->dim);
  output_shape[data->dim - 1] = out_features;

  // vibekit: forward pass not implemented yet, output stays zero-filled.
  return cds_new_tmp(data->arena, output_shape, data->dim, data->type);
}
