#define ARENA_USE_CUDA
#include "_array.h"
#include "_kernel.h"
#include "array.h"
#include "kernel.h"
#include <assert.h>
#include <cstdlib>
#include <limits.h>
#include <cublasLt.h>
#include <cublas_api.h>
#include <cuda_device_runtime_api.h>

CdsContext *cds_context_new(size_t workspace_bytes) {
  CdsContext *self = (CdsContext *)ARENA_MALLOC(sizeof(CdsContext));
  if (!self) {
    CDS_FAIL(CDS_ERR_OUT_OF_MEMORY, "host allocation of CdsContext failed");
    return nullptr;
  }
  self->stream = nullptr;
  self->lt = nullptr;
  self->workspace = nullptr;
  self->workspace_size = workspace_bytes;

  // The stream comes first: creating it initialises the device context that
  // cublasLtCreate probes for.
  if (CDS_CUDA(cudaStreamCreate(&self->stream)) ||
      CDS_CUBLAS(cublasLtCreate(&self->lt)) ||
      (workspace_bytes > 0 &&
       CDS_CUDA(cudaMalloc(&self->workspace, workspace_bytes)))) {
    // Release what was created. The raw calls leave the recorded failure as
    // the last error.
    if (self->lt)
      (void)cublasLtDestroy(self->lt);
    if (self->stream)
      (void)cudaStreamDestroy(self->stream);
    ARENA_FREE(self);
    return nullptr;
  }

  return self;
}

void cds_context_free(CdsContext *ctx) {
  if (ctx == nullptr)
    return;

  // Every step runs even if an earlier one fails, so nothing leaks.
  (void)CDS_CUDA(cudaStreamSynchronize(ctx->stream));
  (void)CDS_CUDA(cudaFree(ctx->workspace));
  (void)CDS_CUDA(cudaStreamDestroy(ctx->stream));
  (void)CDS_CUBLAS(cublasLtDestroy(ctx->lt));
  ARENA_FREE(ctx);
}

CdsError cds_linear(const CdsContext *ctx, const CdsArray *x,
                    const CdsArray *W, const CdsArray *b,
                    CdsActivation activation, CdsArray *y) {
  CDS_TRY(CDS_REQUIRE(ctx != nullptr && x != nullptr && W != nullptr &&
                      y != nullptr));
  CDS_TRY(CDS_REQUIRE(W->dim == 2));
  CDS_TRY(CDS_REQUIRE(W->type == x->type && y->type == x->type));

  const size_t in_features = x->shape[x->dim - 1];
  const size_t out_features = W->shape[0];
  CDS_TRY(CDS_REQUIRE(in_features > 0));
  const size_t batch = x->count / in_features;
  CDS_TRY(CDS_REQUIRE(W->shape[1] == in_features));
  CDS_TRY(CDS_REQUIRE(y->shape[y->dim - 1] == out_features));
  CDS_TRY(CDS_REQUIRE(y->count == batch * out_features));
  if (b) {
    CDS_TRY(CDS_REQUIRE(b->type == x->type));
    CDS_TRY(CDS_REQUIRE(b->count == out_features));
  }

  return cds_matmul_lt(ctx, activation, x, W, b, y);
}

CdsError cds_softmax(const CdsContext *ctx, const CdsArray *x, CdsArray *y) {
  CDS_TRY(CDS_REQUIRE(ctx != nullptr && x != nullptr && y != nullptr));
  CDS_TRY(CDS_REQUIRE(x->dim == y->dim));
  CDS_TRY(CDS_REQUIRE(x->type == y->type));
  CDS_TRY(CDS_REQUIRE(x->count == y->count));

  if (x->count == 0)
    return CDS_OK;

  for (size_t i = 0; i < x->dim; i++)
    CDS_TRY(CDS_REQUIRE(x->shape[i] == y->shape[i]));

  const size_t cols = x->shape[x->dim - 1];
  CDS_TRY(CDS_REQUIRE(cols > 0));

  const size_t rows = x->count / cols;
  CDS_TRY(CDS_REQUIRE(rows <= INT_MAX));
  const unsigned blocks = static_cast<unsigned>(rows);

  switch (x->type) {
  case CDS_TYPE_FLOAT32:
    cds_softmax_lt<float>
        <<<blocks, 32, 0, ctx->stream>>>(static_cast<const float *>(x->data),
                                         static_cast<float *>(y->data), cols);
    break;
  case CDS_TYPE_FLOAT64:
    cds_softmax_lt<double>
        <<<blocks, 32, 0, ctx->stream>>>(static_cast<const double *>(x->data),
                                         static_cast<double *>(y->data), cols);
    break;
  default:
    return CDS_FAIL(CDS_ERR_INVALID, "unknown dtype %d", (int)x->type);
  }

  return CDS_CUDA(cudaGetLastError());
}

CdsError cds_add(const CdsContext *ctx, const CdsArray *a, const CdsArray *b,
                 CdsArray *dest) {
  CDS_TRY(CDS_REQUIRE(ctx != nullptr && a != nullptr && b != nullptr &&
                      dest != nullptr));
  CDS_TRY(CDS_REQUIRE(b->type == a->type && dest->type == a->type));
  CDS_TRY(CDS_REQUIRE(a->dim == b->dim && a->dim == dest->dim));

  if (a->count == 0)
    return CDS_OK;

  for (size_t i = 0; i < a->dim; i++) {
    CDS_TRY(CDS_REQUIRE(a->shape[i] == b->shape[i]));
    CDS_TRY(CDS_REQUIRE(a->shape[i] == dest->shape[i]));
  }

  return cds_add_lt(ctx, a, b, dest);
}

template <typename T>
CdsError cds_scale(const CdsContext *ctx, const CdsArray *a, const T b,
                   CdsArray *dest) {
  CDS_TRY(CDS_REQUIRE(ctx != nullptr && a != nullptr && dest != nullptr));
  CDS_TRY(CDS_REQUIRE(a->type == dest->type));
  CDS_TRY(CDS_REQUIRE(a->dim == dest->dim));
  assert_type(b);

  if (a->count == 0)
    return CDS_OK;

  for (size_t i = 0; i < a->dim; i++)
    CDS_TRY(CDS_REQUIRE(a->shape[i] == dest->shape[i]));

  return cds_scale_lt(ctx, a, b, dest);
}

template CdsError cds_scale<float>(const CdsContext *ctx, const CdsArray *a,
                                   const float b, CdsArray *dest);

template CdsError cds_scale<double>(const CdsContext *ctx, const CdsArray *a,
                                    const double b, CdsArray *dest);

CdsError cds_fill(const CdsContext *ctx, CdsArray *dst, const void *value) {
  return kernel_fill(ctx, dst, value);
}
