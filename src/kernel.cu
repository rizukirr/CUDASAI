#define ARENA_USE_CUDA
#include "_array.h"
#include "_kernel.h"
#include "kernel.h"
#include <assert.h>
#include <cublasLt.h>
#include <cublas_api.h>

CdsContext *cds_context_new(size_t workspace_bytes) {
  CdsContext *self = (CdsContext *)ARENA_MALLOC(sizeof(CdsContext));
  if (!self)
    return nullptr;

  // The stream comes first: creating it initialises the device context that
  // cublasLtCreate probes for.
  CUDA_CHECK(cudaStreamCreate(&self->stream));
  CUBLAS_CHECK(cublasLtCreate(&self->lt));

  self->workspace = nullptr;
  self->workspace_size = workspace_bytes;
  if (workspace_bytes > 0)
    CUDA_CHECK(cudaMalloc(&self->workspace, workspace_bytes));

  return self;
}

void cds_context_free(CdsContext *ctx) {
  if (ctx == nullptr)
    return;

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
  CUDA_CHECK(cudaFree(ctx->workspace));
  CUDA_CHECK(cudaStreamDestroy(ctx->stream));
  CUBLAS_CHECK(cublasLtDestroy(ctx->lt));
  ARENA_FREE(ctx);
}

void cds_linear(const CdsContext *ctx, const CdsArray *x, const CdsArray *W,
                const CdsArray *b, CdsArray *y) {
  assert(x != nullptr);
  assert(W != nullptr);
  assert(y != nullptr);
  assert(W->dim == 2);
  assert(W->type == x->type && y->type == x->type);

  const size_t in_features = x->shape[x->dim - 1];
  const size_t out_features = W->shape[0];
  const size_t batch = x->count / in_features;
  assert(W->shape[1] == in_features);
  assert(y->shape[y->dim - 1] == out_features);
  assert(y->count == batch * out_features);
  if (b) {
    assert(b->type == x->type);
    assert(b->count == out_features);
  }

  cds_matmul_lt(ctx, x, W, b, y);
}

void cds_fill(const CdsContext *ctx, CdsArray *dst, const void *value) {
  kernel_fill(ctx, dst, value);
}
