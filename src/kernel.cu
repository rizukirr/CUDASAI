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

void cds_add(const CdsContext *ctx, const CdsArray *a, const CdsArray *b,
             CdsArray *dest) {

  assert(ctx && a && b && dest);
  assert(b->type == a->type && dest->type == a->type);
  assert(a->dim == b->dim && a->dim == dest->dim);

  if (a->count == 0)
    return;

  for (size_t i = 0; i < a->dim; i++) {
    assert(a->shape[i] == b->shape[i]);
    assert(a->shape[i] == dest->shape[i]);
  }

  cds_add_lt(ctx, a, b, dest);
}

template <typename T>
void cds_scale(const CdsContext *ctx, const CdsArray *a, const T b,
               CdsArray *dest) {
  assert(ctx && a && dest);
  assert(a->type == dest->type);
  assert(a->dim == dest->dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "cds_scale requires float or double");

  if constexpr (std::is_same_v<T, float>) {
    assert(a->type == CDS_FLOAT32);
  } else {
    assert(a->type == CDS_FLOAT64);
  }

  if (a->count == 0)
    return;

  for (size_t i = 0; i < a->dim; i++)
    assert(a->shape[i] == dest->shape[i]);

  cds_scale_lt(ctx, a, b, dest);
}

template void cds_scale<float>(const CdsContext *ctx, const CdsArray *a,
                               const float b, CdsArray *dest);

template void cds_scale<double>(const CdsContext *ctx, const CdsArray *a,
                                const double b, CdsArray *dest);

void cds_fill(const CdsContext *ctx, CdsArray *dst, const void *value) {
  kernel_fill(ctx, dst, value);
}
