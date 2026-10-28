#define ARENA_USE_CUDA
#include "_array.h"
#include "_kernel.h"
#include "array.h"
#include "kernel.h"
#include <assert.h>
#include <cstdlib>
#include <cublasLt.h>
#include <cublas_api.h>
#include <cuda_device_runtime_api.h>

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
                const CdsArray *b, CdsActivation activation, CdsArray *y) {
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

  cds_matmul_lt(ctx, activation, x, W, b, y);
}

void cds_softmax(const CdsContext *ctx, const CdsArray *x, CdsArray *y) {
  assert(ctx && x && y);
  assert(x->dim == y->dim);
  assert(x->type == y->type);
  assert(x->count == y->count);

  if (x->count == 0)
    return;

  for (size_t i = 0; i < x->dim; i++)
    assert(x->shape[i] == y->shape[i]);

  const size_t cols = x->shape[x->dim - 1];
  assert(cols > 0);

  const size_t rows = x->count / cols;
  assert(rows <= INT_MAX);
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
    abort();
  }

  CUDA_CHECK(cudaGetLastError());
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
  assert_type(b);

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
