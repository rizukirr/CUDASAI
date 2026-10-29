#include <cmath>
#define ARENA_USE_CUDA
#include "_array.h"
#include "_kernel.h"
#include "kernel.h"
#include <assert.h>
#include <cublasLt.h>
#include <cublas_api.h>
#include <stdint.h>

constexpr cudaDataType CDS_CUDA_TYPE(CdsType type) {
  switch (type) {
  case CDS_TYPE_FLOAT32:
    return CUDA_R_32F;
  case CDS_TYPE_FLOAT64:
    return CUDA_R_64F;
  // Unreachable: callers validate the type first.
  default:
    return CUDA_R_32F;
  }
}

constexpr cublasComputeType_t CDS_COMPUTE_TYPE(CdsType type) {
  switch (type) {
  case CDS_TYPE_FLOAT32:
    return CUBLAS_COMPUTE_32F;
  case CDS_TYPE_FLOAT64:
    return CUBLAS_COMPUTE_64F;
  // Unreachable: callers validate the type first.
  default:
    return CUBLAS_COMPUTE_32F;
  }
}

template <typename T> __device__ __forceinline__ T softmax_warp_max(T value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmax(value, __shfl_xor_sync(0xffffffffu, value, offset));
  return value;
}

template <typename T> __device__ __forceinline__ T softmax_warp_sum(T value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

CdsError kernel_fill(const CdsContext *ctx, CdsArray *dst, const void *value) {
  CDS_TRY(CDS_REQUIRE(ctx != nullptr && dst != nullptr && value != nullptr));

  const unsigned blocks = cds_blocks_for(dst->count);
  switch (dst->type) {
  case CDS_TYPE_FLOAT32:
    fill_kernel<<<blocks, CDS_THREAD_DEFAULT, 0, ctx->stream>>>(
        (float *)dst->data, dst->count, *((const float *)value));
    break;
  case CDS_TYPE_FLOAT64:
    fill_kernel<<<blocks, CDS_THREAD_DEFAULT, 0, ctx->stream>>>(
        (double *)dst->data, dst->count, *((const double *)value));
    break;
  default:
    return CDS_FAIL(CDS_ERR_INVALID, "unknown dtype %d", (int)dst->type);
  }

  return CDS_CUDA(cudaGetLastError());
}

static void matmul_epilogue(cublasLtEpilogue_t *epi, bool use_bias,
                            CdsActivation activation) {
  assert(epi != nullptr);

  cublasLtEpilogue_t default_epi;
  if (use_bias) {
    switch (activation) {
    case CDS_ACTIVATION_GELU:
      default_epi = CUBLASLT_EPILOGUE_GELU_BIAS;
      break;
    default:
      default_epi = CUBLASLT_EPILOGUE_BIAS;
      break;
    }
  } else {
    switch (activation) {
    case CDS_ACTIVATION_GELU:
      default_epi = CUBLASLT_EPILOGUE_GELU;
      break;
    default:
      default_epi = CUBLASLT_EPILOGUE_DEFAULT;
      break;
    }
  }

  *epi = default_epi;
}

CdsError cds_matmul_lt(const CdsContext *ctx, CdsActivation activation,
                       const CdsArray *x, const CdsArray *w,
                       const CdsArray *bias, CdsArray *y) {
  assert(ctx != nullptr);
  const int K = (int)x->shape[x->dim - 1];
  const int M = (int)((int)x->count / K);
  const int N = (int)w->shape[0];

  cudaDataType dtype = CDS_CUDA_TYPE(x->type);
  cublasComputeType_t ctype = CDS_COMPUTE_TYPE(x->type);
  cudaDataType scale_type = CDS_CUDA_TYPE(x->type);

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr,
                         c_layout = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
  cublasLtEpilogue_t epi;
  matmul_epilogue(&epi, bias != nullptr, activation);
  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  const float alpha_f = 1.0f, beta_f = 0.0f;
  const double alpha_d = 1.0, beta_d = 0.0;
  const void *alpha = x->type == CDS_TYPE_FLOAT32 ? (const void *)&alpha_f
                                                  : (const void *)&alpha_d;
  const void *beta = x->type == CDS_TYPE_FLOAT32 ? (const void *)&beta_f
                                                 : (const void *)&beta_d;

  CdsError err;
  if ((err = CDS_CUBLAS(cublasLtMatmulDescCreate(&op, ctype, scale_type))) ||
      (err = CDS_CUBLAS(cublasLtMatmulDescSetAttribute(
           op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)))) ||
      (err = CDS_CUBLAS(cublasLtMatmulDescSetAttribute(
           op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)))) ||
      (err = CDS_CUBLAS(cublasLtMatmulDescSetAttribute(
           op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)))))
    goto done;

  if (bias && (err = CDS_CUBLAS(cublasLtMatmulDescSetAttribute(
                   op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias->data,
                   sizeof(bias->data)))))
    goto done;

  if ((err = CDS_CUBLAS(cublasLtMatrixLayoutCreate(&a_layout, dtype, K, N, K))) ||
      (err = CDS_CUBLAS(cublasLtMatrixLayoutCreate(&b_layout, dtype, K, M, K))) ||
      (err = CDS_CUBLAS(cublasLtMatrixLayoutCreate(&c_layout, dtype, N, M, N))) ||
      (err = CDS_CUBLAS(cublasLtMatmulPreferenceCreate(&pref))) ||
      (err = CDS_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
           pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ctx->workspace_size,
           sizeof(ctx->workspace_size)))) ||
      (err = CDS_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
           ctx->lt, op, a_layout, b_layout, c_layout, c_layout, pref, 1, &heur,
           &found))))
    goto done;

  if (found == 0) {
    err = CDS_FAIL(CDS_ERR_INTERNAL, "cuBLASLt: no algorithm for M=%d N=%d K=%d",
                   M, N, K);
    goto done;
  }

  err = CDS_CUBLAS(cublasLtMatmul(ctx->lt, op, alpha, w->data, a_layout,
                                  x->data, b_layout, beta, y->data, c_layout,
                                  y->data, c_layout, &heur.algo, ctx->workspace,
                                  ctx->workspace_size, ctx->stream));

done:
  // Release every handle that was created, in reverse order, and report the
  // first failure.
  CdsError cleanup = CDS_OK;
  if (pref) {
    CdsError e = CDS_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (c_layout) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixLayoutDestroy(c_layout));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (b_layout) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixLayoutDestroy(b_layout));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (a_layout) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixLayoutDestroy(a_layout));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (op) {
    CdsError e = CDS_CUBLAS(cublasLtMatmulDescDestroy(op));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  return err != CDS_OK ? err : cleanup;
}

template <typename T>
__global__ void cds_softmax_lt(const T *x, T *y, size_t cols) {
  const size_t base = size_t(blockIdx.x) * cols;
  const unsigned lane = threadIdx.x;

  T maximum = -T(INFINITY);

  for (size_t col = lane; col < cols; col += 32)
    maximum = fmax(maximum, x[base + col]);

  maximum = softmax_warp_max(maximum);

  T sum = T(0);

  for (size_t col = lane; col < cols; col += 32) {
    T value = exp(x[base + col] - maximum);
    y[base + col] = value;
    sum += value;
  }

  sum = softmax_warp_sum(sum);

  for (size_t col = lane; col < cols; col += 32)
    y[base + col] /= sum;
}

template __global__ void cds_softmax_lt<float>(const float *x, float *y,
                                               size_t cols);

template __global__ void cds_softmax_lt<double>(const double *x, double *y,
                                                size_t cols);

CdsError cds_add_lt(const CdsContext *ctx, const CdsArray *a,
                    const CdsArray *b, CdsArray *dest) {
  assert(a->type == b->type);
  assert(a->type == dest->type);

  const uint64_t cols = a->shape[a->dim - 1];
  const uint64_t rows = a->count / cols;
  assert(cols <= INT64_MAX);

  cublasLtMatrixLayout_t layout = nullptr;
  cublasLtMatrixTransformDesc_t transform = nullptr;
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublasOperation_t op = CUBLAS_OP_N;
  const float one_f = 1.0f;
  const double one_d = 1.0;
  const void *one =
      a->type == CDS_TYPE_FLOAT32 ? (const void *)&one_f : (const void *)&one_d;

  CdsError err;
  if ((err = CDS_CUBLAS(cublasLtMatrixLayoutCreate(
           &layout, CDS_CUDA_TYPE(a->type), rows, cols, (int64_t)cols))) ||
      (err = CDS_CUBLAS(cublasLtMatrixLayoutSetAttribute(
           layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)))) ||
      (err = CDS_CUBLAS(cublasLtMatrixTransformDescCreate(
           &transform, CDS_CUDA_TYPE(a->type)))) ||
      (err = CDS_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
           transform, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)))) ||
      (err = CDS_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
           transform, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &op, sizeof(op)))))
    goto done;

  err = CDS_CUBLAS(cublasLtMatrixTransform(ctx->lt, transform, one, a->data,
                                           layout, one, b->data, layout,
                                           dest->data, layout, ctx->stream));

done:
  CdsError cleanup = CDS_OK;
  if (transform) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixTransformDescDestroy(transform));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (layout) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixLayoutDestroy(layout));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  return err != CDS_OK ? err : cleanup;
}

template <typename T>
CdsError cds_scale_lt(const CdsContext *ctx, const CdsArray *a, const T b,
                      CdsArray *dest) {
  assert(ctx != nullptr);

  const uint64_t cols = a->shape[a->dim - 1];
  const uint64_t rows = a->count / cols;
  assert(cols <= INT64_MAX);

  cublasLtMatrixLayout_t layout = nullptr;
  cublasLtMatrixTransformDesc_t transform = nullptr;
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublasOperation_t op = CUBLAS_OP_N;
  const T zero{0};

  CdsError err;
  if ((err = CDS_CUBLAS(cublasLtMatrixLayoutCreate(
           &layout, CDS_CUDA_TYPE(a->type), rows, cols, (int64_t)cols))) ||
      (err = CDS_CUBLAS(cublasLtMatrixLayoutSetAttribute(
           layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)))) ||
      (err = CDS_CUBLAS(cublasLtMatrixTransformDescCreate(
           &transform, CDS_CUDA_TYPE(a->type)))) ||
      (err = CDS_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
           transform, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)))))
    goto done;

  err = CDS_CUBLAS(cublasLtMatrixTransform(ctx->lt, transform, &b, a->data,
                                           layout, &zero, nullptr, nullptr,
                                           dest->data, layout, ctx->stream));

done:
  CdsError cleanup = CDS_OK;
  if (transform) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixTransformDescDestroy(transform));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  if (layout) {
    CdsError e = CDS_CUBLAS(cublasLtMatrixLayoutDestroy(layout));
    if (cleanup == CDS_OK)
      cleanup = e;
  }
  return err != CDS_OK ? err : cleanup;
}

template CdsError cds_scale_lt<float>(const CdsContext *ctx, const CdsArray *a,
                                      const float b, CdsArray *dest);

template CdsError cds_scale_lt<double>(const CdsContext *ctx,
                                       const CdsArray *a, const double b,
                                       CdsArray *dest);
