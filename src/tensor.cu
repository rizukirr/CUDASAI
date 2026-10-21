#include <cublas_api.h>
#define ARENA_USE_CUDA
#include "_array.h"
#include "tensor.h"
#include <assert.h>
#include <cublasLt.h>
#include <string.h>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t cds_st_ = (call);                                           \
    if (cds_st_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "%s:%d: cuBLASLt error %d\n", __FILE__, __LINE__,        \
              (int)cds_st_);                                                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static cublasLtHandle_t cds_lt_handle(void) {
  static cublasLtHandle_t handle = nullptr;
  if (!handle)
    CUBLAS_CHECK(cublasLtCreate(&handle));
  return handle;
}

static void cds_matmul_lt(const Cds *x, const Cds *w, Cds *bias, Cds *y) {
  const int K = (int)x->shape[x->dim - 1];
  const int M = (int)((int)x->count / K);
  const int N = (int)w->shape[0];

  cudaDataType dtype = x->type == CDS_FLOAT32 ? CUDA_R_32F : CUDA_R_64F;
  cublasComputeType_t ctype =
      x->type == CDS_FLOAT32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_64F;
  cudaDataType scale_type = x->type == CDS_FLOAT32 ? CUDA_R_32F : CUDA_R_64F;

  cublasLtMatmulDesc_t op;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, ctype, scale_type));
  cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &t, sizeof(t)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &n, sizeof(n)));

  if (bias) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias->data, sizeof(bias->data)));
  }

  cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout, dtype, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout, dtype, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_layout, dtype, N, M, N));

  cublasLtMatmulPreference_t pref;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspace = 0;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace)));

  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(cds_lt_handle(), op, a_layout,
                                              b_layout, c_layout, c_layout,
                                              pref, 1, &heur, &found));

  if (found == 0) {
    fprintf(stderr, "cuBLASLt: no algorithm for M=%d N=%d K=%d\n", M, N, K);
    abort();
  }

  const float alpha_f = 1.0f, beta_f = 0.0f;
  const double alpha_d = 1.0, beta_d = 0.0;
  const void *alpha =
      x->type == CDS_FLOAT32 ? (const void *)&alpha_f : (const void *)&alpha_d;
  const void *beta =
      x->type == CDS_FLOAT32 ? (const void *)&beta_f : (const void *)&beta_d;

  CUBLAS_CHECK(cublasLtMatmul(
      cds_lt_handle(), op, alpha, w->data, a_layout, x->data, b_layout, beta,
      y->data, c_layout, y->data, c_layout, &heur.algo, nullptr, 0, nullptr));

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
}

Cds *cds_linear(Cds *data, size_t out_features, bool use_bias) {
  assert(out_features > 0);
  assert(data != nullptr);

  size_t w_shape[] = {out_features, data->shape[data->dim - 1]};
  Cds *weights = cds_new_rand(data, w_shape, 2, data->type);
  if (!weights)
    return nullptr;

  Cds *bias = nullptr;
  if (use_bias) {
    size_t b_shape[] = {out_features};
    bias = cds_new_rand(data, b_shape, 1, data->type);
    if (!bias)
      return nullptr;
  }

  size_t output_shape[CDS_MAX_DIM];
  memcpy(output_shape, data->shape, sizeof(size_t) * data->dim);
  output_shape[data->dim - 1] = out_features;

  Cds *output = cds_new_tmp(data, output_shape, data->dim, data->type);
  if (!output)
    return nullptr;

  cds_matmul_lt(data, weights, bias, output);
  return output;
}
