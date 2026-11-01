#include "api_test_helpers.h"
#include "error.h"

// The call failed with `code` and the message names the failing check.
static void expect_error(CdsError got, CdsError code, const char *fragment) {
  CHECK(got == code);
  CHECK(cds_last_error() == code);
  if (!std::strstr(cds_last_error_message(), fragment)) {
    std::fprintf(stderr, "message \"%s\" lacks \"%s\"\n",
                 cds_last_error_message(), fragment);
    std::exit(1);
  }
}

template <typename T> void test_error(Cds *root, CdsContext *context) {
  const size_t two[] = {2}, three[] = {3}, matrix[] = {2, 3};
  CdsArray *a = cds_array(root, two, 1, dtype<T>(), nullptr);
  CdsArray *b = cds_array(root, three, 1, dtype<T>(), nullptr);
  CdsArray *c = cds_array(root, two, 1, dtype<T>(), nullptr);
  CdsArray *m = cds_array(root, matrix, 2, dtype<T>(), nullptr);
  CHECK(a && b && c && m);

  expect_error(cds_add(context, a, b, c), CDS_ERR_INVALID, "shape");
  expect_error(cds_copy(a, b), CDS_ERR_INVALID, "count");
  expect_error(cds_softmax(context, a, b), CDS_ERR_INVALID, "count");
  expect_error(cds_layernorm(context, a, nullptr, nullptr, 1e-5, b),
               CDS_ERR_INVALID, "count");
  expect_error(cds_layernorm(context, m, a, nullptr, 1e-5, m), CDS_ERR_INVALID,
               "gamma");
  expect_error(cds_layernorm(context, m, nullptr, a, 1e-5, m), CDS_ERR_INVALID,
               "beta");
  expect_error(cds_layernorm(context, m, nullptr, nullptr, -1.0, m),
               CDS_ERR_INVALID, "eps");
  expect_error(cds_linear(context, a, m, nullptr, CDS_ACTIVATION_NONE, c),
               CDS_ERR_INVALID, "in_features");
  expect_error(cds_to_host(a, nullptr), CDS_ERR_INVALID, "dst");

  const size_t w23[] = {2, 3}, w32[] = {3, 2}, w22[] = {2, 2};
  CdsArray *m23 = cds_array(root, w23, 2, dtype<T>(), nullptr);
  CdsArray *m32 = cds_array(root, w32, 2, dtype<T>(), nullptr);
  CdsArray *m22 = cds_array(root, w22, 2, dtype<T>(), nullptr);
  CHECK(m23 && m32 && m22);
  // [2, 3] · [2, 3] only fits once one side is flipped.
  expect_error(cds_matmul(context, m23, m23, false, false, m22),
               CDS_ERR_INVALID, "== K");
  expect_error(cds_matmul(context, m23, m23, true, true, m32), CDS_ERR_INVALID,
               "== K");
  // The flags change the output shape too.
  expect_error(cds_matmul(context, m23, m23, false, true, m32), CDS_ERR_INVALID,
               "out->shape");
  expect_error(cds_matmul(context, a, m23, false, false, m22), CDS_ERR_INVALID,
               "a->dim");
  expect_error(cds_matmul(context, m23, nullptr, false, false, m22),
               CDS_ERR_INVALID, "nullptr");

  const size_t row[] = {1};
  const int32_t ints[] = {1, 2};
  CdsArray *i2 = cds_array(root, two, 1, CDS_TYPE_INT32, ints);
  CdsArray *i2b = cds_array(root, two, 1, CDS_TYPE_INT32, nullptr);
  CdsArray *i2c = cds_array(root, two, 1, CDS_TYPE_INT32, nullptr);
  CdsArray *i1 = cds_array(root, row, 1, CDS_TYPE_INT32, nullptr);
  CHECK(i2 && i2b && i2c && i1);

  expect_error(cds_argmax(context, a, c), CDS_ERR_INVALID, "dest->type");
  expect_error(cds_argmax(context, m, i1), CDS_ERR_INVALID, "dest->count");
  expect_error(cds_argmax(context, i2, i1), CDS_ERR_INVALID, "not supported");

  // Integer arrays must not reach the floating point kernels.
  expect_error(cds_add(context, i2, i2b, i2c), CDS_ERR_INVALID, "float");
  expect_error(cds_scale<T>(context, i2, T(2), i2c), CDS_ERR_INVALID, "float");
  const size_t int_w_shape[] = {2, 2};
  CdsArray *int_w = cds_array(root, int_w_shape, 2, CDS_TYPE_INT32, nullptr);
  CHECK(int_w);
  expect_error(
      cds_linear(context, i2, int_w, nullptr, CDS_ACTIVATION_NONE, i2c),
      CDS_ERR_INVALID, "float");
  expect_error(cds_softmax(context, i2, i2c), CDS_ERR_INVALID, "dtype");
  CHECK(cds_array_rand(root, two, 1, CDS_TYPE_INT32) == nullptr);
  expect_error(cds_last_error(), CDS_ERR_INVALID, "not supported");

  CHECK(cds_view(root, a, 1, two, 1) == nullptr);
  expect_error(cds_last_error(), CDS_ERR_INVALID, "offset");

  // Below the size_t overflow limit but far beyond any GPU's memory.
  const size_t huge[] = {size_t(1) << 50};
  CHECK(cds_array(root, huge, 1, dtype<T>(), nullptr) == nullptr);
  expect_error(cds_last_error(), CDS_ERR_OUT_OF_MEMORY, "device allocation");

  // The pending CUDA error from the failed allocation must not leak into
  // later calls.
  CHECK(cds_copy(c, a) == CDS_OK);
  CdsArray *y = cds_array(root, matrix, 2, dtype<T>(), nullptr);
  CHECK(y);
  CHECK(cds_softmax(context, m, y) == CDS_OK);
  CUDA_REQUIRE(cudaDeviceSynchronize());
}

template void test_error<float>(Cds *root, CdsContext *context);
template void test_error<double>(Cds *root, CdsContext *context);
