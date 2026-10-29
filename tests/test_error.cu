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
  expect_error(cds_linear(context, a, m, nullptr, CDS_ACTIVATION_NONE, c),
               CDS_ERR_INVALID, "in_features");
  expect_error(cds_to_host(a, nullptr), CDS_ERR_INVALID, "dst");

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
