#include "api_test_helpers.h"
#include <limits>

template <typename T> void test_array(Cds *root) {
  size_t shape[] = {2, 3};
  const std::vector<T> values = {T(-3), T(0), T(1.5), T(4), T(-2.25), T(9)};
  CdsArray *array = cds_array(root, shape, 2, dtype<T>(), values.data());
  CHECK(array);
  expect_values(array, values);

  // Metadata must outlive the caller's shape buffer.
  shape[0] = 99;
  CdsArrayMeta meta{};
  cds_to_meta(array, &meta);
  CHECK(meta.count == 6);
  CHECK(meta.dim == 2);
  CHECK(meta.shape[0] == 2 && meta.shape[1] == 3);
  const size_t singleton[] = {1};
  CdsArray *zero = cds_array(root, singleton, 1, dtype<T>(), nullptr);
  CHECK(zero);
  expect_values(zero, std::vector<T>{0});
  const size_t large[] = {1025};
  CdsArray *grown = cds_array(root, large, 1, dtype<T>(), nullptr);
  CHECK(grown);
  expect_values(grown, std::vector<T>(1025, 0));
  expect_values(array, values);

  const size_t overflow[] = {std::numeric_limits<size_t>::max(), 2};
  CHECK(cds_array(root, overflow, 2, dtype<T>(), nullptr) == nullptr);
  CHECK(cds_last_error() == CDS_ERR_INVALID);
}

template void test_array<float>(Cds *root);
template void test_array<double>(Cds *root);
