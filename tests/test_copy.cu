#include "api_test_helpers.h"

template <typename T> void test_copy(Cds *root) {
  const size_t flat[] = {6}, matrix[] = {2, 3};
  const std::vector<T> values = {1, -2, 3, -4, 5, -6};
  CdsArray *source = cds_array(root, flat, 1, dtype<T>(), values.data());
  CdsArray *dest = cds_array(root, matrix, 2, dtype<T>(), nullptr);
  CHECK(source && dest);
  cds_copy(dest, source);
  expect_values(dest, values);
  expect_values(source, values);
  const size_t total[] = {12};
  CdsArray *storage = cds_array(root, total, 1, dtype<T>(), nullptr);
  CHECK(storage);
  CdsArray *left = cds_view(root, storage, 0, flat, 1);
  CdsArray *right = cds_view(root, storage, 6, flat, 1);
  CHECK(left && right);
  cds_copy(left, source);
  cds_copy(right, left);
  std::vector<T> doubled = values;
  doubled.insert(doubled.end(), values.begin(), values.end());
  expect_values(storage, doubled);
}

template void test_copy<float>(Cds *root);
template void test_copy<double>(Cds *root);
