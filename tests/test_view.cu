#include "api_test_helpers.h"

template <typename T> void test_view(Cds *root) {
  const size_t shape[] = {6}, window[] = {2, 2}, pair[] = {2};
  std::vector<T> values = {0, 1, 2, 3, 4, 5};
  CdsArray *source = cds_array(root, shape, 1, dtype<T>(), values.data());
  CHECK(source);
  Cds *views = cds_new(64);
  CHECK(views);
  CdsArray *view = cds_view(views, source, 2, window, 2);
  CHECK(view);
  expect_values(view, std::vector<T>{2, 3, 4, 5});
  CdsArray *nested = cds_view(views, view, 1, pair, 1);
  CHECK(nested);
  expect_values(nested, std::vector<T>{3, 4});
  const T replacement[] = {T(-7), T(8)};
  CdsArray *replacement_array =
      cds_array(root, pair, 1, dtype<T>(), replacement);
  CHECK(replacement_array);
  cds_copy(nested, replacement_array);
  values[3] = replacement[0];
  values[4] = replacement[1];
  expect_values(source, values);
  cds_free(views);
  expect_values(source, values);
}

template void test_view<float>(Cds *root);
template void test_view<double>(Cds *root);
