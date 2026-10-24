#include "api_test_helpers.h"

template <typename T> void test_random(Cds *root) {
  const size_t shape[] = {32, 16};
  std::srand(123);
  CdsArray *first = cds_array_rand(root, shape, 2, dtype<T>());
  CHECK(first);
  CdsArrayMeta meta{};
  cds_to_meta(first, &meta);
  CHECK(meta.count == 512);
  std::vector<T> values(512);
  cds_to_host(first, values.data());
  bool different = false;
  for (T value : values) {
    CHECK(std::isfinite(value));
    CHECK(value >= T(-0.25) && value <= T(0.25));
    different = different || value != values[0];
  }
  CHECK(different);
  std::srand(123);
  CdsArray *second = cds_array_rand(root, shape, 2, dtype<T>());
  CHECK(second);
  std::vector<T> repeated(512);
  cds_to_host(second, repeated.data());
  CHECK(values == repeated);
}

template void test_random<float>(Cds *root);
template void test_random<double>(Cds *root);
