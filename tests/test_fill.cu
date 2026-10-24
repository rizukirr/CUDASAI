#include "api_test_helpers.h"

template <typename T> void test_fill(Cds *root, CdsContext *context) {
  for (size_t count :
       {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1025)}) {
    const size_t shape[] = {count};
    CdsArray *array = cds_array(root, shape, 1, dtype<T>(), nullptr);
    CHECK(array);
    for (T value : {T(-2.5), T(0), T(7.25)}) {
      cds_fill(context, array, &value);
      expect_values(array, std::vector<T>(count, value));
    }
  }
}

template void test_fill<float>(Cds *root, CdsContext *context);
template void test_fill<double>(Cds *root, CdsContext *context);
