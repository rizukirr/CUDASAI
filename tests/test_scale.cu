#include "api_test_helpers.h"

template <typename T> void test_scale(Cds *root, CdsContext *context) {
  for (const std::vector<size_t> &shape :
       {std::vector<size_t>{1}, {257}, {3, 5}, {2, 3, 5}}) {
    size_t count = 1;
    for (size_t dimension : shape)
      count *= dimension;
    std::vector<T> a(count), expected(count);
    for (size_t i = 0; i < count; ++i)
      a[i] = T(int(i % 17) - 8) / T(4);

    CdsArray *input =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), a.data());
    CdsArray *dest =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), nullptr);
    CHECK(input && dest);

    for (T scale : {T(-2.5), T(0), T(1), T(0.25), T(3)}) {
      for (size_t i = 0; i < count; ++i)
        expected[i] = a[i] * scale;
      cds_scale(context, input, scale, dest);
      expect_values(dest, expected);
      expect_values(input, a);
    }
  }
}

template void test_scale<float>(Cds *root, CdsContext *context);
template void test_scale<double>(Cds *root, CdsContext *context);
