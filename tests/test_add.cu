#include "api_test_helpers.h"

template <typename T> void test_add(Cds *root, CdsContext *context) {
  for (const std::vector<size_t> &shape :
       {std::vector<size_t>{1}, {257}, {3, 5}, {2, 3, 5}}) {
    size_t count = 1;
    for (size_t dimension : shape)
      count *= dimension;
    std::vector<T> a(count), b(count), expected(count);
    for (size_t i = 0; i < count; ++i) {
      a[i] = T(int(i % 17) - 8) / T(4);
      b[i] = T(int(i % 7) - 3) / T(2);
      expected[i] = a[i] + b[i];
    }
    CdsArray *left =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), a.data());
    CdsArray *right =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), b.data());
    CdsArray *dest =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), nullptr);
    CHECK(left && right && dest);
    cds_add(context, left, right, dest);
    expect_values(dest, expected);
    expect_values(left, a);
    expect_values(right, b);
  }
}

template void test_add<float>(Cds *root, CdsContext *context);
template void test_add<double>(Cds *root, CdsContext *context);
