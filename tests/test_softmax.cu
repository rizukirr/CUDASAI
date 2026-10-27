#include "api_test_helpers.h"
#include <algorithm>

template <typename T>
static void check_softmax(Cds *root, CdsContext *context,
                          const std::vector<size_t> &shape,
                          const std::vector<T> &input) {
  size_t count = 1;
  for (size_t dimension : shape)
    count *= dimension;
  CHECK(input.size() == count);
  const size_t cols = shape.back();

  // Use sequential double precision arithmetic as the CPU reference.
  std::vector<T> expected(count);
  for (size_t base = 0; base < count; base += cols) {
    double maximum = double(input[base]);
    for (size_t col = 1; col < cols; ++col)
      maximum = std::max(maximum, double(input[base + col]));
    double sum = 0;
    for (size_t col = 0; col < cols; ++col)
      sum += std::exp(double(input[base + col]) - maximum);
    for (size_t col = 0; col < cols; ++col)
      expected[base + col] =
          T(std::exp(double(input[base + col]) - maximum) / sum);
  }

  CdsArray *x =
      cds_array(root, shape.data(), shape.size(), dtype<T>(), input.data());
  CdsArray *y =
      cds_array(root, shape.data(), shape.size(), dtype<T>(), nullptr);
  CHECK(x && y);
  const T sentinel = T(-99);
  cds_fill(context, y, &sentinel);
  cds_softmax(context, x, y);
  expect_values(y, expected);
  expect_values(x, input);

  // expect_values has synchronized the context's work before this readback.
  std::vector<T> actual(count);
  cds_to_host(y, actual.data());
  const double tolerance = sizeof(T) == sizeof(float) ? 1e-5 : 1e-12;
  for (size_t base = 0; base < count; base += cols) {
    double sum = 0;
    for (size_t col = 0; col < cols; ++col) {
      CHECK(actual[base + col] >= T(0));
      CHECK(actual[base + col] <= T(1));
      sum += double(actual[base + col]);
    }
    CHECK(std::abs(sum - 1.0) <= tolerance);
  }
}

template <typename T> void test_softmax(Cds *root, CdsContext *context) {
  for (const std::vector<size_t> &shape : {std::vector<size_t>{1},
                                           {3},
                                           {2, 31},
                                           {2, 32},
                                           {3, 33},
                                           {2, 257},
                                           {2, 3, 65},
                                           {2, 1025}}) {
    size_t count = 1;
    for (size_t dimension : shape)
      count *= dimension;
    const size_t cols = shape.back();
    std::vector<T> input(count);
    for (size_t row = 0; row < count / cols; ++row)
      for (size_t col = 0; col < cols; ++col) {
        const T shift = row % 2 ? T(1000) : T(-1000);
        input[row * cols + col] =
            shift + T(int((col * 7 + row * 3) % 29) - 14) / T(4);
      }
    check_softmax(root, context, shape, input);

    // Equal logits must produce a uniform distribution for every row width.
    std::fill(input.begin(), input.end(), T(1000));
    check_softmax(root, context, shape, input);
  }

  // A dominant logit may underflow the smaller probabilities to zero.
  check_softmax<T>(root, context, {3, 3},
                   {T(1000), T(0), T(-1000), T(-1002), T(-1001), T(-1000),
                    T(1000), T(1001), T(1002)});
}

template void test_softmax<float>(Cds *root, CdsContext *context);
template void test_softmax<double>(Cds *root, CdsContext *context);
