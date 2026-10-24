#include "api_test_helpers.h"

template <typename T> void test_linear(Cds *root, CdsContext *context) {
  const size_t weights_shape[] = {4, 3}, bias_shape[] = {4};
  const std::vector<T> weights = {1, -2, 3, 4, 0, -1, -3, 2, 1, 2, 1, -4};
  const std::vector<T> bias = {T(0.5), T(-1.25), T(2), T(-3)};
  CdsArray *w = cds_array(root, weights_shape, 2, dtype<T>(), weights.data());
  CdsArray *b = cds_array(root, bias_shape, 1, dtype<T>(), bias.data());
  CHECK(w && b);
  for (const std::vector<size_t> &shape :
       {std::vector<size_t>{3}, {2, 3}, {2, 2, 3}}) {
    size_t batch = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
      batch *= shape[i];
    std::vector<size_t> output_shape = shape;
    output_shape.back() = 4;
    std::vector<T> input(batch * 3);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = T(int(i) - 4) / T(2);
    CdsArray *x =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), input.data());
    CdsArray *y = cds_array(root, output_shape.data(), output_shape.size(),
                            dtype<T>(), nullptr);
    CHECK(x && y);
    for (bool with_bias : {false, true}) {
      // A nonzero destination catches an incorrect accumulation coefficient.
      T sentinel = T(99);
      cds_fill(context, y, &sentinel);
      cds_linear(context, x, w, with_bias ? b : nullptr, y);
      std::vector<T> expected(batch * 4);
      for (size_t row = 0; row < batch; ++row)
        for (size_t out = 0; out < 4; ++out) {
          T value = with_bias ? bias[out] : T(0);
          for (size_t in = 0; in < 3; ++in)
            value += input[row * 3 + in] * weights[out * 3 + in];
          expected[row * 4 + out] = value;
        }
      expect_values(y, expected);
    }
    expect_values(x, input);
  }
  expect_values(w, weights);
  expect_values(b, bias);
}

template void test_linear<float>(Cds *root, CdsContext *context);
template void test_linear<double>(Cds *root, CdsContext *context);
