#include "api_test_helpers.h"

template <typename T>
static void
check_layernorm(Cds *root, CdsContext *context,
                const std::vector<size_t> &shape, const std::vector<T> &input,
                const std::vector<T> *gamma, const std::vector<T> *beta,
                double eps, bool in_place) {
  size_t count = 1;
  for (size_t dimension : shape)
    count *= dimension;
  CHECK(input.size() == count);
  const size_t cols = shape.back();

  // Use sequential double precision arithmetic as the CPU reference.
  std::vector<T> expected(count);
  for (size_t base = 0; base < count; base += cols) {
    double mean = 0;
    for (size_t col = 0; col < cols; ++col)
      mean += double(input[base + col]);
    mean /= double(cols);
    double var = 0;
    for (size_t col = 0; col < cols; ++col) {
      const double d = double(input[base + col]) - mean;
      var += d * d;
    }
    var /= double(cols);
    for (size_t col = 0; col < cols; ++col) {
      double v = (double(input[base + col]) - mean) / std::sqrt(var + eps);
      if (gamma)
        v *= double((*gamma)[col]);
      if (beta)
        v += double((*beta)[col]);
      expected[base + col] = T(v);
    }
  }

  const size_t param_shape[] = {cols};
  CdsArray *g = gamma
                    ? cds_array(root, param_shape, 1, dtype<T>(), gamma->data())
                    : nullptr;
  CdsArray *b = beta ? cds_array(root, param_shape, 1, dtype<T>(), beta->data())
                     : nullptr;
  CHECK(!gamma || g);
  CHECK(!beta || b);

  CdsArray *x =
      cds_array(root, shape.data(), shape.size(), dtype<T>(), input.data());
  CHECK(x);
  CdsArray *y = x;
  if (!in_place) {
    y = cds_array(root, shape.data(), shape.size(), dtype<T>(), nullptr);
    CHECK(y);
    const T sentinel = T(-99);
    cds_fill(context, y, &sentinel);
  }

  CHECK(cds_layernorm(context, x, g, b, eps, y) == CDS_OK);
  // Float rounding in the row mean is amplified by 1/std, so allow more slack
  // than the elementwise kernels need.
  expect_values(y, expected, sizeof(T) == sizeof(float) ? 1e-4 : 1e-10);
  if (!in_place)
    expect_values(x, input);
  if (gamma)
    expect_values(g, *gamma);
  if (beta)
    expect_values(b, *beta);
}

template <typename T> void test_layernorm(Cds *root, CdsContext *context) {
  constexpr double eps = 1e-5;

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

    // Odd rows sit far from zero, so a variance computed as E[x^2] - mean^2
    // would lose most of its precision.
    std::vector<T> input(count);
    for (size_t row = 0; row < count / cols; ++row)
      for (size_t col = 0; col < cols; ++col) {
        const T shift = row % 2 ? T(100) : T(-3);
        input[row * cols + col] =
            shift + T(int((col * 7 + row * 3) % 29) - 14) / T(4);
      }

    std::vector<T> gamma(cols), beta(cols);
    for (size_t col = 0; col < cols; ++col) {
      gamma[col] = T(int(col % 5) - 2) / T(2) + T(1);
      beta[col] = T(int(col % 3) - 1) / T(4);
    }

    check_layernorm<T>(root, context, shape, input, nullptr, nullptr, eps,
                       false);
    check_layernorm<T>(root, context, shape, input, &gamma, nullptr, eps,
                       false);
    check_layernorm<T>(root, context, shape, input, nullptr, &beta, eps, false);
    check_layernorm<T>(root, context, shape, input, &gamma, &beta, eps, false);
    check_layernorm<T>(root, context, shape, input, &gamma, &beta, eps, true);

    // A constant row has zero variance: eps keeps the division finite and
    // every output collapses to beta.
    std::fill(input.begin(), input.end(), T(7));
    check_layernorm<T>(root, context, shape, input, &gamma, &beta, eps, false);
  }

  // Hand-computed: mean 2.5, variance 1.25, so y = (x - 2.5) / sqrt(1.25).
  const size_t shape[] = {4};
  const std::vector<T> input = {T(1), T(2), T(3), T(4)};
  CdsArray *x = cds_array(root, shape, 1, dtype<T>(), input.data());
  CdsArray *y = cds_array(root, shape, 1, dtype<T>(), nullptr);
  CHECK(x && y);
  CHECK(cds_layernorm(context, x, nullptr, nullptr, 0.0, y) == CDS_OK);
  const T s = T(1 / std::sqrt(1.25));
  expect_values(
      y, std::vector<T>{T(-1.5) * s, T(-0.5) * s, T(0.5) * s, T(1.5) * s});
}

template void test_layernorm<float>(Cds *root, CdsContext *context);
template void test_layernorm<double>(Cds *root, CdsContext *context);
