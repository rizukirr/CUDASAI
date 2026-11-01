#include "api_test_helpers.h"

template <typename T>
static void check_matmul(Cds *root, CdsContext *context, size_t M, size_t K,
                         size_t N, bool transpose_a, bool transpose_b) {
  // Stored shapes before the flags are applied.
  const size_t a_shape[] = {transpose_a ? K : M, transpose_a ? M : K};
  const size_t b_shape[] = {transpose_b ? N : K, transpose_b ? K : N};
  const size_t out_shape[] = {M, N};

  std::vector<T> a_host(M * K), b_host(K * N);
  for (size_t i = 0; i < a_host.size(); ++i)
    a_host[i] = T(int(i * 7 % 11) - 5) / T(2);
  for (size_t i = 0; i < b_host.size(); ++i)
    b_host[i] = T(int(i * 3 % 13) - 6) / T(4);

  // The CPU triple loop, reading through the flags.
  std::vector<T> expected(M * N);
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j) {
      double sum = 0;
      for (size_t k = 0; k < K; ++k) {
        const T left = transpose_a ? a_host[k * M + i] : a_host[i * K + k];
        const T right = transpose_b ? b_host[j * K + k] : b_host[k * N + j];
        sum += double(left) * double(right);
      }
      expected[i * N + j] = T(sum);
    }

  CdsArray *a = cds_array(root, a_shape, 2, dtype<T>(), a_host.data());
  CdsArray *b = cds_array(root, b_shape, 2, dtype<T>(), b_host.data());
  CdsArray *out = cds_array(root, out_shape, 2, dtype<T>(), nullptr);
  CHECK(a && b && out);
  const T sentinel = T(-99);
  cds_fill(context, out, &sentinel);

  CHECK(cds_matmul(context, a, b, transpose_a, transpose_b, out) == CDS_OK);
  expect_values(out, expected);
  expect_values(a, a_host);
  expect_values(b, b_host);
}

template <typename T> void test_matmul(Cds *root, CdsContext *context) {
  const size_t sizes[][3] = {{5, 7, 3}, {3, 7, 5}, {1, 1, 1},   {1, 9, 1},
                             {9, 1, 9}, {4, 4, 4}, {64, 33, 17}};
  for (const auto &size : sizes)
    for (bool transpose_a : {false, true})
      for (bool transpose_b : {false, true})
        check_matmul<T>(root, context, size[0], size[1], size[2], transpose_a,
                        transpose_b);

  // Hand-computed, including the flip of b that attention's Q · K^T needs.
  const size_t shape[] = {2, 2};
  const std::vector<T> left = {T(1), T(2), T(3), T(4)};
  const std::vector<T> right = {T(5), T(6), T(7), T(8)};
  CdsArray *a = cds_array(root, shape, 2, dtype<T>(), left.data());
  CdsArray *b = cds_array(root, shape, 2, dtype<T>(), right.data());
  CdsArray *out = cds_array(root, shape, 2, dtype<T>(), nullptr);
  CHECK(a && b && out);
  CHECK(cds_matmul(context, a, b, false, false, out) == CDS_OK);
  expect_values(out, std::vector<T>{T(19), T(22), T(43), T(50)});
  CHECK(cds_matmul(context, a, b, false, true, out) == CDS_OK);
  expect_values(out, std::vector<T>{T(17), T(23), T(39), T(53)});
  CHECK(cds_matmul(context, a, b, true, false, out) == CDS_OK);
  expect_values(out, std::vector<T>{T(26), T(30), T(38), T(44)});
  CHECK(cds_matmul(context, a, b, true, true, out) == CDS_OK);
  expect_values(out, std::vector<T>{T(23), T(31), T(34), T(46)});
}

template void test_matmul<float>(Cds *root, CdsContext *context);
template void test_matmul<double>(Cds *root, CdsContext *context);
