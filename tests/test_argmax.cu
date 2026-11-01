#include "api_test_helpers.h"
#include <algorithm>
#include <limits>

template <typename T>
static void check_argmax(Cds *root, CdsContext *context,
                         const std::vector<size_t> &shape,
                         const std::vector<T> &input) {
  size_t count = 1;
  for (size_t dimension : shape)
    count *= dimension;
  CHECK(input.size() == count);
  const size_t cols = shape.back();
  const size_t rows = count / cols;

  // The CPU loop: a strict comparison keeps the lowest column on a tie.
  std::vector<int32_t> expected(rows);
  for (size_t row = 0; row < rows; ++row) {
    size_t best = 0;
    for (size_t col = 1; col < cols; ++col)
      if (input[row * cols + col] > input[row * cols + best])
        best = col;
    expected[row] = int32_t(best);
  }

  // The sentinel shows that every row was written.
  const std::vector<int32_t> sentinel(rows, -7);
  const size_t dest_shape[] = {rows};
  CdsArray *x =
      cds_array(root, shape.data(), shape.size(), dtype<T>(), input.data());
  CdsArray *dest =
      cds_array(root, dest_shape, 1, CDS_TYPE_INT32, sentinel.data());
  CHECK(x && dest);

  CHECK(cds_argmax(context, x, dest) == CDS_OK);
  CUDA_REQUIRE(cudaDeviceSynchronize());
  std::vector<int32_t> actual(rows);
  CHECK(cds_to_host(dest, actual.data()) == CDS_OK);
  for (size_t row = 0; row < rows; ++row) {
    if (actual[row] != expected[row]) {
      std::fprintf(stderr, "cols %zu row %zu: expected %d, got %d\n", cols, row,
                   expected[row], actual[row]);
      std::exit(1);
    }
  }

  // Compare bytes: expect_values rejects the infinities used below.
  std::vector<T> unchanged(count);
  CHECK(cds_to_host(x, unchanged.data()) == CDS_OK);
  CHECK(std::memcmp(unchanged.data(), input.data(), count * sizeof(T)) == 0);
}

template <typename T> void test_argmax(Cds *root, CdsContext *context) {
  for (const std::vector<size_t> &shape : {std::vector<size_t>{1},
                                           {3},
                                           {2, 1},
                                           {37, 2},
                                           {37, 31},
                                           {37, 32},
                                           {37, 33},
                                           {37, 64},
                                           {5, 257},
                                           {2, 3, 65},
                                           {3, 1025}}) {
    size_t count = 1;
    for (size_t dimension : shape)
      count *= dimension;
    const size_t cols = shape.back();
    const size_t rows = count / cols;

    std::vector<T> input(count);
    for (size_t i = 0; i < count; ++i)
      input[i] = T(int((i * 7919) % 2003) - 1001);
    check_argmax(root, context, shape, input);

    // The maximum moves between the first, last and middle columns.
    std::fill(input.begin(), input.end(), T(0));
    for (size_t row = 0; row < rows; ++row) {
      const size_t col = row % 3 == 0 ? 0 : row % 3 == 1 ? cols - 1 : cols / 2;
      input[row * cols + col] = T(1);
    }
    check_argmax(root, context, shape, input);

    // Equal values in lanes other than lane 0 must still pick the lowest
    // column.
    std::fill(input.begin(), input.end(), T(0));
    for (size_t row = 0; row < rows; ++row) {
      input[row * cols + cols - 1] = T(2);
      input[row * cols + cols / 2] = T(2);
    }
    check_argmax(root, context, shape, input);

    // A whole row of ties, including a row where nothing beats -infinity. Lanes
    // with no columns must not win.
    std::fill(input.begin(), input.end(), T(5));
    check_argmax(root, context, shape, input);
    std::fill(input.begin(), input.end(), -T(INFINITY));
    check_argmax(root, context, shape, input);
  }

  // A row with NaN has an unspecified answer, but it must be inside the row.
  const size_t cols = 40;
  const size_t nan_shape[] = {2, cols}, dest_shape[] = {2};
  std::vector<T> input(2 * cols, T(1));
  input[0] = std::numeric_limits<T>::quiet_NaN();
  std::fill(input.begin() + cols, input.end(),
            std::numeric_limits<T>::quiet_NaN());
  CdsArray *x = cds_array(root, nan_shape, 2, dtype<T>(), input.data());
  CdsArray *dest = cds_array(root, dest_shape, 1, CDS_TYPE_INT32, nullptr);
  CHECK(x && dest);
  CHECK(cds_argmax(context, x, dest) == CDS_OK);
  CUDA_REQUIRE(cudaDeviceSynchronize());
  int32_t actual[2];
  CHECK(cds_to_host(dest, actual) == CDS_OK);
  CHECK(actual[0] >= 0 && actual[0] < int32_t(cols));
  CHECK(actual[1] >= 0 && actual[1] < int32_t(cols));
}

template void test_argmax<float>(Cds *root, CdsContext *context);
template void test_argmax<double>(Cds *root, CdsContext *context);
