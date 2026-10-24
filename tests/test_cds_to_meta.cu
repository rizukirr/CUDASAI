#include "api_test_helpers.h"

template <typename T> void test_cds_to_meta(Cds *root) {
  for (std::vector<size_t> shape :
       {std::vector<size_t>{7}, {2, 3}, {1, 2, 1, 2, 1, 2, 1, 2}}) {
    const std::vector<size_t> original_shape = shape;
    size_t count = 1;
    for (size_t extent : shape)
      count *= extent;
    const CdsCheckpoint checkpoint = cds_checkpoint(root);
    CdsArray *array =
        cds_array(root, shape.data(), shape.size(), dtype<T>(), nullptr);
    CHECK(array);
    shape[0] = 99;
    CdsArrayMeta meta{};
    cds_to_meta(array, &meta);
    CHECK(meta.count == count);
    CHECK(meta.dim == original_shape.size());
    CHECK(meta.elem_size == sizeof(T));
    CHECK(meta.capacity == count * sizeof(T));
    CHECK(meta.data != nullptr);
    const std::vector<T> values(count, T(3.5));
    CUDA_REQUIRE(cudaMemcpy(meta.data, values.data(), meta.capacity,
                            cudaMemcpyHostToDevice));
    expect_values(array, values);
    for (size_t i = 0; i < meta.dim; ++i)
      CHECK(meta.shape[i] == original_shape[i]);

    // The result is an independent snapshot of the array metadata.
    meta.shape[0] = 123;
    CdsArrayMeta second{};
    cds_to_meta(array, &second);
    CHECK(second.shape[0] == original_shape[0]);
    CHECK(second.data == meta.data);
    cds_restore(root, checkpoint);
    CHECK(second.count == count);
    CHECK(second.shape[0] == original_shape[0]);
  }

  const size_t shape[] = {8}, window[] = {2, 2};
  CdsArray *source = cds_array(root, shape, 1, dtype<T>(), nullptr);
  CHECK(source);
  CdsArray *view = cds_view(root, source, 3, window, 2);
  CHECK(view);
  CdsArrayMeta meta{};
  cds_to_meta(view, &meta);
  CHECK(meta.count == 4);
  CHECK(meta.dim == 2);
  CHECK(meta.elem_size == sizeof(T));
  CHECK(meta.capacity == 4 * sizeof(T));
  CHECK(meta.shape[0] == 2 && meta.shape[1] == 2);
  CdsArrayMeta source_meta{};
  cds_to_meta(source, &source_meta);
  CHECK(meta.data ==
        static_cast<unsigned char *>(source_meta.data) + 3 * sizeof(T));
  const std::vector<T> replacement = {1, 2, 3, 4};
  CUDA_REQUIRE(cudaMemcpy(meta.data, replacement.data(), meta.capacity,
                          cudaMemcpyHostToDevice));
  expect_values(source, std::vector<T>{0, 0, 0, 1, 2, 3, 4, 0});
}

template void test_cds_to_meta<float>(Cds *root);
template void test_cds_to_meta<double>(Cds *root);
