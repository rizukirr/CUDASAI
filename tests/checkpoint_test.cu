#include "api_test_helpers.h"

template <typename T> void test_checkpoint(Cds *root) {
  const CdsCheckpoint empty = cds_checkpoint(root);
  const size_t shape[] = {4}, large[] = {2048};
  const std::vector<T> values = {1, 2, 3, 4};
  CdsArray *persistent = cds_array(root, shape, 1, dtype<T>(), values.data());
  CHECK(persistent);
  const CdsCheckpoint checkpoint = cds_checkpoint(root);
  void *saved_data = nullptr;
  CdsArray *saved_array = nullptr;
  for (int iteration = 0; iteration < 3; ++iteration) {
    CdsArray *temp = cds_array(root, large, 1, dtype<T>(), nullptr);
    CHECK(temp);
    if (iteration == 0) {
      saved_data = cds_data_at(temp, 0);
      saved_array = temp;
    } else {
      CHECK(cds_data_at(temp, 0) == saved_data);
      CHECK(temp == saved_array);
    }
    // Force the host metadata arena to grow as well as the device arena.
    for (int i = 0; i < 40; ++i)
      CHECK(cds_array(root, shape, 1, dtype<T>(), nullptr));
    cds_restore(root, checkpoint);
    expect_values(persistent, values);
  }
  void *first_data = cds_data_at(persistent, 0);
  cds_restore(root, empty);
  CdsArray *again = cds_array(root, shape, 1, dtype<T>(), nullptr);
  CHECK(again);
  CHECK(cds_data_at(again, 0) == first_data);
  expect_values(again, std::vector<T>(4, 0));
}

template void test_checkpoint<float>(Cds *root);
template void test_checkpoint<double>(Cds *root);
