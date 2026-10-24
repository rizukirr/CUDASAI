#include "api_test_helpers.h"

template <typename T> void test_checkpoint(Cds *root) {
  const CdsCheckpoint empty = cds_checkpoint(root);
  const size_t shape[] = {4}, large[] = {2048};
  const std::vector<T> values = {1, 2, 3, 4};
  CdsArray *persistent = cds_array(root, shape, 1, dtype<T>(), values.data());
  CHECK(persistent);
  const CdsCheckpoint checkpoint = cds_checkpoint(root);
  CdsCheckpoint saved_position{};
  CdsArray *saved_array = nullptr;
  for (int iteration = 0; iteration < 3; ++iteration) {
    CdsArray *temp = cds_array(root, large, 1, dtype<T>(), nullptr);
    CHECK(temp);
    const CdsCheckpoint position = cds_checkpoint(root);
    if (iteration == 0) {
      saved_position = position;
      saved_array = temp;
    } else {
      CHECK(position.device.block == saved_position.device.block);
      CHECK(position.device.index == saved_position.device.index);
      CHECK(position.host.block == saved_position.host.block);
      CHECK(position.host.index == saved_position.host.index);
      CHECK(temp == saved_array);
    }
    expect_values(temp, std::vector<T>(2048, 0));
    // Force the host metadata arena to grow as well as the device arena.
    for (int i = 0; i < 40; ++i)
      CHECK(cds_array(root, shape, 1, dtype<T>(), nullptr));
    cds_restore(root, checkpoint);
    expect_values(persistent, values);
  }
  cds_restore(root, empty);
  CdsArray *again = cds_array(root, shape, 1, dtype<T>(), nullptr);
  CHECK(again);
  const CdsCheckpoint restored = cds_checkpoint(root);
  CHECK(restored.device.block == checkpoint.device.block);
  CHECK(restored.device.index == checkpoint.device.index);
  CHECK(restored.host.block == checkpoint.host.block);
  CHECK(restored.host.index == checkpoint.host.index);
  expect_values(again, std::vector<T>(4, 0));
}

template void test_checkpoint<float>(Cds *root);
template void test_checkpoint<double>(Cds *root);
