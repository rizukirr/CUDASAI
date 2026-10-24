#include "api_test_cases.h"
#include "api_test_helpers.h"

template <typename T> void test_context(Cds *root) {
  CdsContext *zero_workspace = cds_context_new(0);
  CHECK(zero_workspace);
  test_fill<T>(root, zero_workspace);
  cds_context_free(zero_workspace);
}

template void test_context<float>(Cds *root);
template void test_context<double>(Cds *root);
