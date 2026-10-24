#include "api_test_helpers.h"

void test_host() {
  CHECK(cds_type_size[CDS_FLOAT32] == sizeof(float));
  CHECK(cds_type_size[CDS_FLOAT64] == sizeof(double));
  CHECK(cds_new(0) == nullptr);
  cds_free(nullptr);
  cds_context_free(nullptr);
}
