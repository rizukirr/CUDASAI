#include "api_test_helpers.h"
#include "error.h"

void test_host() {
  CHECK(cds_type_size[CDS_TYPE_FLOAT32] == sizeof(float));
  CHECK(cds_type_size[CDS_TYPE_FLOAT64] == sizeof(double));
  CHECK(cds_new(0) == nullptr);
  CHECK(cds_last_error() == CDS_ERR_INVALID);
  cds_free(nullptr);
  cds_context_free(nullptr);
  const CdsError codes[] = {CDS_OK, CDS_ERR_INVALID, CDS_ERR_OUT_OF_MEMORY,
                            CDS_ERR_CUDA, CDS_ERR_INTERNAL};
  for (CdsError a : codes) {
    CHECK(cds_error_string(a) != nullptr);
    for (CdsError b : codes)
      CHECK(a == b || std::strcmp(cds_error_string(a), cds_error_string(b)));
  }
  CHECK(cds_last_error_message() != nullptr);
}
