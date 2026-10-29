#ifndef CDS_INTERNAL_ERROR_H
#define CDS_INTERNAL_ERROR_H

#include "error.h"
#include <cublasLt.h>
#include <cuda_runtime.h>

// Every helper records the failure for cds_last_error and returns its code,
// so a call site can `return cds_fail(...)` or test the result directly.
CdsError cds_fail(CdsError code, const char *file, int line, const char *fmt,
                  ...) __attribute__((format(printf, 4, 5)));
CdsError cds_cuda_status(cudaError_t status, const char *file, int line,
                         const char *expr);
CdsError cds_cublas_status(cublasStatus_t status, const char *file, int line,
                           const char *expr);

#define CDS_CUDA(call) cds_cuda_status((call), __FILE__, __LINE__, #call)
#define CDS_CUBLAS(call) cds_cublas_status((call), __FILE__, __LINE__, #call)
#define CDS_REQUIRE(cond)                                                      \
  ((cond) ? CDS_OK : cds_fail(CDS_ERR_INVALID, __FILE__, __LINE__, "%s", #cond))
#define CDS_FAIL(code, ...) cds_fail((code), __FILE__, __LINE__, __VA_ARGS__)

// Return the error from the enclosing CdsError function.
#define CDS_TRY(expr)                                                          \
  do {                                                                         \
    CdsError cds_e_ = (expr);                                                  \
    if (cds_e_ != CDS_OK)                                                      \
      return cds_e_;                                                           \
  } while (0)

#endif // CDS_INTERNAL_ERROR_H
