#include "_error.h"
#include <stdarg.h>
#include <stdio.h>

static thread_local struct {
  CdsError code;
  char message[256];
} last = {CDS_OK, ""};

CdsError cds_fail(CdsError code, const char *file, int line, const char *fmt,
                  ...) {
  int n = snprintf(last.message, sizeof(last.message), "%s:%d: ", file, line);
  if (n > 0 && (size_t)n < sizeof(last.message)) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(last.message + n, sizeof(last.message) - n, fmt, args);
    va_end(args);
  }
  last.code = code;
  return code;
}

CdsError cds_cuda_status(cudaError_t status, const char *file, int line,
                         const char *expr) {
  if (status == cudaSuccess)
    return CDS_OK;
  // Clear the pending error so a later launch check does not report it again.
  (void)cudaGetLastError();
  CdsError code = status == cudaErrorMemoryAllocation ? CDS_ERR_OUT_OF_MEMORY
                                                      : CDS_ERR_CUDA;
  return cds_fail(code, file, line, "%s: %s", expr, cudaGetErrorString(status));
}

CdsError cds_cublas_status(cublasStatus_t status, const char *file, int line,
                           const char *expr) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return CDS_OK;
  CdsError code = status == CUBLAS_STATUS_ALLOC_FAILED ? CDS_ERR_OUT_OF_MEMORY
                                                       : CDS_ERR_CUDA;
  return cds_fail(code, file, line, "%s: cuBLASLt status %d", expr,
                  (int)status);
}

CdsError cds_last_error(void) { return last.code; }

const char *cds_last_error_message(void) { return last.message; }

const char *cds_error_string(CdsError err) {
  switch (err) {
  case CDS_OK:
    return "success";
  case CDS_ERR_INVALID:
    return "invalid argument";
  case CDS_ERR_OUT_OF_MEMORY:
    return "out of memory";
  case CDS_ERR_CUDA:
    return "CUDA error";
  case CDS_ERR_INTERNAL:
    return "internal error";
  }
  return "unknown error";
}
