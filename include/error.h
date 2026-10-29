#ifndef CDS_ERROR_H
#define CDS_ERROR_H

#include "export.h"

typedef enum CdsError {
  CDS_OK = 0,
  CDS_ERR_INVALID,       // bad argument from the caller
  CDS_ERR_OUT_OF_MEMORY, // host or device allocation failed
  CDS_ERR_CUDA,          // CUDA runtime or cuBLASLt reported a failure
  CDS_ERR_INTERNAL,      // a cudasai bug, such as no cuBLASLt algorithm found
} CdsError;

/**
 * Code of this thread's most recent failure. Like errno, a successful call
 * leaves it untouched, so read it only right after a call reported failure.
 */
CDS_API CdsError cds_last_error(void);

/**
 * Message of this thread's most recent failure: source file, line, and the
 * failing expression or CUDA error text. Never null. Valid until the same
 * thread's next failure.
 */
CDS_API const char *cds_last_error_message(void);

/** Fixed description of `err`. Never null. */
CDS_API const char *cds_error_string(CdsError err);

#endif // CDS_ERROR_H
