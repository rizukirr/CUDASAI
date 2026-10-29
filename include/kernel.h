#ifndef CDS_KERNEL_H
#define CDS_KERNEL_H

#include "array.h"
#include "export.h"

typedef enum { CDS_ACTIVATION_GELU, CDS_ACTIVATION_NONE } CdsActivation;

/**
 * GPU state shared by every operation: the cuBLASLt handle, a device
 * workspace cuBLASLt may use as scratch, and the stream all work runs on.
 * Create one before any root and free it after every root is gone.
 */
typedef struct CdsContext CdsContext;

/**
 * Create a context with `workspace_bytes` of device scratch for cuBLASLt.
 * Zero is allowed and limits cuBLASLt to algorithms that need none. 32 MB
 * is a good default. Returns null on failure, with the reason in
 * cds_last_error().
 */
CDS_API CdsContext *cds_context_new(size_t workspace_bytes);

/**
 * Wait for work queued on the context's stream, then release the handle,
 * workspace and stream. Null is a no-op.
 * A failed sync or release is recorded in cds_last_error(), and every resource
 * is still released.
 */
CDS_API void cds_context_free(CdsContext *ctx);

/**
 * Set every element of `dst` to `*value`, read as `dst`'s dtype.
 * Returns CDS_OK, or the failure also recorded in cds_last_error(). Only launch
 * errors are caught here: a fault while the queued work runs is reported by the
 * next synchronizing call, such as cds_to_host.
 */
CDS_API CdsError cds_fill(const CdsContext *ctx, CdsArray *dst,
                          const void *value);

/**
 * y = x · W^T + b, row-major, through cuBLASLt.
 *
 * x is [..., in_features], W is [out_features, in_features], b is
 * [out_features] or null, and y is [..., out_features] with the same leading
 * dimensions as x. All four share one dtype. Nothing is allocated: the caller
 * owns every array, so W and b can live for the whole program while x and y
 * are created per batch.
 * Returns CDS_OK, or the failure also recorded in cds_last_error(). Only launch
 * errors are caught here: a fault while the queued work runs is reported by the
 * next synchronizing call, such as cds_to_host.
 */
CDS_API CdsError cds_linear(const CdsContext *ctx, const CdsArray *x,
                            const CdsArray *W, const CdsArray *b,
                            CdsActivation activation, CdsArray *y);

/**
 * Apply softmax over the last dimension.
 * Input and output must have identical shapes and types.
 * Returns CDS_OK, or the failure also recorded in cds_last_error(). Only launch
 * errors are caught here: a fault while the queued work runs is reported by the
 * next synchronizing call, such as cds_to_host.
 */
CDS_API CdsError cds_softmax(const CdsContext *ctx, const CdsArray *x,
                             CdsArray *y);

/**
 * Elementwise dest = a + b through cuBLASLt, without broadcasting.
 * All arrays must have identical shapes and types. The caller allocates
 * dest, whose device memory must not overlap either input, including views.
 * An empty input performs no work.
 *
 * Queues work on ctx's stream without waiting for completion. Keep the
 * arrays alive until that work finishes and synchronize the stream before
 * reading the result from the host or using it on another stream.
 * Returns CDS_OK, or the failure also recorded in cds_last_error(). Only launch
 * errors are caught here: a fault while the queued work runs is reported by the
 * next synchronizing call, such as cds_to_host.
 */
CDS_API CdsError cds_add(const CdsContext *ctx, const CdsArray *a,
                         const CdsArray *b, CdsArray *dest);

/**
 * Computes dest[i] = a[i] * b through cuBLASLt, where b is a scalar.
 *
 * a and dest must have identical shapes and element types.
 * T must be float for CDS_TYPE_FLOAT32 or double for CDS_TYPE_FLOAT64.
 * The caller allocates dest. Its device memory must not overlap a's
 * device memory, including through views.
 * An empty input performs no work.
 *
 * Queues work on ctx's stream without waiting for completion.
 * Keep the input and output device memory valid until the work finishes.
 * Synchronize before reading the result from the host. Before using it
 * on another stream, establish a dependency, for example with a CUDA event.
 * Returns CDS_OK, or the failure also recorded in cds_last_error(). Only launch
 * errors are caught here: a fault while the queued work runs is reported by the
 * next synchronizing call, such as cds_to_host.
 */
template <typename T>
CDS_API CdsError cds_scale(const CdsContext *ctx, const CdsArray *a, const T b,
                           CdsArray *dest);

#endif // CDS_KERNEL_H
