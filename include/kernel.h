#ifndef CDS_KERNEL_H
#define CDS_KERNEL_H

#include "array.h"
#include "export.h"

/**
 * GPU state shared by every operation: the cuBLASLt handle, a device
 * workspace cuBLASLt may use as scratch, and the stream all work runs on.
 * Create one before any root and free it after every root is gone.
 */
typedef struct CdsContext CdsContext;

/**
 * Create a context with `workspace_bytes` of device scratch for cuBLASLt.
 * Zero is allowed and limits cuBLASLt to algorithms that need none. 32 MB
 * is a good default. Returns null if the host allocation fails.
 */
CDS_API CdsContext *cds_context_new(size_t workspace_bytes);

/**
 * Wait for work queued on the context's stream, then release the handle,
 * workspace and stream. Null is a no-op.
 */
CDS_API void cds_context_free(CdsContext *ctx);

/**
 * Set every element of `dst` to `*value`, read as `dst`'s dtype.
 */
CDS_API void cds_fill(const CdsContext *ctx, CdsArray *dst, const void *value);

/**
 * y = x · W^T + b, row-major, through cuBLASLt.
 *
 * x is [..., in_features], W is [out_features, in_features], b is
 * [out_features] or null, and y is [..., out_features] with the same leading
 * dimensions as x. All four share one dtype. Nothing is allocated: the caller
 * owns every array, so W and b can live for the whole program while x and y
 * are created per batch.
 */
CDS_API void cds_linear(const CdsContext *ctx, const CdsArray *x,
                        const CdsArray *W, const CdsArray *b, CdsArray *y);

/**
 * Elementwise dest = a + b through cuBLASLt, without broadcasting.
 * All arrays must have identical shapes and types. The caller allocates
 * dest, whose device memory must not overlap either input, including views.
 * An empty input performs no work.
 *
 * Queues work on ctx's stream without waiting for completion. Keep the
 * arrays alive until that work finishes and synchronize the stream before
 * reading the result from the host or using it on another stream.
 */
CDS_API void cds_add(const CdsContext *ctx, const CdsArray *a,
                     const CdsArray *b, CdsArray *dest);

#endif // CDS_KERNEL_H
