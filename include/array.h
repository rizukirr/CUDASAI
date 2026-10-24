#ifndef CDS_ARRAY_H
#define CDS_ARRAY_H

#include "arena.h"
#include "export.h"
#include <stddef.h>

constexpr size_t CDS_MAX_DIM = 8;

typedef struct Cds Cds;           // a root: owns device memory
typedef struct CdsArray CdsArray; // an array living inside a root

typedef struct {
  size_t count, dim, capacity, elem_size;
  size_t shape[CDS_MAX_DIM];
  void *data; // Borrowed device pointer, never dereference on the host.
} CdsArrayMeta;

// A position in a root's allocation history. A plain value: copy it, keep it,
// restore to it as many times as you like while the root is alive.
typedef struct CdsCheckpoint {
  ArenaCheckpoint device;
  ArenaCheckpoint host;
} CdsCheckpoint;

typedef enum {
  CDS_FLOAT32,
  CDS_FLOAT64,
} CdsType;

// Indexed by CdsType, so the order must match the enum.
static constexpr size_t cds_type_size[] = {
    sizeof(float),
    sizeof(double),
};

/**
 * Create a root: an owner of device memory. Arrays are created inside it with
 * cds_array and all die together when the root is freed. `device_bytes` is
 * the arena block size, not a limit: the root grows by another block when an
 * array does not fit.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API Cds *cds_new(size_t device_bytes);

/**
 * Create an array inside `root`. `data`, when non-null, points to
 * count * sizeof(type) bytes of host memory that are copied to the device.
 * A null `data` zero-fills the array. The array is released by cds_free on
 * the root, never on its own.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API CdsArray *cds_array(Cds *root, const size_t *shape, size_t dim,
                            CdsType type, const void *data);

/**
 * A new array over the device memory of `src`, starting `offset` elements in
 * and reinterpreted as `shape`. No device memory is allocated and no data is
 * copied: writing through the view writes `src`. The window must fit, so
 * offset + count(shape) <= count(src). The view lives in `root`, so it is
 * reclaimed with `root`, not with `src`.
 */
[[nodiscard]]
CDS_API CdsArray *cds_view(Cds *root, const CdsArray *src, size_t offset,
                           const size_t *shape, size_t dim);

/**
 * Device to device copy of every element of `src` into `dst`. Both must have
 * the same count and type. Shapes may differ. The two ranges must not overlap.
 */
CDS_API void cds_copy(CdsArray *dst, const CdsArray *src);

/**
 * Create an array inside `root` filled with uniform random values in
 * [-1/sqrt(fan_in), 1/sqrt(fan_in)], where fan_in is the last dimension of
 * `shape`. This is the PyTorch default initialisation for linear weights.
 * Draws from rand(), so call srand() first for a reproducible model.
 */
[[nodiscard("check the returned pointer for allocation failure")]]
CDS_API CdsArray *cds_array_rand(Cds *root, const size_t *shape, size_t dim,
                                 CdsType type);

/**
 * Record where `root` is now, so cds_restore can reclaim every array created
 * after this point.
 */
[[nodiscard]]
CDS_API CdsCheckpoint cds_checkpoint(const Cds *root);

/**
 * Reclaim every array created in `root` after `cp` was taken. Those arrays are
 * invalid afterwards, including their shape: the struct memory itself is
 * reused. Waits for the device to go idle first, so no kernel is still reading
 * the memory that the next cds_array will overwrite.
 */
CDS_API void cds_restore(Cds *root, CdsCheckpoint cp);

/**
 * Copy the whole array from the device into `dst`, which must hold
 * `capacity` bytes as reported by cds_to_meta. Blocks until the copy completes.
 */
CDS_API void cds_to_host(const CdsArray *self, void *dst);

/**
 * Copy array metadata into `meta` on the host, without copying array data.
 * `capacity` and `elem_size` are in bytes. Only shape[0..dim) is populated.
 * `data` points to the array's device storage (including any view offset).
 * It does not transfer ownership or extend the storage lifetime. Do not free
 * it or use it after its owning root is freed or restored past its allocation.
 */
CDS_API void cds_to_meta(const CdsArray *self, CdsArrayMeta *meta);

/**
 * Release a root together with every array created inside it. Arrays have no
 * free of their own.
 */
CDS_API void cds_free(Cds *root);

#endif // CDS_ARRAY_H
