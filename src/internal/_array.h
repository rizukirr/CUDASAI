#ifndef CDS_INTERNAL_H
#define CDS_INTERNAL_H

#include "arena.h"
#include "array.h"
#include "_error.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

struct Cds {
  Arena *arena; // device memory for array data
  Arena *host;  // host memory for CdsArray structs
};

struct CdsArray {
  size_t dim;
  size_t elem_size;
  size_t count;
  size_t capacity;
  CdsType type;
  size_t shape[CDS_MAX_DIM];
  size_t strides[CDS_MAX_DIM];
  void *data; // device pointer
};

// Indexed by CdsType, so the order must match the enum.
static constexpr size_t cds_alignof[] = {
    alignof(float),
    alignof(double),
};

// Host side, uniform in [low, high) from rand(). Feeds a staging buffer that
// is then copied to the device.
template <typename T> void random_fill(T *data, size_t count, T low, T high) {
  for (size_t i = 0; i < count; i++)
    data[i] = low + (rand() / ((T)RAND_MAX + (T)1)) * (high - low);
}

template <typename T> static inline void assert_type(T type) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "T must be float or double");
}

/**
 * Fill dim, type, elem_size, count, capacity, shape and strides.
 * Returns false when the byte size overflows.
 */
[[nodiscard("check the returned value for overflow")]]
bool cds_init_meta(CdsArray *self, const size_t *shape, size_t dim,
                   CdsType type);

/**
 * Allocate a CdsArray struct from the root's host arena and fill its metadata.
 * `data` is left null for the caller to point somewhere.
 */
[[nodiscard]]
CdsArray *cds_new_struct(const Cds *root, const size_t *shape, size_t dim,
                         CdsType type);

/**
 * cds_new_struct plus data from the root's device arena, uninitialised.
 * Nothing here needs freeing: both go away with the root.
 */
[[nodiscard]]
CdsArray *cds_new_derived(const Cds *root, const size_t *shape, size_t dim,
                          CdsType type);

#endif // CDS_INTERNAL_H
