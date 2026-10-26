#pragma once

#include "array.h"
#include "kernel.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Keep checks active in Release builds, including calls with side effects.
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUDA_REQUIRE(call)                                                     \
  do {                                                                         \
    cudaError_t cuda_require_status_ = (call);                                 \
    if (cuda_require_status_ != cudaSuccess) {                                 \
      std::fprintf(stderr, "%s: %s\n", #call,                                  \
                   cudaGetErrorString(cuda_require_status_));                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T> constexpr CdsType dtype() {
  return sizeof(T) == sizeof(float) ? CDS_TYPE_FLOAT32 : CDS_TYPE_FLOAT64;
}

template <typename T>
void expect_values(
    const CdsArray *array, const std::vector<T> &expected,
    double tolerance = sizeof(T) == sizeof(float) ? 1e-5 : 1e-12) {
  CdsArrayMeta meta{};
  cds_to_meta(array, &meta);
  CHECK(meta.count == expected.size());
  CHECK(meta.elem_size == sizeof(T));
  CHECK(meta.capacity == expected.size() * sizeof(T));
  // Operations run on a private context stream. Synchronize before readback.
  CUDA_REQUIRE(cudaDeviceSynchronize());
  std::vector<T> actual(expected.size());
  cds_to_host(array, actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i]) ||
        std::abs(double(actual[i]) - double(expected[i])) >
            tolerance * (1 + std::abs(double(expected[i])))) {
      std::fprintf(stderr, "element %zu: expected %.17g, got %.17g\n", i,
                   double(expected[i]), double(actual[i]));
      std::exit(1);
    }
  }
}
