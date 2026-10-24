#include "helper.h"
#include <cuda_runtime.h>
#include <vector>

template <typename T>
static void print_dimension(FILE *output, const CdsArrayMeta &meta,
                            const std::vector<T> &values, size_t depth,
                            size_t &offset) {
  std::fputc('[', output);
  const bool row = depth + 1 == meta.dim;
  for (size_t i = 0; i < meta.shape[depth]; ++i) {
    if (i > 0)
      std::fputs(row ? ", " : ",", output);
    if (row) {
      std::fprintf(output, "%.2f", double(values[offset++]));
    } else {
      if (i > 0)
        std::fprintf(output, "\n%*s", int((depth + 1) * 2), "");
      print_dimension(output, meta, values, depth + 1, offset);
    }
  }
  std::fputc(']', output);
}

template <typename T>
static void print_values(const CdsArray *array, const CdsArrayMeta &meta,
                         FILE *output) {
  std::vector<T> values(meta.count);
  if (meta.count > 0)
    cds_to_host(array, values.data());
  size_t offset = 0;
  print_dimension(output, meta, values, 0, offset);
  std::fputc('\n', output);
}

bool print_array(const CdsArray *array, FILE *output) {
  if (!array || !output)
    return false;
  CdsArrayMeta meta{};
  cds_to_meta(array, &meta);
  if (meta.dim == 0 || meta.dim > CDS_MAX_DIM ||
      (meta.elem_size != sizeof(float) && meta.elem_size != sizeof(double)))
    return false;

  const cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "print_array: %s\n", cudaGetErrorString(status));
    return false;
  }

  std::fprintf(output, "items: %zu, dim: %zu, shape: [", meta.count, meta.dim);
  for (size_t i = 0; i < meta.dim; ++i)
    std::fprintf(output, "%s%zu", i ? ", " : "", meta.shape[i]);
  std::fputs("]\n", output);
  // The public API currently supports only float32 and float64.
  if (meta.elem_size == sizeof(float))
    print_values<float>(array, meta, output);
  else
    print_values<double>(array, meta, output);
  return std::ferror(output) == 0;
}
