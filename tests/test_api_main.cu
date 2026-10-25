#include "api_test_cases.h"
#include "api_test_helpers.h"

template <typename T> void run(const char *name) {
  CdsContext *context = cds_context_new(4 << 20);
  CHECK(context);
  Cds *root = cds_new(128);
  CHECK(root);
  if (std::strcmp(name, "array") == 0)
    test_array<T>(root);
  else if (std::strcmp(name, "cds_to_meta") == 0)
    test_cds_to_meta<T>(root);
  else if (std::strcmp(name, "view") == 0)
    test_view<T>(root);
  else if (std::strcmp(name, "copy") == 0)
    test_copy<T>(root);
  else if (std::strcmp(name, "random") == 0)
    test_random<T>(root);
  else if (std::strcmp(name, "checkpoint") == 0)
    test_checkpoint<T>(root);
  else if (std::strcmp(name, "fill") == 0)
    test_fill<T>(root, context);
  else if (std::strcmp(name, "add") == 0)
    test_add<T>(root, context);
  else if (std::strcmp(name, "scale") == 0)
    test_scale<T>(root, context);
  else if (std::strcmp(name, "linear") == 0)
    test_linear<T>(root, context);
  else if (std::strcmp(name, "context") == 0)
    test_context<T>(root);
  else {
    std::fprintf(stderr, "Unknown test: %s\n", name);
    std::exit(1);
  }
  cds_free(root);
  cds_context_free(context);
}

int main(int argc, char **argv) {
  CHECK(argc >= 2);
  if (std::strcmp(argv[1], "host") == 0) {
    test_host();
    return 0;
  }
  CHECK(argc == 3);
  CHECK(std::strcmp(argv[2], "float") == 0 ||
        std::strcmp(argv[2], "double") == 0);
  int devices = 0;
  cudaError_t status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) {
    std::fprintf(stderr, "SKIP: CUDA device unavailable (%s)\n",
                 cudaGetErrorString(status));
    return 77;
  }
  CUDA_REQUIRE(status);
  if (std::strcmp(argv[2], "float") == 0)
    run<float>(argv[1]);
  else
    run<double>(argv[1]);
  return 0;
}
