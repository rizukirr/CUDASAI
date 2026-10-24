// Compile: c++ -std=c++11 -Itests/cuda_mock -c tests/test_arena_cuda_declaration.cpp -o /tmp/arena-declaration.o
#define ARENA_USE_CUDA
#include "../vendor/arena.h"

void *allocate_without_implementation(size_t size) {
  return ARENA_MALLOC(size);
}
