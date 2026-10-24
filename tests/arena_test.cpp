// CPU: c++ -std=c++11 tests/arena_test.cpp -o /tmp/arena-test && /tmp/arena-test
// Mock CUDA: add -DARENA_USE_CUDA -Itests/cuda_mock to the command above.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstdlib>

#ifdef ARENA_USE_CUDA
#include <cuda_runtime_api.h>

static bool fail_allocation;
static unsigned int allocations;
static unsigned int releases;

cudaError_t cudaMalloc(void **ptr, size_t size) {
  if (fail_allocation)
    return cudaErrorMemoryAllocation;
  *ptr = std::malloc(size);
  if (!*ptr)
    return cudaErrorMemoryAllocation;
  ++allocations;
  return cudaSuccess;
}

cudaError_t cudaFree(void *ptr) {
  assert(ptr);
  ++releases;
  std::free(ptr);
  return cudaSuccess;
}
#endif

#define ARENA_IMPLEMENTATION
#include "../vendor/arena.h"

int main() {
  Arena *arena = arena_create(128);
  assert(arena);
#ifdef ARENA_USE_CUDA
  fail_allocation = true;
  assert(arena_alloc(arena, 16, 16) == nullptr);
  fail_allocation = false;
#endif
  void *first = arena_alloc(arena, 16, 64);
  assert(first && (uintptr_t)first % 64 == 0);
  ArenaCheckpoint checkpoint = arena_checkpoint(arena);
  void *large = arena_alloc(arena, 4096, 256);
  assert(large && (uintptr_t)large % 256 == 0);
#ifdef ARENA_USE_CUDA
  const unsigned int before_reuse = allocations;
#endif
  arena_restore(arena, checkpoint);
  assert(arena_alloc(arena, 4096, 256) == large);
  arena_reset(arena);
  assert(arena_alloc(arena, 16, 64) == first);
#ifdef ARENA_USE_CUDA
  assert(allocations == before_reuse);
  assert(releases == 0);
  fail_allocation = true;
  assert(arena_alloc(arena, 8192, 64) == nullptr);
  fail_allocation = false;
#endif
  arena_free(arena);
#ifdef ARENA_USE_CUDA
  assert(allocations == releases);
#endif
}
