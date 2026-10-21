/*
 * Copyright (c) 2025 Rizki <rizkirr.xyz@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ARENA_H
#define ARENA_H

/* Define ARENA_USE_CUDA consistently across translation units to place block
   data regions in device memory with cudaMalloc/cudaFree. Block headers and the
   Arena struct always live on the host. Allocator hooks below still override
   the defaults. Arena operations run on the host, and pointers returned by
   arena_alloc() under ARENA_USE_CUDA are device addresses. Finish GPU work
   before reusing allocations through reset/restore or releasing them with
   arena_free(). */
#ifdef ARENA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__cplusplus) ||                                                    \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#define ARENA_NULLPTR nullptr
#else
#define ARENA_NULLPTR NULL
#endif

/* Allocator hooks. ARENA_MALLOC/ARENA_FREE allocate host bookkeeping (the
   Arena struct and block headers). ARENA_DATA_MALLOC/ARENA_DATA_FREE allocate
   the block data regions handed out by arena_alloc(). Define any of them before
   including this header to substitute your own allocator. */
#ifdef ARENA_USE_CUDA

#ifndef ARENA_DATA_MALLOC
static inline void *arena_cuda_malloc(size_t size) {
  void *ptr = ARENA_NULLPTR;
  if (cudaMalloc(&ptr, size) != cudaSuccess)
    return ARENA_NULLPTR;
  return ptr;
}
#define ARENA_DATA_MALLOC(size) arena_cuda_malloc(size)
#endif

#ifndef ARENA_DATA_FREE
#define ARENA_DATA_FREE(ptr) ((void)cudaFree(ptr))
#endif

#endif

#if defined(_WIN32)
#include <windows.h>

#ifndef ARENA_MALLOC
#define ARENA_MALLOC(size) HeapAlloc(GetProcessHeap(), 0, (size))
#endif

#ifndef ARENA_FREE
#define ARENA_FREE(ptr)                                                        \
  do {                                                                         \
    if (ptr)                                                                   \
      HeapFree(GetProcessHeap(), 0, (ptr));                                    \
  } while (0)
#endif

#else /* Linux / macOS / POSIX */

#ifndef ARENA_MALLOC
#define ARENA_MALLOC(size) malloc(size)
#endif

#ifndef ARENA_FREE
#define ARENA_FREE(ptr) free(ptr)
#endif

#endif

/* Data regions default to the same allocator as the bookkeeping. */
#ifndef ARENA_DATA_MALLOC
#define ARENA_DATA_MALLOC(size) ARENA_MALLOC(size)
#endif

#ifndef ARENA_DATA_FREE
#define ARENA_DATA_FREE(ptr) ARENA_FREE(ptr)
#endif

/**
 * @brief Get alignment of a type in a portable way.
 *
 * This macro expands to `alignof(type)` when compiling under C11 or newer, and
 * otherwise computes alignment using struct offset hack. This ensures the arena
 * allocator can correctly align memory on all compilers.
 *
 * @param type  Any C type whose alignment is needed.
 */
#if defined(__cplusplus)
#define ARENA_ALIGNOF(type) alignof(type)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
#define ARENA_ALIGNOF(type) alignof(type)
#else
#define ARENA_ALIGNOF(type)                                                    \
  offsetof(                                                                    \
      struct {                                                                 \
        char c;                                                                \
        type d;                                                                \
      },                                                                       \
      d)
#endif

// -----------------------------------------------------------------------------
// PUBLIC API (opaque handles)
// -----------------------------------------------------------------------------

/**
 * @brief Opaque handle for an Arena allocator.
 *
 * The internal structure is hidden from users unless
 * `ARENA_IMPLEMENTATION` is defined. The arena manages memory using
 * fixed-size blocks and fast bump-pointer allocation.
 */
typedef struct Arena Arena;

/**
 * @brief Allocator pair for block data regions, selectable per arena.
 *
 * arena_create() uses ARENA_DATA_MALLOC/ARENA_DATA_FREE. arena_create_with()
 * takes an explicit pair, so one build can hold arenas backed by different
 * memory, for example device memory for tensors and host memory for metadata.
 */
typedef void *(*ArenaDataMalloc)(size_t size);
typedef void (*ArenaDataFree)(void *ptr);

/**
 * @brief Checkpoint structure for saving/restoring arena state.
 *
 * Represents a specific point in the arena's allocation history.
 * Can be used to restore the arena to a previous state, effectively
 * freeing all allocations made after the checkpoint while keeping
 * allocations made before it.
 */
typedef struct ArenaCheckpoint {
  struct ArenaBlock *block; // Block pointer at checkpoint
  size_t index;             // Index within block at checkpoint
} ArenaCheckpoint;

/**
 * @brief Create a new arena allocator.
 *
 * This allocates an `Arena` structure but does **not** allocate any memory
 * blocks yet. Blocks are lazily allocated on the first call to `arena_alloc()`.
 *
 * @param default_block_size  The size (in bytes) of each allocated block.
 *                            Larger allocations will allocate a block sized
 *                            exactly large enough for the request.
 *
 * @return Pointer to a newly initialized Arena, or nullptr if allocation fails.
 */
Arena *arena_create(size_t default_block_size);

/**
 * @brief Create an arena whose block data regions use the given allocator.
 *
 * @see arena_create
 */
Arena *arena_create_with(size_t default_block_size, ArenaDataMalloc data_malloc,
                         ArenaDataFree data_free);

/**
 * @brief Default data allocators, as used by arena_create(). Exposed so a
 * caller can pass them to arena_create_with() alongside custom ones.
 */
void *arena_default_data_malloc(size_t size);
void arena_default_data_free(void *ptr);

/**
 * @brief Host heap allocators, always available regardless of ARENA_USE_CUDA.
 */
void *arena_host_malloc(size_t size);
void arena_host_free(void *ptr);

/**
 * @brief Allocate memory from the arena with a specific alignment.
 *
 * The arena grows by allocating new blocks when needed. Allocations never
 * return memory to the system until `arena_free()` is called.
 *
 * @param arena      Pointer to a valid Arena instance.
 * @param size       Number of bytes to allocate.
 * @param alignment  Alignment requirement (must be power of two).
 *
 * @return Pointer to allocated memory, or nullptr on failure.
 */
void *arena_alloc(Arena *arena, size_t size, size_t alignment);

/**
 * @brief Reset the arena state for reuse.
 *
 * All blocks remain allocated, but their internal `index` pointers are reset
 * to zero. This effectively frees all previously allocated memory but retains
 * the capacity.
 *
 * Behavior note:
 * - The implementation resets **all blocks**.
 * - A different design may free all but the first block.
 *
 * @param arena  Pointer to an Arena instance.
 */
void arena_reset(Arena *arena);

/**
 * @brief Release all memory owned by the arena.
 *
 * This frees all blocks and the Arena structure itself. After this call,
 * the arena pointer must not be used.
 *
 * @param arena  Pointer to an Arena instance.
 */
void arena_free(Arena *arena);

/**
 * @brief Save current arena state as a checkpoint.
 *
 * Returns a checkpoint representing the current allocation position.
 * Allocations made after this point can be freed by restoring to this
 * checkpoint using arena_restore(), while allocations made before remain
 * intact.
 *
 * Supports nested checkpoints. Blocks are rewound rather than freed, so a
 * checkpoint stays valid for the lifetime of the arena and restoring is
 * memory-safe in any order. Restore in LIFO order for the resulting
 * allocation position to be meaningful.
 *
 * @param arena Pointer to Arena instance
 * @return Checkpoint representing current state
 *
 * @example Basic usage:
 *   Arena *arena = arena_create(4096);
 *   void *persistent = arena_alloc(arena, 1024, 8);
 *
 *   ArenaCheckpoint cp = arena_checkpoint(arena);
 *
 *   for (int i = 0; i < 1000; i++) {
 *       void *temp = arena_alloc(arena, 512, 8);
 *       // Use temp...
 *       arena_restore(arena, cp);  // Free temp, keep persistent
 *   }
 */
ArenaCheckpoint arena_checkpoint(Arena *arena);

/**
 * @brief Restore arena to a previous checkpoint.
 *
 * Resets the arena's allocation position to the saved checkpoint state.
 * All allocations made after the checkpoint are effectively freed
 * (their memory becomes available for reuse).
 *
 * IMPORTANT:
 * - The checkpoint must be valid (from the same arena)
 * - Blocks are rewound, never freed; memory is returned to the system only by
 *   arena_free(). A checkpoint's block pointer therefore stays valid for the
 *   lifetime of the arena.
 * - Restore is memory-safe in any order, but checkpoints must be restored in
 *   LIFO order for the resulting allocation position to be meaningful.
 * - Using a checkpoint after arena_free() is undefined behavior
 * - Debug builds include validation checks via assertions
 *
 * @param arena Pointer to Arena instance
 * @param checkpoint Previously saved checkpoint from arena_checkpoint()
 */
void arena_restore(Arena *arena, ArenaCheckpoint checkpoint);

// -----------------------------------------------------------------------------
// IMPLEMENTATION
// -----------------------------------------------------------------------------
#ifdef ARENA_IMPLEMENTATION

/**
 * @brief Internal structure representing a memory block.
 *
 * Each block contains:
 *   - `next` pointer (linked list)
 *   - `data` the region handed out by arena_alloc(), from ARENA_DATA_MALLOC
 *   - `capacity` total size of the data region
 *   - `index` current write position
 *
 * The header lives in host memory from ARENA_MALLOC. The data region is a
 * separate allocation so it can live in device memory under ARENA_USE_CUDA.
 */
struct ArenaBlock {
  struct ArenaBlock *next;
  uint8_t *data;
  size_t capacity;
  size_t index;
};

static uint8_t *arena_block_data(struct ArenaBlock *block) {
  return block->data;
}

/**
 * @brief Internal arena structure.
 *
 * Fields:
 *   - `head`    → first allocated block
 *   - `current` → block currently accepting allocations
 *   - `default_block_size` → minimum block size
 */
struct Arena {
  struct ArenaBlock *head;
  struct ArenaBlock *current;
  size_t default_block_size;
  ArenaDataMalloc data_malloc;
  ArenaDataFree data_free;
};

void *arena_default_data_malloc(size_t size) { return ARENA_DATA_MALLOC(size); }
void arena_default_data_free(void *ptr) { ARENA_DATA_FREE(ptr); }
void *arena_host_malloc(size_t size) { return ARENA_MALLOC(size); }
void arena_host_free(void *ptr) { ARENA_FREE(ptr); }
/**
 * @brief Compute padding needed to align a pointer.
 *
 * This uses a bitmask trick (requires alignment to be power of two):
 *
 *   padding = (-ptr) & (alignment - 1)
 *
 * This ensures:
 *   - If pointer is already aligned → padding = 0
 *   - Otherwise → padding = minimal offset to align
 *
 * @param ptr        Pointer value as integer.
 * @param alignment  Required alignment (must be power of two).
 *
 * @return Number of bytes of padding needed.
 */
static size_t arena_align_padding(uintptr_t ptr, size_t alignment) {
  return ((size_t)0 - (size_t)ptr) & (alignment - 1);
}

static bool arena_add_overflow(size_t a, size_t b, size_t *out) {
  if (a > SIZE_MAX - b)
    return true;
  *out = a + b;
  return false;
}

/* General-purpose blocks are always default_block_size; a request too large to
   fit one gets its own exact-fit dedicated block. Keeping the two cases
   separate stops one large allocation from inflating every later block. */
static size_t arena_block_size_for(const Arena *arena, size_t min_needed) {
  return min_needed > arena->default_block_size ? min_needed
                                                : arena->default_block_size;
}

static struct ArenaBlock *arena_block_create(const Arena *arena,
                                             size_t capacity) {
  struct ArenaBlock *block =
      (struct ArenaBlock *)ARENA_MALLOC(sizeof(struct ArenaBlock));
  if (!block)
    return ARENA_NULLPTR;

  block->data = (uint8_t *)arena->data_malloc(capacity);
  if (!block->data) {
    ARENA_FREE(block);
    return ARENA_NULLPTR;
  }

  block->next = ARENA_NULLPTR;
  block->capacity = capacity;
  block->index = 0;
  return block;
}

static void arena_block_destroy(const Arena *arena, struct ArenaBlock *block) {
  arena->data_free(block->data);
  ARENA_FREE(block);
}

Arena *arena_create_with(size_t default_block_size, ArenaDataMalloc data_malloc,
                         ArenaDataFree data_free) {
  if (default_block_size == 0 || !data_malloc || !data_free)
    return ARENA_NULLPTR;

  Arena *arena = (Arena *)ARENA_MALLOC(sizeof(Arena));
  if (!arena)
    return ARENA_NULLPTR;

  arena->head = ARENA_NULLPTR;
  arena->current = ARENA_NULLPTR;
  arena->default_block_size = default_block_size;
  arena->data_malloc = data_malloc;
  arena->data_free = data_free;
  return arena;
}

Arena *arena_create(size_t default_block_size) {
  return arena_create_with(default_block_size, arena_default_data_malloc,
                           arena_default_data_free);
}

void *arena_alloc(Arena *arena, size_t size, size_t alignment) {
  if (!arena || size == 0 || alignment == 0)
    return ARENA_NULLPTR;

  // Ensure alignment is power of two.
  if (alignment & (alignment - 1))
    return ARENA_NULLPTR;

  size_t min_needed = 0;
  if (arena_add_overflow(size, alignment - 1, &min_needed))
    return ARENA_NULLPTR;

  // Lazily allocate first block.
  if (!arena->current) {
    struct ArenaBlock *block =
        arena_block_create(arena, arena_block_size_for(arena, min_needed));
    if (!block)
      return ARENA_NULLPTR;
    arena->head = arena->current = block;
  }

  for (;;) {
    // Compute padding for alignment in the current block.
    uintptr_t current_ptr =
        (uintptr_t)(arena_block_data(arena->current) + arena->current->index);
    size_t padding = arena_align_padding(current_ptr, alignment);

    size_t used = 0;
    if (arena_add_overflow(arena->current->index, padding, &used) ||
        arena_add_overflow(used, size, &used))
      return ARENA_NULLPTR;

    if (used <= arena->current->capacity) {
      arena->current->index += padding;
      void *ptr = arena_block_data(arena->current) + arena->current->index;
      arena->current->index += size;
      return ptr;
    }

    // Reuse existing next block (important after arena_reset()).
    if (arena->current->next) {
      arena->current = arena->current->next;
      continue;
    }

    struct ArenaBlock *new_block =
        arena_block_create(arena, arena_block_size_for(arena, min_needed));
    if (!new_block)
      return ARENA_NULLPTR;

    arena->current->next = new_block;
    arena->current = new_block;
  }
}

/* Rewind a chain of blocks to empty without releasing any of them. */
static void arena_rewind_chain(struct ArenaBlock *block) {
  while (block) {
    block->index = 0;
    block = block->next;
  }
}

void arena_reset(Arena *arena) {
  if (!arena)
    return;

  arena_rewind_chain(arena->head);
  arena->current = arena->head;
}

void arena_free(Arena *arena) {
  if (!arena)
    return;

  struct ArenaBlock *block = arena->head;
  while (block) {
    struct ArenaBlock *next = block->next;
    arena_block_destroy(arena, block);
    block = next;
  }
  ARENA_FREE(arena);
}

ArenaCheckpoint arena_checkpoint(Arena *arena) {
  ArenaCheckpoint cp = {ARENA_NULLPTR, 0};

  assert(arena != ARENA_NULLPTR && "arena_checkpoint: arena is NULL");
  if (!arena || !arena->current)
    return cp; // Arena not yet allocated: zero checkpoint means "empty state".

  cp.block = arena->current;
  cp.index = arena->current->index;
  return cp;
}

void arena_restore(Arena *arena, ArenaCheckpoint checkpoint) {
  assert(arena != ARENA_NULLPTR && "arena_restore: arena is NULL");
  if (!arena)
    return;

  // Checkpoint taken before the first allocation: rewind everything.
  if (checkpoint.block == ARENA_NULLPTR) {
    assert(checkpoint.index == 0 &&
           "arena_restore: invalid empty-state checkpoint index");
    arena_reset(arena);
    return;
  }

// Debug validation: ensure checkpoint belongs to this arena.
#ifndef NDEBUG
  struct ArenaBlock *owned = arena->head;
  bool found = false;
  while (owned) {
    if (owned == checkpoint.block) {
      found = true;
      break;
    }
    owned = owned->next;
  }
  assert(found && "arena_restore: checkpoint does not belong to this arena");
  assert(checkpoint.index <= checkpoint.block->capacity &&
         "arena_restore: checkpoint index is invalid");
#endif

  // Blocks after the checkpoint are rewound, never freed: freeing them would
  // dangle any outstanding checkpoint pointing into them. Only arena_free()
  // returns memory to the system.
  arena_rewind_chain(checkpoint.block->next);
  checkpoint.block->index = checkpoint.index;
  arena->current = checkpoint.block;
}

#endif // ARENA_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // ARENA_H
