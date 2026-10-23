# cudasai

Tiny LLM inference on CUDA, written from scratch as a learning project. The goal is to run GPT-2 small on the GPU using only the CUDA runtime, cuBLASLt and code in this repository.

## Status

The array layer and a cuBLASLt linear layer are done. Elementwise kernels, attention, the model loader and the generation loop are still to come. See [ROADMAP.md](ROADMAP.md) for the full plan and what each stage needs.

## Requirements

- CUDA toolkit 12 or newer with cuBLASLt
- CMake 3.32 or newer
- A C++20 compiler
- An NVIDIA GPU. The build targets the GPU in the machine by default. Pass `-DCMAKE_CUDA_ARCHITECTURES=89` or similar to target another one.

## Build and run

```
cmake -B build
cmake --build build
./build/bin/cudasai
```

The example in `src/main.cu` creates a random 3 to 2 linear layer, then runs three batches through it. Activations are rewound between batches with a checkpoint, so the loop reuses the same device memory every step.

## How memory works

Everything lives in an arena. A root (`Cds`) owns one device arena and one host arena. Arrays (`CdsArray`) are created inside a root and are never freed one by one. Freeing the root releases all of them.

The intended pattern is two roots: one for parameters that live for the whole run, one for activations that are rewound each step.

```c
Cds *params = cds_new(64 << 20);
Cds *acts = cds_new(16 << 20);

CdsArray *W = cds_array_rand(params, w_shape, 2, CDS_FLOAT32);
CdsArray *b = cds_array_rand(params, b_shape, 1, CDS_FLOAT32);

CdsCheckpoint cp = cds_checkpoint(acts);
for (int step = 0; step < steps; step++) {
  CdsArray *x = cds_array(acts, x_shape, 2, CDS_FLOAT32, x_host);
  CdsArray *y = cds_array(acts, y_shape, 2, CDS_FLOAT32, nullptr);
  cds_linear(x, W, b, y);
  cds_restore(acts, cp);
}

cds_free(acts);
cds_free(params);
```

## Layout

```
include/        public headers: array.h, kernel.h
src/            public implementation and the example
src/internal/   structs, metadata helpers, kernel templates and launchers
vendor/         single header arena allocator
tests/          arena tests, CPU only
```

Public functions carry the `cds_` prefix. Internal files start with an underscore and are not installed.
