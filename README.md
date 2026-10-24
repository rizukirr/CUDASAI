# cudasai

Tiny LLM inference on CUDA, written from scratch as a learning project. The goal is to run GPT-2 small on the GPU using only the CUDA runtime, cuBLASLt and code in this repository.

> [!WARNING]
> This is not production-ready.

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

## License

Apache 2.
