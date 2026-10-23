cmake_minimum_required(VERSION 3.32)
project(cudasai VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CUDASAI_VERSION_SUFFIX "-rc.1")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/version.h"
    @ONLY
)

find_package(CUDAToolkit REQUIRED)

add_library(cdscore STATIC
  src/array.cu
  src/kernel.cu
  src/internal/_array.cu
  src/internal/_kernel.cu
)

target_include_directories(cdscore PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
)
target_include_directories(cdscore PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/internal"
)

target_link_libraries(cdscore PUBLIC CUDA::cudart CUDA::cublasLt)

# cds_data_at is __host__ __device__ and called from kernels in other
# translation units, which needs relocatable device code.
set_target_properties(cdscore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cudasai src/main.cu)
target_link_libraries(cudasai PRIVATE cdscore)
