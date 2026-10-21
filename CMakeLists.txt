cmake_minimum_required(VERSION 3.32)
project(cudasai VERSION 0.1.0 LANGUAGES C CUDA)

set(CUDASAI_VERSION_SUFFIX "-rc.1")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/version.h"
    @ONLY
)


add_executable(cudasai
  src/main.cu
  src/array.cu
  src/tensor.cu
  src/_array.cu
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cudasai PRIVATE
  CUDA::cublasLt -lm)

target_include_directories(cudasai PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
  "${CMAKE_CURRENT_BINARY_DIR}/vendor"
)

set_target_properties(cudasai PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  CUDA_STANDARD 23
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
)
