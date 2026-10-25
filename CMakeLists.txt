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

set_target_properties(cdscore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(example)

include(CTest)
if(BUILD_TESTING)
  add_executable(arena_test tests/test_arena.cpp)
  add_test(NAME arena.cpu COMMAND arena_test)

  add_executable(arena_cuda_mock_test tests/test_arena.cpp)
  target_compile_definitions(arena_cuda_mock_test PRIVATE ARENA_USE_CUDA)
  target_include_directories(arena_cuda_mock_test PRIVATE tests/cuda_mock)
  add_test(NAME arena.cuda_mock COMMAND arena_cuda_mock_test)

  add_library(arena_cuda_declaration_test OBJECT
    tests/test_arena_cuda_declaration.cpp)
  target_include_directories(arena_cuda_declaration_test PRIVATE
    tests/cuda_mock)

  add_executable(public_api_test
    tests/test_api_main.cu
    tests/test_host.cu
    tests/test_array.cu
    tests/test_cds_to_meta.cu
    tests/test_view.cu
    tests/test_copy.cu
    tests/test_random.cu
    tests/test_checkpoint.cu
    tests/test_context.cu
    tests/test_fill.cu
    tests/test_add.cu
    tests/test_scale.cu
    tests/test_linear.cu
  )
  target_link_libraries(public_api_test PRIVATE cdscore)
  set_target_properties(public_api_test
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME api.host COMMAND public_api_test host)
  foreach(dtype float double)
    foreach(case
        array
        cds_to_meta
        view
        copy
        random
        checkpoint
        context
        fill
        add
        scale
        linear)
      add_test(NAME api.${case}.${dtype} COMMAND
        public_api_test ${case} ${dtype})
      set_tests_properties(api.${case}.${dtype} PROPERTIES
        SKIP_RETURN_CODE 77 LABELS gpu TIMEOUT 60)
    endforeach()
  endforeach()
endif()
