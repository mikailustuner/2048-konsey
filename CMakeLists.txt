cmake_minimum_required(VERSION 3.18)
project(konsey LANGUAGES CXX)

# C++20, not 23: Colab's Ubuntu g++ (11.x) must build this too.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native: binaries are machine specific, rebuild on every machine.
# -ffp-contract=off (and --fmad=false for CUDA): no FMA contraction, so CPU and GPU compute
# bit-identical values.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffp-contract=off")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

find_package(Threads REQUIRED)

file(GLOB LIB_SOURCES CONFIGURE_DEPENDS
  engine/*.cpp ntuple/*.cpp search/*.cpp evolve/*.cpp council/*.cpp tools/agents.cpp)
add_library(konsey STATIC ${LIB_SOURCES})
target_include_directories(konsey PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(konsey PUBLIC Threads::Threads)

foreach(tool eval bench replay train evolve distill play)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE konsey)
endforeach()

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE tests/third_party)
target_link_libraries(tests PRIVATE konsey)
add_test(NAME tests COMMAND tests)

# Faz 4b: GPU training/evaluation. Colab A100: -DUSE_CUDA=ON (architecture detected).
option(USE_CUDA "CUDA kernels for batched training and evaluation" OFF)
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false")
  file(GLOB GPU_SOURCES CONFIGURE_DEPENDS gpu/*.cu gpu/*.cpp)
  add_library(konsey_gpu STATIC ${GPU_SOURCES})
  target_link_libraries(konsey_gpu PUBLIC konsey CUDA::cudart_static)
  target_compile_definitions(konsey_gpu PUBLIC K2048_CUDA)
  foreach(tool train evolve eval)
    target_link_libraries(${tool} PRIVATE konsey_gpu)
  endforeach()
  add_executable(train_many tools/train_many.cpp)
  target_link_libraries(train_many PRIVATE konsey_gpu)
  add_executable(gpu_tests tests/main.cpp tests/gpu/test_gpu.cpp)
  target_include_directories(gpu_tests PRIVATE tests/third_party)
  target_link_libraries(gpu_tests PRIVATE konsey_gpu)
  add_test(NAME gpu_tests COMMAND gpu_tests)
endif()
