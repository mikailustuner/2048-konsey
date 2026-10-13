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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)

file(GLOB LIB_SOURCES CONFIGURE_DEPENDS
  engine/*.cpp ntuple/*.cpp search/*.cpp evolve/*.cpp council/*.cpp tools/agents.cpp)
add_library(konsey STATIC ${LIB_SOURCES})
target_include_directories(konsey PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(konsey PUBLIC Threads::Threads)

foreach(tool eval bench replay)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE konsey)
endforeach()

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE tests/third_party)
target_link_libraries(tests PRIVATE konsey)
add_test(NAME tests COMMAND tests)
