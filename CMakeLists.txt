# marincpp: pure C++/CUDA training binary (docs/DESIGN.md §13).
#
# Targets
#   marincpp            static library: every module's objects (libmarincpp.a)
#   marincpp_kernels    shared library with the C ABI of include/marincpp/capi.h, for pytest + ctypes
#   marincpp-train      the training binary (subcommands: train, plan, config, knobs)
#   nm_tests_<stage>    C++ gate tests (tests/<stage>/), registered with ctest
#
# Architectures: portable sources build for NM_ARCHS_PORTABLE (sm_90a for H100 plus sm_80 so tests run on the vis
# node's A100); sources a module lists as SM90A (WGMMA/TMA kernels, FA3) build for NM_ARCHS_SM90A only. A production
# build may set -DNM_ARCHS_PORTABLE=90a-real.
#
# Build (vis node):
#   module load gcc-toolset/14 cudatoolkit/12.8
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8 && ctest --test-dir build
cmake_minimum_required(VERSION 3.30)
project(marincpp VERSION 0.1.0 LANGUAGES C CXX CUDA)   # C only to check that capi.h compiles as C

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Shared cudart: the C ABI library is loaded next to torch's runtime in the test process.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(NM_ARCHS_PORTABLE "90a-real;80-real" CACHE STRING "CUDA architectures of portable kernels")
set(NM_ARCHS_SM90A "90a-real" CACHE STRING "CUDA architectures of sm_90a-only kernels")
# Default of every CUDA target outside nm_module (tests, final targets): the portable split, never CMake's sm_52.
set(CMAKE_CUDA_ARCHITECTURES "${NM_ARCHS_PORTABLE}")

find_package(CUDAToolkit 12.8 REQUIRED)
# Python 3 runs the Python gates (tests/*/test_*.py, no third-party modules). Without one, those gates are registered as
# failing tests (tests/s0/CMakeLists.txt), never dropped. -DNM_PYTHON=<python3> picks the interpreter.
find_package(Python3 3.9 COMPONENTS Interpreter)
set(NM_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python used by the Python gate tests")
message(STATUS "marincpp: Python gates use ${NM_PYTHON}")

include(cmake/MarinFlags.cmake)
include(cmake/MarinModule.cmake)
include(cmake/GitVersion.cmake)

# Modules of DESIGN.md §13.2, in dependency order. Each src/<module>/CMakeLists.txt calls nm_module().
set(NM_MODULES
  core config memory
  data sched
  kernels/elementwise kernels/gemm kernels/attention kernels/cross_entropy
  optim model comm eval metrics checkpoint train)
foreach(m IN LISTS NM_MODULES)
  add_subdirectory(src/${m})
endforeach()
get_property(NM_MODULE_TARGETS GLOBAL PROPERTY NM_MODULE_TARGETS)

# Final targets link the module OBJECT libraries directly: every object (and its static registrations: knobs, blocks,
# commands) is linked, and the modules' usage requirements (CUDA libraries, NCCL) propagate.
add_library(marincpp STATIC src/marincpp.cpp)
target_link_libraries(marincpp PUBLIC ${NM_MODULE_TARGETS})

add_library(marincpp_kernels SHARED src/marincpp.cpp)
target_link_libraries(marincpp_kernels PRIVATE ${NM_MODULE_TARGETS})
# Export only the C ABI (nm_*): no nm:: C++ symbol can interpose with another library loaded in the same process.
target_link_options(marincpp_kernels PRIVATE "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/capi.map")
set_property(TARGET marincpp_kernels APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/cmake/capi.map)

add_executable(marincpp-train src/main.cpp)
target_link_libraries(marincpp-train PRIVATE ${NM_MODULE_TARGETS})

enable_testing()
file(GLOB NM_TEST_DIRS LIST_DIRECTORIES true ${PROJECT_SOURCE_DIR}/tests/*)
foreach(d IN LISTS NM_TEST_DIRS)
  if(IS_DIRECTORY ${d} AND EXISTS ${d}/CMakeLists.txt)
    add_subdirectory(${d})
  endif()
endforeach()
