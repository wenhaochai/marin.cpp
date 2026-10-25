# Stage S4 gates (docs/BUILD_LOG.md): GEMM layer and attention through the C ABI of libmarincpp_kernels.so (torch
# oracles), T1.4 against cache/levref, and two C++ executables. Exit 77 = skipped (no GPU, not sm_90, fixture absent).
#   s4_gemm          linear fwd / dgrad / wgrad (both roundings) vs f64, repeatability, f32 GEMMs
#   s4_attention     reference attention vs f64; every FA3 mode vs the reference at the owner shapes (sm_90)
#   s4_t1_4          T1.4: every built backend against the levanter fixtures with the oracle rule
#   s4_graph_nccl    one process, a thread per GPU: FA3 fwd + bwd and an NCCL all-reduce captured in one graph,
#                    100 replays bitwise equal to the eager run (>= 2 sm_90 GPUs)
#   nm_bench_s4_gemm (not a ctest) the GemmCache search over the owner's GEMM shapes, JSON lines (tests/s4/gemm_report.py)
#   bench_attention.py (not a ctest) E1b: FA3 modes, deterministic on / off, timings on the real segment rows
set(_nm_torch_default "")
if(DEFINED ENV{NANOMARIN_ROOT} AND EXISTS "$ENV{NANOMARIN_ROOT}/envs/nanomarin/bin/python")
  set(_nm_torch_default "$ENV{NANOMARIN_ROOT}/envs/nanomarin/bin/python")
endif()
set(NM_TORCH_PYTHON "${_nm_torch_default}" CACHE FILEPATH "Python with torch + numpy for the kernel gates")

set(_s4_tests s4_gemm:test_gemm.py s4_attention:test_attention.py s4_t1_4:test_t1_4.py)
foreach(pair IN LISTS _s4_tests)
  string(REPLACE ":" ";" p "${pair}")
  list(GET p 0 name)
  list(GET p 1 script)
  if(NM_TORCH_PYTHON)
    add_test(NAME ${name}
             COMMAND ${NM_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/${script} --lib $<TARGET_FILE:marincpp_kernels>
                     --src ${PROJECT_SOURCE_DIR})
    # TIMEOUT: s4_gemm checks every search candidate (~1,000 GEMMs plus f64 references on a shared GPU)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1" TIMEOUT 5400)
  else()
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake
         "message(FATAL_ERROR \"S4 gate: no Python with torch at configure time; reconfigure with -DNM_TORCH_PYTHON=<python>\")\n")
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake)
  endif()
endforeach()

# GEMM bench: the GEMM layer and core only (no FA3 objects), so a CUDA 13.0 tree can build just this target quickly.
add_executable(nm_bench_s4_gemm bench_gemm.cpp)
target_link_libraries(nm_bench_s4_gemm PRIVATE nm_kernels_gemm nm_kernels_gemm_sm90a nm_core)

# NCCL: -DNM_NCCL_ROOT=<dir with include/nccl.h and lib/libnccl.so*>; default the nvidia-nccl wheel of the torch env.
set(_nccl_hint "")
if(DEFINED ENV{NANOMARIN_ROOT})
  file(GLOB _nccl_hint LIST_DIRECTORIES true "$ENV{NANOMARIN_ROOT}/envs/nanomarin/lib/python3*/site-packages/nvidia/nccl")
endif()
set(NM_NCCL_ROOT "${_nccl_hint}" CACHE PATH "NCCL root (include/nccl.h, lib/libnccl.so*)")
find_path(NM_NCCL_INCLUDE nccl.h HINTS ${NM_NCCL_ROOT}/include NO_DEFAULT_PATH)
find_library(NM_NCCL_LIB NAMES nccl libnccl.so.2 HINTS ${NM_NCCL_ROOT}/lib NO_DEFAULT_PATH)
if(NM_NCCL_INCLUDE AND NM_NCCL_LIB)
  add_executable(nm_test_s4_graph_nccl test_graph_nccl.cpp)
  target_include_directories(nm_test_s4_graph_nccl PRIVATE ${NM_NCCL_INCLUDE})
  target_link_libraries(nm_test_s4_graph_nccl PRIVATE ${NM_MODULE_TARGETS} ${NM_NCCL_LIB})
  add_test(NAME s4_graph_nccl COMMAND nm_test_s4_graph_nccl --gpus 4 --replays 100)
  set_tests_properties(s4_graph_nccl PROPERTIES SKIP_RETURN_CODE 77)
else()
  message(WARNING "NCCL not found (NM_NCCL_ROOT): s4_graph_nccl is registered as a failing test")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_nccl.cmake
       "message(FATAL_ERROR \"s4_graph_nccl: NCCL not found at configure time; set -DNM_NCCL_ROOT\")\n")
  add_test(NAME s4_graph_nccl COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_nccl.cmake)
endif()
