# Stage S1 gates (docs/BUILD_LOG.md). One ctest entry per gate; exit 77 = skipped (no GPU / converted set absent).
#   t0_1_order        T0.1 order table sha256 (computed in C++), element-wise vs the levanter table, C ABI
#   t0_2_batches      T0.2 the 12 batch hashes through the loader (converted fineweb set)
#   t0_3_segments     T0.3 segment ids + loss weights of the 12 batches vs the data.md §3.1 formula, element by element,
#                     and the converter's counts (segs_per_seq.u16)
#   t0_3_levanter     T0.3 vs levanter's arrays: levanter_ref.json batches 0, 1, 1000, 4958 and the batch-0 fixture
#                     (skipped, visibly, while a reference is absent)
#   t0_4_paloma       T0.4 16 Paloma domain hashes, 14,529 windows, 114 batches / 65 real rows in batch 113; exact
#                     segment counts of every window (max 291)
#   t0_4_levanter     T0.4 vs levanter's eval batches 19-21 (tokens, tags) and 113 (tokens) (skipped while absent)
#   t0_5_bytes        T0.5 bytes-per-token tables
#   s1_threefry       threefry known answers, levanter keys, block permutation, device == host (GPU)
#   s1_plan           data plans: default, Repeat (max_train_batches restart), reshuffle, explicit, files, fork, passes
#   s1_loader         prefetch ring, pinned slots, H2D (GPU), restart/fork, error propagation (pread EOF, mmap SIGBUS),
#                     stop() while workers block or call, start() input checks
#   s1_convert_outputs  converter outputs (sizes, sha256, T0.1/T0.4/T0.5 from the files) and the data C ABI (Python)
# nm_bench_s1 (not a test): sustained reader throughput on the converted set.
add_executable(nm_tests_s1 ${PROJECT_SOURCE_DIR}/tests/s0/nmtest.cpp test_order.cpp test_batches.cpp test_paloma.cpp
                           test_plan.cpp test_loader.cpp)
target_link_libraries(nm_tests_s1 PRIVATE ${NM_MODULE_TARGETS})

foreach(suite t0_1_order t0_2_batches t0_3_segments t0_3_levanter t0_4_paloma t0_4_levanter t0_5_bytes s1_threefry
              s1_plan s1_loader)
  add_test(NAME ${suite} COMMAND nm_tests_s1 ${suite} --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(${suite} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

add_executable(nm_bench_s1 bench_loader.cpp)
target_link_libraries(nm_bench_s1 PRIVATE ${NM_MODULE_TARGETS})

if(NM_PYTHON)
  add_test(NAME s1_convert_outputs
           COMMAND ${NM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_outputs.py
                   --lib $<TARGET_FILE:marincpp_kernels> --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(s1_convert_outputs PROPERTIES SKIP_RETURN_CODE 77)
else()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake
       "message(FATAL_ERROR \"s1_convert_outputs: no Python 3 interpreter at configure time\")\n")
  add_test(NAME s1_convert_outputs COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake)
endif()
