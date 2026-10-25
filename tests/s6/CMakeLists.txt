# Stage S6 gates (docs/BUILD_LOG.md S6). Exit 77 = skipped (no GPU or no fixture).
#   s6_ownership, s6_zero1, s6_graph, s6_rowpart   C++ (tests/s6/test_engine.cpp): ZeRO-1 ownership plan and layouts,
#                    4 emulated ranks == 1 rank, a captured step == eager steps, row partitions vs an f64 reference
#   s6_ns            TF32 rounding probe, NS vs f64, T1.6 (130m, 1_2b) against cache/levref (tests/s6/test_ns.py)
#   s6_t1_7          T1.7 cases A, B (130m) and A (300m) against cache/levref (tests/s6/test_t1_7.py)
#   s6_t1_7_small_lr T1.7 cases C (130m, lr 5e-6) and B (300m, lr 1e-5): SPECFAIL (exit 1) while item 15 binds there
#                    (BUILD_LOG S6 request 7); case C needs cache/s6/caseC_in from tests/s6/export_case_c.py (once)
#   s6_bench         optimizer step and NS times per size (label bench; report only; tests/s6/bench_s6.py)
add_executable(nm_tests_s6 ${PROJECT_SOURCE_DIR}/tests/s0/nmtest.cpp test_engine.cpp)
target_include_directories(nm_tests_s6 PRIVATE ${PROJECT_SOURCE_DIR}/tests/s0)
target_link_libraries(nm_tests_s6 PRIVATE ${NM_MODULE_TARGETS})
foreach(suite s6_ownership s6_zero1 s6_graph s6_rowpart s6_nsalgo s6_bounds)
  add_test(NAME ${suite} COMMAND nm_tests_s6 ${suite} --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(${suite} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# name:script[:--only cases]
set(_s6_py s6_ns:test_ns.py s6_t1_7:test_t1_7.py:130m_A,130m_B,300m_A s6_t1_7_small_lr:test_t1_7.py:130m_C,300m_B
    s6_bench:bench_s6.py)
if(NM_TORCH_PYTHON)
  foreach(pair IN LISTS _s6_py)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    list(GET p 1 script)
    list(LENGTH p np)
    set(only)
    if(np GREATER 2)
      list(GET p 2 cases)
      set(only --only ${cases})
    endif()
    add_test(NAME ${name}
             COMMAND ${NM_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/${script} --lib $<TARGET_FILE:marincpp_kernels>
                     --src ${PROJECT_SOURCE_DIR} ${only})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endforeach()
  set_tests_properties(s6_bench PROPERTIES LABELS bench)
else()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake
       "message(FATAL_ERROR \"S6 gate: no Python with torch at configure time; reconfigure with -DNM_TORCH_PYTHON=<python>\")\n")
  foreach(pair IN LISTS _s6_py)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake)
  endforeach()
endif()
