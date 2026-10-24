# Stage S0 gates (docs/BUILD_LOG.md). One ctest entry per gate; exit 77 = skipped (no GPU / fixture absent).
#   t0_6_constants      T0.6 optimizer scalar constants vs optim.md §5.3 hex
#   t0_8_params_flops   T0.8 parameter count and flops_per_example vs the spec table and W&B
#   s0_plan_vs_budget   memory planner + step budget vs scripts/design_budget.py and DESIGN.md §2.5/§12.2
#   s0_core, s0_config, s0_memory, s0_arena (GPU), s0_device (GPU): skeleton units
#   s0_headers          (build-time) every public header compiles on its own
add_executable(nm_tests_s0 nmtest.cpp test_core.cpp test_config.cpp test_t0_6.cpp test_t0_8.cpp test_memory.cpp
                           test_arena.cpp)
target_link_libraries(nm_tests_s0 PRIVATE ${NM_MODULE_TARGETS})

foreach(pair "t0_6_constants:t0_6" "t0_8_params_flops:t0_8" "s0_core:s0_core" "s0_config:s0_config"
             "s0_memory:s0_memory" "s0_arena:s0_arena" "s0_device:s0_device")
  string(REPLACE ":" ";" p "${pair}")
  list(GET p 0 test_name)
  list(GET p 1 suite)
  add_test(NAME ${test_name} COMMAND nm_tests_s0 ${suite} --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
set_tests_properties(s0_config PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# The runner reports a case that failed a check and then skipped as FAIL (test_core.cpp, suite s0_runner_selftest).
add_test(NAME s0_runner_selftest COMMAND nm_tests_s0 s0_runner_selftest --src ${PROJECT_SOURCE_DIR})
set_tests_properties(s0_runner_selftest PROPERTIES PASS_REGULAR_EXPRESSION
                     "FAIL s0_runner_selftest.fail_then_skip")

if(NM_PYTHON)
  add_test(NAME s0_plan_vs_budget
           COMMAND ${NM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_plan_vs_budget.py --bin $<TARGET_FILE:marincpp-train>
                   --src ${PROJECT_SOURCE_DIR})
else()
  # The gate must not vanish: without Python it is a test that fails and says why.
  message(WARNING "no Python 3 found: s0_plan_vs_budget is registered as a failing test (set -DNM_PYTHON=<python3>)")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake
       "message(FATAL_ERROR \"s0_plan_vs_budget: no Python 3 interpreter at configure time; reconfigure with -DNM_PYTHON=<python3>\")\n")
  add_test(NAME s0_plan_vs_budget COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake)
endif()

# Header self-containment: one translation unit per public header, compiled with the project flags; capi.h is also
# compiled as C11 (it is the C ABI of libmarincpp_kernels.so).
file(GLOB_RECURSE NM_PUBLIC_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/marincpp/*.hpp
     ${PROJECT_SOURCE_DIR}/include/marincpp/*.h)
set(hdr_srcs "")
foreach(h IN LISTS NM_PUBLIC_HEADERS)
  string(MAKE_C_IDENTIFIER "${h}" id)
  set(f ${CMAKE_CURRENT_BINARY_DIR}/headers/${id}.cpp)
  file(CONFIGURE OUTPUT ${f} CONTENT "#include \"${h}\"\n")
  list(APPEND hdr_srcs ${f})
endforeach()
set(capi_c ${CMAKE_CURRENT_BINARY_DIR}/headers/capi_as_c.c)
file(CONFIGURE OUTPUT ${capi_c} CONTENT "#include \"marincpp/capi.h\"\nint nm_capi_compiles_as_c(void) { return NM_OK; }\n")
set_property(SOURCE ${capi_c} APPEND PROPERTY COMPILE_OPTIONS -Wpedantic -Wstrict-prototypes)
list(APPEND hdr_srcs ${capi_c})
add_library(nm_s0_headers OBJECT ${hdr_srcs})
target_link_libraries(nm_s0_headers PRIVATE nm_flags)
add_dependencies(nm_tests_s0 nm_s0_headers)
