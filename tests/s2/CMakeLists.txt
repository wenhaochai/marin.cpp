# Stage S2 gates (docs/BUILD_LOG.md). One ctest entry per suite; exit 77 = skipped (fixture absent).
#   t0_7_schedules   T0.7: the fixture tables (through the C++ loader of libmarincpp_kernels.so) vs every logged W&B
#                    value, bitwise; holes and k = 0, 1 vs the formula (tests/s2/test_t0_7.py)
#   s2_formula, s2_fixture         schedule shape, XLA:GPU arithmetic port, loader, C ABI (no fixture needed)
#   s2_levanter, s2_probe          formula == jitted-levanter fixture bitwise at every k (cache/levref; skip without)
# The fixtures themselves come from tests/s2/schedule_fixture.py (marin venv, a GPU; see its docstring).
add_executable(nm_tests_s2 ${PROJECT_SOURCE_DIR}/tests/s0/nmtest.cpp test_sched.cpp)
target_include_directories(nm_tests_s2 PRIVATE ${PROJECT_SOURCE_DIR}/tests/s0)
target_link_libraries(nm_tests_s2 PRIVATE ${NM_MODULE_TARGETS})

foreach(suite s2_formula s2_fixture s2_levanter s2_probe)
  add_test(NAME ${suite} COMMAND nm_tests_s2 ${suite} --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(${suite} PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

if(NM_PYTHON)
  add_test(NAME t0_7_schedules
           COMMAND ${NM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_t0_7.py --lib $<TARGET_FILE:marincpp_kernels>
                   --src ${PROJECT_SOURCE_DIR})
  set_tests_properties(t0_7_schedules PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake
       "message(FATAL_ERROR \"t0_7_schedules: no Python 3 interpreter at configure time; reconfigure with -DNM_PYTHON=<python3>\")\n")
  add_test(NAME t0_7_schedules COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_python.cmake)
endif()
