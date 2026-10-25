# Stage S5 gates (docs/BUILD_LOG.md S5): the fused linear CE through the C ABI of libmarincpp_kernels.so, checked with
# torch (f64 autograd oracles, the levanter emulation of numerics.md §5.2) and the T1.5 levref fixture. One ctest entry
# per script; exit 77 = skipped (no GPU).
#   s5_ce           A / B1 / B1.5 / B2 gradient checks at V = 512 / 8256 / 64256 / 128256 x h = 512 / 768 / 1024 /
#                   2048, f32 store, bitwise invariants, in-place and accumulated dX, strict dW, f32 io, odd shapes,
#                   readout heads
#   s5_t15_levref   T1.5 against cache/levref/<size>/t1_units/ce and its f64 oracle (cases skip without a fixture)
#   s5_bench        ms per call at the E2 shapes (label bench; report only)
# The interpreter needs torch and numpy: -DNM_TORCH_PYTHON (tests/s3/CMakeLists.txt).
set(_s5_tests s5_ce:test_ce.py s5_t15_levref:test_t15_levref.py s5_bench:bench_ce.py)

if(NM_TORCH_PYTHON)
  foreach(pair IN LISTS _s5_tests)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    list(GET p 1 script)
    add_test(NAME ${name}
             COMMAND ${NM_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/${script} --lib $<TARGET_FILE:marincpp_kernels>
                     --src ${PROJECT_SOURCE_DIR})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endforeach()
  set_tests_properties(s5_bench PROPERTIES LABELS bench)
else()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake
       "message(FATAL_ERROR \"S5 gate: no Python with torch at configure time; reconfigure with \
-DNM_TORCH_PYTHON=<python>\")\n")
  foreach(pair IN LISTS _s5_tests)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake)
  endforeach()
endif()
