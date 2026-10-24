# Stage S3 gates (docs/BUILD_LOG.md): elementwise kernels through the C ABI of libmarincpp_kernels.so, checked with
# torch (f64 autograd oracles, levanter-rounding emulation) and the levanter fixtures of tools/levref. One ctest entry
# per script; exit 77 = skipped (no GPU).
#   s3_rmsnorm       RMSNorm over D and the sandwich tails (sandwich and pre-LN placement), all storage x rounding forms
#   s3_qknorm_rope   QK-norm + RoPE + q scale forward/backward, dq-postprocess fusion (MHA hd 64, GQA hd 64 / hd 128)
#   s3_swiglu        SwiGLU forward/backward on the interleaved gate/up layout
#   s3_embed         embedding gather, deterministic sorted segment-sum backward (f32 and bf16-sequential)
#   s3_misc          doc_meta, fixed-order reductions, f32 -> bf16 cast
#   s3_t1_levref     T1.2 (RMSNorm) and T1.3 (RoPE) against cache/levref fixtures (cases skip until a fixture exists)
#   s3_bench         achieved GB/s per kernel (label bench; report only)
#
# The interpreter needs torch and numpy: -DNM_TORCH_PYTHON=<python>, default $NANOMARIN_ROOT/envs/nanomarin/bin/python.
set(_nm_torch_default "")
if(DEFINED ENV{NANOMARIN_ROOT} AND EXISTS "$ENV{NANOMARIN_ROOT}/envs/nanomarin/bin/python")
  set(_nm_torch_default "$ENV{NANOMARIN_ROOT}/envs/nanomarin/bin/python")
endif()
set(NM_TORCH_PYTHON "${_nm_torch_default}" CACHE FILEPATH "Python with torch + numpy for the S3 kernel gates")

set(_s3_tests
  s3_rmsnorm:test_rmsnorm.py s3_qknorm_rope:test_qknorm_rope.py s3_swiglu:test_swiglu.py s3_embed:test_embed.py
  s3_misc:test_misc.py s3_t1_levref:test_t1_levref.py s3_bench:bench_s3.py)

if(NM_TORCH_PYTHON)
  message(STATUS "marincpp: S3 kernel gates use ${NM_TORCH_PYTHON}")
  foreach(pair IN LISTS _s3_tests)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    list(GET p 1 script)
    add_test(NAME ${name}
             COMMAND ${NM_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/${script} --lib $<TARGET_FILE:marincpp_kernels>
                     --src ${PROJECT_SOURCE_DIR})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endforeach()
  set_tests_properties(s3_bench PROPERTIES LABELS bench)
else()
  # The gates must not vanish: without a torch interpreter each one is a test that fails and says why.
  message(WARNING "no torch interpreter (NM_TORCH_PYTHON / $NANOMARIN_ROOT/envs/nanomarin): S3 gates registered as failing")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake
       "message(FATAL_ERROR \"S3 gate: no Python with torch at configure time; reconfigure with -DNM_TORCH_PYTHON=<python>\")\n")
  foreach(pair IN LISTS _s3_tests)
    string(REPLACE ":" ";" p "${pair}")
    list(GET p 0 name)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/no_torch.cmake)
  endforeach()
endif()
