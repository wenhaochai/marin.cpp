# Stage S7 gates (docs/BUILD_LOG.md S7). Exit 77 = skipped (no GPU or no fixture).
#   s7_step0   T2.1 / T2.6 (and the T2.5 table): step-0 loss, gradients and group norms of the baseline model on
#              cache/levref/130m/step0_bf16, one GPU, the whole batch in microbatches (tests/s7/step0.cpp)
add_executable(nm_s7_step0 step0.cpp)
target_link_libraries(nm_s7_step0 PRIVATE ${NM_MODULE_TARGETS})
add_test(NAME s7_step0 COMMAND nm_s7_step0 --root ${PROJECT_SOURCE_DIR})
set_tests_properties(s7_step0 PROPERTIES SKIP_RETURN_CODE 77)
#   nm_s7_train  the multi-GPU step loop (tests/s7/train_main.cpp): throughput and the first-steps loss curve; needs NCCL
if(NM_HAVE_NCCL)
  add_executable(nm_s7_train train_main.cpp)
  target_link_libraries(nm_s7_train PRIVATE ${NM_MODULE_TARGETS})
endif()
