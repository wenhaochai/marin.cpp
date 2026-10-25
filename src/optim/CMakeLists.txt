# src/optim + src/kernels/optim: stage S6 (optimizer engine, ownership, Newton-Schulz, C ABI); DESIGN.md §4, §6.1.
# The device kernels live in src/kernels/optim and build here (one module). The update kernels build with
# --fmad=false: every formula spells its operation order with __fmul_rn / __fadd_rn / __fdiv_rn (optim.md).
set(_k ${PROJECT_SOURCE_DIR}/src/kernels/optim)
nm_module(optim
  PORTABLE engine.cpp capi_optim.cpp
           ${_k}/optk_reduce.cu ${_k}/optk_adam.cu ${_k}/optk_muon.cu ${_k}/ns_gemm.cpp
  FMAD_OFF ${_k}/optk_reduce.cu ${_k}/optk_adam.cu ${_k}/optk_muon.cu
  DEPS CUDA::cublasLt)
