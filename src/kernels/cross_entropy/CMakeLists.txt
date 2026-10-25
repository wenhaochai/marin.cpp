# src/kernels/cross_entropy: stage S5, fused linear cross-entropy (implementations A, B1, B1.5, B2; readout heads);
# DESIGN.md §3.8. Portable kernels (mma.sync, sm_80 + sm_90a) and the sm_90a WGMMA/TMA kernels (ce_sm90.cu). The kernel
# units build with --fmad=false: the CE gradient p * g - g and the statistics merges are written as explicit roundings.
nm_module(kernels/cross_entropy
  PORTABLE ce_logits.cu ce_gemm.cu ce_small.cu ce_lt.cpp ce_api.cpp capi_ce.cpp
  SM90A ce_sm90.cu
  FMAD_OFF ce_logits.cu ce_gemm.cu ce_small.cu ce_sm90.cu
  DEPS CUDA::cublasLt)
