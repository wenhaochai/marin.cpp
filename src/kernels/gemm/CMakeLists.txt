# src/kernels/gemm: stage S4 (GemmCache timed search over deterministic cuBLASLt algorithms and custom kernels, C ABI);
# DESIGN.md §3.7.
#   PORTABLE  gemm_cache.cpp (search, persistence), gemm_kernels.cu (portable Levanter wgrad round-add, fills),
#             capi_gemm.cpp (C ABI)
#   SM90A     gemm_sm90_*.cu: CUTLASS SM90 kernels registered as GemmCache candidates (gemm_sm90.cuh)
nm_module(kernels/gemm
  PORTABLE gemm_cache.cpp gemm_kernels.cu capi_gemm.cpp
  SM90A gemm_sm90_fwd.cu gemm_sm90_dgrad.cu gemm_sm90_wgrad.cu gemm_sm90_wgrad_lev.cu
  DEPS CUDA::cublasLt)
# SYSTEM: CUTLASS's sm100 collective headers (pulled in by the collective builder) warn under C++20
# (implicit capture of this), which --Werror=all-warnings would turn into errors.
target_include_directories(nm_kernels_gemm_sm90a SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
                           ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include)
target_compile_definitions(nm_kernels_gemm_sm90a PRIVATE NDEBUG)
target_compile_options(nm_kernels_gemm_sm90a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
