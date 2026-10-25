// CUTLASS SM90 candidates for the linear dgrad dX = dY W (gemm_sm90.cuh).
#include "kernels/gemm/gemm_sm90.cuh"

NM_SM90_GEMM_UNIT("dgrad", cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::bfloat16_t, Fuse::LinComb, Op::N, Op::N, DType::bf16)
