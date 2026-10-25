// CUTLASS SM90 candidates for the linear forward Y = X W^T (gemm_sm90.cuh).
#include "kernels/gemm/gemm_sm90.cuh"

NM_SM90_GEMM_UNIT("fwd", cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::bfloat16_t, Fuse::LinComb, Op::N, Op::T, DType::bf16)
