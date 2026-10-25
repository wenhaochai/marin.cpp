// CUTLASS SM90 candidates for the wgrad under Levanter rounding: dW = C + f32(bf16(dY^T X)) (gemm_sm90.cuh).
#include "kernels/gemm/gemm_sm90.cuh"

NM_SM90_GEMM_UNIT("wgrad_lev", cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, float, Fuse::RoundAdd, Op::T, Op::N, DType::f32)
