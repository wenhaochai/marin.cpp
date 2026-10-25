// CUTLASS SM90 candidates for the wgrad dW += dY^T X (accurate, f32, beta = 1) (gemm_sm90.cuh).
#include "kernels/gemm/gemm_sm90.cuh"

NM_SM90_GEMM_UNIT("wgrad", cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, float, Fuse::LinComb, Op::T, Op::N, DType::f32)
