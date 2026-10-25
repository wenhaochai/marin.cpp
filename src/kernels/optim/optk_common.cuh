// Shared device helpers of the optimizer kernels (src/kernels/optim). Built in --fmad=false units; every formula
// spells its operations with __fmul_rn / __fadd_rn / __fdiv_rn / __fsqrt_rn, so no contraction or fast division can
// change the operation order of optim.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/optim/optim_dev.hpp"

namespace nm::optk::dev {

__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float sqrt_rn(float a) { return __fsqrt_rn(a); }
// jnp.maximum on f32 (NaN-propagating, as XLA's maximum with fast_min_max off)
__device__ __forceinline__ float fmax_nan(float a, float b) { return (a != a || a > b) ? a : b; }

// optax clip_by_global_norm (optim.md §3.1): select(gn < max, g, (g / gn) * max).
__device__ __forceinline__ float clip_g(float g, const ClipDev& c) {
  return c.clip ? mul(div(g, c.gn), c.max_norm) : g;
}

// A norm from an f64 sum of f32 squares: one rounding of the sum to f32, then an f32 square root (XLA: sqrt of the
// f32 reduce).
__device__ __forceinline__ float norm_of(double sumsq) { return sqrt_rn(static_cast<float>(sumsq)); }

// Block-wide sum in a fixed order (blockDim <= 1024): butterfly within each warp, then warp 0 over the warp sums.
// The order depends only on the block shape, so the result is bitwise repeatable. Valid in thread 0.
template <class T>
__device__ __forceinline__ T block_sum(T v) {
  __shared__ T ws[32];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthr = blockDim.x * blockDim.y;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = tid & 31, w = tid >> 5;
  __syncthreads();   // ws may still be read by a previous call
  if (lane == 0) ws[w] = v;
  __syncthreads();
  const int nw = (nthr + 31) >> 5;
  T s = lane < nw ? ws[lane] : T(0);
  if (w == 0)
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

__device__ __forceinline__ double sq(float x) { return static_cast<double>(mul(x, x)); }

// Slice table entry, decoded.
struct Slice {
  int tensor;
  int64_t row;   // full-tensor row of the first element
  int col;
  int len;
  int block;
  int slot;
};
__device__ __forceinline__ Slice slice_at(const SliceTab& s, int i) {
  return Slice{s.tensor[i], s.row[i], s.col[i], s.len[i], s.block[i], s.slot[i]};
}
// Element offset of a slice's first element relative to the tensor's pointers.
__device__ __forceinline__ int64_t slice_off(const TensorTab& t, const Slice& sl) {
  return (sl.row - t.base_row[sl.tensor]) * t.cols[sl.tensor] + sl.col;
}

// The update index k; k outside the tables [0, N) is a program error (a run past its schedule): trap, so the next
// stream synchronization fails instead of reading past the lr and bias-correction tables.
__device__ __forceinline__ int32_t cur_step(const Tables& tb) {
  const int32_t k = *tb.step;
  if (k < 0 || k >= tb.n_steps) __trap();
  return k;
}

// float4 access of a slice: every pointer 16-byte aligned (bf16 copies 8-byte) and the length a multiple of 4.
__device__ __forceinline__ bool al16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
__device__ __forceinline__ bool al8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 7) == 0; }
struct F4 {
  float v[4];
  __device__ __forceinline__ void load(const float* p, int q) {
    const float4 x = reinterpret_cast<const float4*>(p)[q];
    v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
  }
  __device__ __forceinline__ void store(float* p, int q) const {
    reinterpret_cast<float4*>(p)[q] = make_float4(v[0], v[1], v[2], v[3]);
  }
};
__device__ __forceinline__ void store_bf16x4(bf16_t* p16, int q, const F4& x) {
  __nv_bfloat162 a = __floats2bfloat162_rn(x.v[0], x.v[1]), b = __floats2bfloat162_rn(x.v[2], x.v[3]);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&a);
  w.y = *reinterpret_cast<uint32_t*>(&b);
  reinterpret_cast<uint2*>(p16)[q] = w;
}

// Frozen block at step k (row-partition blocks; -1 = no partition).
__device__ __forceinline__ bool block_frozen(const RowPartTab& rp, int block, int32_t k) {
  return block >= 0 && k >= rp.block_frozen_from[block];
}
// lr multiplier of a block at step k (1 without a table).
__device__ __forceinline__ float block_lr_mult(const RowPartTab& rp, const Tables& tb, int block, int32_t k) {
  if (block < 0) return 1.0f;
  const int tab = rp.block_lr_table[block];
  return tab < 0 ? 1.0f : tb.lr[tab][k];
}
// Lazy rows: the update skips rows not touched this step.
__device__ __forceinline__ bool row_skipped(const RowPartTab& rp, int block, int64_t row) {
  if (block < 0) return false;
  const uint8_t* touched = rp.touched[rp.block_rp[block]];
  return touched != nullptr && touched[row] == 0;
}

}  // namespace nm::optk::dev
