// Row framework of the norm kernels (RMSNorm over D, QK-norm over hd). Not a public header.
//
// A row of `width` elements is handled by a row group of G = width / 8 threads, one 8-element vector per thread. G is a
// power of two <= 32 (sub-warp groups, shuffle reductions) or a multiple of 32 up to 512 (warp shuffles plus one shared
// memory step). A CTA holds `groups` = blockDim.x / G row groups. The T rows form P parts (norm_parts); CTA p takes
// part p = rows [p T / P, (p + 1) T / P), and in it every row group takes rows part_begin + it * groups + grp.
// Gain-gradient accumulators live in shared memory, one private 8-float slot per thread and gain (no race, no
// registers), and are flushed once per part into partial[p, width] by summing the row groups of the CTA in ascending
// order: every partial depends only on T and the device's SM count, never on timing.
#pragma once

#include "kernels/elementwise/ew_common.cuh"

namespace nm::ew {

inline constexpr int kMaxRowG = 512;              // width <= 4096
inline constexpr int kRedSlots = 2 * 16 * 4;      // [2 phases][<= 16 warps][<= 4 values]

struct RowGeo {
  int G, groups, grp, lig, col0;
  __device__ __forceinline__ explicit RowGeo(int width) {
    G = width >> 3;
    groups = blockDim.x / G;
    grp = threadIdx.x / G;
    lig = threadIdx.x - grp * G;
    col0 = lig * 8;
  }
};

// Sum each of v[0..N) over the row group; every thread of the group receives the sums. Must be called by every thread
// of the CTA the same number of times (it synchronizes the CTA when G > 32). Deterministic: butterfly inside a warp,
// then the warps of the group in ascending order.
template <int N>
__device__ __forceinline__ void row_sum(float (&v)[N], const RowGeo& g, float* red, int& phase) {
  if (g.G <= 32) {
#pragma unroll
    for (int k = 0; k < N; ++k)
      for (int o = g.G >> 1; o > 0; o >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], o, g.G);
    return;
  }
#pragma unroll
  for (int k = 0; k < N; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], o);
  const int W = g.G >> 5;
  float* buf = red + phase * (kRedSlots / 2);
  phase ^= 1;
  if ((threadIdx.x & 31) == 0) {
    const int w = g.lig >> 5;
#pragma unroll
    for (int k = 0; k < N; ++k) buf[(g.grp * W + w) * N + k] = v[k];
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < N; ++k) {
    float s = 0.f;
    for (int w = 0; w < W; ++w) s += buf[(g.grp * W + w) * N + k];
    v[k] = s;
  }
}

// Shared-memory gain accumulators: slot [grp * width + col0 + j] belongs to one thread (blockDim.x * 8 floats).
__device__ __forceinline__ void acc_zero(float* acc, const RowGeo& g, int width) {
  float4* p = reinterpret_cast<float4*>(acc + g.grp * width + g.col0);
  p[0] = make_float4(0.f, 0.f, 0.f, 0.f);
  p[1] = make_float4(0.f, 0.f, 0.f, 0.f);
}
__device__ __forceinline__ void acc_add(float* acc, const RowGeo& g, int width, const float (&v)[8]) {
  float4* p = reinterpret_cast<float4*>(acc + g.grp * width + g.col0);
  float4 a = p[0], b = p[1];
  a.x += v[0]; a.y += v[1]; a.z += v[2]; a.w += v[3];
  b.x += v[4]; b.y += v[5]; b.z += v[6]; b.w += v[7];
  p[0] = a;
  p[1] = b;
}
// Flush one part: partial_row[col] = sum over row groups (ascending) of acc[grp][col]; then zero the slots. Called by
// every thread of the CTA.
__device__ __forceinline__ void acc_flush(float* acc, const RowGeo& g, int width, float* partial_row) {
  __syncthreads();
  if (partial_row)
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
      float s = 0.f;
      for (int q = 0; q < g.groups; ++q) s += acc[q * width + c];
      partial_row[c] = s;
    }
  __syncthreads();
  acc_zero(acc, g, width);
}

// ------------------------------------------------------------------------------------------------ norm math
__device__ __forceinline__ float sum8_sq(const float (&x)[8]) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s = __fmaf_rn(x[j], x[j], s);   // a row sum: its order is free (class S)
  return s;
}
__device__ __forceinline__ float sum8_dot(const float (&x)[8], const float (&a)[8]) {
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s = __fmaf_rn(x[j], a[j], s);
  return s;
}

// Forward of one norm on a thread's 8 elements: y = round(x * r), out = g * y (a rounding point under levanter).
template <class S, bool LEV>
__device__ __forceinline__ R8<S, LEV> norm_out(const float (&xf)[8], float r, const R8<S, LEV>& g) {
  float t[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) t[j] = xf[j] * r;
  return R8<S, LEV>::mul(g, R8<S, LEV>::round(t));
}

// Backward of one norm after the row sums: returns the gain contributions gc = f32(round(dy * round(x * r))) and
// dx = a * r + c * (2 x) in f32 with c = (s * (-0.5 * (r / (ms + eps)))) / n (c * (2 x) is computed as (2 c) * x: the
// same exact product, one rounding).
template <class S, bool LEV>
__device__ __forceinline__ void norm_dx(const float (&xf)[8], const R8<S, LEV>& dy, const float (&af)[8], float r,
                                        float ms, float s, float eps, int n, float (&gc)[8], float (&dx)[8]) {
  const float c2 = 2.f * __fdiv_rn(s * (-0.5f * __fdiv_rn(r, ms + eps)), static_cast<float>(n));
  float t[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) t[j] = xf[j] * r;
  R8<S, LEV>::mul(dy, R8<S, LEV>::round(t)).f32(gc);
#pragma unroll
  for (int j = 0; j < 8; ++j) dx[j] = af[j] * r + c2 * xf[j];
}

// CTA size for row groups of G threads: a multiple of G, at least 256 threads when G <= 256.
inline int row_block(int G) { return G >= 256 ? G : G * (256 / G); }

inline void check_row_width(int width, const char* what) {
  NM_REQUIRE(width % 8 == 0 && width >= 8, std::string(what) + ": width must be a positive multiple of 8, got " +
                                               std::to_string(width));
  const int G = width / 8;
  NM_REQUIRE((G <= 32 && (G & (G - 1)) == 0) || (G % 32 == 0 && G <= kMaxRowG),
             std::string(what) + ": width / 8 must be a power of two <= 32 or a multiple of 32 <= 512, got width " +
                 std::to_string(width));
}

}  // namespace nm::ew
