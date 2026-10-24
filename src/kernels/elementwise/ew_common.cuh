// Device helpers shared by the elementwise kernels (stage S3). Not a public header.
//
// Storage: every kernel is a template over the storage type S (bf16 for the bf16 policy, float for the f32 policy) and
// the rounding mode LEV (true = Rounding::Levanter). Each thread owns one "vector" of 8 consecutive elements (16 B of
// bf16 or 32 B of f32), so the same index math serves both storages.
//
// Rounding: rnd<S, LEV>(x) is a levanter rounding point: under Levanter it rounds x to the storage precision (bf16 RNE
// for bf16 storage; identity for f32 storage, where levanter's compute dtype is f32), under Accurate it is the identity.
// st<S>(x) is the one rounding at a kernel output (store). Contraction (DESIGN.md §3.0): the units that instantiate the
// levanter-mode kernels build with --fmad=false, so `a * b + c` is two roundings there and a contraction is written as
// __fmaf_rn where wanted; the accurate-mode instantiations (*_acc.cu) keep nvcc's default contraction.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "marincpp/core.hpp"

namespace nm::ew {

__device__ __forceinline__ float bf2f(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }   // round to bf16 precision (RNE)

// ------------------------------------------------------------------------------------------------ storage traits
template <class S> struct St;

template <> struct St<bf16_t> {
  static constexpr int kBytes = 2;
  __device__ __forceinline__ static void load8(const bf16_t* p, float v[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[2 * i] = __uint_as_float(w[i] << 16);
      v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
  // Plain (coherent) load: for buffers this kernel or an earlier kernel in the same graph may have written.
  __device__ __forceinline__ static void load8_cg(const bf16_t* p, float v[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[2 * i] = __uint_as_float(w[i] << 16);
      v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
  __device__ __forceinline__ static void store8(bf16_t* p, const float v[8]) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = f2bf(v[2 * i]) | (f2bf(v[2 * i + 1]) << 16);
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  __device__ __forceinline__ static float round(float x) { return rbf(x); }
};

template <> struct St<float> {
  static constexpr int kBytes = 4;
  __device__ __forceinline__ static void load8(const float* p, float v[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  }
  __device__ __forceinline__ static void load8_cg(const float* p, float v[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  }
  __device__ __forceinline__ static void store8(float* p, const float v[8]) {
    reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
  }
  __device__ __forceinline__ static float round(float x) { return x; }
};

// ------------------------------------------------------------------------------------------------ packed bf16 (levanter)
// V8: 8 bf16 values as 4 bf16x2 words, element 2i in the low half of w[i] (memory order). Used by the bf16-storage
// levanter instantiations, where most ops take two bf16 operands and round to bf16:
//   * the product of two bf16 values is exact in f32 (16 significant bits; the f32 normal range), so __hmul2_rn (one
//     RNE rounding of the exact product) equals bf16(f32(a) * f32(b));
//   * the sum of two bf16 values is exact in f32 when their exponents differ by <= 15; otherwise the smaller one is
//     below 2^-15 of the larger and both roundings return the larger: __hadd2_rn equals bf16(f32(a) + f32(b)).
//   (A product below 2^-126 is an f32 subnormal: the two roundings could then differ at a tie. No such case was seen.)
// So the packed ops are bit-identical to the f32-then-round chain (the S3 tests compare them on every element) at a
// quarter of the instructions. The _rn forms forbid fma contraction.
struct V8 {
  uint32_t w[4];
};
__device__ __forceinline__ V8 ld8(const bf16_t* p) {
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
  return V8{{u.x, u.y, u.z, u.w}};
}
__device__ __forceinline__ V8 ld8_cg(const bf16_t* p) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  return V8{{u.x, u.y, u.z, u.w}};
}
__device__ __forceinline__ void st8(bf16_t* p, const V8& v) {
  *reinterpret_cast<uint4*>(p) = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
}
__device__ __forceinline__ __nv_bfloat162 b2(uint32_t u) { return *reinterpret_cast<__nv_bfloat162*>(&u); }
__device__ __forceinline__ uint32_t u2(__nv_bfloat162 b) { return *reinterpret_cast<uint32_t*>(&b); }
__device__ __forceinline__ V8 vmul(const V8& a, const V8& b) {
  V8 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r.w[i] = u2(__hmul2_rn(b2(a.w[i]), b2(b.w[i])));
  return r;
}
__device__ __forceinline__ V8 vadd(const V8& a, const V8& b) {
  V8 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r.w[i] = u2(__hadd2_rn(b2(a.w[i]), b2(b.w[i])));
  return r;
}
__device__ __forceinline__ V8 vneg_if(const V8& a, bool neg) {   // exact sign flip
  const uint32_t m = neg ? 0x80008000u : 0u;
  return V8{{a.w[0] ^ m, a.w[1] ^ m, a.w[2] ^ m, a.w[3] ^ m}};
}
__device__ __forceinline__ V8 vsplat(float x) {   // x must be a bf16 value
  const uint32_t h = f2bf(x);
  const uint32_t w = h | (h << 16);
  return V8{{w, w, w, w}};
}
__device__ __forceinline__ void unpack(const V8& v, float (&f)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(v.w[i] << 16);
    f[2 * i + 1] = __uint_as_float(v.w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ V8 pack_rn(const float (&f)[8]) {
  V8 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r.w[i] = u2(__floats2bfloat162_rn(f[2 * i], f[2 * i + 1]));
  return r;
}
template <class S, bool LEV> inline constexpr bool kPacked = false;
template <> inline constexpr bool kPacked<bf16_t, true> = true;

template <class S, bool LEV> __device__ __forceinline__ float rnd(float x);

// R8<S, LEV>: one thread's 8 elements in the form its instantiation computes on. Packed (bf16 storage, levanter):
// V8 with bf16x2 ops. Otherwise float[8], where mul / add / round are levanter rounding points (rnd: identity for f32
// storage and for accurate mode) and store rounds once to the storage type. Kernels written against R8 therefore
// compute exactly the formulas of their header comment in every instantiation.
template <class S, bool LEV, bool PK = kPacked<S, LEV>> struct R8;
template <class S, bool LEV> struct R8<S, LEV, true> {
  V8 v;
  __device__ __forceinline__ static R8 load(const S* p) { return R8{ld8(p)}; }
  __device__ __forceinline__ static R8 load_cg(const S* p) { return R8{ld8_cg(p)}; }
  __device__ __forceinline__ static R8 load_z(const S* p, bool on) { return on ? load(p) : R8{V8{{0, 0, 0, 0}}}; }
  __device__ __forceinline__ static R8 load_z_cg(const S* p, bool on) {
    return on ? load_cg(p) : R8{V8{{0, 0, 0, 0}}};
  }
  __device__ __forceinline__ static R8 zero() { return R8{V8{{0, 0, 0, 0}}}; }
  __device__ __forceinline__ void f32(float (&f)[8]) const { unpack(v, f); }
  __device__ __forceinline__ static R8 round(const float (&f)[8]) { return R8{pack_rn(f)}; }
  __device__ __forceinline__ static R8 mul(const R8& a, const R8& b) { return R8{vmul(a.v, b.v)}; }
  __device__ __forceinline__ static R8 add(const R8& a, const R8& b) { return R8{vadd(a.v, b.v)}; }
  __device__ __forceinline__ void store(S* p) const { st8(p, v); }
  __device__ __forceinline__ R8 shfl_xor(int mask, int width) const {
    R8 r;
#pragma unroll
    for (int i = 0; i < 4; ++i) r.v.w[i] = __shfl_xor_sync(0xffffffffu, v.w[i], mask, width);
    return r;
  }
  __device__ __forceinline__ R8 neg_if(bool n) const { return R8{vneg_if(v, n)}; }
  __device__ __forceinline__ static R8 splat(float x) { return R8{vsplat(x)}; }
  __device__ __forceinline__ static R8 select(bool c, const R8& a, const R8& b) {   // by value: no local memory
    R8 r;
#pragma unroll
    for (int i = 0; i < 4; ++i) r.v.w[i] = c ? a.v.w[i] : b.v.w[i];
    return r;
  }
};
template <class S, bool LEV> struct R8<S, LEV, false> {
  float x[8];
  __device__ __forceinline__ static R8 load(const S* p) {
    R8 r;
    St<S>::load8(p, r.x);
    return r;
  }
  __device__ __forceinline__ static R8 load_cg(const S* p) {
    R8 r;
    St<S>::load8_cg(p, r.x);
    return r;
  }
  __device__ __forceinline__ static R8 zero() {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = 0.f;
    return r;
  }
  __device__ __forceinline__ static R8 load_z(const S* p, bool on) { return on ? load(p) : zero(); }
  __device__ __forceinline__ static R8 load_z_cg(const S* p, bool on) { return on ? load_cg(p) : zero(); }
  __device__ __forceinline__ void f32(float (&f)[8]) const {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = x[j];
  }
  __device__ __forceinline__ static R8 round(const float (&f)[8]) {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = rnd<S, LEV>(f[j]);
    return r;
  }
  __device__ __forceinline__ static R8 mul(const R8& a, const R8& b) {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = rnd<S, LEV>(a.x[j] * b.x[j]);
    return r;
  }
  __device__ __forceinline__ static R8 add(const R8& a, const R8& b) {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = rnd<S, LEV>(a.x[j] + b.x[j]);
    return r;
  }
  __device__ __forceinline__ void store(S* p) const { St<S>::store8(p, x); }
  __device__ __forceinline__ R8 shfl_xor(int mask, int width) const {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = __shfl_xor_sync(0xffffffffu, x[j], mask, width);
    return r;
  }
  __device__ __forceinline__ R8 neg_if(bool n) const {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = n ? -x[j] : x[j];
    return r;
  }
  __device__ __forceinline__ static R8 splat(float v) {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = v;
    return r;
  }
  __device__ __forceinline__ static R8 select(bool c, const R8& a, const R8& b) {
    R8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.x[j] = c ? a.x[j] : b.x[j];
    return r;
  }
};

template <class S, bool LEV> __device__ __forceinline__ float rnd(float x) { return LEV ? St<S>::round(x) : x; }
template <class S> __device__ __forceinline__ float st(float x) { return St<S>::round(x); }

__device__ __forceinline__ void load8_f32(const float* p, float v[8]) { St<float>::load8(p, v); }

// ------------------------------------------------------------------------------------------------ group reductions
// Sum over the G threads of one row group (G a power of two <= 32, aligned inside a warp): fixed butterfly order, so
// every thread of the group gets the same, deterministic value.
template <int G> __device__ __forceinline__ float group_sum_small(float v) {
#pragma unroll
  for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, G);
  return v;
}
template <int G> __device__ __forceinline__ float group_shfl_xor(float v, int mask) {
  return __shfl_xor_sync(0xffffffffu, v, mask, G);
}

// ------------------------------------------------------------------------------------------------ launch helpers
// Partition of the T rows of a norm kernel into P parts of consecutive rows: part p = rows [p T / P, (p + 1) T / P).
// P = min(1024, ceil(T / 16), kNormCtasPerSm x SMs): one CTA per part, and every SM holds kNormCtasPerSm of them at
// once (the kernels' __launch_bounds__ guarantee that occupancy), so no SM gets one part more than another. The
// per-part gain partials therefore depend on T and on the device's SM count (fixed for a device type), never on
// timing: a step is bitwise repeatable on a given GPU type (DESIGN.md §11).
inline constexpr int kMaxParts = 1024;        // the planner reserves [1024, width] f32 per gain-partial slot
inline constexpr int kMinRowsPerPart = 16;
inline constexpr int kNormCtasPerSm = 4;      // 256-thread CTAs, <= 64 registers per thread
int64_t norm_parts(int64_t T);

// Grid for a grid-stride loop over n work items of `per_cta` items each: at most the resident capacity.
int grid_for_items(int64_t n, int64_t per_cta, const void* kernel, int block, size_t smem);

// Dispatch a launcher template fn<S, LEV>(a, s) on the storage type and the rounding mode.
#define NM_EW_DISPATCH(fn, st, r, a, s)                                                   \
  do {                                                                                    \
    const bool lev_ = (r) == Rounding::Levanter;                                          \
    if ((st) == DType::bf16) {                                                            \
      if (lev_) fn<bf16_t, true>(a, s); else fn<bf16_t, false>(a, s);                     \
    } else if ((st) == DType::f32) {                                                      \
      if (lev_) fn<float, true>(a, s); else fn<float, false>(a, s);                       \
    } else {                                                                              \
      fatal(NM_WHERE, "elementwise kernels take bf16 or f32 storage");                    \
    }                                                                                     \
  } while (0)

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace nm::ew
