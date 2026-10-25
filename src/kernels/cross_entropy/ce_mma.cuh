// Portable tensor-core building blocks of the CE kernels (sm_80 and sm_90a): cp.async with zero fill, ldmatrix,
// mma.sync m16n8k16 bf16 -> f32, and the XOR swizzles of the shared-memory tiles.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nm::ce {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte global -> shared copy; pred == false fills the 16 bytes with zeros (src is not read).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// D += A B: A 16x16 row-major fragment (4 regs), B 16x8 column fragment (2 regs), f32 accumulators (4).
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Swizzled 16-byte chunk index of a tile whose rows hold CPR chunks: conflict-free ldmatrix over 8 consecutive rows.
template <int CPR>
__device__ __forceinline__ int swz(int row, int chunk) {
  static_assert(CPR == 4 || CPR >= 8, "rows of 64 B or of at least 128 B");
  if constexpr (CPR == 4) return chunk ^ ((row >> 1) & 3);
  else return chunk ^ (row & 7);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const bf16 l = __float2bfloat16_rn(lo), h = __float2bfloat16_rn(hi);
  return uint32_t(__bfloat16_as_ushort(l)) | (uint32_t(__bfloat16_as_ushort(h)) << 16);
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// exp(d) of a statistics term d = z - max <= 0. Accurate (default): libdevice expf, what XLA:GPU emits. Fast
// (Backends::fast_stats_exp, off until the A/B numerics.md §1 asks for fast-math exponentials): ex2.approx.ftz of the
// rounded d * log2(e), 3 instructions instead of 9; per term ~3 ulp at |d| < 1 (the terms that carry the sum), more for
// terms below e^-20 that do not move it, results below 2^-126 flushed to 0 (the sum is >= 1).
template <bool FAST>
__device__ __forceinline__ float exp_stat(float d) {
  if constexpr (FAST) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(__fmul_rn(d, 1.4426950408889634f)));
    return r;
  } else {
    return expf(d);
  }
}

// The CE gradient of one logit (numerics.md §5.2, levanter's _backward_b_tiled_from_lse): p = exp(z_s - lse) in f32,
// dz = p * g, and at the target dz + (-g). Explicit roundings: no contraction of p * g - g into one FMA.
__device__ __forceinline__ float ce_dz(float zs, float lse, float g, bool is_target) {
  const float p = expf(__fsub_rn(zs, lse));
  const float dz = __fmul_rn(p, g);
  return is_target ? __fsub_rn(dz, g) : dz;
}
// ce_dz where `ok`, else 0, without a branch: the exponential is always evaluated (on garbage when !ok) and selected
// away. A conditional expf compiles to a branch per element, which serializes the exponentials of a thread.
__device__ __forceinline__ float ce_dz_sel(float zs, float lse, float g, bool is_target, bool ok) {
  const float d = ce_dz(zs, lse, g, is_target);
  return ok ? d : 0.f;
}

}  // namespace nm::ce
