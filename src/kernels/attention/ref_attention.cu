// Portable reference attention (RefF32 backend; the oracle the FA3 tests compare against). DESIGN.md §3.0, §3.4.
//
// Causal and same-segment softmax attention, f32 math, f32 running statistics. Forward and dQ: one warp per (query,
// head); keys of the query's segment up to the query, in blocks of 1024 keys in ascending order (block max first, then
// exp, sum and P V against the running statistics), 32 keys per step with lane j on key j (whole dot product per lane,
// q / dO staged in shared memory). dK / dV: one warp per (key, kv head); the G query heads of the group in ascending
// order, the queries of the segment from the key on in ascending order. No atomics: every output element is written
// once by one warp, so results are bitwise repeatable. Storage bf16 or f32 (template), accumulation f32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

#include "kernels/attention/attn_internal.hpp"

namespace nm::ref {

namespace {

constexpr int kWarps = 4;
constexpr int kKeyBlock = 1024;

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

struct Seg {
  int lo, hi;   // [lo, hi) token range of the segment, absolute
};

// Segment of token t: packed cu_seqlens (largest i with cu[i] <= t over slots + 1 entries), or the per-row table.
__device__ Seg seg_of(int t, const int* cu, int slots, const int* rss, int S, int max_segs) {
  if (cu != nullptr) {
    int a = 0, b = slots;
    while (a < b) {
      const int m = (a + b + 1) >> 1;
      if (cu[m] <= t) a = m; else b = m - 1;
    }
    return {cu[a], cu[a + 1]};
  }
  const int r = t / S, p = t - r * S;
  const int* rs = rss + int64_t(r) * (max_segs + 1);
  int a = 0, b = max_segs;
  while (a < b) {
    const int m = (a + b + 1) >> 1;
    if (rs[m] <= p) a = m; else b = m - 1;
  }
  int e = S;
  for (int i = a + 1; i <= max_segs; ++i) {
    if (rs[i] > p) { e = rs[i]; break; }
  }
  return {r * S + rs[a], r * S + e};
}

struct Geo {
  int64_t T;
  int H, KV, hd, S, slots, max_segs;
  const int* cu;
  const int* rss;
  float scale;
};

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Forward: warp per (query t, head h). o row and lse [H, T].
template <class T, int HD>
__global__ void __launch_bounds__(32 * kWarps) k_ref_fwd(Geo g, const T* __restrict__ q, int64_t q_ld,
                                                         const T* __restrict__ k, int64_t k_ld,
                                                         const T* __restrict__ v, int64_t v_ld, T* __restrict__ o,
                                                         int64_t o_ld, float* __restrict__ lse) {
  __shared__ float sq[kWarps][HD];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = int64_t(blockIdx.x) * kWarps + w;
  if (item >= g.T * g.H) return;
  const int t = int(item / g.H), h = int(item % g.H), hk = h / (g.H / g.KV);
  for (int d = lane; d < HD; d += 32) sq[w][d] = ld(q + int64_t(t) * q_ld + h * HD + d);
  __syncwarp();
  const Seg sg = seg_of(t, g.cu, g.slots, g.rss, g.S, g.max_segs);
  constexpr int DPL = HD / 32;   // dims per lane
  float acc[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) acc[i] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;
  const T* kh = k + hk * HD;
  const T* vh = v + hk * HD;
  for (int b0 = sg.lo; b0 <= t; b0 += kKeyBlock) {
    const int b1 = min(b0 + kKeyBlock, t + 1);
    float m_blk = -INFINITY;
    for (int j0 = b0; j0 < b1; j0 += 32) {
      const int j = j0 + lane;
      float s = -INFINITY;
      if (j < b1) {
        float d0 = 0.f;
        const T* kr = kh + int64_t(j) * k_ld;
#pragma unroll 8
        for (int d = 0; d < HD; ++d) d0 += sq[w][d] * ld(kr + d);
        s = d0 * g.scale;
      }
      m_blk = fmaxf(m_blk, warp_max(s));
    }
    const float m_new = fmaxf(m_run, m_blk);
    const float corr = m_run == -INFINITY ? 0.f : expf(m_run - m_new);
    l_run *= corr;
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] *= corr;
    for (int j0 = b0; j0 < b1; j0 += 32) {
      const int j = j0 + lane;
      float p = 0.f;
      if (j < b1) {
        float d0 = 0.f;
        const T* kr = kh + int64_t(j) * k_ld;
#pragma unroll 8
        for (int d = 0; d < HD; ++d) d0 += sq[w][d] * ld(kr + d);
        p = expf(d0 * g.scale - m_new);
      }
      l_run += warp_sum(p);
      const int n = min(32, b1 - j0);
      for (int jj = 0; jj < n; ++jj) {
        const float pj = __shfl_sync(0xffffffffu, p, jj);
        const T* vr = vh + int64_t(j0 + jj) * v_ld;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] += pj * ld(vr + lane + 32 * i);
      }
    }
    m_run = m_new;
  }
  const float inv = 1.f / l_run;
#pragma unroll
  for (int i = 0; i < DPL; ++i) st(o + int64_t(t) * o_ld + h * HD + lane + 32 * i, acc[i] * inv);
  if (lane == 0) lse[int64_t(h) * g.T + t] = m_run + logf(l_run);
}

// D[h, t] = rowsum(dO * O) from the stored O.
template <class T, int HD>
__global__ void k_ref_dsum(Geo g, const T* __restrict__ o, int64_t o_ld, const T* __restrict__ dO, int64_t do_ld,
                           float* __restrict__ D) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = int64_t(blockIdx.x) * kWarps + w;
  if (item >= g.T * g.H) return;
  const int t = int(item / g.H), h = int(item % g.H);
  float s = 0.f;
  for (int d = lane; d < HD; d += 32) s += ld(o + int64_t(t) * o_ld + h * HD + d) * ld(dO + int64_t(t) * do_ld + h * HD + d);
  s = warp_sum(s);
  if (lane == 0) D[int64_t(h) * g.T + t] = s;
}

// dQ: warp per (query, head): dq = scale * sum_j P_j (dP_j - D) k_j, keys ascending.
template <class T, int HD>
__global__ void __launch_bounds__(32 * kWarps) k_ref_dq(Geo g, const T* __restrict__ q, int64_t q_ld,
                                                        const T* __restrict__ k, int64_t k_ld,
                                                        const T* __restrict__ v, int64_t v_ld,
                                                        const T* __restrict__ dO, int64_t do_ld,
                                                        const float* __restrict__ lse, const float* __restrict__ D,
                                                        T* __restrict__ dq, int64_t dq_ld) {
  __shared__ float sq[kWarps][HD];
  __shared__ float sdo[kWarps][HD];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = int64_t(blockIdx.x) * kWarps + w;
  if (item >= g.T * g.H) return;
  const int t = int(item / g.H), h = int(item % g.H), hk = h / (g.H / g.KV);
  for (int d = lane; d < HD; d += 32) {
    sq[w][d] = ld(q + int64_t(t) * q_ld + h * HD + d);
    sdo[w][d] = ld(dO + int64_t(t) * do_ld + h * HD + d);
  }
  __syncwarp();
  const Seg sg = seg_of(t, g.cu, g.slots, g.rss, g.S, g.max_segs);
  const float L = lse[int64_t(h) * g.T + t], Dt = D[int64_t(h) * g.T + t];
  constexpr int DPL = HD / 32;
  float acc[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) acc[i] = 0.f;
  const T* kh = k + hk * HD;
  const T* vh = v + hk * HD;
  for (int j0 = sg.lo; j0 <= t; j0 += 32) {
    const int j = j0 + lane;
    float ds = 0.f;
    if (j <= t) {
      float s = 0.f, dp = 0.f;
      const T* kr = kh + int64_t(j) * k_ld;
      const T* vr = vh + int64_t(j) * v_ld;
#pragma unroll 8
      for (int d = 0; d < HD; ++d) {
        s += sq[w][d] * ld(kr + d);
        dp += sdo[w][d] * ld(vr + d);
      }
      const float p = expf(s * g.scale - L);
      ds = p * (dp - Dt);
    }
    const int n = min(32, t + 1 - j0);
    for (int jj = 0; jj < n; ++jj) {
      const float dsj = __shfl_sync(0xffffffffu, ds, jj);
      const T* kr = kh + int64_t(j0 + jj) * k_ld;
#pragma unroll
      for (int i = 0; i < DPL; ++i) acc[i] += dsj * ld(kr + lane + 32 * i);
    }
  }
#pragma unroll
  for (int i = 0; i < DPL; ++i) st(dq + int64_t(t) * dq_ld + h * HD + lane + 32 * i, acc[i] * g.scale);
}

// dK, dV: warp per (key, kv head); heads of the group ascending, queries of the segment from the key on ascending.
template <class T, int HD>
__global__ void __launch_bounds__(32 * kWarps) k_ref_dkv(Geo g, const T* __restrict__ q, int64_t q_ld,
                                                         const T* __restrict__ k, int64_t k_ld,
                                                         const T* __restrict__ v, int64_t v_ld,
                                                         const T* __restrict__ dO, int64_t do_ld,
                                                         const float* __restrict__ lse, const float* __restrict__ D,
                                                         T* __restrict__ dk, int64_t dk_ld, T* __restrict__ dv,
                                                         int64_t dv_ld) {
  __shared__ float sk[kWarps][HD];
  __shared__ float sv[kWarps][HD];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = int64_t(blockIdx.x) * kWarps + w;
  if (item >= g.T * g.KV) return;
  const int c = int(item / g.KV), hk = int(item % g.KV), G = g.H / g.KV;
  for (int d = lane; d < HD; d += 32) {
    sk[w][d] = ld(k + int64_t(c) * k_ld + hk * HD + d);
    sv[w][d] = ld(v + int64_t(c) * v_ld + hk * HD + d);
  }
  __syncwarp();
  const Seg sg = seg_of(c, g.cu, g.slots, g.rss, g.S, g.max_segs);
  constexpr int DPL = HD / 32;
  float ak[DPL], av[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) ak[i] = av[i] = 0.f;
  for (int gi = 0; gi < G; ++gi) {
    const int h = hk * G + gi;
    for (int i0 = c; i0 < sg.hi; i0 += 32) {
      const int r = i0 + lane;
      float p = 0.f, ds = 0.f;
      if (r < sg.hi) {
        float s = 0.f, dp = 0.f;
        const T* qr = q + int64_t(r) * q_ld + h * HD;
        const T* dr = dO + int64_t(r) * do_ld + h * HD;
#pragma unroll 8
        for (int d = 0; d < HD; ++d) {
          s += sk[w][d] * ld(qr + d);
          dp += sv[w][d] * ld(dr + d);
        }
        p = expf(s * g.scale - lse[int64_t(h) * g.T + r]);
        ds = p * (dp - D[int64_t(h) * g.T + r]);
      }
      const int n = min(32, sg.hi - i0);
      for (int jj = 0; jj < n; ++jj) {
        const float pj = __shfl_sync(0xffffffffu, p, jj);
        const float dsj = __shfl_sync(0xffffffffu, ds, jj);
        const T* qr = q + int64_t(i0 + jj) * q_ld + h * HD;
        const T* dr = dO + int64_t(i0 + jj) * do_ld + h * HD;
#pragma unroll
        for (int i = 0; i < DPL; ++i) {
          ak[i] += dsj * ld(qr + lane + 32 * i);
          av[i] += pj * ld(dr + lane + 32 * i);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < DPL; ++i) {
    st(dk + int64_t(c) * dk_ld + hk * HD + lane + 32 * i, ak[i] * g.scale);
    st(dv + int64_t(c) * dv_ld + hk * HD + lane + 32 * i, av[i]);
  }
}

Geo geo_of(const AttnShape& s, const AttnFwdArgs& a) {
  NM_REQUIRE(s.hd == 64 || s.hd == 128, "reference attention: hd 64 or 128");
  NM_REQUIRE(s.KV > 0 && s.H % s.KV == 0, "reference attention: H must be a multiple of KV");
  NM_REQUIRE(s.total_q == int64_t(s.rows) * s.S && s.total_q < (int64_t(1) << 31), "reference attention: total_q");
  NM_REQUIRE(a.cu_seqlens != nullptr || a.row_seg_start != nullptr, "reference attention: cu_seqlens or row_seg_start");
  NM_REQUIRE(s.io == DType::bf16 || s.io == DType::f32, "reference attention: bf16 or f32 storage");
  return Geo{s.total_q, s.H, s.KV, s.hd, s.S, s.rows * s.max_segs_per_seq, s.max_segs_per_seq,
             a.cu_seqlens, a.cu_seqlens ? nullptr : a.row_seg_start, a.softmax_scale};
}

unsigned blocks_for(int64_t items) { return unsigned((items + kWarps - 1) / kWarps); }

template <class T, int HD>
void fwd_t(const Geo& g, const AttnFwdArgs& a, cudaStream_t st) {
  k_ref_fwd<T, HD><<<blocks_for(g.T * g.H), 32 * kWarps, 0, st>>>(
      g, static_cast<const T*>(a.q), a.q_ld, static_cast<const T*>(a.k), a.k_ld, static_cast<const T*>(a.v), a.v_ld,
      static_cast<T*>(a.o), a.o_ld, a.lse);
  NM_CUDA(cudaGetLastError());
}

template <class T, int HD>
void bwd_t(const Geo& g, const AttnBwdArgs& a, float* D, cudaStream_t st) {
  const AttnFwdArgs& f = a.fwd;
  const T* q = static_cast<const T*>(f.q);
  const T* k = static_cast<const T*>(f.k);
  const T* v = static_cast<const T*>(f.v);
  const T* dO = static_cast<const T*>(a.d_o);
  if (a.softmax_d == nullptr) {
    k_ref_dsum<T, HD><<<blocks_for(g.T * g.H), 32 * kWarps, 0, st>>>(g, static_cast<const T*>(f.o), f.o_ld, dO, a.do_ld, D);
    NM_CUDA(cudaGetLastError());
  }
  const float* Dp = a.softmax_d ? a.softmax_d : D;
  k_ref_dq<T, HD><<<blocks_for(g.T * g.H), 32 * kWarps, 0, st>>>(g, q, f.q_ld, k, f.k_ld, v, f.v_ld, dO, a.do_ld, f.lse,
                                                                  Dp, static_cast<T*>(a.dq), a.dq_ld);
  NM_CUDA(cudaGetLastError());
  k_ref_dkv<T, HD><<<blocks_for(g.T * g.KV), 32 * kWarps, 0, st>>>(g, q, f.q_ld, k, f.k_ld, v, f.v_ld, dO, a.do_ld,
                                                                    f.lse, Dp, static_cast<T*>(a.dk), a.dk_ld,
                                                                    static_cast<T*>(a.dv), a.dv_ld);
  NM_CUDA(cudaGetLastError());
}

}  // namespace

uint64_t workspace_bytes(const AttnShape& s, bool training) {
  return training ? uint64_t(s.H) * uint64_t(s.total_q) * 4 : 0;
}

void fwd(const AttnShape& s, const AttnFwdArgs& a, cudaStream_t st) {
  const Geo g = geo_of(s, a);
  NM_REQUIRE(a.q && a.k && a.v && a.o && a.lse, "reference attention fwd: q, k, v, o, lse required");
  NM_REQUIRE(a.extra_k == nullptr, "reference attention: extra_kv (moda) is not built");
  if (s.io == DType::bf16) {
    if (s.hd == 64) fwd_t<__nv_bfloat16, 64>(g, a, st); else fwd_t<__nv_bfloat16, 128>(g, a, st);
  } else {
    if (s.hd == 64) fwd_t<float, 64>(g, a, st); else fwd_t<float, 128>(g, a, st);
  }
}

void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st) {
  const Geo g = geo_of(s, a.fwd);
  NM_REQUIRE(a.dq_accum == nullptr && a.dq && a.dk && a.dv && a.d_o, "reference attention bwd: dq, dk, dv, dO required");
  NM_REQUIRE(a.softmax_d != nullptr || ws != nullptr, "reference attention bwd: workspace for softmax_d");
  float* D = static_cast<float*>(ws);
  if (s.io == DType::bf16) {
    if (s.hd == 64) bwd_t<__nv_bfloat16, 64>(g, a, D, st); else bwd_t<__nv_bfloat16, 128>(g, a, D, st);
  } else {
    if (s.hd == 64) bwd_t<float, 64>(g, a, D, st); else bwd_t<float, 128>(g, a, D, st);
  }
}

}  // namespace nm::ref
