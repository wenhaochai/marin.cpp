// QK-norm + RoPE (+ q scale), forward and backward, and the dq-postprocess fusion. DESIGN.md §3.3-§3.4;
// numerics.md §3.2-§3.4.
//
// Per (token, head) row of hd elements: RMSNorm over hd with the layer's shared gain (q_norm or k_norm), RoPE with the
// NeoX half split at absolute position pos[t] (rotate_half(x) = [-x2, x1]), then q *= q_scale.
// Levanter (every op of levanter's jaxpr rounds to the compute dtype; numerics.md §3.3: "each multiply and each add
// rounds to bf16"):
//   n = bf16(g * bf16(x * r));  o = bf16(bf16(n * cos) + bf16(rot(n) * sin));  q: o = bf16(o * q_scale)
// backward (transpose of the same ops; rot^T(u) = [u2, -u1]):
//   q: d = bf16(dq_rot * q_scale);  dn = bf16(bf16(d * cos) + rot^T(bf16(d * sin)));  then the RMSNorm backward of
//   rmsnorm.cu over hd with dg accumulated over tokens and heads.
// Accurate: the same chain in f32 with one rounding per stored output.
// A row group of hd / 8 threads (8 or 16 lanes, inside one warp) owns one (token, head) row; the rotation partner of a
// thread's 8 elements is the thread hd / 16 lanes away (lane ^ (G / 2)).
//
// Kernels and launchers (templates over storage S and rounding LEV). The levanter instantiations (LEV = true) are built
// in qknorm_rope.cu with --fmad=false, the accurate ones (LEV = false) in qknorm_rope_acc.cu with nvcc's default
// contraction (DESIGN.md §3.0); NM_EW_QK_LAUNCHERS lists the launchers both units instantiate.
#pragma once

#include <string>
#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_rows.cuh"

namespace nm::ew::qk {

template <int G> __device__ __forceinline__ float gsum(float v) {
#pragma unroll
  for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, G);
  return v;
}

struct QkGeo {
  int64_t T;
  int H, KV, hd;
  int h_begin, h_end;   // combined head index: [0, H) q heads, [H, H + KV) k heads
};

// Locate item i of part [t0, t1): token and combined head.
__device__ __forceinline__ bool qk_item(const QkGeo& q, int64_t t0, int64_t t1, int64_t i, int64_t& t, int& h) {
  const int nh = q.h_end - q.h_begin;
  t = t0 + i / nh;
  h = q.h_begin + static_cast<int>(i % nh);
  return t < t1;
}

// RoPE of one thread's 8 elements of a normed row: o = n * cos + rot(n) * sin (each op a rounding point under
// levanter); rot(n) takes the partner thread's elements (lane ^ G/2), negated in the first half.
template <class S, bool LEV, int G>
__device__ __forceinline__ R8<S, LEV> rope8(const R8<S, LEV>& n, const R8<S, LEV>& c, const R8<S, LEV>& s,
                                            bool first_half) {
  using V = R8<S, LEV>;
  const V rot = n.shfl_xor(G / 2, G).neg_if(first_half);
  return V::add(V::mul(n, c), V::mul(rot, s));
}

// ------------------------------------------------------------------------------------------------ forward
// NORM = false: RoPE (+ scale) only, on rows that are already normed (rope_fwd_dt; T1.3 tests); gains and rstd unused.
template <class S, bool LEV, int G, bool NORM>
__global__ void __launch_bounds__(256, 4) k_qknorm_rope_fwd(const S* __restrict__ qkv, int64_t qkv_ld,
                                                            const S* __restrict__ gq, const S* __restrict__ gk,
                                                            const S* __restrict__ cos_tab,
                                                            const S* __restrict__ sin_tab,
                                                            const int32_t* __restrict__ pos, S* __restrict__ q_rot,
                                                            S* __restrict__ k_rot, float* __restrict__ rstd_q,
                                                            float* __restrict__ rstd_k, QkGeo q, float eps,
                                                            float q_scale, int64_t parts, int64_t rows_per_part) {
  using V = R8<S, LEV>;
  constexpr int hd = 8 * G;
  const int groups = blockDim.x / G, grp = threadIdx.x / G, lig = threadIdx.x % G, col0 = lig * 8;
  const bool first_half = lig < G / 2;
  const V vgq = NORM ? V::load(gq + col0) : V::zero();
  const V vgk = NORM ? V::load(gk + col0) : V::zero();
  const V vsc = V::splat(q_scale);
  const int nh = q.h_end - q.h_begin;
  const int64_t iters = (rows_per_part * nh + groups - 1) / groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t t0 = p * q.T / parts, t1 = (p + 1) * q.T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      int64_t t;
      int h;
      const bool on = qk_item(q, t0, t1, it * groups + grp, t, h);
      const bool isq = h < q.H;
      const int64_t tt = on ? t : t0;   // a valid token for inactive groups (they compute and discard)
      const V x = V::load(qkv + tt * qkv_ld + int64_t(h) * hd + col0);   // q | k column blocks are contiguous
      const int64_t prow = int64_t(__ldg(pos + tt)) * hd + col0;
      const V c = V::load(cos_tab + prow), sn = V::load(sin_tab + prow);
      float r = 1.f;
      V n = x;
      if (NORM) {
        float xf[8];
        x.f32(xf);
        r = rsqrtf(__fdiv_rn(gsum<G>(sum8_sq(xf)), static_cast<float>(hd)) + eps);
        n = norm_out<S, LEV>(xf, r, V::select(isq, vgq, vgk));
      }
      V o = rope8<S, LEV, G>(n, c, sn, first_half);
      if (isq) o = V::mul(o, vsc);
      if (on) {
        if (isq) {
          if (q_rot) o.store(q_rot + (t * q.H + h) * hd + col0);
          if (NORM && rstd_q && lig == 0) rstd_q[t * q.H + h] = r;
        } else {
          const int kh = h - q.H;
          if (k_rot) o.store(k_rot + (t * q.KV + kh) * hd + col0);
          if (NORM && rstd_k && lig == 0) rstd_k[t * q.KV + kh] = r;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ backward
// FROM_ACCUM: the q gradient comes from FA3's f32 accumulator [H, T, hd] (dq_post_qknorm_rope_bwd), else from
// dq_rot [T, H*hd] / dk_rot [T, KV*hd].
template <class S, bool LEV, int G, bool FROM_ACCUM>
__global__ void __launch_bounds__(256, 4) k_qknorm_rope_bwd(const S* __restrict__ dq_rot,
                                                            const S* __restrict__ dk_rot,
                                                            const float* __restrict__ dq_accum,
                                                            const S* __restrict__ qkv, int64_t qkv_ld,
                                                            const S* __restrict__ gq, const S* __restrict__ gk,
                                                            const S* __restrict__ cos_tab,
                                                            const S* __restrict__ sin_tab,
                                                            const int32_t* __restrict__ pos,
                                                            const float* __restrict__ rstd_q,
                                                            const float* __restrict__ rstd_k, S* __restrict__ dqkv,
                                                            int64_t dqkv_ld, float* __restrict__ pgq,
                                                            float* __restrict__ pgk, QkGeo q, float eps, float q_scale,
                                                            int64_t parts, int64_t rows_per_part) {
  using V = R8<S, LEV>;
  constexpr int hd = 8 * G;
  __shared__ __align__(16) float accq[256 * 8];
  __shared__ __align__(16) float acck[256 * 8];
  const RowGeo geo(hd);
  const int groups = geo.groups, grp = geo.grp, lig = geo.lig, col0 = geo.col0;
  const bool first_half = lig < G / 2;
  acc_zero(accq, geo, hd);
  acc_zero(acck, geo, hd);
  const V vgq = V::load(gq + col0), vgk = V::load(gk + col0);
  const V vsc = V::splat(q_scale);
  const int nh = q.h_end - q.h_begin;
  const int64_t iters = (rows_per_part * nh + groups - 1) / groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t t0 = p * q.T / parts, t1 = (p + 1) * q.T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      int64_t t;
      int h;
      const bool on = qk_item(q, t0, t1, it * groups + grp, t, h);
      const bool isq = h < q.H;
      const int64_t tt = on ? t : t0;
      V d;
      if (isq) {
        if (FROM_ACCUM) {
          float af[8];
          load8_f32(dq_accum + (int64_t(h) * q.T + tt) * hd + col0, af);
          if (LEV) {
            d = V::mul(V::round(af), vsc);   // FA3 postprocess bf16(acc) at softmax_scale 1, then bf16(dq * scale)
          } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) af[j] *= q_scale;
            d = V::round(af);
          }
        } else {
          d = V::mul(V::load(dq_rot + (tt * q.H + h) * hd + col0), vsc);
        }
      } else {
        d = V::load(dk_rot + (tt * q.KV + (h - q.H)) * hd + col0);
      }
      const int64_t prow = int64_t(__ldg(pos + tt)) * hd + col0;
      const V c = V::load(cos_tab + prow), sn = V::load(sin_tab + prow);
      // dn = d * cos + rot^T(d * sin), rot^T(u) = [u2, -u1]
      const V u = V::mul(d, sn);
      const V dn = V::add(V::mul(d, c), u.shfl_xor(G / 2, G).neg_if(!first_half));
      // RMSNorm backward over hd on the raw q / k columns of qkv.
      const V x = V::load(qkv + tt * qkv_ld + int64_t(h) * hd + col0);
      float xf[8], af[8];
      x.f32(xf);
      V::mul(V::select(isq, vgq, vgk), dn).f32(af);
      const float ss = gsum<G>(sum8_sq(xf));
      const float sd = gsum<G>(sum8_dot(xf, af));
      const float ms = __fdiv_rn(ss, static_cast<float>(hd));
      const float* rs = isq ? rstd_q : rstd_k;
      const int64_t ridx = isq ? tt * q.H + h : tt * q.KV + (h - q.H);
      const float r = rs ? __ldg(rs + ridx) : rsqrtf(ms + eps);
      float gc[8], dx[8];
      norm_dx<S, LEV>(xf, dn, af, r, ms, sd, eps, hd, gc, dx);
      if (on) {
        acc_add(isq ? accq : acck, geo, hd, gc);
        V::round(dx).store(dqkv + t * dqkv_ld + int64_t(h) * hd + col0);
      }
    }
    acc_flush(accq, geo, hd, pgq ? pgq + p * hd : nullptr);
    if (!FROM_ACCUM) acc_flush(acck, geo, hd, pgk ? pgk + p * hd : nullptr);
  }
}

// ------------------------------------------------------------------------------------------------ launchers
struct QkLaunch {
  int grid;
  int64_t parts, rows_per_part;
};
inline QkLaunch qk_launch(int64_t T) {
  QkLaunch l;
  l.parts = norm_parts(T);
  l.rows_per_part = (T + l.parts - 1) / l.parts;
  l.grid = static_cast<int>(l.parts);
  return l;
}

template <class S> const S* P(const void* p) { return static_cast<const S*>(p); }
template <class S> S* Pm(void* p) { return static_cast<S*>(p); }

template <class S, bool LEV, int G>
void fwd_g(const QkNormRopeFwd& a, cudaStream_t s) {
  auto k = k_qknorm_rope_fwd<S, LEV, G, true>;
  const QkLaunch l = qk_launch(a.T);
  const QkGeo q{a.T, a.H, a.KV, a.hd, 0, a.H + a.KV};
  k<<<l.grid, 256, 0, s>>>(P<S>(a.qkv), a.qkv_ld, P<S>(a.gq), P<S>(a.gk), P<S>(a.cos_tab), P<S>(a.sin_tab), a.pos,
                           Pm<S>(a.q_rot), Pm<S>(a.k_rot), a.rstd_q, a.rstd_k, q, a.eps, a.q_scale, l.parts,
                           l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void fwd(const QkNormRopeFwd& a, cudaStream_t s) {
  if (a.hd == 64) fwd_g<S, LEV, 8>(a, s); else fwd_g<S, LEV, 16>(a, s);
}

// RoPE only: rows [T, nheads] of hd at stride ld, all treated as "q" heads (scale applied; 1.0 = none).
struct RopeArgs {
  const void* x; int64_t ld; const void* cos_tab; const void* sin_tab; const int32_t* pos; void* out;
  int64_t T; int nheads, hd; float scale;
};
template <class S, bool LEV, int G>
void rope_g(const RopeArgs& a, cudaStream_t s) {
  auto k = k_qknorm_rope_fwd<S, LEV, G, false>;
  const QkLaunch l = qk_launch(a.T);
  const QkGeo q{a.T, a.nheads, 0, a.hd, 0, a.nheads};
  k<<<l.grid, 256, 0, s>>>(P<S>(a.x), a.ld, nullptr, nullptr, P<S>(a.cos_tab), P<S>(a.sin_tab), a.pos,
                           Pm<S>(a.out), nullptr, nullptr, nullptr, q, 0.f, a.scale, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void rope(const RopeArgs& a, cudaStream_t s) {
  if (a.hd == 64) rope_g<S, LEV, 8>(a, s); else rope_g<S, LEV, 16>(a, s);
}

template <class S, bool LEV, int G>
void bwd_g(const QkNormRopeBwd& a, cudaStream_t s) {
  auto k = k_qknorm_rope_bwd<S, LEV, G, false>;
  const QkLaunch l = qk_launch(a.T);
  const QkGeo q{a.T, a.H, a.KV, a.hd, a.dq_rot ? 0 : a.H, a.dk_rot ? a.H + a.KV : a.H};
  k<<<l.grid, 256, 0, s>>>(P<S>(a.dq_rot), P<S>(a.dk_rot), nullptr, P<S>(a.qkv), a.qkv_ld, P<S>(a.gq), P<S>(a.gk),
                           P<S>(a.cos_tab), P<S>(a.sin_tab), a.pos, a.rstd_q, a.rstd_k, Pm<S>(a.dqkv), a.dqkv_ld,
                           a.dgq_partial, a.dgk_partial, q, a.eps, a.q_scale, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void bwd(const QkNormRopeBwd& a, cudaStream_t s) {
  if (a.hd == 64) bwd_g<S, LEV, 8>(a, s); else bwd_g<S, LEV, 16>(a, s);
}

template <class S, bool LEV, int G>
void post_g(const DqPostQkNormRopeBwd& a, cudaStream_t s) {
  auto k = k_qknorm_rope_bwd<S, LEV, G, true>;
  const QkLaunch l = qk_launch(a.T);
  const QkGeo q{a.T, a.H, 0, a.hd, 0, a.H};
  k<<<l.grid, 256, 0, s>>>(nullptr, nullptr, a.dq_accum, P<S>(a.qkv), a.qkv_ld, P<S>(a.gq), P<S>(a.gq),
                           P<S>(a.cos_tab), P<S>(a.sin_tab), a.pos, a.rstd_q, nullptr, Pm<S>(a.dqkv), a.dqkv_ld,
                           a.dgq_partial, nullptr, q, a.eps, a.q_scale, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void post(const DqPostQkNormRopeBwd& a, cudaStream_t s) {
  if (a.hd == 64) post_g<S, LEV, 8>(a, s); else post_g<S, LEV, 16>(a, s);
}

// Every launcher for one rounding mode: PREFIX = template (explicit instantiation) or extern template.
#define NM_EW_QK_LAUNCHERS(PREFIX, LEV) \
  PREFIX void fwd<bf16_t, LEV>(const QkNormRopeFwd&, cudaStream_t); \
  PREFIX void fwd<float, LEV>(const QkNormRopeFwd&, cudaStream_t); \
  PREFIX void rope<bf16_t, LEV>(const RopeArgs&, cudaStream_t); \
  PREFIX void rope<float, LEV>(const RopeArgs&, cudaStream_t); \
  PREFIX void bwd<bf16_t, LEV>(const QkNormRopeBwd&, cudaStream_t); \
  PREFIX void bwd<float, LEV>(const QkNormRopeBwd&, cudaStream_t); \
  PREFIX void post<bf16_t, LEV>(const DqPostQkNormRopeBwd&, cudaStream_t); \
  PREFIX void post<float, LEV>(const DqPostQkNormRopeBwd&, cudaStream_t);

}  // namespace nm::ew::qk
