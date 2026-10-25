// Adam and AdamH (optim.md §5), multi-tensor over owned slices. Operation order of the H100 program levanter ran:
//   mu = (1-b1)*g + b1*mu;  nu = (1-b2)*(g*g) + b2*nu;  a = mu / (bc1 * (sqrt(nu / bc2) + eps))
// The jaxpr's (mu / bc1) / (sqrt(nu / bc2) + eps) is rewritten by XLA's algebraic simplifier into one division by a
// product (x / y / z -> x / (y * z)): the optimized HLO of the 130m train step (cache/levref/130m/t1_precision,
// fusions of lm_head's AdamH and of every Adam leaf: `divide(mu, multiply(bc1, add(sqrt(divide(nu, bc2)), eps)))`,
// the multiply named "multiply.4x" with no jaxpr frame). The order below is that program's.
//   Adam:  u = (-lr) * a;  p = p + u
//   AdamH: w = p - ((eta*a)*n) / max(m, 1e-10);  u = (w / ||w||) * n - p  (2-D branch: no floor);  p = p + u
// g is the clipped gradient (g * grad_scale, then (g / gn) * max_norm when the clip group clips).
#include "kernels/optim/optk_common.cuh"

namespace nm::optk {
namespace {

using namespace dev;

constexpr int kThreads = 256;

__device__ __forceinline__ float adam_dir(float mu, float nu, float bc1, float bc2, float eps) {
  return div(mu, mul(bc1, add(sqrt_rn(div(nu, bc2)), eps)));
}

__device__ __forceinline__ void moments(float g, float& mu, float& nu, const GroupDev& gr) {
  mu = add(mul(gr.one_minus_b1, g), mul(gr.b1, mu));
  nu = add(mul(gr.one_minus_b2, mul(g, g)), mul(gr.b2, nu));
}

// Rows of a slice: a slice with a row partition holds whole rows (col = 0), so element i is on row sl.row + i / cols.
__device__ __forceinline__ bool skip_elem(const RowPartTab& rp, const Slice& sl, int64_t cols, int i) {
  return sl.block >= 0 && row_skipped(rp, sl.block, sl.row + i / cols);
}

__global__ void __launch_bounds__(kThreads) k_adam(TensorTab t, SliceTab s, RowPartTab rp, const GroupDev* groups,
                                                   const ClipDev* clip, Tables tb, int s_begin) {
  const Slice sl = slice_at(s, s_begin + blockIdx.x);
  const int32_t k = cur_step(tb);
  if (block_frozen(rp, sl.block, k)) return;
  const GroupDev gr = groups[t.group[sl.tensor]];
  const ClipDev cl = clip[gr.clip];
  const int64_t off = slice_off(t, sl), cols = t.cols[sl.tensor];
  float* p = t.p32[sl.tensor] + off;
  bf16_t* p16 = t.p16[sl.tensor] ? t.p16[sl.tensor] + off : nullptr;
  const float* g = t.g[sl.tensor] + off;
  float* m = t.s0[sl.tensor] + off;
  float* v = t.s1[sl.tensor] + off;
  const float gs = t.grad_scale[sl.tensor], us = t.update_scale[sl.tensor];
  float* uo = t.u_out[sl.tensor] ? t.u_out[sl.tensor] + off : nullptr;
  const float bc1 = tb.bc1[k], bc2 = tb.bc2[k];
  const float lr = mul(mul(tb.lr[gr.lr_table][k], gr.lr_scale), block_lr_mult(rp, tb, sl.block, k));
  const float neg_lr = -lr;
  // one element: false when the row is skipped (lazy); p, mu, nu then keep their values
  auto elem = [&](int i, float gv, float& mu, float& nu, float& pv, float& uv) {
    if (skip_elem(rp, sl, cols, i)) return;
    const float gi = clip_g(mul(gv, gs), cl);
    moments(gi, mu, nu, gr);
    float a = adam_dir(mu, nu, bc1, bc2, gr.eps);
    if (gr.weight_decay != 0.0f) a = add(a, mul(gr.weight_decay, pv));   // optax add_decayed_weights
    uv = mul(mul(neg_lr, a), us);
    pv = add(pv, uv);
  };
  if (sl.len % 4 == 0 && al16(p) && al16(g) && al16(m) && al16(v) && (!uo || al16(uo)) && (!p16 || al8(p16))) {
    for (int q = threadIdx.x; q < sl.len / 4; q += kThreads) {
      F4 G, M, V, P, U{};
      G.load(g, q), M.load(m, q), V.load(v, q), P.load(p, q);
      for (int j = 0; j < 4; ++j) elem(4 * q + j, G.v[j], M.v[j], V.v[j], P.v[j], U.v[j]);
      M.store(m, q), V.store(v, q), P.store(p, q);
      if (uo) U.store(uo, q);
      if (p16) store_bf16x4(p16, q, P);
    }
    return;
  }
  for (int i = threadIdx.x; i < sl.len; i += kThreads) {
    if (skip_elem(rp, sl, cols, i)) continue;
    float mu = m[i], nu = v[i], pv = p[i], uv = 0.0f;
    elem(i, g[i], mu, nu, pv, uv);
    m[i] = mu;
    v[i] = nu;
    p[i] = pv;
    if (uo) uo[i] = uv;
    if (p16) reinterpret_cast<__nv_bfloat16*>(p16)[i] = __float2bfloat16_rn(pv);
  }
}

__global__ void __launch_bounds__(kThreads) k_adamh1(TensorTab t, SliceTab s, RowPartTab rp, const GroupDev* groups,
                                                     const ClipDev* clip, Tables tb, int s_begin, double* partial2) {
  const Slice sl = slice_at(s, s_begin + blockIdx.x);
  const int32_t k = cur_step(tb);
  double pp = 0.0, aa = 0.0;
  if (!block_frozen(rp, sl.block, k)) {
    const GroupDev gr = groups[t.group[sl.tensor]];
    const ClipDev cl = clip[gr.clip];
    const int64_t off = slice_off(t, sl), cols = t.cols[sl.tensor];
    const float* p = t.p32[sl.tensor] + off;
    const float* g = t.g[sl.tensor] + off;
    float* m = t.s0[sl.tensor] + off;
    float* v = t.s1[sl.tensor] + off;
    const float gs = t.grad_scale[sl.tensor];
    const float bc1 = tb.bc1[k], bc2 = tb.bc2[k];
    auto elem = [&](int i, float gv, float& mu, float& nu, float pv) {
      if (skip_elem(rp, sl, cols, i)) return;
      const float gi = clip_g(mul(gv, gs), cl);
      moments(gi, mu, nu, gr);
      const float a = adam_dir(mu, nu, bc1, bc2, gr.eps);
      pp += sq(pv);
      aa += sq(a);
    };
    if (sl.len % 4 == 0 && al16(p) && al16(g) && al16(m) && al16(v)) {
      for (int q = threadIdx.x; q < sl.len / 4; q += kThreads) {
        F4 G, M, V, P;
        G.load(g, q), M.load(m, q), V.load(v, q), P.load(p, q);
        for (int j = 0; j < 4; ++j) elem(4 * q + j, G.v[j], M.v[j], V.v[j], P.v[j]);
        M.store(m, q), V.store(v, q);
      }
    } else {
      for (int i = threadIdx.x; i < sl.len; i += kThreads) {
        float mu = m[i], nu = v[i];
        elem(i, g[i], mu, nu, p[i]);
        m[i] = mu;
        v[i] = nu;
      }
    }
  }
  pp = block_sum(pp);
  aa = block_sum(aa);
  if (threadIdx.x == 0) {
    partial2[2 * blockIdx.x] = pp;
    partial2[2 * blockIdx.x + 1] = aa;
  }
}

// w of one element (pass 2 and 3 recompute a from the stored moments: same bits as pass 1).
struct WArgs {
  float eta, n, m_floor, bc1, bc2;
};
__device__ __forceinline__ float adamh_w(float p, float mu, float nu, const GroupDev& gr, const WArgs& w) {
  const float a = adam_dir(mu, nu, w.bc1, w.bc2, gr.eps);
  return sub(p, div(mul(mul(w.eta, a), w.n), w.m_floor));
}
__device__ __forceinline__ WArgs w_args(const RowPartTab& rp, const Tables& tb, const GroupDev& gr, const Slice& sl,
                                        const double* slot_pa, int32_t k) {
  WArgs w;
  w.eta = mul(mul(tb.lr[gr.lr_table][k], gr.lr_scale), block_lr_mult(rp, tb, sl.block, k));
  w.n = norm_of(slot_pa[2 * sl.slot]);
  w.m_floor = fmax_nan(norm_of(slot_pa[2 * sl.slot + 1]), kNormFloor);
  w.bc1 = tb.bc1[k];
  w.bc2 = tb.bc2[k];
  return w;
}

__global__ void __launch_bounds__(kThreads) k_adamh2(TensorTab t, SliceTab s, RowPartTab rp, const GroupDev* groups,
                                                     Tables tb, const double* slot_pa, int s_begin, double* partial) {
  const Slice sl = slice_at(s, s_begin + blockIdx.x);
  const int32_t k = cur_step(tb);
  double ww = 0.0;
  if (!block_frozen(rp, sl.block, k)) {
    const GroupDev gr = groups[t.group[sl.tensor]];
    const WArgs wa = w_args(rp, tb, gr, sl, slot_pa, k);
    const int64_t off = slice_off(t, sl), cols = t.cols[sl.tensor];
    const float* p = t.p32[sl.tensor] + off;
    const float* m = t.s0[sl.tensor] + off;
    const float* v = t.s1[sl.tensor] + off;
    if (sl.len % 4 == 0 && al16(p) && al16(m) && al16(v)) {
      for (int q = threadIdx.x; q < sl.len / 4; q += kThreads) {
        F4 M, V, P;
        M.load(m, q), V.load(v, q), P.load(p, q);
        for (int j = 0; j < 4; ++j)
          if (!skip_elem(rp, sl, cols, 4 * q + j)) ww += sq(adamh_w(P.v[j], M.v[j], V.v[j], gr, wa));
      }
    } else {
      for (int i = threadIdx.x; i < sl.len; i += kThreads) {
        if (skip_elem(rp, sl, cols, i)) continue;
        ww += sq(adamh_w(p[i], m[i], v[i], gr, wa));
      }
    }
  }
  ww = block_sum(ww);
  if (threadIdx.x == 0) partial[blockIdx.x] = ww;
}

__global__ void __launch_bounds__(kThreads) k_adamh3(TensorTab t, SliceTab s, RowPartTab rp, const GroupDev* groups,
                                                     Tables tb, const double* slot_pa, const double* slot_w,
                                                     int s_begin) {
  const Slice sl = slice_at(s, s_begin + blockIdx.x);
  const int32_t k = cur_step(tb);
  if (block_frozen(rp, sl.block, k)) return;
  const GroupDev gr = groups[t.group[sl.tensor]];
  const WArgs wa = w_args(rp, tb, gr, sl, slot_pa, k);
  const float nw = norm_of(slot_w[sl.slot]);   // 2-D branch: new_param / jnp.linalg.norm(new_param), no floor
  const int64_t off = slice_off(t, sl), cols = t.cols[sl.tensor];
  float* p = t.p32[sl.tensor] + off;
  bf16_t* p16 = t.p16[sl.tensor] ? t.p16[sl.tensor] + off : nullptr;
  const float* m = t.s0[sl.tensor] + off;
  const float* v = t.s1[sl.tensor] + off;
  const float us = t.update_scale[sl.tensor];
  float* uo = t.u_out[sl.tensor] ? t.u_out[sl.tensor] + off : nullptr;
  auto elem = [&](int i, float& pv, float mu, float nu, float& uv) {
    if (skip_elem(rp, sl, cols, i)) return;
    const float w = adamh_w(pv, mu, nu, gr, wa);
    uv = mul(sub(mul(div(w, nw), wa.n), pv), us);
    pv = add(pv, uv);
  };
  if (sl.len % 4 == 0 && al16(p) && al16(m) && al16(v) && (!uo || al16(uo)) && (!p16 || al8(p16))) {
    for (int q = threadIdx.x; q < sl.len / 4; q += kThreads) {
      F4 M, V, P, U{};
      M.load(m, q), V.load(v, q), P.load(p, q);
      for (int j = 0; j < 4; ++j) elem(4 * q + j, P.v[j], M.v[j], V.v[j], U.v[j]);
      P.store(p, q);
      if (uo) U.store(uo, q);
      if (p16) store_bf16x4(p16, q, P);
    }
    return;
  }
  for (int i = threadIdx.x; i < sl.len; i += kThreads) {
    if (skip_elem(rp, sl, cols, i)) continue;
    float pv = p[i], uv = 0.0f;
    elem(i, pv, m[i], v[i], uv);
    p[i] = pv;
    if (uo) uo[i] = uv;
    if (p16) reinterpret_cast<__nv_bfloat16*>(p16)[i] = __float2bfloat16_rn(pv);
  }
}

}  // namespace

void adam_update(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const ClipDev* clip, const Tables& tb, int s_begin, int s_end, cudaStream_t st) {
  if (s_end <= s_begin) return;
  k_adam<<<s_end - s_begin, kThreads, 0, st>>>(t, s, rp, groups, clip, tb, s_begin);
  NM_CUDA(cudaGetLastError());
}

void adamh_pass1(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const ClipDev* clip, const Tables& tb, int s_begin, int s_end, double* partial2, cudaStream_t st) {
  if (s_end <= s_begin) return;
  k_adamh1<<<s_end - s_begin, kThreads, 0, st>>>(t, s, rp, groups, clip, tb, s_begin, partial2);
  NM_CUDA(cudaGetLastError());
}

void adamh_pass2(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const Tables& tb, const double* slot_pa, int s_begin, int s_end, double* partial, cudaStream_t st) {
  if (s_end <= s_begin) return;
  k_adamh2<<<s_end - s_begin, kThreads, 0, st>>>(t, s, rp, groups, tb, slot_pa, s_begin, partial);
  NM_CUDA(cudaGetLastError());
}

void adamh_pass3(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const Tables& tb, const double* slot_pa, const double* slot_w, int s_begin, int s_end,
                 cudaStream_t st) {
  if (s_end <= s_begin) return;
  k_adamh3<<<s_end - s_begin, kThreads, 0, st>>>(t, s, rp, groups, tb, slot_pa, slot_w, s_begin);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::optk
