// RMSNorm over D (block norms, final norm) and the fused sandwich tails. DESIGN.md §3.3, §3.6; numerics.md §3.2.
//
// Levanter rounding (numerics.md §3.2, the rounding points of levanter's jaxpr; built with --fmad=false):
//   forward   ms = sum(x32^2) / n;  r = rsqrt(ms + eps);  y = bf16(x32 * r);  out = bf16(g * y)
//   backward  a = bf16(g * dy);  dg += f32(bf16(dy * y));  s = sum(x32 * f32(a));
//             c = (s * (-0.5 * (r / (ms + eps)))) / n;  dx = bf16(f32(a) * r + c * (2 * x32))
// Accurate: f32 throughout, one rounding per stored output (dg accumulates dy * x32 * r in f32).
// rsqrt is CUDA's rsqrtf (libdevice __nv_rsqrtf, the function XLA:GPU calls); divisions are IEEE (__fdiv_rn).
// The residual stream is bf16: x_out = bf16(x_in + y); residual-gradient adds are bf16 adds of bf16 operands under
// levanter (dx_mid = bf16(dx_out + bf16(norm_bwd))) and one rounding under accurate.
//
// Kernels and launchers (templates over storage S and rounding LEV). The levanter instantiations (LEV = true) are built
// in rmsnorm.cu with --fmad=false, the accurate ones (LEV = false) in rmsnorm_acc.cu with nvcc's default contraction
// (DESIGN.md §3.0); NM_EW_RMS_LAUNCHERS lists the launchers both units instantiate.
#pragma once

#include <string>
#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_rows.cuh"

namespace nm::ew::rms {

// ------------------------------------------------------------------------------------------------ k_rmsnorm_fwd
template <class S, bool LEV>
__global__ void __launch_bounds__(512, 2) k_rmsnorm_fwd(const S* __restrict__ x, const S* __restrict__ gain,
                                                        S* __restrict__ y, float* __restrict__ rstd, int64_t T, int D,
                                                        float eps, int64_t parts, int64_t rows_per_part) {
  using V = R8<S, LEV>;
  __shared__ float red[kRedSlots];
  int phase = 0;
  const RowGeo geo(D);
  const V g = V::load(gain + geo.col0);
  const int64_t iters = (rows_per_part + geo.groups - 1) / geo.groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t r0 = p * T / parts, r1 = (p + 1) * T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      const int64_t row = r0 + it * geo.groups + geo.grp;
      const bool on = row < r1;
      float xf[8];
      V::load_z(x + row * D + geo.col0, on).f32(xf);
      float ss[1] = {sum8_sq(xf)};
      row_sum<1>(ss, geo, red, phase);
      const float r = rsqrtf(__fdiv_rn(ss[0], static_cast<float>(D)) + eps);
      if (on) {
        if (y) norm_out<S, LEV>(xf, r, g).store(y + row * D + geo.col0);
        if (rstd && geo.lig == 0) rstd[row] = r;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ k_rmsnorm_bwd_add
template <class S, bool LEV>
__global__ void __launch_bounds__(512, 2) k_rmsnorm_bwd_add(const S* __restrict__ dy, const S* __restrict__ x,
                                                            const S* __restrict__ gain,
                                                            const float* __restrict__ rstd, const S* dres, S* dx,
                                                            float* __restrict__ partial, int64_t T, int D, float eps,
                                                            int64_t parts, int64_t rows_per_part) {
  using V = R8<S, LEV>;
  __shared__ float red[kRedSlots];
  __shared__ __align__(16) float acc[kMaxRowG * 8];
  int phase = 0;
  const RowGeo geo(D);
  acc_zero(acc, geo, D);
  const V g = V::load(gain + geo.col0);
  const int64_t iters = (rows_per_part + geo.groups - 1) / geo.groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t r0 = p * T / parts, r1 = (p + 1) * T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      const int64_t row = r0 + it * geo.groups + geo.grp;
      const bool on = row < r1;
      const int64_t off = row * D + geo.col0;
      const V xv = V::load_z(x + off, on), dyv = V::load_z(dy + off, on);
      float xf[8], af[8];
      xv.f32(xf);
      V::mul(g, dyv).f32(af);   // a = round(g * dy)
      float sums[2] = {sum8_sq(xf), sum8_dot(xf, af)};
      row_sum<2>(sums, geo, red, phase);
      const float ms = __fdiv_rn(sums[0], static_cast<float>(D));
      const float r = (rstd && on) ? __ldg(rstd + row) : rsqrtf(ms + eps);
      float d[8], gc[8];
      norm_dx<S, LEV>(xf, dyv, af, r, ms, sums[1], eps, D, gc, d);
      acc_add(acc, geo, D, gc);
      if (on) {
        if (dres) {
          if (LEV) {
            V::add(V::load_cg(dres + off), V::round(d)).store(dx + off);   // bf16(dres + bf16(dx))
          } else {
            float rv[8];
            V::load_cg(dres + off).f32(rv);
#pragma unroll
            for (int j = 0; j < 8; ++j) d[j] += rv[j];
            V::round(d).store(dx + off);
          }
        } else {
          V::round(d).store(dx + off);
        }
      }
    }
    acc_flush(acc, geo, D, partial ? partial + p * D : nullptr);
  }
}

// ------------------------------------------------------------------------------------------------ k_sandwich_tail_fwd
template <class S, bool LEV>
__global__ void __launch_bounds__(512, 2) k_sandwich_fwd(const S* __restrict__ branch, const S* __restrict__ x_in,
                                                         const S* __restrict__ g_post, const S* __restrict__ g_next,
                                                         S* __restrict__ x_out, S* __restrict__ a_next,
                                                         float* __restrict__ rstd_post, float* __restrict__ rstd_next,
                                                         int64_t T, int D, float eps, int64_t parts,
                                                         int64_t rows_per_part) {
  using V = R8<S, LEV>;
  __shared__ float red[kRedSlots];
  int phase = 0;
  const RowGeo geo(D);
  const V gp = g_post ? V::load(g_post + geo.col0) : V::zero();
  const V gn = V::load(g_next + geo.col0);
  const int64_t iters = (rows_per_part + geo.groups - 1) / geo.groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t r0 = p * T / parts, r1 = (p + 1) * T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      const int64_t row = r0 + it * geo.groups + geo.grp;
      const bool on = row < r1;
      const int64_t off = row * D + geo.col0;
      const V b = V::load_z(branch + off, on), xi = V::load_z(x_in + off, on);
      V yv = b;
      if (g_post) {
        float bf[8];
        b.f32(bf);
        float ss[1] = {sum8_sq(bf)};
        row_sum<1>(ss, geo, red, phase);
        const float r = rsqrtf(__fdiv_rn(ss[0], static_cast<float>(D)) + eps);
        yv = norm_out<S, LEV>(bf, r, gp);
        if (on && rstd_post && geo.lig == 0) rstd_post[row] = r;
      }
      // the residual stream is stored (bf16): x_out = round_storage(x_in + y)
      float xof[8];
      if (LEV) {
        V::add(xi, yv).f32(xof);
      } else {
        float a[8], c[8];
        xi.f32(a);
        yv.f32(c);
#pragma unroll
        for (int j = 0; j < 8; ++j) xof[j] = st<S>(a[j] + c[j]);
      }
      float ss[1] = {sum8_sq(xof)};
      row_sum<1>(ss, geo, red, phase);
      const float r = rsqrtf(__fdiv_rn(ss[0], static_cast<float>(D)) + eps);
      if (on) {
        V::round(xof).store(x_out + off);
        if (a_next) norm_out<S, LEV>(xof, r, gn).store(a_next + off);
        if (rstd_next && geo.lig == 0) rstd_next[row] = r;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ k_sandwich_tail_bwd
template <class S, bool LEV>
__global__ void __launch_bounds__(512, 2) k_sandwich_bwd(const S* dx_out, const S* __restrict__ da_next,
                                                         const S* __restrict__ x_out, const S* __restrict__ g_next,
                                                         const float* __restrict__ rstd_next,
                                                         const S* __restrict__ branch, const S* __restrict__ g_post,
                                                         const float* __restrict__ rstd_post, S* dx_in, S* dbranch,
                                                         float* __restrict__ pg_next, float* __restrict__ pg_post,
                                                         int64_t T, int D, float eps, int64_t parts,
                                                         int64_t rows_per_part) {
  using V = R8<S, LEV>;
  __shared__ float red[kRedSlots];
  __shared__ __align__(16) float acc_n[kMaxRowG * 8];
  __shared__ __align__(16) float acc_p[kMaxRowG * 8];
  int phase = 0;
  const RowGeo geo(D);
  acc_zero(acc_n, geo, D);
  acc_zero(acc_p, geo, D);
  const V gn = V::load(g_next + geo.col0);
  const V gp = g_post ? V::load(g_post + geo.col0) : V::zero();
  const int64_t iters = (rows_per_part + geo.groups - 1) / geo.groups;
  for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
    const int64_t r0 = p * T / parts, r1 = (p + 1) * T / parts;
    for (int64_t it = 0; it < iters; ++it) {
      const int64_t row = r0 + it * geo.groups + geo.grp;
      const bool on = row < r1;
      const int64_t off = row * D + geo.col0;
      const V xo = V::load_z(x_out + off, on), da = V::load_z(da_next + off, on);
      const V b = g_post ? V::load_z(branch + off, on) : V::zero();
      float s3[3];
      {
        float xf[8], af[8], bf[8];
        xo.f32(xf);
        V::mul(gn, da).f32(af);
        b.f32(bf);
        s3[0] = sum8_sq(xf);
        s3[1] = sum8_dot(xf, af);
        s3[2] = sum8_sq(bf);
      }
      row_sum<3>(s3, geo, red, phase);
      const float ms_n = __fdiv_rn(s3[0], static_cast<float>(D));
      const float r_n = (rstd_next && on) ? __ldg(rstd_next + row) : rsqrtf(ms_n + eps);
      V dxm;
      {
        float xf[8], af[8], dxn[8], gc[8];
        xo.f32(xf);
        V::mul(gn, da).f32(af);
        norm_dx<S, LEV>(xf, da, af, r_n, ms_n, s3[1], eps, D, gc, dxn);
        acc_add(acc_n, geo, D, gc);
        // dx_out may alias dx_in (in-place residual gradient): a coherent load, never ld.global.nc, on a buffer this
        // kernel writes
        const V dxo = V::load_z_cg(dx_out + off, on);
        if (LEV) {
          dxm = V::add(dxo, V::round(dxn));   // bf16(dx_out + bf16(norm_bwd))
        } else {
          float o[8];
          dxo.f32(o);
#pragma unroll
          for (int j = 0; j < 8; ++j) o[j] += dxn[j];
          dxm = V::round(o);   // accurate: f32, rounded at the stores
        }
      }
      if (on) dxm.store(dx_in + off);
      if (g_post) {
        float bf[8], apf[8];
        b.f32(bf);
        V::mul(gp, dxm).f32(apf);
        float s1[1] = {sum8_dot(bf, apf)};
        row_sum<1>(s1, geo, red, phase);
        const float ms_p = __fdiv_rn(s3[2], static_cast<float>(D));
        const float r_p = (rstd_post && on) ? __ldg(rstd_post + row) : rsqrtf(ms_p + eps);
        float db[8], gc[8];
        norm_dx<S, LEV>(bf, dxm, apf, r_p, ms_p, s1[0], eps, D, gc, db);
        acc_add(acc_p, geo, D, gc);
        if (on) V::round(db).store(dbranch + off);
      } else if (on && dbranch != dx_in) {
        dxm.store(dbranch + off);
      }
    }
    acc_flush(acc_n, geo, D, pg_next ? pg_next + p * D : nullptr);
    if (g_post) acc_flush(acc_p, geo, D, pg_post ? pg_post + p * D : nullptr);
  }
}

// ------------------------------------------------------------------------------------------------ launchers
struct RowLaunch {
  int block;
  int grid;
  int64_t parts, rows_per_part;
};
inline RowLaunch row_launch(int64_t T, int D) {
  RowLaunch l;
  l.block = row_block(D / 8);
  l.parts = norm_parts(T);
  l.rows_per_part = (T + l.parts - 1) / l.parts;   // the most rows any part holds (sets the loop count)
  l.grid = static_cast<int>(l.parts);
  return l;
}

template <class S> const S* P(const void* p) { return static_cast<const S*>(p); }
template <class S> S* Pm(void* p) { return static_cast<S*>(p); }

template <class S, bool LEV>
void launch_fwd(const RmsNormFwd& a, cudaStream_t s) {
  auto k = k_rmsnorm_fwd<S, LEV>;
  const RowLaunch l = row_launch(a.T, a.D);
  k<<<l.grid, l.block, 0, s>>>(P<S>(a.x), P<S>(a.gain), Pm<S>(a.y), a.rstd, a.T, a.D, a.eps, l.parts,
                               l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void launch_bwd(const RmsNormBwdAdd& a, cudaStream_t s) {
  auto k = k_rmsnorm_bwd_add<S, LEV>;
  const RowLaunch l = row_launch(a.T, a.D);
  k<<<l.grid, l.block, 0, s>>>(P<S>(a.dy), P<S>(a.x), P<S>(a.gain), a.rstd, P<S>(a.dres), Pm<S>(a.dx),
                               a.dgain_partial, a.T, a.D, a.eps, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void launch_sw_fwd(const SandwichTailFwd& a, cudaStream_t s) {
  auto k = k_sandwich_fwd<S, LEV>;
  const RowLaunch l = row_launch(a.T, a.D);
  k<<<l.grid, l.block, 0, s>>>(P<S>(a.branch), P<S>(a.x_in), P<S>(a.g_post), P<S>(a.g_next), Pm<S>(a.x_out),
                               Pm<S>(a.a_next), a.rstd_post, a.rstd_next, a.T, a.D, a.eps, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void launch_sw_bwd(const SandwichTailBwd& a, cudaStream_t s) {
  auto k = k_sandwich_bwd<S, LEV>;
  const RowLaunch l = row_launch(a.T, a.D);
  k<<<l.grid, l.block, 0, s>>>(P<S>(a.dx_out), P<S>(a.da_next), P<S>(a.x_out), P<S>(a.g_next), a.rstd_next,
                               P<S>(a.branch), P<S>(a.g_post), a.rstd_post, Pm<S>(a.dx_in), Pm<S>(a.dbranch),
                               a.dg_next_partial, a.dg_post_partial, a.T, a.D, a.eps, l.parts, l.rows_per_part);
  NM_CUDA(cudaGetLastError());
}

// Every launcher for one rounding mode: PREFIX = template (explicit instantiation) or extern template.
#define NM_EW_RMS_LAUNCHERS(PREFIX, LEV) \
  PREFIX void launch_fwd<bf16_t, LEV>(const RmsNormFwd&, cudaStream_t); \
  PREFIX void launch_fwd<float, LEV>(const RmsNormFwd&, cudaStream_t); \
  PREFIX void launch_bwd<bf16_t, LEV>(const RmsNormBwdAdd&, cudaStream_t); \
  PREFIX void launch_bwd<float, LEV>(const RmsNormBwdAdd&, cudaStream_t); \
  PREFIX void launch_sw_fwd<bf16_t, LEV>(const SandwichTailFwd&, cudaStream_t); \
  PREFIX void launch_sw_fwd<float, LEV>(const SandwichTailFwd&, cudaStream_t); \
  PREFIX void launch_sw_bwd<bf16_t, LEV>(const SandwichTailBwd&, cudaStream_t); \
  PREFIX void launch_sw_bwd<float, LEV>(const SandwichTailBwd&, cudaStream_t);

}  // namespace nm::ew::rms
