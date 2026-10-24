// SwiGLU forward and backward on the interleaved gate/up layout. DESIGN.md §3.5; numerics.md §3.5.
//
// Layout: gu [T, 2F] holds 64-column blocks gate_b | up_b (columns 128b..128b+63 = gate columns 64b..64b+63, columns
// 128b+64..128b+127 = the matching up columns); h and dh are [T, F] in plain column order; dgu has gu's layout.
// Levanter (levanter's jaxpr, every op rounds; XLA expands logistic(x) to 1 / (1 + exp(-x)) in the operand dtype):
//   e = bf16(exp(-g));  s = bf16(1 / bf16(1 + e));  silu = bf16(g * s);  h = bf16(silu * u)
//   backward:  sp = bf16(s * bf16(1 - s));  dsilu = bf16(dh * u);  du = bf16(dh * silu);
//              dg = bf16(bf16(dsilu * s) + bf16(bf16(g * dsilu) * sp))
// Accurate: s = 1 / (1 + exp(-g)) in f32;  h = bf16(g * s * u);  dg = bf16(dh * u * s * (1 + g * (1 - s)));
//           du = bf16(dh * g * s).
// exp is CUDA's expf (libdevice __nv_expf, what XLA:GPU calls; a bf16 exp is the f32 exp rounded to bf16).
//
// Kernels and launchers (templates over storage S and rounding LEV). The levanter instantiations (LEV = true) are built
// in swiglu.cu with --fmad=false, the accurate ones (LEV = false) in swiglu_acc.cu with nvcc's default contraction
// (DESIGN.md §3.0); NM_EW_SW_LAUNCHERS lists the launchers both units instantiate.
#pragma once

#include <string>
#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_common.cuh"

namespace nm::ew::sw {

template <class S, bool LEV>
__device__ __forceinline__ float logistic(float g) {
  if (LEV) {
    const float e = rnd<S, LEV>(expf(-g));
    return rnd<S, LEV>(__fdiv_rn(1.f, rnd<S, LEV>(1.f + e)));
  }
  return __fdiv_rn(1.f, 1.f + expf(-g));
}

// Vector v of 8 h columns: row t, h columns [8c, 8c + 8) with c = v % (F / 8); gate columns at 128*(c/8) + 8*(c%8).
__device__ __forceinline__ void swiglu_cols(int64_t v, int F8, int64_t& t, int64_t& hcol, int64_t& gcol) {
  t = v / F8;
  const int64_t c = v - t * F8;
  hcol = c * 8;
  gcol = (c >> 3) * 128 + (c & 7) * 8;
}

template <class S, bool LEV>
__global__ void __launch_bounds__(256) k_swiglu_fwd(const S* __restrict__ gu, S* __restrict__ h, int64_t nvec, int F) {
  const int F8 = F / 8;
  for (int64_t v = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; v < nvec; v += int64_t(gridDim.x) * blockDim.x) {
    int64_t t, hc, gc;
    swiglu_cols(v, F8, t, hc, gc);
    const S* row = gu + t * 2 * int64_t(F);
    float g[8], u[8], o[8];
    St<S>::load8(row + gc, g);
    St<S>::load8(row + gc + 64, u);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = logistic<S, LEV>(g[j]);
      o[j] = LEV ? rnd<S, LEV>(rnd<S, LEV>(g[j] * s) * u[j]) : (g[j] * s) * u[j];
    }
    St<S>::store8(h + t * int64_t(F) + hc, o);
  }
}

template <class S, bool LEV>
__global__ void __launch_bounds__(256) k_swiglu_bwd(const S* __restrict__ dh, const S* __restrict__ gu,
                                                    S* __restrict__ dgu, int64_t nvec, int F) {
  const int F8 = F / 8;
  for (int64_t v = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; v < nvec; v += int64_t(gridDim.x) * blockDim.x) {
    int64_t t, hc, gc;
    swiglu_cols(v, F8, t, hc, gc);
    const S* row = gu + t * 2 * int64_t(F);
    float g[8], u[8], d[8], dg[8], du[8];
    St<S>::load8(row + gc, g);
    St<S>::load8(row + gc + 64, u);
    St<S>::load8(dh + t * int64_t(F) + hc, d);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = logistic<S, LEV>(g[j]);
      if (LEV) {
        const float silu = rnd<S, LEV>(g[j] * s);
        const float sp = rnd<S, LEV>(s * rnd<S, LEV>(1.f - s));
        const float dsilu = rnd<S, LEV>(d[j] * u[j]);
        du[j] = d[j] * silu;
        dg[j] = rnd<S, LEV>(dsilu * s) + rnd<S, LEV>(rnd<S, LEV>(g[j] * dsilu) * sp);
      } else {
        const float dsilu = d[j] * u[j];
        du[j] = d[j] * (g[j] * s);
        dg[j] = dsilu * s * (1.f + g[j] * (1.f - s));
      }
    }
    S* orow = dgu + t * 2 * int64_t(F);
    St<S>::store8(orow + gc, dg);
    St<S>::store8(orow + gc + 64, du);
  }
}

template <class S> const S* P(const void* p) { return static_cast<const S*>(p); }
template <class S> S* Pm(void* p) { return static_cast<S*>(p); }

struct SwArgs {
  const void* dh;
  const void* gu;
  void* out;
  int64_t T;
  int F;
};

template <class S, bool LEV>
void fwd(const SwArgs& a, cudaStream_t s) {
  auto k = k_swiglu_fwd<S, LEV>;
  const int64_t nvec = a.T * (a.F / 8);
  const int grid = grid_for_items(nvec, 256, reinterpret_cast<const void*>(k), 256, 0);
  k<<<grid, 256, 0, s>>>(P<S>(a.gu), Pm<S>(a.out), nvec, a.F);
  NM_CUDA(cudaGetLastError());
}
template <class S, bool LEV>
void bwd(const SwArgs& a, cudaStream_t s) {
  auto k = k_swiglu_bwd<S, LEV>;
  const int64_t nvec = a.T * (a.F / 8);
  const int grid = grid_for_items(nvec, 256, reinterpret_cast<const void*>(k), 256, 0);
  k<<<grid, 256, 0, s>>>(P<S>(a.dh), P<S>(a.gu), Pm<S>(a.out), nvec, a.F);
  NM_CUDA(cudaGetLastError());
}

// Every launcher for one rounding mode: PREFIX = template (explicit instantiation) or extern template.
#define NM_EW_SW_LAUNCHERS(PREFIX, LEV) \
  PREFIX void fwd<bf16_t, LEV>(const SwArgs&, cudaStream_t); \
  PREFIX void fwd<float, LEV>(const SwArgs&, cudaStream_t); \
  PREFIX void bwd<bf16_t, LEV>(const SwArgs&, cudaStream_t); \
  PREFIX void bwd<float, LEV>(const SwArgs&, cudaStream_t);

}  // namespace nm::ew::sw
