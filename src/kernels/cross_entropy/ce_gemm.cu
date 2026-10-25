// Portable (sm_80 + sm_90a) dX and dW GEMMs of the fused CE: plain bf16 dz (B1, B1.5's wgrad), or dz computed from the
// logits store on the fly (B1.5's dgrad, B2), with the epilogues of ce_internal.hpp. mma.sync m16n8k16 bf16 -> f32,
// 4-stage cp.async pipelines, no split-K: every output element is one sequential K loop, so dX of a row does not depend
// on the number of rows in the call and the result is bitwise repeatable. Built with --fmad=false (ce_dz).
#include <algorithm>

#include "kernels/cross_entropy/ce_internal.hpp"
#include "kernels/cross_entropy/ce_mma.cuh"

namespace nm::ce {
namespace {

// SRC: 0 plain bf16 dz, 1 bf16 logits store (dz in place in shared memory), 2 f32 logits store (dz into a separate
// bf16 tile)
template <int SRC> struct Src;
template <> struct Src<0> { using T = bf16; static constexpr int kElem = 2; };
template <> struct Src<1> { using T = bf16; static constexpr int kElem = 2; };
template <> struct Src<2> { using T = float; static constexpr int kElem = 4; };

__device__ __forceinline__ float dz_or_zero(float z, float lse, float gs, bool tgt, bool ok) {
  return ce_dz_sel(z, lse, gs, tgt, ok);
}

// ------------------------------------------------------------------------------------------------ dgrad
// dx[rows, D] = dz[rows, V] W[V, D]. CTA: 64 rows x 128 columns of D, 4 warps of 32 x 64, BK = 32 over V.
constexpr int DBM = 64, DBN = 128, DBK = 32, DST = 4, DTH = 128;

template <int SRC>
struct DgradSmem {
  static constexpr int kA = DBM * DBK * Src<SRC>::kElem;   // store / dz tile of one stage
  static constexpr int kB = DBK * DBN * 2;                  // W tile [DBK][DBN]
  static constexpr int kStage = kA + kB;
  static constexpr int kDz = SRC == 2 ? DBM * DBK * 2 : 0;  // bf16 dz tile (x2, alternating) for an f32 store
  static constexpr int kMeta = 3 * DBM * 4;
  static constexpr int kTotal = DST * kStage + 2 * kDz + kMeta;
};

template <int SRC>
__global__ void __launch_bounds__(DTH) k_ce_dgrad(DgradArgs p) {
  using SM = DgradSmem<SRC>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.x * DBM;
  const uint32_t sbase = smem_u32(sm);
  const uint32_t dzbase = sbase + DST * SM::kStage;
  float* lse_s = reinterpret_cast<float*>(sm + DST * SM::kStage + 2 * SM::kDz);
  float* gs_s = lse_s + DBM;
  int* tg_s = reinterpret_cast<int*>(gs_s + DBM);
  if constexpr (SRC != 0) {
    if (tid < DBM) {
      const int r = m0 + tid;
      const bool ok = r < p.rows;
      lse_s[tid] = ok ? p.lse[r] : 0.f;
      gs_s[tid] = ok ? p.gscale[r] : 0.f;
      tg_s[tid] = ok ? p.targets[r] : -1;
    }
  }
  const typename Src<SRC>::T* A = static_cast<const typename Src<SRC>::T*>(p.store);
  const bf16* W = static_cast<const bf16*>(p.w);
  const int KT = (p.V + DBK - 1) / DBK;
  const int n_tiles = (p.D + DBN - 1) / DBN;
  const int nt_begin = p.write_dz ? 0 : blockIdx.y, nt_end = p.write_dz ? n_tiles : blockIdx.y + 1;
  __syncthreads();

  for (int nt = nt_begin; nt < nt_end; ++nt) {
    const int n0 = nt * DBN;
    const bool last_pass = p.write_dz && nt == n_tiles - 1;
    auto load_stage = [&](int stage, int kt) {
      const int k0 = kt * DBK;
      const uint32_t sa = sbase + stage * SM::kStage, sb = sa + SM::kA;
      if constexpr (SRC == 2) {   // f32 tile [64][8 chunks of 4], unswizzled
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int q = tid + i * DTH, r = q >> 3, c = q & 7, gm = m0 + r, k = k0 + c * 4;
          const bool pr = gm < p.rows && k < p.V;
          cp_async16(sa + r * 128 + c * 16, pr ? A + int64_t(gm) * p.ldS + k : A, pr);
        }
      } else {                     // bf16 tile [64][4 chunks of 8], swz<4>
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int q = tid + i * DTH, r = q >> 2, c = q & 3, gm = m0 + r, k = k0 + c * 8;
          const bool pr = gm < p.rows && k < p.V;
          cp_async16(sa + r * 64 + swz<4>(r, c) * 16, pr ? A + int64_t(gm) * p.ldS + k : A, pr);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {   // W tile [32][16 chunks], swz<16>
        const int q = tid + i * DTH, r = q >> 4, c = q & 15, v = k0 + r, d = n0 + c * 8;
        const bool pr = v < p.V && d < p.D;
        cp_async16(sb + r * 256 + swz<16>(r, c) * 16, pr ? W + int64_t(v) * p.D + d : W, pr);
      }
    };

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int s = 0; s < DST - 1; ++s) {
      if (s < KT) load_stage(s, s);
      cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
      cp_async_wait<DST - 2>();
      __syncthreads();
      {
        const int nk = kt + DST - 1;
        if (nk < KT) load_stage(nk % DST, nk);
        cp_async_commit();
      }
      const int k0 = kt * DBK;
      uint8_t* sa_g = sm + (kt % DST) * SM::kStage;
      uint32_t sa = sbase + (kt % DST) * SM::kStage;
      const uint32_t sb = sbase + (kt % DST) * SM::kStage + SM::kA;
      // ---- A operand: transform the store tile into bf16 dz (and write it back to global in B1.5's last pass)
      if constexpr (SRC == 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int q = tid + i * DTH, r = q >> 2, c = q & 3, v0 = k0 + c * 8;
          uint4* cp = reinterpret_cast<uint4*>(sa_g + r * 64 + swz<4>(r, c) * 16);
          uint4 raw = *cp;
          uint32_t* u = reinterpret_cast<uint32_t*>(&raw);
          const float lse = lse_s[r], gs = gs_s[r];
          const int tg = tg_s[r];
          const bool rok = m0 + r < p.rows;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int v = v0 + 2 * e;
            u[e] = pack_bf16x2(dz_or_zero(bf16lo(u[e]), lse, gs, v == tg, rok && v < p.V),
                               dz_or_zero(bf16hi(u[e]), lse, gs, v + 1 == tg, rok && v + 1 < p.V));
          }
          *cp = raw;
          if (last_pass && rok && v0 < p.V)
            *reinterpret_cast<uint4*>(static_cast<bf16*>(const_cast<void*>(p.store)) + int64_t(m0 + r) * p.ldS + v0) =
                raw;
        }
        __syncthreads();
      } else if constexpr (SRC == 2) {
        uint8_t* dz_g = sm + DST * SM::kStage + (kt & 1) * SM::kDz;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int q = tid + i * DTH, r = q >> 3, cf = q & 7, v0 = k0 + cf * 4;
          const float4 z = *reinterpret_cast<const float4*>(sa_g + r * 128 + cf * 16);
          const float lse = lse_s[r], gs = gs_s[r];
          const int tg = tg_s[r];
          const bool rok = m0 + r < p.rows;
          const uint2 d = make_uint2(pack_bf16x2(dz_or_zero(z.x, lse, gs, v0 == tg, rok && v0 < p.V),
                                                 dz_or_zero(z.y, lse, gs, v0 + 1 == tg, rok && v0 + 1 < p.V)),
                                     pack_bf16x2(dz_or_zero(z.z, lse, gs, v0 + 2 == tg, rok && v0 + 2 < p.V),
                                                 dz_or_zero(z.w, lse, gs, v0 + 3 == tg, rok && v0 + 3 < p.V)));
          const int c = cf >> 1;
          *reinterpret_cast<uint2*>(dz_g + r * 64 + swz<4>(r, c) * 16 + (cf & 1) * 8) = d;
          // bytes [2 v0, 2 v0 + 8) of the f32 row: f32 columns below v0, already read by this CTA's last pass
          if (last_pass && rok && v0 < p.V)
            *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(const_cast<void*>(p.store)) + int64_t(m0 + r) * 2 * p.ldS +
                                      v0) = d;
        }
        sa = dzbase + (kt & 1) * SM::kDz;
        __syncthreads();
      } else {
        if (k0 + DBK > p.V) {   // plain dz: zero the columns past V (an f32-store dz row has unwritten padding there)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int q = tid + i * DTH, r = q >> 2, c = q & 3, v0 = k0 + c * 8;
            bf16* e8 = reinterpret_cast<bf16*>(sa_g + r * 64 + swz<4>(r, c) * 16);
#pragma unroll
            for (int e = 0; e < 8; ++e)
              if (v0 + e >= p.V) e8[e] = __float2bfloat16_rn(0.f);
          }
          __syncthreads();
        }
      }
      // ---- MMA
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t af[2][4], bq[4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int r = wm * 32 + i * 16 + (lane & 15), c = kk * 2 + (lane >> 4);
          ldsm_x4(sa + r * 64 + swz<4>(r, c) * 16, af[i]);
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, c = wn * 8 + jj * 2 + (lane >> 4);
          ldsm_x4_t(sb + r * 256 + swz<16>(r, c) * 16, bq[jj]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) mma16816(acc[i][j], af[i], bq[j >> 1][(j & 1) * 2], bq[j >> 1][(j & 1) * 2 + 1]);
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    // ---- epilogue: dx = bf16(acc) | bf16(old + f32(bf16(acc))) (levanter) | bf16(old + acc) (accurate)
    const bf16* old = static_cast<const bf16*>(p.dx_old);
    bf16* out = static_cast<bf16*>(p.dx_out);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int gm = m0 + wm * 32 + i * 16 + h * 8 + g;
        if (gm >= p.rows) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int gn = n0 + wn * 64 + j * 8 + 2 * t4;
          if (gn >= p.D) continue;
          float a0 = acc[i][j][h * 2], a1 = acc[i][j][h * 2 + 1];
          const int64_t o = int64_t(gm) * p.D + gn;
          if (old) {
            const uint32_t ov = *reinterpret_cast<const uint32_t*>(old + o);
            if (p.r == Rounding::Levanter) {
              a0 = __bfloat162float(__float2bfloat16_rn(a0));
              a1 = __bfloat162float(__float2bfloat16_rn(a1));
            }
            a0 = __fadd_rn(bf16lo(ov), a0);
            a1 = __fadd_rn(bf16hi(ov), a1);
          }
          *reinterpret_cast<uint32_t*>(out + o) = pack_bf16x2(a0, a1);
        }
      }
    __syncthreads();   // the next N pass reuses the pipeline buffers
  }
}

// ------------------------------------------------------------------------------------------------ wgrad
// dw[V, D] (+)= dz[rows, V]^T x[rows, D]. CTA: 128 vocab rows x 128 columns of D, 4 warps of 64 x 64, BK = 32 rows.
constexpr int WBM = 128, WBN = 128, WBK = 32, WST = 4, WTH = 128;

template <int SRC>
struct WgradSmem {
  static constexpr int kA = WBK * WBM * Src<SRC>::kElem;   // store / dz tile [WBK rows][WBM vocab]
  static constexpr int kB = WBK * WBN * 2;                  // x tile [WBK][WBN]
  static constexpr int kStage = kA + kB;
  static constexpr int kDz = SRC == 2 ? WBK * WBM * 2 : 0;
  static constexpr int kTotal = WST * kStage + 2 * kDz;
};

template <int SRC>
__global__ void __launch_bounds__(WTH) k_ce_wgrad(WgradArgs p) {
  using SM = WgradSmem<SRC>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.x * WBM, n0 = blockIdx.y * WBN;
  const uint32_t sbase = smem_u32(sm);
  const typename Src<SRC>::T* A = static_cast<const typename Src<SRC>::T*>(p.store);
  const bf16* X = static_cast<const bf16*>(p.x);
  const int KT = (p.rows + WBK - 1) / WBK;

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * WBK;
    const uint32_t sa = sbase + stage * SM::kStage, sb = sa + SM::kA;
    if constexpr (SRC == 2) {   // f32 tile [32][32 chunks of 4], unswizzled
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q = tid + i * WTH, r = q >> 5, c = q & 31, gr = k0 + r, v = m0 + c * 4;
        const bool pr = gr < p.rows && v < p.V;
        cp_async16(sa + r * 512 + c * 16, pr ? A + int64_t(gr) * p.ldS + v : A, pr);
      }
    } else {                     // bf16 tile [32][16 chunks of 8], swz<16>
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int q = tid + i * WTH, r = q >> 4, c = q & 15, gr = k0 + r, v = m0 + c * 8;
        const bool pr = gr < p.rows && v < p.V;
        cp_async16(sa + r * 256 + swz<16>(r, c) * 16, pr ? A + int64_t(gr) * p.ldS + v : A, pr);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {   // x tile [32][16 chunks], swz<16>
      const int q = tid + i * WTH, r = q >> 4, c = q & 15, gr = k0 + r, d = n0 + c * 8;
      const bool pr = gr < p.rows && d < p.D;
      cp_async16(sb + r * 256 + swz<16>(r, c) * 16, pr ? X + int64_t(gr) * p.D + d : X, pr);
    }
  };

  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < WST - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<WST - 2>();
    __syncthreads();
    {
      const int nk = kt + WST - 1;
      if (nk < KT) load_stage(nk % WST, nk);
      cp_async_commit();
    }
    const int k0 = kt * WBK;
    uint8_t* sa_g = sm + (kt % WST) * SM::kStage;
    uint32_t sa = sbase + (kt % WST) * SM::kStage;
    const uint32_t sb = sa + SM::kA;
    if constexpr (SRC == 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int q = tid + i * WTH, r = q >> 4, c = q & 15, gr = k0 + r, v0 = m0 + c * 8;
        uint4* cp = reinterpret_cast<uint4*>(sa_g + r * 256 + swz<16>(r, c) * 16);
        uint4 raw = *cp;
        uint32_t* u = reinterpret_cast<uint32_t*>(&raw);
        const bool rok = gr < p.rows;
        const float lse = rok ? p.lse[gr] : 0.f, gs = rok ? p.gscale[gr] : 0.f;
        const int tg = rok ? p.targets[gr] : -1;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int v = v0 + 2 * e;
          u[e] = pack_bf16x2(dz_or_zero(bf16lo(u[e]), lse, gs, v == tg, rok && v < p.V),
                             dz_or_zero(bf16hi(u[e]), lse, gs, v + 1 == tg, rok && v + 1 < p.V));
        }
        *cp = raw;
      }
      __syncthreads();
    } else if constexpr (SRC == 2) {
      uint8_t* dz_g = sm + WST * SM::kStage + (kt & 1) * SM::kDz;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q = tid + i * WTH, r = q >> 5, cf = q & 31, gr = k0 + r, v0 = m0 + cf * 4;
        const float4 z = *reinterpret_cast<const float4*>(sa_g + r * 512 + cf * 16);
        const bool rok = gr < p.rows;
        const float lse = rok ? p.lse[gr] : 0.f, gs = rok ? p.gscale[gr] : 0.f;
        const int tg = rok ? p.targets[gr] : -1;
        const uint2 d = make_uint2(pack_bf16x2(dz_or_zero(z.x, lse, gs, v0 == tg, rok && v0 < p.V),
                                               dz_or_zero(z.y, lse, gs, v0 + 1 == tg, rok && v0 + 1 < p.V)),
                                   pack_bf16x2(dz_or_zero(z.z, lse, gs, v0 + 2 == tg, rok && v0 + 2 < p.V),
                                               dz_or_zero(z.w, lse, gs, v0 + 3 == tg, rok && v0 + 3 < p.V)));
        const int c = cf >> 1;
        *reinterpret_cast<uint2*>(dz_g + r * 256 + swz<16>(r, c) * 16 + (cf & 1) * 8) = d;
      }
      sa = sbase + WST * SM::kStage + (kt & 1) * SM::kDz;
      __syncthreads();
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[4][4], bq[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {   // A = dz^T from the [k][m] tile: transposed loads
        const int q = lane >> 3, r = kk * 16 + (q >> 1) * 8 + (lane & 7), c = (wm * 64 + i * 16) / 8 + (q & 1);
        ldsm_x4_t(sa + r * 256 + swz<16>(r, c) * 16, af[i]);
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, c = wn * 8 + jj * 2 + (lane >> 4);
        ldsm_x4_t(sb + r * 256 + swz<16>(r, c) * 16, bq[jj]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma16816(acc[i][j], af[i], bq[j >> 1][(j & 1) * 2], bq[j >> 1][(j & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();

  // ---- epilogue: dw += acc | dw += f32(bf16(acc)); each element owned by this CTA (no atomics)
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int gv = m0 + wm * 64 + i * 16 + h * 8 + g;
      if (gv >= p.V) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int gn = n0 + wn * 64 + j * 8 + 2 * t4;
        if (gn >= p.D) continue;
        float a0 = acc[i][j][h * 2], a1 = acc[i][j][h * 2 + 1];
        if (p.epi_round_chunk) {
          a0 = __bfloat162float(__float2bfloat16_rn(a0));
          a1 = __bfloat162float(__float2bfloat16_rn(a1));
        }
        float2* o = reinterpret_cast<float2*>(p.dw + int64_t(gv) * p.D + gn);
        float2 c = *o;
        c.x = __fadd_rn(c.x, a0);
        c.y = __fadd_rn(c.y, a1);
        *o = c;
      }
    }
}

template <class K>
void set_smem(K kernel, int bytes) {
  NM_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <int SRC>
void dgrad_t(const DgradArgs& a, cudaStream_t st) {
  const int mt = (a.rows + DBM - 1) / DBM, nt = (a.D + DBN - 1) / DBN;
  const dim3 grid(mt, a.write_dz ? 1 : nt);
  set_smem(k_ce_dgrad<SRC>, DgradSmem<SRC>::kTotal);
  k_ce_dgrad<SRC><<<grid, DTH, DgradSmem<SRC>::kTotal, st>>>(a);
  NM_CUDA(cudaGetLastError());
}

template <int SRC>
void wgrad_t(const WgradArgs& a, cudaStream_t st) {
  const dim3 grid((a.V + WBM - 1) / WBM, (a.D + WBN - 1) / WBN);
  set_smem(k_ce_wgrad<SRC>, WgradSmem<SRC>::kTotal);
  k_ce_wgrad<SRC><<<grid, WTH, WgradSmem<SRC>::kTotal, st>>>(a);
  NM_CUDA(cudaGetLastError());
}

}  // namespace

void launch_dgrad_portable(DType store_dt, const DgradArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  NM_REQUIRE(a.D % 8 == 0 && a.ldS % 8 == 0, "fused CE dgrad: D and the store ld must be multiples of 8");
  NM_REQUIRE(!(a.write_dz && store_dt == DType::u8), "fused CE dgrad: write_dz needs a logits store");
  if (store_dt == DType::u8) dgrad_t<0>(a, st);
  else if (store_dt == DType::bf16) dgrad_t<1>(a, st);
  else if (store_dt == DType::f32) dgrad_t<2>(a, st);
  else NM_REQUIRE(false, "fused CE dgrad: store dtype must be bf16, f32 or u8 (plain dz)");
}

void launch_wgrad_portable(DType store_dt, const WgradArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  NM_REQUIRE(a.D % 8 == 0 && a.ldS % 8 == 0, "fused CE wgrad: D and the store ld must be multiples of 8");
  if (store_dt == DType::u8) wgrad_t<0>(a, st);
  else if (store_dt == DType::bf16) wgrad_t<1>(a, st);
  else if (store_dt == DType::f32) wgrad_t<2>(a, st);
  else NM_REQUIRE(false, "fused CE wgrad: store dtype must be bf16, f32 or u8 (plain dz)");
}

}  // namespace nm::ce
