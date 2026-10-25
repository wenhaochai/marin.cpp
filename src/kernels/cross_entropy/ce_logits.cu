// k_ce_logits_stats, k_ce_finalize_rows, the per-chunk loss sum, B1's in-place dlogits and implementation A's row
// kernel. DESIGN.md §3.8 steps 1-3. Built with --fmad=false: every f32 op is one rounded op as written (no contraction
// of p * g - g); exp and log are libdevice expf / logf (what XLA:GPU emits).
#include <algorithm>
#include <cfloat>

#include "kernels/cross_entropy/ce_internal.hpp"
#include "kernels/cross_entropy/ce_mma.cuh"

namespace nm::ce {
namespace {

// ------------------------------------------------------------------------------------------------ logits + statistics
// z = x W^T (+ bias) for a 128-row x 128-column tile with f32 accumulators (4 warps of 64 x 64, BK 32, 4-stage cp.async
// pipeline). Epilogue from the f32 accumulators: per row the tile's (max, sum exp(z - max)) over its valid columns,
// z[target] when the target falls in the tile, and the logits store (bf16 or f32) through a shared-memory staging tile
// so every global store is a full 16-byte chunk.
constexpr int LBM = 128, LBN = 128, LBK = 32, LST = 4, LTH = 128;
constexpr int kLStageA = LBM * LBK * 2, kLStageB = LBN * LBK * 2;
constexpr int kLPipe = LST * (kLStageA + kLStageB);   // 65,536 B
constexpr int kLSmem = kLPipe + 2 * 2 * LBM * 4;      // + red_m[2][128], red_s[2][128]
static_assert(LBN == kTile, "one statistics tile per CTA column tile");
static_assert(LBM * 136 * 2 <= kLPipe && 64 * 132 * 4 <= kLPipe, "staging tiles fit in the pipeline buffers");

template <int STORE, bool FAST>   // STORE: 0 none, 1 bf16, 2 f32; FAST: exp_stat<true>
__global__ void __launch_bounds__(LTH, 2) k_ce_logits_stats(LogitsArgs p) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.x * LBM, n0 = blockIdx.y * LBN;
  const bf16* X = static_cast<const bf16*>(p.x);
  const bf16* W = static_cast<const bf16*>(p.w);
  const int KT = (p.D + LBK - 1) / LBK;
  const uint32_t sbase = smem_u32(sm);

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * LBK;
    const uint32_t sa = sbase + stage * (kLStageA + kLStageB), sb = sa + kLStageA;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = tid + i * LTH;
      const int r = q >> 2, c = q & 3, k = k0 + c * 8;
      const int gm = m0 + r, gn = n0 + r;
      const bool pa = gm < p.rows && k < p.D, pb = gn < p.V && k < p.D;
      const uint32_t off = r * 64 + swz<4>(r, c) * 16;
      cp_async16(sa + off, pa ? X + int64_t(gm) * p.D + k : X, pa);
      cp_async16(sb + off, pb ? W + int64_t(gn) * p.D + k : W, pb);
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
  for (int s = 0; s < LST - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<LST - 2>();
    __syncthreads();
    {
      const int nk = kt + LST - 1;
      if (nk < KT) load_stage(nk % LST, nk);
      cp_async_commit();
    }
    const uint32_t sa = sbase + (kt % LST) * (kLStageA + kLStageB), sb = sa + kLStageA;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[4][4], bq[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = wm * 64 + i * 16 + (lane & 15), c = kk * 2 + (lane >> 4);
        ldsm_x4(sa + r * 64 + swz<4>(r, c) * 16, af[i]);
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int r = wn * 64 + jj * 16 + (lane & 7) + ((lane >> 4) << 3), c = kk * 2 + ((lane >> 3) & 1);
        ldsm_x4(sb + r * 64 + swz<4>(r, c) * 16, bq[jj]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma16816(acc[i][j], af[i], bq[j >> 1][(j & 1) * 2], bq[j >> 1][(j & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- epilogue: z = acc (+ bias), per-row tile statistics, z[target], store
  bool colok[8][2];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int gn = n0 + wn * 64 + j * 8 + 2 * t4 + e;
      colok[j][e] = gn < p.V;
      if (p.bias) {
        const float b = colok[j][e] ? p.bias[gn] : 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[i][j][e] = __fadd_rn(acc[i][j][e], b);
          acc[i][j][2 + e] = __fadd_rn(acc[i][j][2 + e], b);
        }
      }
    }
  float* red_m = reinterpret_cast<float*>(sm + kLPipe);
  float* red_s = red_m + 2 * LBM;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int rl = wm * 64 + i * 16 + h * 8 + g;
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          if (colok[j][e]) mx = fmaxf(mx, acc[i][j][h * 2 + e]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      // branch-free (see ce_dz_sel): every exponential is evaluated against a safe max and selected
      const float ms = mx == -INFINITY ? 0.f : mx;
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float x = exp_stat<FAST>(__fsub_rn(acc[i][j][h * 2 + e], ms));
          s = __fadd_rn(s, colok[j][e] ? x : 0.f);
        }
      s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
      s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
      if (t4 == 0) {
        red_m[wn * LBM + rl] = mx;
        red_s[wn * LBM + rl] = s;
      }
      const int gm = m0 + rl;
      if (gm < p.rows) {
        const int tg = p.targets[gm];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (n0 + wn * 64 + j * 8 + 2 * t4 + e == tg) p.zt[gm] = acc[i][j][h * 2 + e];
      }
    }

  if constexpr (STORE == 1) {
    bf16* stg = reinterpret_cast<bf16*>(sm);   // [128][136]
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int rl = wm * 64 + i * 16 + h * 8 + g, cl = wn * 64 + j * 8 + 2 * t4;
          *reinterpret_cast<uint32_t*>(stg + rl * 136 + cl) = pack_bf16x2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
        }
    __syncthreads();
    bf16* out = static_cast<bf16*>(p.store);
    for (int q = tid; q < LBM * 16; q += LTH) {
      const int r = q >> 4, c = q & 15, gm = m0 + r, gc = n0 + c * 8;
      if (gm < p.rows && gc < p.V)
        *reinterpret_cast<uint4*>(out + int64_t(gm) * p.ldS + gc) = *reinterpret_cast<const uint4*>(stg + r * 136 + c * 8);
    }
  } else if constexpr (STORE == 2) {
    float* stg = reinterpret_cast<float*>(sm);   // [64][132], one warp row (64 rows) at a time
    float* out = static_cast<float*>(p.store);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      if (wm == half) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int rl = i * 16 + h * 8 + g, cl = wn * 64 + j * 8 + 2 * t4;
              *reinterpret_cast<float2*>(stg + rl * 132 + cl) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
            }
      }
      __syncthreads();
      for (int q = tid; q < 64 * 32; q += LTH) {
        const int r = q >> 5, c = q & 31, gm = m0 + half * 64 + r, gc = n0 + c * 4;
        if (gm < p.rows && gc < p.V)
          *reinterpret_cast<float4*>(out + int64_t(gm) * p.ldS + gc) =
              *reinterpret_cast<const float4*>(stg + r * 132 + c * 4);
      }
      __syncthreads();
    }
  } else {
    __syncthreads();
  }

  // merge the two 64-column halves of each row (fixed order: half 0, then half 1)
  {
    const int gm = m0 + tid;
    if (gm < p.rows) {
      const float ma = red_m[tid], sa = red_s[tid], mb = red_m[LBM + tid], sb = red_s[LBM + tid];
      const float M = fmaxf(ma, mb);
      float S = 0.f;
      if (M != -INFINITY)
        S = __fadd_rn(__fmul_rn(sa, expf(__fsub_rn(ma, M))), __fmul_rn(sb, expf(__fsub_rn(mb, M))));
      p.tmax[int64_t(blockIdx.y) * p.ldT + gm] = M;
      p.tsum[int64_t(blockIdx.y) * p.ldT + gm] = S;
    }
  }
}

// ------------------------------------------------------------------------------------------------ finalize
// One thread per row; tiles merged in ascending order: M = max_i m_i (0 if not finite, as jax.nn.logsumexp),
// S = sum_i s_i exp(m_i - M), lse = log(S) + M, CE = lse - z[target].
__global__ void __launch_bounds__(128) k_ce_finalize_rows(FinalizeArgs p) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= p.rows) return;
  float M = -INFINITY;
#pragma unroll 8
  for (int i = 0; i < p.NT; ++i) M = fmaxf(M, p.tmax[int64_t(i) * p.ldT + r]);
  if (!isfinite(M)) M = 0.f;
  float S = 0.f;
#pragma unroll 8
  for (int i = 0; i < p.NT; ++i) {
    const int64_t o = int64_t(i) * p.ldT + r;
    S = __fadd_rn(S, __fmul_rn(p.tsum[o], expf(__fsub_rn(p.tmax[o], M))));
  }
  const float lse = __fadd_rn(logf(S), M);
  const int tg = p.targets[r];
  const float w = p.weight[r];
  const bool valid = tg >= 0 && tg < p.V;   // a target outside [0, V) is ignored: no loss, no gradient
  float loss = 0.f;
  if (valid && w != 0.f) loss = __fsub_rn(lse, p.zt[r]);
  p.lse[r] = lse;
  p.loss_tok[r] = loss;
  p.gscale[r] = (p.grad_scale && valid) ? __fmul_rn(w, *p.grad_scale) : 0.f;
}

// sum_t weight[t] * loss[t], one CTA of 1024 threads: thread-strided sequential sums, then a fixed binary tree.
__global__ void __launch_bounds__(1024) k_ce_chunk_loss(const float* loss, const float* w, int rows, float* out) {
  __shared__ float red[1024];
  float s = 0.f;
  for (int r = threadIdx.x; r < rows; r += 1024) s = __fadd_rn(s, __fmul_rn(w[r], loss[r]));
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = 512; o > 0; o >>= 1) {
    if (int(threadIdx.x) < o) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = red[0];
}

// ------------------------------------------------------------------------------------------------ dlogits (B1)
// bf16 store: each thread rewrites one 16-byte chunk (8 logits) of kDzRows rows as bf16 dz, in place; the loads of
// the rows are issued before any math (memory-level parallelism: one chunk per thread left HBM at 2 TB/s on the
// H100), and the grid is a few CTAs per SM that stride over the row groups.
constexpr int kDzRows = 4;
__global__ void __launch_bounds__(256) k_ce_dlogits_bf16(DzArgs p) {
  const int nch = (p.V + 7) / 8;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nch) return;
  const int v0 = c * 8;
  for (int r0 = blockIdx.y * kDzRows; r0 < p.rows; r0 += gridDim.y * kDzRows) {
    uint4 raw[kDzRows];
#pragma unroll
    for (int q = 0; q < kDzRows; ++q)
      if (r0 + q < p.rows)
        raw[q] = *reinterpret_cast<const uint4*>(static_cast<const bf16*>(p.store) + int64_t(r0 + q) * p.ldS + v0);
#pragma unroll
    for (int q = 0; q < kDzRows; ++q) {
      const int r = r0 + q;
      if (r >= p.rows) break;
      const float lse = p.lse[r], gs = p.gscale[r];
      const int tg = p.targets[r];
      uint32_t* u = reinterpret_cast<uint32_t*>(&raw[q]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int v = v0 + 2 * e;
        const float d0 = ce_dz_sel(bf16lo(u[e]), lse, gs, v == tg, v < p.V);
        const float d1 = ce_dz_sel(bf16hi(u[e]), lse, gs, v + 1 == tg, v + 1 < p.V);
        u[e] = pack_bf16x2(d0, d1);
      }
      *reinterpret_cast<uint4*>(static_cast<bf16*>(p.store) + int64_t(r) * p.ldS + v0) = raw[q];
    }
  }
}

// f32 store: bf16 dz written over the first half of each f32 row (ld 2 ldS bf16 elements). One CTA walks a row in
// blocks of 1024 logits in ascending order: block b writes bytes [2048 b, 2048 b + 2048) of the row, i.e. f32 elements
// [512 b, 512 b + 512), which blocks <= b have read; the barrier between the reads and the writes of a block covers
// b = 0 (and orders every earlier block's reads before this block's writes).
__global__ void __launch_bounds__(256) k_ce_dlogits_f32(DzArgs p) {
  for (int r = blockIdx.x; r < p.rows; r += gridDim.x) {
    const float* zr = static_cast<const float*>(p.store) + int64_t(r) * p.ldS;
    bf16* dr = reinterpret_cast<bf16*>(static_cast<float*>(p.store) + int64_t(r) * p.ldS);
    const float lse = p.lse[r], gs = p.gscale[r];
    const int tg = p.targets[r];
    for (int v0 = 0; v0 < p.V; v0 += 1024) {
      const int v = v0 + int(threadIdx.x) * 4;
      float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      if (v < p.V) z = *reinterpret_cast<const float4*>(zr + v);
      __syncthreads();
      if (v < p.V) {
        const float d0 = ce_dz(z.x, lse, gs, v == tg);
        const float d1 = ce_dz_sel(z.y, lse, gs, v + 1 == tg, v + 1 < p.V);
        const float d2 = ce_dz_sel(z.z, lse, gs, v + 2 == tg, v + 2 < p.V);
        const float d3 = ce_dz_sel(z.w, lse, gs, v + 3 == tg, v + 3 < p.V);
        *reinterpret_cast<uint2*>(dr + v) = make_uint2(pack_bf16x2(d0, d1), pack_bf16x2(d2, d3));
      }
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------ implementation A
// One CTA per row over the f32 logits: max, sum exp(z - max) (thread-strided sequential sums, then a fixed tree), the
// row's lse / CE / g, then dz in place. io bf16: bf16 dz over the first half of the row (the block order of
// k_ce_dlogits_f32); io f32: f32 dz in place.
constexpr int kRowThreads = 512;

__device__ __forceinline__ float block_reduce(float v, float* red, bool is_max) {
  red[threadIdx.x] = v;
  __syncthreads();
  for (int o = kRowThreads / 2; o > 0; o >>= 1) {
    if (int(threadIdx.x) < o)
      red[threadIdx.x] = is_max ? fmaxf(red[threadIdx.x], red[threadIdx.x + o])
                                : __fadd_rn(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

__global__ void __launch_bounds__(kRowThreads) k_ce_row_a(RowArgs p) {
  __shared__ float red[kRowThreads];
  for (int r = blockIdx.x; r < p.rows; r += gridDim.x) {
    float* zr = p.logits + int64_t(r) * p.ldS;
    auto zval = [&](int v) { return p.bias ? __fadd_rn(zr[v], p.bias[v]) : zr[v]; };
    float mx = -INFINITY;
    for (int v = threadIdx.x; v < p.V; v += kRowThreads) mx = fmaxf(mx, zval(v));
    float M = block_reduce(mx, red, true);
    if (!isfinite(M)) M = 0.f;
    float s = 0.f;
    for (int v = threadIdx.x; v < p.V; v += kRowThreads) s = __fadd_rn(s, expf(__fsub_rn(zval(v), M)));
    const float S = block_reduce(s, red, false);
    const float lse = __fadd_rn(logf(S), M);
    const int tg = p.targets[r];
    const float w = p.weight[r];
    const bool valid = tg >= 0 && tg < p.V;
    const float gs = (p.grad_scale && valid) ? __fmul_rn(w, *p.grad_scale) : 0.f;
    if (threadIdx.x == 0) {
      float loss = 0.f;
      if (valid && w != 0.f) loss = __fsub_rn(lse, zval(tg));
      p.lse[r] = lse;
      p.loss_tok[r] = loss;
      p.gscale[r] = gs;
    }
    if (!p.grads) continue;
    __syncthreads();   // thread 0 has read z[target] above; the dz below overwrites the row in place
    if (p.io == DType::f32) {
      for (int v = threadIdx.x; v < p.V; v += kRowThreads) zr[v] = ce_dz(zval(v), lse, gs, v == tg);
      __syncthreads();
    } else {
      bf16* dr = reinterpret_cast<bf16*>(zr);
      for (int v0 = 0; v0 < p.V; v0 += kRowThreads) {
        const int v = v0 + int(threadIdx.x);
        float z = 0.f;
        if (v < p.V) z = zval(v);
        __syncthreads();
        if (v < p.V) {
          const float zs = p.emulate_bf16_store ? __bfloat162float(__float2bfloat16_rn(z)) : z;
          dr[v] = __float2bfloat16_rn(ce_dz(zs, lse, gs, v == tg));
        }
      }
      __syncthreads();
    }
  }
}

template <class K>
void set_smem(K kernel, int bytes) {
  // a function attribute: legal during graph capture, cheap to repeat
  NM_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

void launch_logits_stats_portable(DType store_dt, const LogitsArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  NM_REQUIRE(a.D % 8 == 0, "fused CE: D must be a multiple of 8");
  NM_REQUIRE(a.ldT >= a.rows, "fused CE: tile-statistics ld below the row count");
  const dim3 grid((a.rows + LBM - 1) / LBM, (a.V + LBN - 1) / LBN);
  NM_REQUIRE(grid.y <= 65535u, "fused CE: V too large for the logits grid");
  auto go = [&](auto kernel) {
    set_smem(kernel, kLSmem);
    kernel<<<grid, LTH, kLSmem, st>>>(a);
  };
  NM_REQUIRE(!a.store || store_dt == DType::bf16 || store_dt == DType::f32, "fused CE: logits store must be bf16 or f32");
  const int sel = (a.store ? (store_dt == DType::bf16 ? 1 : 2) : 0) * 2 + (a.fast_exp ? 1 : 0);
  switch (sel) {
    case 0: go(k_ce_logits_stats<0, false>); break;
    case 1: go(k_ce_logits_stats<0, true>); break;
    case 2: go(k_ce_logits_stats<1, false>); break;
    case 3: go(k_ce_logits_stats<1, true>); break;
    case 4: go(k_ce_logits_stats<2, false>); break;
    default: go(k_ce_logits_stats<2, true>); break;
  }
  NM_CUDA(cudaGetLastError());
}

void launch_finalize(const FinalizeArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  k_ce_finalize_rows<<<(a.rows + 127) / 128, 128, 0, st>>>(a);
  NM_CUDA(cudaGetLastError());
}

void launch_chunk_loss(const float* loss_tok, const float* weight, int rows, float* out, cudaStream_t st) {
  k_ce_chunk_loss<<<1, 1024, 0, st>>>(loss_tok, weight, rows, out);
  NM_CUDA(cudaGetLastError());
}

void launch_dlogits_inplace(DType store_dt, const DzArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  if (store_dt == DType::bf16) {
    const int nch = (a.V + 7) / 8;
    const int gx = (nch + 255) / 256;
    const int groups = (a.rows + kDzRows - 1) / kDzRows;
    const dim3 grid(gx, std::max(1, std::min(groups, (16 * sm_count() + gx - 1) / gx)));
    k_ce_dlogits_bf16<<<grid, 256, 0, st>>>(a);
  } else {
    k_ce_dlogits_f32<<<std::min(a.rows, 16 * sm_count()), 256, 0, st>>>(a);
  }
  NM_CUDA(cudaGetLastError());
}

void launch_row_a(const RowArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  k_ce_row_a<<<std::min(a.rows, 8 * sm_count()), kRowThreads, 0, st>>>(a);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::ce
