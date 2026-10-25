// Stage S5 internal API of the fused linear cross-entropy module (include/marincpp/kernels/cross_entropy.hpp holds the
// public one). DESIGN.md §3.8.
//
// Workspace layout of one call (every region starts on a 256-byte boundary, in this order, so the planner's buffers
// ws.ce.logits, ws.ce.tile_max, ws.ce.tile_sum, ws.ce.lse, ws.ce.gscale laid out back to back are exactly this block):
//   B1 / B1.5 / B2:  store [rows, ldS] (bf16 or f32 logits store; ldS = V rounded up to 8)
//                    tile_max, tile_sum [NTp, rows] f32 each (NTp = V / 128 + 1 >= ceil(V / 128); tile-major, so the
//                    logits kernel writes consecutive rows per tile and the finalize reads them coalesced)
//                    lse, gscale [rows] f32
//                    dxtmp: [rows, D] bf16 chunk of dX, used only by an in-place B1.5 call (dx == x) and by a
//                    levanter-rounded dx accumulation through cuBLASLt. It aliases tile_max..tile_sum (free after the
//                    finalize) when that span is large enough, else it is one more region at the end (not in the
//                    planner's ws.ce.* buffers: V < 32 D - 128, e.g. V = 8,256 at D = 2,048).
//   A:               logits [rows, ldS] f32 (cuBLASLt output; dz is written in place over it), lse, gscale, dxtmp
//                    (bf16 io only: the f32 policy needs none, so its layout is never larger than the bf16-io one).
//   all:             ltws [kLtWsBytes]: the cuBLASLt workspace of implementation A and of the cuBLASLt plain-dz GEMMs
//                    (not in the planner's ws.ce.* buffers: BUILD_LOG S5 request).
// rows = CeShape::chunk_rows (the most rows one chunk holds).
#pragma once

#include <cstdint>

#include "marincpp/kernels/cross_entropy.hpp"

namespace nm::ce {

constexpr int kTile = 128;   // vocab columns per (max, sumexp) partial
constexpr uint64_t kLtWsBytes = 32ull << 20;   // cuBLASLt workspace region of the CE workspace

inline int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }
inline int64_t store_ld(int V) { return round_up(V, 8); }
inline int stats_tiles(int V) { return V / kTile + 1; }   // planner shape [rows, V / 128 + 1]

struct Ws {
  void* store = nullptr; int64_t ldS = 0;       // logits store (A: f32 logits), ld in elements of its dtype
  float* tmax = nullptr; float* tsum = nullptr; int64_t ldT = 0; int NT = 0;   // NT = ceil(V / 128) written
  float* lse = nullptr; float* gscale = nullptr;
  void* dxtmp = nullptr;                         // [rows, D] bf16 (bf16 io only)
  void* ltws = nullptr;                          // [kLtWsBytes]
  uint64_t bytes = 0;                            // total of the layout
};
Ws carve(CeImpl impl, const CeShape& s, void* base);   // base == nullptr: sizes only

// Which kernel runs the dX and dW GEMMs of a plain bf16 dz (B1, B1.5's wgrad). Lt = cuBLASLt (one algorithm per shape
// class, chosen for a canonical row count so the result of a row does not depend on the chunk's row count), Own = this
// module's kernels (sm_90a WGMMA on H100, mma.sync elsewhere; the two give the same bits). The strict per-chunk dW
// rounding and every on-the-fly dz GEMM always use Own. Defaults: the faster on H100 (BUILD_LOG S5: cuBLASLt at 740-820
// TFLOP/s against 450-550 for Own); tests set both.
enum class GemmBackend : uint8_t { Lt = 0, Own = 1 };
constexpr int kSm90Logits = 1, kSm90Gemm = 2;   // Backends::sm90 bits: the sm_90a logits kernel, dX / dW kernels
struct Backends {
  GemmBackend dgrad = GemmBackend::Lt;
  GemmBackend wgrad = GemmBackend::Lt;
  int sm90 = kSm90Logits | kSm90Gemm;   // which sm_90a kernels may run on H100 (0: the portable ones everywhere)
  bool fast_stats_exp = false;           // exp_stat<true> in the logits statistics (ce_mma.cuh; off: numerics.md §1)
};
Backends& backends();   // process-wide; set before any graph capture

// One call of the module with an explicit cuBLASLt handle: fused_linear_ce() and the C ABI both end here. `lt` may be
// nullptr: the module's per-device handle is used (created on first use, before any graph capture).
void run(CeImpl impl, Rounding r, const CeShape& s, const CeArgs& a, void* ws, cublasLtHandle_t lt, cudaStream_t st);

// Readout (DESIGN.md §3.8 multi-head). eps = model.norm_eps (ReadoutCall carries none yet: BUILD_LOG S5 request).
void readout(CeImpl impl, Rounding r, const CeShape& s, const ReadoutCall& c, const CeArgs& base, float eps, void* ws,
             cublasLtHandle_t lt, cudaStream_t st);

// out = mult / sum(w[0..n)) in f32, the sum in a fixed order: the per-token gradient scale of the weighted mean
// (numerics.md §3.6: levanter's cotangent of sum(loss_tok * w) / sum(w) is w_t * f32(1 / sum(w))). For a mean over
// several microbatches or ranks the caller sums the weights first and passes them as w.
void grad_scale(const float* w, int64_t n, float mult, float* out, cudaStream_t st);

int device_arch();   // 80, 86, 90 ... of the current device (cached per device)
int sm_count();

// ------------------------------------------------------------------------------------------------ kernel launchers
// `store_dt` is the logits store dtype (DType::bf16 or DType::f32); DType::u8 marks "plain bf16 dz" for the GEMMs.
struct LogitsArgs {
  const void* x; const void* w; const float* bias; const int32_t* targets;
  int rows, D, V;
  void* store; int64_t ldS;               // store == nullptr: statistics only (forward-only / eval)
  float* tmax; float* tsum; int64_t ldT;  // [NTp, ldT]
  float* zt;                              // [rows] f32 z[target] (rows whose target is outside [0, V) are untouched)
  bool fast_exp;                          // Backends::fast_stats_exp
};
void launch_logits_stats(DType store_dt, const LogitsArgs& a, cudaStream_t st);           // dispatcher
void launch_logits_stats_portable(DType store_dt, const LogitsArgs& a, cudaStream_t st);  // mma.sync (sm_80+)
bool launch_logits_stats_sm90(DType store_dt, const LogitsArgs& a, cudaStream_t st);      // false: shape not taken

struct FinalizeArgs {
  const float* tmax; const float* tsum; int64_t ldT; int NT;
  const float* zt; const int32_t* targets; const float* weight; const float* grad_scale;
  int rows, V;
  float* lse; float* loss_tok; float* gscale;    // loss_tok may equal zt (read, then overwritten per row)
};
void launch_finalize(const FinalizeArgs& a, cudaStream_t st);
// out[0] = sum_t weight[t] * loss_tok[t] over rows t in a fixed order (one CTA).
void launch_chunk_loss(const float* loss_tok, const float* weight, int rows, float* out, cudaStream_t st);

// dz in place over the store (B1): bf16 store -> bf16 dz at the same place (ld ldS); f32 store -> bf16 dz written over
// the first half of each f32 row (ld 2 ldS bf16 elements). Columns [V, round_up(V, 8)) of dz are written as 0.
struct DzArgs {
  void* store; int64_t ldS;
  const float* lse; const float* gscale; const int32_t* targets;
  int rows, V;
};
void launch_dlogits_inplace(DType store_dt, const DzArgs& a, cudaStream_t st);

// Implementation A's row kernel on f32 logits [rows, ldS] (+ bias): lse, z[target], loss, g; with grads, dz in place
// (io bf16: bf16 dz over the first half of each row, p from f32(bf16(z)) when emulate_bf16_store; io f32: f32 dz).
struct RowArgs {
  float* logits; int64_t ldS; const float* bias;
  const int32_t* targets; const float* weight; const float* grad_scale;
  int rows, V;
  float* lse; float* loss_tok; float* gscale;
  bool grads; bool emulate_bf16_store; DType io;
};
void launch_row_a(const RowArgs& a, cudaStream_t st);

// dX = dz W, dz from the store on the fly (B1.5, B2) or a plain bf16 dz (store_dt == u8).
//   out = bf16(acc)                          (dx_old == nullptr)
//   out = bf16(f32(dx_old) + f32(bf16(acc))) (levanter accumulate)  |  bf16(f32(dx_old) + acc)  (accurate)
// write_dz (B1.5): the CTA owns its rows over all of D (N tiles looped inside the CTA) and writes bf16 dz in place over
// the store during the last N pass (bf16 store: ld ldS; f32 store: first half of each f32 row, ld 2 ldS).
struct DgradArgs {
  const void* store; int64_t ldS;         // store dtype elements; plain dz: bf16 [rows, ldS]
  const void* w;                          // [V, D] bf16
  const float* lse; const float* gscale; const int32_t* targets;
  int rows, D, V;
  const void* dx_old; void* dx_out;       // [rows, D] bf16 (dx_old may equal dx_out)
  bool write_dz; Rounding r;
};
void launch_dgrad(DType store_dt, const DgradArgs& a, cudaStream_t st);           // dispatcher
void launch_dgrad_portable(DType store_dt, const DgradArgs& a, cudaStream_t st);
bool launch_dgrad_sm90(DType store_dt, const DgradArgs& a, cudaStream_t st);

// dW (+)= dz^T x, dz from the store on the fly (B2) or plain bf16 dz (store_dt == u8).
//   epi_round_chunk == false:  dw[v, d] += acc (f32)                      (fast default)
//   epi_round_chunk == true:   dw[v, d] += f32(bf16(acc))   (dw = the shared per-microbatch accumulator, strict)
struct WgradArgs {
  const void* store; int64_t ldS;
  const void* x;                          // [rows, D] bf16
  const float* lse; const float* gscale; const int32_t* targets;
  int rows, D, V;
  float* dw;                              // [V, D] f32
  bool epi_round_chunk;
};
void launch_wgrad(DType store_dt, const WgradArgs& a, cudaStream_t st);           // dispatcher
void launch_wgrad_portable(DType store_dt, const WgradArgs& a, cudaStream_t st);
bool launch_wgrad_sm90(DType store_dt, const WgradArgs& a, cudaStream_t st);

// small kernels
// dx = tmp (accumulate false) | levanter bf16(f32(dx_old) + f32(tmp)) | accurate: the same (tmp is already rounded)
void launch_dx_combine(const void* tmp, const void* dx_old, void* dx, int64_t n, DType io, bool accumulate,
                       cudaStream_t st);
void launch_finish_microbatch(float* dw, float* dw_mb, int64_t n, cudaStream_t st);   // dw += f32(bf16(mb)); mb = 0
void launch_zero_cols(void* x, int64_t rows, int D, int from, DType dt, cudaStream_t st);  // x[:, from:] = 0
void launch_sum_scale(const float* w, int64_t n, float mult, float* out, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ cuBLASLt GEMMs
// Row-major formulas; `ws` = the kLtWsBytes region (Ws::ltws); one algorithm per (kind, io, D, V, ld) chosen by the
// heuristic for a canonical row count and reused for every row count it supports (ce_lt.cpp). io = bf16
// (CUBLAS_COMPUTE_32F) or f32 (CUBLAS_COMPUTE_32F_PEDANTIC).
cublasLtHandle_t lt_handle();
// C[rows, ldc] f32 = x[rows, D] W[V, D]^T
void lt_logits(cublasLtHandle_t lt, DType io, const void* x, const void* w, float* c, int64_t ldc, int rows, int D,
               int V, void* ws, cudaStream_t st);
// dx[rows, D] = dz[rows, V] W[V, D] + beta dx   (beta 0 or 1; dx and dz in io dtype)
void lt_dgrad(cublasLtHandle_t lt, DType io, const void* dz, int64_t ldz, const void* w, void* dx, float beta,
              int rows, int D, int V, void* ws, cudaStream_t st);
// dw[V, D] f32 += dz[rows, V]^T x[rows, D]
void lt_wgrad(cublasLtHandle_t lt, DType io, const void* dz, int64_t ldz, const void* x, float* dw, int rows, int D,
              int V, void* ws, cudaStream_t st);

}  // namespace nm::ce
