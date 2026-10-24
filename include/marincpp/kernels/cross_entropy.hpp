// Fused linear cross-entropy: x [T, D] x W [V, D] -> per-token CE, dx, dW (+= f32). DESIGN.md §3.8.
// Full-vocab f32 logits are never materialized in the bf16 policy (except as the optional f32 logits STORE of one
// chunk). One numeric path for every microbatch size: rows are processed in chunks of ce_chunk_rows; only the summation
// order of dW depends on the chunking.
//
// Math per row t (natural log), with grad scale g_t = weight[t] * (*grad_scale):
//   z     = x_t W^T                       f32 accumulate
//   lse_t = m + log(sum exp(z - m))       from the f32 accumulators (levanter forward, numerics.md §5.2)
//   CE_t  = lse_t - z[target_t]
//   p     = exp(z_s - lse_t)              z_s = the stored logit: f32(bf16(z)) with a bf16 store, z with an f32 store.
//                                         The store dtype is exec.ce_logits = the T1.0 item 2 fact (does XLA keep the
//                                         bf16 output of levanter's backward logit dot?)
//   dz    = p * g_t;  dz[target_t] -= g_t;  dzb = bf16(dz)
//   dx_t  = bf16(dzb W)                   one GEMM over the full V
//   dW   += dzb^T x                       f32; strict mode rounds each chunk's dW to bf16 (CeDwRound::PerChunkBf16)
//
// Implementations (exec.ce), extra HBM traffic beyond the GEMM operands:
//   A    cuBLASLt f32 logits of the chunk + row kernel; the reference and the f32-policy path. ~14V B/token.
//   B1   k_ce_logits_stats writes the store + per-(row, 128-col tile) partial (max, sumexp) + z[target];
//        k_ce_finalize_rows; k_ce_dlogits_inplace rewrites the store as bf16 dz; dgrad + wgrad read dz.   6V B/token
//   B1.5 as B1, but the dgrad kernel computes dz from the store on the fly and writes bf16 dz for the wgrad. 4V B/token
//   B2   dgrad and wgrad both compute dz from the store in a transform warpgroup (store tile in smem -> bf16 dz ->
//        smem -> WGMMA), as FA3 overlaps its softmax.                                                    2V B/token
// Kernel requirements (all B stages): persistent, TMA, warp-specialized. The logits kernel uses ping-pong consumer
// warpgroups so one warpgroup's exp/statistics epilogue overlaps the other's mainloop (at D = 512 the exp + f32 work is
// ~50% of the tensor time: 16 exp2 vs 4096 bf16 FLOP per SM-clock; part of the exps may run as an FMA polynomial).
// dgrad and wgrad use N = 512 output tiles (two consumer warpgroups sharing A, or a 2-CTA cluster with TMA multicast);
// with N = 256 each dz/store element is read D/256 times and the GEMM sits at 256 FLOP/B, below the 295 FLOP/B ridge.
// Decision rule (O2): per D in {512, 768, 1024, 2048}, the fastest implementation that passes T1.5.
#pragma once

#include <cstdint>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

class GemmCache;

enum class CeImpl : uint8_t { A = 0, B1 = 1, B1_5 = 2, B2 = 3 };

struct CeShape {
  int64_t T;          // rows (tokens) in the call
  int D, V;
  int chunk_rows;     // resolved exec.ce_chunk_rows: smallest 8192*2^i with head-grad RMW <= 0.5% of the step, capped
                      // by fit (130m/300m 32768, 520m/1.2B/pls 16384); strict preset 8192
  DType logits_store; // bf16 or f32 (Numerics::ce_logits_store)
  DType io = DType::bf16;   // bf16, or f32 for implementation A under the f32 policy
};

struct CeArgs {
  const void* x;             // [T, D]
  const void* w;             // [V, D] (OutputHead may pass a gathered W_P for sampled softmax, V = P)
  const int32_t* targets;    // [T] (remapped to candidate positions for sampled softmax)
  const float* weight;       // [T] 0/1 (last position of a row, padded rows and masked labels are 0)
  const float* grad_scale;   // device scalar: readout weight / (B_global * (S - 1))
  const float* logit_bias;   // [V] or nullptr (composed / biased heads)
  float* loss_tok;           // [T] CE (0 where weight 0)
  float* lse;                // [T] (exposed to LossTerms through HeadOutputs)
  float* loss_partial;       // [ceil(T/chunk_rows)] f32 sum of weight*CE per chunk (fixed-order reduction later)
  void* dx;                  // [T, D]; nullptr: no dX GEMM (eval, probes, detached backbone, before late start)
  bool accumulate_dx;        // dx += (readout heads add into the residual-gradient buffer of their layer)
  float* dw;                 // [V, D] f32 accumulator; nullptr: no head gradient (frozen or detached head)
  float* dw_microbatch;      // CeDwRound::PerChunkBf16: shared per-microbatch f32 accumulator [V, D] (Microbatch
                             // region); rounded to bf16 and added to dw by ce_finish_microbatch()
  CeDwRound dw_round;
  bool forward_only;         // eval: GEMM + statistics only (no store, no gradients)
};

uint64_t ce_workspace_bytes(CeImpl impl, const CeShape& s);  // B1 at 16384 rows, bf16 store: 4,334,288,896 B
void fused_linear_ce(CeImpl impl, Rounding r, const CeShape& s, const CeArgs& a, void* ws, GemmCache& gemm,
                     cudaStream_t st);
// Strict mode only: dw += f32(bf16(dw_microbatch)); dw_microbatch = 0.
void ce_finish_microbatch(const CeShape& s, const CeArgs& a, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ readout heads (pls)
// One readout: the readout state after layer `layer` (Block::readout_state; the residual by default), its own RMSNorm
// (separate heads) or the final norm (shared), its own head W (separate) or lm_head (shared). Called from the backward
// schedule when the backward reaches `layer`, so the CE workspace is reused by every head and memory does not grow
// with the number of heads. dW accumulates in place in the head's f32 gradient; d(state) goes to
// Block::readout_state_bwd. A readout with a static train_backbone == false (probe, monitor, detached backbone, before
// readout.on) runs with dx = nullptr and the readout norm backward computes only its gain gradient.
struct ReadoutCall {
  int layer;
  const bf16_t* state;           // [T, D] readout state
  const bf16_t* norm_gain;       // bf16 copy of the readout norm gain
  float* norm_rstd_ws;           // [T]
  bf16_t* norm_out_ws;           // [T, D] (recomputed per call; Transient)
  const bf16_t* w;               // [V, D] (waits on its ZeRO-1 all-gather event)
  float* dw;                     // [V, D] f32 grad (nullptr if detached head)
  float* dgain_partial;          // [n_blocks, D]
  const float* grad_scale;       // device scalar, includes the scheduled readout weight (readout.off, shape)
  bf16_t* d_state;               // [T, D] gradient w.r.t. the readout state (nullptr: no backbone gradient)
  int visible_dims;              // readout.visible_frac * D (coordinates beyond are zeroed in the readout input)
};
void run_readout(CeImpl impl, Rounding r, const CeShape& s, const ReadoutCall& c, const CeArgs& base, void* ws,
                 GemmCache& gemm, cudaStream_t st);

}  // namespace nm
