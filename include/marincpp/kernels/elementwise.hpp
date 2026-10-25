// Memory-bound kernels: document metadata, embedding, RMSNorm (+ sandwich tails), QK-norm + RoPE, SwiGLU, and the
// dq postprocess fused with q-norm/RoPE backward. DESIGN.md §3.1-§3.6.
// Every kernel has a Rounding argument (core.hpp) and an f32-storage instantiation linked into the main library (f32
// compute policy, gate V1, tight gradient checks). The stage-1 schedule fuses several of these into GEMM epilogues
// (gemm.hpp Epilogue); the kernels here remain the unfused reference path (exec.fused=false), and in levanter mode the
// fused outputs must equal them bitwise. Levanter-mode units build with --fmad=false and explicit contractions.
//
// Conventions: T = tokens in the microbatch (rows * S), row-major activations [T, width], bf16 storage, f32 math.
// "partial" outputs are per-CTA partial sums [n_blocks, width] f32, reduced later in a fixed order (reduce.hpp), so
// every reduction is deterministic.
#pragma once

#include <cstdint>

#include "marincpp/core.hpp"

namespace nm {

// ------------------------------------------------------------------------------------------------ microbatch metadata
// k_doc_meta: one CTA per row. From the int32 tokens of the microbatch rows it writes
//   targets[t]  = tokens[t + 1] within the row; the last position of the row takes tokens[row start] (levanter's roll)
//   weight[t]   = 0 at the last position of every row and on padded eval rows, else 1 (f32)
//   cu_seqlens  = packed segment starts over the microbatch, where a new segment starts after every EOS token
//                 (data.md §3.1: segment ids from the EOS value) and at every row start; entries past the real count
//                 repeat the total (the varlen forward and the persistent backward read *n_segments; no padding rows)
//   row_seg_start[r][i] = start of segment i of row r within the row, then S (doc_causal attention)
//   pos[t]      = t mod S (RoPE positions are absolute window positions, never reset at documents)
// and sets *overflow if a row has more than max_segs_per_seq segments (summed across ranks by the metrics reduction:
// fatal at the next metrics copy).
struct DocMetaArgs {
  const int32_t* tokens;      // [rows, S]
  const uint8_t* row_valid;   // [rows] or nullptr (training: all valid)
  int rows, S;
  int32_t eos_id;             // 128001
  int max_segs_per_seq;       // from DatasetMeta (17 for fineweb-edu-10B at S=4096; 291 for Paloma)
  int32_t* targets;           // [T]
  float* weight;              // [T]
  int32_t* pos;               // [T]
  int32_t* cu_seqlens;        // [rows*max_segs_per_seq + 1]
  int32_t* n_segments;        // device scalar
  int32_t* row_seg_start;     // [rows, max_segs_per_seq + 1]
  uint32_t* overflow;         // device flag
};
void doc_meta(const DocMetaArgs& a, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ embedding
// k_embed_fwd: out[t, :] = table_bf16[tokens[t], :]  (bf16 gather, x1.0; model.md §2)
void embed_fwd(const bf16_t* table, const int32_t* tokens, bf16_t* out, int64_t T, int D, cudaStream_t s);

// Deterministic backward (numerics.md §9.1 item 1 replaced by a sorted segment sum; DESIGN.md §3.2):
//   1. cub::DeviceRadixSort::SortPairs over (token id, position) pairs, ceil(log2 V) key bits (17 at V = 128,256;
//      24 for 12.8M-row tables); stable, so each token's positions ascend.
//   2. EmbedAccum::F32 (k_embed_chunks + k_embed_merge): fixed chunks of 256 sorted entries; one warp per (chunk,
//      256-column slice) sums each run piece of its chunk in ascending position order in f32. A run inside one chunk
//      is added to dtable[token] at once; the pieces of a run that crosses chunks go to per-chunk head / tail partials
//      (2 x ceil(T/256) x D f32, in cub_temp) and are merged in chunk order by the chunk where the run starts.
//      EmbedAccum::Bf16Sequential (k_embed_seq; strict preset): a bf16 running sum in ascending position order
//      (levanter's precision), one warp per (run, 64-column slice), then added to dtable[token] in f32.
//   dtable accumulates across microbatches in microbatch order; touched_rows[token] = 1 for every token present (the
//   mask for lazy updates). Measured at T = 131,072, D = 512, sort included, A100: 0.51 ms (F32), 0.62-1.0 ms
//   (Bf16Sequential, uniform to Zipf tokens; one token repeated T times: 28 ms, the sequential critical path).
// No atomics; bitwise repeatable. Rows of tokens absent from the batch are untouched (Adam still moves them unless the
// row partition is lazy).
struct EmbedBwdWorkspace {
  void* cub_temp; size_t cub_bytes;       // CUB temp storage, then the F32 chunk partials (D <= 4096 in the sizing)
  uint32_t* keys_in; uint32_t* keys_out;  // [T]
  int32_t* vals_in; int32_t* vals_out;    // [T]
  int32_t* run_start;                      // unused, null (no run list is built); to be removed with its callers
  int32_t* n_runs;                         // unused, null
};
uint64_t embed_bwd_workspace_bytes(int64_t T, int V);
void embed_bwd_sorted(const bf16_t* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                      EmbedAccum acc, const EmbedBwdWorkspace& ws, uint8_t* touched_rows /* [V] or nullptr */,
                      cudaStream_t s);

// ------------------------------------------------------------------------------------------------ RMSNorm family
// Levanter rounding (numerics.md §3.2):  y = bf16(x32 * rsqrt(mean(x32^2) + eps));  out = bf16(g_bf16 * y)
//   backward:  a = bf16(g_bf16 * dy);  dg += sum_t f32(bf16(dy * y));  s = sum_d x32*f32(a);
//              c = (s * (-0.5 * (r / (ms + eps)))) / n;  dx = bf16(f32(a)*r + c*(2*x32))
// Accurate: f32 throughout, out = bf16(g32 * x32 * r), dx rounded once.
struct RmsNormFwd {
  const bf16_t* x; const bf16_t* gain;   // gain = bf16 copy of the f32 parameter
  bf16_t* y; float* rstd;                // y may be nullptr (remat=min: only rstd kept, y recomputed)
  int64_t T; int D; float eps;
};
void rmsnorm_fwd(const RmsNormFwd& a, Rounding r, cudaStream_t s);                      // k_rmsnorm_fwd

// dx_out = dres (+) norm_bwd(dy): the first layer's input norm and the final norm use this form.
struct RmsNormBwdAdd {
  const bf16_t* dy; const bf16_t* x; const bf16_t* gain; const float* rstd;
  const bf16_t* dres;                    // residual gradient to add (nullptr = none)
  bf16_t* dx; float* dgain_partial;      // [n_blocks, D]
  int64_t T; int D; float eps;
};
void rmsnorm_bwd_add(const RmsNormBwdAdd& a, Rounding r, cudaStream_t s);              // k_rmsnorm_bwd_add

// Sandwich tail, fused (one read of the branch output and the residual, one write of each output):
//   y      = RMSNorm_post(branch)            (skipped when g_post == nullptr: pre-LN blocks)
//   x_out  = x_in + y                        (bf16 add: the residual stream is bf16, numerics.md §3.5)
//   a_next = RMSNorm_next(x_out)             (the next sublayer's pre-norm, or the final norm)
struct SandwichTailFwd {
  const bf16_t* branch; const bf16_t* x_in;
  const bf16_t* g_post; const bf16_t* g_next;
  bf16_t* x_out; bf16_t* a_next;          // a_next nullptr under remat=min (recomputed in the backward)
  float* rstd_post; float* rstd_next;
  int64_t T; int D; float eps;
};
void sandwich_tail_fwd(const SandwichTailFwd& a, Rounding r, cudaStream_t s);          // k_sandwich_tail_fwd

// Its backward:
//   dx_mid  = dx_out + norm_next_bwd(da_next)          (residual gradient through the next pre-norm)
//   dbranch = norm_post_bwd(dx_mid)                    (gradient of the branch output: o_proj or down_proj output)
//   dx_in   = dx_mid                                   (written to the ping-pong residual-gradient buffer)
struct SandwichTailBwd {
  const bf16_t* dx_out; const bf16_t* da_next;
  const bf16_t* x_out; const bf16_t* g_next; const float* rstd_next;
  const bf16_t* branch; const bf16_t* g_post; const float* rstd_post;
  bf16_t* dx_in; bf16_t* dbranch;
  float* dg_next_partial; float* dg_post_partial;
  int64_t T; int D; float eps;
};
void sandwich_tail_bwd(const SandwichTailBwd& a, Rounding r, cudaStream_t s);          // k_sandwich_tail_bwd

// ------------------------------------------------------------------------------------------------ QK-norm + RoPE
// Per (token, head): RMSNorm over hd with the layer's shared q_norm / k_norm gain, then RoPE (NeoX half split,
// rotate_half(x) = [-x2, x1]) with bf16 cos/sin tables at absolute position pos[t], then q *= q_scale.
// q_scale is applied as one multiply by the value passed; the caller chooses it (DESIGN.md §3.4).
// Levanter: every multiply/add rounds to bf16 (numerics.md §3.3-§3.4); q_scale = bf16(1/sqrt(hd)), multiplied in bf16
// (with bf16 storage it must be a bf16 value: checked).
// Accurate: f32 chain, one rounding; q_scale is one f32 multiply inside the chain. FA3 takes the scale under accurate,
// so the caller passes q_scale = 1 here and the whole scale to dq_post_qknorm_rope_bwd.
// f32 policy: the scale is f32(1/sqrt(hd)) (levanter's Python-float constant in the f32 compute dtype).
// The raw q, k (pre-norm) stay in the qkv buffer for the backward; outputs go to q_rot, k_rot ([T, H*hd], [T, KV*hd]).
struct QkNormRopeFwd {
  const bf16_t* qkv; int64_t qkv_ld;      // [T, (H+2KV)*hd]: q | k | v column blocks, GQA head order h = kv*G + g
  const bf16_t* gq; const bf16_t* gk;     // [hd]
  const bf16_t* cos_tab; const bf16_t* sin_tab;  // [S_max, hd]
  const int32_t* pos;                     // [T]
  bf16_t* q_rot; bf16_t* k_rot;           // nullptr under remat=min in the backward-only recompute call
  float* rstd_q; float* rstd_k;           // [T, H], [T, KV]
  int64_t T; int H, KV, hd; float eps; float q_scale;
};
void qknorm_rope_fwd(const QkNormRopeFwd& a, Rounding r, cudaStream_t s);              // k_qknorm_rope_fwd

// Backward: dq' (from attention) -> * q_scale (bf16, levanter: dq = bf16(dq_scaled * scale)) -> inverse rotation ->
// norm backward with the saved rstd -> dq written into the q columns of dqkv; same for k (no scale); dv is already in
// place (attention writes dv into the v columns of dqkv).
struct QkNormRopeBwd {
  const bf16_t* dq_rot; const bf16_t* dk_rot;
  const bf16_t* qkv; int64_t qkv_ld;
  const bf16_t* gq; const bf16_t* gk; const bf16_t* cos_tab; const bf16_t* sin_tab; const int32_t* pos;
  const float* rstd_q; const float* rstd_k;
  bf16_t* dqkv; int64_t dqkv_ld;
  float* dgq_partial; float* dgk_partial;  // [n_blocks, hd]
  int64_t T; int H, KV, hd; float eps; float q_scale;
};
void qknorm_rope_bwd(const QkNormRopeBwd& a, Rounding r, cudaStream_t s);              // k_qknorm_rope_bwd

// Stage-1 fusion of FA3's dq postprocess with the q side of the backward above: reads FA3's f32 dq accumulator
// [H, T, hd], rounds it as FA3's postprocess would (softmax_scale 1 under levanter), then dq *= q_scale in bf16, inverse
// rotation, q-norm backward, and writes the q columns of dqkv. Saves the bf16 dq round trip and one launch per layer.
// The k side runs in FA3's dK epilogue (or in qknorm_rope_bwd with dq_rot = nullptr).
struct DqPostQkNormRopeBwd {
  const float* dq_accum;                   // [H, T, hd] f32 (FA3 layout)
  const bf16_t* qkv; int64_t qkv_ld;
  const bf16_t* gq; const bf16_t* cos_tab; const bf16_t* sin_tab; const int32_t* pos;
  const float* rstd_q;
  bf16_t* dqkv; int64_t dqkv_ld;
  float* dgq_partial;
  int64_t T; int H, hd; float eps; float q_scale;
};
void dq_post_qknorm_rope_bwd(const DqPostQkNormRopeBwd& a, Rounding r, cudaStream_t s);  // k_dq_post_qknorm_rope_bwd

// Host: f32 inv_freq built as numerics.md §3.3 (k = 0,2,..,hd-2; e = k/hd in f32; p = theta^e in f32; 1/p in f32),
// angle = f32(f32(pos) * inv_freq), cos/sin of the f32 angle evaluated in double and rounded to f32, then to bf16.
// Test T1.3 compares the tables bitwise with levanter's; on a mismatch the run loads the levanter tables (O4).
void build_rope_tables(int S_max, int hd, double theta, bf16_t* cos_host, bf16_t* sin_host, float* inv_freq_host);

// ------------------------------------------------------------------------------------------------ SwiGLU
// gu = gate and up outputs of one GEMM ([T, 2F]); the Wgu rows are interleaved in 64-row blocks (gate block, up block,
// ...) so a GEMM tile holds matching gate and up columns (fused epilogue); these kernels read the same layout.
// h = silu(gate) * up.
// Levanter (numerics.md §3.5): s = bf16(logistic(g)); silu = bf16(g*s); h = bf16(silu*u);
//   backward: s; sp = bf16(s*bf16(1-s)); dsilu = dh*u; du = dh*silu; dg = bf16(bf16(dsilu*s) + bf16(bf16(g*dsilu)*sp))
//   (every op bf16); Accurate: f32, dg = dsilu * s * (1 + g*(1-s)) rounded once.
void swiglu_fwd(const bf16_t* gu, bf16_t* h, int64_t T, int F, Rounding r, cudaStream_t s);          // k_swiglu_fwd
void swiglu_bwd(const bf16_t* dh, const bf16_t* gu, bf16_t* dgu, int64_t T, int F, Rounding r,
                cudaStream_t s);                                                                    // k_swiglu_bwd

// ------------------------------------------------------------------------------------------------ casts
// Multi-tensor f32 -> bf16 (round to nearest even) for parameters whose optimizer did not already write the bf16 copy.
struct CastList { const float* const* src; bf16_t* const* dst; const int64_t* numel; int n; };
void cast_f32_bf16_multi(const CastList& l, cudaStream_t s);                                        // k_cast_multi

}  // namespace nm
