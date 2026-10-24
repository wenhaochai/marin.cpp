// Attention: causal AND same-document, any sequence length, hd 64 (MHA/GQA) and hd 128 (GQA). DESIGN.md §3.4.
//
// Backends behind one interface:
//   FA3        vendored FlashAttention-3 hopper/ (third_party/fa3-hopper @767386d + marincpp patches), torch-free
//              entry points. Forward: varlen with VarlenDynamicPersistentTileScheduler. Backward: NEVER FA3's
//              SingleTileBwdLPTScheduler over a padded batch (non-persistent grid ceil(max_seqlen/kBlockN) x h x
//              num_batch with host-fixed num_batch: ~95% empty CTAs at rows x 24 segments). One of:
//                Persistent  varlen persistent backward scheduler: grid = SM count; a prepare kernel in the graph reads
//                            cu_seqlens and the device segment count and builds the LPT-ordered work list.
//                DocCausal   patch: batch = rows, per-row segment-start table; tiles outside a segment skipped,
//                            boundary tiles masked by segment id; no padding rows; grid (S/128) x H x rows.
//              (exec.attn_bwd_sched; decided by E1b, DESIGN.md O1.) f32 softmax statistics.
//   CudnnSdpa  cuDNN 9.19 SDPA through cudnn-frontend 1.x with THD ragged offsets; graph built at plan time.
//   LevRef     levanter's numerics exactly (bf16 running max/sum/accumulator, block 1024, ascending key blocks,
//              bf16(-1e10) fill, L stored f32 with bf16 precision, model.md §2.3): levanter_strict preset and tests.
//   RefF32     f32 math, block 1024, ascending key blocks, f32 running statistics: the f32 compute policy and tests.
#pragma once

#include <cstdint>
#include <memory>

#include "marincpp/config.hpp"
#include "marincpp/core.hpp"

namespace nm {

struct DeviceContext;

enum class AttnBackendKind : uint8_t { FA3 = 0, CudnnSdpa = 1, LevRef = 2, RefF32 = 3 };

struct AttnShape {
  int64_t total_q;        // tokens in the microbatch (rows * S)
  int rows;               // sequences in the microbatch
  int S;                  // window length (max segment length)
  int H, KV, hd;
  int max_segs_per_seq;   // sizes the segment tables and the persistent scheduler's work list; never padding rows
  AttnBwdSched bwd_sched;
  bool deterministic;     // ordered dq accumulation; ordered GQA dk/dv accumulation with first-head store (no memset)
  DType io = DType::bf16; // bf16, or f32 for RefF32
};

struct AttnFwdArgs {
  const void* q; int64_t q_ld;       // [T, H*hd] (row stride q_ld elements); already normed, rotated, scaled (levanter)
  const void* k; int64_t k_ld;       // [T, KV*hd]
  const void* v; int64_t v_ld;       // [T, KV*hd] (the v columns of the qkv buffer: no copy)
  void* o; int64_t o_ld;             // [T, H*hd]
  float* lse;                        // [H, T] natural-log log-sum-exp, f32 (LevRef stores bf16 precision)
  const int32_t* cu_seqlens;         // [rows*max_segs_per_seq + 1] packed segment starts (varlen forward, Persistent)
  const int32_t* n_segs;             // device segment count (read by the Persistent backward scheduler's prepare kernel)
  const int32_t* row_seg_start;      // [rows, max_segs_per_seq + 1] per-row segment starts (DocCausal)
  float softmax_scale;               // Levanter rounding: 1.0 (q pre-scaled in bf16); Accurate: bf16(1/sqrt(hd))
  // Reserved for moda (depth keys and values of the same token inside the same softmax). nullptr until the FA3
  // extra_kv forward/backward patch exists; the knob value model.block=moda is not registered before then.
  const void* extra_k = nullptr; const void* extra_v = nullptr; int extra_per_token = 0;
};

struct AttnBwdArgs {
  AttnFwdArgs fwd;                   // the same inputs, plus the saved o and lse
  const void* d_o; int64_t do_ld;
  const float* softmax_d;            // [H, T] rowsum(dO*O) from the o_proj dgrad epilogue (stage-1 fusion), or
                                     // nullptr: the backend runs FA3's preprocess kernel itself
  float* dq_accum;                   // [H, T, hd] f32: consumed by k_dq_post_qknorm_rope_bwd (stage-1 fusion);
                                     // nullptr: the backend runs FA3's postprocess and writes dq
  void* dq; int64_t dq_ld;           // [T, H*hd] (only when dq_accum == nullptr)
  void* dk; int64_t dk_ld;           // [T, KV*hd]
  void* dv; int64_t dv_ld;           // written straight into the v columns of dqkv
};

class AttentionBackend {
 public:
  virtual ~AttentionBackend() = default;
  virtual AttnBackendKind kind() const = 0;
  // Bytes of scratch for one fwd+bwd at this shape. FA3: dq_accum f32 [H, T, hd] (DocCausal; Persistent adds FA3's
  // per-segment row padding), GQA dk/dv accumulators [KV, T, hd], softmax_d and lse_log2 [H, T], scheduler semaphores
  // and work lists. Every glue buffer of hopper/flash_api.cpp is an arena buffer; zeroing is a memset node.
  virtual uint64_t workspace_bytes(const AttnShape& s, bool training) const = 0;
  // Called once at graph-build time: chooses kernels, builds cuDNN graphs, records nothing on a stream.
  virtual void prepare(const AttnShape& s, bool training) = 0;
  virtual void fwd(const AttnShape& s, const AttnFwdArgs& a, void* ws, cudaStream_t st) = 0;
  virtual void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st) = 0;
};

std::unique_ptr<AttentionBackend> make_attention(AttnBackendKind k, const DeviceContext& ctx);

// FA3 entry points compiled from the vendored sources (src/kernels/attention/fa3_entry.cu; replaces
// hopper/flash_api.cpp, no torch). Instantiations: {fwd, bwd} x {hd64, hd128} x bf16 x sm90 x {varlen, doc_causal},
// packgqa fwd for GQA, the persistent backward scheduler.
namespace fa3 {
void fwd(const AttnShape& s, const AttnFwdArgs& a, void* ws, cudaStream_t st);
void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st);
uint64_t workspace_bytes(const AttnShape& s, bool training);
}  // namespace fa3

}  // namespace nm
