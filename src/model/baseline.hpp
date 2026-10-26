// Stage S7: the baseline model (Qwen3 sandwich blocks, embedding input, untied fused-CE head) as one rank records it:
// parameter table in marincpp's storage layout, buffer declarations, and the forward + hand-written backward of one
// microbatch built from the S3 (elementwise), S4 (GEMM, attention) and S5 (fused CE) kernels. DESIGN.md §3.1 (the
// unfused schedule; the fused epilogues are S8), §3.2-§3.8; model.md §2 (order of operations).
//
// Everything is recorded on one stream with no host synchronization, so the same calls are captured into g_step.
// Parameter gradients accumulate (+=) in f32 across microbatches; the caller zeroes them once per update.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/core.hpp"
#include "marincpp/kernels/attention.hpp"
#include "marincpp/kernels/cross_entropy.hpp"
#include "marincpp/kernels/gemm.hpp"
#include "marincpp/memory.hpp"
#include "marincpp/model.hpp"

namespace nm::model {

struct BaselineSpec {
  ModelShape m;
  int micro_rows = 32;            // sequences per microbatch on this rank
  int global_rows = 128;          // sequences per update over all ranks (sets the loss scale)
  int max_segs = 24;              // segment-table width per row (dataset meta)
  int ce_chunk = 8192;
  Numerics num;
  AttnBackendKind attn = AttnBackendKind::FA3;
  AttnBwdSched attn_sched = AttnBwdSched::DocCausal;
  bool deterministic = true;
  CeImpl ce = CeImpl::B1;
  int32_t eos_id = 128001;
};

// Storage tensors in canonical order: per layer Wqkv (q | k | v row blocks), Wo, Wgu (gate / up interleaved in 64-row
// blocks, the layout the SwiGLU kernels read), Wdown, the four block gains, q_norm, k_norm; then final_norm, embedding,
// lm_head. Leaves carry the levanter names (one MuonH matrix per leaf).
std::vector<ParamSpec> baseline_params(const ModelShape& m);

class Baseline {
 public:
  explicit Baseline(const BaselineSpec& s);
  ~Baseline();
  Baseline(const Baseline&) = delete;
  Baseline& operator=(const Baseline&) = delete;

  const BaselineSpec& spec() const { return s_; }
  const std::vector<ParamSpec>& params() const { return params_; }
  int64_t offset(int tensor) const { return offsets_[tensor]; }   // elements, into param.* and grad.f32
  int64_t numel(int tensor) const;
  int64_t total() const { return total_; }
  int find(const std::string& storage_name) const;                // fatal if absent
  int64_t T() const { return int64_t(s_.micro_rows) * s_.m.S; }

  // 1. GEMM shapes: plan them in `gemm`, which the caller then searches (choose_all) before declare().
  void plan_gemms(GemmCache& gemm);
  size_t gemm_search_scratch() const { return gemm_scratch_; }   // scratch choose_all needs for these shapes
  // 2. Every buffer of the model, workspaces sized from the kernels; `gemm_ws` = gemm.workspace_bytes().
  void declare(MemoryPlan& plan, uint64_t gemm_ws) const;
  // 3. Pointers into the arena; uploads the RoPE tables and the loss scale (synchronous: before capture).
  void bind(const Arena& arena, GemmCache& gemm, cudaStream_t s);

  // State buffers (full tensors on this rank).
  float* p32() const { return p32_; }
  bf16_t* p16() const { return p16_; }
  float* grad() const { return grad_; }
  int32_t* tokens() const { return tokens_; }   // [micro_rows, S] staging for the microbatch being recorded

  void zero_grads(cudaStream_t s) const;
  // Records forward + backward of the microbatch in tokens(). Its weighted loss sum goes to loss slot `index`
  // (slots: max_micro, zeroed by zero_grads).
  void record_microbatch(int index, cudaStream_t s) { record_microbatch(index, tokens_, s); }
  // The same, reading the microbatch rows from `tokens` ([micro_rows, S], device): a trainer stages a whole step's rows
  // once and points each microbatch into them (the pointers are fixed for the life of a captured graph).
  void record_microbatch(int index, const int32_t* tokens, cudaStream_t s);
  const float* loss_slots() const { return loss_mb_; }
  static constexpr int kMaxMicro = 256;
  float loss_scale() const;     // 1 / (global_rows * (S - 1))

 private:
  struct L;                     // per-layer pointers
  void* bind_buf(const Arena& a, const std::string& name, DType t, std::initializer_list<int64_t> shape) const;
  bf16_t* w16(int tensor) const { return p16_ + offsets_[tensor]; }
  float* g32(int tensor) const { return grad_ + offsets_[tensor]; }
  void reduce_gain(const float* partial, int tensor, int width, cudaStream_t s) const;

  BaselineSpec s_;
  std::vector<ParamSpec> params_;
  std::vector<int64_t> offsets_;
  int64_t total_ = 0;
  std::unique_ptr<AttentionBackend> attn_;
  AttnShape attn_shape_{};
  CeShape ce_shape_{};
  GemmCache* gemm_ = nullptr;
  int id_fwd_[4] = {}, id_dgrad_[4] = {}, id_wgrad_[4] = {};   // qkv, o, gu, down
  std::vector<L> layers_;
  int t_final_ = -1, t_embed_ = -1, t_head_ = -1;
  int64_t n_part_ = 0;          // gain-partial rows for T tokens
  size_t gemm_scratch_ = 0;

  float *p32_ = nullptr, *grad_ = nullptr;
  bf16_t* p16_ = nullptr;
  int32_t* tokens_ = nullptr;
  int32_t *targets_ = nullptr, *pos_ = nullptr, *cu_seqlens_ = nullptr, *n_segs_ = nullptr, *row_seg_ = nullptr;
  uint32_t* overflow_ = nullptr;
  float* weight_ = nullptr;
  bf16_t *rope_cos_ = nullptr, *rope_sin_ = nullptr;
  bf16_t *fin_x_ = nullptr, *fin_xf_ = nullptr;
  float* fin_rstd_ = nullptr;
  // transients
  bf16_t *dx_ = nullptr, *dA_ = nullptr, *dxf_ = nullptr, *dy_ = nullptr, *dh_ = nullptr, *dgu_ = nullptr,
         *dm_ = nullptr, *doproj_ = nullptr, *dattn_ = nullptr, *dq_ = nullptr, *dk_ = nullptr, *dqkv_ = nullptr,
         *zeros_ = nullptr;
  float *part_a_ = nullptr, *part_b_ = nullptr, *part_q_ = nullptr, *part_k_ = nullptr;
  float *loss_tok_ = nullptr, *ce_lse_ = nullptr, *loss_part_ = nullptr, *loss_mb_ = nullptr, *gscale_ = nullptr;
  void *ws_attn_ = nullptr, *ws_ce_ = nullptr, *ws_embed_ = nullptr, *ws_gemm_ = nullptr;
  uint64_t ws_embed_bytes_ = 0;
};

}  // namespace nm::model
