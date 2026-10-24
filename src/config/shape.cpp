// Derived shapes and optimizer constants (config.hpp). Gate T0.8 checks n_params and flops_per_example; T0.6 checks
// optim_constants.
#include <cmath>

#include "marincpp/config.hpp"
#include "marincpp/data.hpp"

namespace nm {

ModelShape model_shape(const Config& c) {
  ModelShape m{};
  m.V = c.model_vocab;
  m.D = c.model_hidden;
  m.L = c.model_layers;
  m.H = c.model_heads;
  m.KV = c.model_kv_heads;
  NM_REQUIRE(m.H > 0 && m.KV > 0 && m.H % m.KV == 0, "model.heads must be a positive multiple of model.kv_heads");
  m.hd = c.model_head_dim ? c.model_head_dim : m.D / m.H;
  m.F = c.model_mlp;
  m.S = c.model_seq_len;
  m.G = m.H / m.KV;
  m.qkv_cols = (m.H + 2 * m.KV) * m.hd;
  const bool separate = c.readout_kind == "separate";
  m.n_readouts = separate ? m.L : 1;                       // L-1 intermediate heads + the final head
  m.n_ce_heads = c.readout_kind == "none" ? 1 : m.L;
  const int64_t D = m.D, V = m.V, H = m.H, KV = m.KV, hd = m.hd, F = m.F;
  // model.md §5: attention + MLP + 4 block norms + q/k norms per layer; final norm; embedding; heads (untied) and the
  // norms of separate readouts.
  const int64_t per_layer = D * H * hd + 2 * D * KV * hd + H * hd * D + 3 * D * F + 4 * D + 2 * hd;
  const int64_t heads = (c.model_tie_embeddings && m.n_readouts == 1) ? 0 : int64_t(m.n_readouts) * V * D;
  m.n_params = m.L * per_layer + D + int64_t(m.n_readouts - 1) * D + V * D + heads;
  // model.md §6, levanter's lm_flops_per_token(glu=True) with hd' = D/H as a Python float, same operation order.
  const double hdf = double(m.D) / double(m.H);
  const int64_t S = m.S, L = m.L;
  const double fpt = double(L * (2 * 3 * D * F + 2 * D * D)) +
                     double(L) * (double(2 * D) * (double(H) * hdf + double(2 * KV) * hdf) + double(2 * S * H) * hdf +
                                  double(3 * S * H) + double(2 * S) * hdf * double(H)) +
                     double(2 * D * V * m.n_ce_heads);
  m.flops_per_example = 3.0 * fpt * double(S);
  // class M: bf16(1/sqrt(hd)), rounded once from the f64 value through f32
  m.attn_scale_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(static_cast<float>(1.0 / std::sqrt(double(m.hd)))));
  return m;
}

ExecShape exec_shape(const Config& c, const ModelShape& m, const DatasetMeta* d) {
  ExecShape e{};
  e.ranks = c.exec_gpus;
  NM_REQUIRE(c.data_global_batch % e.ranks == 0, "data.global_batch must be a multiple of exec.gpus");
  e.seqs_per_rank = c.data_global_batch / e.ranks;
  NM_REQUIRE(c.exec_micro_seqs > 0, "exec.micro_seqs is unresolved (run choose_plan and resolve first)");
  NM_REQUIRE(e.seqs_per_rank % c.exec_micro_seqs == 0, "exec.micro_seqs must divide the sequences per rank");
  e.micro_seqs = c.exec_micro_seqs;
  e.n_micro = e.seqs_per_rank / e.micro_seqs;
  e.micro_tokens = int64_t(e.micro_seqs) * m.S;
  e.max_segs_per_seq = d ? d->max_segs_per_seq : kDefaultMaxSegsPerSeq;
  NM_REQUIRE(c.exec_ce_chunk_rows > 0, "exec.ce_chunk_rows is unresolved");
  e.ce_chunk_rows = c.exec_ce_chunk_rows;
  NM_REQUIRE(c.optim_zero1 != "auto" && c.optim_zero2_vocab != "auto", "optim.zero1 / optim.zero2_vocab unresolved");
  e.zero1 = c.optim_zero1 == "on";
  e.zero2_vocab = c.optim_zero2_vocab == "on";
  NM_REQUIRE(c.exec_remat != "auto", "exec.remat is unresolved");
  e.remat = c.exec_remat == "all" ? Remat::All : c.exec_remat == "min" ? Remat::Min : Remat::Layer;
  e.attn_bwd = c.exec_attn_bwd_sched == "persistent"   ? AttnBwdSched::Persistent
               : c.exec_attn_bwd_sched == "doc_causal" ? AttnBwdSched::DocCausal
                                                       : AttnBwdSched::Undecided;
  e.num.policy = c.exec_policy == "f32" ? ComputePolicy::F32 : ComputePolicy::BF16;
  e.num.rounding = c.exec_rounding == "accurate" ? Rounding::Accurate : Rounding::Levanter;
  e.num.ce_logits_store = c.exec_ce_logits == "f32" ? DType::f32 : DType::bf16;
  e.num.embed_accum = c.exec_embed_accum == "bf16_seq" ? EmbedAccum::Bf16Sequential : EmbedAccum::F32;
  e.num.ce_dw_round = c.exec_ce_dw_round == "chunk" ? CeDwRound::PerChunkBf16 : CeDwRound::None;
  e.num.grad_reduce = c.exec_grad_reduce == "bf16" ? DType::bf16 : DType::f32;
  e.fused = c.exec_fused;
  return e;
}

OptimConstants optim_constants(const Config& c) {
  NM_REQUIRE(c.optim_ns_coeffs == "simple", "only optim.ns_coeffs = simple is implemented");
  OptimConstants k{};
  // optim.md §5.3 rule 1: Python float expressions in f64, one rounding to f32.
  k.b1 = static_cast<float>(c.optim_beta1);
  k.b2 = static_cast<float>(c.optim_beta2);
  k.one_minus_b1 = static_cast<float>(1.0 - c.optim_beta1);
  k.one_minus_b2 = static_cast<float>(1.0 - c.optim_beta2);
  k.momentum = static_cast<float>(c.optim_momentum);
  k.eps = static_cast<float>(c.optim_eps);
  k.muon_eps = static_cast<float>(c.optim_muon_eps);
  k.max_grad_norm = static_cast<float>(c.optim_max_grad_norm);
  k.ns_a = static_cast<float>(3.4445);
  k.ns_b = static_cast<float>(-4.7750);
  k.ns_c = static_cast<float>(2.0315);
  k.norm_floor = static_cast<float>(1e-10);
  k.lr_peak = static_cast<float>(c.optim_lr);
  k.adam_lr_peak = static_cast<float>(c.optim_adam_lr);
  k.warmup_init_minus_end = static_cast<float>(0.0 - c.optim_lr);
  k.adam_warmup_init_minus_end = static_cast<float>(0.0 - c.optim_adam_lr);
  k.cos_one_minus_alpha = static_cast<float>(1.0 - c.optim_min_lr_ratio);
  k.pi = static_cast<float>(3.141592653589793);
  k.half = 0.5f;
  return k;
}

}  // namespace nm
