// build_plan: every buffer of one rank (DESIGN.md §2.2-§2.5). The sizes are the planner formulas of
// scripts/design_budget.py detailed(), buffer by buffer; test s0_plan_vs_budget compares them byte for byte.
//
// Owner shards (ZeRO-1) are the even split ceil(n / ranks) until the optimizer plan (S6, plan_ownership) assigns whole
// MuonH matrices and row shards; the plan then takes the largest rank's shard per group.
#include <algorithm>
#include <string>

#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"
#include "memory/internal.hpp"

namespace nm {
namespace {

struct Adder {
  MemoryPlan& p;
  void operator()(const std::string& name, const std::string& group, Lifetime life, DType t,
                  std::initializer_list<int64_t> shape, int layer = -1, Recompute rc = Recompute::Never) {
    BufferSpec b;
    b.name = name;
    b.group = group;
    b.life = life;
    b.dtype = t;
    b.ndim = static_cast<int>(shape.size());
    int i = 0;
    for (int64_t s : shape) b.shape[i++] = s;
    b.layer = layer;
    b.recompute = rc;
    p.add(std::move(b));
  }
};

}  // namespace

MemoryPlan build_plan(const Config& c, const ModelShape& m, const ExecShape& e, const DatasetMeta& d) {
  (void)d;  // max_segs_per_seq already in ExecShape
  MemoryPlan plan;
  Adder add{plan};
  const ParamCounts p = param_counts(c, m);
  const int64_t D = m.D, H = m.H, KV = m.KV, hd = m.hd, F = m.F, V = m.V, S = m.S;
  const int64_t T = e.micro_tokens;
  const int64_t r = e.ranks;
  auto shard = [&](int64_t n) { return e.zero1 ? (n + r - 1) / r : n; };
  const auto P = Lifetime::Persistent, MB = Lifetime::Microbatch, A = Lifetime::Activation, TR = Lifetime::Transient,
             W = Lifetime::Workspace;
  const DType f32 = DType::f32, bf16 = DType::bf16, i32 = DType::i32, u32 = DType::u32, u8 = DType::u8;

  // ---- persistent
  add("param.f32.muonh", "param.f32.muonh", P, f32, {shard(p.muon)});
  add("param.f32.adam", "param.f32.adam", P, f32, {shard(p.emb + p.norms)});
  if (p.heads) add("param.f32.adamh", "param.f32.adamh", P, f32, {shard(p.heads)});
  add("param.bf16", "param.bf16", P, bf16, {p.total});
  add("grad.f32", "grad.f32", P, f32, {p.total});
  add("muonh.momentum.f32", "muonh.momentum", P, f32, {shard(p.muon)});
  add("adam.mu.f32", "adam.mu_nu", P, f32, {shard(p.emb + p.norms)});
  add("adam.nu.f32", "adam.mu_nu", P, f32, {shard(p.emb + p.norms)});
  if (p.heads) {
    add("adamh.mu.f32", "adamh.mu_nu", P, f32, {shard(p.heads)});
    add("adamh.nu.f32", "adamh.mu_nu", P, f32, {shard(p.heads)});
  }
  {
    // X, X', A, B for a batch of 4 matrices of the largest oriented shape (DESIGN.md §4.2)
    const int64_t mats[3][2] = {{D, H * hd}, {KV * hd, D}, {D, F}};
    int64_t mxn = 0, mm = 0;
    for (auto& s : mats) {
      mxn = std::max(mxn, s[0] * s[1]);
      mm = std::max(mm, std::min(s[0], s[1]) * std::min(s[0], s[1]));
    }
    add("ns.workspace.f32", "ns.workspace", P, f32, {4 * (2 * mxn + 2 * mm)});
  }
  const int64_t N = c.optim_num_steps;
  for (const char* t : {"lr", "adam_lr", "bc1", "bc2"}) add(std::string("sched.") + t, "sched.tables", P, f32, {N});
  add("rope.cos", "rope", P, bf16, {S, hd});
  add("rope.sin", "rope", P, bf16, {S, hd});
  add("data.tokens.0", "data.tokens", P, i32, {e.seqs_per_rank, S});
  add("data.tokens.1", "data.tokens", P, i32, {e.seqs_per_rank, S});
  add("metrics.ring", "metrics.ring", P, u8, {int64_t(64) << 20});

  // ---- microbatch
  add("mb.targets", "mb.targets_pos", MB, i32, {T});
  add("mb.pos", "mb.targets_pos", MB, i32, {T});
  add("mb.seg_table", "mb.seg_table", MB, i32, {int64_t(e.micro_seqs) * e.max_segs_per_seq + 1});
  if (e.num.ce_dw_round == CeDwRound::PerChunkBf16)
    add("mb.ce_dw_accum", "mb.ce_dw_accum", MB, f32, {V, D});  // strict: per-microbatch head-gradient accumulator

  // ---- activations, per layer (DESIGN.md §2.3 table; remat=min drops the Elementwise rows, remat=layer keeps x_in)
  struct Act { const char* name; DType t; int64_t cols; Recompute rc; };
  const Act acts[] = {
      {"x_in", bf16, D, Recompute::Never},          {"a", bf16, D, Recompute::Elementwise},
      {"qkv", bf16, (H + 2 * KV) * hd, Recompute::Never}, {"qk_rot", bf16, (H + KV) * hd, Recompute::Elementwise},
      {"attn_o", bf16, H * hd, Recompute::Never},   {"oproj", bf16, D, Recompute::Never},
      {"x_mid", bf16, D, Recompute::Never},         {"m", bf16, D, Recompute::Elementwise},
      {"gu", bf16, 2 * F, Recompute::Never},        {"h", bf16, F, Recompute::Elementwise},
      {"y", bf16, D, Recompute::Never},
      {"f32_stats", f32, 4 + 2 * H + KV, Recompute::Never},  // 4 norm rstd, q/k-norm rstd (H + KV), lse (H)
  };
  int64_t full_per_token = 0;
  for (const Act& a : acts) full_per_token += a.cols * int64_t(dtype_size(a.t));
  for (int l = 0; l < m.L; ++l)
    for (const Act& a : acts) {
      if (e.remat == Remat::Min && a.rc == Recompute::Elementwise) continue;
      if (e.remat == Remat::Layer && std::string(a.name) != "x_in") continue;
      add("act.L" + std::to_string(l) + "." + a.name, std::string("act.L*.") + a.name, A, a.t, {T, a.cols}, l, a.rc);
    }
  add("act.final.x", "act.final", A, bf16, {T, D}, m.L);
  add("act.final.xf", "act.final", A, bf16, {T, D}, m.L);
  add("act.final.rstd", "act.final", A, f32, {T}, m.L);

  // ---- transients (one slot per role, shared by all layers)
  if (e.remat == Remat::Layer) add("tr.layer_recompute", "tr.layer_recompute", TR, u8, {full_per_token * T});
  if (m.n_readouts > 1) add("tr.head_norm_out", "tr.head_norm_out", TR, bf16, {T, D});
  add("tr.dx.0", "tr.dx", TR, bf16, {T, D});
  add("tr.dx.1", "tr.dx", TR, bf16, {T, D});
  add("tr.da", "tr.da_dm_do", TR, bf16, {T, D});
  add("tr.dqkv", "tr.dqkv", TR, bf16, {T, (H + 2 * KV) * hd});
  add("tr.dO", "tr.dO", TR, bf16, {T, H * hd});
  add("tr.dgu", "tr.dgu", TR, bf16, {T, 2 * F});
  add("tr.dh", "tr.dh", TR, bf16, {T, F});
  if (e.remat == Remat::Min)
    add("tr.recompute", "tr.recompute", TR, bf16, {T, std::max({D, F, (H + KV) * hd})});
  for (int i = 0; i < 4; ++i)
    add("tr.norm_gain_partials." + std::to_string(i), "tr.norm_gain_partials", TR, f32, {1024, D});

  // ---- workspaces (one slot per kernel family)
  add("ws.fa3.dq_accum", "ws.fa3.dq_accum", W, f32, {H, T, hd});
  if (H != KV) {
    add("ws.fa3.dk_accum", "ws.fa3.dkv_accum", W, f32, {KV, T, hd});
    add("ws.fa3.dv_accum", "ws.fa3.dkv_accum", W, f32, {KV, T, hd});
  }
  add("ws.fa3.softmax_d", "ws.fa3.stats", W, f32, {H, T});
  add("ws.fa3.lse_log2", "ws.fa3.stats", W, f32, {H, T});
  const int64_t rows = e.ce_chunk_rows;
  add("ws.ce.logits", "ws.ce.logits", W, e.num.ce_logits_store, {rows, V});
  add("ws.ce.tile_max", "ws.ce.row_stats", W, f32, {rows, V / 128 + 1});
  add("ws.ce.tile_sum", "ws.ce.row_stats", W, f32, {rows, V / 128 + 1});
  add("ws.ce.lse", "ws.ce.row_stats", W, f32, {rows});
  add("ws.ce.gscale", "ws.ce.row_stats", W, f32, {rows});
  for (const char* t : {"keys_in", "keys_out", "idx_in", "idx_out"})
    add(std::string("ws.embed_bwd.") + t, "ws.embed_bwd", W, u32, {T});
  add("ws.embed_bwd.cub_temp", "ws.embed_bwd", W, u8, {int64_t(32) << 20});
  add("ws.cublaslt.0", "ws.cublaslt", W, u8, {int64_t(64) << 20});
  add("ws.cublaslt.1", "ws.cublaslt", W, u8, {int64_t(64) << 20});
  return plan;
}

}  // namespace nm
