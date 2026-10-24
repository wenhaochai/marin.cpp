// choose_plan: the planner rules of DESIGN.md §2.3 (micro_seqs, remat) and §3.8 (CE chunk), memory.hpp.
#include <algorithm>
#include <cstdio>

#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"

namespace nm {

uint64_t mem_budget_bytes(const Config& c) {
  return static_cast<uint64_t>(c.exec_mem_budget_gib * 1073741824.0);
}

int ce_rows_floor(int64_t micro_tokens) {
  const int64_t rows = std::min<int64_t>(8192, micro_tokens);
  return static_cast<int>((rows + 127) / 128 * 128);
}

namespace {

const char* remat_text(Remat r) { return r == Remat::All ? "all" : r == Remat::Min ? "min" : "layer"; }

struct Planner {
  const Config& c;
  const ModelShape& m;
  const DatasetMeta& d;
  bool zero1, zero2;
  bool canaries;   // exec.debug_arena != off: the arena carries guard bands, and the fit test counts them

  ExecShape shape(int micro, Remat r, int ce_rows) const {
    Config k = c;
    k.exec_micro_seqs = micro;
    k.exec_remat = remat_text(r);
    k.exec_ce_chunk_rows = ce_rows;
    k.optim_zero1 = zero1 ? "on" : "off";
    k.optim_zero2_vocab = zero2 ? "on" : "off";
    if (k.exec_ce_logits == "auto") k.exec_ce_logits = "bf16";
    return exec_shape(k, m, &d);
  }
  // The CE-cap headroom of §3.8 is a payload rule (scripts/design_budget.py); the fit test is on the arena.
  uint64_t payload(int micro, Remat r, int ce_rows) const {
    return build_plan(c, m, shape(micro, r, ce_rows), d).payload_bytes();
  }
  uint64_t arena(int micro, Remat r, int ce_rows) const {
    MemoryPlan p = build_plan(c, m, shape(micro, r, ce_rows), d);
    p.finalize(UINT64_MAX, canaries);
    return p.device_bytes();
  }
  // CE chunk of the fit test: the requested chunk, else the smallest chunk the rule can choose.
  int fit_rows(int micro) const {
    return c.exec_ce_chunk_rows > 0 ? c.exec_ce_chunk_rows : ce_rows_floor(int64_t(micro) * m.S);
  }
};

CeStage ce_stage_of(const Config& c) {
  return c.exec_ce == "b1_5" ? CeStage::B1_5 : c.exec_ce == "b2" ? CeStage::B2 : CeStage::B1;
}
NsPrecision ns_of(const Config& c) {
  return c.optim_ns_precision == "f32" ? NsPrecision::F32
         : c.optim_ns_precision == "3xtf32" ? NsPrecision::X3TF32
                                            : NsPrecision::TF32;
}

}  // namespace

PlanChoice choose_plan(const Config& c, const ModelShape& m, const DatasetMeta& d, std::string* report) {
  NM_REQUIRE(c.exec_gpus > 0 && c.data_global_batch % c.exec_gpus == 0, "global batch must split over the GPUs");
  // ZeRO-2 for vocab-sized gradients is not implemented (memory.hpp PlanChoice): auto resolves to off.
  Planner pl{c, m, d, c.optim_zero1 != "off", false, c.exec_debug_arena != "off"};
  const uint64_t budget = mem_budget_bytes(c);
  const int spr = c.data_global_batch / c.exec_gpus;
  const int64_t S = m.S;
  std::vector<int> cands;
  if (c.exec_micro_seqs > 0) {
    cands.push_back(c.exec_micro_seqs);
  } else {
    for (int k = spr; k >= 1; --k)
      if (spr % k == 0) cands.push_back(k);
  }
  std::string log;
  auto note = [&](int mb, Remat r, int rows, uint64_t bytes, bool ok) {
    char b[160];
    std::snprintf(b, sizeof(b), "micro_seqs=%d remat=%s ce_rows=%d arena=%.2f GiB: %s\n", mb, remat_text(r), rows,
                  bytes / 1073741824.0, ok ? "fits" : "does not fit");
    log += b;
  };
  auto largest = [&](Remat r) {
    for (int mb : cands) {
      const int rows = pl.fit_rows(mb);
      const uint64_t bytes = pl.arena(mb, r, rows);
      const bool ok = bytes <= budget;
      note(mb, r, rows, bytes, ok);
      if (ok) return mb;
    }
    return 0;
  };

  PlanChoice pc;
  pc.zero1 = pl.zero1;
  pc.zero2_vocab = pl.zero2;
  int mb = 0;
  Remat remat = Remat::All;
  if (c.exec_remat != "auto") {
    remat = c.exec_remat == "all" ? Remat::All : c.exec_remat == "min" ? Remat::Min : Remat::Layer;
    mb = largest(remat);
  } else if (int m_all = largest(Remat::All)) {
    mb = m_all;
    if (int64_t(m_all) * S < 16384) {
      int m_min = largest(Remat::Min);
      if (m_min > 0 && m_min >= 2 * m_all) {
        mb = m_min;
        remat = Remat::Min;
      }
    }
  } else if (int m_min = largest(Remat::Min)) {
    mb = m_min;
    remat = Remat::Min;
  } else {
    remat = Remat::Layer;
    mb = largest(Remat::Layer);
  }
  if (report) *report = log;
  pc.remat = remat;
  pc.micro_seqs = mb;
  if (mb == 0) {
    if (report) *report += "nothing fits in the budget\n";
    pc.device_bytes = 0;
    return pc;
  }

  const int64_t T = int64_t(mb) * S;
  if (c.exec_ce_chunk_rows > 0 || T < 8192) {
    pc.ce_chunk_rows = pl.fit_rows(mb);   // the chunk the fit test used
  } else {
    // smallest 8192 * 2^i whose head-gradient RMW <= 0.5% of the step, capped by what fits with 2 GiB headroom
    const int64_t store = (c.exec_ce_logits == "f32") ? 4 : 2;
    const int64_t left = int64_t(budget) - int64_t(pl.payload(mb, remat, 8192)) - (int64_t(2) << 30);
    int64_t cap = 8192;
    while (cap * 2 <= T && (cap * 2 - 8192) * m.V * store <= left) cap *= 2;
    int64_t rows = 8192;
    while (rows < cap) {
      StepBudget sb = step_budget(c, m, pl.shape(mb, remat, int(rows)), ce_stage_of(c), c.exec_fused, ns_of(c));
      if (sb.row("ce_head_grad_rmw") <= 0.005 * sb.step_s) break;
      rows *= 2;
    }
    pc.ce_chunk_rows = int(rows);
  }
  pc.device_bytes = pl.arena(mb, remat, pc.ce_chunk_rows);
  return pc;
}

}  // namespace nm
