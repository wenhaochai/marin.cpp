// s0_memory: plan layout, bind-by-name rules on the CPU side, planner choices (DESIGN.md §2.3, §2.5, §3.8). The byte
// comparison of every plan against scripts/design_budget.py is test s0_plan_vs_budget (Python).
#include <cmath>
#include <cstdio>

#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"
#include "nmtest.hpp"

using namespace nm;

namespace {

BufferSpec buf(const std::string& name, Lifetime life, DType t, std::initializer_list<int64_t> shape, int layer = -1) {
  BufferSpec b;
  b.name = name;
  b.life = life;
  b.dtype = t;
  b.ndim = static_cast<int>(shape.size());
  int i = 0;
  for (int64_t s : shape) b.shape[i++] = s;
  b.layer = layer;
  return b;
}

DatasetMeta meta_for(const ModelShape& m) {
  DatasetMeta d;
  d.seq_len = m.S;
  d.vocab = m.V;
  d.max_segs_per_seq = kDefaultMaxSegsPerSeq;
  return d;
}

PlanChoice choose(const std::vector<std::string>& sets, const std::vector<std::string>& presets = {}) {
  Config c = load_config({}, presets, sets);
  ModelShape m = model_shape(c);
  std::string rep;
  return choose_plan(c, m, meta_for(m), &rep);
}

// The CLI flow (main.cpp resolve_all): choose, resolve, validate; returns the resolved config.
Config resolve_cli(const std::vector<std::string>& sets, PlanChoice* pc_out = nullptr) {
  Config c = load_config({}, {}, sets);
  validate(c);
  ModelShape m = model_shape(c);
  std::string rep;
  PlanChoice pc = choose_plan(c, m, meta_for(m), &rep);
  NM_REQUIRE(pc.micro_seqs > 0, rep);
  resolve(c, nullptr, nullptr, &pc);
  validate(c);
  if (pc_out) *pc_out = pc;
  return c;
}

MemoryPlan plan_of(const Config& c) {
  ModelShape m = model_shape(c);
  DatasetMeta d = meta_for(m);
  return build_plan(c, m, exec_shape(c, m, &d), d);
}

}  // namespace

NM_TEST(s0_memory, layout_alignment_and_order) {
  MemoryPlan p;
  p.add(buf("ws.x", Lifetime::Workspace, DType::u8, {100}));
  p.add(buf("act.L1.a", Lifetime::Activation, DType::bf16, {3, 5}, 1));
  p.add(buf("act.L0.a", Lifetime::Activation, DType::bf16, {7}, 0));
  p.add(buf("param", Lifetime::Persistent, DType::f32, {10}));
  p.add(buf("tr.t", Lifetime::Transient, DType::f32, {1}));
  CHECK_THROWS(p.add(buf("param", Lifetime::Persistent, DType::f32, {1})), "duplicate buffer name param");
  CHECK_THROWS(p.add(buf("empty", Lifetime::Persistent, DType::f32, {0})), "empty dimension");
  CHECK_EQ(p.payload_bytes(), uint64_t(100 + 30 + 14 + 40 + 4));
  CHECK_EQ(p.payload_bytes('A'), uint64_t(44));
  p.finalize(1 << 20, false);
  auto off = [&](const char* n) { return p.offset(p.find(n)); };
  CHECK_EQ(off("param"), uint64_t(0));                 // persistent first
  CHECK_EQ(off("act.L0.a"), uint64_t(256));            // activations by layer
  CHECK_EQ(off("act.L1.a"), uint64_t(512));
  CHECK_EQ(off("tr.t"), uint64_t(768));                // then transients, then workspaces
  CHECK_EQ(off("ws.x"), uint64_t(1024));
  CHECK_EQ(p.device_bytes(), uint64_t(1280));
  CHECK(!p.find("nope").valid());
  MemoryPlan q;
  q.add(buf("a", Lifetime::Persistent, DType::f32, {10}));
  q.add(buf("b", Lifetime::Persistent, DType::f32, {10}));
  q.finalize(1 << 20, true);
  // [band][a][band][pad][band][b][band]: a band right before and right after every buffer
  CHECK_EQ(q.offset(q.find("a")), kCanaryBytes);
  const uint64_t end_a = kCanaryBytes + 40 + kCanaryBytes;
  CHECK_EQ(q.offset(q.find("b")), (end_a + 255) / 256 * 256 + kCanaryBytes);
  CHECK_EQ(q.device_bytes(), (q.offset(q.find("b")) + 40 + kCanaryBytes + 255) / 256 * 256);
  MemoryPlan r;
  r.add(buf("big", Lifetime::Persistent, DType::f32, {1000}));
  CHECK_THROWS(r.finalize(1000, false), "budget");
  // absurd shapes saturate at 1 PiB per buffer: the plan does not fit, and nothing wraps around
  BufferSpec huge = buf("huge", Lifetime::Persistent, DType::f32, {int64_t(1) << 40, int64_t(1) << 40});
  CHECK_EQ(huge.numel(), INT64_MAX);
  CHECK_EQ(huge.bytes(), kMaxBufferBytes);
  CHECK_EQ(buf("big2", Lifetime::Persistent, DType::f32, {int64_t(1) << 49}).bytes(), kMaxBufferBytes);
}

NM_TEST(s0_memory, poison_never_touches_saved_tensors) {
  // 130m at micro 32 with remat=all saves a, qk_rot, m, h (tagged Elementwise): the poison set is exactly the
  // Transient and Workspace buffers.
  Config c = resolve_cli({"exec.remat=all"});
  MemoryPlan p = plan_of(c);
  int saved_elementwise = 0, targets = 0;
  for (int i = 0; i < p.size(); ++i) {
    const BufferSpec& b = p.spec(BufferId{i});
    if (b.life == Lifetime::Activation && b.recompute == Recompute::Elementwise) ++saved_elementwise;
    const bool want = b.life == Lifetime::Transient || b.life == Lifetime::Workspace;
    if (poison_target(b) != want) nmtest::fail(__FILE__, __LINE__, "poison_target wrong for " + b.name);
    targets += poison_target(b);
  }
  CHECK_EQ(saved_elementwise, 24);   // 6 layers x {a, qk_rot, m, h}
  CHECK(targets > 0);
  for (const char* pol : {"min", "layer"}) {   // the recompute slots are transients, so they are poisoned
    MemoryPlan q = plan_of(resolve_cli({std::string("exec.remat=") + pol, "exec.micro_seqs=8"}));
    const char* slot = std::string(pol) == "min" ? "tr.recompute" : "tr.layer_recompute";
    CHECK(poison_target(q.spec(q.find(slot))));
  }
}

NM_TEST(s0_memory, planner_choices) {
  // DESIGN.md §2.5 chosen plans
  PlanChoice a = choose({});
  CHECK_EQ(a.micro_seqs, 32);
  CHECK(a.remat == Remat::All);
  CHECK_EQ(a.ce_chunk_rows, 32768);
  CHECK(a.zero1);
  PlanChoice b = choose({"model.size=520m"});
  CHECK_EQ(b.micro_seqs, 8);
  CHECK(b.remat == Remat::All);
  CHECK_EQ(b.ce_chunk_rows, 16384);
  PlanChoice pa = choose({"model.size=pls_d48", "optim.zero1=off"});   // option A
  CHECK_EQ(pa.micro_seqs, 4);
  CHECK(pa.remat == Remat::Min);
  CHECK_EQ(pa.ce_chunk_rows, 8192);
  CHECK(!pa.zero1);
  PlanChoice pb = choose({"model.size=pls_d48"});                      // option B
  CHECK_EQ(pb.micro_seqs, 8);
  CHECK(pb.remat == Remat::All);
  CHECK_EQ(pb.ce_chunk_rows, 16384);
  // requested values are honoured; strict preset fixes the chunk
  PlanChoice s = choose({"model.size=520m", "exec.micro_seqs=32", "exec.remat=layer"});
  CHECK_EQ(s.micro_seqs, 32);
  CHECK(s.remat == Remat::Layer);
  PlanChoice st = choose({"model.size=1_2b", "exec.ce_chunk_rows=8192"});
  CHECK_EQ(st.ce_chunk_rows, 8192);
  // a fixed microbatch that does not fit with remat=all falls back to min, then layer
  PlanChoice f = choose({"model.size=1_2b", "exec.micro_seqs=32"});
  CHECK_EQ(f.micro_seqs, 32);
  CHECK(f.remat == Remat::Layer);
  // nothing fits a tiny budget
  PlanChoice n = choose({"exec.mem_budget_gib=2"});
  CHECK_EQ(n.micro_seqs, 0);
  CHECK_EQ(n.device_bytes, uint64_t(0));
  // the strict preset as a strict run passes it
  PlanChoice sp = choose({"model.size=520m"}, {"levanter_strict"});
  CHECK_EQ(sp.micro_seqs, 32);
  CHECK(sp.remat == Remat::Layer);
  CHECK_EQ(sp.ce_chunk_rows, 8192);
}

NM_TEST(s0_memory, ce_chunk_for_any_seq_len) {
  // microbatches below 8,192 tokens: one chunk of the microbatch rounded up to a multiple of 128, so the resolved
  // config passes its own exec.ce_chunk_rows check (model.seq_len takes any value in 128..65536)
  struct Case { int S, B; };
  for (Case k : {Case{2000, 16}, Case{1000, 4}, Case{129, 4}, Case{4096, 4}, Case{3000, 128}}) {
    PlanChoice pc;
    Config c = resolve_cli({"model.seq_len=" + std::to_string(k.S), "data.global_batch=" + std::to_string(k.B)}, &pc);
    const int64_t T = int64_t(pc.micro_seqs) * k.S;
    CHECK_EQ(c.exec_ce_chunk_rows % 128, 0);
    if (T < 8192) CHECK_EQ(int64_t(c.exec_ce_chunk_rows), (T + 127) / 128 * 128);
    else CHECK(c.exec_ce_chunk_rows >= 8192 && c.exec_ce_chunk_rows <= T);
    ModelShape m = model_shape(c);
    DatasetMeta d = meta_for(m);
    StepBudget sb = step_budget(c, m, exec_shape(c, m, &d), CeStage::B1, true, NsPrecision::TF32);
    CHECK(sb.row("ce_head_grad_rmw") > 0);   // a chunk larger than the microbatch still costs one RMW
  }
  CHECK_EQ(ce_rows_floor(8000), 8064);
  CHECK_EQ(ce_rows_floor(128), 128);
  CHECK_EQ(ce_rows_floor(1 << 20), 8192);
}

NM_TEST(s0_memory, fit_uses_the_requested_ce_chunk_and_the_arena) {
  // 300m with a 65,536-row chunk: micro 32 does not fit with that chunk, micro 16 does (largest that fits, §2.3)
  PlanChoice pc;
  Config c = resolve_cli({"model.size=300m", "exec.ce_chunk_rows=65536"}, &pc);
  CHECK_EQ(pc.micro_seqs, 16);
  CHECK_EQ(c.exec_ce_chunk_rows, 65536);
  MemoryPlan p = plan_of(c);
  p.finalize(UINT64_MAX, false);
  CHECK(p.device_bytes() <= mem_budget_bytes(c));
  CHECK_EQ(pc.device_bytes, p.device_bytes());   // the choice reports the arena it fitted
  // with guard bands the arena is what is checked against the budget
  PlanChoice pg;
  Config g = resolve_cli({"exec.debug_arena=canary"}, &pg);
  MemoryPlan q = plan_of(g);
  q.finalize(UINT64_MAX, true);
  CHECK_EQ(pg.device_bytes, q.device_bytes());
  CHECK(q.device_bytes() > q.payload_bytes() + 2 * kCanaryBytes * uint64_t(q.size() - 1));
  // a budget between payload and arena (canary mode) is refused at choose time, never after
  const double gib = (q.payload_bytes() + (q.device_bytes() - q.payload_bytes()) / 2) / 1073741824.0;
  char budget[64];
  std::snprintf(budget, sizeof(budget), "exec.mem_budget_gib=%.17g", gib);
  PlanChoice tight = choose({"exec.debug_arena=canary", "exec.micro_seqs=32", "exec.remat=all",
                             "exec.ce_chunk_rows=" + std::to_string(g.exec_ce_chunk_rows), budget});
  CHECK_EQ(tight.micro_seqs, 0);
}

NM_TEST(s0_memory, step_budget_has_no_integer_overflow) {
  // products like 6 T D V heads exceed int64 inside the knob ranges; they are exact (__int128) and positive
  Config c = load_config({}, {}, {"readout.kind=shared", "model.layers=48", "model.seq_len=65536",
                                  "data.global_batch=8192", "exec.gpus=1", "exec.micro_seqs=1", "exec.remat=layer",
                                  "exec.ce_chunk_rows=8192", "optim.zero1=on", "optim.zero2_vocab=off",
                                  "exec.ce_logits=bf16"});
  validate(c);
  ModelShape m = model_shape(c);
  DatasetMeta d = meta_for(m);
  StepBudget sb = step_budget(c, m, exec_shape(c, m, &d), CeStage::B1, true, NsPrecision::TF32);
  const double T = 8192.0 * 65536.0;
  CHECK_EQ(sb.row("ce_gemm"), 6.0 * T * 512.0 * 128256.0 * 48.0 / (0.60 * 989.5e12));
  for (auto& [n, v] : sb.rows)
    if (!(v >= 0 && std::isfinite(v)))
      nmtest::fail(__FILE__, __LINE__, std::string("budget row ") + n + " is negative or not finite");
  CHECK(sb.step_s > 0 && sb.mfu_pct > 0 && std::isfinite(sb.mfu_pct));
}

NM_TEST(s0_memory, remat_policies_drop_the_right_tensors) {
  for (const char* pol : {"all", "min", "layer"}) {
    Config c = load_config({}, {}, {"exec.micro_seqs=8", std::string("exec.remat=") + pol, "exec.ce_chunk_rows=8192",
                                    "optim.zero1=on", "optim.zero2_vocab=off", "exec.ce_logits=bf16"});
    ModelShape m = model_shape(c);
    DatasetMeta d = meta_for(m);
    MemoryPlan p = build_plan(c, m, exec_shape(c, m, &d), d);
    std::string P = pol;
    CHECK(p.find("act.L5.x_in").valid());
    CHECK_EQ(p.find("act.L0.a").valid(), P == "all");
    CHECK_EQ(p.find("act.L0.h").valid(), P == "all");
    CHECK_EQ(p.find("act.L0.gu").valid(), P != "layer");
    CHECK_EQ(p.find("tr.recompute").valid(), P == "min");
    CHECK_EQ(p.find("tr.layer_recompute").valid(), P == "layer");
    CHECK(!p.find("act.L6.x_in").valid());
    CHECK(!p.find("ws.fa3.dk_accum").valid());          // MHA: no GQA accumulators
  }
  Config g = load_config({}, {"levanter_strict"}, {"model.size=520m", "optim.zero1=on", "optim.zero2_vocab=off",
                                                   "exec.ce_logits=bf16"});
  ModelShape m = model_shape(g);
  DatasetMeta d = meta_for(m);
  MemoryPlan p = build_plan(g, m, exec_shape(g, m, &d), d);
  CHECK(p.find("ws.fa3.dk_accum").valid());             // GQA
  CHECK(p.find("mb.ce_dw_accum").valid());              // strict per-microbatch head-gradient accumulator
  CHECK_EQ(p.spec(p.find("mb.ce_dw_accum")).bytes(), uint64_t(128256) * 1024 * 4);
}

NM_TEST(s0_memory, step_budget_matches_design_table) {
  // DESIGN.md §12.2 step (B1) row, ms, as printed (one decimal)
  const std::pair<const char*, double> want[] = {{"130m", 187.7}, {"300m", 378.3}, {"520m", 1626.8}, {"1_2b", 3069.7}};
  for (auto& [size, ms] : want) {
    Config c = load_config({}, {}, {std::string("model.size=") + size});
    ModelShape m = model_shape(c);
    DatasetMeta d = meta_for(m);
    std::string rep;
    PlanChoice pc = choose_plan(c, m, d, &rep);
    resolve(c, nullptr, nullptr, &pc);
    StepBudget b = step_budget(c, m, exec_shape(c, m, &d), CeStage::B1, true, NsPrecision::TF32);
    CHECK(std::abs(b.step_s * 1e3 - ms) < 0.05 + 1e-9);
  }
}
