// Step budget (memory.hpp; DESIGN.md §12): port of scripts/design_budget.py perf() and show()'s MFU / eval terms. Every
// expression keeps the Python operation order and int/float split, so the results are equal in double. Python's ints
// are unbounded: the integer products here are __int128 (I), which the knob ranges of knobs.def keep below 2^100, and
// are converted to double (correctly rounded, as Python's int -> float) where Python converts.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "marincpp/config.hpp"
#include "marincpp/memory.hpp"
#include "memory/internal.hpp"

namespace nm {
namespace {
using I = __int128;
double f(I x) { return static_cast<double>(x); }
}  // namespace

double StepBudget::row(const char* name) const {
  for (auto& [n, v] : rows)
    if (std::strcmp(n, name) == 0) return v;
  fatal(NM_WHERE, std::string("no budget row ") + name);
}

StepBudget step_budget(const Config& c, const ModelShape& m, const ExecShape& e, CeStage stage, bool fused,
                       NsPrecision ns_prec) {
  // ESTIMATES until the H100 benches land (DESIGN.md §12.1)
  const double PEAK = 989.5e12, TF32 = 494.7e12, FP32_FMA = 67e12, HBM = 3.35e12 * 0.80, NVL = 350e9;
  const double CAUSAL_COMPUTED = 0.504;
  const int64_t EVAL_TOKENS = int64_t(14529) * 4096;
  const int64_t D = m.D, L = m.L, H = m.H, KV = m.KV, hd = m.hd, F = m.F, V = m.V, S = m.S;
  const int64_t B = c.data_global_batch, NG = e.ranks, heads = m.n_ce_heads;
  const bool layer = e.remat == Remat::Layer, min = e.remat == Remat::Min;
  const ParamCounts p = param_counts(c, m);
  const int64_t T = B * S / NG;   // B <= 2^20, S <= 2^16
  const double eff = D <= 512 ? 0.60 : D <= 768 ? 0.66 : D <= 1024 ? 0.72 : 0.78;

  StepBudget out;
  auto put = [&](const char* n, double v) { out.rows.emplace_back(n, v); };
  const I fwd_gemm = I(2) * T * (I(D) * (H + 2 * KV) * hd + I(H) * hd * D + I(3) * D * F);
  put("gemm_trunk", f(3 * fwd_gemm * L) / (eff * PEAK) * (layer ? 4.0 / 3.0 : 1.0));
  const double keys = CAUSAL_COMPUTED * double(S) / 2;
  const double fa_fwd = f(I(4) * T) * keys * double(H) * double(hd);
  const double ef = hd == 64 ? 0.40 : 0.55, eb = hd == 64 ? 0.30 : 0.42;
  put("attn_fwd", double(L) * fa_fwd / (ef * PEAK) * (layer ? 2.0 : 1.0));
  put("attn_bwd", double(L) * 2.5 * fa_fwd / (eb * PEAK));
  put("ce_gemm", f(I(6) * T * D * V * heads) / (eff * PEAK));
  const int64_t extra = stage == CeStage::B1 ? 6 : stage == CeStage::B1_5 ? 4 : 2;
  put("ce_mem", f(I(extra) * V * T * heads) / HBM);
  // chunks per head per step: every microbatch runs ceil(micro tokens / chunk) chunks
  const int64_t chunks_per_head = int64_t(e.n_micro) * ((e.micro_tokens + e.ce_chunk_rows - 1) / e.ce_chunk_rows);
  put("ce_head_grad_rmw", f(I(heads) * chunks_per_head * 2 * V * D * 4) / HBM);
  I ew = I(40) * D + I(16) * F + I(10) * (H + KV) * hd;
  if (fused)
    ew -= I(4) * F + I(4) * F + I(2) * (H + KV) * hd + I(4) * (H + KV) * hd + I(2) * H * hd + (D == 512 ? 4 * D : 0);
  I ew_side = I(14) * H * hd;
  if (fused) ew_side -= I(2) * H * hd;
  I ew_tot = (ew + ew_side) * T * L;
  if (min) ew_tot += (I(4) * D + I(4) * D + I(6) * F + I(4) * (H + KV) * hd) * T * L;
  put("elementwise", f(ew_tot) / HBM);
  put("embed_fb", f(I(8 * D + 24) * T) / HBM);
  const double zf = e.zero1 ? 1.0 / double(NG) : 1.0;
  const double opt_bytes =
      f(I(40) * p.muon + I(30) * (p.norms + p.emb) + I(54) * p.heads) * zf + f(I(2) * p.total) * zf;
  put("optimizer_elementwise", opt_bytes / HBM);
  const int64_t mats[7][2] = {{D, H * hd}, {KV * hd, D}, {KV * hd, D}, {D, H * hd}, {D, F}, {D, F}, {D, F}};
  I ns = 0;
  for (auto& s : mats) {
    const I a = std::min(s[0], s[1]), b = std::max(s[0], s[1]);
    ns += 5 * (4 * a * a * b + 2 * a * a * a);
  }
  ns *= L;
  const double rate = ns_prec == NsPrecision::TF32     ? 0.5 * TF32
                      : ns_prec == NsPrecision::X3TF32 ? 2 * 0.75 * FP32_FMA / 0.7
                                                       : 0.75 * FP32_FMA;
  const double newton_schulz = f(ns) / rate * zf;
  put("newton_schulz", newton_schulz);
  const int64_t wire = e.zero1 ? 4 + 2 : 8;
  // Python divides int by int here (correctly rounded); equal to this while the numerators stay below 2^53
  const double comm = f(I(wire) * p.total * (NG - 1)) / double(NG) / NVL;
  const double emb_rs = f(I(4) * p.emb * (NG - 1)) / double(NG) / NVL;
  put("comm_exposed", 0.10 * comm + std::max(0.0, emb_rs - newton_schulz));
  // Python's sum() of floats (3.12+): Neumaier compensated summation, compensation added once at the end.
  double step = 0, comp = 0;
  for (auto& [n, v] : out.rows) {
    const double t = step + v;
    comp += std::fabs(step) >= std::fabs(v) ? (step - t) + v : (v - t) + step;
    step = t;
  }
  if (comp != 0 && std::isfinite(comp)) step += comp;
  out.step_s = step;

  // one eval: forward only over EVAL_TOKENS / NG tokens, every readout
  const double tok = double(EVAL_TOKENS) / double(NG);
  // fwd_gemm / T is exact (fwd_gemm = 2 T x ...), as Python's int / int; ew * L / 3 likewise below 2^53
  out.eval_s = (f(fwd_gemm / T) * double(L) / (eff * PEAK) +
                4 * keys * double(H) * double(hd) * double(L) / (ef * PEAK) + f(I(2) * D * V * heads) / (eff * PEAK) +
                f(ew * L) / 3 / HBM) *
               tok;
  const int64_t N = c.optim_num_steps;
  const int periodic = c.eval_every > 0 ? int((N - 1) / c.eval_every) : 0;   // s % every == 0, 0 < s < N
  out.n_evals = (c.eval_init_eval ? 1 : 0) + periodic + 1;
  out.tokens_per_s = f(I(B) * S) / step;
  out.mfu_pct = m.flops_per_example * double(B) / step / (double(NG) * PEAK) * 100;
  return out;
}

}  // namespace nm
