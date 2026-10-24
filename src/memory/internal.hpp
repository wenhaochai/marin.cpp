// Internal to src/memory: parameter counts by optimizer group, shared by the planner and the step budget.
#pragma once

#include <cstdint>

#include "marincpp/config.hpp"

namespace nm {

// scripts/design_budget.py params(): MuonH matrices, Adam norm gains, embedding, AdamH heads.
struct ParamCounts {
  int64_t muon, norms, emb, heads, total;
};
inline ParamCounts param_counts(const Config& c, const ModelShape& m) {
  const int64_t D = m.D, L = m.L, H = m.H, KV = m.KV, hd = m.hd, F = m.F, V = m.V;
  ParamCounts p{};
  p.muon = L * (D * H * hd + 2 * D * KV * hd + H * hd * D + 3 * D * F);
  p.norms = L * (4 * D + 2 * hd) + D + int64_t(m.n_readouts - 1) * D;
  p.emb = V * D;
  p.heads = (c.model_tie_embeddings && m.n_readouts == 1) ? 0 : int64_t(m.n_readouts) * V * D;
  p.total = p.muon + p.norms + p.emb + p.heads;
  return p;
}

}  // namespace nm
