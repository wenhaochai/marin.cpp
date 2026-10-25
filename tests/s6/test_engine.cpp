// Stage S6 C++ gates of the optimizer engine (src/optim/engine.hpp) on synthetic data.
//   s6_ownership  plan_ownership_specs on the 130m and 1.2B baseline layouts (fused Wqkv / Wgu storage with 64-row
//                 interleaved leaves, as model.hpp describes): every row of every parameter owned exactly once, MuonH
//                 leaves owned whole by one rank, vocab-sized tensors row-split over all ranks, LPT balance, engine
//                 layouts (matrices from leaves, owned ranges), zero1 = off -> every rank owns everything
//   s6_zero1      (GPU) 4 emulated ranks (one engine each, sharded, sums added between the phases as the trainer's
//                 all-reduce does) against 1 rank on the same fused 130m-shaped tensors: MuonH, Adam and AdamH
//                 parameters and states bitwise equal; clip sums within 1e-12 relative (f64 partial order)
//   s6_graph      (GPU) the whole step captured in a CUDA graph (global capture mode: any allocation or sync inside
//                 fails) and replayed 3 times == 3 eager steps, bitwise; the step counter advances on device
//   s6_rowpart    (GPU) row partitions: frozen blocks get no update and enter no norm, per-block lr multiplier tables
//                 and own AdamH norm scopes, against a host f64 reference of optim.md §5; the scalar fallback of the
//                 elementwise kernels (unaligned slices) against their float4 path; MuonH matrices whose OUT and IN are
//                 not multiples of the tile edge against a host f64 reference; the one-tensor C ABI entry points
//                 (nm_adam_update, nm_adamh_update, nm_muonh_update) against host f32 bit patterns of optim.md §5.3;
//                 lazy Adam rows and decoupled weight decay against host f32; configurations the engine refuses
//   s6_nsalgo     (GPU) the NS GEMM algorithm is a function of the matrix shape alone: every oriented NS shape of the
//                 four sizes planned at several batch counts gets the same cuBLASLt algorithm configuration, no split-K
//   s6_bounds     (GPU) a step at k = N - 1 runs; a step at k = N (past the tables) traps (last case: the context dies)
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "marincpp/capi.h"
#include "marincpp/config.hpp"
#include "nmtest.hpp"
#include "optim/engine.hpp"

using namespace nm;
using namespace nm::opt;

namespace {

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Baseline parameter specs in marincpp's storage layout: per layer a fused Wqkv [(H + 2 KV) hd, D] whose q, k, v
// leaves interleave 64-row blocks (q block, k block, v block, ... while each leaf has rows left), Wo [D, H hd], a fused
// Wgu [2F, D] with gate / up interleaved in 64-row blocks, Wdown [D, F], the four block gains and q/k gains; then the
// final gain, the embedding (adam, vocab) and lm_head (adamh, vocab).
std::vector<ParamSpec> baseline_specs(int L, int D, int H, int KV, int hd, int F, int V) {
  std::vector<ParamSpec> ps;
  auto add = [&](std::string name, std::vector<int64_t> shape, std::string group, int layer) -> ParamSpec& {
    ParamSpec p;
    p.name = std::move(name);
    p.shape = std::move(shape);
    p.group = std::move(group);
    p.layer = layer;
    p.rows = 1;
    for (size_t i = 0; i + 1 < p.shape.size(); ++i) p.rows *= p.shape[i];
    p.cols = p.shape.back();
    ps.push_back(p);
    return ps.back();
  };
  auto interleave = [](std::vector<std::pair<std::string, int64_t>> leaves) {
    std::vector<LeafSlice> out(leaves.size());
    std::vector<int64_t> left;
    for (size_t i = 0; i < leaves.size(); ++i) out[i].name = leaves[i].first, left.push_back(leaves[i].second);
    int64_t row = 0;
    for (bool any = true; any;) {
      any = false;
      for (size_t i = 0; i < leaves.size(); ++i)
        if (left[i] > 0) {
          const int64_t n = std::min<int64_t>(64, left[i]);
          auto& r = out[i].rows;
          if (!r.empty() && r.back() == row) r.back() += n;
          else r.push_back(row), r.push_back(row + n);
          row += n, left[i] -= n, any = true;
        }
    }
    return out;
  };
  for (int l = 0; l < L; ++l) {
    const std::string pre = "transformer.layers." + std::to_string(l) + ".";
    auto& qkv = add("L" + std::to_string(l) + ".Wqkv", {int64_t(H + 2 * KV) * hd, D}, "muonh", l);
    qkv.leaves = interleave({{pre + "self_attn.q_proj.weight", int64_t(H) * hd},
                             {pre + "self_attn.k_proj.weight", int64_t(KV) * hd},
                             {pre + "self_attn.v_proj.weight", int64_t(KV) * hd}});
    add("L" + std::to_string(l) + ".Wo", {D, int64_t(H) * hd}, "muonh", l);
    auto& gu = add("L" + std::to_string(l) + ".Wgu", {2 * int64_t(F), D}, "muonh", l);
    gu.leaves = interleave({{pre + "mlp.gate_proj.weight", F}, {pre + "mlp.up_proj.weight", F}});
    add("L" + std::to_string(l) + ".Wdown", {D, F}, "muonh", l);
    for (const char* g : {"input_layernorm", "post_attention_layernorm", "post_attn_layernorm", "post_mlp_layernorm"})
      add("L" + std::to_string(l) + "." + g, {D}, "adam", l);
    add("L" + std::to_string(l) + ".q_norm", {hd}, "adam", l);
    add("L" + std::to_string(l) + ".k_norm", {hd}, "adam", l);
  }
  add("final_norm", {D}, "adam", -1);
  add("embedding", {V, D}, "adam", -1).vocab_sized = true;
  add("lm_head", {V, D}, "adamh", -1).vocab_sized = true;
  return ps;
}

std::vector<EngGroup> groups130() {
  Config c;
  apply_preset(c, "130m");
  return baseline_groups(c);
}

void check_ownership(const std::vector<ParamSpec>& ps, const std::vector<EngGroup>& gs, int nranks) {
  const Ownership own = plan_ownership_specs(ps, gs, nranks, true);
  std::vector<std::vector<int>> owner(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) owner[i].assign(ps[i].rows, -2);
  for (const auto& s : own.slices) {
    CHECK(s.owner >= 0 && s.owner < nranks);
    for (int64_t r = s.row_begin; r < s.row_end; ++r) {
      CHECK_EQ(owner[s.param][r], -2);   // owned once
      owner[s.param][r] = s.owner;
    }
  }
  for (size_t i = 0; i < ps.size(); ++i) {
    for (int o : owner[i]) CHECK(o >= 0);   // owned at all
    if (ps[i].group == "muonh") {
      if (ps[i].leaves.empty()) {
        CHECK(std::all_of(owner[i].begin(), owner[i].end(), [&](int o) { return o == owner[i][0]; }));
      }
      for (const LeafSlice& l : ps[i].leaves) {
        const int o = owner[i][l.rows[0]];
        for (size_t k = 0; k + 1 < l.rows.size(); k += 2)
          for (int64_t r = l.rows[k]; r < l.rows[k + 1]; ++r) CHECK_EQ(owner[i][r], o);
      }
    }
    if (ps[i].vocab_sized) {
      std::set<int> rs(owner[i].begin(), owner[i].end());
      CHECK_EQ(static_cast<int>(rs.size()), nranks);
    }
  }
  // LPT balance: the most loaded rank within 15% of the mean
  const double mean = std::accumulate(own.load_per_rank.begin(), own.load_per_rank.end(), 0.0) / nranks;
  const double mx = *std::max_element(own.load_per_rank.begin(), own.load_per_rank.end());
  std::printf("  ranks %d: load max / mean = %.4f\n", nranks, mx / mean);
  CHECK(mx <= 1.15 * mean);
  // engine layouts: every MuonH matrix of a rank owned whole; leaves become matrices with row maps
  int64_t mats = 0;
  for (int r = 0; r < nranks; ++r) {
    auto ts = engine_layout(ps, gs, own, r);
    for (size_t i = 0; i < ts.size(); ++i) {
      if (gs[ts[i].group].rule != OptRule::MuonH) continue;
      CHECK_EQ(ts[i].mats.size(), ps[i].leaves.empty() ? size_t(1) : ps[i].leaves.size());
      for (const EngMatrix& m : ts[i].mats) {
        const int64_t r0 = m.row_map.empty() ? m.row0 : m.row_map[0];
        bool mine = false;
        for (auto [b, e] : ts[i].owned) mine |= r0 >= b && r0 < e;
        mats += mine;
      }
    }
  }
  int64_t want = 0;
  for (const auto& p : ps)
    if (p.group == "muonh") want += p.leaves.empty() ? 1 : static_cast<int64_t>(p.leaves.size());
  CHECK_EQ(mats, want);
}

}  // namespace

NM_TEST(s6_ownership, baseline_layouts) {
  const auto gs = groups130();
  const auto p130 = baseline_specs(6, 512, 8, 8, 64, 1792, 128256);
  const auto p12b = baseline_specs(16, 2048, 16, 8, 128, 7168, 128256);
  for (int n : {2, 4, 8}) {
    check_ownership(p130, gs, n);
    check_ownership(p12b, gs, n);
  }
  // replicated: every slice owner -1, every rank owns all rows
  const Ownership rep = plan_ownership_specs(p130, gs, 4, false);
  for (const auto& s : rep.slices) CHECK_EQ(s.owner, -1);
  const auto ts = engine_layout(p130, gs, rep, 3);
  for (size_t i = 0; i < ts.size(); ++i) {
    CHECK_EQ(ts[i].owned.size(), size_t(1));
    CHECK_EQ(ts[i].owned[0].second - ts[i].owned[0].first, p130[i].rows);
  }
  // an unknown group name is refused
  auto bad = p130;
  bad[0].group = "nope";
  CHECK_THROWS(plan_ownership_specs(bad, gs, 4, true), "no optimizer group");
}

// optim.md §5.3: the kernels' f32 constants of the baseline groups, as bit patterns, and equal to optim_constants().
NM_TEST(s6_ownership, constants) {
  auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
  for (const char* size : {"130m", "300m"}) {
    Config c;
    apply_preset(c, size);
    const auto gs = baseline_groups(c);
    const OptimConstants k = optim_constants(c);
    const optk::GroupDev mu = group_dev(gs[0], 0), ah = group_dev(gs[1], 1), ad = group_dev(gs[2], 2);
    CHECK_EQ(bits(ah.one_minus_b1), 0x3DCCCCCDu);
    CHECK_EQ(bits(ah.one_minus_b2), 0x3CA3D70Au);
    CHECK_EQ(bits(ah.b1), 0x3F666666u);
    CHECK_EQ(bits(ah.b2), 0x3F7AE148u);
    CHECK_EQ(bits(ad.one_minus_b1), bits(k.one_minus_b1));
    CHECK_EQ(bits(ad.eps), bits(k.eps));
    CHECK_EQ(bits(mu.momentum), bits(k.momentum));
    CHECK_EQ(bits(ah.eps), std::string(size) == "130m" ? 0x1E3CE508u : 0x26901D7Du);
    CHECK_EQ(bits(mu.momentum), std::string(size) == "130m" ? 0x3F733333u : 0x3F7AE148u);
    CHECK_EQ(gs[0].lr_table, 0);
    CHECK_EQ(gs[1].lr_table, 0);   // AdamH: main learning rate
    CHECK_EQ(gs[2].lr_table, 1);   // Adam: adam_lr
    CHECK_EQ(ad.weight_decay, 0.0f);
  }
  const EngOptions o;
  CHECK_EQ(bits(static_cast<float>(o.ns_a)), 0x405C72B0u);
  CHECK_EQ(bits(static_cast<float>(o.ns_b)), 0xC098CCCDu);
  CHECK_EQ(bits(static_cast<float>(o.ns_c)), 0x40020419u);
  CHECK_EQ(bits(static_cast<float>(o.muon_eps)), 0x3727C5ACu);
  CHECK_EQ(bits(optk::kNormFloor), 0x2EDBE6FFu);
}

// ------------------------------------------------------------------------------------------------ GPU cases
namespace {

struct DevBuf {
  float* p = nullptr;
  size_t n = 0;
  explicit DevBuf(size_t count = 0) : n(count) {
    if (n) NM_CUDA(cudaMalloc(&p, n * sizeof(float)));
  }
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
  void up(const std::vector<float>& v) { NM_CUDA(cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice)); }
  std::vector<float> down() const {
    std::vector<float> v(n);
    NM_CUDA(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
  }
};

std::vector<float> randn(size_t n, uint32_t seed, float scale) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// One full copy of every tensor (params, grads, states) on the device.
struct World {
  std::vector<ParamSpec> specs;
  std::vector<DevBuf> p, g, s0, s1;
  World(const std::vector<ParamSpec>& ps, uint32_t seed) : specs(ps) {
    for (size_t i = 0; i < ps.size(); ++i) {
      const size_t n = ps[i].rows * ps[i].cols;
      p.emplace_back(n), g.emplace_back(n), s0.emplace_back(n), s1.emplace_back(n);
      p.back().up(randn(n, seed + 4 * i, 0.02f));
      g.back().up(randn(n, seed + 4 * i + 1, ps[i].vocab_sized ? 1e-4f : 1e-3f));
      s0.back().up(randn(n, seed + 4 * i + 2, 1e-4f));
      std::vector<float> nu = randn(n, seed + 4 * i + 3, 1e-6f);
      for (auto& x : nu) x = x * x;
      s1.back().up(nu);
    }
  }
  std::vector<EngTensor> bind(std::vector<EngTensor> ts) {
    for (size_t i = 0; i < ts.size(); ++i) ts[i].p32 = p[i].p, ts[i].g = g[i].p, ts[i].s0 = s0[i].p, ts[i].s1 = s1[i].p;
    return ts;
  }
};

struct Tabs {
  DevBuf lr{4}, alr{4}, bc1{4}, bc2{4};
  int32_t* step = nullptr;
  Tabs() {
    lr.up({0.02f, 0.019f, 0.018f, 0.017f});
    alr.up({0.008f, 0.0079f, 0.0078f, 0.0077f});
    bc1.up({0.1f, 0.19f, 0.271f, 0.3439f});
    bc2.up({0.02f, 0.0396f, 0.058808f, 0.07763184f});
    NM_CUDA(cudaMalloc(&step, sizeof(int32_t)));
    reset(0);
  }
  ~Tabs() { cudaFree(step); }
  void reset(int32_t k) { NM_CUDA(cudaMemcpy(step, &k, sizeof(k), cudaMemcpyHostToDevice)); }
  void bind(Engine& e) { e.set_tables({lr.p, alr.p}, bc1.p, bc2.p, step, 4); }
};

std::vector<double> d2h(const double* p, int n) {
  std::vector<double> v(n);
  if (n) NM_CUDA(cudaMemcpy(v.data(), p, n * sizeof(double), cudaMemcpyDeviceToHost));
  return v;
}
void h2d(double* p, const std::vector<double>& v) {
  if (!v.empty()) NM_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
}

}  // namespace

NM_TEST(s6_zero1, four_ranks_equal_one) {
  if (!have_gpu()) SKIP("no GPU");
  // 130m shapes with 2 layers and a 16k vocab (fits the shared A100 next to other jobs)
  const auto ps = baseline_specs(2, 512, 8, 8, 64, 1792, 16384);
  const auto gs = groups130();
  World w1(ps, 7), w4(ps, 7);
  Tabs t1, t4;
  EngOptions o;
  o.ns_chunk_bytes = size_t(64) << 20;   // several NS chunks per shape
  Engine e1(gs, w1.bind(engine_layout(ps, gs, plan_ownership_specs(ps, gs, 1, true), 0)), {}, {}, o);
  e1.bind(nullptr);
  t1.bind(e1);
  e1.prepare(nullptr);
  e1.record_step(nullptr, nullptr, true);
  const Ownership own = plan_ownership_specs(ps, gs, 4, true);
  o.sharded = true;
  std::vector<std::unique_ptr<Engine>> es;
  for (int r = 0; r < 4; ++r) {
    es.push_back(std::make_unique<Engine>(gs, w4.bind(engine_layout(ps, gs, own, r)), std::vector<RowPartition>{},
                                          std::vector<const uint8_t*>{}, o));
    es.back()->bind(nullptr);
    t4.bind(*es.back());
    es.back()->prepare(nullptr);
  }
  // phases in lockstep, the all-reduce done on the host between them (as ncclAllReduce of f64 sums)
  auto allreduce = [&](auto get, int n) {
    std::vector<double> sum(n, 0.0);
    for (auto& e : es) {
      const auto v = d2h(get(*e), n);
      for (int i = 0; i < n; ++i) sum[i] += v[i];
    }
    for (auto& e : es) h2d(get(*e), sum);
    return sum;
  };
  for (auto& e : es) e->record_sumsq(nullptr);
  const auto clip4 = allreduce([](Engine& e) { return e.clip_sumsq(); }, es[0]->n_clip());
  for (auto& e : es) e->record_update(nullptr);
  allreduce([](Engine& e) { return e.slot_pa(); }, 2 * es[0]->n_slots());
  for (auto& e : es) e->record_adamh_w(nullptr);
  allreduce([](Engine& e) { return e.slot_w(); }, es[0]->n_slots());
  // The emulated ranks share one step counter: none advances it (a rank that did would move k under the others).
  for (int r = 0; r < 4; ++r) es[r]->record_finish(nullptr, false);
  NM_CUDA(cudaDeviceSynchronize());
  const auto clip1 = d2h(e1.clip_sumsq(), e1.n_clip());
  for (size_t i = 0; i < clip1.size(); ++i) CHECK(std::fabs(clip1[i] - clip4[i]) <= 1e-12 * clip1[i]);
  int mats = 0;
  for (auto& e : es) mats += e->n_matrices();
  CHECK_EQ(mats, e1.n_matrices());
  size_t differ = 0;
  for (size_t i = 0; i < ps.size(); ++i) {
    for (auto [a, b] : {std::pair{&w1.p[i], &w4.p[i]}, std::pair{&w1.s0[i], &w4.s0[i]}, std::pair{&w1.s1[i], &w4.s1[i]}}) {
      const auto x = a->down(), y = b->down();
      if (std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) != 0) {
        ++differ;
        double d2 = 0, n2 = 0;
        size_t nd = 0;
        for (size_t j = 0; j < x.size(); ++j) {
          d2 += (double(x[j]) - y[j]) * (double(x[j]) - y[j]), n2 += double(x[j]) * x[j];
          nd += x[j] != y[j];
        }
        std::printf("  differs: %s (%zu of %zu elements, rel %.3e)\n", ps[i].name.c_str(), nd, x.size(),
                    std::sqrt(d2 / n2));
      }
    }
  }
  CHECK_EQ(differ, size_t(0));
}

NM_TEST(s6_graph, capture_equals_eager) {
  if (!have_gpu()) SKIP("no GPU");
  const auto ps = baseline_specs(1, 512, 8, 8, 64, 1792, 8192);
  const auto gs = groups130();
  const Ownership own = plan_ownership_specs(ps, gs, 1, true);
  World we(ps, 11), wg(ps, 11);
  Tabs te, tg;
  EngOptions o;
  Engine ee(gs, we.bind(engine_layout(ps, gs, own, 0)), {}, {}, o);
  Engine eg(gs, wg.bind(engine_layout(ps, gs, own, 0)), {}, {}, o);
  for (auto [e, t] : {std::pair{&ee, &te}, std::pair{&eg, &tg}}) {
    e->bind(nullptr);
    t->bind(*e);
  }
  cudaStream_t s;
  NM_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  ee.prepare(s);
  eg.prepare(s);
  // eager: the clip sums per tensor (the trainer's per-bucket order, last tensor first), then the rest of the step
  for (int i = 0; i < 3; ++i) {
    for (int t = static_cast<int>(ps.size()) - 1; t >= 0; --t) ee.record_sumsq_tensor(t, s);
    ee.record_clip_sums(s);
    ee.record_update(s);
    ee.record_adamh_w(s);
    ee.record_finish(s, true);
  }
  cudaGraph_t graph;
  NM_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  eg.record_step(s, nullptr, true);
  NM_CUDA(cudaStreamEndCapture(s, &graph));
  cudaGraphExec_t exec;
  NM_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 3; ++i) NM_CUDA(cudaGraphLaunch(exec, s));
  NM_CUDA(cudaStreamSynchronize(s));
  int32_t ke = 0, kg = 0;
  NM_CUDA(cudaMemcpy(&ke, te.step, 4, cudaMemcpyDeviceToHost));
  NM_CUDA(cudaMemcpy(&kg, tg.step, 4, cudaMemcpyDeviceToHost));
  CHECK_EQ(ke, 3);
  CHECK_EQ(kg, 3);
  for (size_t i = 0; i < ps.size(); ++i) {
    const auto x = we.p[i].down(), y = wg.p[i].down();
    CHECK(std::memcmp(x.data(), y.data(), x.size() * 4) == 0);
    const auto a = we.s0[i].down(), b = wg.s0[i].down();
    CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
}

NM_TEST(s6_rowpart, frozen_blocks_lr_mult_own_norms) {
  if (!have_gpu()) SKIP("no GPU");
  // One Adam and one AdamH tensor [96, 64]; blocks: rows [0, 32) block 0 (lr x 0.5, own norm), [32, 64) block 1
  // (frozen), [64, 96) block 2 (plain, shares the tensor norm).
  const int R = 96, Cc = 64;
  std::vector<EngGroup> gs = groups130();
  gs[1].eps = 1e-20, gs[2].eps = 1e-20;
  RowPartition rp;
  rp.block_of_row.resize(R);
  for (int r = 0; r < R; ++r) rp.block_of_row[r] = r / 32;
  rp.blocks.resize(3);
  rp.blocks[0].lr_table = 2;
  rp.blocks[0].own_norm = true;
  rp.blocks[1].frozen = true;
  rp.blocks[2].own_norm = false;
  std::vector<DevBuf> p, g, m, v;
  std::vector<std::vector<float>> hp, hg, hm, hv;
  for (int i = 0; i < 2; ++i) {
    hp.push_back(randn(R * Cc, 100 + i, 0.02f));
    hg.push_back(randn(R * Cc, 200 + i, 1e-3f));
    hm.push_back(randn(R * Cc, 300 + i, 1e-4f));
    hv.push_back(randn(R * Cc, 400 + i, 1e-3f));
    for (auto& x : hv.back()) x = x * x;
    p.emplace_back(R * Cc), g.emplace_back(R * Cc), m.emplace_back(R * Cc), v.emplace_back(R * Cc);
    p.back().up(hp.back()), g.back().up(hg.back()), m.back().up(hm.back()), v.back().up(hv.back());
  }
  std::vector<EngTensor> ts(2);
  for (int i = 0; i < 2; ++i) {
    ts[i].name = i ? "adamh" : "adam";
    ts[i].group = i ? 1 : 2;
    ts[i].rows = R, ts[i].cols = Cc;
    ts[i].p32 = p[i].p, ts[i].g = g[i].p, ts[i].s0 = m[i].p, ts[i].s1 = v[i].p;
    ts[i].row_partition = 0;
    ts[i].owned = {{0, R}};
  }
  Tabs t;
  DevBuf mult(4);
  mult.up({0.5f, 0.5f, 0.5f, 0.5f});
  EngOptions o;
  Engine e(gs, ts, {rp}, {nullptr}, o);
  e.bind(nullptr);
  e.set_tables({t.lr.p, t.alr.p, mult.p}, t.bc1.p, t.bc2.p, t.step, 4);
  e.prepare(nullptr);
  e.record_step(nullptr, nullptr, false);
  NM_CUDA(cudaDeviceSynchronize());
  CHECK_EQ(e.n_slots(), 3);   // the tensor, block 0 and the frozen block 1 (own norm, empty)
  // host f64 reference (k = 0; no clip: the gradients are small)
  const double b1 = 0.9, b2 = 0.98, bc1 = 0.1f, bc2 = 0.02f, eps = 1e-20;
  double clip_sq[3] = {0, 0, 0};
  for (int i = 0; i < 2; ++i)
    for (int r = 0; r < R; ++r)
      if (r / 32 != 1)
        for (int c = 0; c < Cc; ++c) clip_sq[i ? 1 : 2] += double(hg[i][r * Cc + c]) * hg[i][r * Cc + c];
  CHECK(std::sqrt(clip_sq[1]) < 1.0 && std::sqrt(clip_sq[2]) < 1.0);
  const auto cs = d2h(e.clip_sumsq(), e.n_clip());
  CHECK(std::fabs(cs[1] - clip_sq[1]) <= 1e-6 * clip_sq[1]);
  CHECK(std::fabs(cs[2] - clip_sq[2]) <= 1e-6 * clip_sq[2]);
  std::vector<double> a[2], mu[2], nu[2];
  for (int i = 0; i < 2; ++i) {
    a[i].resize(R * Cc), mu[i].resize(R * Cc), nu[i].resize(R * Cc);
    for (int j = 0; j < R * Cc; ++j) {
      const double gg = hg[i][j];
      mu[i][j] = (1 - b1) * gg + b1 * hm[i][j];
      nu[i][j] = (1 - b2) * gg * gg + b2 * hv[i][j];
      a[i][j] = (mu[i][j] / bc1) / (std::sqrt(nu[i][j] / bc2) + eps);
    }
  }
  double worst_adam = 0.0, worst_adamh = 0.0;
  // Adam: u = -lr * mult * a (|u_got - u| within 1e-6 |u| (|u| >= lr: a is O(1) or larger) plus the rounding of
  // p + u); frozen rows unchanged
  {
    const auto np = p[0].down(), nm_ = m[0].down();
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < Cc; ++c) {
        const int j = r * Cc + c;
        if (r / 32 == 1) {
          CHECK_EQ(np[j], hp[0][j]);
          CHECK_EQ(nm_[j], hm[0][j]);
          continue;
        }
        const double lr = double(0.008f) * (r / 32 == 0 ? 0.5 : 1.0);
        const double want = -lr * a[0][j];
        const double err = std::fabs((double(np[j]) - hp[0][j]) - want);
        worst_adam = std::max(worst_adam, err / (1e-6 * std::max(std::fabs(want), lr) +
                                                 std::ldexp(std::fabs(hp[0][j]), -23)));
      }
  }
  // AdamH: block 0 its own norms, block 2 the tensor slot (rows 64..95 only: frozen rows enter no norm)
  {
    const auto np = p[1].down();
    for (int blk : {0, 2}) {
      double pp = 0, aa = 0;
      for (int r = blk * 32; r < blk * 32 + 32; ++r)
        for (int c = 0; c < Cc; ++c)
          pp += double(hp[1][r * Cc + c]) * hp[1][r * Cc + c], aa += a[1][r * Cc + c] * a[1][r * Cc + c];
      const double n = std::sqrt(pp), mm = std::sqrt(aa), eta = double(0.02f) * (blk == 0 ? 0.5 : 1.0);
      double ww = 0;
      std::vector<double> wv(32 * Cc);
      for (int j = 0; j < 32 * Cc; ++j) {
        const int k = blk * 32 * Cc + j;
        wv[j] = hp[1][k] - eta * a[1][k] * n / mm;
        ww += wv[j] * wv[j];
      }
      double got_n = 0;
      for (int j = 0; j < 32 * Cc; ++j) {
        const int k = blk * 32 * Cc + j;
        const double want = wv[j] / std::sqrt(ww) * n;
        worst_adamh = std::max(worst_adamh, std::fabs(np[k] - want) / std::fabs(want));
        got_n += double(np[k]) * np[k];
      }
      CHECK(std::fabs(std::sqrt(got_n) - n) <= 1e-6 * n);   // the block norm is preserved
    }
    for (int r = 32; r < 64; ++r)
      for (int c = 0; c < Cc; ++c) CHECK_EQ(np[r * Cc + c], hp[1][r * Cc + c]);
  }
  std::printf("  Adam worst |du - u| / (1e-6 max(|u|, lr) + |p| 2^-23) = %.3f; AdamH worst element rel = %.3e\n", worst_adam,
              worst_adamh);
  CHECK(worst_adam <= 1.0);
  CHECK(worst_adamh <= 2e-6);
}

// The scalar fallback of the elementwise kernels (a slice that is not 16-byte aligned or not a multiple of 4 long)
// against the float4 path on the same values: Adam bitwise (elementwise math), AdamH within 1e-6 (its norms sum in
// another order); the MuonH NS normalization of an odd-sized matrix is covered by tests/s6/test_ns.py.
NM_TEST(s6_rowpart, scalar_and_vector_paths) {
  if (!have_gpu()) SKIP("no GPU");
  const int R = 96, Cc = 64, n = R * Cc;
  const auto gs = groups130();
  std::vector<std::vector<float>> hp, hg, hm, hv;
  for (int i = 0; i < 2; ++i) {
    hp.push_back(randn(n, 500 + i, 0.02f));
    hg.push_back(randn(n, 600 + i, 1e-3f));
    hm.push_back(randn(n, 700 + i, 1e-4f));
    hv.push_back(randn(n, 800 + i, 1e-3f));
    for (auto& x : hv.back()) x = x * x;
  }
  std::vector<float> out[2][2];   // [aligned, unaligned][tensor] params after the step
  for (int shift = 0; shift < 2; ++shift) {
    std::vector<DevBuf> p, g, m, v;
    std::vector<EngTensor> ts(2);
    for (int i = 0; i < 2; ++i) {
      p.emplace_back(n + 4), g.emplace_back(n + 4), m.emplace_back(n + 4), v.emplace_back(n + 4);
      auto pad = [&](const std::vector<float>& x) {
        std::vector<float> y(n + 4, 0.0f);
        std::copy(x.begin(), x.end(), y.begin() + shift);
        return y;
      };
      p.back().up(pad(hp[i])), g.back().up(pad(hg[i])), m.back().up(pad(hm[i])), v.back().up(pad(hv[i]));
      ts[i].name = i ? "adamh" : "adam";
      ts[i].group = i ? 1 : 2;
      ts[i].rows = R, ts[i].cols = Cc;
      ts[i].p32 = p[i].p + shift, ts[i].g = g[i].p + shift, ts[i].s0 = m[i].p + shift, ts[i].s1 = v[i].p + shift;
      ts[i].owned = {{0, R}};
    }
    Tabs t;
    EngOptions o;
    Engine e(gs, ts, {}, {}, o);
    e.bind(nullptr);
    t.bind(e);
    e.prepare(nullptr);
    e.record_step(nullptr, nullptr, false);
    NM_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i) {
      const auto all = p[i].down();
      out[shift][i].assign(all.begin() + shift, all.begin() + shift + n);
    }
  }
  CHECK(std::memcmp(out[0][0].data(), out[1][0].data(), n * sizeof(float)) == 0);
  double d2 = 0, n2 = 0;
  for (int j = 0; j < n; ++j) {
    const double u0 = double(out[0][1][j]) - hp[1][j], u1 = double(out[1][1][j]) - hp[1][j];
    d2 += (u0 - u1) * (u0 - u1), n2 += u0 * u0;
  }
  std::printf("  AdamH update, scalar vs float4 path: rel %.3e\n", std::sqrt(d2 / n2));
  CHECK(std::sqrt(d2 / n2) <= 1e-5);
}

// ------------------------------------------------------------------------------------------------ MuonH edge tiles
namespace {
uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
// f32 arithmetic of the host reference, one rounding per operation (volatile: no contraction into an FMA)
float fmul(float a, float b) { volatile float r = a * b; return r; }
float fadd(float a, float b) { volatile float r = a + b; return r; }

// optim.md §4 in f64 for one matrix u [out, in] (row-major) after the momentum: the NS output O [out, in].
std::vector<double> ns_f64(const std::vector<double>& u, int out, int in) {
  const bool tr = out > in;
  const int m = tr ? in : out, n = tr ? out : in;
  double uu = 0;
  for (double x : u) uu += x * x;
  const double den = std::sqrt(uu) + double(1e-5f);
  std::vector<double> X(size_t(m) * n), A(size_t(m) * m), B(size_t(m) * m), X2(size_t(m) * n);
  for (int r = 0; r < out; ++r)
    for (int c = 0; c < in; ++c) (tr ? X[size_t(c) * n + r] : X[size_t(r) * n + c]) = u[size_t(r) * in + c] / den;
  const double a = double(3.4445f), b = double(-4.7750f), cc = double(2.0315f);
  for (int it = 0; it < 5; ++it) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += X[size_t(i) * n + k] * X[size_t(j) * n + k];
        A[size_t(i) * m + j] = s;
      }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) {
        double s = 0;
        for (int k = 0; k < m; ++k) s += A[size_t(i) * m + k] * A[size_t(k) * m + j];
        B[size_t(i) * m + j] = b * A[size_t(i) * m + j] + cc * s;
      }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int k = 0; k < m; ++k) s += B[size_t(i) * m + k] * X[size_t(k) * n + j];
        X2[size_t(i) * n + j] = a * X[size_t(i) * n + j] + s;
      }
    std::swap(X, X2);
  }
  std::vector<double> O(size_t(out) * in);
  for (int r = 0; r < out; ++r)
    for (int c = 0; c < in; ++c) O[size_t(r) * in + c] = tr ? X[size_t(c) * n + r] : X[size_t(r) * n + c];
  return O;
}
}  // namespace

// MuonH tensors whose matrices are not multiples of the 64 x 64 tile (nor of its 32 x 32 sub-tiles): [2 x 100, 37]
// (OUT > IN: the transposed NS orientation) and [2 x 33, 65], two layer matrices each, true-f32 NS. The momentum
// buffer is elementwise (bitwise against host f32); the update (target - p) per matrix against an f64 reference of the
// whole chain (rel <= 2e-5: the f32 NS error and the renormalization's rounding); every element is written.
NM_TEST(s6_rowpart, muonh_edge_tiles) {
  if (!have_gpu()) SKIP("no GPU");
  const auto gs = groups130();
  struct Shp { int out, in; };
  const Shp shp[2] = {{100, 37}, {33, 65}};
  const int L = 2;
  std::vector<DevBuf> p, g, b, u;
  std::vector<std::vector<float>> hp, hg, hb;
  std::vector<EngTensor> ts(2);
  for (int i = 0; i < 2; ++i) {
    const size_t n = size_t(L) * shp[i].out * shp[i].in;
    hp.push_back(randn(n, 900 + i, 0.02f));
    hg.push_back(randn(n, 910 + i, 1e-3f));
    hb.push_back(randn(n, 920 + i, 1e-4f));
    p.emplace_back(n), g.emplace_back(n), b.emplace_back(n), u.emplace_back(n);
    p[i].up(hp[i]), g[i].up(hg[i]), b[i].up(hb[i]);
    NM_CUDA(cudaMemset(u[i].p, 0xff, n * sizeof(float)));   // NaN: an element the kernels skip stays NaN
    ts[i].name = "edge" + std::to_string(i);
    ts[i].group = 0;
    ts[i].rows = int64_t(L) * shp[i].out, ts[i].cols = shp[i].in;
    ts[i].p32 = p[i].p, ts[i].g = g[i].p, ts[i].s0 = b[i].p, ts[i].u_out = u[i].p;
    for (int l = 0; l < L; ++l)
      ts[i].mats.push_back({"m" + std::to_string(l), int64_t(l) * shp[i].out, shp[i].out, {}});
    ts[i].owned = {{0, ts[i].rows}};
  }
  Tabs t;
  EngOptions o;
  o.ns = optk::NsPrec::F32;
  Engine e(gs, ts, {}, {}, o);
  e.bind(nullptr);
  t.bind(e);
  e.prepare(nullptr);
  e.record_step(nullptr, nullptr, false);
  NM_CUDA(cudaDeviceSynchronize());
  // no clip (the muonh clip norm is < 1)
  double gg = 0;
  for (const auto& v : hg)
    for (float x : v) gg += double(x) * x;
  CHECK(std::sqrt(gg) < 1.0);
  const float mu = 0.95f, eta = 0.02f;
  double worst = 0;
  for (int i = 0; i < 2; ++i) {
    const int out = shp[i].out, in = shp[i].in, mn = out * in;
    const auto nb = b[i].down(), nu_ = u[i].down();
    size_t bad_buf = 0;
    for (int l = 0; l < L; ++l) {
      std::vector<double> uu(mn), P(mn);
      for (int j = 0; j < mn; ++j) {
        const size_t k = size_t(l) * mn + j;
        const float buf = fadd(fmul(mu, hb[i][k]), hg[i][k]);
        bad_buf += fbits(nb[k]) != fbits(buf);
        uu[j] = double(mu) * buf + hg[i][k];
        P[j] = hp[i][k];
      }
      const auto O = ns_f64(uu, out, in);
      double pp = 0, oo = 0;
      for (int j = 0; j < mn; ++j) pp += P[j] * P[j], oo += O[j] * O[j];
      const double nrm = std::sqrt(pp), mm = std::max(std::sqrt(oo), 1e-10);
      std::vector<double> W(mn);
      double ww = 0;
      for (int j = 0; j < mn; ++j) W[j] = P[j] - double(eta) * O[j] * nrm / mm, ww += W[j] * W[j];
      double d2 = 0, n2 = 0;
      for (int j = 0; j < mn; ++j) {
        const double want = W[j] / std::sqrt(ww) * nrm - P[j];
        const double got = nu_[size_t(l) * mn + j];
        d2 += (got - want) * (got - want), n2 += want * want;   // NaN (an unwritten element) fails the bound
      }
      const double r = std::sqrt(d2 / n2);
      std::printf("  [%d x %d] layer %d: update rel to f64 %.3e\n", out, in, l, r);
      worst = std::isnan(r) ? INFINITY : std::max(worst, r);
    }
    CHECK_EQ(bad_buf, size_t(0));
  }
  CHECK(worst <= 2e-5);
}

// The one-tensor C ABI entry points (capi.h: b1, b2, eps, mu are f32 arguments). One step from mu = nu = 0 at g = 0.5:
// mu = f32(1 - 0.9) * 0.5 and nu = f32(1 - 0.98) * 0.25 carry optim.md §5.3 rule 1's constants 0x3DCCCCCD /
// 0x3CA3D70A (not f32(1 - f32(0.9)) = 0x3DCCCCD0); Adam's parameters and the MuonH momentum buffer bitwise against
// host f32 in the kernels' operation order; AdamH and MuonH keep their norms.
NM_TEST(s6_rowpart, capi_one_tensor_bits) {
  if (!have_gpu()) SKIP("no GPU");
  const int R = 64, Cc = 64, n = R * Cc;
  const float b1 = 0.9f, b2 = 0.98f, eps = 1e-20f, lr = 0.008f;
  const float omb1 = static_cast<float>(1.0 - 0.9), omb2 = static_cast<float>(1.0 - 0.98);
  CHECK_EQ(fbits(omb1), 0x3DCCCCCDu);
  CHECK_EQ(fbits(omb2), 0x3CA3D70Au);
  const float mu_want = fmul(omb1, 0.5f), nu_want = fmul(omb2, fmul(0.5f, 0.5f));
  const auto hp = randn(n, 1000, 0.02f);
  for (int rule = 0; rule < 2; ++rule) {   // 0 Adam, 1 AdamH
    DevBuf p(n), g(n), m(n), v(n);
    p.up(hp), g.up(std::vector<float>(n, 0.5f)), m.up(std::vector<float>(n, 0.0f)), v.up(std::vector<float>(n, 0.0f));
    const int rc = rule == 0 ? nm_adam_update(p.p, nullptr, g.p, m.p, v.p, R, Cc, 1, lr, b1, b2, eps, 0.0f, 0, 1.0f,
                                              nullptr, nullptr, nullptr, 0, nullptr)
                             : nm_adamh_update(p.p, nullptr, g.p, m.p, v.p, R, Cc, 1, 0.02f, b1, b2, eps, 0.0f, 0, 1.0f,
                                               nullptr, nullptr, nullptr, 0, nullptr);
    CHECK_EQ(rc, 0);
    const auto nm_ = m.down(), nv = v.down(), np = p.down();
    size_t bad = 0;
    for (int j = 0; j < n; ++j) bad += fbits(nm_[j]) != fbits(mu_want) || fbits(nv[j]) != fbits(nu_want);
    std::printf("  %s: mu bits %08x (want %08x), nu bits %08x (want %08x)\n", rule ? "AdamH" : "Adam", fbits(nm_[0]),
                fbits(mu_want), fbits(nv[0]), fbits(nu_want));
    CHECK_EQ(bad, size_t(0));
    if (rule == 0) {   // p + (-lr) * mu / (bc1 * (sqrt(nu / bc2) + eps)), bc = 1 - f32(b)^1
      const float bc1 = 1.0f - b1, bc2 = 1.0f - b2;
      const float a = mu_want / fmul(bc1, fadd(std::sqrt(nu_want / bc2), eps));
      size_t badp = 0;
      for (int j = 0; j < n; ++j) badp += fbits(np[j]) != fbits(fadd(hp[j], fmul(-lr, a)));
      CHECK_EQ(badp, size_t(0));
    } else {
      double n0 = 0, n1 = 0;
      for (int j = 0; j < n; ++j) n0 += double(hp[j]) * hp[j], n1 += double(np[j]) * np[j];
      CHECK(std::fabs(std::sqrt(n1) - std::sqrt(n0)) <= 1e-6 * std::sqrt(n0));
    }
  }
  // MuonH (2 layers of [32, 64]): buf = f32(mu * buf + g) bitwise; each layer's Frobenius norm kept
  {
    const int Lm = 2, out = 32;
    const auto hb = randn(n, 1001, 1e-3f), hg = randn(n, 1002, 1e-3f);
    DevBuf p(n), g(n), b(n);
    p.up(hp), g.up(hg), b.up(hb);
    CHECK_EQ(nm_muonh_update(p.p, nullptr, g.p, b.p, Lm, out, Cc, 0.95f, 0.02f, 1e-5f, 1, 0.0f, 0, 1.0f, nullptr,
                             nullptr),
             0);
    const auto nb = b.down(), np = p.down();
    size_t bad = 0;
    for (int j = 0; j < n; ++j) bad += fbits(nb[j]) != fbits(fadd(fmul(0.95f, hb[j]), hg[j]));
    CHECK_EQ(bad, size_t(0));
    for (int l = 0; l < Lm; ++l) {
      double n0 = 0, n1 = 0;
      for (int j = l * out * Cc; j < (l + 1) * out * Cc; ++j) n0 += double(hp[j]) * hp[j], n1 += double(np[j]) * np[j];
      CHECK(std::fabs(std::sqrt(n1) - std::sqrt(n0)) <= 1e-6 * std::sqrt(n0));
    }
  }
}

// Lazy Adam rows and decoupled weight decay (implemented, unused by the baseline): an Adam tensor [64, 64] with a
// one-block lazy row partition (even rows touched) and weight_decay 0.1. Untouched rows keep p, mu and nu bitwise;
// touched rows equal host f32 in the kernels' order: a = mu / (bc1 * (sqrt(nu / bc2) + eps)) + wd * p (optax
// add_decayed_weights), u = (-lr) * a, p + u.
NM_TEST(s6_rowpart, lazy_adam_weight_decay) {
  if (!have_gpu()) SKIP("no GPU");
  const int R = 64, Cc = 64, n = R * Cc;
  auto gs = groups130();
  gs[2].weight_decay = 0.1;
  const auto hp = randn(n, 1100, 0.02f), hg = randn(n, 1101, 1e-3f), hm = randn(n, 1102, 1e-4f);
  auto hv = randn(n, 1103, 1e-3f);
  for (auto& x : hv) x = x * x;
  DevBuf p(n), g(n), m(n), v(n);
  p.up(hp), g.up(hg), m.up(hm), v.up(hv);
  std::vector<uint8_t> touched(R);
  for (int r = 0; r < R; ++r) touched[r] = r % 2 == 0;
  uint8_t* d_touched = nullptr;
  NM_CUDA(cudaMalloc(&d_touched, R));
  NM_CUDA(cudaMemcpy(d_touched, touched.data(), R, cudaMemcpyHostToDevice));
  RowPartition rp;
  rp.block_of_row.assign(R, 0);
  rp.blocks.resize(1);
  rp.lazy = true;
  EngTensor t;
  t.name = "adam_lazy";
  t.group = 2;
  t.rows = R, t.cols = Cc;
  t.p32 = p.p, t.g = g.p, t.s0 = m.p, t.s1 = v.p;
  t.row_partition = 0;
  t.owned = {{0, R}};
  Tabs tb;
  Engine e(gs, {t}, {rp}, {d_touched}, EngOptions{});
  e.bind(nullptr);
  tb.bind(e);
  e.prepare(nullptr);
  e.record_step(nullptr, nullptr, false);
  NM_CUDA(cudaDeviceSynchronize());
  cudaFree(d_touched);
  const auto np = p.down(), nm_ = m.down(), nv = v.down();
  const float omb1 = static_cast<float>(1.0 - 0.9), b1 = 0.9f, omb2 = static_cast<float>(1.0 - 0.98), b2 = 0.98f;
  const float bc1 = 0.1f, bc2 = 0.02f, eps = static_cast<float>(gs[2].eps), wd = 0.1f, lr = 0.008f;
  size_t bad_skip = 0, bad_upd = 0;
  for (int j = 0; j < n; ++j) {
    if (!touched[j / Cc]) {
      bad_skip += fbits(np[j]) != fbits(hp[j]) || fbits(nm_[j]) != fbits(hm[j]) || fbits(nv[j]) != fbits(hv[j]);
      continue;
    }
    const float mu = fadd(fmul(omb1, hg[j]), fmul(b1, hm[j]));
    const float nu = fadd(fmul(omb2, fmul(hg[j], hg[j])), fmul(b2, hv[j]));
    const float a = fadd(mu / fmul(bc1, fadd(std::sqrt(nu / bc2), eps)), fmul(wd, hp[j]));
    bad_upd += fbits(nm_[j]) != fbits(mu) || fbits(nv[j]) != fbits(nu) ||
               fbits(np[j]) != fbits(fadd(hp[j], fmul(-lr, a)));
  }
  std::printf("  untouched rows changed: %zu elements; touched rows off host f32: %zu elements\n", bad_skip, bad_upd);
  CHECK_EQ(bad_skip, size_t(0));
  CHECK_EQ(bad_upd, size_t(0));
}

// Configurations the engine refuses: Adam / AdamH groups with different betas (one bias-correction table pair), lazy
// AdamH rows (whole-tensor norms), tables of 0 entries.
NM_TEST(s6_rowpart, refusals) {
  if (!have_gpu()) SKIP("no GPU");
  DevBuf p(64 * 64), g(64 * 64), m(64 * 64), v(64 * 64);
  auto tensor = [&](int group) {
    EngTensor t;
    t.name = "t";
    t.group = group;
    t.rows = 64, t.cols = 64;
    t.p32 = p.p, t.g = g.p, t.s0 = m.p, t.s1 = v.p;
    t.owned = {{0, 64}};
    return t;
  };
  auto gs = groups130();
  gs[2].b2 = 0.95;
  CHECK_THROWS(Engine(gs, {tensor(1), tensor(2)}, {}, {}, EngOptions{}), "different b1 / b2");
  gs = groups130();
  RowPartition rp;
  rp.block_of_row.assign(64, 0);
  rp.blocks.resize(1);
  uint8_t* touched = nullptr;
  NM_CUDA(cudaMalloc(&touched, 64));
  EngTensor t1 = tensor(1);
  t1.row_partition = 0;
  CHECK_THROWS(Engine(gs, {t1}, {rp}, {touched}, EngOptions{}), "lazy");
  EngTensor t2 = tensor(2);   // lazy Adam is allowed
  t2.row_partition = 0;
  Engine ok(gs, {t2}, {rp}, {touched}, EngOptions{});
  ok.bind(nullptr);
  Tabs t;
  CHECK_THROWS(ok.set_tables({t.lr.p, t.alr.p}, t.bc1.p, t.bc2.p, t.step, 0), "entries");
  cudaFree(touched);
}

// ------------------------------------------------------------------------------------------------ NS algorithms
// ZeRO-1 ranks own different counts of each MuonH shape, so "1 rank == N ranks bitwise" needs every batch count of a
// shape to run the same GEMM kernel. For every oriented NS shape (m <= n) of the four sizes, in TF32 and true f32,
// full and symmetric-block products: the plan at batch 1, 2, 3, 6 and the engine's 1 GiB chunk capacity chooses the
// same cuBLASLt algorithm configuration (NsGemm::algo_config: kernel, tile, stages, cluster, ...) for every GEMM of
// the iteration, with reduction scheme NONE (no split-K). Not the opaque algorithm bytes: on the H100 the first
// heuristic call of a process leaves garbage in a field outside the configuration (same kernel, other bytes).
NM_TEST(s6_nsalgo, shape_only) {
  if (!have_gpu()) SKIP("no GPU");
  struct Shape { const char* size; int64_t m, n; };
  const Shape shapes[] = {{"130m", 512, 512},   {"130m", 512, 1792},  {"300m", 768, 768},   {"300m", 768, 2688},
                          {"520m", 1024, 1024}, {"520m", 512, 1024},  {"520m", 1024, 3584}, {"1_2b", 2048, 2048},
                          {"1_2b", 1024, 2048}, {"1_2b", 2048, 7168}};
  const EngOptions o;
  int checked = 0, fallbacks = 0;
  for (const Shape& sh : shapes) {
    const size_t per = size_t(2 * sh.m * sh.n + 3 * sh.m * sh.m) * sizeof(float);
    const int cap = static_cast<int>(std::max<size_t>(1, o.ns_chunk_bytes / per));
    for (optk::NsPrec prec : {optk::NsPrec::TF32, optk::NsPrec::F32})
      for (bool sym : {false, true}) {
        std::vector<std::vector<int64_t>> ref;
        for (int batch : {1, 2, 3, 6, cap}) {
          optk::NsGemm gm(prec);
          const optk::NsIterPlan pl = optk::plan_ns_iteration(gm, sh.m, sh.n, batch, float(o.ns_a), float(o.ns_b),
                                                              float(o.ns_c), sym);
          gm.choose(false, nullptr);
          std::vector<int> ids;
          for (int i = 0; i < 3; ++i)
            for (int id : {pl.g1[i], pl.g2[i]})
              if (id >= 0) ids.push_back(id);
          ids.push_back(pl.g3);
          std::vector<std::vector<int64_t>> got;
          for (int id : ids) {
            got.push_back(gm.algo_config(id));
            fallbacks += gm.batch_fallback(id);
            if (gm.reduction_scheme(id) != CUBLASLT_REDUCTION_SCHEME_NONE) {
              std::printf("  %s %ldx%ld prec %d sym %d batch %d: GEMM %d uses split-K (scheme %u)\n", sh.size,
                          long(sh.m), long(sh.n), int(prec), int(sym), batch, id, gm.reduction_scheme(id));
              CHECK(false);
            }
          }
          if (ref.empty()) {
            ref = got;
          } else if (got != ref) {
            std::printf("  %s %ldx%ld prec %d sym %d: batch %d chooses other algorithms than batch 1\n", sh.size,
                        long(sh.m), long(sh.n), int(prec), int(sym), batch);
            for (size_t i = 0; i < got.size(); ++i)
              if (got[i] != ref[i]) {
                std::printf("    GEMM %zu config (id tile stages splitk red swz cust inner cluster): batch 1", i);
                for (int64_t v : ref[i]) std::printf(" %ld", long(v));
                std::printf(" / batch %d", batch);
                for (int64_t v : got[i]) std::printf(" %ld", long(v));
                std::printf("\n");
              }
            CHECK(false);
          }
          ++checked;
        }
      }
  }
  std::printf("  %d plans checked, %d GEMMs from the batch-dependent fallback\n", checked, fallbacks);
}

// ------------------------------------------------------------------------------------------------ table bounds
// The lr and bias-correction tables have N entries: a step at k = N - 1 runs, a step at k = N traps (the stream reports
// a launch failure) instead of reading past them. Last in its process: the trap ends the CUDA context.
NM_TEST(s6_bounds, step_past_tables_traps) {
  if (!have_gpu()) SKIP("no GPU");
  const auto ps = baseline_specs(1, 512, 8, 8, 64, 1792, 1024);
  const auto gs = groups130();
  World w(ps, 13);
  Tabs t;
  Engine e(gs, w.bind(engine_layout(ps, gs, plan_ownership_specs(ps, gs, 1, true), 0)), {}, {}, EngOptions{});
  e.bind(nullptr);
  t.bind(e);
  e.prepare(nullptr);
  t.reset(3);
  e.record_step(nullptr, nullptr, true);
  NM_CUDA(cudaDeviceSynchronize());
  int32_t k = 0;
  NM_CUDA(cudaMemcpy(&k, t.step, 4, cudaMemcpyDeviceToHost));
  CHECK_EQ(k, 4);
  std::string err;
  try {
    e.record_step(nullptr, nullptr, true);
    NM_CUDA(cudaDeviceSynchronize());
  } catch (const std::exception& x) {
    err = x.what();
  }
  std::printf("  k = N: %s\n", err.empty() ? "no error" : err.c_str());
  CHECK(err.find("launch failure") != std::string::npos || err.find("illegal instruction") != std::string::npos);
}
