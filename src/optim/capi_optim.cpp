// C ABI of the optimizer (capi.h optimizer block and src/optim/capi_optim.h). Test path: the one-tensor entry points
// build a one-tensor engine per call and synchronize the stream before returning; the trainer records the engine's
// phases into g_step instead.
#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/capi_util.hpp"
#include "kernels/optim/ns_gemm.hpp"
#include "kernels/optim/optim_dev.hpp"
#include "marincpp/capi.h"
#include "optim/capi_optim.h"
#include "optim/engine.hpp"
#include "sched/sched.hpp"

namespace nm {
namespace {

cudaStream_t S(void* p) { return static_cast<cudaStream_t>(p); }

optk::NsPrec ns_prec(int p) {
  NM_REQUIRE(p >= 0 && p <= 2, "ns precision must be 0 (f32), 1 (tf32) or 2 (3xtf32)");
  return static_cast<optk::NsPrec>(p);
}

// Device scratch of one C ABI call, freed after the stream is synchronized.
class Scratch {
 public:
  explicit Scratch(cudaStream_t s) : s_(s) {}
  ~Scratch() {
    cudaStreamSynchronize(s_);
    for (void* p : ptrs_) cudaFree(p);
  }
  template <class T>
  T* alloc(size_t n) {
    void* p = nullptr;
    NM_CUDA(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    ptrs_.push_back(p);
    return static_cast<T*>(p);
  }
  template <class T>
  T* upload(const std::vector<T>& v) {
    T* d = alloc<T>(v.size());
    NM_CUDA(cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s_));
    return d;
  }

 private:
  cudaStream_t s_;
  std::vector<void*> ptrs_;
};

// The decimal an f32 argument was written as (shortest round-trip digits, then f64): capi.h passes b1 / b2 / eps / mu
// as float, and optim.md §5.3 rule 1 forms f32(1 - b) from the decimal (0.9 -> 0x3DCCCCCD), not from f32(0.9)
// (0x3DCCCCD0). Every f32 of a decimal with <= 7 significant digits maps back to that decimal.
double decimal_of(float x) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf) - 1, x);
  NM_REQUIRE(r.ec == std::errc(), "decimal_of: to_chars failed");
  *r.ptr = '\0';
  return std::strtod(buf, nullptr);
}

float bias_correction(double beta, int64_t t) {
  NM_REQUIRE(t >= 1 && t < sched::kMaxPowExponent, "t must be in [1, 2^24)");
  return 1.0f - sched::xla_powf(static_cast<float>(beta), static_cast<float>(t));   // 1 - b**t, f32
}

// One tensor through a one-group engine at k = 0 of one-entry tables; the clip state is the caller's.
void run_one(opt::EngGroup grp, opt::EngTensor t, std::vector<RowPartition> rps, float lr, float bc1, float bc2,
             const std::vector<float>& block_mult, int ns_precision, double muon_eps, float clip_gn, int clip,
             float max_norm, cudaStream_t s) {
  Scratch sc(s);
  std::vector<const float*> lr_tabs;
  lr_tabs.push_back(sc.upload(std::vector<float>{lr}));
  lr_tabs.push_back(lr_tabs[0]);
  for (float m : block_mult) lr_tabs.push_back(sc.upload(std::vector<float>{m}));
  const float* d_bc1 = sc.upload(std::vector<float>{bc1});
  const float* d_bc2 = sc.upload(std::vector<float>{bc2});
  int32_t* d_step = sc.upload(std::vector<int32_t>{0});
  const float clip_factor = clip ? max_norm / clip_gn : 1.0f;
  optk::ClipDev* d_clip = sc.upload(std::vector<optk::ClipDev>{{clip_gn, clip, max_norm, clip_factor}});
  t.owned = {{0, t.rows}};
  grp.lr_table = 0;
  opt::EngOptions o;
  o.ns = ns_prec(ns_precision);
  o.muon_eps = muon_eps;
  opt::Engine e({grp}, {t}, std::move(rps), {}, o);
  e.bind(nullptr);
  e.set_tables(lr_tabs, d_bc1, d_bc2, d_step, 1);
  e.prepare(s);
  e.record_update(s, d_clip);
  e.record_adamh_w(s);
  e.record_finish(s, false);
  NM_CUDA(cudaStreamSynchronize(s));
}

// Row blocks of the one-tensor Adam / AdamH entry points: block lr multiplier j is lr table 2 + j.
std::vector<RowPartition> row_blocks(const int32_t* block_of_row, const float* block_lr_mult,
                                     const uint8_t* block_frozen, int n_blocks, int64_t rows,
                                     std::vector<float>& mult_host, cudaStream_t s) {
  if (!block_of_row) return {};
  NM_REQUIRE(n_blocks > 0, "n_blocks must be > 0 with block_of_row");
  RowPartition rp;
  rp.block_of_row.resize(rows);
  NM_CUDA(cudaMemcpyAsync(rp.block_of_row.data(), block_of_row, rows * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  mult_host.assign(n_blocks, 1.0f);
  std::vector<uint8_t> frozen(n_blocks, 0);
  if (block_lr_mult)
    NM_CUDA(cudaMemcpyAsync(mult_host.data(), block_lr_mult, n_blocks * sizeof(float), cudaMemcpyDeviceToHost, s));
  if (block_frozen)
    NM_CUDA(cudaMemcpyAsync(frozen.data(), block_frozen, n_blocks, cudaMemcpyDeviceToHost, s));
  NM_CUDA(cudaStreamSynchronize(s));
  for (int32_t b : rp.block_of_row) NM_REQUIRE(b >= 0 && b < n_blocks, "block_of_row out of range");
  for (int b = 0; b < n_blocks; ++b) {
    RowBlock blk;
    blk.lr_table = 2 + b;
    blk.frozen = frozen[b] != 0;
    blk.own_norm = true;
    rp.blocks.push_back(blk);
  }
  rp.param = 0;
  return {rp};
}

opt::EngTensor plain_tensor(float* p, void* p16, const float* g, float* s0, float* s1, int64_t rows, int64_t cols) {
  opt::EngTensor t;
  t.name = "tensor";
  t.group = 0;
  t.rows = rows;
  t.cols = cols;
  t.p32 = p;
  t.p16 = static_cast<bf16_t*>(p16);
  t.g = g;
  t.s0 = s0;
  t.s1 = s1;
  return t;
}

}  // namespace
}  // namespace nm

using nm::capi_call;

extern "C" {

int nm_newton_schulz(int precision, int symmetric, float* x, int64_t batch, int64_t rows, int64_t cols, int steps,
                     float a, float b, float c, float eps, void* workspace, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(workspace == nullptr, "nm_newton_schulz allocates its own workspace (pass NULL)");
    NM_REQUIRE(batch >= 1 && rows >= 1 && cols >= 1 && steps >= 1, "empty NS call");
    const cudaStream_t s = nm::S(stream);
    nm::Scratch sc(s);
    const int64_t m = std::min(rows, cols), n = std::max(rows, cols);
    const int cnt = static_cast<int>(batch);
    nm::optk::NsGemm gemm(nm::ns_prec(precision));
    const nm::optk::NsIterPlan plan =
        nm::optk::plan_ns_iteration(gemm, m, n, cnt, a, b, c, (symmetric & 1) != 0, (symmetric & 2) != 0);
    gemm.choose(false, s);
    gemm.bind_workspace(sc.alloc<char>(gemm.workspace_bytes()));
    float* X = sc.alloc<float>(batch * m * n);
    float* X2 = sc.alloc<float>(batch * m * n);
    float* A = sc.alloc<float>(batch * m * m);
    float* cA = sc.alloc<float>(batch * m * m);
    float* B = sc.alloc<float>(batch * m * m);
    const int64_t tiles = ((rows + nm::optk::kTile - 1) / nm::optk::kTile) * ((cols + nm::optk::kTile - 1) / nm::optk::kTile);
    double* partial = sc.alloc<double>(batch * tiles);
    double* u2 = sc.alloc<double>(batch);
    std::vector<int64_t> seg(batch + 1);
    for (int64_t i = 0; i <= batch; ++i) seg[i] = i * tiles;
    const int64_t* dseg = sc.upload(seg);
    nm::optk::ns_load(x, X, cnt, static_cast<int>(rows), static_cast<int>(cols), partial, s);
    nm::optk::segment_sum(partial, nullptr, dseg, cnt, 1, u2, 0, s);
    nm::optk::ns_normalize(X, cnt, m * n, u2, eps, s);
    X = nm::optk::run_ns_iteration(gemm, plan, steps, X, X2, A, cA, B, s);
    nm::optk::ns_store(X, x, cnt, static_cast<int>(rows), static_cast<int>(cols), s);
    NM_CUDA(cudaStreamSynchronize(s));
  });
}

int nm_muonh_update(float* p, void* p16, const float* g, float* momentum, int64_t layers, int64_t rows, int64_t cols,
                    float mu, float eta, float muon_eps, int ns_precision, float clip_gn, int clip, float max_norm,
                    void* workspace, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(workspace == nullptr, "nm_muonh_update allocates its own workspace (pass NULL)");
    nm::opt::EngGroup grp;
    grp.name = "muonh";
    grp.rule = nm::OptRule::MuonH;
    grp.momentum = nm::decimal_of(mu);
    grp.max_grad_norm = max_norm;
    nm::opt::EngTensor t = nm::plain_tensor(p, p16, g, momentum, nullptr, layers * rows, cols);
    for (int64_t l = 0; l < layers; ++l) t.mats.push_back({"layer" + std::to_string(l), l * rows, rows, {}});
    nm::run_one(grp, t, {}, eta, 1.0f, 1.0f, {}, ns_precision, muon_eps, clip_gn, clip, max_norm, nm::S(stream));
  });
}

int nm_adamh_update(float* p, void* p16, const float* g, float* mu, float* nu, int64_t rows, int64_t cols, int64_t t,
                    float eta, float b1, float b2, float eps, float clip_gn, int clip, float max_norm,
                    const int32_t* block_of_row, const float* block_lr_mult, const uint8_t* block_frozen,
                    int n_blocks, void* stream) {
  return capi_call([&] {
    nm::opt::EngGroup grp;
    grp.name = "adamh";
    grp.rule = nm::OptRule::AdamH;
    grp.b1 = nm::decimal_of(b1), grp.b2 = nm::decimal_of(b2), grp.eps = nm::decimal_of(eps);
    grp.max_grad_norm = max_norm;
    std::vector<float> mult;
    auto rps = nm::row_blocks(block_of_row, block_lr_mult, block_frozen, n_blocks, rows, mult, nm::S(stream));
    nm::opt::EngTensor ten = nm::plain_tensor(p, p16, g, mu, nu, rows, cols);
    ten.row_partition = rps.empty() ? -1 : 0;
    nm::run_one(grp, ten, rps, eta, nm::bias_correction(b1, t), nm::bias_correction(b2, t), mult, 1, 1e-5, clip_gn,
                clip, max_norm, nm::S(stream));
  });
}

int nm_adam_update(float* p, void* p16, const float* g, float* mu, float* nu, int64_t rows, int64_t cols, int64_t t,
                   float adam_lr, float b1, float b2, float eps, float clip_gn, int clip, float max_norm,
                   const int32_t* block_of_row, const float* block_lr_mult, const uint8_t* block_frozen,
                   int n_blocks, void* stream) {
  return capi_call([&] {
    nm::opt::EngGroup grp;
    grp.name = "adam";
    grp.rule = nm::OptRule::Adam;
    grp.b1 = nm::decimal_of(b1), grp.b2 = nm::decimal_of(b2), grp.eps = nm::decimal_of(eps);
    grp.max_grad_norm = max_norm;
    std::vector<float> mult;
    auto rps = nm::row_blocks(block_of_row, block_lr_mult, block_frozen, n_blocks, rows, mult, nm::S(stream));
    nm::opt::EngTensor ten = nm::plain_tensor(p, p16, g, mu, nu, rows, cols);
    ten.row_partition = rps.empty() ? -1 : 0;
    nm::run_one(grp, ten, rps, adam_lr, nm::bias_correction(b1, t), nm::bias_correction(b2, t), mult, 1, 1e-5,
                clip_gn, clip, max_norm, nm::S(stream));
  });
}

int nm_opt_create(const nm_opt_group* groups, int n_groups, const nm_opt_tensor* tensors, int n_tensors,
                  int ns_precision, int ns_steps, double muon_eps, int64_t ns_chunk_bytes, int timed_gemm,
                  int sharded, void** engine) {
  return capi_call([&] {
    NM_REQUIRE(groups && n_groups > 0 && tensors && n_tensors > 0 && engine, "nm_opt_create: empty input");
    std::vector<nm::opt::EngGroup> gs;
    for (int i = 0; i < n_groups; ++i) {
      const nm_opt_group& c = groups[i];
      NM_REQUIRE(c.rule >= 0 && c.rule <= 3, "bad rule");
      nm::opt::EngGroup g;
      g.name = "g" + std::to_string(i);
      g.rule = static_cast<nm::OptRule>(c.rule);
      g.lr_table = c.lr_table;
      g.lr_scale = c.lr_scale, g.b1 = c.b1, g.b2 = c.b2, g.eps = c.eps, g.momentum = c.momentum;
      g.max_grad_norm = c.max_grad_norm, g.weight_decay = c.weight_decay;
      g.clip_group = "c" + std::to_string(c.clip_group);
      gs.push_back(g);
    }
    std::vector<nm::opt::EngTensor> ts;
    for (int i = 0; i < n_tensors; ++i) {
      const nm_opt_tensor& c = tensors[i];
      nm::opt::EngTensor t = nm::plain_tensor(c.p32, c.p16, c.g, c.s0, c.s1, c.rows, c.cols);
      t.name = "t" + std::to_string(i);
      t.group = c.group;
      t.base_row = c.base_row;
      t.grad_scale = c.grad_scale;
      t.update_scale = c.update_scale;
      t.u_out = c.u_out;
      for (int64_t k = 0; k < c.n_mats; ++k) {
        nm::opt::EngMatrix m;
        m.name = t.name + ".m" + std::to_string(k);
        m.out = c.mat_rows;
        if (c.row_map) m.row_map.assign(c.row_map + k * c.mat_rows, c.row_map + (k + 1) * c.mat_rows);
        else m.row0 = k * c.mat_rows;
        t.mats.push_back(std::move(m));
      }
      if (c.n_owned == 0) t.owned = {{0, c.rows}};
      for (int64_t k = 0; k < c.n_owned; ++k) t.owned.push_back({c.owned[2 * k], c.owned[2 * k + 1]});
      ts.push_back(std::move(t));
    }
    nm::opt::EngOptions o;
    o.ns = nm::ns_prec(ns_precision & 3);
    o.ns_symmetric = (ns_precision & 4) != 0;
    o.ns_fold_c = (ns_precision & 8) != 0;
    o.ns_steps = ns_steps;
    o.muon_eps = muon_eps;
    if (ns_chunk_bytes > 0) o.ns_chunk_bytes = static_cast<size_t>(ns_chunk_bytes);
    o.timed_gemm = timed_gemm != 0;
    o.sharded = sharded != 0;
    auto e = std::make_unique<nm::opt::Engine>(std::move(gs), std::move(ts), std::vector<nm::RowPartition>{},
                                               std::vector<const uint8_t*>{}, o);
    e->bind(nullptr);
    *engine = e.release();
  });
}

int nm_opt_set_tables(void* engine, const float* lr_main, const float* lr_adam, const float* bc1, const float* bc2,
                      int32_t* step, int64_t n_steps) {
  return capi_call(
      [&] { static_cast<nm::opt::Engine*>(engine)->set_tables({lr_main, lr_adam}, bc1, bc2, step, n_steps); });
}

int nm_opt_prepare(void* engine, void* stream) {
  return capi_call([&] { static_cast<nm::opt::Engine*>(engine)->prepare(nm::S(stream)); });
}

int nm_opt_phase(void* engine, int phase, void* stream) {
  return capi_call([&] {
    auto* e = static_cast<nm::opt::Engine*>(engine);
    const cudaStream_t s = nm::S(stream);
    switch (phase) {
      case 0: e->record_sumsq(s); break;
      case 1: e->record_update(s); break;
      case 2: e->record_adamh_w(s); break;
      case 3: e->record_finish(s, false); break;
      case 4: e->record_finish(s, true); break;
      default: nm::fatal(NM_WHERE, "phase must be 0..4");
    }
  });
}

int nm_opt_step(void* engine, int advance, void* stream) {
  return capi_call([&] {
    auto* e = static_cast<nm::opt::Engine*>(engine);
    NM_REQUIRE(!e->sharded(), "nm_opt_step runs an unsharded engine; drive a sharded one by phases");
    e->record_step(nm::S(stream), nullptr, advance != 0);
  });
}

int nm_opt_buffers(void* engine, double** clip_sumsq, int* n_clip, double** slot_pa, double** slot_w, int* n_slots,
                   void** clip_states, int* n_mats, int* n_chunks) {
  return capi_call([&] {
    auto* e = static_cast<nm::opt::Engine*>(engine);
    if (clip_sumsq) *clip_sumsq = e->clip_sumsq();
    if (n_clip) *n_clip = e->n_clip();
    if (slot_pa) *slot_pa = e->slot_pa();
    if (slot_w) *slot_w = e->slot_w();
    if (n_slots) *n_slots = e->n_slots();
    if (clip_states) *clip_states = const_cast<nm::optk::ClipDev*>(e->clip_states());
    if (n_mats) *n_mats = e->n_matrices();
    if (n_chunks) *n_chunks = e->n_ns_chunks();
  });
}

int nm_opt_destroy(void* engine) {
  return capi_call([&] { delete static_cast<nm::opt::Engine*>(engine); });
}

int nm_ns_gemm(int precision, const float* A, const float* B, float* D, int64_t M, int64_t N, int64_t K,
               void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    nm::Scratch sc(s);
    nm::optk::NsGemm gemm(nm::ns_prec(precision));
    const int id = gemm.plan({M, N, K, false, 1, M * K, K * N, M * N, 0.0f});
    gemm.choose(false, s);
    gemm.bind_workspace(sc.alloc<char>(gemm.workspace_bytes()));
    gemm.run(id, A, B, nullptr, D, s);
    NM_CUDA(cudaStreamSynchronize(s));
  });
}

}  // extern "C"
