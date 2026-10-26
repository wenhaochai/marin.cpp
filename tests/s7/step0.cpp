// S7 gate driver: step-0 loss and gradients of the baseline model against the levanter fixtures (parity plan T2.1,
// T2.5, T2.6). One GPU runs the whole global batch as `--micro-rows` microbatches, eagerly.
//
//   nm_s7_step0 --root <repo> [--size 130m] [--fixture step0_bf16] [--micro-rows 32] [--attn fa3|ref]
//               [--ce b1|a] [--rounding levanter|accurate] [--ce-chunk 8192] [--steps 1] [--embed f32|bf16]
// --steps K > 1 also applies K-1 optimizer updates (S6 engine) on batches 1.. of the converted dataset (levanter order)
// and prints the update error against the fixture's u0 and the step-1 / step-2 losses against levanter's (T2.7).
// Prints the loss, the per-tensor gradient error (rel L2 and cosine against the fixture's g0), the three pre-clip group
// norms, and PASS/FAIL lines for T2.1 (|dL| <= 1e-4) and T2.6 (muonh, adamh within 1e-4 relative). Exit 0 iff all
// printed gates pass, 77 if there is no GPU or no fixture.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/kernels/gemm.hpp"
#include "marincpp/memory.hpp"
#include "model/baseline.hpp"
#include "model/fixture.hpp"
#include "model/npy.hpp"
#include "train/optimizer.hpp"

using namespace nm;

namespace {

std::string arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}

bool exists(const std::string& p) { return std::ifstream(p).good(); }

// First number after `"<key>"` ... `"norm":` in group_norms.json (g0[0].<group>.norm).
double group_norm_ref(const std::string& json, const std::string& group) {
  const size_t g = json.find("\"" + group + "\"");
  const size_t n = json.find("\"norm\":", g);
  return std::stod(json.substr(n + 7));
}
double first_number_after(const std::string& json, const std::string& key) {
  const size_t k = json.find(key);
  const size_t v = json.find("\"value\":", k);
  return std::stod(json.substr(v + 8));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = arg(argc, argv, "--root", ".");
  const std::string size = arg(argc, argv, "--size", "130m");
  const std::string fixture = root + "/cache/levref/" + size + "/" + arg(argc, argv, "--fixture", "step0_bf16");
  const std::string batch_dir = root + "/cache/levref/" + size + "/batch0";
  const int micro_rows = std::stoi(arg(argc, argv, "--micro-rows", "32"));
  const std::string attn = arg(argc, argv, "--attn", "fa3");
  const std::string ce = arg(argc, argv, "--ce", "b1");
  const std::string rounding = arg(argc, argv, "--rounding", "levanter");
  const int ce_chunk = std::stoi(arg(argc, argv, "--ce-chunk", "8192"));
  const int steps = std::stoi(arg(argc, argv, "--steps", "1"));
  const std::string embed = arg(argc, argv, "--embed", "f32");   // f32 (fast default) | bf16 (levanter's accumulator)
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::puts("SKIP: no GPU"); return 77; }
  if (!exists(fixture + "/p0/index.json") || !exists(batch_dir + "/tokens.npy")) {
    std::printf("SKIP: fixture missing under %s\n", fixture.c_str());
    return 77;
  }
  try {
    Config c;
    apply_preset(c, size);
    const ModelShape m = model_shape(c);
    std::vector<int64_t> tshape;
    const std::vector<int32_t> tokens = npy::load_i32(batch_dir + "/tokens.npy", &tshape);
    const int B = static_cast<int>(tshape[0]);
    NM_REQUIRE(tshape[1] == m.S, "batch0 sequence length differs from the model's");
    NM_REQUIRE(B % micro_rows == 0, "--micro-rows must divide the batch");

    model::BaselineSpec bs;
    bs.m = m;
    bs.micro_rows = micro_rows;
    bs.global_rows = B;
    bs.max_segs = 24;
    bs.ce_chunk = ce_chunk;
    bs.num.rounding = rounding == "accurate" ? Rounding::Accurate : Rounding::Levanter;
    bs.attn = attn == "ref" ? AttnBackendKind::RefF32 : AttnBackendKind::FA3;
    bs.ce = ce == "a" ? CeImpl::A : CeImpl::B1;
    bs.num.embed_accum = embed == "bf16" ? EmbedAccum::Bf16Sequential : EmbedAccum::F32;

    cudaStream_t s;
    NM_CUDA(cudaSetDevice(0));
    NM_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cublasLtHandle_t lt;
    NM_CUBLAS(cublasLtCreate(&lt));
    model::Baseline net(bs);
    GemmCache gc(lt, "");
    net.plan_gemms(gc);
    {
      void* scratch = nullptr;
      NM_CUDA(cudaMalloc(&scratch, net.gemm_search_scratch()));
      gc.choose_all(s, scratch, net.gemm_search_scratch());
      NM_CUDA(cudaStreamSynchronize(s));
      NM_CUDA(cudaFree(scratch));
    }
    MemoryPlan plan;
    net.declare(plan, gc.workspace_bytes());
    std::unique_ptr<train::Optimizer> opt;
    if (steps > 1) {
      opt = std::make_unique<train::Optimizer>(c, net);
      opt->declare(plan);
    }
    size_t free_b = 0, total_b = 0;
    NM_CUDA(cudaMemGetInfo(&free_b, &total_b));
    plan.finalize(free_b - (size_t(2) << 30), false);
    std::printf("arena %.2f GiB (P %.2f, A %.2f, T %.2f, W %.2f)\n", plan.device_bytes() / 1073741824.0,
                plan.payload_bytes('P') / 1073741824.0, plan.payload_bytes('A') / 1073741824.0,
                plan.payload_bytes('T') / 1073741824.0, plan.payload_bytes('W') / 1073741824.0);
    Arena arena;
    arena.allocate(plan, 0, ArenaDebug::Off);
    net.bind(arena, gc, s);
    if (opt) opt->bind(arena, net, s, 0);

    // parameters: f32 masters and their bf16 compute copies from the fixture's init
    const std::vector<float> p0 = model::load_fixture_storage(fixture + "/p0", net.params());
    NM_REQUIRE(static_cast<int64_t>(p0.size()) == net.total(), "fixture size differs from the parameter table");
    std::vector<bf16_t> p16(p0.size());
    for (size_t i = 0; i < p0.size(); ++i) p16[i].bits = f32_to_bf16_bits(p0[i]);
    NM_CUDA(cudaMemcpy(net.p32(), p0.data(), p0.size() * 4, cudaMemcpyHostToDevice));
    NM_CUDA(cudaMemcpy(net.p16(), p16.data(), p16.size() * 2, cudaMemcpyHostToDevice));

    net.zero_grads(s);
    const int n_micro = B / micro_rows;
    NM_REQUIRE(n_micro <= model::Baseline::kMaxMicro, "too many microbatches");
    cudaEvent_t e0, e1;
    NM_CUDA(cudaEventCreate(&e0));
    NM_CUDA(cudaEventCreate(&e1));
    NM_CUDA(cudaEventRecord(e0, s));
    for (int j = 0; j < n_micro; ++j) {
      NM_CUDA(cudaMemcpyAsync(net.tokens(), tokens.data() + size_t(j) * micro_rows * m.S,
                              size_t(micro_rows) * m.S * 4, cudaMemcpyHostToDevice, s));
      net.record_microbatch(j, s);
    }
    NM_CUDA(cudaEventRecord(e1, s));
    NM_CUDA(cudaStreamSynchronize(s));
    float ms = 0;
    NM_CUDA(cudaEventElapsedTime(&ms, e0, e1));

    std::vector<float> slots(n_micro);
    NM_CUDA(cudaMemcpy(slots.data(), net.loss_slots(), n_micro * 4, cudaMemcpyDeviceToHost));
    double lsum = 0;
    for (float x : slots) lsum += x;
    const double loss = lsum * net.loss_scale();
    std::ifstream lf(fixture + "/losses.json");
    std::stringstream lss;
    lss << lf.rdbuf();
    const double L0 = first_number_after(lss.str(), "\"L0_ref\"");
    std::printf("step0 loss %.7f  levanter %.7f  diff %+.2e  (%d microbatches of %d rows, %.1f ms eager)\n", loss, L0,
                loss - L0, n_micro, micro_rows, ms);

    // gradients against the fixture g0
    std::vector<float> g(net.total());
    NM_CUDA(cudaMemcpy(g.data(), net.grad(), g.size() * 4, cudaMemcpyDeviceToHost));
    const std::vector<float> gref = model::load_fixture_storage(fixture + "/g0", net.params());
    std::map<std::string, double> ss_nm, ss_ref;
    std::printf("%-30s %12s %12s %10s %8s\n", "tensor", "|g|", "|g_lev|", "rel", "cos");
    for (size_t i = 0; i < net.params().size(); ++i) {
      const ParamSpec& p = net.params()[i];
      const int64_t o = net.offset(static_cast<int>(i)), n = net.numel(static_cast<int>(i));
      double a2 = 0, b2 = 0, d2 = 0, ab = 0;
      for (int64_t k = 0; k < n; ++k) {
        const double x = g[o + k], y = gref[o + k];
        a2 += x * x, b2 += y * y, d2 += (x - y) * (x - y), ab += x * y;
      }
      ss_nm[p.group] += a2;
      ss_ref[p.group] += b2;
      if (p.layer <= 0 || p.layer == m.L - 1)
        std::printf("%-30s %12.5e %12.5e %10.3e %8.5f\n", p.name.c_str(), std::sqrt(a2), std::sqrt(b2),
                    std::sqrt(d2 / std::max(b2, 1e-300)), ab / std::sqrt(std::max(a2 * b2, 1e-300)));
    }
    std::ifstream gf(fixture + "/group_norms.json");
    std::stringstream gss;
    gss << gf.rdbuf();
    bool ok = std::fabs(loss - L0) <= 1e-4;
    std::printf("%s T2.1 step-0 loss |dL| = %.2e (bound 1e-4)\n", ok ? "PASS" : "FAIL", std::fabs(loss - L0));
    for (const char* grp : {"muonh", "adamh", "adam"}) {
      const double ours = std::sqrt(ss_nm[grp]), ref = group_norm_ref(gss.str(), grp);
      const double rel = std::fabs(ours - ref) / ref;
      const bool pass = rel <= 1e-4;
      if (std::string(grp) != "adam") ok = ok && pass;
      std::printf("%s T2.6 %-6s norm %.6f  levanter %.6f  rel %.2e%s\n", pass ? "PASS" : (std::string(grp) == "adam" ? "INFO" : "FAIL"),
                  grp, ours, ref, rel, std::string(grp) == "adam" ? " (embedding accumulator differs by design, cache/levref/README §5)" : "");
    }

    // ---- further updates (T2.7): u0 against the fixture, then the losses of batches 1.. under the updated params
    if (opt) {
      const char* sroot = std::getenv("NANOMARIN_ROOT");
      NM_REQUIRE(sroot, "--steps > 1 needs NANOMARIN_ROOT (the converted dataset lives under it)");
      const DatasetMeta meta = load_dataset_meta(std::string(sroot) + "/marin_store_big/marincpp/fineweb-edu-10B-marin128k");
      TokenFile tf;
      tf.open(meta);
      const LevanterOrder order(42, meta.n_seqs, 2048);
      auto batch = [&](int k) {
        std::vector<int32_t> v(size_t(B) * m.S);
        for (int r = 0; r < B; ++r) {
          const uint32_t* w = tf.window(order(uint64_t(k) * B + r), m.S);
          for (int j = 0; j < m.S; ++j) v[size_t(r) * m.S + j] = static_cast<int32_t>(w[j]);
        }
        return v;
      };
      const bool same0 = batch(0) == tokens;
      std::printf("%s batch 0 from the converted dataset equals the fixture batch\n", same0 ? "PASS" : "FAIL");
      ok = ok && same0;
      std::ifstream lf2(fixture + "/losses.json");
      std::stringstream ls2;
      ls2 << lf2.rdbuf();
      const std::string lj = ls2.str();
      std::vector<double> lev_train;   // L_train_step values
      for (size_t at = lj.find("\"L_train_step\""), v; (v = lj.find("\"value\":", at)) != std::string::npos; at = v + 8)
        lev_train.push_back(std::stod(lj.substr(v + 8)));
      for (int k = 1; k < steps; ++k) {
        opt->record_step(s);   // update k-1 from the gradients now in grad.f32
        NM_CUDA(cudaStreamSynchronize(s));
        if (k == 1) {
          std::vector<float> p1(net.total());
          NM_CUDA(cudaMemcpy(p1.data(), net.p32(), p1.size() * 4, cudaMemcpyDeviceToHost));
          const std::vector<float> u0 = model::load_fixture_storage(fixture + "/u0", net.params());
          double worst = 0;
          std::string worst_name;
          for (size_t i = 0; i < net.params().size(); ++i) {
            const int64_t o = net.offset(static_cast<int>(i)), n = net.numel(static_cast<int>(i));
            double d2 = 0, b2 = 0;
            for (int64_t q = 0; q < n; ++q) {
              const double u = double(p1[o + q]) - double(p0[o + q]), y = u0[o + q];
              d2 += (u - y) * (u - y), b2 += y * y;
            }
            const double rel = std::sqrt(d2 / std::max(b2, 1e-300));
            if (net.params()[i].name != "embedding" && rel > worst) worst = rel, worst_name = net.params()[i].name;
            if (net.params()[i].layer <= 0) std::printf("  u0 %-28s rel %.3e\n", net.params()[i].name.c_str(), rel);
          }
          std::printf("INFO update 0 worst rel against levanter u0 (embedding excluded): %.3e (%s)\n", worst,
                      worst_name.c_str());
        }
        const std::vector<int32_t> tk = batch(k);
        net.zero_grads(s);
        for (int j = 0; j < n_micro; ++j) {
          NM_CUDA(cudaMemcpyAsync(net.tokens(), tk.data() + size_t(j) * micro_rows * m.S,
                                  size_t(micro_rows) * m.S * 4, cudaMemcpyHostToDevice, s));
          net.record_microbatch(j, s);
        }
        NM_CUDA(cudaStreamSynchronize(s));
        NM_CUDA(cudaMemcpy(slots.data(), net.loss_slots(), n_micro * 4, cudaMemcpyDeviceToHost));
        double sum = 0;
        for (float x : slots) sum += x;
        const double lk = sum * net.loss_scale();
        const double ref = k < static_cast<int>(lev_train.size()) ? lev_train[k] : NAN;
        const bool pass = std::fabs(lk - ref) <= 2e-3;
        if (k <= 2) ok = ok && pass;
        std::printf("%s T2.7 step-%d loss %.7f  levanter %.7f  diff %+.2e (bound 2e-3)\n", pass ? "PASS" : "FAIL", k, lk,
                    ref, lk - ref);
      }
    }
    cublasLtDestroy(lt);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 2;
  }
}
