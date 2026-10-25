// GEMM search bench (BUILD_LOG decision 4, 5; DESIGN.md §3.7, O8, O9): one GemmCache over the linear-layer GEMMs of the
// owner shapes, the timed search of choose_all, one JSON line per shape and orientation on stdout (chosen kernel,
// median ms, TFLOP/s, candidate count, the fastest candidates). No torch: the same source builds against CUDA 12.8 and
// 13.0, so the two cuBLASLt versions are compared by running the two binaries (tests/s4/gemm_report.py).
//
//   nm_bench_s4_gemm [--set e1|owner|all] [--m M] [--quick]
// --set e1: the 11 (M, K, N) shapes of docs/bench/RESULTS.md §2 (the bars); owner: every trunk linear layer of 130m,
// 300m, 520m, 1.2B at 131,072 tokens (qkv, o_proj, gate+up interleaved, down) and the lm head at 16,384 rows.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/gemm/gemm_internal.hpp"
#include "marincpp/kernels/gemm.hpp"

namespace {

struct Shape {
  const char* name;
  int64_t M, K, N;   // tokens, in features, out features
};

const Shape kE1[] = {
    {"e1", 131072, 512, 1536},   {"e1", 131072, 512, 1792},   {"e1", 131072, 768, 2304},  {"e1", 131072, 768, 2688},
    {"e1", 131072, 1024, 3584},  {"e1", 131072, 2048, 7168},  {"e1", 131072, 512, 128256}, {"e1", 131072, 768, 128256},
    {"e1", 16384, 512, 128256},  {"e1", 16384, 768, 128256},  {"e1", 8192, 768, 128256},
};

std::vector<Shape> owner_shapes(int64_t M) {
  struct Size { const char* n; int64_t D, H, KV, hd, F; };
  const Size sizes[] = {{"130m", 512, 8, 8, 64, 1792}, {"300m", 768, 12, 12, 64, 2688}, {"520m", 1024, 16, 8, 64, 3584},
                        {"1_2b", 2048, 16, 8, 128, 7168}};
  std::vector<Shape> v;
  static std::vector<std::string> names;
  names.reserve(64);
  for (const Size& s : sizes) {
    names.push_back(std::string(s.n) + ".qkv");
    v.push_back({names.back().c_str(), M, s.D, (s.H + 2 * s.KV) * s.hd});
    names.push_back(std::string(s.n) + ".o_proj");
    v.push_back({names.back().c_str(), M, s.H * s.hd, s.D});
    names.push_back(std::string(s.n) + ".gate_up");
    v.push_back({names.back().c_str(), M, s.D, 2 * s.F});
    names.push_back(std::string(s.n) + ".down");
    v.push_back({names.back().c_str(), M, s.F, s.D});
    names.push_back(std::string(s.n) + ".lm_head16k");
    v.push_back({names.back().c_str(), 16384, s.D, 128256});
  }
  return v;
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string set = "all";
  int64_t M = 131072;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--set") && i + 1 < argc) set = argv[++i];
    else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) M = std::atoll(argv[++i]);
    else { std::fprintf(stderr, "usage: %s [--set e1|owner|all] [--m M]\n", argv[0]); return 2; }
  }
  std::vector<Shape> shapes;
  if (set == "e1" || set == "all") shapes.insert(shapes.end(), std::begin(kE1), std::end(kE1));
  if (set == "owner" || set == "all") {
    auto o = owner_shapes(M);
    shapes.insert(shapes.end(), o.begin(), o.end());
  }
  int dev = 0;
  check(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  check(cudaGetDeviceProperties(&p, dev), "props");
  int rt = 0;
  cudaRuntimeGetVersion(&rt);
  cudaStream_t s;
  check(cudaStreamCreate(&s), "stream");
  cublasLtHandle_t lt;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return 1;
  std::fprintf(stderr, "device %s, cudart %d, cublasLt %zu\n", p.name, rt, cublasLtGetVersion());
  for (const Shape& sh : shapes) {
    for (int orient = 0; orient < 4; ++orient) {   // fwd, dgrad, wgrad accurate, wgrad levanter
      nm::GemmCache cache(lt, "");
      nm::GemmDesc d;
      const char* on = "";
      if (orient == 0) { d = nm::linear_fwd_desc(sh.M, sh.N, sh.K, nm::Epilogue::None, "fwd"); on = "fwd"; }
      if (orient == 1) { d = nm::linear_dgrad_desc(sh.M, sh.N, sh.K, nm::Epilogue::None, "dgrad"); on = "dgrad"; }
      if (orient == 2) { d = nm::linear_wgrad_desc(sh.M, sh.N, sh.K, nm::Rounding::Accurate, "wgrad"); on = "wgrad"; }
      if (orient == 3) { d = nm::linear_wgrad_desc(sh.M, sh.N, sh.K, nm::Rounding::Levanter, "wgrad_lev"); on = "wgrad_lev"; }
      cache.plan(d);
      // scratch: A, B, C, D, the levanter transient (as choose_all lays them out) + 256 MiB of split-K workspace
      auto al = [](size_t x) { return (x + 255) / 256 * 256; };
      const int64_t a_rows = d.op_a == nm::Op::N ? d.m : d.k, b_rows = d.op_b == nm::Op::N ? d.k : d.n;
      size_t need = al(size_t(a_rows * d.lda) * nm::dtype_size(d.a)) + al(size_t(b_rows * d.ldb) * nm::dtype_size(d.b)) +
                    al(size_t(d.m * d.ldd) * nm::dtype_size(d.d)) + (size_t(300) << 20);
      if (d.beta != 0.f) need += al(size_t(d.m * d.ldc) * nm::dtype_size(d.c));
      if (d.epilogue == nm::Epilogue::Bf16RoundAddF32) need += al(size_t(d.m * d.n) * 2);
      if (d.op_b == nm::Op::N) need += al(size_t(d.k * d.n) * 2);   // B^T of the transposed-B candidates
      void* scratch = nullptr;
      check(cudaMalloc(&scratch, need), "cudaMalloc scratch");
      try {
        cache.choose_all(s, scratch, need);
      } catch (const std::exception& e) {
        std::printf("{\"set\": \"%s\", \"orient\": \"%s\", \"M\": %lld, \"K\": %lld, \"N\": %lld, \"error\": \"%s\"}\n",
                    sh.name, on, (long long)sh.M, (long long)sh.K, (long long)sh.N, e.what());
        cudaFree(scratch);
        continue;
      }
      check(cudaFree(scratch), "cudaFree");
      const auto rep = nm::gemm::report(cache);
      const auto& r = rep.back();
      // second measurement of the chosen kernel through GemmCache::run (the trainer's path), fresh operands
      float retime = 0.f;
      {
        const int64_t a_rows = d.op_a == nm::Op::N ? d.m : d.k, b_rows = d.op_b == nm::Op::N ? d.k : d.n;
        const int64_t na = a_rows * d.lda, nb = b_rows * d.ldb, nd = d.m * d.ldd;
        void *A, *B, *D, *W = nullptr;
        check(cudaMalloc(&A, size_t(na) * nm::dtype_size(d.a)), "A");
        check(cudaMalloc(&B, size_t(nb) * nm::dtype_size(d.b)), "B");
        check(cudaMalloc(&D, size_t(nd) * nm::dtype_size(d.d)), "D");
        nm::gemm::fill_uniform(A, d.a, na, 7, s);
        nm::gemm::fill_uniform(B, d.b, nb, 8, s);
        nm::gemm::fill_uniform(D, d.d, nd, 9, s);
        if (cache.workspace_bytes()) check(cudaMalloc(&W, cache.workspace_bytes()), "W");
        cache.bind_workspace(W);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::vector<float> t;
        for (int i = 0; i < 25; ++i) {
          cudaEventRecord(e0, s);
          cache.run(0, A, B, D, D, s);
          cudaEventRecord(e1, s);
          check(cudaEventSynchronize(e1), "sync");
          float ms = 0.f;
          cudaEventElapsedTime(&ms, e0, e1);
          if (i >= 5) t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        retime = t[t.size() / 2];
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(A); cudaFree(B); cudaFree(D);
        if (W) cudaFree(W);
      }
      std::printf("{\"set\": \"%s\", \"orient\": \"%s\", \"M\": %lld, \"K\": %lld, \"N\": %lld, \"cudart\": %d, "
                  "\"cublaslt\": %zu, \"gpu\": \"%s\", \"chosen\": \"%s\", \"ms\": %.5f, \"tflops\": %.1f, "
                  "\"n_candidates\": %d, \"retime_ms\": %.5f, \"retime_tflops\": %.1f, \"top\": [",
                  sh.name, on, (long long)sh.M, (long long)sh.K, (long long)sh.N, rt, cublasLtGetVersion(), p.name,
                  r.chosen.c_str(), r.ms, r.tflops, r.n_candidates, retime,
                  2.0 * double(d.m) * double(d.n) * double(d.k) / (double(retime) * 1e-3) / 1e12);
      for (size_t i = 0; i < r.top.size(); ++i) {
        std::printf("%s{\"name\": \"%s\", \"ms\": %.5f, \"ws\": %zu}", i ? ", " : "", r.top[i].name.c_str(),
                    r.top[i].ms, r.top[i].workspace);
      }
      std::printf("]}\n");
      std::fflush(stdout);
    }
  }
  cublasLtDestroy(lt);
  return 0;
}
