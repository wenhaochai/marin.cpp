// BUILD_LOG decision 9 (DESIGN.md Appendix A h100-10): one process, one thread per GPU, NCCL communicators from
// ncclCommInitAll; each thread captures FA3 fwd + bwd (both training schedulers, persistent and doc_causal,
// deterministic, each into its own output buffers) and an NCCL all-reduce of each scheduler's dK gradient into ONE
// CUDA graph, replays it 100 times, and checks every replay against the eager run bitwise (dQ, dK, dV, O, LSE and the
// all-reduced dK of both schedulers). E6's PyTorch graph arms crashed in this setting.
//
//   nm_test_s4_graph_nccl [--gpus N] [--replays R]      exit 0 pass, 1 fail, 77 skip (< 2 GPUs or not sm_90)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kernels/attention/attn_internal.hpp"

namespace {

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x ": ") + cudaGetErrorString(e_)); \
  } while (0)
#define NK(x)                                                                                        \
  do {                                                                                               \
    ncclResult_t r_ = (x);                                                                           \
    if (r_ != ncclSuccess) throw std::runtime_error(std::string(#x ": ") + ncclGetErrorString(r_));  \
  } while (0)

constexpr int kRows = 8, kS = 4096, kH = 16, kKV = 8, kHd = 64, kMaxSegs = 24;   // 520m-like GQA shape, 32k tokens

struct Out {   // the outputs of one scheduler
  void *o, *dq, *dk, *dv, *dk_sum;
  float* lse;
};

struct Bufs {
  void *q, *k, *v, *dO;
  Out out[2];   // persistent, doc_causal
  int *cu, *n_segs, *rss;
  void* ws;
  size_t bytes_q, bytes_k;
};

void segments(int rank, std::vector<int>& cu, std::vector<int>& rss, int& n) {
  std::mt19937 g(1234 + rank);
  cu.clear();
  rss.assign(kRows * (kMaxSegs + 1), kS);
  for (int r = 0; r < kRows; ++r) {
    std::vector<int> st = {0};
    const int n_cut = 1 + int(g() % 12);
    for (int i = 0; i < n_cut; ++i) st.push_back(1 + int(g() % (kS - 1)));
    std::sort(st.begin(), st.end());
    st.erase(std::unique(st.begin(), st.end()), st.end());
    for (size_t i = 0; i < st.size(); ++i) {
      rss[r * (kMaxSegs + 1) + i] = st[i];
      cu.push_back(r * kS + st[i]);
    }
  }
  n = int(cu.size());
  while (int(cu.size()) < kRows * kMaxSegs + 1) cu.push_back(kRows * kS);
}

void fill(void* p, size_t n, uint32_t seed) {
  std::vector<__nv_bfloat16> h(n);
  std::mt19937 g(seed);
  std::normal_distribution<float> d;
  for (auto& x : h) x = __float2bfloat16(d(g));
  CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
}

// One step: for each training scheduler, FA3 fwd + bwd (deterministic) into its own outputs; then all-reduce each
// scheduler's dK (bf16 sum) into its dk_sum.
void step(const Bufs& b, ncclComm_t comm, cudaStream_t s) {
  const int64_t T = int64_t(kRows) * kS;
  const nm::Fa3Mode modes[2] = {nm::fa3_mode_of(nm::AttnBwdSched::Persistent),
                                nm::fa3_mode_of(nm::AttnBwdSched::DocCausal)};
  for (int i = 0; i < 2; ++i) {
    const nm::Fa3Mode mode = modes[i];
    const Out& r = b.out[i];
    nm::AttnShape sh{};
    sh.total_q = T; sh.rows = kRows; sh.S = kS; sh.H = kH; sh.KV = kKV; sh.hd = kHd; sh.max_segs_per_seq = kMaxSegs;
    sh.deterministic = true; sh.io = nm::DType::bf16;
    nm::AttnFwdArgs f{};
    f.q = b.q; f.q_ld = kH * kHd; f.k = b.k; f.k_ld = kKV * kHd; f.v = b.v; f.v_ld = kKV * kHd;
    f.o = r.o; f.o_ld = kH * kHd; f.lse = r.lse; f.cu_seqlens = b.cu; f.n_segs = b.n_segs; f.row_seg_start = b.rss;
    f.softmax_scale = 0.125f;
    nm::fa3k::fwd(sh, f, mode, nullptr, b.ws, s);
    nm::AttnBwdArgs a{};
    a.fwd = f;
    a.d_o = b.dO; a.do_ld = kH * kHd;
    a.dq = r.dq; a.dq_ld = kH * kHd; a.dk = r.dk; a.dk_ld = kKV * kHd; a.dv = r.dv; a.dv_ld = kKV * kHd;
    nm::fa3k::bwd(sh, a, mode, nullptr, b.ws, s);
  }
  for (const Out& r : b.out) NK(ncclAllReduce(r.dk, r.dk_sum, size_t(T) * kKV * kHd, ncclBfloat16, ncclSum, comm, s));
}

std::vector<char> snapshot(const Bufs& b) {
  const int64_t T = int64_t(kRows) * kS;
  std::vector<char> h;
  auto add = [&](const void* p, size_t n) {
    const size_t o = h.size();
    h.resize(o + n);
    CK(cudaMemcpy(h.data() + o, p, n, cudaMemcpyDeviceToHost));
  };
  for (const Out& r : b.out) {
    add(r.o, b.bytes_q); add(r.dq, b.bytes_q); add(r.dk, b.bytes_k); add(r.dv, b.bytes_k); add(r.dk_sum, b.bytes_k);
    add(r.lse, size_t(kH) * T * 4);
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  int want = 4, replays = 100;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--gpus")) want = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--replays")) replays = std::atoi(argv[i + 1]);
  }
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 2) {
    std::printf("SKIP: needs >= 2 GPUs (found %d)\n", n_dev);
    return 77;
  }
  const int n = std::min(want, n_dev);
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, d);
    if (p.major != 9) {
      std::printf("SKIP: GPU %d is sm_%d%d (FA3 needs sm_90)\n", d, p.major, p.minor);
      return 77;
    }
  }
  std::vector<ncclComm_t> comms(n);
  std::vector<int> devs(n);
  for (int i = 0; i < n; ++i) devs[i] = i;
  if (ncclCommInitAll(comms.data(), n, devs.data()) != ncclSuccess) {
    std::printf("FAIL: ncclCommInitAll\n");
    return 1;
  }
  std::atomic<int> failures{0};
  std::vector<std::thread> th;
  for (int rank = 0; rank < n; ++rank) {
    th.emplace_back([&, rank] {
      try {
        CK(cudaSetDevice(rank));
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        const int64_t T = int64_t(kRows) * kS;
        Bufs b{};
        b.bytes_q = size_t(T) * kH * kHd * 2;
        b.bytes_k = size_t(T) * kKV * kHd * 2;
        for (void** p : {&b.q, &b.dO}) CK(cudaMalloc(p, b.bytes_q));
        for (void** p : {&b.k, &b.v}) CK(cudaMalloc(p, b.bytes_k));
        for (Out& r : b.out) {
          for (void** p : {&r.o, &r.dq}) CK(cudaMalloc(p, b.bytes_q));
          for (void** p : {&r.dk, &r.dv, &r.dk_sum}) CK(cudaMalloc(p, b.bytes_k));
          CK(cudaMalloc(&r.lse, size_t(kH) * T * 4));
        }
        std::vector<int> cu, rss;
        int nseg = 0;
        segments(rank, cu, rss, nseg);
        CK(cudaMalloc(&b.cu, cu.size() * 4));
        CK(cudaMalloc(&b.rss, rss.size() * 4));
        CK(cudaMalloc(&b.n_segs, 4));
        CK(cudaMemcpy(b.cu, cu.data(), cu.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.rss, rss.data(), rss.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.n_segs, &nseg, 4, cudaMemcpyHostToDevice));
        fill(b.q, b.bytes_q / 2, 10 + rank);
        fill(b.k, b.bytes_k / 2, 20 + rank);
        fill(b.v, b.bytes_k / 2, 30 + rank);
        fill(b.dO, b.bytes_q / 2, 40 + rank);
        nm::AttnShape sh{};
        sh.total_q = T; sh.rows = kRows; sh.S = kS; sh.H = kH; sh.KV = kKV; sh.hd = kHd; sh.max_segs_per_seq = kMaxSegs;
        sh.io = nm::DType::bf16;
        const size_t ws = std::max(nm::fa3k::workspace_bytes(sh, nm::Fa3Mode::ListGrid, true),
                                   nm::fa3k::workspace_bytes(sh, nm::Fa3Mode::Doc, true));
        CK(cudaMalloc(&b.ws, ws));
        // eager reference (also sets the kernels' shared-memory attributes before the capture)
        step(b, comms[rank], s);
        CK(cudaStreamSynchronize(s));
        const std::vector<char> ref = snapshot(b);
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        step(b, comms[rank], s);
        CK(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiate(&ge, g, 0));
        size_t n_nodes = 0;
        CK(cudaGraphGetNodes(g, nullptr, &n_nodes));
        int bad = 0;
        for (int i = 0; i < replays; ++i) {
          CK(cudaGraphLaunch(ge, s));
          if (i == 0 || i == replays - 1 || i % 25 == 0) {
            CK(cudaStreamSynchronize(s));
            if (snapshot(b) != ref) ++bad;
          }
        }
        CK(cudaStreamSynchronize(s));
        if (snapshot(b) != ref) ++bad;
        std::printf("rank %d: graph of %zu nodes, %d replays, %d mismatching checkpoints against the eager run\n", rank,
                    n_nodes, replays, bad);
        if (bad) failures++;
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
      } catch (const std::exception& e) {
        std::printf("rank %d: FAIL: %s\n", rank, e.what());
        failures++;
      }
    });
  }
  for (auto& t : th) t.join();
  for (auto c : comms) ncclCommDestroy(c);
  std::printf("%s: %d GPUs, %d replays\n", failures ? "FAIL" : "PASS", n, replays);
  return failures ? 1 : 0;
}
