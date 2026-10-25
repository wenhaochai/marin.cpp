// GemmCache: one GEMM kernel per shape, chosen by a timed search over deterministic candidates. DESIGN.md §3.7, §11.
//
// Candidates per shape
//   cuBLASLt: the heuristic's configured algorithms (up to kHeuristic), each also with split-K counts {2, 3, 4, 6, 8,
//   12, 16} under every allowed reduction scheme. Allowed schemes: NONE always; COMPUTE_TYPE always (f32 partials,
//   one rounding at the output); INPLACE and OUTPUT_TYPE only when D is f32 (they reduce in the output type, which
//   for bf16 D would round each partial). cuBLASLt has no float-atomic reduction (cuBLAS: bitwise reproducible on one
//   GPU type and toolkit), INPLACE serializes the splits through workspace counters.
//   Custom kernels registered with register_custom_gemm (CUTLASS SM90, src/kernels/gemm/gemm_sm90.cu) that accept
//   the desc.
// Timing: every candidate 2 warm-up + 5 timed calls (median, CUDA events), then the 4 fastest 5 warm-up + 20 timed
// calls (BUILD_LOG decision 2); the fastest wins, a custom kernel only when faster than the best cuBLASLt candidate by
// more than 1% (O9: ties go to cuBLASLt). The cuBLASLt workspace is the largest the chosen algorithms need.
// Persistence: gemm_algos.bin holds one record per shape key (desc + device name + SM count + cuBLASLt version); a
// resumed run reloads it and executes the same kernels (bitwise resume). The candidate set does not depend on the
// scratch size (the heuristic is asked with kMaxSearchWorkspace; candidates needing more than the scratch holds are not
// timed), so a resume with other scratch finds every saved name. Choices are timing-dependent and per process: ranks
// and A/B twins that must run the same kernels share one gemm_algos.bin (one process searches, the others load).
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include <unistd.h>

#include "kernels/gemm/gemm_internal.hpp"
#include "marincpp/kernels/gemm.hpp"

namespace nm {

namespace {

constexpr int kHeuristic = 24;
constexpr size_t kMaxSearchWorkspace = size_t(256) << 20;   // split-K partials beyond this are not worth it
const int kSplits[] = {2, 3, 4, 6, 8, 12, 16};

cudaDataType_t cuda_type(DType t) {
  switch (t) {
    case DType::bf16: return CUDA_R_16BF;
    case DType::f32: return CUDA_R_32F;
    case DType::f16: return CUDA_R_16F;
    default: fatal(NM_WHERE, std::string("GEMM: unsupported dtype ") + dtype_name(t));
  }
}

cublasComputeType_t compute_type(Compute c) {
  switch (c) {
    case Compute::F32: return CUBLAS_COMPUTE_32F;
    case Compute::TF32: return CUBLAS_COMPUTE_32F_FAST_TF32;
    case Compute::F32_PEDANTIC: return CUBLAS_COMPUTE_32F_PEDANTIC;
    case Compute::F32_3XTF32: break;
  }
  fatal(NM_WHERE, "GEMM: 3xTF32 runs only as a custom kernel");
}

std::string device_tag() {
  int dev = 0, sms = 0;
  NM_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  NM_CUDA(cudaGetDeviceProperties(&p, dev));
  sms = p.multiProcessorCount;
  std::ostringstream o;
  o << p.name << "|sm" << p.major << p.minor << "|" << sms << "SMs|lt" << cublasLtGetVersion();
  return o.str();
}

std::string desc_key(const GemmDesc& d) {
  std::ostringstream o;
  o << "m" << d.m << "n" << d.n << "k" << d.k << "op" << int(d.op_a) << int(d.op_b) << "t" << int(d.a) << int(d.b)
    << int(d.c) << int(d.d) << "ld" << d.lda << "," << d.ldb << "," << d.ldc << "," << d.ldd << "b" << d.batch << "s"
    << d.stride_a << "," << d.stride_b << "," << d.stride_c << "," << d.stride_d << "c" << int(d.compute) << "a"
    << d.alpha << "be" << d.beta << "e" << int(d.epilogue) << "sp" << int(d.split) << "sy" << d.symmetric_output;
  return o.str();
}

struct LtAlgo {
  cublasLtMatmulAlgo_t algo;
  size_t ws = 0;
  std::string name;
};

// Every configuration attribute of the algorithm, each read with its own type (cublasLt.h): two algorithms differ in
// name exactly when they differ in configuration.
std::string algo_name(const cublasLtMatmulAlgo_t& a) {
  auto get = [&](cublasLtMatmulAlgoConfigAttributes_t attr, auto v) {
    size_t w = 0;
    NM_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(&a, attr, &v, sizeof(v), &w));
    return int64_t(v);
  };
  std::ostringstream o;
  o << "lt:" << get(CUBLASLT_ALGO_CONFIG_ID, int32_t(0)) << ":tile" << get(CUBLASLT_ALGO_CONFIG_TILE_ID, uint32_t(0))
    << ":st" << get(CUBLASLT_ALGO_CONFIG_STAGES_ID, uint32_t(0)) << ":split"
    << get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, int32_t(0)) << ":red"
    << get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, uint32_t(0)) << ":swz"
    << get(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, uint32_t(0)) << ":co"
    << get(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, uint32_t(0)) << ":is"
    << get(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, uint16_t(0)) << ":cl"
    << get(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, uint16_t(0));
  return o.str();
}

std::mutex g_reg_mu;
std::map<const GemmCache*, std::vector<gemm::ShapeReport>>& g_reports() {
  static std::map<const GemmCache*, std::vector<gemm::ShapeReport>> m;
  return m;
}
std::map<std::pair<const GemmCache*, int>, std::string>& g_chosen() {
  static std::map<std::pair<const GemmCache*, int>, std::string> m;
  return m;
}
std::map<const GemmCache*, std::string>& g_forced() {
  static std::map<const GemmCache*, std::string> m;
  return m;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ entries
struct GemmCache::Entry {
  GemmDesc d;
  std::string key;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr, lt_tmp = nullptr;
  bool chosen = false;
  bool use_custom = false;
  int custom = -1;
  std::string custom_name;
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
  size_t tmp_bytes = 0;   // bf16 [m, n] transient of the portable Levanter wgrad path (cuBLASLt + round-add)
  // cuBLASLt view (column-major): D^T[n, m] = op(B)^T op(A)^T
  int64_t lt_m = 0, lt_n = 0, lt_k = 0;
  // Transposed-B variant (op_b N, bf16, batch 1: the dgrad's weight): B^T [n, k] is written to the workspace by
  // gemm::transpose_bf16 and the GEMM runs on K-major operands (op_b T); the transpose is part of the timed call.
  cublasLtMatmulDesc_t op_t = nullptr;
  cublasLtMatrixLayout_t la_t = nullptr;
  size_t tb_bytes = 0;
  bool use_tb = false;
};

namespace {

bool lt_round_add(const GemmDesc& d) { return d.epilogue == Epilogue::Bf16RoundAddF32; }

void make_layout(cublasLtMatrixLayout_t* l, DType t, int64_t rows, int64_t cols, int64_t ld, int batch,
                 int64_t stride) {
  NM_CUBLAS(cublasLtMatrixLayoutCreate(l, cuda_type(t), uint64_t(rows), uint64_t(cols), ld));
  if (batch > 1) {
    NM_CUBLAS(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    NM_CUBLAS(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                               sizeof(stride)));
  }
}

void validate(const GemmDesc& d) {
  NM_REQUIRE(d.m > 0 && d.n > 0 && d.k > 0 && d.batch >= 1, "GEMM: empty shape " + desc_key(d));
  const int64_t a_cols = d.op_a == Op::N ? d.k : d.m;
  const int64_t b_cols = d.op_b == Op::N ? d.n : d.k;
  NM_REQUIRE(d.lda >= a_cols && d.ldb >= b_cols && d.ldd >= d.n, "GEMM: leading dimension too small: " + desc_key(d));
  NM_REQUIRE(d.beta == 0.f || d.ldc >= d.n, "GEMM: ldc too small: " + desc_key(d));
  NM_REQUIRE(d.epilogue == Epilogue::None || d.epilogue == Epilogue::Bf16RoundAddF32,
             "GEMM: fused epilogue " + std::to_string(int(d.epilogue)) + " is built in S8 (DESIGN.md §3.1)");
  if (d.epilogue == Epilogue::Bf16RoundAddF32) {
    NM_REQUIRE(d.c == DType::f32 && d.d == DType::f32 && d.beta == 1.f && d.alpha == 1.f && d.batch == 1,
               "GEMM: Bf16RoundAddF32 is the wgrad epilogue D = C + f32(bf16(acc)), f32 C and D, alpha = beta = 1");
  }
  NM_REQUIRE(!d.symmetric_output || d.m == d.n, "GEMM: symmetric_output needs a square output");
  if (d.compute == Compute::F32) {
    NM_REQUIRE(d.a == DType::bf16 && d.b == DType::bf16, "GEMM: Compute::F32 takes bf16 A and B");
  } else {
    NM_REQUIRE(d.a == DType::f32 && d.b == DType::f32 && d.c == DType::f32 && d.d == DType::f32,
               "GEMM: TF32 / F32_PEDANTIC / 3xTF32 take f32 operands");
  }
}

}  // namespace

GemmCache::GemmCache(cublasLtHandle_t lt, std::string cache_path) : lt_(lt), path_(std::move(cache_path)) {}

int GemmCache::plan(const GemmDesc& d) {
  validate(d);
  const std::string key = desc_key(d);
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i]->key == key) return int(i);
  }
  auto* e = new Entry;
  e->d = d;
  e->key = key;
  if (d.compute != Compute::F32_3XTF32) {
    // row-major D[m, n] = op(A) op(B)  <=>  column-major D^T[n, m] = op(B)^T op(A)^T
    e->lt_m = d.n; e->lt_n = d.m; e->lt_k = d.k;
    NM_CUBLAS(cublasLtMatmulDescCreate(&e->op, compute_type(d.compute), CUDA_R_32F));
    cublasOperation_t ta = d.op_b == Op::N ? CUBLAS_OP_N : CUBLAS_OP_T;   // Lt A = our B
    cublasOperation_t tb = d.op_a == Op::N ? CUBLAS_OP_N : CUBLAS_OP_T;   // Lt B = our A
    NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    // Lt A: op(B)^T [n, k]; stored column-major as [n, k] (N) or [k, n] (T) with ld = ldb
    if (ta == CUBLAS_OP_N) make_layout(&e->la, d.b, d.n, d.k, d.ldb, d.batch, d.stride_b);
    else make_layout(&e->la, d.b, d.k, d.n, d.ldb, d.batch, d.stride_b);
    if (tb == CUBLAS_OP_N) make_layout(&e->lb, d.a, d.k, d.m, d.lda, d.batch, d.stride_a);
    else make_layout(&e->lb, d.a, d.m, d.k, d.lda, d.batch, d.stride_a);
    if (lt_round_add(d)) {   // cuBLASLt writes bf16(acc) to a transient; gemm::bf16_round_add adds it to C
      make_layout(&e->lt_tmp, DType::bf16, d.n, d.m, d.n, 1, 0);
      e->lc = e->lt_tmp;
      e->ld = e->lt_tmp;
      e->tmp_bytes = size_t(d.m) * d.n * 2;
    } else {
      make_layout(&e->lc, d.beta != 0.f ? d.c : d.d, d.n, d.m, d.beta != 0.f ? d.ldc : d.ldd, d.batch,
                  d.beta != 0.f ? d.stride_c : d.stride_d);
      make_layout(&e->ld, d.d, d.n, d.m, d.ldd, d.batch, d.stride_d);
    }
    if (d.op_b == Op::N && d.b == DType::bf16 && d.batch == 1) {
      NM_CUBLAS(cublasLtMatmulDescCreate(&e->op_t, compute_type(d.compute), CUDA_R_32F));
      const cublasOperation_t tt = CUBLAS_OP_T;
      NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op_t, CUBLASLT_MATMUL_DESC_TRANSA, &tt, sizeof(tt)));
      NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op_t, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      make_layout(&e->la_t, d.b, d.k, d.n, d.k, 1, 0);   // B^T [n, k] row-major = column-major [k, n], ld k
      e->tb_bytes = size_t(d.n) * d.k * 2;
    }
  }
  entries_.push_back(e);
  return int(entries_.size() - 1);
}

namespace {

float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.f : v[v.size() / 2];
}

struct Bufs {
  const void* A;
  const void* B;
  void* C;
  void* D;
  void* ws;
  size_t ws_bytes;
  void* tmp;
  void* tb;   // B^T of the transposed-B variant
};

}  // namespace

namespace {

// One call of entry e with candidate (algo or custom): the exact code path run() takes. (A template so the private
// GemmCache::Entry is never named outside the members that call it.)
template <class E>
void launch(cublasLtHandle_t lt, const E& e, bool custom, int ci, bool tb, const cublasLtMatmulAlgo_t* algo,
            const Bufs& b, cudaStream_t s) {
  const GemmDesc& d = e.d;
  if (custom) {
    gemm::custom_gemms()[size_t(ci)].launch(d, b.A, b.B, b.C, b.D, b.ws, s);
    return;
  }
  const float alpha = d.alpha;
  const cublasLtMatmulDesc_t op = tb ? e.op_t : e.op;
  const cublasLtMatrixLayout_t la = tb ? e.la_t : e.la;
  const void* B = b.B;
  if (tb) {
    gemm::transpose_bf16(b.B, d.ldb, b.tb, d.k, d.k, d.n, s);
    B = b.tb;
  }
  if (lt_round_add(d)) {
    const float zero = 0.f;
    NM_CUBLAS(cublasLtMatmul(lt, op, &alpha, B, la, b.A, e.lb, &zero, b.tmp, e.lc, b.tmp, e.ld, algo, b.ws,
                             b.ws_bytes, s));
    gemm::bf16_round_add(static_cast<const float*>(b.C), d.ldc, b.tmp, d.n, static_cast<float*>(b.D), d.ldd, d.m, d.n,
                         s);
    return;
  }
  const float beta = d.beta;
  NM_CUBLAS(cublasLtMatmul(lt, op, &alpha, B, la, b.A, e.lb, &beta, d.beta != 0.f ? b.C : b.D, e.lc, b.D, e.ld,
                           algo, b.ws, b.ws_bytes, s));
}

}  // namespace

void GemmCache::choose_all(cudaStream_t s, void* scratch, size_t scratch_bytes) {
  const std::string dev = device_tag();
  // previous choices of this run (resume)
  std::map<std::string, std::string> saved;
  if (!path_.empty()) {
    std::ifstream f(path_, std::ios::binary);
    std::string line;
    while (f && std::getline(f, line)) {
      const size_t t = line.find('\t');
      if (t != std::string::npos) saved[line.substr(0, t)] = line.substr(t + 1);
    }
  }
  std::string forced;
  {
    std::lock_guard<std::mutex> g(g_reg_mu);
    auto it = g_forced().find(this);
    if (it != g_forced().end()) forced = it->second;
  }
  std::vector<gemm::ShapeReport> reports;
  cudaEvent_t ev0, ev1;
  NM_CUDA(cudaEventCreate(&ev0));
  NM_CUDA(cudaEventCreate(&ev1));
  auto time_ms = [&](const std::function<void()>& call, int warm, int reps) {
    for (int i = 0; i < warm; ++i) call();
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
      NM_CUDA(cudaEventRecord(ev0, s));
      call();
      NM_CUDA(cudaEventRecord(ev1, s));
      NM_CUDA(cudaEventSynchronize(ev1));
      float ms = 0.f;
      NM_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
      t.push_back(ms);
    }
    return median(t);
  };
  std::vector<std::pair<int, std::string>> chosen_now;
  for (size_t ei = 0; ei < entries_.size(); ++ei) {
    Entry* e = entries_[ei];
    if (e->chosen) continue;
    const GemmDesc& d = e->d;
    // operand scratch: A, B, C (or D in place), D, transient, workspace
    const int64_t a_rows = d.op_a == Op::N ? d.m : d.k;
    const int64_t b_rows = d.op_b == Op::N ? d.k : d.n;
    const size_t a_bytes = size_t((d.batch - 1) * d.stride_a + a_rows * d.lda) * dtype_size(d.a);
    const size_t b_bytes = size_t((d.batch - 1) * d.stride_b + b_rows * d.ldb) * dtype_size(d.b);
    const size_t d_bytes = size_t((d.batch - 1) * d.stride_d + d.m * d.ldd) * dtype_size(d.d);
    const size_t c_bytes = d.beta != 0.f ? size_t((d.batch - 1) * d.stride_c + d.m * d.ldc) * dtype_size(d.c) : 0;
    auto al = [](size_t x) { return (x + 255) / 256 * 256; };
    const size_t need_ops = al(a_bytes) + al(b_bytes) + al(c_bytes) + al(d_bytes) + al(e->tmp_bytes) + al(e->tb_bytes);
    NM_REQUIRE(scratch != nullptr && scratch_bytes > need_ops + (size_t(8) << 20),
               "GemmCache::choose_all: scratch of " + std::to_string(scratch_bytes) + " B is too small for " + e->key +
                   " (needs " + std::to_string(need_ops) + " B + workspace)");
    char* p = static_cast<char*>(scratch);
    Bufs b{};
    b.A = p; p += al(a_bytes);
    b.B = p; p += al(b_bytes);
    b.C = c_bytes ? p : nullptr; p += al(c_bytes);
    b.D = p; p += al(d_bytes);
    b.tmp = e->tmp_bytes ? p : nullptr; p += al(e->tmp_bytes);
    b.tb = e->tb_bytes ? p : nullptr; p += al(e->tb_bytes);
    b.ws = p;
    b.ws_bytes = std::min(kMaxSearchWorkspace, scratch_bytes - need_ops);
    gemm::fill_uniform(const_cast<void*>(b.A), d.a, int64_t(a_bytes / dtype_size(d.a)), 1, s);
    gemm::fill_uniform(const_cast<void*>(b.B), d.b, int64_t(b_bytes / dtype_size(d.b)), 2, s);
    if (c_bytes) gemm::fill_uniform(b.C, d.c, int64_t(c_bytes / dtype_size(d.c)), 3, s);
    // candidates
    struct Cand {
      bool custom = false;
      bool tb = false;   // transposed-B variant
      int ci = -1;
      cublasLtMatmulAlgo_t algo{};
      size_t ws = 0;
      std::string name;
      float ms = 0.f;
    };
    std::vector<Cand> cands;
    for (int variant = 0; variant < 2 && e->op != nullptr; ++variant) {
      const bool tbv = variant == 1;
      if (tbv && e->op_t == nullptr) break;
      const cublasLtMatmulDesc_t vop = tbv ? e->op_t : e->op;
      const cublasLtMatrixLayout_t vla = tbv ? e->la_t : e->la;
      cublasLtMatmulPreference_t pref;
      NM_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
      uint64_t wmax = kMaxSearchWorkspace;   // fixed: the candidate set must not depend on the scratch size
      NM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wmax, sizeof(wmax)));
      const bool f32_out = !lt_round_add(d) && d.d == DType::f32;
      const uint32_t mask = f32_out ? uint32_t(CUBLASLT_REDUCTION_SCHEME_MASK)
                                    : uint32_t(CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE);
      NM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask,
                                                     sizeof(mask)));
      cublasLtMatmulHeuristicResult_t res[kHeuristic];
      int n_res = 0;
      NM_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt_, vop, vla, e->lb, e->lc, e->ld, pref, kHeuristic, res, &n_res));
      NM_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
      std::map<std::string, bool> seen;
      auto add_lt = [&](const cublasLtMatmulAlgo_t& a) {
        cublasLtMatmulHeuristicResult_t chk{};
        if (cublasLtMatmulAlgoCheck(lt_, vop, vla, e->lb, e->lc, e->ld, &a, &chk) != CUBLAS_STATUS_SUCCESS) return;
        if (chk.workspaceSize > kMaxSearchWorkspace) return;
        const std::string nm = (tbv ? "ltT" : "") + algo_name(a).substr(tbv ? 2 : 0);
        if (seen.count(nm)) return;
        seen[nm] = true;
        Cand c;
        c.tb = tbv;
        c.algo = a;
        c.ws = chk.workspaceSize;
        c.name = nm;
        cands.push_back(c);
      };
      for (int i = 0; i < n_res; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
        add_lt(res[i].algo);
        int32_t splitk_ok = 0;
        uint32_t red_mask = 0;
        size_t w = 0;
        cublasLtMatmulAlgoCapGetAttribute(&res[i].algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, sizeof(splitk_ok), &w);
        cublasLtMatmulAlgoCapGetAttribute(&res[i].algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &red_mask,
                                          sizeof(red_mask), &w);
        if (!splitk_ok) continue;
        for (int sk : kSplits) {
          for (uint32_t red : {uint32_t(CUBLASLT_REDUCTION_SCHEME_INPLACE), uint32_t(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE),
                               uint32_t(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE)}) {
            if (!(red & mask) || !(red & red_mask)) continue;
            cublasLtMatmulAlgo_t a = res[i].algo;
            const uint32_t skn = uint32_t(sk);
            if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &skn, sizeof(skn)) !=
                CUBLAS_STATUS_SUCCESS)
              continue;
            if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) !=
                CUBLAS_STATUS_SUCCESS)
              continue;
            add_lt(a);
          }
        }
      }
    }
    const auto& customs = gemm::custom_gemms();
    for (size_t i = 0; i < customs.size(); ++i) {
      if (!customs[i].accepts(d)) continue;
      Cand c;
      c.custom = true;
      c.ci = int(i);
      c.ws = customs[i].workspace ? customs[i].workspace(d) : 0;
      if (c.ws > kMaxSearchWorkspace) continue;
      c.name = std::string("custom:") + customs[i].name;
      cands.push_back(c);
    }
    NM_REQUIRE(!cands.empty(), "GemmCache: no candidate for " + e->key + (e->op ? "" : " (needs a custom kernel)"));
    // a forced name (tests: gemm::force_choice) or a saved choice of this shape key wins (resume executes the same
    // kernel); neither is timed, so neither needs the workspace in the scratch
    const std::string full_key = e->key + "|" + dev;
    int pick = -1;
    if (!forced.empty() || saved.count(full_key)) {
      const std::string& want = forced.empty() ? saved[full_key] : forced;
      for (size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].name == want) pick = int(i);
      }
      NM_REQUIRE(pick >= 0, "GemmCache: " + std::string(forced.empty() ? "saved" : "forced") + " kernel " + want +
                                " of " + e->key + " is not a candidate on this build" +
                                (forced.empty() ? " (delete " + path_ + " to search again)" : ""));
    } else {
      // candidates whose workspace exceeds the scratch are not timed
      cands.erase(std::remove_if(cands.begin(), cands.end(), [&](const Cand& c) { return c.ws > b.ws_bytes; }),
                  cands.end());
      NM_REQUIRE(!cands.empty(), "GemmCache: no candidate of " + e->key + " fits the scratch workspace");
      for (auto& c : cands) {   // a candidate that fails to launch (e.g. a custom kernel's host check) is dropped
        try {
          c.ms = time_ms([&] { launch(lt_, *e, c.custom, c.ci, c.tb, &c.algo, b, s); }, 2, 5);
        } catch (const Fatal&) {
          NM_CUDA(cudaStreamSynchronize(s));
          c.ms = 1e30f;
        }
      }
      cands.erase(std::remove_if(cands.begin(), cands.end(), [](const Cand& c) { return c.ms >= 1e30f; }),
                  cands.end());
      NM_REQUIRE(!cands.empty(), "GemmCache: every candidate of " + e->key + " failed to launch");
      std::vector<size_t> order(cands.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return cands[x].ms < cands[y].ms; });
      const size_t top = std::min<size_t>(4, order.size());
      for (size_t i = 0; i < top; ++i) {
        Cand& c = cands[order[i]];
        c.ms = time_ms([&] { launch(lt_, *e, c.custom, c.ci, c.tb, &c.algo, b, s); }, 5, 20);
      }
      std::stable_sort(order.begin(), order.begin() + top, [&](size_t x, size_t y) { return cands[x].ms < cands[y].ms; });
      pick = int(order[0]);
      if (cands[size_t(pick)].custom) {   // O9: a custom kernel must beat the best cuBLASLt candidate by > 1%
        for (size_t i = 0; i < top; ++i) {
          const Cand& c = cands[order[i]];
          if (!c.custom && c.ms <= cands[size_t(pick)].ms * 1.01f) { pick = int(order[i]); break; }
        }
      }
      gemm::ShapeReport r;
      r.key = e->key;
      r.tag = d.tag;
      r.n_candidates = int(cands.size());
      r.chosen = cands[size_t(pick)].name;
      r.ms = cands[size_t(pick)].ms;
      r.tflops = 2.0 * double(d.m) * double(d.n) * double(d.k) * d.batch / (double(r.ms) * 1e-3) / 1e12;
      for (size_t i = 0; i < std::min<size_t>(8, top); ++i) {
        const Cand& c = cands[order[i]];
        r.top.push_back({c.name, c.ms, c.ws});
      }
      for (size_t i : order) r.all.push_back({cands[i].name, cands[i].ms, cands[i].ws});
      reports.push_back(r);
    }
    const Cand& c = cands[size_t(pick)];
    e->use_custom = c.custom;
    e->use_tb = c.tb;
    e->custom = c.ci;
    e->custom_name = c.custom ? c.name : "";
    e->algo = c.algo;
    e->ws = c.ws;
    e->chosen = true;
    chosen_now.emplace_back(int(ei), c.name);
  }
  NM_CUDA(cudaEventDestroy(ev0));
  NM_CUDA(cudaEventDestroy(ev1));
  ws_bytes_ = 0;
  for (Entry* e : entries_) {
    ws_bytes_ = std::max(ws_bytes_, e->ws + (e->tmp_bytes + 255) / 256 * 256 +
                                        (e->use_tb ? (e->tb_bytes + 255) / 256 * 256 : 0));
  }
  std::lock_guard<std::mutex> g(g_reg_mu);
  auto& rep = g_reports()[this];
  rep.insert(rep.end(), reports.begin(), reports.end());
  for (const auto& [id, name] : chosen_now) g_chosen()[{this, id}] = name;
}

size_t GemmCache::workspace_bytes() const { return ws_bytes_; }

void GemmCache::bind_workspace(void* ws) { ws_ = ws; }

void GemmCache::run(int id, const void* A, const void* B, const void* C, void* D, cudaStream_t s) const {
  NM_REQUIRE(id >= 0 && size_t(id) < entries_.size(), "GemmCache::run: bad id");
  const Entry& e = *entries_[size_t(id)];
  NM_REQUIRE(e.chosen, "GemmCache::run: choose_all has not run for " + e.key);
  NM_REQUIRE(ws_ != nullptr || ws_bytes_ == 0, "GemmCache::run: no workspace bound");
  char* w = static_cast<char*>(ws_);
  Bufs b{A, B, const_cast<void*>(C), D, nullptr, 0, nullptr, nullptr};
  if (e.tmp_bytes) {
    b.tmp = w;
    w += (e.tmp_bytes + 255) / 256 * 256;
  }
  if (e.use_tb) {
    b.tb = w;
    w += (e.tb_bytes + 255) / 256 * 256;
  }
  b.ws = w;
  b.ws_bytes = e.ws;
  launch(lt_, e, e.use_custom, e.custom, e.use_tb, &e.algo, b, s);
}

void GemmCache::save() const {
  if (path_.empty()) return;
  const std::string dev = device_tag();
  // per-process temporary: ranks that share one path never write the same file
  std::ostringstream tn;
  tn << path_ << ".tmp." << getpid() << "." << std::hash<std::thread::id>{}(std::this_thread::get_id());
  const std::string tmp = tn.str();
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    NM_REQUIRE(f.good(), "GemmCache::save: cannot write " + tmp);
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (!entries_[i]->chosen) continue;
      f << entries_[i]->key << "|" << dev << "\t" << gemm::chosen_name(*this, int(i)) << "\n";
    }
    NM_REQUIRE(f.good(), "GemmCache::save: write failed " + tmp);
  }
  NM_REQUIRE(std::rename(tmp.c_str(), path_.c_str()) == 0, "GemmCache::save: rename failed " + path_);
}

std::vector<gemm::ShapeReport> gemm::report(const GemmCache& c) {
  std::lock_guard<std::mutex> g(g_reg_mu);
  auto it = g_reports().find(&c);
  return it == g_reports().end() ? std::vector<ShapeReport>{} : it->second;
}

std::string gemm::chosen_name(const GemmCache& c, int id) {
  std::lock_guard<std::mutex> g(g_reg_mu);
  auto it = g_chosen().find({&c, id});
  return it == g_chosen().end() ? std::string() : it->second;
}

void gemm::force_choice(const GemmCache& c, const std::string& name) {
  std::lock_guard<std::mutex> g(g_reg_mu);
  g_forced()[&c] = name;
}

void gemm::forget(const GemmCache& c) {
  std::lock_guard<std::mutex> g(g_reg_mu);
  g_reports().erase(&c);
  g_forced().erase(&c);
  for (auto it = g_chosen().begin(); it != g_chosen().end();) {
    if (it->first.first == &c) it = g_chosen().erase(it); else ++it;
  }
}

// ------------------------------------------------------------------------------------------------ linear descs
GemmDesc linear_fwd_desc(int64_t T, int64_t N, int64_t K, Epilogue e, const char* tag) {
  GemmDesc d;   // Y[T, N] = X[T, K] W[N, K]^T
  d.m = T; d.n = N; d.k = K;
  d.op_a = Op::N; d.op_b = Op::T;
  d.lda = K; d.ldb = K; d.ldc = N; d.ldd = N;
  d.epilogue = e;
  d.tag = tag ? tag : "";
  return d;
}

GemmDesc linear_dgrad_desc(int64_t T, int64_t N, int64_t K, Epilogue e, const char* tag) {
  GemmDesc d;   // dX[T, K] = dY[T, N] W[N, K]
  d.m = T; d.n = K; d.k = N;
  d.op_a = Op::N; d.op_b = Op::N;
  d.lda = N; d.ldb = K; d.ldc = K; d.ldd = K;
  d.epilogue = e;
  d.tag = tag ? tag : "";
  return d;
}

GemmDesc linear_wgrad_desc(int64_t T, int64_t N, int64_t K, Rounding r, const char* tag) {
  GemmDesc d;   // dW[N, K] += dY[T, N]^T X[T, K], f32, in place across microbatches
  d.m = N; d.n = K; d.k = T;
  d.op_a = Op::T; d.op_b = Op::N;
  d.lda = N; d.ldb = K; d.ldc = K; d.ldd = K;
  d.c = DType::f32; d.d = DType::f32;
  d.beta = 1.f;
  d.epilogue = r == Rounding::Levanter ? Epilogue::Bf16RoundAddF32 : Epilogue::None;
  d.tag = tag ? tag : "";
  return d;
}

// ------------------------------------------------------------------------------------------------ custom kernels
namespace {
std::vector<CustomGemm>& custom_list() {
  static std::vector<CustomGemm> v;
  return v;
}
}  // namespace

void register_custom_gemm(const CustomGemm& g) { custom_list().push_back(g); }
const std::vector<CustomGemm>& gemm::custom_gemms() { return custom_list(); }

}  // namespace nm
