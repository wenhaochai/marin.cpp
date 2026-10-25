// cuBLASLt GEMMs of the fused CE (implementation A's logits, and the plain-dz dgrad / wgrad of B1 and B1.5). Row-major
// formulas of ce_internal.hpp mapped onto column-major cuBLASLt. The workspace is the CE workspace's kLtWsBytes region
// (Hopper cuBLASLt offers no algorithm without one at these shapes). The algorithm of a (kind, io, D, V, ld) class is
// the heuristic's first choice for kCanonRows rows whose split-K reduction (if any) is f32 before a bf16 output; every
// cuBLASLt reduction scheme is deterministic. It is reused for
// every row count it supports (cublasLtMatmulAlgoCheck), so that a row's result does not depend on the row count of the
// call as long as the kernel's K partition does not depend on M (tests/s5 bitwise checks it); a row count the canonical
// algorithm does not support gets its own heuristic choice (reported once on stderr).
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>

#include "kernels/cross_entropy/ce_internal.hpp"

namespace nm::ce {
namespace {

constexpr int kCanonRows = 16384;
constexpr int64_t kMaxPlansPerDev = 256;

enum Kind { kLogits = 0, kDgrad = 1, kWgrad = 2 };

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

using Key = std::tuple<int, int, int, int, int, int64_t, int64_t>;   // dev, kind, io, D, V, ld, rows

std::mutex g_mu;
std::map<int, cublasLtHandle_t> g_handles;
std::map<Key, Plan> g_plans;
std::map<std::tuple<int, int, int, int, int, int64_t>, cublasLtMatmulAlgo_t> g_canon;   // per class

int cur_dev() {
  int d = 0;
  NM_CUDA(cudaGetDevice(&d));
  return d;
}

// Descriptors of one (kind, io, D, V, ld, rows). Column-major view: see each case.
Plan make_desc(int kind, DType io, int D, int V, int64_t ld, int rows) {
  Plan p;
  const bool f32 = io == DType::f32;
  const cudaDataType_t ab = f32 ? CUDA_R_32F : CUDA_R_16BF;
  const cublasComputeType_t ct = f32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
  NM_CUBLAS(cublasLtMatmulDescCreate(&p.op, ct, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
  cudaDataType_t cd = CUDA_R_32F;
  uint64_t am = 0, an = 0, bm = 0, bn = 0, cm = 0, cn = 0;
  int64_t lda = 0, ldb = 0, ldc = 0;
  if (kind == kLogits) {
    // C^T [V x rows] (ld) = op_T(W: D x V, ld D) * x^T (D x rows, ld D)
    ta = CUBLAS_OP_T;
    am = D; an = V; lda = D;
    bm = D; bn = rows; ldb = D;
    cm = V; cn = rows; ldc = ld;
    cd = CUDA_R_32F;
  } else if (kind == kDgrad) {
    // dx^T [D x rows] (ld D) = W^T (D x V, ld D) * dz^T (V x rows, ld)
    am = D; an = V; lda = D;
    bm = V; bn = rows; ldb = ld;
    cm = D; cn = rows; ldc = D;
    cd = ab;
  } else {
    // dw^T [D x V] (ld D) += x^T (D x rows, ld D) * op_T(dz^T: V x rows, ld)
    tb = CUBLAS_OP_T;
    am = D; an = rows; lda = D;
    bm = V; bn = rows; ldb = ld;
    cm = D; cn = V; ldc = D;
    cd = CUDA_R_32F;
  }
  NM_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  NM_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  NM_CUBLAS(cublasLtMatrixLayoutCreate(&p.a, ab, am, an, lda));
  NM_CUBLAS(cublasLtMatrixLayoutCreate(&p.b, ab, bm, bn, ldb));
  NM_CUBLAS(cublasLtMatrixLayoutCreate(&p.c, cd, cm, cn, ldc));
  return p;
}

void free_desc(Plan& p) {
  if (p.a) cublasLtMatrixLayoutDestroy(p.a);
  if (p.b) cublasLtMatrixLayoutDestroy(p.b);
  if (p.c) cublasLtMatrixLayoutDestroy(p.c);
  if (p.op) cublasLtMatmulDescDestroy(p.op);
  p = Plan{};
}

// A split-K algorithm whose output is bf16 must reduce its partial sums in f32 (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
// before the one rounding: OUTPUT_TYPE and INPLACE add bf16 partials, an extra rounding per part (DESIGN.md §3.7; on
// the A100 it raised the dX error at V = 128,256 to 1.33x levanter's).
bool reduction_ok(cublasLtHandle_t lt, const Plan& p, bool bf16_out, cublasLtMatmulAlgo_t* a) {
  int32_t split = 1;
  uint32_t scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
  size_t n = 0;
  cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split), &n);
  cublasLtMatmulAlgoConfigGetAttribute(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme, sizeof(scheme), &n);
  if (!bf16_out || split <= 1 || scheme == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return true;
  auto try_config = [&](int32_t sp, uint32_t sc) {
    cublasLtMatmulAlgo_t b = *a;
    if (cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sp, sizeof(sp)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &sc, sizeof(sc)) !=
            CUBLAS_STATUS_SUCCESS)
      return false;
    cublasLtMatmulHeuristicResult_t chk;
    if (cublasLtMatmulAlgoCheck(lt, p.op, p.a, p.b, p.c, p.c, &b, &chk) != CUBLAS_STATUS_SUCCESS ||
        chk.workspaceSize > kLtWsBytes)
      return false;
    *a = b;
    return true;
  };
  // the same split with an f32 reduction, else no split
  return try_config(split, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) || try_config(1, CUBLASLT_REDUCTION_SCHEME_NONE);
}

bool heuristic(cublasLtHandle_t lt, const Plan& p, bool bf16_out, cublasLtMatmulAlgo_t* out) {
  cublasLtMatmulPreference_t pref;
  NM_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  const uint64_t ws = kLtWsBytes;
  NM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t res[16];
  int n = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.a, p.b, p.c, p.c, pref, 16, res, &n);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS) return false;
  for (int i = 0; i < n; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > kLtWsBytes) continue;
    cublasLtMatmulAlgo_t a = res[i].algo;
    if (!reduction_ok(lt, p, bf16_out, &a)) continue;
    *out = a;
    return true;
  }
  return false;
}

const Plan& get_plan(cublasLtHandle_t lt, int kind, DType io, int D, int V, int64_t ld, int rows) {
  const int dev = cur_dev();
  const bool bf16_out = kind == kDgrad && io == DType::bf16;
  std::lock_guard<std::mutex> lk(g_mu);
  const Key key{dev, kind, int(io), D, V, ld, rows};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  const auto ckey = std::make_tuple(dev, kind, int(io), D, V, ld);
  auto cit = g_canon.find(ckey);
  if (cit == g_canon.end()) {
    Plan canon = make_desc(kind, io, D, V, ld, kCanonRows);
    cublasLtMatmulAlgo_t a{};
    const bool ok = heuristic(lt, canon, bf16_out, &a);
    free_desc(canon);
    NM_REQUIRE(ok, "fused CE: no cuBLASLt algorithm within the CE cuBLASLt workspace for this shape");
    cit = g_canon.emplace(ckey, a).first;
  }
  // every distinct tail-chunk row count adds a plan: past kMaxPlansPerDev plans of this device, drop them all (the
  // canonical algorithms stay, so a dropped plan costs one descriptor build and check). Safe because one host thread
  // drives a device and consumes each returned plan before its next call
  int64_t on_dev = 0;
  for (const auto& [k, v] : g_plans) on_dev += std::get<0>(k) == dev;
  if (on_dev >= kMaxPlansPerDev) {
    for (auto i = g_plans.begin(); i != g_plans.end();) {
      if (std::get<0>(i->first) != dev) { ++i; continue; }
      free_desc(i->second);
      i = g_plans.erase(i);
    }
  }
  Plan p = make_desc(kind, io, D, V, ld, rows);
  cublasLtMatmulHeuristicResult_t chk;
  p.algo = cit->second;
  if (cublasLtMatmulAlgoCheck(lt, p.op, p.a, p.b, p.c, p.c, &p.algo, &chk) != CUBLAS_STATUS_SUCCESS ||
      chk.workspaceSize > kLtWsBytes) {
    NM_REQUIRE(heuristic(lt, p, bf16_out, &p.algo), "fused CE: no cuBLASLt algorithm within the CE cuBLASLt workspace");
    std::fprintf(stderr, "marincpp CE: cuBLASLt kind %d rows %d D %d V %d uses its own algorithm (the canonical one "
                         "does not support it)\n", kind, rows, D, V);
  }
  return g_plans.emplace(key, p).first->second;
}

void matmul(cublasLtHandle_t lt, const Plan& p, const void* A, const void* B, void* C, float beta, void* ws,
            cudaStream_t st) {
  const float alpha = 1.f;
  NM_REQUIRE(ws, "fused CE: the cuBLASLt workspace region is missing");
  NM_CUBLAS(cublasLtMatmul(lt, p.op, &alpha, A, p.a, B, p.b, &beta, C, p.c, C, p.c, &p.algo, ws, kLtWsBytes, st));
}

}  // namespace

cublasLtHandle_t lt_handle() {
  const int dev = cur_dev();
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_handles.find(dev);
  if (it != g_handles.end()) return it->second;
  cublasLtHandle_t h;
  NM_CUBLAS(cublasLtCreate(&h));
  g_handles[dev] = h;
  return h;
}

void lt_logits(cublasLtHandle_t lt, DType io, const void* x, const void* w, float* c, int64_t ldc, int rows, int D,
               int V, void* ws, cudaStream_t st) {
  if (rows <= 0) return;
  matmul(lt, get_plan(lt, kLogits, io, D, V, ldc, rows), w, x, c, 0.f, ws, st);
}

void lt_dgrad(cublasLtHandle_t lt, DType io, const void* dz, int64_t ldz, const void* w, void* dx, float beta,
              int rows, int D, int V, void* ws, cudaStream_t st) {
  if (rows <= 0) return;
  matmul(lt, get_plan(lt, kDgrad, io, D, V, ldz, rows), w, dz, dx, beta, ws, st);
}

void lt_wgrad(cublasLtHandle_t lt, DType io, const void* dz, int64_t ldz, const void* x, float* dw, int rows, int D,
              int V, void* ws, cudaStream_t st) {
  if (rows <= 0) return;
  matmul(lt, get_plan(lt, kWgrad, io, D, V, ldz, rows), x, dz, dw, 1.f, ws, st);
}

}  // namespace nm::ce
