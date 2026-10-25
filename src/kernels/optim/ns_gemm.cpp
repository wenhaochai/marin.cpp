// Newton-Schulz GEMMs on cuBLASLt (ns_gemm.hpp).
#include "kernels/optim/ns_gemm.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "kernels/optim/optim_dev.hpp"

namespace nm::optk {

namespace {
constexpr size_t kLtWorkspace = size_t(32) << 20;   // cuBLASLt workspace (Hopper recommendation)
constexpr int kCandidates = 16;

NsGemmDesc dense(NsGemmDesc d) {
  if (!d.ld_p) d.ld_p = d.K;
  if (!d.ld_q) d.ld_q = d.q_trans ? d.K : d.N;
  if (!d.ld_d) d.ld_d = d.N;
  return d;
}
bool same(const NsGemmDesc& a, const NsGemmDesc& b) {
  return a.M == b.M && a.N == b.N && a.K == b.K && a.q_trans == b.q_trans && a.batch == b.batch &&
         a.stride_p == b.stride_p && a.stride_q == b.stride_q && a.stride_d == b.stride_d && a.beta == b.beta &&
         a.ld_p == b.ld_p && a.ld_q == b.ld_q && a.ld_d == b.ld_d && a.alpha == b.alpha;
}
// Column-major layouts of a row-major desc (A = Q, B = P, C = D) with a batch count and the desc's strides.
void make_layouts(const NsGemmDesc& d, int batch, cublasLtMatrixLayout_t* l) {
  if (d.q_trans) NM_CUBLAS(cublasLtMatrixLayoutCreate(&l[0], CUDA_R_32F, d.K, d.N, d.ld_q));
  else NM_CUBLAS(cublasLtMatrixLayoutCreate(&l[0], CUDA_R_32F, d.N, d.K, d.ld_q));
  NM_CUBLAS(cublasLtMatrixLayoutCreate(&l[1], CUDA_R_32F, d.K, d.M, d.ld_p));
  NM_CUBLAS(cublasLtMatrixLayoutCreate(&l[2], CUDA_R_32F, d.N, d.M, d.ld_d));
  const int32_t bc = batch;
  const int64_t s[3] = {d.stride_q, d.stride_p, d.stride_d};
  for (int j = 0; j < 3; ++j) {
    NM_CUBLAS(cublasLtMatrixLayoutSetAttribute(l[j], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    NM_CUBLAS(cublasLtMatrixLayoutSetAttribute(l[j], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s[j], sizeof(s[j])));
  }
}
int64_t p_elems(const NsGemmDesc& d) { return d.batch * d.M * d.K; }
int64_t q_elems(const NsGemmDesc& d) { return d.batch * d.K * d.N; }
}  // namespace

struct NsGemm::Entry {
  NsGemmDesc d;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool chosen = false;
  bool fallback = false;   // chosen from this layout's own heuristic (batch-dependent)
};

NsGemm::NsGemm(NsPrec prec) : prec_(prec) { NM_CUBLAS(cublasLtCreate(&lt_)); }

NsGemm::~NsGemm() {
  for (Entry* e : entries_) {
    if (e->op) cublasLtMatmulDescDestroy(e->op);
    if (e->la) cublasLtMatrixLayoutDestroy(e->la);
    if (e->lb) cublasLtMatrixLayoutDestroy(e->lb);
    if (e->lc) cublasLtMatrixLayoutDestroy(e->lc);
    delete e;
  }
  if (lt_) cublasLtDestroy(lt_);
}

// Row-major D[M,N] = P[M,K] op(Q) is column-major D^T[N,M] = op(Q)^T P^T: cuBLASLt A = Q (op T when q_trans),
// B = P, m = N, n = M.
int NsGemm::plan(const NsGemmDesc& d0) {
  const NsGemmDesc d = dense(d0);
  NM_REQUIRE(d.M > 0 && d.N > 0 && d.K > 0 && d.batch > 0, "empty NS GEMM");
  for (size_t i = 0; i < entries_.size(); ++i)
    if (same(entries_[i]->d, d)) return static_cast<int>(i);
  auto* e = new Entry;
  e->d = d;
  const cublasComputeType_t ct = prec_ == NsPrec::F32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F_FAST_TF32;
  NM_CUBLAS(cublasLtMatmulDescCreate(&e->op, ct, CUDA_R_32F));
  const cublasOperation_t ta = d.q_trans ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
  NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  NM_CUBLAS(cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtMatrixLayout_t l[3];
  make_layouts(d, d.batch, l);
  e->la = l[0], e->lb = l[1], e->lc = l[2];
  entries_.push_back(e);
  return static_cast<int>(entries_.size() - 1);
}

void NsGemm::launch(const Entry& e, const void* algo, const float* P, const float* Q, const float* C, float* D,
                    float beta, cudaStream_t s) const {
  const float alpha = e.d.alpha;
  NM_CUBLAS(cublasLtMatmul(lt_, e.op, &alpha, Q, e.la, P, e.lb, &beta, C ? C : D, e.lc, D, e.lc,
                           static_cast<const cublasLtMatmulAlgo_t*>(algo), ws_, kLtWorkspace, s));
}

void NsGemm::choose(bool timed, cudaStream_t s) {
  cublasLtMatmulPreference_t pref;
  NM_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsb = kLtWorkspace;
  NM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
  const uint32_t no_split = CUBLASLT_REDUCTION_SCHEME_NONE;
  NM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &no_split,
                                                 sizeof(no_split)));
  // Timing scratch: the largest P, Q, D of the planned descs plus a workspace, allocated here and freed below.
  float *P = nullptr, *Q = nullptr, *D = nullptr;
  void* ws = nullptr;
  void* saved_ws = ws_;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (timed) {
    int64_t np = 0, nq = 0, nd = 0;
    for (const Entry* e : entries_) {
      const NsGemmDesc& d = e->d;
      np = std::max(np, d.stride_p * (d.batch - 1) + d.ld_p * d.M);
      nq = std::max(nq, d.stride_q * (d.batch - 1) + d.ld_q * (d.q_trans ? d.N : d.K));
      nd = std::max(nd, d.stride_d * (d.batch - 1) + d.ld_d * d.M);
    }
    NM_CUDA(cudaMalloc(&P, np * sizeof(float)));
    NM_CUDA(cudaMalloc(&Q, nq * sizeof(float)));
    NM_CUDA(cudaMalloc(&D, nd * sizeof(float)));
    NM_CUDA(cudaMalloc(&ws, kLtWorkspace));
    NM_CUDA(cudaMemsetAsync(P, 0, np * sizeof(float), s));
    NM_CUDA(cudaMemsetAsync(Q, 0, nq * sizeof(float), s));
    NM_CUDA(cudaMemsetAsync(D, 0, nd * sizeof(float), s));
    NM_CUDA(cudaEventCreate(&e0));
    NM_CUDA(cudaEventCreate(&e1));
    ws_ = ws;
  }
  for (Entry* e : entries_) {
    if (e->chosen) continue;
    // Candidates without split-K (reduction scheme NONE: a split-K workspace grows with the batch, so its validity
    // and the choice would depend on it), from the heuristic of a fixed batch-2 layout of the shape, kept if they
    // also run this desc's layout: the choice is then a function of the matrix shape alone, so a matrix gets the
    // same bits whatever the batch it is in (ZeRO-1 ranks own different counts of each shape: 1 rank == N ranks
    // bitwise). Timed mode times them on the real batch and may pick per batch (not rank-invariant unless the choice
    // is saved and loaded).
    cublasLtMatmulHeuristicResult_t res[kCandidates];
    int n = 0;
    {
      cublasLtMatrixLayout_t two[3];
      make_layouts(e->d, 2, two);
      int n2 = 0;
      const cublasStatus_t st =
          cublasLtMatmulAlgoGetHeuristic(lt_, e->op, two[0], two[1], two[2], two[2], pref, kCandidates, res, &n2);
      for (int j = 0; j < 3; ++j) cublasLtMatrixLayoutDestroy(two[j]);
      if (st == CUBLAS_STATUS_SUCCESS)
        for (int i = 0; i < n2; ++i) {
          cublasLtMatmulHeuristicResult_t chk{};
          if (res[i].state == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoCheck(lt_, e->op, e->la, e->lb, e->lc, e->lc, &res[i].algo, &chk) ==
                  CUBLAS_STATUS_SUCCESS &&
              chk.workspaceSize <= kLtWorkspace)
            res[n++] = res[i];
        }
    }
    if (n == 0) {   // no batch-2 candidate runs this layout: this layout's own heuristic (batch-dependent choice)
      NM_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt_, e->op, e->la, e->lb, e->lc, e->lc, pref, kCandidates, res, &n));
      e->fallback = true;
    }
    NM_REQUIRE(n > 0, "cuBLASLt has no algorithm for an NS GEMM of shape " + std::to_string(e->d.M) + "x" +
                          std::to_string(e->d.N) + "x" + std::to_string(e->d.K));
    int best = 0;
    if (timed && n > 1) {
      float best_ms = 1e30f;
      for (int i = 0; i < n; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
        launch(*e, &res[i].algo, P, Q, nullptr, D, e->d.beta, s);   // warm-up
        NM_CUDA(cudaEventRecord(e0, s));
        for (int r = 0; r < 5; ++r) launch(*e, &res[i].algo, P, Q, nullptr, D, e->d.beta, s);
        NM_CUDA(cudaEventRecord(e1, s));
        NM_CUDA(cudaEventSynchronize(e1));
        float ms = 0.0f;
        NM_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best_ms) best_ms = ms, best = i;
      }
    }
    e->algo = res[best].algo;
    e->chosen = true;
  }
  if (timed) {
    NM_CUDA(cudaStreamSynchronize(s));
    cudaFree(P), cudaFree(Q), cudaFree(D), cudaFree(ws);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    ws_ = saved_ws;
  }
  cublasLtMatmulPreferenceDestroy(pref);
}

size_t NsGemm::workspace_bytes() const {
  size_t split = 0;
  if (prec_ == NsPrec::X3TF32)
    for (const Entry* e : entries_)
      split = std::max(split, size_t(2 * (p_elems(e->d) + q_elems(e->d))) * sizeof(float));
  return kLtWorkspace + split;
}

void NsGemm::bind_workspace(void* ws) { ws_ = ws; }

void NsGemm::run(int id, const float* P, const float* Q, const float* C, float* D, cudaStream_t s) const {
  const Entry& e = *entries_.at(id);
  NM_REQUIRE(e.chosen && ws_, "NsGemm::run before choose() and bind_workspace()");
  if (prec_ != NsPrec::X3TF32) {
    launch(e, &e.algo, P, Q, C, D, e.d.beta, s);
    return;
  }
  // 3xTF32: D = Ps Qb + beta C;  D += Pb Qs;  D += Pb Qb (small products first). Operands are batch-contiguous.
  NM_REQUIRE(e.d.alpha == 1.0f, "3xTF32 NS GEMMs take alpha = 1");
  NM_REQUIRE(e.d.stride_p == e.d.M * e.d.K && e.d.stride_q == e.d.K * e.d.N && e.d.ld_p == e.d.K &&
                 e.d.ld_q == (e.d.q_trans ? e.d.K : e.d.N),
             "3xTF32 needs dense, contiguous batches (no symmetric blocks)");
  float* base = reinterpret_cast<float*>(static_cast<char*>(ws_) + kLtWorkspace);
  const int64_t np = p_elems(e.d), nq = q_elems(e.d);
  float *pb = base, *ps = base + np, *qb = base + 2 * np, *qs = base + 2 * np + nq;
  tf32_split(P, pb, ps, np, s);
  if (Q == P) qb = pb, qs = ps;
  else tf32_split(Q, qb, qs, nq, s);
  launch(e, &e.algo, ps, qb, C, D, e.d.beta, s);
  launch(e, &e.algo, pb, qs, D, D, 1.0f, s);
  launch(e, &e.algo, pb, qb, D, D, 1.0f, s);
}

NsIterPlan plan_ns_iteration(NsGemm& g, int64_t m, int64_t n, int count, float a, float b, float c, bool symmetric,
                             bool fold_c) {
  NsIterPlan p;
  p.m = m, p.n = n, p.count = count, p.c = c;
  p.fold_c = fold_c && g.prec() != NsPrec::X3TF32;   // 3xTF32 sums three products: alpha would scale only one
  const float alpha = p.fold_c ? c : 1.0f;
  p.symmetric = symmetric && g.prec() != NsPrec::X3TF32 && m >= 64;
  const int64_t mm = m * m, mn = m * n;
  if (!p.symmetric) {
    p.g1[0] = g.plan({m, m, n, true, count, mn, mn, mm, 0.0f});        // A = X X^T
    p.g2[0] = g.plan({m, m, m, false, count, mm, mm, mm, b, 0, 0, 0, alpha});   // B = (cA) A + b A
  } else {
    p.m1 = ((m / 2 + 31) / 32) * 32;
    const int64_t m1 = p.m1, m2 = m - m1;
    // A_ij = X_i X_j^T: P = X rows i (ld n), Q = X rows j (q_trans, ld n), D = A block (ld m)
    p.g1[0] = g.plan({m1, m1, n, true, count, mn, mn, mm, 0.0f, n, n, m});
    p.g1[1] = g.plan({m1, m2, n, true, count, mn, mn, mm, 0.0f, n, n, m});
    p.g1[2] = g.plan({m2, m2, n, true, count, mn, mn, mm, 0.0f, n, n, m});
    // B_ij = cA rows i . A cols j + b A_ij (ld m everywhere)
    p.g2[0] = g.plan({m1, m1, m, false, count, mm, mm, mm, b, m, m, m, alpha});
    p.g2[1] = g.plan({m1, m2, m, false, count, mm, mm, mm, b, m, m, m, alpha});
    p.g2[2] = g.plan({m2, m2, m, false, count, mm, mm, mm, b, m, m, m, alpha});
  }
  p.g3 = g.plan({m, n, m, false, count, mm, mn, mn, a});               // X' = B X + a X
  return p;
}

float* run_ns_iteration(const NsGemm& g, const NsIterPlan& p, int steps, float* X, float* X2, float* A, float* cA,
                        float* B, cudaStream_t s) {
  const int64_t m = p.m, m1 = p.m1, n = p.n;
  const int64_t mm = int64_t(p.count) * m * m;
  const float* P2 = p.fold_c ? A : cA;   // GEMM2's left operand: c * A, or A with alpha = c
  for (int it = 0; it < steps; ++it) {
    if (!p.symmetric) {
      g.run(p.g1[0], X, X, nullptr, A, s);
      if (!p.fold_c) ns_scale(cA, A, p.c, mm, s);
      g.run(p.g2[0], P2, A, A, B, s);
    } else {
      g.run(p.g1[0], X, X, nullptr, A, s);
      g.run(p.g1[1], X, X + m1 * n, nullptr, A + m1, s);
      g.run(p.g1[2], X + m1 * n, X + m1 * n, nullptr, A + m1 * m + m1, s);
      ns_mirror(A, p.count, static_cast<int>(m), static_cast<int>(m1), s);
      if (!p.fold_c) ns_scale(cA, A, p.c, mm, s);
      g.run(p.g2[0], P2, A, A, B, s);
      g.run(p.g2[1], P2, A + m1, A + m1, B + m1, s);
      g.run(p.g2[2], P2 + m1 * m, A + m1, A + m1 * m + m1, B + m1 * m + m1, s);
      ns_mirror(B, p.count, static_cast<int>(m), static_cast<int>(m1), s);
    }
    g.run(p.g3, B, X, X, X2, s);
    std::swap(X, X2);
  }
  return X;
}

std::string NsGemm::save() const {
  std::string out;
  for (const Entry* e : entries_) {
    NM_REQUIRE(e->chosen, "NsGemm::save before choose()");
    out.append(reinterpret_cast<const char*>(&e->d), sizeof(NsGemmDesc));
    out.append(reinterpret_cast<const char*>(&e->algo), sizeof(cublasLtMatmulAlgo_t));
  }
  return out;
}

std::vector<int64_t> NsGemm::algo_config(int id) const {
  const Entry& e = *entries_.at(id);
  NM_REQUIRE(e.chosen, "NsGemm::algo_config before choose()");
  std::vector<int64_t> out;
  for (cublasLtMatmulAlgoConfigAttributes_t a :
       {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
        CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID}) {
    uint32_t v32 = 0;   // 32-bit attributes; INNER_SHAPE_ID and CLUSTER_SHAPE_ID are 16-bit
    uint16_t v16 = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&e.algo, a, &v32, sizeof(v32), &written) == CUBLAS_STATUS_SUCCESS)
      out.push_back(v32);
    else if (cublasLtMatmulAlgoConfigGetAttribute(&e.algo, a, &v16, sizeof(v16), &written) == CUBLAS_STATUS_SUCCESS)
      out.push_back(v16);
    else
      out.push_back(-1);
  }
  return out;
}

uint32_t NsGemm::reduction_scheme(int id) const {
  const Entry& e = *entries_.at(id);
  NM_REQUIRE(e.chosen, "NsGemm::reduction_scheme before choose()");
  uint32_t scheme = 0;
  size_t written = 0;
  NM_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(&e.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme,
                                                 sizeof(scheme), &written));
  return scheme;
}

bool NsGemm::batch_fallback(int id) const { return entries_.at(id)->fallback; }

void NsGemm::load(const std::string& bytes) {
  const size_t rec = sizeof(NsGemmDesc) + sizeof(cublasLtMatmulAlgo_t);
  NM_REQUIRE(bytes.size() == rec * entries_.size(), "NsGemm::load: saved algorithms are for another plan");
  for (size_t i = 0; i < entries_.size(); ++i) {
    NsGemmDesc d;
    std::memcpy(&d, bytes.data() + i * rec, sizeof(d));
    NM_REQUIRE(same(d, entries_[i]->d), "NsGemm::load: saved algorithms are for another plan");
    std::memcpy(&entries_[i]->algo, bytes.data() + i * rec + sizeof(d), sizeof(cublasLtMatmulAlgo_t));
    entries_[i]->chosen = true;
  }
}

}  // namespace nm::optk
