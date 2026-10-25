// Batched f32 GEMMs of the Newton-Schulz iteration (optim.md §4.3) on cuBLASLt. DESIGN.md §4.2, BUILD_LOG "Decisions
// for S4-S6" item 14: TF32 operands are rounded to nearest (cuBLASLt's CUBLAS_COMPUTE_32F_FAST_TF32 does, as XLA's
// emitters do; tests/s6 probe), true f32 is CUBLAS_COMPUTE_32F_PEDANTIC, 3xTF32 splits each operand into
// big = tf32_rna(x) and small = x - big and sums small*big + big*small + big*big on TF32 tensor cores.
//
// Row-major convention: D[M, N] = P[M, K] op(Q) + beta * C, with op(Q) = Q[K, N] or Q[N, K]^T, C laid out as D, every
// operand a strided batch. No split-K atomics are ever used (cuBLASLt's reduction schemes are deterministic); the
// algorithm of each shape is fixed at plan time (heuristic or timed search) and can be saved and reloaded, so a step
// is bitwise repeatable and a resumed run executes the same kernels. run() is graph-capturable (no allocation, no
// host synchronization) once bind_workspace() has been called.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/core.hpp"

namespace nm::optk {

enum class NsPrec : int { F32 = 0, TF32 = 1, X3TF32 = 2 };   // capi.h order: 0 f32 pedantic, 1 tf32, 2 3xtf32

struct NsGemmDesc {
  int64_t M = 0, N = 0, K = 0;
  bool q_trans = false;          // op(Q) = Q^T (Q stored [N, K])
  int batch = 1;
  int64_t stride_p = 0, stride_q = 0, stride_d = 0;
  float beta = 0.0f;             // D = P op(Q) + beta * C (C may alias D; C has D's layout)
  int64_t ld_p = 0, ld_q = 0, ld_d = 0;   // row strides; 0 = dense (K, K or N, N)
  float alpha = 1.0f;            // D = alpha * P op(Q) + beta * C
};

class NsGemm;

// The Newton-Schulz iteration of one chunk (`count` matrices X [m, n] in the wide orientation, m <= n, contiguous):
//   A = X X^T;  cA = c * A;  B = cA A + b A;  X' = B X + a X
// symmetric: A and B are symmetric (B = c A^2 + b A), so their GEMMs compute the blocks [0:m1, 0:m1], [0:m1, m1:m],
// [m1:m, m1:m] of a 2 x 2 split (m1 = m / 2 rounded up to 32) and mirror the upper-right block into the lower-left:
// 3/4 of the X X^T and cA A products. The mirrored blocks equal their transposes exactly; the full products of XLA
// are symmetric only up to rounding (same error class: T1.6). Not for 3xTF32.
struct NsIterPlan {
  int64_t m = 0, n = 0, m1 = 0;
  int count = 0;
  bool symmetric = false;
  int g1[3] = {-1, -1, -1}, g2[3] = {-1, -1, -1}, g3 = -1;   // NsGemm ids (blocks 11, 12, 22; full: [0] only)
  float c = 0.0f;
  bool fold_c = false;           // B = c (A A) + b A in one GEMM (alpha = c), no cA = c * A pass
};
NsIterPlan plan_ns_iteration(NsGemm& g, int64_t m, int64_t n, int count, float a, float b, float c, bool symmetric,
                             bool fold_c = false);
// Runs `steps` iterations on X; returns the buffer that holds the result (X or X2). A, cA, B: [count, m, m].
float* run_ns_iteration(const NsGemm& g, const NsIterPlan& p, int steps, float* X, float* X2, float* A, float* cA,
                        float* B, cudaStream_t s);

class NsGemm {
 public:
  explicit NsGemm(NsPrec prec);
  ~NsGemm();
  NsGemm(const NsGemm&) = delete;
  NsGemm& operator=(const NsGemm&) = delete;

  NsPrec prec() const { return prec_; }
  int plan(const NsGemmDesc& d);                         // same desc -> same id
  // Picks the algorithm of every planned desc: timed = fastest of the heuristic candidates (each timed on scratch
  // buffers it allocates and frees here), else the heuristic's first. Before graph capture.
  void choose(bool timed, cudaStream_t s);
  size_t workspace_bytes() const;                        // cuBLASLt workspace + 3xTF32 operand splits
  void bind_workspace(void* ws);
  void run(int id, const float* P, const float* Q, const float* C, float* D, cudaStream_t s) const;
  // The chosen algorithms (opaque bytes) for a resumed run; load() refuses a different set of descs.
  std::string save() const;
  void load(const std::string& bytes);
  // Introspection (tests/s6 s6_nsalgo): the configuration of the chosen algorithm of desc `id` (the
  // CUBLASLT_ALGO_CONFIG_* attributes ID, TILE_ID, STAGES_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING,
  // CUSTOM_OPTION, INNER_SHAPE_ID, CLUSTER_SHAPE_ID, in that order; not the opaque bytes, which also carry a field
  // left uninitialized on the H100), its reduction scheme (NONE = no split-K), and whether the choice fell back to
  // the desc's own (batch-dependent) heuristic because no batch-2 candidate runs its layout.
  std::vector<int64_t> algo_config(int id) const;
  uint32_t reduction_scheme(int id) const;
  bool batch_fallback(int id) const;

 private:
  struct Entry;
  void launch(const Entry& e, const void* algo, const float* P, const float* Q, const float* C, float* D, float beta,
              cudaStream_t s) const;
  NsPrec prec_;
  cublasLtHandle_t lt_ = nullptr;
  void* ws_ = nullptr;
  std::vector<Entry*> entries_;
};

}  // namespace nm::optk
