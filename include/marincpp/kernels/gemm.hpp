// GEMMs: cuBLASLt and CUTLASS SM90 kernels behind one id per shape, chosen by a timed search over deterministic
// candidates at plan time; fused epilogues for the stage-1 schedule. DESIGN.md §3.1, §3.7.
//
// Layout convention: every activation is row-major [tokens, features]; every weight is row-major [out, in]
// (levanter's out_first). Then, for Y[T,N] = X[T,K] W[N,K]^T:
//   linear_fwd   Y  = X  W^T     (bf16 x bf16 -> bf16, f32 accumulate)
//   linear_dgrad dX = dY W       (bf16 x bf16 -> bf16)
//   linear_wgrad dW += dY^T X    (bf16 x bf16 -> f32, beta = 1, in place across microbatches). Rounding::Levanter
//                                 uses the epilogue Bf16RoundAddF32: D = C_f32 + f32(bf16(acc)) on the COMPLETE
//                                 accumulator (levanter's per-microbatch bf16 gradient, no transient, no extra pass)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

enum class Compute : uint8_t {
  F32 = 0,           // CUBLAS_COMPUTE_32F with bf16 inputs: tensor cores, f32 accumulate (all bf16-policy model GEMMs)
  TF32 = 1,          // f32 inputs on TF32 tensor cores (Newton-Schulz if T1.0 shows levanter used TF32)
  F32_PEDANTIC = 2,  // true f32 FFMA (f32 compute policy; NS if 3xTF32 fails T1.6)
  F32_3XTF32 = 3,    // split-precision f32 on TF32 tensor cores (CUTLASS): f32-level error at ~2x the FFMA rate
};

enum class Op : uint8_t { N = 0, T = 1 };  // of the row-major operand as written in the formulas above

// Fused epilogues (CUTLASS SM90 epilogue visitor trees). Each keeps levanter's rounding points: the accumulator is first
// rounded to bf16 (the tensor levanter stores), then the bf16 op chain runs. Fused output == unfused path, bitwise.
enum class Epilogue : uint8_t {
  None = 0,
  Bf16RoundAddF32 = 1,   // wgrad under Levanter: D = C + f32(bf16(acc))
  QkNormRope = 2,        // QKV fwd: also write q_rot, k_rot, rstd_q/k (N tile holds whole heads)
  SwiGLU = 3,            // gate/up fwd with interleaved 64-row blocks: write gu and h
  SwiGLUBwd = 4,         // down dgrad: load gu, write dgu (dh never exists)
  SandwichTail = 5,      // o_proj / down fwd at D = 512 (full-row 128 x 512 tile): post-norm + residual + next pre-norm
  RowsumDoO = 6,         // o_proj dgrad: also write FA3's softmax_d = rowsum(dO * O) per head
  Bf16ChunkToF32Acc = 7, // CE wgrad, strict: acc_mb += f32(bf16(acc)) per chunk
};

// Split-K policy: only deterministic reductions. cuBLASLt reduction schemes NONE, INPLACE, COMPUTE_TYPE, OUTPUT_TYPE are
// allowed, atomics never; CUTLASS stream-K only with deterministic fixup. When the epilogue rounds the result
// (Bf16RoundAddF32, bf16 D), only schemes that reduce in f32 before the rounding are candidates.
enum class SplitPolicy : uint8_t { NoSplit = 0, DeterministicSplit = 1 };

struct GemmDesc {
  // D[m, n] = alpha * op(A)[m, k] * op(B)[k, n] + beta * C[m, n], all row-major with leading dimensions in elements
  int64_t m = 0, n = 0, k = 0;
  Op op_a = Op::N, op_b = Op::N;
  DType a = DType::bf16, b = DType::bf16, c = DType::bf16, d = DType::bf16;
  int64_t lda = 0, ldb = 0, ldc = 0, ldd = 0;
  int batch = 1;                                   // strided batch (Newton-Schulz over same-shape matrices)
  int64_t stride_a = 0, stride_b = 0, stride_c = 0, stride_d = 0;
  Compute compute = Compute::F32;
  float alpha = 1.f, beta = 0.f;
  Epilogue epilogue = Epilogue::None;
  const void* epilogue_args = nullptr;             // per-epilogue struct (gains, tables, extra outputs)
  SplitPolicy split = SplitPolicy::DeterministicSplit;
  bool symmetric_output = false;                   // compute upper-triangle tiles only (NS: X X^T, A A), mirror after
  std::string tag;                                 // "L*.qkv.fwd", printed by --profile-ops
};

// Owns the descriptors and the chosen kernel of every GEMM shape in the step. plan() records the desc; choose_all()
// runs the timed search (default; ~40 shapes, a few seconds) over cuBLASLt algorithms, CUTLASS kernels (incl. stream-K)
// and registered custom kernels, keeps the fastest deterministic candidate per shape, sizes the cuBLASLt workspace from
// the chosen split counts, and writes <run>/gemm_algos.bin; every resumed segment reloads it, so a resumed run executes
// the same kernels (bitwise resume, DESIGN.md §11).
class GemmCache {
 public:
  GemmCache(cublasLtHandle_t lt, std::string cache_path);
  int plan(const GemmDesc& d);                      // returns an id; same desc -> same id
  void choose_all(cudaStream_t s, void* scratch, size_t scratch_bytes);   // at plan time, before graph capture
  size_t workspace_bytes() const;                   // after choose_all (split-K counts); declared in the arena
  void bind_workspace(void* ws);
  void run(int id, const void* A, const void* B, const void* C, void* D, cudaStream_t s) const;
  void save() const;

 private:
  cublasLtHandle_t lt_;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 0;
  std::string path_;
  struct Entry;
  std::vector<Entry*> entries_;
};

// Convenience wrappers that build the GemmDesc for the three linear-layer GEMMs.
GemmDesc linear_fwd_desc(int64_t T, int64_t N, int64_t K, Epilogue e, const char* tag);
GemmDesc linear_dgrad_desc(int64_t T, int64_t N, int64_t K, Epilogue e, const char* tag);
GemmDesc linear_wgrad_desc(int64_t T, int64_t N, int64_t K, Rounding r, const char* tag);

// Custom SM90 kernels (CuTe/WGMMA) register the descs they accept; GemmCache::choose_all times them with the rest.
// A custom kernel takes a shape when it is the fastest deterministic candidate and passes the same tests.
struct CustomGemm {
  const char* name;
  bool (*accepts)(const GemmDesc&);
  void (*launch)(const GemmDesc&, const void* A, const void* B, const void* C, void* D, void* ws, cudaStream_t);
  size_t (*workspace)(const GemmDesc&);
};
void register_custom_gemm(const CustomGemm& g);

}  // namespace nm
