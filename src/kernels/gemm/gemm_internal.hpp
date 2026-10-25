// Internal interface of src/kernels/gemm.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/kernels/gemm.hpp"

namespace nm::gemm {

// Portable kernels (gemm_kernels.cu).
// D_f32 = C_f32 + f32(t_bf16), elementwise over rows x cols (row strides in elements): the Levanter wgrad epilogue
// when it runs after a cuBLASLt GEMM that wrote bf16(acc) to t (reduction in f32 before the rounding).
void bf16_round_add(const float* C, int64_t ldc, const void* t, int64_t ldt, float* D, int64_t ldd, int64_t rows,
                    int64_t cols, cudaStream_t s);
// out[c, r] = in[r, c] for a bf16 [rows, cols] matrix (row strides in elements): the transposed-B GEMM variant.
void transpose_bf16(const void* in, int64_t ld_in, void* out, int64_t ld_out, int64_t rows, int64_t cols,
                    cudaStream_t s);
// Deterministic pseudo-random fill (timing inputs): bf16 or f32 values ~ U(-1, 1).
void fill_uniform(void* p, DType t, int64_t n, uint32_t seed, cudaStream_t s);

// One timed candidate of a shape (for the search report).
struct Candidate {
  std::string name;       // "lt:<algo id>:tile<t>:stages<s>:split<k>:red<r>:swz<z>:co<c>:cl<c>" or "custom:<name>"
  float ms = 0.f;         // median ms
  size_t workspace = 0;
};

// The search report of one shape (GemmCache::report).
struct ShapeReport {
  std::string key;
  std::string tag;
  std::string chosen;
  float ms = 0.f;
  double tflops = 0.0;
  int n_candidates = 0;
  std::vector<Candidate> top;   // fastest first (up to 8)
  std::vector<Candidate> all;   // every candidate that launched, fastest first (first-pass times past the top 4)
};

// The search reports of every choose_all of cache c so far (shapes loaded from gemm_algos.bin have none).
std::vector<ShapeReport> report(const GemmCache& c);
// Name of the kernel chosen for shape id of cache c ("" before choose_all).
std::string chosen_name(const GemmCache& c, int id);
// Tests: the next choose_all of c takes the candidate of this name for every new shape (no timing; an unknown name
// fails), so a gate can run every candidate a search may pick.
void force_choice(const GemmCache& c, const std::string& name);
// Drops the side-table records (reports, chosen names, forced name) of c; call before c is destroyed (GemmCache has
// no destructor yet, BUILD_LOG request), so a later cache at the same address starts clean.
void forget(const GemmCache& c);

// Custom kernels registered by register_custom_gemm (CUTLASS SM90 units register at static-init time).
const std::vector<CustomGemm>& custom_gemms();

// The process-wide cache the C ABI entry points use (one per device, searched on first use of a shape, never saved).
GemmCache& capi_cache();

}  // namespace nm::gemm
