// marincpp core types. Header-only declarations; no CUDA headers are required to include this file.
// Design: docs/DESIGN.md §1-§3. Every public header of marincpp includes this one.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

// Opaque CUDA / library handles. These typedefs are identical to the ones in the CUDA, NCCL and cuBLASLt headers,
// so a translation unit may include both (a repeated identical typedef is legal C++).
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;
typedef struct ncclComm* ncclComm_t;
typedef struct cublasLtContext* cublasLtHandle_t;

namespace nm {

// ------------------------------------------------------------------------------------------------ scalar types
enum class DType : uint8_t { f32 = 0, bf16 = 1, f16 = 2, i32 = 3, u32 = 4, i64 = 5, u8 = 6, f64 = 7 };
constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::f32: case DType::i32: case DType::u32: return 4;
    case DType::bf16: case DType::f16: return 2;
    case DType::i64: case DType::f64: return 8;
    case DType::u8: return 1;
  }
  return 0;
}
const char* dtype_name(DType t);

// Storage type for bfloat16 in host code and in headers. Device code uses __nv_bfloat16 (same bits).
struct bf16_t { uint16_t bits; };
static_assert(sizeof(bf16_t) == 2);

// Host f32 <-> bf16, round to nearest even (the rounding of XLA's convert and of __float2bfloat16_rn). NaN stays a
// quiet NaN.
inline uint16_t f32_to_bf16_bits(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return static_cast<uint16_t>((u >> 16) | 0x0040u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline float bf16_bits_to_f32(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

// ------------------------------------------------------------------------------------------------ numerics policy
// DESIGN.md §3.0. Three independent axes.
//
// ComputePolicy (exec.policy): BF16 = training (bf16 activations, bf16 GEMM inputs, f32 accumulate). F32 = parity
// tests T2.2/T2.4 and gradient gate V1 only: f32 activations and residual, true-f32 GEMMs (CUBLAS_COMPUTE_32F_PEDANTIC),
// RefF32 attention, CE implementation A in f32; eager (no graph).
enum class ComputePolicy : uint8_t { BF16 = 0, F32 = 1 };
//
// Rounding (exec.rounding): Levanter = copy every class-P rounding point local to one kernel (bf16 casts between
// elementwise ops, bf16 dlogits, bf16 scale multiply of q and dq, bf16 per-microbatch trunk/gain gradients).
// Accurate = f32 inside every kernel, one rounding at the kernel output. The M-class items (eps inside rsqrt,
// bf16(1/sqrt(hd)) as the attention scale, absolute RoPE positions, EOS-derived segments) are identical in both.
enum class Rounding : uint8_t { Levanter = 0, Accurate = 1 };
//
// The class-P items where the fast configuration departs from levanter, one knob each; the preset levanter_strict
// sets every one to levanter's choice (DESIGN.md §3.0 table). The attention backend and the microbatch split are
// separate knobs (exec.attention, exec.micro_seqs / exec.remat).
enum class EmbedAccum : uint8_t { F32 = 0, Bf16Sequential = 1 };   // exec.embed_accum
enum class CeDwRound : uint8_t { None = 0, PerChunkBf16 = 1 };      // exec.ce_dw_round (strict: bf16 per chunk + per microbatch)
struct Numerics {
  ComputePolicy policy = ComputePolicy::BF16;
  Rounding rounding = Rounding::Levanter;
  DType ce_logits_store = DType::bf16;   // exec.ce_logits resolved from T1.0 item 2 (bf16 or f32)
  EmbedAccum embed_accum = EmbedAccum::F32;
  CeDwRound ce_dw_round = CeDwRound::None;
  DType grad_reduce = DType::f32;        // exec.grad_reduce (strict: bf16)
};
//
// Activation policy (exec.remat, DESIGN.md §2.3): All = save everything; Min = recompute elementwise outputs;
// Layer = keep only each layer's input and recompute the whole layer forward, GEMMs and attention included (levanter's
// policy; strict plans and variants that need it).
enum class Remat : uint8_t { All = 0, Min = 1, Layer = 2 };
//
// FMA contraction (DESIGN.md §3.0): host code builds with -ffp-contract=off. Device units of levanter-mode elementwise
// kernels, optimizer, init and eval aggregation build with --fmad=false and write each contraction explicitly
// (__fmaf_rn, or __fmul_rn/__fadd_rn to forbid one), as the T1.0 XLA PTX dump shows per formula. Means are sum / n.

// ------------------------------------------------------------------------------------------------ tensor view
// A non-owning view of device (or pinned host) memory. Row-major unless strides say otherwise. Kernels take raw
// pointers plus explicit shapes; TensorView is the currency of the C++ host code and of the memory planner.
struct TensorView {
  void* data = nullptr;
  DType dtype = DType::f32;
  int ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};
  int64_t stride[4] = {0, 0, 0, 0};  // in elements

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return ndim ? n : 0;
  }
  size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype); }
  template <class T> T* as() const { return static_cast<T*>(data); }
};

TensorView make_view(void* data, DType t, std::initializer_list<int64_t> shape);  // contiguous strides

// ------------------------------------------------------------------------------------------------ errors
// All failures are fatal in the training binary. fatal() throws nm::Fatal; nothing catches it below main() (exit 1),
// the C ABI entry points (status code + nm_last_error) and a worker thread's top frame (which records it and stops
// the process). No exception crosses a CUDA graph: graphs hold no host code. `where` is __FILE__:__LINE__.
struct Fatal : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// A failed CUDA, NCCL or cuBLAS call (NM_CUDA / NM_NCCL / NM_CUBLAS); the C ABI reports it as NM_ERR_DEVICE.
struct DeviceError : Fatal {
  using Fatal::Fatal;
};
[[noreturn]] void fatal(const char* where, const std::string& msg);
void cuda_check(int err, const char* expr, const char* where);
void nccl_check(int err, const char* expr, const char* where);
void cublas_check(int err, const char* expr, const char* where);

#define NM_STR2(x) #x
#define NM_STR(x) NM_STR2(x)
#define NM_WHERE __FILE__ ":" NM_STR(__LINE__)
#define NM_CUDA(x) ::nm::cuda_check(static_cast<int>(x), #x, NM_WHERE)
#define NM_NCCL(x) ::nm::nccl_check(static_cast<int>(x), #x, NM_WHERE)
#define NM_CUBLAS(x) ::nm::cublas_check(static_cast<int>(x), #x, NM_WHERE)
#define NM_REQUIRE(cond, msg) \
  do { if (!(cond)) ::nm::fatal(NM_WHERE, std::string("requirement failed: " #cond ": ") + (msg)); } while (0)

// ------------------------------------------------------------------------------------------------ hashing
// SHA-256 (FIPS 180-4) as 64 lowercase hex digits: run identity (config.hpp) and content hashes of input files.
std::string sha256_hex(const void* data, size_t n);
inline std::string sha256_hex(std::string_view s) { return sha256_hex(s.data(), s.size()); }
std::string sha256_file_hex(const std::string& path);   // fatal if the file cannot be read

// ------------------------------------------------------------------------------------------------ constants
inline constexpr int kMaxGpus = 8;            // one node, at most 8 H100 (owner scope)
inline constexpr size_t kArenaAlign = 256;    // every planned buffer starts on a 256-byte boundary
inline constexpr int kWarp = 32;

// Exit codes of the training process (wq --partial semantics, tools/wq/README.md; DESIGN.md §7.5).
enum class ExitCode : int {
  Done = 0,              // all N updates and the final eval are durable; DONE marker written
  ConfigError = 1,       // invalid config, hash mismatch on resume, missing data file: never retried
  NumericsAbort = 2,     // non-finite loss or gradient norm (state of the last good checkpoint kept)
  InfraRetry = 99,       // node health probe failed, or a voluntary stop before the deadline: wq requeues
  Interrupted = 143,     // SIGTERM/SIGINT received; state saved (or a recent checkpoint judged sufficient)
  // 124 is set by `timeout` (wq host end), never by the trainer; wq requeues 99, 124 and 143.
};

}  // namespace nm
