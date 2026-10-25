// CUTLASS SM90 GEMM candidates of the GemmCache search (DESIGN.md §3.7): TMA warp-specialized kernels (cooperative and
// ping-pong, persistent, and stream-K with the deterministic fixup), bf16 A / B, f32 accumulate, for the three
// linear-layer orientations and the Levanter wgrad epilogue D = C + f32(bf16(acc)) as an epilogue visitor tree (no
// transient, no extra pass). Included by gemm_sm90_*.cu (one operand-layout / epilogue combination per unit, so the
// build parallelizes); each unit registers its kernels with register_custom_gemm at static-init time.
//
// Row-major D[m, n] = op(A)[m, k] op(B)[k, n] maps to CUTLASS's D = A (M x K) * B (K x N) with
//   op_a N: A RowMajor (K-major)      op_a T: A ColumnMajor (M-major)
//   op_b T: B ColumnMajor (K-major)   op_b N: B RowMajor (N-major)
// fwd = (N, T), dgrad = (N, N), wgrad = (T, N). C and D row-major. Operands must be packed (ld = columns) and 16-byte
// aligned.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/gemm/gemm_internal.hpp"

namespace nm::gemm::sm90 {

using namespace cute;

enum class Fuse { LinComb, RoundAdd };   // alpha acc + beta C (any D) | D = C + f32(bf16(acc)) (f32 C, D)

template <class LayoutA, class LayoutB, class ElementD, Fuse F, class Tile, class Cluster, class KSched, class ESched,
          class TSched>
struct Kernel {
  using ElementC = std::conditional_t<std::is_same_v<ElementD, float>, float, ElementD>;
  static constexpr int kAlignD = 16 / int(sizeof(ElementD));
  static constexpr int kAlignC = 16 / int(sizeof(ElementC));
  static constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;
  using RoundAcc = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, RN>,
      cutlass::epilogue::fusion::Sm90AccFetch>;
  using RoundAddEVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, float, float, RN>,
      cutlass::epilogue::fusion::Sm90SrcFetch<float>, RoundAcc>;
  using LinComb = cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float, RN>;
  using FusionOp = std::conditional_t<F == Fuse::RoundAdd, RoundAddEVT, LinComb>;

  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, cutlass::layout::RowMajor, kAlignC,
      ElementD, cutlass::layout::RowMajor, kAlignD, ESched, FusionOp>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::bfloat16_t, LayoutA, 8, cutlass::bfloat16_t,
      LayoutB, 8, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      KSched>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, TSched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static constexpr bool kStreamK = std::is_same_v<TSched, cutlass::gemm::StreamKScheduler>;

  static typename Gemm::Arguments args(const GemmDesc& d, const void* A, const void* B, const void* C, void* D) {
    using SA = typename GemmKernel::StrideA;
    using SB = typename GemmKernel::StrideB;
    using SC = typename GemmKernel::StrideC;
    using SD = typename GemmKernel::StrideD;
    const int M = int(d.m), N = int(d.n), K = int(d.k);
    SA sa = cutlass::make_cute_packed_stride(SA{}, make_shape(M, K, 1));
    SB sb = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
    SC sc = cutlass::make_cute_packed_stride(SC{}, make_shape(M, N, 1));
    SD sd = cutlass::make_cute_packed_stride(SD{}, make_shape(M, N, 1));
    typename Gemm::Arguments a{};
    a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {M, N, K, 1};
    a.mainloop = {static_cast<const cutlass::bfloat16_t*>(A), sa, static_cast<const cutlass::bfloat16_t*>(B), sb};
    a.epilogue.ptr_C = static_cast<const ElementC*>(d.beta != 0.f || F == Fuse::RoundAdd ? C : nullptr);
    a.epilogue.dC = sc;
    a.epilogue.ptr_D = static_cast<ElementD*>(D);
    a.epilogue.dD = sd;
    if constexpr (F == Fuse::LinComb) {
      a.epilogue.thread.alpha = d.alpha;
      a.epilogue.thread.beta = d.beta;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    a.hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(dev);
    if constexpr (kStreamK) {
      a.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
      a.scheduler.decomposition_mode = cutlass::gemm::kernel::detail::DecompositionMode::StreamK;
    }
    return a;
  }

  static size_t workspace(const GemmDesc& d) {
    return Gemm::get_workspace_size(args(d, nullptr, nullptr, nullptr, nullptr));
  }

  static void launch(const GemmDesc& d, const void* A, const void* B, const void* C, void* D, void* ws,
                     cudaStream_t s) {
    Gemm g;
    auto a = args(d, A, B, C, D);
    cutlass::Status st = g.can_implement(a);
    NM_REQUIRE(st == cutlass::Status::kSuccess, std::string("CUTLASS can_implement: ") + cutlass::cutlassGetStatusString(st));
    st = g.initialize(a, ws, s);
    NM_REQUIRE(st == cutlass::Status::kSuccess, std::string("CUTLASS initialize: ") + cutlass::cutlassGetStatusString(st));
    st = g.run(s);
    if (st != cutlass::Status::kSuccess) {
      throw DeviceError(std::string("CUTLASS run: ") + cutlass::cutlassGetStatusString(st) + ": " +
                        cudaGetErrorString(cudaGetLastError()));
    }
  }
};

// Which descs a kernel of (op_a, op_b, D type, epilogue) takes: packed, 16-byte aligned, single batch, bf16 x bf16
// with f32 accumulation.
inline bool on_sm90() {   // the current device runs sm_90a code (these kernels have no other image)
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
  return major == 9;
}

inline bool accepts_common(const GemmDesc& d, Op op_a, Op op_b, DType out, Fuse f) {
  if (!on_sm90()) return false;
  if (d.compute != Compute::F32 || d.a != DType::bf16 || d.b != DType::bf16 || d.batch != 1) return false;
  if (d.op_a != op_a || d.op_b != op_b || d.d != out || d.symmetric_output) return false;
  const int64_t a_cols = d.op_a == Op::N ? d.k : d.m;
  const int64_t b_cols = d.op_b == Op::N ? d.n : d.k;
  if (d.lda != a_cols || d.ldb != b_cols || d.ldd != d.n) return false;
  if (a_cols % 8 || b_cols % 8 || d.n % (out == DType::f32 ? 4 : 8)) return false;
  if (d.m > INT32_MAX || d.n > INT32_MAX || d.k > INT32_MAX) return false;
  if (f == Fuse::RoundAdd) return d.epilogue == Epilogue::Bf16RoundAddF32 && d.c == DType::f32 && d.ldc == d.n;
  if (d.epilogue != Epilogue::None || d.alpha != 1.f) return false;
  return d.beta == 0.f || (d.c == out && d.ldc == d.n);
}

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ECoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EPing = cutlass::epilogue::TmaWarpSpecialized;
using Persist = cutlass::gemm::PersistentScheduler;
using StreamK = cutlass::gemm::StreamKScheduler;

}  // namespace nm::gemm::sm90

// One unit: NM_SM90_GEMM_UNIT(tag, LayoutA, LayoutB, ElementD, Fuse, op_a, op_b, DType)
#define NM_SM90_GEMM_UNIT(TAG, LA, LB, ED, FU, OPA, OPB, OUT)                                                       \
  namespace nm::gemm::sm90 {                                                                                         \
  namespace {                                                                                                        \
  using K1 = Kernel<LA, LB, ED, FU, Shape<_128, _256, _64>, Shape<_2, _1, _1>, Coop, ECoop, Persist>;                \
  using K2 = Kernel<LA, LB, ED, FU, Shape<_256, _128, _64>, Shape<_1, _2, _1>, Coop, ECoop, Persist>;                \
  using K3 = Kernel<LA, LB, ED, FU, Shape<_128, _128, _64>, Shape<_2, _1, _1>, Ping, EPing, Persist>;                \
  using K4 = Kernel<LA, LB, ED, FU, Shape<_128, _256, _64>, Shape<_1, _1, _1>, Coop, ECoop, StreamK>;                \
  struct Reg {                                                                                                       \
    Reg() {                                                                                                          \
      struct R1 {}; struct R2 {}; struct R3 {}; struct R4 {};                                                        \
      add_one<K1, R1>("cutlass:" TAG ":128x256x64:c2x1:coop:persistent");                                            \
      add_one<K2, R2>("cutlass:" TAG ":256x128x64:c1x2:coop:persistent");                                            \
      add_one<K3, R3>("cutlass:" TAG ":128x128x64:c2x1:pingpong:persistent");                                        \
      add_one<K4, R4>("cutlass:" TAG ":128x256x64:c1x1:coop:streamk-det");                                           \
    }                                                                                                                \
    template <class K, class Tag>                                                                                    \
    static void add_one(const char* name) {                                                                          \
      CustomGemm g;                                                                                                  \
      g.name = name;                                                                                                 \
      g.accepts = [](const GemmDesc& d) { return accepts_common(d, OPA, OPB, OUT, FU); };                             \
      g.launch = &K::launch;                                                                                         \
      g.workspace = &K::workspace;                                                                                   \
      register_custom_gemm(g);                                                                                       \
    }                                                                                                                \
  } reg_;                                                                                                            \
  }                                                                                                                  \
  }
