// SwiGLU: argument checks, dispatch and the public entry points; the kernels are in swiglu_kernels.cuh. This unit
// builds with --fmad=false and instantiates the levanter-mode launchers; the accurate-mode ones come from swiglu_acc.cu
// (nvcc's default contraction, DESIGN.md §3.0).
#include <string>

#include "kernels/elementwise/swiglu_kernels.cuh"

namespace nm::ew {
namespace sw {
NM_EW_SW_LAUNCHERS(template, true)
NM_EW_SW_LAUNCHERS(extern template, false)
}  // namespace sw
using namespace sw;

namespace {

void check(const SwArgs& a, const char* what) {
  NM_REQUIRE(a.T >= 0, std::string(what) + ": T < 0");
  NM_REQUIRE(a.F > 0 && a.F % 64 == 0, std::string(what) + ": F must be a positive multiple of 64 (interleave block)");
  NM_REQUIRE(a.gu && a.out, std::string(what) + ": missing tensor");
  NM_REQUIRE(aligned16(a.gu) && aligned16(a.out) && (!a.dh || aligned16(a.dh)),
             std::string(what) + ": every tensor must be 16-byte aligned");
}

}  // namespace

void swiglu_fwd_dt(DType st, const void* gu, void* h, int64_t T, int F, Rounding r, cudaStream_t s) {
  const SwArgs a{nullptr, gu, h, T, F};
  check(a, "swiglu_fwd");
  if (T == 0) return;
  NM_EW_DISPATCH(fwd, st, r, a, s);
}
void swiglu_bwd_dt(DType st, const void* dh, const void* gu, void* dgu, int64_t T, int F, Rounding r, cudaStream_t s) {
  const SwArgs a{dh, gu, dgu, T, F};
  check(a, "swiglu_bwd");
  NM_REQUIRE(dh, "swiglu_bwd: dh is required");
  if (T == 0) return;
  NM_EW_DISPATCH(bwd, st, r, a, s);
}

}  // namespace nm::ew

namespace nm {
void swiglu_fwd(const bf16_t* gu, bf16_t* h, int64_t T, int F, Rounding r, cudaStream_t s) {
  ew::swiglu_fwd_dt(DType::bf16, gu, h, T, F, r, s);
}
void swiglu_bwd(const bf16_t* dh, const bf16_t* gu, bf16_t* dgu, int64_t T, int F, Rounding r, cudaStream_t s) {
  ew::swiglu_bwd_dt(DType::bf16, dh, gu, dgu, T, F, r, s);
}
}  // namespace nm
