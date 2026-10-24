// RMSNorm and the sandwich tails: argument checks, dispatch and the public entry points; the kernels are in
// rmsnorm_kernels.cuh. This unit builds with --fmad=false and instantiates the levanter-mode launchers; the accurate-
// mode ones come from rmsnorm_acc.cu (nvcc's default contraction, DESIGN.md §3.0).
#include <string>

#include "kernels/elementwise/rmsnorm_kernels.cuh"

namespace nm::ew {
namespace rms {
NM_EW_RMS_LAUNCHERS(template, true)
NM_EW_RMS_LAUNCHERS(extern template, false)
}  // namespace rms
using namespace rms;

namespace {

void check_common(int64_t T, int D, const char* what) {
  NM_REQUIRE(T >= 0, std::string(what) + ": T < 0");
  check_row_width(D, what);
}
void check_ptrs(std::initializer_list<const void*> ps, const char* what) {
  for (const void* p : ps)
    NM_REQUIRE(p == nullptr || aligned16(p), std::string(what) + ": every tensor must be 16-byte aligned");
}

}  // namespace

int64_t norm_partial_rows(int64_t T) { return norm_parts(T); }

void rmsnorm_fwd_dt(DType st, const RmsNormFwd& a, Rounding r, cudaStream_t s) {
  check_common(a.T, a.D, "rmsnorm_fwd");
  NM_REQUIRE(a.x && a.gain, "rmsnorm_fwd: x and gain are required");
  check_ptrs({a.x, a.gain, a.y}, "rmsnorm_fwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(launch_fwd, st, r, a, s);
}
void rmsnorm_bwd_add_dt(DType st, const RmsNormBwdAdd& a, Rounding r, cudaStream_t s) {
  check_common(a.T, a.D, "rmsnorm_bwd_add");
  NM_REQUIRE(a.dy && a.x && a.gain && a.dx, "rmsnorm_bwd_add: dy, x, gain and dx are required");
  check_ptrs({a.dy, a.x, a.gain, a.dres, a.dx}, "rmsnorm_bwd_add");
  if (a.T == 0) return;
  NM_EW_DISPATCH(launch_bwd, st, r, a, s);
}
void sandwich_tail_fwd_dt(DType st, const SandwichTailFwd& a, Rounding r, cudaStream_t s) {
  check_common(a.T, a.D, "sandwich_tail_fwd");
  NM_REQUIRE(a.branch && a.x_in && a.g_next && a.x_out, "sandwich_tail_fwd: branch, x_in, g_next, x_out required");
  check_ptrs({a.branch, a.x_in, a.g_post, a.g_next, a.x_out, a.a_next}, "sandwich_tail_fwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(launch_sw_fwd, st, r, a, s);
}
void sandwich_tail_bwd_dt(DType st, const SandwichTailBwd& a, Rounding r, cudaStream_t s) {
  check_common(a.T, a.D, "sandwich_tail_bwd");
  NM_REQUIRE(a.dx_out && a.da_next && a.x_out && a.g_next && a.dx_in && a.dbranch,
             "sandwich_tail_bwd: dx_out, da_next, x_out, g_next, dx_in, dbranch required");
  NM_REQUIRE(!a.g_post || a.branch, "sandwich_tail_bwd: g_post needs branch");
  check_ptrs({a.dx_out, a.da_next, a.x_out, a.g_next, a.branch, a.g_post, a.dx_in, a.dbranch}, "sandwich_tail_bwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(launch_sw_bwd, st, r, a, s);
}

}  // namespace nm::ew

namespace nm {
void rmsnorm_fwd(const RmsNormFwd& a, Rounding r, cudaStream_t s) { ew::rmsnorm_fwd_dt(DType::bf16, a, r, s); }
void rmsnorm_bwd_add(const RmsNormBwdAdd& a, Rounding r, cudaStream_t s) {
  ew::rmsnorm_bwd_add_dt(DType::bf16, a, r, s);
}
void sandwich_tail_fwd(const SandwichTailFwd& a, Rounding r, cudaStream_t s) {
  ew::sandwich_tail_fwd_dt(DType::bf16, a, r, s);
}
void sandwich_tail_bwd(const SandwichTailBwd& a, Rounding r, cudaStream_t s) {
  ew::sandwich_tail_bwd_dt(DType::bf16, a, r, s);
}
}  // namespace nm
