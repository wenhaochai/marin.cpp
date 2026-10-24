// QK-norm + RoPE: argument checks, dispatch and the public entry points; the kernels are in qknorm_rope_kernels.cuh.
// This unit builds with --fmad=false and instantiates the levanter-mode launchers; the accurate-mode ones come from
// qknorm_rope_acc.cu (nvcc's default contraction, DESIGN.md §3.0).
#include <string>

#include "kernels/elementwise/qknorm_rope_kernels.cuh"

namespace nm::ew {
namespace qk {
NM_EW_QK_LAUNCHERS(template, true)
NM_EW_QK_LAUNCHERS(extern template, false)
}  // namespace qk
using namespace qk;

namespace {

void check_shape(int64_t T, int H, int KV, int hd, int64_t ld, int64_t dld, const char* what) {
  NM_REQUIRE(T >= 0 && H > 0 && KV >= 0, std::string(what) + ": bad T / H / KV");
  NM_REQUIRE(hd == 64 || hd == 128, std::string(what) + ": hd must be 64 or 128, got " + std::to_string(hd));
  NM_REQUIRE(ld % 8 == 0 && ld >= int64_t(H + 2 * KV) * hd,
             std::string(what) + ": qkv_ld must be a multiple of 8 and >= (H + 2 KV) hd");
  NM_REQUIRE(dld % 8 == 0 && dld >= 0, std::string(what) + ": dqkv_ld must be a multiple of 8");
}
// Under levanter with bf16 storage the scale is a bf16 constant (numerics.md §3.4: bf16(1/sqrt(hd))), applied as a
// bf16 multiply.
void check_scale(DType st, Rounding r, float scale, const char* what) {
  if (st == DType::bf16 && r == Rounding::Levanter)
    NM_REQUIRE(bf16_bits_to_f32(f32_to_bf16_bits(scale)) == scale,
               std::string(what) + ": under levanter rounding the q scale must be a bf16 value (bf16(1/sqrt(hd)))");
}
void check_ptrs(std::initializer_list<const void*> ps, const char* what) {
  for (const void* p : ps)
    NM_REQUIRE(p == nullptr || aligned16(p), std::string(what) + ": every tensor must be 16-byte aligned");
}

}  // namespace

void qknorm_rope_fwd_dt(DType st, const QkNormRopeFwd& a, Rounding r, cudaStream_t s) {
  check_shape(a.T, a.H, a.KV, a.hd, a.qkv_ld, 0, "qknorm_rope_fwd");
  NM_REQUIRE(a.KV > 0 && a.H % a.KV == 0, "qknorm_rope_fwd: H must be a multiple of KV");
  NM_REQUIRE(a.qkv && a.gq && a.gk && a.cos_tab && a.sin_tab && a.pos, "qknorm_rope_fwd: missing input");
  check_ptrs({a.qkv, a.gq, a.gk, a.cos_tab, a.sin_tab, a.q_rot, a.k_rot}, "qknorm_rope_fwd");
  check_scale(st, r, a.q_scale, "qknorm_rope_fwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(fwd, st, r, a, s);
}
void rope_fwd_dt(DType st, const void* x, int64_t ld, const void* cos_tab, const void* sin_tab, const int32_t* pos,
                 void* out, int64_t T, int nheads, int hd, float scale, Rounding r, cudaStream_t s) {
  NM_REQUIRE(T >= 0 && nheads > 0, "rope_fwd: bad T / nheads");
  NM_REQUIRE(hd == 64 || hd == 128, "rope_fwd: hd must be 64 or 128");
  NM_REQUIRE(ld % 8 == 0 && ld >= int64_t(nheads) * hd, "rope_fwd: ld must be a multiple of 8 and >= nheads hd");
  NM_REQUIRE(x && cos_tab && sin_tab && pos && out, "rope_fwd: missing tensor");
  check_ptrs({x, cos_tab, sin_tab, out}, "rope_fwd");
  check_scale(st, r, scale, "rope_fwd");
  if (T == 0) return;
  const RopeArgs a{x, ld, cos_tab, sin_tab, pos, out, T, nheads, hd, scale};
  NM_EW_DISPATCH(rope, st, r, a, s);
}

void qknorm_rope_bwd_dt(DType st, const QkNormRopeBwd& a, Rounding r, cudaStream_t s) {
  check_shape(a.T, a.H, a.KV, a.hd, a.qkv_ld, a.dqkv_ld, "qknorm_rope_bwd");
  NM_REQUIRE(a.KV > 0 && a.H % a.KV == 0, "qknorm_rope_bwd: H must be a multiple of KV");
  NM_REQUIRE(a.dq_rot || a.dk_rot, "qknorm_rope_bwd: dq_rot and dk_rot are both null");
  NM_REQUIRE(a.qkv && a.gq && a.gk && a.cos_tab && a.sin_tab && a.pos && a.dqkv, "qknorm_rope_bwd: missing input");
  NM_REQUIRE(a.dqkv_ld >= int64_t(a.H + a.KV) * a.hd, "qknorm_rope_bwd: dqkv_ld < (H + KV) hd");
  check_ptrs({a.dq_rot, a.dk_rot, a.qkv, a.gq, a.gk, a.cos_tab, a.sin_tab, a.dqkv}, "qknorm_rope_bwd");
  check_scale(st, r, a.q_scale, "qknorm_rope_bwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(bwd, st, r, a, s);
}

}  // namespace nm::ew

namespace nm {
void qknorm_rope_fwd(const QkNormRopeFwd& a, Rounding r, cudaStream_t s) {
  ew::qknorm_rope_fwd_dt(DType::bf16, a, r, s);
}
void qknorm_rope_bwd(const QkNormRopeBwd& a, Rounding r, cudaStream_t s) {
  ew::qknorm_rope_bwd_dt(DType::bf16, a, r, s);
}
void dq_post_qknorm_rope_bwd(const DqPostQkNormRopeBwd& a, Rounding r, cudaStream_t s) {
  ew::check_shape(a.T, a.H, 0, a.hd, a.qkv_ld, a.dqkv_ld, "dq_post_qknorm_rope_bwd");
  NM_REQUIRE(a.dq_accum && a.qkv && a.gq && a.cos_tab && a.sin_tab && a.pos && a.dqkv,
             "dq_post_qknorm_rope_bwd: missing input");
  ew::check_ptrs({a.dq_accum, a.qkv, a.gq, a.cos_tab, a.sin_tab, a.dqkv}, "dq_post_qknorm_rope_bwd");
  ew::check_scale(DType::bf16, r, a.q_scale, "dq_post_qknorm_rope_bwd");
  if (a.T == 0) return;
  NM_EW_DISPATCH(ew::qk::post, DType::bf16, r, a, s);
}
}  // namespace nm
