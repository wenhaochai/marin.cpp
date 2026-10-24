// Accurate-mode (Rounding::Accurate) instantiations of the QK-norm + RoPE launchers (qknorm_rope_kernels.cuh), built
// with nvcc's default contraction (DESIGN.md §3.0: accurate-mode kernels keep it); the levanter-mode ones are in
// qknorm_rope.cu (--fmad=false).
#include "kernels/elementwise/qknorm_rope_kernels.cuh"

namespace nm::ew::qk {
NM_EW_QK_LAUNCHERS(template, false)
}  // namespace nm::ew::qk
