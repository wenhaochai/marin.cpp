// Accurate-mode (Rounding::Accurate) instantiations of the SwiGLU launchers (swiglu_kernels.cuh), built with nvcc's
// default contraction (DESIGN.md §3.0: accurate-mode kernels keep it); the levanter-mode ones are in swiglu.cu
// (--fmad=false).
#include "kernels/elementwise/swiglu_kernels.cuh"

namespace nm::ew::sw {
NM_EW_SW_LAUNCHERS(template, false)
}  // namespace nm::ew::sw
