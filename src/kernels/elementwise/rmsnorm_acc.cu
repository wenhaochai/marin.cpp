// Accurate-mode (Rounding::Accurate) instantiations of the RMSNorm and the sandwich tails launchers
// (rmsnorm_kernels.cuh), built with nvcc's default contraction (DESIGN.md §3.0: accurate-mode kernels keep it); the
// levanter-mode ones are in rmsnorm.cu (--fmad=false).
#include "kernels/elementwise/rmsnorm_kernels.cuh"

namespace nm::ew::rms {
NM_EW_RMS_LAUNCHERS(template, false)
}  // namespace nm::ew::rms
