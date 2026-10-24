// sm_90a-only unit (NM_ARCHS_SM90A): has no sm_80 image, so on an A100 the launch fails with "no kernel image".
#include <cuda_runtime.h>

#include "core/device_probe.hpp"

namespace nm {

int run_probe(void (*kernel)(int*));

namespace {
__global__ void k_probe_sm90a(int* out) {
#ifdef __CUDA_ARCH__
  *out = __CUDA_ARCH__;
#endif
}
}  // namespace

int probe_sm90a_kernel_arch() { return run_probe(k_probe_sm90a); }

}  // namespace nm
