// Portable unit (NM_ARCHS_PORTABLE): runs on sm_80 and sm_90a.
#include <cuda_runtime.h>

#include "core/device_probe.hpp"
#include "marincpp/core.hpp"

namespace nm {
namespace {

__global__ void k_probe_portable(int* out) {
#ifdef __CUDA_ARCH__
  *out = __CUDA_ARCH__;
#endif
}

}  // namespace

// Shared by both probe units: launch, copy the result back, map "no kernel image" to 0.
int run_probe(void (*kernel)(int*)) {
  int* d = nullptr;
  NM_CUDA(cudaMalloc(&d, sizeof(int)));
  NM_CUDA(cudaMemset(d, 0, sizeof(int)));
  kernel<<<1, 1>>>(d);
  cudaError_t e = cudaGetLastError();
  int h = 0;
  if (e == cudaSuccess) {
    NM_CUDA(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  } else if (e != cudaErrorNoKernelImageForDevice && e != cudaErrorInvalidDeviceFunction) {
    cudaFree(d);
    NM_CUDA(e);
  }
  NM_CUDA(cudaFree(d));
  return h;
}

int probe_portable_kernel_arch() { return run_probe(k_probe_portable); }

}  // namespace nm
