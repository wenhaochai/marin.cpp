// Device threefry (capi.h nm_threefry2x32_device): the same inline body as the host port (threefry_impl.hpp), so the
// device bits equal the host bits by construction; the test compares them on every input.
#include <cuda_runtime.h>

#include "core/capi_util.hpp"
#include "data/threefry_impl.hpp"
#include "marincpp/capi.h"

namespace {
__global__ void k_threefry2x32(uint32_t k0, uint32_t k1, uint32_t* x0, uint32_t* x1, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t a = x0[i], b = x1[i];
    nm::tf::threefry2x32_20(k0, k1, a, b);
    x0[i] = a;
    x1[i] = b;
  }
}
}  // namespace

extern "C" int nm_threefry2x32_device(uint32_t k0, uint32_t k1, uint32_t* x0, uint32_t* x1, int64_t n,
                                      void* stream) {
  return nm::capi_call([&] {
    if (n <= 0) return;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    k_threefry2x32<<<static_cast<unsigned>(blocks < 4096 ? blocks : 4096), threads, 0,
                     static_cast<cudaStream_t>(stream)>>>(k0, k1, x0, x1, n);
    NM_CUDA(cudaGetLastError());
  });
}
