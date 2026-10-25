// Small kernels of the fused CE module: dX combination, the strict per-microbatch dW rounding, column zeroing for
// partial-width readouts, and the fixed-order weight sum of the gradient scale. Built with --fmad=false.
#include <algorithm>

#include "kernels/cross_entropy/ce_internal.hpp"
#include "kernels/cross_entropy/ce_mma.cuh"

namespace nm::ce {
namespace {

__global__ void k_dx_combine_bf16(const bf16* tmp, const bf16* old, bf16* dx, int64_t n, bool acc) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const float t = __bfloat162float(tmp[i]);
    dx[i] = acc ? __float2bfloat16_rn(__fadd_rn(__bfloat162float(old[i]), t)) : tmp[i];
  }
}
__global__ void k_dx_combine_f32(const float* tmp, const float* old, float* dx, int64_t n, bool acc) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    dx[i] = acc ? __fadd_rn(old[i], tmp[i]) : tmp[i];
}

__global__ void k_finish_microbatch(float4* dw, float4* mb, int64_t n4) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n4; i += int64_t(gridDim.x) * blockDim.x) {
    float4 a = dw[i];
    const float4 b = mb[i];
    a.x = __fadd_rn(a.x, __bfloat162float(__float2bfloat16_rn(b.x)));
    a.y = __fadd_rn(a.y, __bfloat162float(__float2bfloat16_rn(b.y)));
    a.z = __fadd_rn(a.z, __bfloat162float(__float2bfloat16_rn(b.z)));
    a.w = __fadd_rn(a.w, __bfloat162float(__float2bfloat16_rn(b.w)));
    dw[i] = a;
    mb[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}
__global__ void k_finish_microbatch_tail(float* dw, float* mb, int64_t from, int64_t n) {
  for (int64_t i = from + threadIdx.x; i < n; i += blockDim.x) {
    dw[i] = __fadd_rn(dw[i], __bfloat162float(__float2bfloat16_rn(mb[i])));
    mb[i] = 0.f;
  }
}

template <class T>
__global__ void k_zero_cols(T* x, int64_t rows, int D, int from) {
  const int w = D - from;
  const int64_t n = rows * w;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    x[(i / w) * D + from + i % w] = T{};
}

// out = mult / sum(w): thread-strided sequential sums, then a fixed binary tree (one CTA).
__global__ void __launch_bounds__(1024) k_sum_scale(const float* w, int64_t n, float mult, float* out) {
  __shared__ float red[1024];
  float s = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += 1024) s = __fadd_rn(s, w[i]);
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = 512; o > 0; o >>= 1) {
    if (int(threadIdx.x) < o) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = __fdiv_rn(mult, red[0]);
}

int grid_for(int64_t n, int threads) {
  return int(std::min<int64_t>((n + threads - 1) / threads, int64_t(sm_count()) * 16));
}

}  // namespace

void launch_dx_combine(const void* tmp, const void* dx_old, void* dx, int64_t n, DType io, bool accumulate,
                       cudaStream_t st) {
  if (n <= 0) return;
  if (io == DType::bf16)
    k_dx_combine_bf16<<<grid_for(n, 256), 256, 0, st>>>(static_cast<const bf16*>(tmp), static_cast<const bf16*>(dx_old),
                                                         static_cast<bf16*>(dx), n, accumulate);
  else
    k_dx_combine_f32<<<grid_for(n, 256), 256, 0, st>>>(static_cast<const float*>(tmp),
                                                        static_cast<const float*>(dx_old), static_cast<float*>(dx), n,
                                                        accumulate);
  NM_CUDA(cudaGetLastError());
}

void launch_finish_microbatch(float* dw, float* dw_mb, int64_t n, cudaStream_t st) {
  if (n <= 0) return;
  const bool aligned = (reinterpret_cast<uintptr_t>(dw) % 16 == 0) && (reinterpret_cast<uintptr_t>(dw_mb) % 16 == 0);
  const int64_t n4 = aligned ? n / 4 : 0;
  if (n4 > 0)
    k_finish_microbatch<<<grid_for(n4, 256), 256, 0, st>>>(reinterpret_cast<float4*>(dw), reinterpret_cast<float4*>(dw_mb),
                                                           n4);
  if (n4 * 4 < n) k_finish_microbatch_tail<<<1, 256, 0, st>>>(dw, dw_mb, n4 * 4, n);
  NM_CUDA(cudaGetLastError());
}

void launch_zero_cols(void* x, int64_t rows, int D, int from, DType dt, cudaStream_t st) {
  if (rows <= 0 || from >= D) return;
  const int64_t n = rows * (D - from);
  if (dt == DType::bf16) k_zero_cols<bf16><<<grid_for(n, 256), 256, 0, st>>>(static_cast<bf16*>(x), rows, D, from);
  else k_zero_cols<float><<<grid_for(n, 256), 256, 0, st>>>(static_cast<float*>(x), rows, D, from);
  NM_CUDA(cudaGetLastError());
}

void launch_sum_scale(const float* w, int64_t n, float mult, float* out, cudaStream_t st) {
  k_sum_scale<<<1, 1024, 0, st>>>(w, n, mult, out);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::ce
