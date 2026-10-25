// Portable GEMM helper kernels (sm_80 and sm_90a).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "kernels/gemm/gemm_internal.hpp"

namespace nm::gemm {

namespace {

// D = C + f32(t): t already holds bf16(acc) (cuBLASLt rounded the f32 accumulator once, RNE).
__global__ void k_bf16_round_add(const float* __restrict__ C, int64_t ldc, const __nv_bfloat16* __restrict__ t,
                                 int64_t ldt, float* __restrict__ D, int64_t ldd, int64_t rows, int64_t cols) {
  const int64_t n = rows * cols;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / cols, c = i - r * cols;
    D[r * ldd + c] = C[r * ldc + c] + __bfloat162float(t[r * ldt + c]);
  }
}

__device__ __forceinline__ uint32_t mix(uint32_t x) {   // lowbias32
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

template <class T>
__global__ void k_fill(T* p, int64_t n, uint32_t seed) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const float u = float(mix(uint32_t(i) * 2654435761U ^ mix(seed + uint32_t(i >> 32)))) * (2.f / 4294967296.f) - 1.f;
    if constexpr (sizeof(T) == 2) p[i] = __float2bfloat16_rn(u); else p[i] = u;
  }
}

// 32 x 32 tiles through shared memory (padded against bank conflicts), 32 x 8 threads.
__global__ void k_transpose_bf16(const __nv_bfloat16* __restrict__ in, int64_t ld_in, __nv_bfloat16* __restrict__ out,
                                 int64_t ld_out, int64_t rows, int64_t cols) {
  __shared__ __nv_bfloat16 t[32][33];
  const int64_t r0 = int64_t(blockIdx.y) * 32, c0 = int64_t(blockIdx.x) * 32;
  for (int i = threadIdx.y; i < 32; i += 8) {
    const int64_t r = r0 + i, c = c0 + threadIdx.x;
    if (r < rows && c < cols) t[i][threadIdx.x] = in[r * ld_in + c];
  }
  __syncthreads();
  for (int i = threadIdx.y; i < 32; i += 8) {
    const int64_t c = c0 + i, r = r0 + threadIdx.x;
    if (r < rows && c < cols) out[c * ld_out + r] = t[threadIdx.x][i];
  }
}

}  // namespace

void transpose_bf16(const void* in, int64_t ld_in, void* out, int64_t ld_out, int64_t rows, int64_t cols,
                    cudaStream_t s) {
  dim3 grid(unsigned((cols + 31) / 32), unsigned((rows + 31) / 32));
  k_transpose_bf16<<<grid, dim3(32, 8), 0, s>>>(static_cast<const __nv_bfloat16*>(in), ld_in,
                                                static_cast<__nv_bfloat16*>(out), ld_out, rows, cols);
  NM_CUDA(cudaGetLastError());
}

void bf16_round_add(const float* C, int64_t ldc, const void* t, int64_t ldt, float* D, int64_t ldd, int64_t rows,
                    int64_t cols, cudaStream_t s) {
  const int64_t n = rows * cols;
  const int blocks = int(std::min<int64_t>((n + 255) / 256, 4096));
  k_bf16_round_add<<<blocks, 256, 0, s>>>(C, ldc, static_cast<const __nv_bfloat16*>(t), ldt, D, ldd, rows, cols);
  NM_CUDA(cudaGetLastError());
}

void fill_uniform(void* p, DType t, int64_t n, uint32_t seed, cudaStream_t s) {
  if (n <= 0) return;
  const int blocks = int(std::min<int64_t>((n + 255) / 256, 8192));
  if (t == DType::bf16) k_fill<<<blocks, 256, 0, s>>>(static_cast<__nv_bfloat16*>(p), n, seed);
  else if (t == DType::f32) k_fill<<<blocks, 256, 0, s>>>(static_cast<float*>(p), n, seed);
  else fatal(NM_WHERE, "fill_uniform: bf16 or f32");
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::gemm
