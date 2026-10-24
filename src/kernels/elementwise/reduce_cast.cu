// Fixed-order reductions (reduce.hpp: reduce_partials_rows, sum_fixed), the multi-tensor f32 -> bf16 cast, and the
// launch helpers of the elementwise module. DESIGN.md §11: no floating-point atomics; every order depends only on the
// shape.
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_common.cuh"

namespace nm::ew {

// ------------------------------------------------------------------------------------------------ launch helpers
namespace {
int resident_ctas(const void* kernel, int block, size_t smem) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int, size_t, int>, int> cache;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  const auto key = std::make_tuple(kernel, block, smem, dev);
  {
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }
  int per_sm = 0, sms = 0;
  NM_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block, smem));
  NM_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int cap = per_sm > 0 ? per_sm * sms : sms;
  std::lock_guard<std::mutex> g(mu);
  cache[key] = cap;
  return cap;
}
}  // namespace

int64_t norm_parts(int64_t T) {
  if (T <= 0) return 1;
  static std::mutex mu;
  static std::map<int, int> sms_of;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  int sms = 0;
  {
    std::lock_guard<std::mutex> g(mu);
    auto it = sms_of.find(dev);
    if (it != sms_of.end()) sms = it->second;
  }
  if (sms == 0) {
    NM_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    std::lock_guard<std::mutex> g(mu);
    sms_of[dev] = sms;
  }
  int64_t p = (T + kMinRowsPerPart - 1) / kMinRowsPerPart;
  if (p > kMaxParts) p = kMaxParts;
  if (p > int64_t(kNormCtasPerSm) * sms) p = int64_t(kNormCtasPerSm) * sms;
  return p;
}

int grid_for_items(int64_t n, int64_t per_cta, const void* kernel, int block, size_t smem) {
  if (n <= 0) return 1;
  const int64_t want = (n + per_cta - 1) / per_cta;
  const int64_t cap = resident_ctas(kernel, block, smem);
  return static_cast<int>(want < cap ? want : cap);
}

namespace {

// ------------------------------------------------------------------------------------------------ k_reduce_rows
// CTA = 8 row groups x 32 columns. Row group q sums partial rows [q * nb / 8, (q + 1) * nb / 8) in ascending order;
// the 8 group sums are then added in ascending q. The order depends only on (n_blocks, width).
__global__ void __launch_bounds__(256) k_reduce_rows(const float* __restrict__ partial, int nb, int width,
                                                     float* __restrict__ out, bool accumulate, bool round_bf16) {
  __shared__ float sm[8][33];
  const int cx = threadIdx.x & 31, q = threadIdx.x >> 5;
  const int col = blockIdx.x * 32 + cx;
  const int b0 = static_cast<int>(int64_t(q) * nb / 8), b1 = static_cast<int>(int64_t(q + 1) * nb / 8);
  float s = 0.f;
  if (col < width)
    for (int b = b0; b < b1; ++b) s += partial[int64_t(b) * width + col];
  sm[q][cx] = s;
  __syncthreads();
  if (q == 0 && col < width) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += sm[i][cx];
    if (round_bf16) t = rbf(t);
    out[col] = accumulate ? out[col] + t : t;
  }
}

// ------------------------------------------------------------------------------------------------ k_sum_fixed
__global__ void __launch_bounds__(1024) k_sum_fixed(const float* __restrict__ x, int64_t n, float* __restrict__ out) {
  __shared__ float sm[1024];
  float s = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += 1024) s += x[i];
  sm[threadIdx.x] = s;
  __syncthreads();
  for (int w = 512; w > 0; w >>= 1) {
    if (threadIdx.x < w) sm[threadIdx.x] += sm[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = sm[0];
}

// ------------------------------------------------------------------------------------------------ k_cast_multi
// blockIdx.y = tensor; device arrays src[n], dst[n], numel[n]. Round to nearest even.
__global__ void __launch_bounds__(256) k_cast_multi(const float* const* __restrict__ src, bf16_t* const* __restrict__ dst,
                                                    const int64_t* __restrict__ numel) {
  const int k = blockIdx.y;
  const float* s = src[k];
  uint16_t* d = reinterpret_cast<uint16_t*>(dst[k]);
  const int64_t n = numel[k];
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    d[i] = static_cast<uint16_t>(f2bf(s[i]));
}

}  // namespace
}  // namespace nm::ew

namespace nm {

void reduce_partials_rows(const float* partial, int n_blocks, int width, float* out, bool accumulate, bool round_bf16,
                          cudaStream_t s) {
  NM_REQUIRE(partial && out && n_blocks > 0 && width > 0, "reduce_partials_rows: bad arguments");
  ew::k_reduce_rows<<<(width + 31) / 32, 256, 0, s>>>(partial, n_blocks, width, out, accumulate, round_bf16);
  NM_CUDA(cudaGetLastError());
}

void sum_fixed(const float* x, int64_t n, float* out, cudaStream_t s) {
  NM_REQUIRE(out && n >= 0 && (x || n == 0), "sum_fixed: bad arguments");
  ew::k_sum_fixed<<<1, 1024, 0, s>>>(x, n, out);
  NM_CUDA(cudaGetLastError());
}

// CastList arrays (src, dst, numel) are device arrays of l.n entries (planned buffers, so the launch is graph-safe).
void cast_f32_bf16_multi(const CastList& l, cudaStream_t s) {
  NM_REQUIRE(l.n >= 0 && l.n <= 65535, "cast_f32_bf16_multi: 0 <= n <= 65535 tensors");
  if (l.n == 0) return;
  NM_REQUIRE(l.src && l.dst && l.numel, "cast_f32_bf16_multi: missing list");
  ew::k_cast_multi<<<dim3(256, l.n), 256, 0, s>>>(l.src, l.dst, l.numel);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm
