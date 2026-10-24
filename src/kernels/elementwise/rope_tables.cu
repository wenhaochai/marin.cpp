// RoPE tables (DESIGN.md §3.4, numerics.md §3.3, T1.3). Levanter's expression, all f32:
//   k = int32 [0, 2, .., hd-2];  e = f32(k) / f32(hd);  p = f32(theta) ** e;  inv_freq = 1 / p;  (/ factor 1.0)
//   angle[pos, i] = f32(pos) * inv_freq[i mod hd/2];  cos, sin of the f32 angle, cast to bf16.
// HostDouble (DESIGN.md): pow, cos and sin evaluated in double and rounded to f32 (the correctly rounded f32 values in
// all but rare ties), divisions and the angle product in f32 on the host (-ffp-contract=off).
// DeviceLibdevice: the same expression on the GPU with libdevice powf, IEEE division, cosf and sinf (no fast math),
// which are the functions XLA:GPU emits for these HLO ops.
#include <cmath>
#include <vector>

#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_common.cuh"

namespace nm::ew {
namespace {

__global__ void k_rope_inv(int hd, float theta, float* inv) {
  const int i = threadIdx.x;
  if (i < hd / 2) {
    const float e = __fdiv_rn(static_cast<float>(2 * i), static_cast<float>(hd));
    inv[i] = __fdiv_rn(1.f, powf(theta, e));
  }
}
__global__ void k_rope_tab(int S, int hd, const float* inv, uint16_t* c, uint16_t* s) {
  const int64_t n = int64_t(S) * hd;
  for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; idx < n; idx += int64_t(gridDim.x) * blockDim.x) {
    const int p = static_cast<int>(idx / hd), i = static_cast<int>(idx % hd);
    const float a = __fmul_rn(static_cast<float>(p), inv[i % (hd / 2)]);
    c[idx] = static_cast<uint16_t>(f2bf(cosf(a)));
    s[idx] = static_cast<uint16_t>(f2bf(sinf(a)));
  }
}

}  // namespace

void build_rope_tables_method(RopeMethod m, int S_max, int hd, double theta, bf16_t* cos_host, bf16_t* sin_host,
                              float* inv_host) {
  NM_REQUIRE(S_max > 0 && hd > 0 && hd % 2 == 0, "build_rope_tables: S_max > 0 and even hd > 0 required");
  NM_REQUIRE(theta > 0, "build_rope_tables: theta must be > 0");
  const int half = hd / 2;
  const float theta_f = static_cast<float>(theta);
  std::vector<float> inv(half);
  if (m == RopeMethod::HostDouble) {
    for (int i = 0; i < half; ++i) {
      const float e = static_cast<float>(2 * i) / static_cast<float>(hd);
      const float p = static_cast<float>(std::pow(static_cast<double>(theta_f), static_cast<double>(e)));
      inv[i] = 1.0f / p;
    }
    for (int p = 0; p < S_max; ++p)
      for (int i = 0; i < hd; ++i) {
        const float a = static_cast<float>(p) * inv[i % half];
        const float cf = static_cast<float>(std::cos(static_cast<double>(a)));
        const float sf = static_cast<float>(std::sin(static_cast<double>(a)));
        if (cos_host) cos_host[int64_t(p) * hd + i].bits = f32_to_bf16_bits(cf);
        if (sin_host) sin_host[int64_t(p) * hd + i].bits = f32_to_bf16_bits(sf);
      }
  } else {
    float* d_inv = nullptr;
    uint16_t* d_tab = nullptr;
    const int64_t n = int64_t(S_max) * hd;
    NM_CUDA(cudaMalloc(&d_inv, half * sizeof(float)));
    NM_CUDA(cudaMalloc(&d_tab, 2 * n * sizeof(uint16_t)));
    k_rope_inv<<<1, 1024>>>(hd, theta_f, d_inv);
    NM_CUDA(cudaGetLastError());
    k_rope_tab<<<256, 256>>>(S_max, hd, d_inv, d_tab, d_tab + n);
    NM_CUDA(cudaGetLastError());
    NM_CUDA(cudaMemcpy(inv.data(), d_inv, half * sizeof(float), cudaMemcpyDeviceToHost));
    if (cos_host) NM_CUDA(cudaMemcpy(cos_host, d_tab, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (sin_host) NM_CUDA(cudaMemcpy(sin_host, d_tab + n, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    NM_CUDA(cudaFree(d_inv));
    NM_CUDA(cudaFree(d_tab));
  }
  if (inv_host)
    for (int i = 0; i < half; ++i) inv_host[i] = inv[i];
}

}  // namespace nm::ew

namespace nm {
void build_rope_tables(int S_max, int hd, double theta, bf16_t* cos_host, bf16_t* sin_host, float* inv_freq_host) {
  ew::build_rope_tables_method(ew::RopeMethod::HostDouble, S_max, hd, theta, cos_host, sin_host, inv_freq_host);
}
}  // namespace nm
