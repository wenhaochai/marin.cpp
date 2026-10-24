// Threefry-2x32 with 20 rounds, as JAX runs it (data.md §4.1): one inline body for host and device code, so the
// host data order, the host init and the device in-graph RNG share the same bits (DESIGN.md §9.3 RNG gate).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define NM_TF_HD __host__ __device__ __forceinline__
#else
#define NM_TF_HD inline
#endif

namespace nm::tf {

NM_TF_HD uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

// ks = (k0, k1, k0 ^ k1 ^ 0x1BD11BDA); x0 += ks0; x1 += ks1; five groups of four rounds (rotations 13,15,26,6 for even
// groups, 17,29,16,24 for odd ones), each round x0 += x1; x1 = rotl(x1, r); x1 ^= x0; after group i:
// x0 += ks[(i+1)%3]; x1 += ks[(i+2)%3] + (i+1). All uint32 arithmetic wraps (jax/_src/prng.py threefry2x32).
NM_TF_HD void threefry2x32_20(uint32_t k0, uint32_t k1, uint32_t& x0, uint32_t& x1) {
  const uint32_t ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  uint32_t a = x0 + ks[0], b = x1 + ks[1];
#define NM_TF_ROUND(r) \
  a += b;              \
  b = rotl32(b, r);    \
  b ^= a;
  NM_TF_ROUND(13) NM_TF_ROUND(15) NM_TF_ROUND(26) NM_TF_ROUND(6)
  a += ks[1]; b += ks[2] + 1u;
  NM_TF_ROUND(17) NM_TF_ROUND(29) NM_TF_ROUND(16) NM_TF_ROUND(24)
  a += ks[2]; b += ks[0] + 2u;
  NM_TF_ROUND(13) NM_TF_ROUND(15) NM_TF_ROUND(26) NM_TF_ROUND(6)
  a += ks[0]; b += ks[1] + 3u;
  NM_TF_ROUND(17) NM_TF_ROUND(29) NM_TF_ROUND(16) NM_TF_ROUND(24)
  a += ks[1]; b += ks[2] + 4u;
  NM_TF_ROUND(13) NM_TF_ROUND(15) NM_TF_ROUND(26) NM_TF_ROUND(6)
  a += ks[2]; b += ks[0] + 5u;
#undef NM_TF_ROUND
  x0 = a;
  x1 = b;
}

}  // namespace nm::tf
