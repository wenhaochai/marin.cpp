// S1: threefry port, keys, permutation, LevanterOrder; T0.1 (order table sha256).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "marincpp/capi.h"
#include "s1_util.hpp"

using namespace nm;

NM_TEST(s1_threefry, known_answers) {
  // Random123 known-answer vectors of threefry2x32_20 (kat_vectors): key, counter -> output
  struct V { uint32_t k0, k1, x0, x1, y0, y1; };
  const V v[] = {{0u, 0u, 0u, 0u, 0x6b200159u, 0x99ba4efeu},
                 {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0x1cb996fcu, 0xbb002be7u},
                 {0x13198a2eu, 0x03707344u, 0x243f6a88u, 0x85a308d3u, 0xc4923a9cu, 0x483df7a0u}};
  for (const V& t : v) {
    uint32_t a = t.x0, b = t.x1;
    threefry2x32(Key{t.k0, t.k1}, a, b);
    CHECK_EQ(a, t.y0);
    CHECK_EQ(b, t.y1);
  }
}

NM_TEST(s1_threefry, levanter_keys) {
  // data.md §4.1 [verified]: data_key = PRNGKey(42); mix, shuffle = split(data_key); perm = split(shuffle)[1]
  const Key dk = prng_key(42);
  CHECK_EQ(dk.k0, 0u);
  CHECK_EQ(dk.k1, 42u);
  const Key mix = split(dk, 0), shuf = split(dk, 1), perm = split(shuf, 1);
  CHECK_EQ(mix.k0, 1832780943u);
  CHECK_EQ(mix.k1, 270669613u);
  CHECK_EQ(shuf.k0, 64467757u);
  CHECK_EQ(shuf.k1, 2916123636u);
  CHECK_EQ(perm.k0, 2350016172u);
  CHECK_EQ(perm.k1, 1168365246u);
  // randint(perm_key, (), 0, 2^30) = bits32(split(perm_key)[1], 1)[0] mod 2^30 = the PCG64 seed 937906096
  CHECK_EQ(bits32(split(perm, 1), 0) % (1u << 30), 937906096u);
}

NM_TEST(s1_threefry, block_permutation_and_first_values) {
  const uint32_t head[16] = {416, 814, 925, 86, 1936, 560, 614, 1951, 1832, 907, 1744, 1707, 1455, 78, 844, 762};
  const auto pi = data::block_permutation(split(prng_key(42), 0), 0, 2048);
  for (int i = 0; i < 16; ++i) CHECK_EQ(pi[i], head[i]);
  std::vector<uint32_t> sorted(pi);
  std::sort(sorted.begin(), sorted.end());
  for (uint32_t i = 0; i < 2048; ++i) CHECK_EQ(sorted[i], i);
  LevanterOrder o(42, 2441406, 2048);
  const uint64_t s0[8] = {1909879, 335277, 2165778, 1350661, 1748337, 1843177, 292285, 1995702};   // data.md §4.4
  for (int j = 0; j < 8; ++j) CHECK_EQ(o(j), s0[j]);
  // n <= 1: no rounds; tiny n: still a permutation
  uint32_t one[1] = {7};
  permutation(prng_key(1), 1, one);
  CHECK_EQ(one[0], 0u);
  CHECK_THROWS(LevanterOrder(43, 2441406, 2048), "no LCG constants for data_seed 43");
}

NM_TEST(s1_threefry, device_equals_host) {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    cudaGetLastError();
    SKIP("no CUDA device");
  }
  const int64_t n = 1 << 20;
  std::vector<uint32_t> x0(n), x1(n), h0, h1;
  for (int64_t i = 0; i < n; ++i) {
    x0[i] = static_cast<uint32_t>(i * 2654435761u);
    x1[i] = static_cast<uint32_t>(i ^ 0xdeadbeefu);
  }
  h0 = x0;
  h1 = x1;
  CHECK_EQ(nm_threefry2x32(0x12345678u, 0x9abcdef0u, h0.data(), h1.data(), n), NM_OK);
  uint32_t *d0 = nullptr, *d1 = nullptr;
  NM_CUDA(cudaMalloc(&d0, n * 4));
  NM_CUDA(cudaMalloc(&d1, n * 4));
  NM_CUDA(cudaMemcpy(d0, x0.data(), n * 4, cudaMemcpyHostToDevice));
  NM_CUDA(cudaMemcpy(d1, x1.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK_EQ(nm_threefry2x32_device(0x12345678u, 0x9abcdef0u, d0, d1, n, nullptr), NM_OK);
  NM_CUDA(cudaMemcpy(x0.data(), d0, n * 4, cudaMemcpyDeviceToHost));
  NM_CUDA(cudaMemcpy(x1.data(), d1, n * 4, cudaMemcpyDeviceToHost));
  cudaFree(d0);
  cudaFree(d1);
  CHECK(x0 == h0);
  CHECK(x1 == h1);
}

// T0.1: s(j), j = 0..5,859,327, little-endian uint32 -> sha256 dde0ca7b...655b (data.md §1), and element-wise against
// the levanter-verified table when cache/data is present.
NM_TEST(t0_1_order, table_sha256) {
  LevanterOrder o(42, 2441406, 2048);
  CHECK_EQ(o.table_sha256(5859328), std::string("dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b"));
}

NM_TEST(t0_1_order, table_vs_levanter_file_and_c_abi) {
  const std::string p = s1::repo_file_or_skip("cache/data/fineweb_order_seed42.u32");
  const auto ref = s1::read_vec<uint32_t>(p);
  CHECK_EQ(ref.size(), size_t(5859328));
  std::vector<uint32_t> got(ref.size());
  CHECK_EQ(nm_levanter_order(42, 2441406, 2048, 0, got.size(), got.data()), NM_OK);
  size_t bad = 0;
  for (size_t j = 0; j < ref.size(); ++j) bad += got[j] != ref[j];
  CHECK_EQ(bad, size_t(0));
  // the thread-safe form used by DataPlan agrees on a stride through every pass
  for (uint64_t j = 0; j < ref.size(); j += 997) CHECK_EQ(data::levanter_seq(42, 2441406, 2048, 2441406, j), uint64_t(ref[j]));
  // first wrapped mixture index (data.md §4.5): block 1192 holds the first r >= N
  CHECK_EQ(data::first_index_of_pass(1, 2441406, 2048), uint64_t(2441216));
  CHECK_EQ(data::first_index_of_pass(2, 2441406, 2048), uint64_t(4882432));
  CHECK_EQ(nm_levanter_order(43, 2441406, 2048, 0, 1, got.data()), NM_ERR_INVALID);
  CHECK(std::string(nm_last_error()).find("no LCG constants") != std::string::npos);
}

NM_TEST(t0_1_order, converted_order_file) {
  const std::string d = s1::converted_or_skip(s1::kFineweb);
  const DatasetMeta m = load_dataset_meta(s1::kFineweb);
  CHECK_EQ(sha256_file_hex(d + "/order_seed42.u32"),
           std::string("dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b"));
  CHECK_EQ(m.n_seqs, uint64_t(2441406));
  const json::Value meta = json::parse_file(d + "/meta.json");
  CHECK_EQ(meta.at("lcg").at("42").at("a").as_u64(), uint64_t(830293));
  CHECK_EQ(meta.at("lcg").at("42").at("b").as_u64(), uint64_t(746237));
}
