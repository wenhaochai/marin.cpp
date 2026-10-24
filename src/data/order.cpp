// JAX threefry port (partitionable semantics) and the exact levanter data order. data.md §4, DESIGN.md §5.3.
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>

#include "data/internal.hpp"
#include "data/threefry_impl.hpp"
#include "marincpp/data.hpp"

namespace nm {

void threefry2x32(Key k, uint32_t& x0, uint32_t& x1) { tf::threefry2x32_20(k.k0, k.k1, x0, x1); }

Key prng_key(uint32_t seed) { return Key{0u, seed}; }   // PRNGKey(seed) for 0 <= seed < 2^32: (seed >> 32, seed)

Key split(Key k, uint32_t i) {   // split(k)[i] = T(k, (0, i)) under jax_threefry_partitionable
  uint32_t x0 = 0, x1 = i;
  threefry2x32(k, x0, x1);
  return Key{x0, x1};
}

Key fold_in(Key k, uint32_t d) {   // T(k, threefry_seed(d)) = T(k, (0, d))
  uint32_t x0 = 0, x1 = d;
  threefry2x32(k, x0, x1);
  return Key{x0, x1};
}

uint32_t bits32(Key k, uint64_t i) {   // counters (i >> 32, i & 0xffffffff); 32-bit draw = x0 ^ x1
  uint32_t x0 = static_cast<uint32_t>(i >> 32), x1 = static_cast<uint32_t>(i);
  threefry2x32(k, x0, x1);
  return x0 ^ x1;
}

// jax.random.permutation(key, arange(n)) for 1-D input (jax/_src/random.py _shuffle): rounds =
// ceil(3 ln(max(1, n)) / ln(2^32 - 1)) in float64; each round key, sub = split(key); keys = bits32(sub, n); stable sort
// of the current array by keys (lax.sort_key_val, is_stable). Key i belongs to the element now at position i.
void permutation(Key k, uint32_t n, uint32_t* out) {
  for (uint32_t i = 0; i < n; ++i) out[i] = i;
  if (n <= 1) return;
  const int rounds = static_cast<int>(std::ceil(3.0 * std::log(static_cast<double>(n)) / std::log(4294967295.0)));
  std::vector<std::pair<uint32_t, uint32_t>> kv(n);   // (sort key, element)
  for (int r = 0; r < rounds; ++r) {
    const Key sub = split(k, 1);
    k = split(k, 0);
    for (uint32_t i = 0; i < n; ++i) kv[i] = {bits32(sub, i), out[i]};
    std::stable_sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (uint32_t i = 0; i < n; ++i) out[i] = kv[i].second;
  }
}

namespace data {
namespace {

std::mutex g_lcg_mu;
std::map<std::pair<uint32_t, uint64_t>, Lcg>& lcg_table() {
  static std::map<std::pair<uint32_t, uint64_t>, Lcg> t = {
      {{42u, 2441406ull}, Lcg{830293ull, 746237ull}},   // data.md §4.2 [verified]; converter meta.json repeats it
  };
  return t;
}

struct BlockCache {   // per-thread cache of block permutations, keyed by (mix key, block size, block index)
  struct E { uint32_t k0 = 0, k1 = 0; int block = 0; uint64_t blk = ~0ull; std::vector<uint32_t> pi; };
  E e[8];
  unsigned next = 0;
  const std::vector<uint32_t>& get(Key k, int block, uint64_t blk) {
    for (E& x : e)
      if (x.blk == blk && x.block == block && x.k0 == k.k0 && x.k1 == k.k1) return x.pi;
    E& x = e[next++ % 8];
    x.k0 = k.k0;
    x.k1 = k.k1;
    x.block = block;
    x.blk = blk;
    x.pi = block_permutation(k, blk, block);
    return x.pi;
  }
};

uint64_t sigma(const Lcg& l, uint64_t n, uint64_t r) {   // exact: a < N, r < N < 2^32
  return static_cast<uint64_t>((static_cast<unsigned __int128>(l.a) * r + l.b) % n);
}

}  // namespace

void register_lcg(uint32_t seed, uint64_t n_seqs, uint64_t a, uint64_t b) {
  NM_REQUIRE(n_seqs > 0 && a > 0 && a < n_seqs && b < n_seqs && std::gcd(a, n_seqs) == 1,
             "LCG constants (seed " + std::to_string(seed) + ", N " + std::to_string(n_seqs) + ")");
  std::lock_guard<std::mutex> g(g_lcg_mu);
  auto [it, inserted] = lcg_table().insert({{seed, n_seqs}, Lcg{a, b}});
  if (!inserted && (it->second.a != a || it->second.b != b))
    fatal(NM_WHERE, "conflicting LCG constants for data_seed " + std::to_string(seed) + ", N " + std::to_string(n_seqs));
}

Lcg lcg_for(uint32_t seed, uint64_t n_seqs) {
  std::lock_guard<std::mutex> g(g_lcg_mu);
  auto it = lcg_table().find({seed, n_seqs});
  if (it == lcg_table().end())
    fatal(NM_WHERE, "no LCG constants for data_seed " + std::to_string(seed) + " and N = " + std::to_string(n_seqs) +
                        " sequences: run tools/convert/levanter_to_flat.py with --seeds including it (levanter draws "
                        "a, b from numpy PCG64; they are copied, never recomputed)");
  return it->second;
}

std::vector<uint32_t> block_permutation(Key mix_key, uint64_t blk, int block) {
  NM_REQUIRE(blk <= 0xffffffffull, "block index exceeds 2^32");
  std::vector<uint32_t> pi(static_cast<size_t>(block));
  permutation(fold_in(mix_key, static_cast<uint32_t>(blk)), static_cast<uint32_t>(block), pi.data());
  return pi;
}

uint64_t levanter_seq(uint32_t seed, uint64_t n_seqs, int block, uint64_t wrap, uint64_t j) {
  NM_REQUIRE(wrap > 0 && wrap <= n_seqs, "wrap must be in [1, N]");
  thread_local BlockCache cache;
  thread_local uint32_t c_seed = ~0u;
  thread_local uint64_t c_n = 0;
  thread_local Lcg c_lcg;
  thread_local Key c_mix{};
  if (seed != c_seed || n_seqs != c_n) {
    c_lcg = lcg_for(seed, n_seqs);
    c_mix = split(prng_key(seed), 0);
    c_seed = seed;
    c_n = n_seqs;
  }
  const uint64_t blk = j / static_cast<uint64_t>(block);
  const auto& pi = cache.get(c_mix, block, blk);
  const uint64_t r = (blk * static_cast<uint64_t>(block) + pi[j % static_cast<uint64_t>(block)]) % wrap;
  return sigma(c_lcg, n_seqs, r);
}

uint64_t first_index_of_pass(uint64_t q, uint64_t wrap, int block) {
  return (q * wrap / static_cast<uint64_t>(block)) * static_cast<uint64_t>(block);
}

}  // namespace data

// ------------------------------------------------------------------------------------------------ LevanterOrder
LevanterOrder::LevanterOrder(uint32_t data_seed, uint64_t n_seqs, int mix_block)
    : N_(n_seqs), block_(mix_block) {
  NM_REQUIRE(n_seqs > 0 && n_seqs < (1ull << 32), "N must be in [1, 2^32)");
  NM_REQUIRE(mix_block > 0, "mix_block must be positive");
  const data::Lcg l = data::lcg_for(data_seed, n_seqs);
  a_ = l.a;
  b_ = l.b;
  mix_key_ = split(prng_key(data_seed), 0);   // mix_key, shuffle_key = split(PRNGKey(data_seed))
}

uint64_t LevanterOrder::operator()(uint64_t j) const {
  const uint64_t blk = j / static_cast<uint64_t>(block_);
  if (static_cast<int64_t>(blk) != cache_blk_) {
    cache_pi_ = data::block_permutation(mix_key_, blk, block_);
    cache_blk_ = static_cast<int64_t>(blk);
  }
  const uint64_t r = (blk * static_cast<uint64_t>(block_) + cache_pi_[j % static_cast<uint64_t>(block_)]) % N_;
  return static_cast<uint64_t>((static_cast<unsigned __int128>(a_) * r + b_) % N_);
}

std::string LevanterOrder::table_sha256(uint64_t count) const {
  std::vector<uint8_t> buf(count * 4);
  for (uint64_t j = 0; j < count; ++j) {
    const uint32_t s = static_cast<uint32_t>((*this)(j));
    buf[4 * j] = static_cast<uint8_t>(s);
    buf[4 * j + 1] = static_cast<uint8_t>(s >> 8);
    buf[4 * j + 2] = static_cast<uint8_t>(s >> 16);
    buf[4 * j + 3] = static_cast<uint8_t>(s >> 24);
  }
  return sha256_hex(buf.data(), buf.size());
}

}  // namespace nm
