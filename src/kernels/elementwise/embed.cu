// Embedding forward (gather) and deterministic backward (sorted segment sum). DESIGN.md §3.2, §11.
//
// Backward:
//   1. keys = token ids, values = positions; cub::DeviceRadixSort::SortPairs over ceil(log2 V) key bits. The sort is
//      stable, so inside every run of equal tokens the positions ascend.
//   2. EmbedAccum::F32: the sorted entries form fixed chunks of kEmbedChunk entries. One warp per (chunk, 256-column
//      slice) sums each run piece of its chunk in ascending position order in f32. A run that lies inside one chunk is
//      added to dtable[token] at once; the piece of a run that continues from the previous chunk goes to the chunk's
//      head slot, the piece that continues into the next chunk to its tail slot. A merge pass, owned by the chunk where
//      a multi-chunk run starts, adds tail[k] + head[k+1] + ... in chunk order to dtable. The summation order depends
//      only on the token sequence: no atomics, bitwise repeatable, and no warp walks a long run alone.
//   3. EmbedAccum::Bf16Sequential (strict preset): levanter's precision. Each run (per 64-column slice) is summed by one
//      warp, the warp of the chunk where the run starts: a running sum rounded to the storage dtype after every add, in
//      ascending position order, then added to dtable[token] in f32. The dout rows are loaded 32 ahead of the adds.
// dtable accumulates across microbatches (+=). Rows of tokens absent from the batch are not touched.
#include <cub/device/device_radix_sort.cuh>

#include <string>

#include "kernels/elementwise/ew_api.hpp"
#include "kernels/elementwise/ew_common.cuh"

namespace nm::ew {
namespace {

constexpr uint32_t kNoKey = 0xffffffffu;
constexpr int kSlice = 256;   // columns per warp (32 lanes x 8)

uint64_t align256(uint64_t b) { return (b + 255) & ~uint64_t(255); }
int key_bits(int V) {
  int b = 1;
  while ((int64_t(1) << b) < V) ++b;
  return b;
}

// ------------------------------------------------------------------------------------------------ forward
template <int VB>   // bytes per 8-element vector: 16 (bf16) or 32 (f32)
__global__ void __launch_bounds__(256) k_embed_fwd(const uint8_t* __restrict__ table, const int32_t* __restrict__ tok,
                                                   uint8_t* __restrict__ out, int64_t nvec, int D8) {
  for (int64_t v = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; v < nvec; v += int64_t(gridDim.x) * blockDim.x) {
    const int64_t t = v / D8, c = v - t * D8;
    const int64_t row = __ldg(tok + t);
    const uint4* src = reinterpret_cast<const uint4*>(table + (row * D8 + c) * VB);
    uint4* dst = reinterpret_cast<uint4*>(out + (t * D8 + c) * VB);
    dst[0] = __ldg(src);
    if (VB == 32) dst[1] = __ldg(src + 1);
  }
}

// ------------------------------------------------------------------------------------------------ backward
__global__ void k_embed_prep(const int32_t* __restrict__ tok, uint32_t* __restrict__ keys, int32_t* __restrict__ vals,
                             int64_t T) {
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < T; t += int64_t(gridDim.x) * blockDim.x) {
    keys[t] = static_cast<uint32_t>(tok[t]);
    vals[t] = static_cast<int32_t>(t);
  }
}

__device__ __forceinline__ void add_row_f32(float* p, const float (&a)[8]) {
  float4* q = reinterpret_cast<float4*>(p);
  float4 x = q[0], y = q[1];
  x.x += a[0]; x.y += a[1]; x.z += a[2]; x.w += a[3];
  y.x += a[4]; y.y += a[5]; y.z += a[6]; y.w += a[7];
  q[0] = x;
  q[1] = y;
}
__device__ __forceinline__ void put_row_f32(float* p, const float (&a)[8]) {
  float4* q = reinterpret_cast<float4*>(p);
  q[0] = make_float4(a[0], a[1], a[2], a[3]);
  q[1] = make_float4(a[4], a[5], a[6], a[7]);
}

// F32 path, pass 1: one warp per (chunk, slice).
template <class S>
__global__ void __launch_bounds__(256) k_embed_chunks(const uint32_t* __restrict__ keys,
                                                      const int32_t* __restrict__ pos, const S* __restrict__ dout,
                                                      float* __restrict__ dtable, float* __restrict__ head,
                                                      float* __restrict__ tail, uint8_t* __restrict__ touched,
                                                      int64_t T, int D, int64_t nchunks, int nslices) {
  const int lane = threadIdx.x & 31;
  const int64_t nwarps = (int64_t(gridDim.x) * blockDim.x) >> 5;
  for (int64_t item = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) >> 5; item < nchunks * nslices;
       item += nwarps) {
    const int64_t k = item / nslices;
    const int sl = static_cast<int>(item - k * nslices);
    const int col = sl * kSlice + lane * 8;
    const bool con = col < D;
    const int64_t i0 = k * kEmbedChunk, i1 = min(T, i0 + kEmbedChunk);
    const uint32_t prev = k > 0 ? keys[i0 - 1] : kNoKey;
    const uint32_t next = i1 < T ? keys[i1] : kNoKey;
    uint32_t cur = keys[i0];
    bool first = true;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto flush = [&](bool closed) {
      const bool from_prev = first && cur == prev;
      if (con) {
        if (!from_prev && closed) add_row_f32(dtable + int64_t(cur) * D + col, acc);
        else if (from_prev) put_row_f32(head + k * D + col, acc);
        else put_row_f32(tail + k * D + col, acc);
      }
      if (touched && sl == 0 && lane == 0) touched[cur] = 1;
    };
    for (int64_t b = i0; b < i1; b += 32) {
      const int n = static_cast<int>(i1 - b < 32 ? i1 - b : 32);
      const uint32_t kk = lane < n ? keys[b + lane] : kNoKey;
      const int32_t pp = lane < n ? pos[b + lane] : 0;
      for (int j0 = 0; j0 < n; j0 += 8) {
        float rows[8][8];
        uint32_t kj[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int j = j0 + u;
          kj[u] = __shfl_sync(0xffffffffu, kk, j & 31);
          const int32_t pj = __shfl_sync(0xffffffffu, pp, j & 31);
          if (j < n && con) {
            St<S>::load8(dout + int64_t(pj) * D + col, rows[u]);
          } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) rows[u][e] = 0.f;
          }
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          if (j0 + u >= n) break;
          if (kj[u] != cur) {
            flush(true);
            cur = kj[u];
            first = false;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.f;
          }
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] += rows[u][e];
        }
      }
    }
    flush(next != cur);
  }
}

// F32 path, pass 2: the chunk where a multi-chunk run starts adds its pieces in chunk order.
__global__ void __launch_bounds__(256) k_embed_merge(const uint32_t* __restrict__ keys, float* __restrict__ dtable,
                                                     const float* __restrict__ head, const float* __restrict__ tail,
                                                     int64_t T, int D, int64_t nchunks, int nslices) {
  const int lane = threadIdx.x & 31;
  const int64_t nwarps = (int64_t(gridDim.x) * blockDim.x) >> 5;
  for (int64_t item = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) >> 5; item < nchunks * nslices;
       item += nwarps) {
    const int64_t k = item / nslices;
    const int col = static_cast<int>(item - k * nslices) * kSlice + lane * 8;
    const int64_t i0 = k * kEmbedChunk, i1 = min(T, i0 + kEmbedChunk);
    if (i1 >= T) continue;
    const uint32_t K = keys[i1 - 1];
    if (keys[i1] != K) continue;                                    // the last run ends in this chunk
    if (k > 0 && keys[i0 - 1] == K && keys[i0] == K) continue;      // a through chunk: the run started earlier
    if (col >= D) continue;
    float acc[8];
    St<float>::load8_cg(tail + k * D + col, acc);
    for (int64_t j = k + 1; j < nchunks; ++j) {
      float h[8];
      St<float>::load8_cg(head + j * D + col, h);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] += h[e];
      const int64_t j1 = min(T, (j + 1) * kEmbedChunk);
      if (!(j1 < T && keys[j1 - 1] == K && keys[j1] == K)) break;   // the run ends in chunk j
    }
    add_row_f32(dtable + int64_t(K) * D + col, acc);
  }
}

// Bf16Sequential path: one warp per (chunk, 64-column slice) sums the runs that start in its chunk; the last one may
// continue past the chunk end. Running sum rounded to the storage dtype after each add (levanter's scatter-add), in
// ascending position order. Only the adds are sequential:
//   * the keys are sorted, so the first own entry (past a run that started in an earlier chunk) and the end of the last
//     run come from binary searches;
//   * entries go in batches of 32: lane j loads entry j's whole 64-column row slice (no shuffles), the warp transposes
//     the batch through shared memory into registers, and each lane then adds its 2 columns of the 32 rows in order;
//     the rows of batch i + 1 and the keys / positions of batch i + 2 are in flight while batch i is added; run
//     boundaries are one ballot, and a batch inside one run is a branch-free chain of 32 adds;
//   * under bf16 storage a lane's 2 columns are one bf16x2 word and each add is one packed __hadd2_rn: the sum of two
//     bf16 values rounded once equals bf16(f32(acc) + f32(row)) (ew_common.cuh V8), so the dependent chain per row is
//     a single instruction.
constexpr int kSeqCpl = 2;               // columns per lane
constexpr int kSeqSlice = 32 * kSeqCpl;  // columns per warp
constexpr int kSeqWarps = 4;             // warps per CTA (shared tile: 4 warps x 32 rows x 64 columns)

template <class S> struct SeqAcc;
template <> struct SeqAcc<bf16_t> {      // acc and rows as bf16x2 words
  using Raw = uint32_t;
  __device__ __forceinline__ static Raw lds(const uint4* row, int lane) {
    return reinterpret_cast<const uint32_t*>(row)[lane];
  }
  __device__ __forceinline__ static Raw zero() { return 0u; }
  __device__ __forceinline__ static Raw add(Raw acc, Raw r) { return u2(__hadd2_rn(b2(acc), b2(r))); }
  __device__ __forceinline__ static float2 f32(Raw a) {
    return make_float2(__uint_as_float(a << 16), __uint_as_float(a & 0xffff0000u));
  }
};
template <> struct SeqAcc<float> {       // f32 storage: the rounding is the identity, a plain f32 running sum
  using Raw = float2;
  __device__ __forceinline__ static Raw lds(const uint4* row, int lane) {
    return reinterpret_cast<const float2*>(row)[lane];
  }
  __device__ __forceinline__ static Raw zero() { return make_float2(0.f, 0.f); }
  __device__ __forceinline__ static Raw add(Raw acc, Raw r) { return make_float2(acc.x + r.x, acc.y + r.y); }
  __device__ __forceinline__ static float2 f32(Raw a) { return a; }
};

// First index in [lo, hi) whose key differs from K, given keys[lo..hi) ascending and every key before it <= K.
__device__ __forceinline__ int64_t first_after(const uint32_t* __restrict__ keys, int64_t lo, int64_t hi, uint32_t K) {
  while (lo < hi) {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (__ldg(keys + mid) <= K) lo = mid + 1; else hi = mid;
  }
  return lo;
}

template <class S>
__global__ void __launch_bounds__(32 * kSeqWarps) k_embed_seq(const uint32_t* __restrict__ keys,
                                                              const int32_t* __restrict__ pos,
                                                              const S* __restrict__ dout, float* __restrict__ dtable,
                                                              uint8_t* __restrict__ touched, int64_t T, int D,
                                                              int64_t nchunks, int nslices) {
  using A = SeqAcc<S>;
  constexpr int kVec = kSeqSlice * int(sizeof(S)) / 16;   // 16-byte vectors per row slice: 8 (bf16), 16 (f32)
  constexpr int kVecCols = 16 / int(sizeof(S));
  __shared__ __align__(16) uint4 tile[kSeqWarps][32][kVec];
  const int lane = threadIdx.x & 31;
  uint4(*tl)[kVec] = tile[threadIdx.x >> 5];
  const int64_t nwarps = (int64_t(gridDim.x) * blockDim.x) >> 5;
  for (int64_t item = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) >> 5; item < nchunks * nslices;
       item += nwarps) {
    const int64_t k = item / nslices;
    const int sl = static_cast<int>(item - k * nslices);
    const int c0 = sl * kSeqSlice;
    const bool con = c0 + lane * kSeqCpl < D;
    const int64_t i0 = k * kEmbedChunk, i1 = min(T, i0 + kEmbedChunk);
    const int64_t s0 = k > 0 ? first_after(keys, i0, i1, __ldg(keys + i0 - 1)) : i0;   // skip a continuing run
    if (s0 >= i1) continue;                                                            // a through chunk
    const int64_t e = first_after(keys, i1, T, __ldg(keys + i1 - 1));                 // the last run's end
    // lane j: entry j of a batch (key, position, row slice)
    auto load_kp = [&](int64_t b, uint32_t& kk, int32_t& pp) {
      const bool v = b + lane < e;
      kk = v ? __ldg(keys + b + lane) : kNoKey;
      pp = v ? __ldg(pos + b + lane) : 0;
    };
    auto load_row = [&](int64_t b, int32_t pp, uint4 (&rv)[kVec]) {
      const bool v = b + lane < e;
      const uint4* src = reinterpret_cast<const uint4*>(dout + int64_t(pp) * D + c0);
#pragma unroll
      for (int q = 0; q < kVec; ++q)
        rv[q] = (v && c0 + q * kVecCols < D) ? __ldg(src + q) : make_uint4(0u, 0u, 0u, 0u);
    };
    uint32_t cur = __ldg(keys + s0);
    typename A::Raw acc = A::zero();
    auto flush = [&]() {
      if (con) {
        const float2 a = A::f32(acc);
        float2* q = reinterpret_cast<float2*>(dtable + int64_t(cur) * D + c0 + lane * kSeqCpl);
        float2 v = *q;
        v.x += a.x;
        v.y += a.y;
        *q = v;
      }
      if (touched && sl == 0 && lane == 0) touched[cur] = 1;
    };
    uint32_t kk, kk1;
    int32_t pp, pp1;
    uint4 rows[kVec];
    load_kp(s0, kk, pp);
    load_kp(s0 + 32, kk1, pp1);
    load_row(s0, pp, rows);
    for (int64_t b = s0; b < e; b += 32) {
      const int n = static_cast<int>(e - b < 32 ? e - b : 32);
#pragma unroll
      for (int q = 0; q < kVec; ++q) tl[lane][q] = rows[q];
      __syncwarp();
      uint32_t kk2;
      int32_t pp2;
      load_row(b + 32, pp1, rows);   // batch i + 1 rows and batch i + 2 keys / positions in flight during the adds
      load_kp(b + 64, kk2, pp2);
      const uint32_t up = __shfl_up_sync(0xffffffffu, kk, 1);
      const uint32_t bmask = __ballot_sync(0xffffffffu, lane < n && kk != (lane == 0 ? cur : up));
      typename A::Raw r[32];
#pragma unroll
      for (int u = 0; u < 32; ++u) r[u] = A::lds(tl[u], lane);
      __syncwarp();
      if (n == 32 && bmask == 0) {   // inside one run (the long runs): a branch-free chain of 32 adds
#pragma unroll
        for (int u = 0; u < 32; ++u) acc = A::add(acc, r[u]);
      } else {
#pragma unroll
        for (int u = 0; u < 32; ++u) {
          if (u < n) {
            if ((bmask >> u) & 1u) {   // warp-uniform: entry b + u starts a run
              flush();
              cur = __shfl_sync(0xffffffffu, kk, u);
              acc = A::zero();
            }
            acc = A::add(acc, r[u]);
          }
        }
      }
      kk = kk1;
      pp = pp1;
      kk1 = kk2;
      pp1 = pp2;
    }
    flush();
  }
}

uint64_t cub_bytes_for(int64_t T, int bits) {
  size_t bytes = 0;
  NM_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                          static_cast<uint32_t*>(nullptr), static_cast<const int32_t*>(nullptr),
                                          static_cast<int32_t*>(nullptr), static_cast<int>(T), 0, bits));
  return bytes;
}
uint64_t partial_bytes(int64_t T, int D) {
  const int64_t nchunks = (T + kEmbedChunk - 1) / kEmbedChunk;
  return 2 * align256(uint64_t(nchunks) * D * sizeof(float));
}

}  // namespace

uint64_t embed_bwd_cub_bytes(int64_t T, int V) { return cub_bytes_for(T, key_bits(V)); }

EmbedBwdWorkspace embed_bwd_carve(void* base, uint64_t bytes, int64_t T, int V) {
  (void)V;
  EmbedBwdWorkspace w{};
  uint8_t* p = static_cast<uint8_t*>(base);
  const uint64_t a = align256(uint64_t(T) * 4);
  NM_REQUIRE(bytes >= 4 * a, "embed_bwd_carve: workspace smaller than the sort arrays");
  w.keys_in = reinterpret_cast<uint32_t*>(p);
  w.keys_out = reinterpret_cast<uint32_t*>(p + a);
  w.vals_in = reinterpret_cast<int32_t*>(p + 2 * a);
  w.vals_out = reinterpret_cast<int32_t*>(p + 3 * a);
  w.run_start = nullptr;   // not used: the chunked segment sum needs no run list
  w.n_runs = nullptr;
  w.cub_temp = p + 4 * a;
  w.cub_bytes = bytes - 4 * a;
  return w;
}

void embed_fwd_dt(DType st, const void* table, const int32_t* tokens, void* out, int64_t T, int D, cudaStream_t s) {
  NM_REQUIRE(T >= 0 && D > 0 && D % 8 == 0, "embed_fwd: D must be a positive multiple of 8");
  NM_REQUIRE(st == DType::bf16 || st == DType::f32, "embed_fwd: bf16 or f32 storage");
  NM_REQUIRE(table && tokens && out && aligned16(table) && aligned16(out), "embed_fwd: missing or unaligned tensor");
  if (T == 0) return;
  const int64_t nvec = T * (D / 8);
  if (st == DType::bf16) {
    auto k = k_embed_fwd<16>;
    const int grid = grid_for_items(nvec, 256, reinterpret_cast<const void*>(k), 256, 0);
    k<<<grid, 256, 0, s>>>(static_cast<const uint8_t*>(table), tokens, static_cast<uint8_t*>(out), nvec, D / 8);
  } else {
    auto k = k_embed_fwd<32>;
    const int grid = grid_for_items(nvec, 256, reinterpret_cast<const void*>(k), 256, 0);
    k<<<grid, 256, 0, s>>>(static_cast<const uint8_t*>(table), tokens, static_cast<uint8_t*>(out), nvec, D / 8);
  }
  NM_CUDA(cudaGetLastError());
}

template <class S>
static void embed_bwd_t(const S* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V, EmbedAccum acc,
                        const EmbedBwdWorkspace& ws, uint8_t* touched, cudaStream_t s) {
  const int bits = key_bits(V);
  const uint64_t need = align256(cub_bytes_for(T, bits));
  const uint64_t pbytes = acc == EmbedAccum::F32 ? partial_bytes(T, D) : 0;
  NM_REQUIRE(ws.cub_temp && ws.cub_bytes >= need + pbytes,
             "embed_bwd_sorted: workspace temp region " + std::to_string(ws.cub_bytes) + " B < " +
                 std::to_string(need + pbytes) + " B (sort temp + chunk partials)");
  {
    const int grid = grid_for_items(T, 256, reinterpret_cast<const void*>(k_embed_prep), 256, 0);
    k_embed_prep<<<grid, 256, 0, s>>>(tokens, ws.keys_in, ws.vals_in, T);
    NM_CUDA(cudaGetLastError());
  }
  size_t tb = need;
  NM_CUDA(cub::DeviceRadixSort::SortPairs(ws.cub_temp, tb, ws.keys_in, ws.keys_out, ws.vals_in, ws.vals_out,
                                          static_cast<int>(T), 0, bits, s));
  const int64_t nchunks = (T + kEmbedChunk - 1) / kEmbedChunk;
  const int nslices = (D + kSlice - 1) / kSlice;
  const int64_t warps = nchunks * nslices;
  if (acc == EmbedAccum::F32) {
    float* head = reinterpret_cast<float*>(static_cast<uint8_t*>(ws.cub_temp) + need);
    float* tail = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(head) + pbytes / 2);
    auto k1 = k_embed_chunks<S>;
    const int g1 = grid_for_items(warps, 8, reinterpret_cast<const void*>(k1), 256, 0);
    k1<<<g1, 256, 0, s>>>(ws.keys_out, ws.vals_out, dout, dtable, head, tail, touched, T, D, nchunks, nslices);
    NM_CUDA(cudaGetLastError());
    const int g2 = grid_for_items(warps, 8, reinterpret_cast<const void*>(k_embed_merge), 256, 0);
    k_embed_merge<<<g2, 256, 0, s>>>(ws.keys_out, dtable, head, tail, T, D, nchunks, nslices);
    NM_CUDA(cudaGetLastError());
  } else {
    auto k = k_embed_seq<S>;
    const int sslices = (D + kSeqSlice - 1) / kSeqSlice;
    const int g = grid_for_items(nchunks * sslices, kSeqWarps, reinterpret_cast<const void*>(k), 32 * kSeqWarps, 0);
    k<<<g, 32 * kSeqWarps, 0, s>>>(ws.keys_out, ws.vals_out, dout, dtable, touched, T, D, nchunks, sslices);
    NM_CUDA(cudaGetLastError());
  }
}

void embed_bwd_sorted_dt(DType st, const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                         EmbedAccum acc, const EmbedBwdWorkspace& ws, uint8_t* touched, cudaStream_t s) {
  NM_REQUIRE(T >= 0 && T < (int64_t(1) << 31), "embed_bwd_sorted: T out of range");
  NM_REQUIRE(D > 0 && D % 8 == 0, "embed_bwd_sorted: D must be a positive multiple of 8");
  NM_REQUIRE(V > 1, "embed_bwd_sorted: V must be > 1");
  NM_REQUIRE(dout && tokens && dtable && ws.keys_in && ws.keys_out && ws.vals_in && ws.vals_out,
             "embed_bwd_sorted: missing tensor or workspace array");
  NM_REQUIRE(aligned16(dout) && aligned16(dtable), "embed_bwd_sorted: dout and dtable must be 16-byte aligned");
  if (T == 0) return;
  if (st == DType::bf16)
    embed_bwd_t(static_cast<const bf16_t*>(dout), tokens, dtable, T, D, V, acc, ws, touched, s);
  else if (st == DType::f32)
    embed_bwd_t(static_cast<const float*>(dout), tokens, dtable, T, D, V, acc, ws, touched, s);
  else
    fatal(NM_WHERE, "embed_bwd_sorted: bf16 or f32 storage");
}

}  // namespace nm::ew

namespace nm {
void embed_fwd(const bf16_t* table, const int32_t* tokens, bf16_t* out, int64_t T, int D, cudaStream_t s) {
  ew::embed_fwd_dt(DType::bf16, table, tokens, out, T, D, s);
}
uint64_t embed_bwd_workspace_bytes(int64_t T, int V) {
  const uint64_t a = (uint64_t(T) * 4 + 255) & ~uint64_t(255);
  const int64_t nchunks = (T + ew::kEmbedChunk - 1) / ew::kEmbedChunk;
  const uint64_t part = 2 * ((uint64_t(nchunks) * ew::kEmbedMaxD * 4 + 255) & ~uint64_t(255));
  return 4 * a + ((ew::embed_bwd_cub_bytes(T, V) + 255) & ~uint64_t(255)) + part;
}
void embed_bwd_sorted(const bf16_t* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V, EmbedAccum acc,
                      const EmbedBwdWorkspace& ws, uint8_t* touched_rows, cudaStream_t s) {
  ew::embed_bwd_sorted_dt(DType::bf16, dout, tokens, dtable, T, D, V, acc, ws, touched_rows, s);
}
}  // namespace nm
