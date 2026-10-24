// k_doc_meta: microbatch metadata from the token rows (elementwise.hpp DocMetaArgs; data.md §3.1; DESIGN.md §3.4).
//   kernel 1, one CTA per row: targets (levanter's roll), weights, positions, the row's segment starts (a segment starts
//            at the row start and after every EOS token) into row_seg_start[r][0..n_r), then S up to max_segs; sets
//            *overflow when n_r > max_segs (the first max_segs starts are kept).
//   kernel 2, one CTA: packed cu_seqlens over the rows (row offset r * S + start), the total, and the device segment
//            count; entries past the count repeat the total token count.
#include <cub/block/block_scan.cuh>

#include <string>

#include "kernels/elementwise/ew_common.cuh"
#include "marincpp/kernels/elementwise.hpp"

namespace nm::ew {
namespace {

constexpr int kRowThreads = 512;

__global__ void __launch_bounds__(kRowThreads) k_doc_meta_rows(const int32_t* __restrict__ tokens,
                                                               const uint8_t* __restrict__ row_valid, int S,
                                                               int32_t eos, int max_segs, int32_t* __restrict__ targets,
                                                               float* __restrict__ weight, int32_t* __restrict__ pos,
                                                               int32_t* __restrict__ row_seg_start,
                                                               uint32_t* __restrict__ overflow) {
  using Scan = cub::BlockScan<int, kRowThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ int total;
  const int r = blockIdx.x;
  const int32_t* row = tokens + int64_t(r) * S;
  const int64_t base = int64_t(r) * S;
  const float wv = (row_valid == nullptr || row_valid[r]) ? 1.f : 0.f;
  const int32_t first = row[0];
  // contiguous range of positions per thread, so the scan yields the starts in ascending order
  const int per = (S + kRowThreads - 1) / kRowThreads;
  const int p0 = min(S, threadIdx.x * per), p1 = min(S, p0 + per);
  int cnt = 0;
  for (int p = p0; p < p1; ++p) {
    targets[base + p] = p + 1 < S ? row[p + 1] : first;
    weight[base + p] = p + 1 < S ? wv : 0.f;
    pos[base + p] = p;
    cnt += (p == 0 || row[p - 1] == eos) ? 1 : 0;
  }
  int off = 0;
  Scan(scan_tmp).ExclusiveSum(cnt, off);
  if (threadIdx.x == kRowThreads - 1) total = off + cnt;
  int32_t* rs = row_seg_start + int64_t(r) * (max_segs + 1);
  for (int p = p0; p < p1; ++p)
    if (p == 0 || row[p - 1] == eos) {
      if (off < max_segs) rs[off] = p;
      ++off;
    }
  __syncthreads();
  const int n = total;
  if (n > max_segs && threadIdx.x == 0) *overflow = 1u;
  for (int i = min(n, max_segs) + threadIdx.x; i <= max_segs; i += kRowThreads) rs[i] = S;
}

__global__ void __launch_bounds__(256) k_doc_meta_pack(const int32_t* __restrict__ row_seg_start, int rows, int S,
                                                       int max_segs, int32_t* __restrict__ cu_seqlens,
                                                       int32_t* __restrict__ n_segments) {
  extern __shared__ int off[];   // [rows + 1]
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int32_t* rs = row_seg_start + int64_t(r) * (max_segs + 1);
    int c = 0;
    while (c < max_segs && rs[c] < S) ++c;
    off[r + 1] = c;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    off[0] = 0;
    for (int r = 0; r < rows; ++r) off[r + 1] += off[r];
    *n_segments = off[rows];
  }
  __syncthreads();
  const int total = off[rows];
  const int32_t T = rows * S;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int32_t* rs = row_seg_start + int64_t(r) * (max_segs + 1);
    for (int i = 0; i < off[r + 1] - off[r]; ++i) cu_seqlens[off[r] + i] = r * S + rs[i];
  }
  for (int i = total + threadIdx.x; i <= rows * max_segs; i += blockDim.x) cu_seqlens[i] = T;
}

}  // namespace
}  // namespace nm::ew

namespace nm {
void doc_meta(const DocMetaArgs& a, cudaStream_t s) {
  NM_REQUIRE(a.rows > 0 && a.S > 0 && a.max_segs_per_seq > 0, "doc_meta: rows, S and max_segs_per_seq must be > 0");
  NM_REQUIRE(int64_t(a.rows) * a.S < (int64_t(1) << 31), "doc_meta: rows * S must fit int32 (cu_seqlens)");
  NM_REQUIRE(a.tokens && a.targets && a.weight && a.pos && a.cu_seqlens && a.n_segments && a.row_seg_start &&
                 a.overflow,
             "doc_meta: missing output");
  NM_REQUIRE(a.rows <= 16384, "doc_meta: at most 16384 rows per call");
  ew::k_doc_meta_rows<<<a.rows, ew::kRowThreads, 0, s>>>(a.tokens, a.row_valid, a.S, a.eos_id, a.max_segs_per_seq,
                                                         a.targets, a.weight, a.pos, a.row_seg_start, a.overflow);
  NM_CUDA(cudaGetLastError());
  // [rows + 1] ints of dynamic shared memory: past 48 KiB (rows > 12287) the kernel must opt in (64 KiB at 16384 rows,
  // within every sm_80 / sm_90 limit). A function attribute, not a stream operation: legal during graph capture.
  const size_t smem = size_t(a.rows + 1) * sizeof(int);
  if (smem > 48 * 1024)
    NM_CUDA(cudaFuncSetAttribute(ew::k_doc_meta_pack, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  ew::k_doc_meta_pack<<<1, 256, smem, s>>>(a.row_seg_start, a.rows, a.S, a.max_segs_per_seq, a.cu_seqlens,
                                           a.n_segments);
  NM_CUDA(cudaGetLastError());
}
}  // namespace nm
