// FA3 backend: torch-free entry points over the vendored FlashAttention-3 (replaces hopper/flash_api.cpp). DESIGN.md §3.4.
//
// Two training modes (exec.attn_bwd_sched) and the E1b bench arms share one parameter fill:
//   Persistent  forward: FA3 varlen over the packed segments (b = rows * max_segs segment slots, empty slots have
//               length 0; VarlenDynamicPersistentTileScheduler). Backward: FA3 varlen causal over a device work list
//               built in the graph by k_bwd_worklists (segments longest first, then head, then n block descending);
//               the preprocess / postprocess grids cover only the tiles in the lists; main kernel grid = one CTA per
//               work-list slot (ListGrid; the hardware balances the load), or the SM count with the KVEmpty handoff
//               and static striding (ListPersistent, an E1b arm). Accumulators keep FA3's per-segment row padding.
//   DocCausal   batch = rows, seqlen = S, FA3's Is_local path with the doc-causal patch: per-position segment start /
//               end tables (k_doc_tables) bound the n / m loops and the masks. Grids (S / kBlock) x H x rows, all
//               useful; no padding rows. Forward at hd 64: the varlen forward of Persistent (faster there, E1b; it
//               writes O [T, H*hd] and lse [H, T], the layout the doc backward reads); at hd 128 the doc forward.
//   Bench arms (E1b): Stock padded = FA3's own varlen backward on the padded segment slots (the version-1 grid);
//               Stock exact = the same with the real segment count and maximum length from the host (E1's setup).
//
// Deterministic: ordered dQ accumulation (semaphores), GQA dK/dV accumulation ordered by head, no float atomics.
// Determinism off: TMA reduce-add of dQ (and GQA dK/dV) in arrival order.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#include "kernels/attention/attn_internal.hpp"
#include "kernels/attention/fa3_internal.hpp"
#include "marincpp/kernels/attention.hpp"

namespace nm::fa3k {

namespace {

constexpr int kCtaWl = 1024;   // threads of k_bwd_worklists (also the cost-bin limit)

// One CTA. Segment lengths from cu_seqlens (slots past *n_segs, or of length 0, have no tiles). Writes
//   mtiles: (segment, m_block) for every kBlockM-row tile, segment-major (pre- and postprocess);
//   items:  the main kernel's work items (segment, n_block | head << 16): segments by n-block count descending (ties in
//           segment order), inside a segment head by head, n descending (H * tiles items; FA3's deterministic dQ
//           order: the higher n blocks reach an m block first);
//   ktiles: (segment, n_block) for every kBlockN-key tile in the same segment order (GQA dK/dV postprocess).
__global__ void __launch_bounds__(kCtaWl) k_bwd_worklists(const int* __restrict__ cu, const int* __restrict__ n_segs,
                                                          int slots, int block_m, int block_n, int H,
                                                          int2* __restrict__ items, int* __restrict__ item_count,
                                                          int2* __restrict__ ktiles, int* __restrict__ ktile_count,
                                                          int2* __restrict__ mtiles, int* __restrict__ mtile_count) {
  extern __shared__ int smem[];
  int* len = smem;                 // [slots]
  int* mstart = smem + slots;      // [slots + 1] exclusive prefix of m tiles
  __shared__ int bin_count[kCtaWl];
  __shared__ int bin_base[kCtaWl];
  __shared__ int warp_tot[32];
  const int nseg = n_segs ? min(*n_segs, slots) : slots;
  for (int i = threadIdx.x; i < slots; i += blockDim.x) len[i] = i < nseg ? cu[i + 1] - cu[i] : 0;
  __syncthreads();
  // m-tile prefix over segments (serial chunks of 1024 with a warp-shuffle block scan)
  int carry = 0;
  for (int base = 0; base < slots; base += blockDim.x) {
    const int i = base + threadIdx.x;
    int v = i < slots ? (len[i] + block_m - 1) / block_m : 0;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, x, o);
      if ((threadIdx.x & 31) >= o) x += y;
    }
    if ((threadIdx.x & 31) == 31) warp_tot[threadIdx.x >> 5] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
      int w = threadIdx.x < (blockDim.x >> 5) ? warp_tot[threadIdx.x] : 0;
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, w, o);
        if (threadIdx.x >= o) w += y;
      }
      warp_tot[threadIdx.x] = w;   // inclusive over warps
    }
    __syncthreads();
    const int excl = carry + x - v + ((threadIdx.x >> 5) ? warp_tot[(threadIdx.x >> 5) - 1] : 0);
    if (i < slots) mstart[i] = excl;
    const int tot = warp_tot[(blockDim.x >> 5) - 1];
    __syncthreads();
    carry += tot;
  }
  if (threadIdx.x == 0) { mstart[slots] = carry; *mtile_count = carry; }
  __syncthreads();
  for (int i = threadIdx.x; i < nseg; i += blockDim.x) {
    const int nm = (len[i] + block_m - 1) / block_m;
    for (int m = 0; m < nm; ++m) mtiles[mstart[i] + m] = make_int2(i, m);
  }
  // segments by n-block count: thread c owns count c + 1
  const int c = threadIdx.x + 1;
  int cnt = 0;
  for (int i = 0; i < nseg; ++i) cnt += (len[i] + block_n - 1) / block_n == c;
  bin_count[threadIdx.x] = cnt * c;   // key tiles of this bin
  __syncthreads();
  if (threadIdx.x == 0) {   // descending count: base of bin c = key tiles of longer segments (<= 1024 bins, once)
    int acc = 0;
    for (int b = blockDim.x - 1; b >= 0; --b) { bin_base[b] = acc; acc += bin_count[b]; }
    *ktile_count = acc;
    *item_count = acc * H;
  }
  __syncthreads();
  int kpos = bin_base[threadIdx.x];
  for (int i = 0; i < nseg && cnt > 0; ++i) {
    if ((len[i] + block_n - 1) / block_n != c) continue;
    for (int n = 0; n < c; ++n) ktiles[kpos + n] = make_int2(i, n);
    for (int h = 0; h < H; ++h) {
      for (int n = 0; n < c; ++n) items[int64_t(kpos) * H + h * c + n] = make_int2(i, (c - 1 - n) | (h << 16));
    }
    kpos += c;
    --cnt;
  }
}

// doc_lo / doc_hi [rows, S] from row_seg_start [rows, max_segs + 1] (segment starts within the row, then S).
__global__ void k_doc_tables(const int* __restrict__ rss, int rows, int S, int max_segs, int* __restrict__ lo,
                             int* __restrict__ hi) {
  const int r = blockIdx.y;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || p >= S) return;
  const int* rs = rss + int64_t(r) * (max_segs + 1);
  int a = 0, b = max_segs;   // largest i with rs[i] <= p (rs[0] = 0, rs is non-decreasing)
  while (a < b) {
    const int m = (a + b + 1) >> 1;
    if (rs[m] <= p) a = m; else b = m - 1;
  }
  int e = S;
  for (int i = a + 1; i <= max_segs; ++i) {
    if (rs[i] > p) { e = rs[i]; break; }
  }
  lo[int64_t(r) * S + p] = rs[a];
  hi[int64_t(r) * S + p] = e;
}

int sm_count_of_current_device() {
  static std::mutex mu;
  static int cached[64] = {};
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  std::lock_guard<std::mutex> g(mu);
  if (dev < 64 && cached[dev]) return cached[dev];
  int n = 0;
  NM_CUDA(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  if (dev < 64) cached[dev] = n;
  return n;
}

int64_t round_up(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

struct Ws {   // byte offsets into the caller's workspace (256-byte aligned)
  size_t total = 0;
  size_t sched = 0, sched_bytes = 0;          // forward scheduler metadata + tile semaphore
  size_t doc_lo = 0, doc_hi = 0;              // Doc
  size_t dq_accum = 0, dpsum = 0, lse_log2 = 0;
  size_t dk_accum = 0, dv_accum = 0;          // GQA
  size_t dq_sem = 0, dkv_sem = 0, dkv_sem_bytes = 0, zero_bytes_gqa = 0;
  size_t ntiles = 0, ktiles = 0, mtiles = 0, counts = 0;  // List
  int64_t rows_q_pad = 0, rows_k_pad = 0;     // accumulator rows (per head)
  int slots = 0, max_ntiles = 0, max_mtiles = 0;
};

size_t take(size_t& at, size_t bytes) {
  const size_t o = at;
  at = round_up(at + bytes, 256);
  return o;
}

bool is_varlen(Fa3Mode m) { return m != Fa3Mode::Doc; }
// The forward runs FA3 varlen over the packed segments: every mode but Doc, and Doc at hd 64 (E1b: varlen 0.62 / 0.92 /
// 1.22 ms against doc 0.68 / 1.00 / 1.31 at 130m / 300m / 520m; at hd 128 the doc forward is faster, 1.72 vs 1.75).
bool fwd_varlen(const AttnShape& s, Fa3Mode m) { return is_varlen(m) || s.hd == 64; }
BwdKind bwd_kind(Fa3Mode m) {
  switch (m) {
    case Fa3Mode::Doc: return BwdKind::Doc;
    case Fa3Mode::ListPersistent: case Fa3Mode::ListGrid: return BwdKind::List;
    default: return BwdKind::Stock;
  }
}

Ws layout(const AttnShape& s, Fa3Mode mode, bool training) {
  Ws w;
  size_t at = 0;
  const int H = s.H, KV = s.KV, hd = s.hd;
  const int bm = bwd_block_m(hd), bn = bwd_block_n(hd);
  const bool gqa = H != KV;
  const int slots = s.rows * s.max_segs_per_seq;
  w.slots = slots;
  if (fwd_varlen(s, mode)) {
    const int b4 = int(round_up(slots, 4));
    w.sched_bytes = size_t(b4) * 4 * 4 + 4 * 4;   // num_splits, num_m_blocks, batch_idx, nheads_in_l2 + semaphore
  } else {
    w.sched_bytes = 16;                           // DynamicPersistentTileScheduler semaphore
  }
  if (!is_varlen(mode)) {
    w.doc_lo = take(at, size_t(s.rows) * s.S * 4);
    w.doc_hi = take(at, size_t(s.rows) * s.S * 4);
  }
  w.sched = take(at, w.sched_bytes);
  if (training) {
    if (is_varlen(mode)) {
      w.rows_q_pad = round_up(s.total_q + int64_t(slots) * bm, bm);
      w.rows_k_pad = round_up(s.total_q + int64_t(slots) * bn, bn);
    } else {   // flash_api.cpp: seqlen_q_rounded = seqlen_k_rounded = round_up(S, 128)
      w.rows_q_pad = int64_t(s.rows) * round_up(s.S, 128);
      w.rows_k_pad = int64_t(s.rows) * round_up(s.S, 128);
    }
    const int b = is_varlen(mode) ? slots : s.rows;
    w.dq_accum = take(at, size_t(H) * w.rows_q_pad * hd * 4);
    w.dpsum = take(at, size_t(H) * w.rows_q_pad * 4);
    w.lse_log2 = take(at, size_t(H) * w.rows_q_pad * 4);
    w.dq_sem = take(at, size_t((s.S + bm - 1) / bm) * b * H * 4);
    if (gqa) {
      w.zero_bytes_gqa = size_t(KV) * w.rows_k_pad * hd * 4;
      w.dk_accum = take(at, w.zero_bytes_gqa);
      w.dv_accum = take(at, w.zero_bytes_gqa);
      w.dkv_sem_bytes = size_t((s.S + bn - 1) / bn) * b * KV * 4;
      w.dkv_sem = take(at, 2 * w.dkv_sem_bytes);
    }
    if (bwd_kind(mode) == BwdKind::List) {
      w.max_ntiles = int(s.total_q / bn + slots + 1);
      w.max_mtiles = int(s.total_q / bm + slots + 1);
      w.ntiles = take(at, size_t(w.max_ntiles) * H * 8);   // work items
      w.ktiles = take(at, size_t(w.max_ntiles) * 8);
      w.mtiles = take(at, size_t(w.max_mtiles) * 8);
      w.counts = take(at, 16);
    }
  }
  w.total = at;
  return w;
}

void check_shape(const AttnShape& s) {
  NM_REQUIRE(s.io == DType::bf16, "FA3 takes bf16 q, k, v");
  NM_REQUIRE((s.hd == 64 && (s.H == s.KV || s.H % s.KV == 0)) || (s.hd == 128 && s.H != s.KV && s.H % s.KV == 0),
             "FA3 is built for hd 64 (MHA, GQA) and hd 128 (GQA); got hd " + std::to_string(s.hd) + ", H " +
                 std::to_string(s.H) + ", KV " + std::to_string(s.KV));
  NM_REQUIRE(s.rows > 0 && s.S > 0 && s.total_q == int64_t(s.rows) * s.S, "FA3: total_q must equal rows * S");
  NM_REQUIRE(s.total_q < (int64_t(1) << 31), "FA3: total_q must fit int32");
  NM_REQUIRE(s.max_segs_per_seq >= 1, "FA3: max_segs_per_seq >= 1");
}

// The fields every call sets (flash_api.cpp set_params_fprop for bf16 q/k/v with row-major [T, heads * hd] rows).
// varlen: packed segments (cu_seqlens); otherwise the doc-causal batch of rows.
void fill_common(Flash_fwd_params& p, const AttnShape& s, const AttnFwdArgs& a, Fa3Mode mode, bool varlen,
                 const Ws& w, char* ws, const Fa3Exact* exact) {
  std::memset(&p, 0, sizeof(p));
  p.is_bf16 = true;
  p.q_ptr = const_cast<void*>(a.q);
  p.k_ptr = const_cast<void*>(a.k);
  p.v_ptr = const_cast<void*>(a.v);
  p.o_ptr = a.o;
  p.q_row_stride = a.q_ld; p.k_row_stride = a.k_ld; p.v_row_stride = a.v_ld; p.o_row_stride = a.o_ld;
  p.q_head_stride = s.hd; p.k_head_stride = s.hd; p.v_head_stride = s.hd; p.o_head_stride = s.hd;
  p.v_dim_stride = 1;
  p.softmax_lse_ptr = a.lse;
  p.h = s.H; p.h_k = s.KV;
  p.d = s.hd; p.d_rounded = s.hd; p.dv = s.hd; p.dv_rounded = s.hd;
  p.scale_softmax = a.softmax_scale;
  p.softcap = 0.f;
  p.p_dropout = 1.f; p.rp_dropout = 1.f; p.p_dropout_in_uint8_t = 255;
  p.num_splits = 1;
  p.arch = 90;
  p.num_sm = sm_count_of_current_device();
  if (varlen) {
    const bool ex = mode == Fa3Mode::StockExact;
    p.b = ex ? exact->n_segs : w.slots;
    p.seqlen_q = p.seqlen_k = ex ? exact->max_seqlen : s.S;
    p.cu_seqlens_q = p.cu_seqlens_k = const_cast<int*>(a.cu_seqlens);
    p.total_q = p.total_k = int(s.total_q);
    p.is_causal = true;
    p.window_size_left = -1;
    p.window_size_right = 0;
  } else {
    p.b = s.rows;
    p.seqlen_q = p.seqlen_k = s.S;
    p.total_q = p.total_k = int(s.total_q);
    p.q_batch_stride = int64_t(s.S) * a.q_ld; p.k_batch_stride = int64_t(s.S) * a.k_ld;
    p.v_batch_stride = int64_t(s.S) * a.v_ld; p.o_batch_stride = int64_t(s.S) * a.o_ld;
    p.is_local = true;               // Is_local kernels; the doc tables replace the left window
    p.window_size_left = s.S - 1;
    p.window_size_right = 0;
    p.doc_lo = reinterpret_cast<const int*>(ws + w.doc_lo);
    p.doc_hi = reinterpret_cast<const int*>(ws + w.doc_hi);
    p.lse_head_stride = s.total_q;   // lse is one [H, T] tensor
    p.lse_batch_stride = s.S;
  }
  p.seqlen_q_rounded = int(round_up(p.seqlen_q, 128));
  p.seqlen_k_rounded = int(round_up(p.seqlen_k, 128));
}

void build_doc_tables(const AttnShape& s, const AttnFwdArgs& a, const Ws& w, char* ws, cudaStream_t st) {
  NM_REQUIRE(a.row_seg_start != nullptr, "FA3 doc_causal needs row_seg_start");
  dim3 grid((s.S + 255) / 256, s.rows);
  k_doc_tables<<<grid, 256, 0, st>>>(a.row_seg_start, s.rows, s.S, s.max_segs_per_seq,
                                     reinterpret_cast<int*>(ws + w.doc_lo), reinterpret_cast<int*>(ws + w.doc_hi));
  NM_CUDA(cudaGetLastError());
}

}  // namespace

uint64_t workspace_bytes(const AttnShape& s, Fa3Mode mode, bool training) {
  check_shape(s);
  return layout(s, mode, training).total;
}

void fwd(const AttnShape& s, const AttnFwdArgs& a, Fa3Mode mode, const Fa3Exact* exact, void* ws_, cudaStream_t st) {
  check_shape(s);
  NM_REQUIRE(a.extra_k == nullptr && a.extra_v == nullptr, "FA3: extra_kv (moda) is not built");
  NM_REQUIRE(a.q && a.k && a.v && a.o && a.lse, "FA3 fwd: q, k, v, o, lse required");
  NM_REQUIRE(mode != Fa3Mode::StockExact || exact != nullptr, "FA3 StockExact needs the host segment count");
  char* ws = static_cast<char*>(ws_);
  const Ws w = layout(s, mode, false);
  Flash_fwd_params p;
  const bool varlen = fwd_varlen(s, mode);
  fill_common(p, s, a, mode, varlen, w, ws, exact);
  const bool gqa = s.H != s.KV;
  int* sched = reinterpret_cast<int*>(ws + w.sched);
  try {
    if (varlen) {
      NM_REQUIRE(a.cu_seqlens != nullptr, "FA3 varlen needs cu_seqlens");
      const int b4 = int(round_up(p.b, 4));
      // flash_api.cpp mha_fwd, varlen: prepare_varlen_num_blocks fills these and zeroes the tile semaphore.
      p.pack_gqa = gqa;                        // FA3's heuristic for varlen q (heuristics.h should_pack_gqa)
      p.varlen_sort_batches = true;            // flash_api.cpp: !is_local
      p.head_swizzle = true;
      p.num_splits_dynamic_ptr = sched;
      p.num_m_blocks_ptr = sched + b4;
      p.varlen_batch_idx_ptr = sched + 2 * b4;
      p.num_nheads_in_l2_ptr = sched + 3 * b4;
      p.tile_count_semaphore = sched + 4 * b4;
      p.tile_count_semaphore_offset = 4 * b4;
      p.prepare_varlen_pdl = false;
      p.skip_scheduler_metadata_computation = false;
      if (s.hd == 64) {
        if (gqa) fwd_hd64_varlen_packgqa(p, st); else fwd_hd64_varlen(p, st);
      } else {
        fwd_hd128_varlen_packgqa(p, st);
      }
    } else {
      build_doc_tables(s, a, w, ws, st);
      NM_CUDA(cudaMemsetAsync(sched, 0, 4, st));
      p.tile_count_semaphore = sched;
      p.pack_gqa = false;
      p.head_swizzle = true;
      if (s.hd == 64) fwd_hd64_doc(p, st); else fwd_hd128_doc(p, st);
    }
  } catch (const std::runtime_error& e) {
    if (dynamic_cast<const Fatal*>(&e)) throw;
    throw DeviceError(e.what());
  }
}

void bwd(const AttnShape& s, const AttnBwdArgs& a, Fa3Mode mode, const Fa3Exact* exact, void* ws_, cudaStream_t st) {
  check_shape(s);
  const AttnFwdArgs& f = a.fwd;
  NM_REQUIRE(f.q && f.k && f.v && f.o && f.lse && a.d_o && a.dk && a.dv, "FA3 bwd: q, k, v, o, lse, dO, dk, dv required");
  NM_REQUIRE(a.softmax_d == nullptr, "FA3 bwd: a precomputed softmax_d (o_proj epilogue, S8) is not built yet");
  NM_REQUIRE(a.dq_accum == nullptr && a.dq != nullptr,
             "FA3 bwd: dq_accum output is not built (FA3 stores dQ in its MMA fragment order per tile); pass dq");
  NM_REQUIRE(mode != Fa3Mode::StockExact || exact != nullptr, "FA3 StockExact needs the host segment count");
  char* ws = static_cast<char*>(ws_);
  const Ws w = layout(s, mode, true);
  Flash_bwd_params p;
  fill_common(p, s, f, mode, is_varlen(mode), w, ws, exact);
  const bool gqa = s.H != s.KV;
  const bool det = s.deterministic;
  p.do_ptr = const_cast<void*>(a.d_o);
  p.do_row_stride = a.do_ld; p.do_head_stride = s.hd;
  p.dq_ptr = a.dq; p.dk_ptr = a.dk; p.dv_ptr = a.dv;
  p.dq_row_stride = a.dq_ld; p.dk_row_stride = a.dk_ld; p.dv_row_stride = a.dv_ld;
  p.dq_head_stride = s.hd; p.dk_head_stride = s.hd; p.dv_head_stride = s.hd;
  if (!is_varlen(mode)) {
    p.do_batch_stride = int64_t(s.S) * a.do_ld;
    p.dq_batch_stride = int64_t(s.S) * a.dq_ld;
    p.dk_batch_stride = int64_t(s.S) * a.dk_ld;
    p.dv_batch_stride = int64_t(s.S) * a.dv_ld;
  }
  p.dq_accum_ptr = ws + w.dq_accum;
  p.dsoftmax_sum = ws + w.dpsum;
  p.softmax_lse_log2_ptr = ws + w.lse_log2;
  p.dq_semaphore = reinterpret_cast<int*>(ws + w.dq_sem);   // zeroed by the preprocess kernel
  p.deterministic = det;
  if (gqa) {
    p.dk_accum_ptr = ws + w.dk_accum;
    p.dv_accum_ptr = ws + w.dv_accum;
    p.dk_semaphore = reinterpret_cast<int*>(ws + w.dkv_sem);
    p.dv_semaphore = reinterpret_cast<int*>(ws + w.dkv_sem + w.dkv_sem_bytes);
  }
  const BwdKind kind = bwd_kind(mode);
  try {
    if (!is_varlen(mode)) build_doc_tables(s, f, w, ws, st);
    if (gqa) {
      // Deterministic training modes: the first head of every kv group stores its dK/dV tile and the others add in
      // head order, so the accumulators need no zeroing (DESIGN.md §3.4). Otherwise (FA3's atomic order, and the
      // stock bench arms) they start at zero as in flash_api.cpp: memset nodes in a graph. The semaphores always do.
      p.nm_gqa_first_store = det && kind != BwdKind::Stock;
      if (!p.nm_gqa_first_store) {
        NM_CUDA(cudaMemsetAsync(ws + w.dk_accum, 0, w.zero_bytes_gqa, st));
        NM_CUDA(cudaMemsetAsync(ws + w.dv_accum, 0, w.zero_bytes_gqa, st));
      }
      NM_CUDA(cudaMemsetAsync(ws + w.dkv_sem, 0, 2 * w.dkv_sem_bytes, st));
    }
    if (kind == BwdKind::List) {
      NM_REQUIRE(f.cu_seqlens != nullptr, "FA3 persistent backward needs cu_seqlens");
      const int bm = bwd_block_m(s.hd), bn = bwd_block_n(s.hd);
      NM_REQUIRE((s.S + bn - 1) / bn <= kCtaWl, "FA3 work list: S / kBlockN must be <= 1024");
      int2* items = reinterpret_cast<int2*>(ws + w.ntiles);
      int2* ktiles = reinterpret_cast<int2*>(ws + w.ktiles);
      int2* mtiles = reinterpret_cast<int2*>(ws + w.mtiles);
      int* counts = reinterpret_cast<int*>(ws + w.counts);   // items, ktiles, mtiles
      const size_t smem = size_t(2 * w.slots + 1) * 4;
      NM_REQUIRE(smem <= 48 * 1024, "FA3 work list: too many segment slots (rows * max_segs <= 6143)");
      NM_REQUIRE(s.H < (1 << 15) && (s.S + bn - 1) / bn < (1 << 16), "FA3 work list: item encoding");
      k_bwd_worklists<<<1, kCtaWl, smem, st>>>(f.cu_seqlens, f.n_segs, w.slots, bm, bn, s.H, items, counts, ktiles,
                                               counts + 1, mtiles, counts + 2);
      NM_CUDA(cudaGetLastError());
      p.nm_ntiles = items; p.nm_ntile_count = counts;
      p.nm_ktiles = ktiles; p.nm_ktile_count = counts + 1;
      p.nm_mtiles = mtiles; p.nm_mtile_count = counts + 2;
      p.nm_persistent = mode == Fa3Mode::ListPersistent;
      p.nm_grid = p.nm_persistent ? p.num_sm : w.max_ntiles * s.H;
      p.nm_pre_grid = w.max_mtiles;
      p.nm_post_kv_grid = w.max_ntiles;
    }
    const int k = int(kind);
    if (s.hd == 64) {
      switch (k * 2 + det) {
        case 0: bwd_hd64_k0_d0(p, gqa, st); break;
        case 1: bwd_hd64_k0_d1(p, gqa, st); break;
        case 2: bwd_hd64_k1_d0(p, gqa, st); break;
        case 3: bwd_hd64_k1_d1(p, gqa, st); break;
        case 4: bwd_hd64_k2_d0(p, gqa, st); break;
        default: bwd_hd64_k2_d1(p, gqa, st); break;
      }
    } else {
      switch (k * 2 + det) {
        case 0: bwd_hd128_k0_d0(p, gqa, st); break;
        case 1: bwd_hd128_k0_d1(p, gqa, st); break;
        case 2: bwd_hd128_k1_d0(p, gqa, st); break;
        case 3: bwd_hd128_k1_d1(p, gqa, st); break;
        case 4: bwd_hd128_k2_d0(p, gqa, st); break;
        default: bwd_hd128_k2_d1(p, gqa, st); break;
      }
    }
  } catch (const std::runtime_error& e) {
    if (dynamic_cast<const Fatal*>(&e)) throw;
    throw DeviceError(e.what());
  }
}

}  // namespace nm::fa3k
