// MuonH elementwise kernels (optim.md §4): Nesterov momentum into the Newton-Schulz buffer, NS normalization and
// scaling, the 3xTF32 operand split, and the three hyperball passes. One CTA per 64 x 64 tile (kTile) of a layer
// matrix [OUT, IN], walked as four 32 x 32 sub-tiles; a transposed matrix (OUT > IN) goes through a shared-memory
// sub-tile so both the storage and the NS buffer (wide orientation) are accessed in coalesced rows.
#include "kernels/optim/optk_common.cuh"

namespace nm::optk {
namespace {

using namespace dev;

constexpr int kSub = 32;               // sub-tile edge; a CTA walks the (kTile / kSub)^2 sub-tiles of its tile
constexpr int kTy = 8;                 // blockDim = (32, 8): each thread handles 4 rows of a sub-tile
constexpr int kRowsPerThread = kSub / kTy;
constexpr int kSubs = (kTile / kSub) * (kTile / kSub);

struct TileRef {
  int mat;
  int out, in;
  int r0, c0;
  bool trans;
  int64_t x_off;
};

__device__ __forceinline__ TileRef tile_ref(const MatTab& m, int m_begin, int m_end, int64_t tile) {
  int lo = m_begin, hi = m_end - 1;   // last matrix whose tile0 <= tile
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (m.tile0[mid] <= tile) lo = mid; else hi = mid - 1;
  }
  TileRef t;
  t.mat = lo;
  t.out = m.out[lo];
  t.in = m.in[lo];
  const int64_t tt = tile - m.tile0[lo];
  const int tiles_c = (t.in + kTile - 1) / kTile;
  t.r0 = static_cast<int>(tt / tiles_c) * kTile;
  t.c0 = static_cast<int>(tt % tiles_c) * kTile;
  t.trans = m.trans[lo] != 0;
  t.x_off = m.x_off[lo];
  return t;
}

// Element offset of matrix element (r, c) relative to the tensor pointers.
__device__ __forceinline__ int64_t stor(const TensorTab& t, const MatTab& m, int tensor, int mat, int r, int c,
                                        int in) {
  const int32_t* map = m.row_map[mat];
  const int64_t row = map ? static_cast<int64_t>(map[r]) : m.row0[mat] + r;
  return (row - t.base_row[tensor]) * static_cast<int64_t>(in) + c;
}

// X index of matrix element (r, c): X = U [OUT, IN] or X = U^T [IN, OUT].
__device__ __forceinline__ int64_t xidx(const TileRef& tr, int r, int c) {
  return tr.trans ? tr.x_off + static_cast<int64_t>(c) * tr.out + r : tr.x_off + static_cast<int64_t>(r) * tr.in + c;
}

// Sub-tile s of a tile (row-major order of the 2 x 2 sub-tiles); valid() is false past the matrix edge (uniform in
// the CTA, so skipping it keeps every __syncthreads reached by all threads).
__device__ __forceinline__ TileRef sub_tile(const TileRef& t, int s) {
  TileRef u = t;
  u.r0 += (s / (kTile / kSub)) * kSub;
  u.c0 += (s % (kTile / kSub)) * kSub;
  return u;
}
__device__ __forceinline__ bool valid(const TileRef& t) { return t.r0 < t.out && t.c0 < t.in; }

// Loads the tile of X in [OUT, IN] coordinates into v[j] (row r0 + ty + 8j, column c0 + tx). A transposed matrix is
// read along X's rows through shared memory.
__device__ __forceinline__ void load_x_tile(const TileRef& tr, const float* x, float (*sm)[kSub + 1],
                                            float v[kRowsPerThread]) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  if (tr.trans) {
    // X rows = matrix columns c0 + ty + 8j, X columns = matrix rows r0 + tx
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int c = tr.c0 + ty + kTy * j, r = tr.r0 + tx;
      sm[ty + kTy * j][tx] = (r < tr.out && c < tr.in) ? x[tr.x_off + static_cast<int64_t>(c) * tr.out + r] : 0.0f;
    }
    __syncthreads();
    for (int j = 0; j < kRowsPerThread; ++j) v[j] = sm[tx][ty + kTy * j];
    __syncthreads();
  } else {
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = tr.r0 + ty + kTy * j, c = tr.c0 + tx;
      v[j] = (r < tr.out && c < tr.in) ? x[xidx(tr, r, c)] : 0.0f;
    }
  }
}

// Stores v[j] (matrix coordinates as in load_x_tile) into X.
__device__ __forceinline__ void store_x_tile(const TileRef& tr, float* x, float (*sm)[kSub + 1],
                                             const float v[kRowsPerThread]) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  if (tr.trans) {
    for (int j = 0; j < kRowsPerThread; ++j) sm[tx][ty + kTy * j] = v[j];
    __syncthreads();
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int c = tr.c0 + ty + kTy * j, r = tr.r0 + tx;
      if (r < tr.out && c < tr.in) x[tr.x_off + static_cast<int64_t>(c) * tr.out + r] = sm[ty + kTy * j][tx];
    }
    __syncthreads();
  } else {
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = tr.r0 + ty + kTy * j, c = tr.c0 + tx;
      if (r < tr.out && c < tr.in) x[xidx(tr, r, c)] = v[j];
    }
  }
}

__global__ void __launch_bounds__(kSub* kTy) k_muon_momentum(TensorTab t, MatTab m, const GroupDev* groups,
                                                              const ClipDev* clip, int m_begin, int m_end,
                                                              int64_t tile_begin, float* x, double* partial) {
  __shared__ float sm[kSub][kSub + 1];
  const int64_t tile = tile_begin + blockIdx.x;
  const TileRef tr = tile_ref(m, m_begin, m_end, tile);
  const int tensor = m.tensor[tr.mat];
  const GroupDev gr = groups[t.group[tensor]];
  const ClipDev cl = clip[gr.clip];
  const float gs = t.grad_scale[tensor];
  const float* g = t.g[tensor];
  float* buf = t.s0[tensor];
  float v[kRowsPerThread];
  double acc = 0.0;
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      v[j] = 0.0f;
      if (r < st.out && c < st.in) {
        const int64_t e = stor(t, m, tensor, st.mat, r, c, st.in);
        const float gi = clip_g(mul(g[e], gs), cl);
        const float b = add(mul(gr.momentum, buf[e]), gi);   // buf = momentum * m + g
        buf[e] = b;
        const float u = add(mul(gr.momentum, b), gi);        // nesterov: momentum * buf + g, with the new buf
        v[j] = u;
        acc += sq(u);
      }
    }
    store_x_tile(st, x, sm, v);
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0 && threadIdx.y == 0) partial[tile] = acc;
}

// blockIdx.y = matrix; float4 over the matrix when its elements are 16-byte aligned.
__global__ void k_ns_normalize(float* x, int64_t mn, const double* u2, float eps) {
  const float d = add(norm_of(u2[blockIdx.y]), eps);   // X /= jnp.linalg.norm(X) + eps
  float* xm = x + int64_t(blockIdx.y) * mn;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  const int64_t i0 = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if ((mn & 3) == 0 && (reinterpret_cast<uintptr_t>(xm) & 15) == 0) {
    float4* x4 = reinterpret_cast<float4*>(xm);
    for (int64_t i = i0; i < mn / 4; i += stride) {
      float4 v = x4[i];
      v.x = div(v.x, d), v.y = div(v.y, d), v.z = div(v.z, d), v.w = div(v.w, d);
      x4[i] = v;
    }
  } else {
    for (int64_t i = i0; i < mn; i += stride) xm[i] = div(xm[i], d);
  }
}

__global__ void k_ns_scale(float* dst, const float* src, float c, int64_t n) {
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  const int64_t i0 = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if ((n & 3) == 0 && ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(dst);
    for (int64_t i = i0; i < n / 4; i += stride) {
      const float4 v = s4[i];
      d4[i] = make_float4(mul(c, v.x), mul(c, v.y), mul(c, v.z), mul(c, v.w));
    }
  } else {
    for (int64_t i = i0; i < n; i += stride) dst[i] = mul(c, src[i]);
  }
}

// Lower-left block [m1:m, 0:m1] of each [m, m] matrix = transpose of the upper-right block [0:m1, m1:m], through a
// 32 x 32 shared-memory tile (blockIdx.x over the tiles of the lower-left block, blockIdx.y = matrix).
__global__ void __launch_bounds__(kSub* kTy) k_ns_mirror(float* d, int m, int m1) {
  __shared__ float sm[kSub][kSub + 1];
  const int m2 = m - m1;
  const int tiles_c = (m1 + kSub - 1) / kSub;
  const int r0 = static_cast<int>(blockIdx.x / tiles_c) * kSub;   // row offset inside [m1:m]
  const int c0 = static_cast<int>(blockIdx.x % tiles_c) * kSub;   // column offset inside [0:m1]
  float* x = d + int64_t(blockIdx.y) * m * m;
  // read the upper-right source rows c0.. (columns m1 + r0..), coalesced along the source rows
  for (int j = 0; j < kRowsPerThread; ++j) {
    const int sr = c0 + threadIdx.y + kTy * j, sc = r0 + threadIdx.x;
    if (sr < m1 && sc < m2) sm[threadIdx.y + kTy * j][threadIdx.x] = x[int64_t(sr) * m + m1 + sc];
  }
  __syncthreads();
  for (int j = 0; j < kRowsPerThread; ++j) {
    const int r = r0 + threadIdx.y + kTy * j, c = c0 + threadIdx.x;
    if (r < m2 && c < m1) x[int64_t(m1 + r) * m + c] = sm[threadIdx.x][threadIdx.y + kTy * j];
  }
}

__global__ void k_tf32_split(const float* x, float* big, float* small, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const float v = x[i];
    uint32_t b;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b) : "f"(v));
    const float bf = __uint_as_float(b);
    big[i] = bf;
    small[i] = sub(v, bf);
  }
}

__global__ void __launch_bounds__(kSub* kTy) k_hb_norms(TensorTab t, MatTab m, int m_begin, int m_end,
                                                         int64_t tile_begin, const float* x, double* partial2) {
  __shared__ float sm[kSub][kSub + 1];
  const int64_t tile = tile_begin + blockIdx.x;
  const TileRef tr = tile_ref(m, m_begin, m_end, tile);
  const int tensor = m.tensor[tr.mat];
  const float* p = t.p32[tensor];
  float o[kRowsPerThread];
  double pp = 0.0, oo = 0.0;
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    load_x_tile(st, x, sm, o);
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      if (r < st.out && c < st.in) {
        pp += sq(p[stor(t, m, tensor, st.mat, r, c, st.in)]);
        oo += sq(o[j]);
      }
    }
  }
  pp = block_sum(pp);
  oo = block_sum(oo);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    partial2[2 * tile] = pp;
    partial2[2 * tile + 1] = oo;
  }
}

__global__ void __launch_bounds__(kSub* kTy) k_hb_w(TensorTab t, MatTab m, const GroupDev* groups, Tables tb,
                                                     int m_begin, int m_end, int64_t tile_begin, float* x,
                                                     const double* mat_po, double* partial) {
  __shared__ float sm[kSub][kSub + 1];
  const int64_t tile = tile_begin + blockIdx.x;
  const TileRef tr = tile_ref(m, m_begin, m_end, tile);
  const int tensor = m.tensor[tr.mat];
  const GroupDev gr = groups[t.group[tensor]];
  const float eta = mul(tb.lr[gr.lr_table][cur_step(tb)], gr.lr_scale);
  const float n = norm_of(mat_po[2 * tr.mat]);
  const float mfl = fmax_nan(norm_of(mat_po[2 * tr.mat + 1]), kNormFloor);
  const float* p = t.p32[tensor];
  float o[kRowsPerThread];
  double ww = 0.0;
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    load_x_tile(st, x, sm, o);
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      if (r < st.out && c < st.in) {
        // param - learning_rate * update * param_norm / jnp.maximum(update_norm, _NORM_FLOOR), left to right
        const float w = sub(p[stor(t, m, tensor, st.mat, r, c, st.in)], div(mul(mul(eta, o[j]), n), mfl));
        o[j] = w;
        ww += sq(w);
      }
    }
    store_x_tile(st, x, sm, o);
  }
  ww = block_sum(ww);
  if (threadIdx.x == 0 && threadIdx.y == 0) partial[tile] = ww;
}

__global__ void __launch_bounds__(kSub* kTy) k_hb_apply(TensorTab t, MatTab m, int m_begin, int m_end,
                                                         int64_t tile_begin, const float* x, const double* mat_po,
                                                         const double* mat_w) {
  __shared__ float sm[kSub][kSub + 1];
  const int64_t tile = tile_begin + blockIdx.x;
  const TileRef tr = tile_ref(m, m_begin, m_end, tile);
  const int tensor = m.tensor[tr.mat];
  const float n = norm_of(mat_po[2 * tr.mat]);
  const float nw = fmax_nan(norm_of(mat_w[tr.mat]), kNormFloor);
  const float us = t.update_scale[tensor];
  float* p = t.p32[tensor];
  float* uo = t.u_out[tensor];
  __nv_bfloat16* p16 = reinterpret_cast<__nv_bfloat16*>(t.p16[tensor]);
  float w[kRowsPerThread];
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    load_x_tile(st, x, sm, w);
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      if (r < st.out && c < st.in) {
        const int64_t e = stor(t, m, tensor, st.mat, r, c, st.in);
        const float pi = p[e];
        // new_param / jnp.maximum(new_param_norm, _NORM_FLOOR) * param_norm - param; the caller adds p + u
        const float u = mul(sub(mul(div(w[j], nw), n), pi), us);
        const float pn = add(pi, u);
        p[e] = pn;
        if (uo) uo[e] = u;
        if (p16) p16[e] = __float2bfloat16_rn(pn);
      }
    }
  }
}

// Standalone NS: tile (blockIdx.x) of matrix blockIdx.y of a dense [batch, rows, cols] array.
__device__ __forceinline__ TileRef dense_tile(int rows, int cols) {
  TileRef t;
  t.mat = blockIdx.y;
  t.out = rows;
  t.in = cols;
  const int tiles_c = (cols + kTile - 1) / kTile;
  t.r0 = static_cast<int>(blockIdx.x / tiles_c) * kTile;
  t.c0 = static_cast<int>(blockIdx.x % tiles_c) * kTile;
  t.trans = rows > cols;
  t.x_off = int64_t(blockIdx.y) * rows * cols;
  return t;
}

__global__ void __launch_bounds__(kSub* kTy) k_ns_load(const float* src, float* x, int rows, int cols,
                                                        double* partial) {
  __shared__ float sm[kSub][kSub + 1];
  const TileRef tr = dense_tile(rows, cols);
  float v[kRowsPerThread];
  double acc = 0.0;
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      v[j] = (r < rows && c < cols) ? src[st.x_off + int64_t(r) * cols + c] : 0.0f;
      acc += sq(v[j]);
    }
    store_x_tile(st, x, sm, v);
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0 && threadIdx.y == 0) partial[int64_t(blockIdx.y) * gridDim.x + blockIdx.x] = acc;
}

__global__ void __launch_bounds__(kSub* kTy) k_ns_store(const float* x, float* dst, int rows, int cols) {
  __shared__ float sm[kSub][kSub + 1];
  const TileRef tr = dense_tile(rows, cols);
  float v[kRowsPerThread];
  for (int s = 0; s < kSubs; ++s) {
    const TileRef st = sub_tile(tr, s);
    if (!valid(st)) continue;
    load_x_tile(st, x, sm, v);
    for (int j = 0; j < kRowsPerThread; ++j) {
      const int r = st.r0 + threadIdx.y + kTy * j, c = st.c0 + threadIdx.x;
      if (r < rows && c < cols) dst[st.x_off + int64_t(r) * cols + c] = v[j];
    }
  }
}

int grid_for(int64_t n) {
  const int64_t b = (n + 255) / 256;
  return static_cast<int>(b < 65535 * 16 ? b : 65535 * 16);
}

}  // namespace

void muon_momentum(const TensorTab& t, const MatTab& m, const GroupDev* groups, const ClipDev* clip, int m_begin,
                   int m_end, int64_t tile_begin, int64_t tile_end, float* x, double* partial, cudaStream_t st) {
  if (tile_end <= tile_begin) return;
  k_muon_momentum<<<static_cast<unsigned>(tile_end - tile_begin), dim3(kSub, kTy), 0, st>>>(
      t, m, groups, clip, m_begin, m_end, tile_begin, x, partial);
  NM_CUDA(cudaGetLastError());
}

void ns_normalize(float* x, int count, int64_t mn, const double* u2, float eps, cudaStream_t st) {
  if (count <= 0 || mn <= 0) return;
  const int64_t per = (mn / 4 + 255) / 256;
  k_ns_normalize<<<dim3(static_cast<unsigned>(per < 4096 ? per : 4096), count), 256, 0, st>>>(x, mn, u2, eps);
  NM_CUDA(cudaGetLastError());
}

void ns_scale(float* dst, const float* src, float c, int64_t n, cudaStream_t st) {
  if (n <= 0) return;
  k_ns_scale<<<grid_for(n / 4 + 1), 256, 0, st>>>(dst, src, c, n);
  NM_CUDA(cudaGetLastError());
}

void ns_mirror(float* d, int count, int m, int m1, cudaStream_t st) {
  if (count <= 0 || m1 <= 0 || m1 >= m) return;
  const unsigned tiles = ((m - m1 + kSub - 1) / kSub) * ((m1 + kSub - 1) / kSub);
  k_ns_mirror<<<dim3(tiles, count), dim3(kSub, kTy), 0, st>>>(d, m, m1);
  NM_CUDA(cudaGetLastError());
}

void tf32_split(const float* x, float* big, float* small, int64_t n, cudaStream_t st) {
  if (n <= 0) return;
  k_tf32_split<<<grid_for(n), 256, 0, st>>>(x, big, small, n);
  NM_CUDA(cudaGetLastError());
}

void ns_load(const float* src, float* x, int batch, int rows, int cols, double* partial, cudaStream_t st) {
  const unsigned tiles = ((rows + kTile - 1) / kTile) * ((cols + kTile - 1) / kTile);
  k_ns_load<<<dim3(tiles, batch), dim3(kSub, kTy), 0, st>>>(src, x, rows, cols, partial);
  NM_CUDA(cudaGetLastError());
}

void ns_store(const float* x, float* dst, int batch, int rows, int cols, cudaStream_t st) {
  const unsigned tiles = ((rows + kTile - 1) / kTile) * ((cols + kTile - 1) / kTile);
  k_ns_store<<<dim3(tiles, batch), dim3(kSub, kTy), 0, st>>>(x, dst, rows, cols);
  NM_CUDA(cudaGetLastError());
}

void hyperball_norms(const TensorTab& t, const MatTab& m, int m_begin, int m_end, int64_t tile_begin,
                     int64_t tile_end, const float* x, double* partial2, cudaStream_t st) {
  if (tile_end <= tile_begin) return;
  k_hb_norms<<<static_cast<unsigned>(tile_end - tile_begin), dim3(kSub, kTy), 0, st>>>(t, m, m_begin, m_end,
                                                                                         tile_begin, x, partial2);
  NM_CUDA(cudaGetLastError());
}

void hyperball_w(const TensorTab& t, const MatTab& m, const GroupDev* groups, const Tables& tb, int m_begin,
                 int m_end, int64_t tile_begin, int64_t tile_end, float* x, const double* mat_po, double* partial,
                 cudaStream_t st) {
  if (tile_end <= tile_begin) return;
  k_hb_w<<<static_cast<unsigned>(tile_end - tile_begin), dim3(kSub, kTy), 0, st>>>(t, m, groups, tb, m_begin, m_end,
                                                                                     tile_begin, x, mat_po, partial);
  NM_CUDA(cudaGetLastError());
}

void hyperball_apply(const TensorTab& t, const MatTab& m, int m_begin, int m_end, int64_t tile_begin,
                     int64_t tile_end, const float* x, const double* mat_po, const double* mat_w, cudaStream_t st) {
  if (tile_end <= tile_begin) return;
  k_hb_apply<<<static_cast<unsigned>(tile_end - tile_begin), dim3(kSub, kTy), 0, st>>>(t, m, m_begin, m_end,
                                                                                         tile_begin, x, mat_po, mat_w);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::optk
