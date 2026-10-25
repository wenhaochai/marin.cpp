// Deterministic reductions of the optimizer: clip sums of squares per slice, fixed-order segment sums, clip state,
// step advance. optim.md §3.1; DESIGN.md §4.1, §11.
#include "kernels/optim/optk_common.cuh"

namespace nm::optk {
namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) k_sumsq_slices(TensorTab t, SliceTab s, RowPartTab rp, Tables tb,
                                                           int s_begin, double* partial) {
  const int si = s_begin + blockIdx.x;
  const dev::Slice sl = dev::slice_at(s, si);
  double acc = 0.0;
  if (!dev::block_frozen(rp, sl.block, dev::cur_step(tb))) {
    const float* g = t.g[sl.tensor] + dev::slice_off(t, sl);
    const float gs = t.grad_scale[sl.tensor];
    if (sl.len % 4 == 0 && dev::al16(g)) {
      for (int q = threadIdx.x; q < sl.len / 4; q += kThreads) {
        dev::F4 G;
        G.load(g, q);
        for (int j = 0; j < 4; ++j) acc += dev::sq(dev::mul(G.v[j], gs));
      }
    } else {
      for (int i = threadIdx.x; i < sl.len; i += kThreads) acc += dev::sq(dev::mul(g[i], gs));
    }
  }
  acc = dev::block_sum(acc);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

// One block per segment; each thread sums a contiguous run of the segment in index order, then the runs are summed
// in thread order (block_sum's order is fixed for a given blockDim). The order depends only on the segment length.
__global__ void __launch_bounds__(kThreads) k_segment_sum(const double* partial, const int32_t* idx, const int64_t* seg,
                                                          int stride, double* out, int accumulate) {
  const int i = blockIdx.x, c = blockIdx.y;
  const int64_t b = seg[i], e = seg[i + 1], n = e - b;
  const int64_t per = (n + kThreads - 1) / kThreads;
  const int64_t lo = b + per * threadIdx.x, hi = lo + per < e ? lo + per : e;
  double acc = 0.0;
  for (int64_t j = lo; j < hi; ++j) acc += partial[(idx ? idx[j] : j) * stride + c];
  acc = dev::block_sum(acc);
  if (threadIdx.x == 0) out[int64_t(i) * stride + c] = accumulate ? out[int64_t(i) * stride + c] + acc : acc;
}

__global__ void k_clip_state(const double* clip_sumsq, const float* max_norm, int n, ClipDev* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gn = dev::norm_of(clip_sumsq[i]);
  const float mx = max_norm[i];
  const int32_t clip = !(gn < mx);
  out[i] = ClipDev{gn, clip, mx, clip ? dev::div(mx, gn) : 1.0f};
}

__global__ void k_step_advance(int32_t* step, int32_t n_steps, const float* lr_main, const float* lr_adam,
                               float* ring) {
  const int32_t k = *step;
  if (k < 0 || k >= n_steps) __trap();
  if (ring) {
    ring[0] = lr_main ? lr_main[k] : 0.0f;
    ring[1] = lr_adam ? lr_adam[k] : 0.0f;
  }
  *step = k + 1;
}

}  // namespace

void sumsq_slices(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const Tables& tb, int s_begin,
                  int s_end, double* partial, cudaStream_t st) {
  if (s_end <= s_begin) return;
  k_sumsq_slices<<<s_end - s_begin, kThreads, 0, st>>>(t, s, rp, tb, s_begin, partial);
  NM_CUDA(cudaGetLastError());
}

void segment_sum(const double* partial, const int32_t* idx, const int64_t* seg, int n_seg, int stride, double* out,
                 int accumulate, cudaStream_t st) {
  if (n_seg <= 0) return;
  k_segment_sum<<<dim3(n_seg, stride), kThreads, 0, st>>>(partial, idx, seg, stride, out, accumulate);
  NM_CUDA(cudaGetLastError());
}

void clip_state(const double* clip_sumsq, const float* max_norm, int n, ClipDev* out, cudaStream_t st) {
  if (n <= 0) return;
  k_clip_state<<<(n + 127) / 128, 128, 0, st>>>(clip_sumsq, max_norm, n, out);
  NM_CUDA(cudaGetLastError());
}

void step_advance(int32_t* step, int32_t n_steps, const float* lr_main, const float* lr_adam, float* ring,
                  cudaStream_t st) {
  k_step_advance<<<1, 1, 0, st>>>(step, n_steps, lr_main, lr_adam, ring);
  NM_CUDA(cudaGetLastError());
}

}  // namespace nm::optk
