// C ABI of the elementwise kernels (capi.h "elementwise" block and capi_ew.h). Test path: gain gradients are reduced
// here from partials allocated with cudaMallocAsync; the trainer calls the C++ entry points with planned buffers.
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "core/capi_util.hpp"
#include "kernels/elementwise/capi_ew.h"
#include "kernels/elementwise/ew_api.hpp"
#include "marincpp/capi.h"

namespace nm {
namespace {

cudaStream_t S(void* p) { return static_cast<cudaStream_t>(p); }

DType storage(int dtype) {
  if (dtype == 0) return DType::f32;
  if (dtype == 1) return DType::bf16;
  fatal(NM_WHERE, "dtype must be 0 (f32 storage) or 1 (bf16 storage), got " + std::to_string(dtype));
}
Rounding rounding(int r) {
  if (r == 0) return Rounding::Levanter;
  if (r == 1) return Rounding::Accurate;
  fatal(NM_WHERE, "rounding must be 0 (levanter) or 1 (accurate), got " + std::to_string(r));
}
template <class T> bf16_t* Bm(T* p) { return reinterpret_cast<bf16_t*>(p); }
inline const bf16_t* Bv(const void* p) { return static_cast<const bf16_t*>(p); }
inline bf16_t* Bvm(void* p) { return static_cast<bf16_t*>(p); }

// Partial buffers of one call: allocated and freed stream-ordered from the device's default pool, which keeps its
// memory (release threshold = max) so a repeated call does not go back to the driver.
void keep_pool_memory() {
  thread_local int done_dev = -1;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  if (done_dev == dev) return;
  cudaMemPool_t pool;
  NM_CUDA(cudaDeviceGetDefaultMemPool(&pool, dev));
  uint64_t thr = UINT64_MAX;
  NM_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  done_dev = dev;
}
struct Partials {
  float* p = nullptr;
  cudaStream_t s;
  Partials(int64_t rows, int width, int count, cudaStream_t st) : s(st) {
    keep_pool_memory();
    NM_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&p), size_t(rows) * width * count * sizeof(float), s));
  }
  ~Partials() {
    if (p) cudaFreeAsync(p, s);
  }
  Partials(const Partials&) = delete;
  Partials& operator=(const Partials&) = delete;
  float* at(int64_t rows, int width, int i) const { return p + int64_t(i) * rows * width; }
};

void reduce_into(const float* partial, int64_t rows, int width, float* out, bool round_bf16, cudaStream_t s) {
  if (out) reduce_partials_rows(partial, static_cast<int>(rows), width, out, false, round_bf16, s);
}

}  // namespace
}  // namespace nm

using nm::capi_call;

extern "C" {

int nm_doc_meta(const int32_t* tokens, const uint8_t* row_valid, int rows, int S, int32_t eos, int max_segs,
                int32_t* targets, float* weight, int32_t* pos, int32_t* cu_seqlens, int32_t* n_segs,
                int32_t* row_seg_start, uint32_t* overflow, void* stream) {
  return capi_call([&] {
    nm::DocMetaArgs a{tokens, row_valid, rows, S, eos, max_segs, targets, weight, pos, cu_seqlens, n_segs,
                      row_seg_start, overflow};
    nm::doc_meta(a, nm::S(stream));
  });
}

int nm_embed_fwd(const void* table, const int32_t* tokens, void* out, int64_t T, int D, void* stream) {
  return capi_call([&] { nm::ew::embed_fwd_dt(nm::DType::bf16, table, tokens, out, T, D, nm::S(stream)); });
}
int nm_embed_fwd_dt(int dtype, const void* table, const int32_t* tokens, void* out, int64_t T, int D, void* stream) {
  return capi_call([&] { nm::ew::embed_fwd_dt(nm::storage(dtype), table, tokens, out, T, D, nm::S(stream)); });
}

uint64_t nm_embed_bwd_workspace_bytes(int64_t T, int V) {
  uint64_t b = 0;
  if (capi_call([&] { b = nm::embed_bwd_workspace_bytes(T, V); }) != NM_OK) return 0;
  return b;
}

int nm_embed_bwd_dt(int dtype, const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                    int accum, uint8_t* touched_rows, void* workspace, uint64_t workspace_bytes, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(accum == 0 || accum == 1, "accum must be 0 (f32) or 1 (bf16 sequential)");
    const nm::EmbedBwdWorkspace ws = nm::ew::embed_bwd_carve(workspace, workspace_bytes, T, V);
    nm::ew::embed_bwd_sorted_dt(nm::storage(dtype), dout, tokens, dtable, T, D, V,
                                accum ? nm::EmbedAccum::Bf16Sequential : nm::EmbedAccum::F32, ws, touched_rows,
                                nm::S(stream));
  });
}
int nm_embed_bwd(const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V, int accum,
                 uint8_t* touched_rows, void* workspace, uint64_t workspace_bytes, void* stream) {
  return nm_embed_bwd_dt(1, dout, tokens, dtable, T, D, V, accum, touched_rows, workspace, workspace_bytes, stream);
}

int nm_rmsnorm_fwd(int dtype, int rounding, const void* x, const void* gain, void* y, float* rstd, int64_t T, int D,
                   float eps, void* stream) {
  return capi_call([&] {
    nm::RmsNormFwd a{nm::Bv(x), nm::Bv(gain), nm::Bvm(y), rstd, T, D, eps};
    nm::ew::rmsnorm_fwd_dt(nm::storage(dtype), a, nm::rounding(rounding), nm::S(stream));
  });
}

int nm_rmsnorm_bwd_add_partial(int dtype, int rounding, const void* dy, const void* x, const void* gain,
                               const float* rstd, const void* dres, void* dx, float* dgain_partial, int64_t T, int D,
                               float eps, void* stream) {
  return capi_call([&] {
    nm::RmsNormBwdAdd a{nm::Bv(dy), nm::Bv(x), nm::Bv(gain), rstd, nm::Bv(dres), nm::Bvm(dx), dgain_partial, T, D, eps};
    nm::ew::rmsnorm_bwd_add_dt(nm::storage(dtype), a, nm::rounding(rounding), nm::S(stream));
  });
}

int nm_rmsnorm_bwd_add(int dtype, int rounding, const void* dy, const void* x, const void* gain, const float* rstd,
                       const void* dres, void* dx, float* dgain, int64_t T, int D, float eps, void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    const int64_t rows = nm::ew::norm_partial_rows(T);
    nm::Partials part(dgain ? rows : 0, D, 1, s);
    nm::RmsNormBwdAdd a{nm::Bv(dy), nm::Bv(x), nm::Bv(gain), rstd, nm::Bv(dres), nm::Bvm(dx),
                        dgain ? part.p : nullptr, T, D, eps};
    const nm::Rounding r = nm::rounding(rounding);
    nm::ew::rmsnorm_bwd_add_dt(nm::storage(dtype), a, r, s);
    nm::reduce_into(part.p, rows, D, dgain, r == nm::Rounding::Levanter && dtype == 1, s);
  });
}

int nm_sandwich_tail_fwd(int dtype, int rounding, const void* branch, const void* x_in, const void* g_post,
                         const void* g_next, void* x_out, void* a_next, float* rstd_post, float* rstd_next, int64_t T,
                         int D, float eps, void* stream) {
  return capi_call([&] {
    nm::SandwichTailFwd a{nm::Bv(branch), nm::Bv(x_in), nm::Bv(g_post), nm::Bv(g_next), nm::Bvm(x_out),
                          nm::Bvm(a_next), rstd_post, rstd_next, T, D, eps};
    nm::ew::sandwich_tail_fwd_dt(nm::storage(dtype), a, nm::rounding(rounding), nm::S(stream));
  });
}

int nm_sandwich_tail_bwd(int dtype, int rounding, const void* dx_out, const void* da_next, const void* x_out,
                         const void* g_next, const float* rstd_next, const void* branch, const void* g_post,
                         const float* rstd_post, void* dx_in, void* dbranch, float* dg_next, float* dg_post,
                         int64_t T, int D, float eps, void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    const int64_t rows = nm::ew::norm_partial_rows(T);
    nm::Partials part(rows, D, 2, s);
    nm::SandwichTailBwd a{nm::Bv(dx_out), nm::Bv(da_next), nm::Bv(x_out), nm::Bv(g_next), rstd_next,
                          nm::Bv(branch), nm::Bv(g_post),  rstd_post,     nm::Bvm(dx_in), nm::Bvm(dbranch),
                          part.at(rows, D, 0), part.at(rows, D, 1), T, D, eps};
    const nm::Rounding r = nm::rounding(rounding);
    nm::ew::sandwich_tail_bwd_dt(nm::storage(dtype), a, r, s);
    const bool rb = r == nm::Rounding::Levanter && dtype == 1;
    nm::reduce_into(part.at(rows, D, 0), rows, D, dg_next, rb, s);
    if (g_post) nm::reduce_into(part.at(rows, D, 1), rows, D, dg_post, rb, s);
  });
}

int nm_qknorm_rope_fwd(int dtype, int rounding, const void* qkv, int64_t qkv_ld, const void* gq, const void* gk,
                       const void* cos_tab, const void* sin_tab, const int32_t* pos, void* q_rot, void* k_rot,
                       float* rstd_q, float* rstd_k, int64_t T, int H, int KV, int hd, float eps, float q_scale,
                       void* stream) {
  return capi_call([&] {
    nm::QkNormRopeFwd a{nm::Bv(qkv),   qkv_ld, nm::Bv(gq), nm::Bv(gk), nm::Bv(cos_tab), nm::Bv(sin_tab),
                        pos,           nm::Bvm(q_rot), nm::Bvm(k_rot), rstd_q, rstd_k, T, H, KV, hd, eps, q_scale};
    nm::ew::qknorm_rope_fwd_dt(nm::storage(dtype), a, nm::rounding(rounding), nm::S(stream));
  });
}

int nm_qknorm_rope_bwd(int dtype, int rounding, const void* dq_rot, const void* dk_rot, const void* qkv,
                       int64_t qkv_ld, const void* gq, const void* gk, const void* cos_tab, const void* sin_tab,
                       const int32_t* pos, const float* rstd_q, const float* rstd_k, void* dqkv, int64_t dqkv_ld,
                       float* dgq, float* dgk, int64_t T, int H, int KV, int hd, float eps, float q_scale,
                       void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    const int64_t rows = nm::ew::norm_partial_rows(T);
    nm::Partials part(rows, hd, 2, s);
    nm::QkNormRopeBwd a{nm::Bv(dq_rot), nm::Bv(dk_rot), nm::Bv(qkv), qkv_ld, nm::Bv(gq), nm::Bv(gk),
                        nm::Bv(cos_tab), nm::Bv(sin_tab), pos, rstd_q, rstd_k, nm::Bvm(dqkv), dqkv_ld,
                        part.at(rows, hd, 0), part.at(rows, hd, 1), T, H, KV, hd, eps, q_scale};
    const nm::Rounding r = nm::rounding(rounding);
    nm::ew::qknorm_rope_bwd_dt(nm::storage(dtype), a, r, s);
    const bool rb = r == nm::Rounding::Levanter && dtype == 1;
    if (dq_rot) nm::reduce_into(part.at(rows, hd, 0), rows, hd, dgq, rb, s);
    if (dk_rot) nm::reduce_into(part.at(rows, hd, 1), rows, hd, dgk, rb, s);
  });
}

int nm_dq_post_qknorm_rope_bwd(int rounding, const float* dq_accum, const void* qkv, int64_t qkv_ld, const void* gq,
                               const void* cos_tab, const void* sin_tab, const int32_t* pos, const float* rstd_q,
                               void* dqkv, int64_t dqkv_ld, float* dgq, int64_t T, int H, int hd, float eps,
                               float q_scale, void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    const int64_t rows = nm::ew::norm_partial_rows(T);
    nm::Partials part(rows, hd, 1, s);
    nm::DqPostQkNormRopeBwd a{dq_accum, nm::Bv(qkv), qkv_ld, nm::Bv(gq), nm::Bv(cos_tab), nm::Bv(sin_tab), pos,
                              rstd_q, nm::Bvm(dqkv), dqkv_ld, part.p, T, H, hd, eps, q_scale};
    const nm::Rounding r = nm::rounding(rounding);
    nm::dq_post_qknorm_rope_bwd(a, r, s);
    nm::reduce_into(part.p, rows, hd, dgq, r == nm::Rounding::Levanter, s);
  });
}

int nm_rope_tables(int S_max, int hd, double theta, uint16_t* cos_host, uint16_t* sin_host, float* inv_freq_host) {
  return capi_call([&] {
    nm::build_rope_tables(S_max, hd, theta, nm::Bm(cos_host), nm::Bm(sin_host), inv_freq_host);
  });
}
int nm_rope_tables_method(int method, int S_max, int hd, double theta, uint16_t* cos_host, uint16_t* sin_host,
                          float* inv_freq_host) {
  return capi_call([&] {
    NM_REQUIRE(method == 0 || method == 1, "method must be 0 (host double) or 1 (device libdevice)");
    nm::ew::build_rope_tables_method(static_cast<nm::ew::RopeMethod>(method), S_max, hd, theta, nm::Bm(cos_host),
                                     nm::Bm(sin_host), inv_freq_host);
  });
}

int nm_swiglu_fwd(int dtype, int rounding, const void* gu, void* h, int64_t T, int F, void* stream) {
  return capi_call([&] {
    nm::ew::swiglu_fwd_dt(nm::storage(dtype), gu, h, T, F, nm::rounding(rounding), nm::S(stream));
  });
}
int nm_swiglu_bwd(int dtype, int rounding, const void* dh, const void* gu, void* dgu, int64_t T, int F, void* stream) {
  return capi_call([&] {
    nm::ew::swiglu_bwd_dt(nm::storage(dtype), dh, gu, dgu, T, F, nm::rounding(rounding), nm::S(stream));
  });
}

int64_t nm_norm_partial_rows(int64_t T) { return nm::ew::norm_partial_rows(T); }

int nm_rope_fwd(int dtype, int rounding, const void* x, int64_t ld, const void* cos_tab, const void* sin_tab,
                const int32_t* pos, void* out, int64_t T, int nheads, int hd, float scale, void* stream) {
  return capi_call([&] {
    nm::ew::rope_fwd_dt(nm::storage(dtype), x, ld, cos_tab, sin_tab, pos, out, T, nheads, hd, scale,
                        nm::rounding(rounding), nm::S(stream));
  });
}

int nm_reduce_partials_rows(const float* partial, int n_blocks, int width, float* out, int accumulate, int round_bf16,
                            void* stream) {
  return capi_call([&] {
    nm::reduce_partials_rows(partial, n_blocks, width, out, accumulate != 0, round_bf16 != 0, nm::S(stream));
  });
}
int nm_sum_fixed(const float* x, int64_t n, float* out, void* stream) {
  return capi_call([&] { nm::sum_fixed(x, n, out, nm::S(stream)); });
}

int nm_cast_f32_bf16(const float* src, void* dst, int64_t n, void* stream) {
  return capi_call([&] {
    const cudaStream_t s = nm::S(stream);
    struct List { const float* src; nm::bf16_t* dst; int64_t n; } h{src, static_cast<nm::bf16_t*>(dst), n};
    void* d = nullptr;
    NM_CUDA(cudaMallocAsync(&d, sizeof(List), s));
    NM_CUDA(cudaMemcpyAsync(d, &h, sizeof(List), cudaMemcpyHostToDevice, s));
    List* dl = static_cast<List*>(d);
    const nm::CastList l{reinterpret_cast<const float* const*>(&dl->src), reinterpret_cast<nm::bf16_t* const*>(&dl->dst),
                         &dl->n, 1};
    nm::cast_f32_bf16_multi(l, s);
    NM_CUDA(cudaStreamSynchronize(s));   // h is a host stack object: the copy must finish before it goes away
    NM_CUDA(cudaFreeAsync(d, s));
  });
}

}  // extern "C"
