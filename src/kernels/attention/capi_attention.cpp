// C ABI of the attention backends (capi.h "attention" block and capi_attn.h).
#include <cuda_runtime.h>

#include <string>

#include "core/capi_util.hpp"
#include "kernels/attention/attn_internal.hpp"
#include "kernels/attention/capi_attn.h"
#include "marincpp/capi.h"
#include "marincpp/device.hpp"

namespace nm {
namespace {

cudaStream_t S(void* p) { return static_cast<cudaStream_t>(p); }

AttnBackendKind backend_of(int b) {
  switch (b) {
    case 0: return AttnBackendKind::FA3;
    case 1: return AttnBackendKind::CudnnSdpa;
    case 2: return AttnBackendKind::LevRef;
    case 3: return AttnBackendKind::RefF32;
  }
  fatal(NM_WHERE, "attention backend must be 0 fa3, 1 cudnn, 2 levref or 3 ref_f32, got " + std::to_string(b));
}

AttnBwdSched sched_of(int s) {
  if (s == 0) return AttnBwdSched::Persistent;
  if (s == 1) return AttnBwdSched::DocCausal;
  fatal(NM_WHERE, "bwd_sched must be 0 (persistent) or 1 (doc_causal), got " + std::to_string(s));
}

Fa3Mode mode_of(int m) {
  NM_REQUIRE(m >= 0 && m <= 4, "FA3 mode must be 0..4, got " + std::to_string(m));
  return static_cast<Fa3Mode>(m);
}

AttnShape shape_of(int64_t total_q, int rows, int S_, int H, int KV, int hd, int max_segs, AttnBwdSched sched,
                   bool det, DType io) {
  AttnShape s{};
  s.total_q = total_q; s.rows = rows; s.S = S_; s.H = H; s.KV = KV; s.hd = hd; s.max_segs_per_seq = max_segs;
  s.bwd_sched = sched; s.deterministic = det; s.io = io;
  return s;
}

AttnFwdArgs fwd_args(const void* q, int64_t q_ld, const void* k, int64_t k_ld, const void* v, int64_t v_ld, void* o,
                     int64_t o_ld, float* lse, const int32_t* cu, const int32_t* n_segs, const int32_t* rss,
                     float scale) {
  AttnFwdArgs a{};
  a.q = q; a.q_ld = q_ld; a.k = k; a.k_ld = k_ld; a.v = v; a.v_ld = v_ld; a.o = o; a.o_ld = o_ld; a.lse = lse;
  a.cu_seqlens = cu; a.n_segs = n_segs; a.row_seg_start = rss; a.softmax_scale = scale;
  return a;
}

// The C ABI attention entry points take bf16 storage for FA3 and f32 or bf16 for ref_f32: the reference reads the
// storage type from the backend (3 = ref_f32 with bf16 storage; the f32 policy calls the C++ entry with io = f32).
DType io_of(AttnBackendKind) { return DType::bf16; }

std::unique_ptr<AttentionBackend> backend(AttnBackendKind k) {
  DeviceContext ctx;
  return make_attention(k, ctx);
}

}  // namespace
}  // namespace nm

using namespace nm;

extern "C" {

uint64_t nm_attention_workspace_bytes(int backend_id, int bwd_sched, int64_t total_q, int rows, int S_, int H, int KV,
                                      int hd, int max_segs_per_seq, int training) {
  uint64_t out = 0;
  int rc = capi_call([&] {
    const AttnBackendKind k = backend_of(backend_id);
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq,
                                 k == AttnBackendKind::FA3 ? sched_of(bwd_sched) : AttnBwdSched::DocCausal, true,
                                 io_of(k));
    out = backend(k)->workspace_bytes(s, training != 0);
  });
  return rc == NM_OK ? out : UINT64_MAX;
}

int nm_attention_fwd(int backend_id, int bwd_sched, const void* q, int64_t q_ld, const void* k, int64_t k_ld,
                     const void* v, int64_t v_ld, void* o, int64_t o_ld, float* lse, const int32_t* cu_seqlens,
                     const int32_t* n_segs, const int32_t* row_seg_start, int64_t total_q, int rows, int S_, int H,
                     int KV, int hd, int max_segs_per_seq, float softmax_scale, void* workspace, void* stream) {
  return capi_call([&] {
    const AttnBackendKind kind = backend_of(backend_id);
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq,
                                 kind == AttnBackendKind::FA3 ? sched_of(bwd_sched) : AttnBwdSched::DocCausal, true,
                                 io_of(kind));
    auto b = backend(kind);
    b->prepare(s, false);
    b->fwd(s, fwd_args(q, q_ld, k, k_ld, v, v_ld, o, o_ld, lse, cu_seqlens, n_segs, row_seg_start, softmax_scale),
           workspace, S(stream));
  });
}

int nm_attention_bwd(int backend_id, int bwd_sched, const void* q, int64_t q_ld, const void* k, int64_t k_ld,
                     const void* v, int64_t v_ld, const void* o, int64_t o_ld, const float* lse, const void* d_o,
                     int64_t do_ld, const float* softmax_d, float* dq_accum, void* dq, int64_t dq_ld, void* dk,
                     int64_t dk_ld, void* dv, int64_t dv_ld, const int32_t* cu_seqlens, const int32_t* n_segs,
                     const int32_t* row_seg_start, int64_t total_q, int rows, int S_, int H, int KV, int hd,
                     int max_segs_per_seq, float softmax_scale, int deterministic, void* workspace, void* stream) {
  return capi_call([&] {
    const AttnBackendKind kind = backend_of(backend_id);
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq,
                                 kind == AttnBackendKind::FA3 ? sched_of(bwd_sched) : AttnBwdSched::DocCausal,
                                 deterministic != 0, io_of(kind));
    AttnBwdArgs a{};
    a.fwd = fwd_args(q, q_ld, k, k_ld, v, v_ld, const_cast<void*>(o), o_ld, const_cast<float*>(lse), cu_seqlens,
                     n_segs, row_seg_start, softmax_scale);
    a.d_o = d_o; a.do_ld = do_ld; a.softmax_d = softmax_d; a.dq_accum = dq_accum;
    a.dq = dq; a.dq_ld = dq_ld; a.dk = dk; a.dk_ld = dk_ld; a.dv = dv; a.dv_ld = dv_ld;
    auto b = backend(kind);
    b->prepare(s, true);
    b->bwd(s, a, workspace, S(stream));
  });
}

uint64_t nm_fa3_mode_workspace_bytes(int mode, int64_t total_q, int rows, int S_, int H, int KV, int hd,
                                     int max_segs_per_seq, int training) {
  uint64_t out = 0;
  int rc = capi_call([&] {
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq, AttnBwdSched::Persistent, true,
                                 DType::bf16);
    out = fa3k::workspace_bytes(s, mode_of(mode), training != 0);
  });
  return rc == NM_OK ? out : UINT64_MAX;
}

int nm_fa3_mode_fwd(int mode, int exact_n_segs, int exact_max_seqlen, const void* q, int64_t q_ld, const void* k,
                    int64_t k_ld, const void* v, int64_t v_ld, void* o, int64_t o_ld, float* lse,
                    const int32_t* cu_seqlens, const int32_t* n_segs, const int32_t* row_seg_start, int64_t total_q,
                    int rows, int S_, int H, int KV, int hd, int max_segs_per_seq, float softmax_scale,
                    void* workspace, void* stream) {
  return capi_call([&] {
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq, AttnBwdSched::Persistent, true,
                                 DType::bf16);
    const Fa3Exact ex{exact_n_segs, exact_max_seqlen};
    fa3k::fwd(s, fwd_args(q, q_ld, k, k_ld, v, v_ld, o, o_ld, lse, cu_seqlens, n_segs, row_seg_start, softmax_scale),
              mode_of(mode), &ex, workspace, S(stream));
  });
}

int nm_fa3_mode_bwd(int mode, int exact_n_segs, int exact_max_seqlen, const void* q, int64_t q_ld, const void* k,
                    int64_t k_ld, const void* v, int64_t v_ld, const void* o, int64_t o_ld, const float* lse,
                    const void* d_o, int64_t do_ld, void* dq, int64_t dq_ld, void* dk, int64_t dk_ld, void* dv,
                    int64_t dv_ld, const int32_t* cu_seqlens, const int32_t* n_segs, const int32_t* row_seg_start,
                    int64_t total_q, int rows, int S_, int H, int KV, int hd, int max_segs_per_seq, float softmax_scale,
                    int deterministic, void* workspace, void* stream) {
  return capi_call([&] {
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq, AttnBwdSched::Persistent,
                                 deterministic != 0, DType::bf16);
    const Fa3Exact ex{exact_n_segs, exact_max_seqlen};
    AttnBwdArgs a{};
    a.fwd = fwd_args(q, q_ld, k, k_ld, v, v_ld, const_cast<void*>(o), o_ld, const_cast<float*>(lse), cu_seqlens,
                     n_segs, row_seg_start, softmax_scale);
    a.d_o = d_o; a.do_ld = do_ld; a.dq = dq; a.dq_ld = dq_ld; a.dk = dk; a.dk_ld = dk_ld; a.dv = dv; a.dv_ld = dv_ld;
    fa3k::bwd(s, a, mode_of(mode), &ex, workspace, S(stream));
  });
}

int nm_ref_attention_fwd(int dtype, const void* q, int64_t q_ld, const void* k, int64_t k_ld, const void* v,
                         int64_t v_ld, void* o, int64_t o_ld, float* lse, const int32_t* cu_seqlens,
                         const int32_t* row_seg_start, int64_t total_q, int rows, int S_, int H, int KV, int hd,
                         int max_segs_per_seq, float softmax_scale, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(dtype == 0 || dtype == 1, "dtype must be 0 (f32) or 1 (bf16)");
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq, AttnBwdSched::DocCausal, true,
                                 dtype == 0 ? DType::f32 : DType::bf16);
    ref::fwd(s, fwd_args(q, q_ld, k, k_ld, v, v_ld, o, o_ld, lse, cu_seqlens, nullptr, row_seg_start, softmax_scale),
             S(stream));
  });
}

int nm_ref_attention_bwd(int dtype, const void* q, int64_t q_ld, const void* k, int64_t k_ld, const void* v,
                         int64_t v_ld, const void* o, int64_t o_ld, const float* lse, const void* d_o, int64_t do_ld,
                         void* dq, int64_t dq_ld, void* dk, int64_t dk_ld, void* dv, int64_t dv_ld,
                         const int32_t* cu_seqlens, const int32_t* row_seg_start, int64_t total_q, int rows, int S_,
                         int H, int KV, int hd, int max_segs_per_seq, float softmax_scale, void* workspace,
                         void* stream) {
  return capi_call([&] {
    NM_REQUIRE(dtype == 0 || dtype == 1, "dtype must be 0 (f32) or 1 (bf16)");
    const AttnShape s = shape_of(total_q, rows, S_, H, KV, hd, max_segs_per_seq, AttnBwdSched::DocCausal, true,
                                 dtype == 0 ? DType::f32 : DType::bf16);
    AttnBwdArgs a{};
    a.fwd = fwd_args(q, q_ld, k, k_ld, v, v_ld, const_cast<void*>(o), o_ld, const_cast<float*>(lse), cu_seqlens,
                     nullptr, row_seg_start, softmax_scale);
    a.d_o = d_o; a.do_ld = do_ld; a.dq = dq; a.dq_ld = dq_ld; a.dk = dk; a.dk_ld = dk_ld; a.dv = dv; a.dv_ld = dv_ld;
    ref::bwd(s, a, workspace, S(stream));
  });
}

}  // extern "C"
