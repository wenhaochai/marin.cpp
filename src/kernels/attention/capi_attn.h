/* Stage S4 additions to the C ABI of libmarincpp_kernels.so for tests and the E1b bench (capi.h holds the core set).
 * Conventions as in capi.h. `mode` = FA3 launch mode (src/kernels/attention/attn_internal.hpp Fa3Mode): 0 persistent
 * (work list, grid = SM count), 1 doc_causal, 2 stock padded (FA3's own varlen backward over rows * max_segs segment
 * slots), 3 stock exact (FA3 with the real segment count `exact_n_segs` and longest segment `exact_max_seqlen`, E1's
 * setup), 4 work list with one CTA per list slot. The workspace of a mode holds nm_fa3_mode_workspace_bytes(training)
 * bytes; a training workspace serves the forward and the backward of the same shape.                              */
#ifndef MARINCPP_CAPI_ATTN_H
#define MARINCPP_CAPI_ATTN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t nm_fa3_mode_workspace_bytes(int mode, int64_t total_q, int rows, int S, int H, int KV, int hd,
                                     int max_segs_per_seq, int training);
int nm_fa3_mode_fwd(int mode, int exact_n_segs, int exact_max_seqlen, const void* q, int64_t q_ld, const void* k,
                    int64_t k_ld, const void* v, int64_t v_ld, void* o, int64_t o_ld, float* lse,
                    const int32_t* cu_seqlens, const int32_t* n_segs, const int32_t* row_seg_start, int64_t total_q,
                    int rows, int S, int H, int KV, int hd, int max_segs_per_seq, float softmax_scale,
                    void* workspace, void* stream);
int nm_fa3_mode_bwd(int mode, int exact_n_segs, int exact_max_seqlen, const void* q, int64_t q_ld, const void* k,
                    int64_t k_ld, const void* v, int64_t v_ld, const void* o, int64_t o_ld, const float* lse,
                    const void* d_o, int64_t do_ld, void* dq, int64_t dq_ld, void* dk, int64_t dk_ld, void* dv,
                    int64_t dv_ld, const int32_t* cu_seqlens, const int32_t* n_segs, const int32_t* row_seg_start,
                    int64_t total_q, int rows, int S, int H, int KV, int hd, int max_segs_per_seq, float softmax_scale,
                    int deterministic, void* workspace, void* stream);

/* The reference attention (RefF32 backend) with explicit storage: dtype 0 = f32, 1 = bf16. Segments from cu_seqlens
 * (packed slots) or, when cu_seqlens is NULL, row_seg_start. Workspace (backward): nm_attention_workspace_bytes(3, ...). */
int nm_ref_attention_fwd(int dtype, const void* q, int64_t q_ld, const void* k, int64_t k_ld, const void* v,
                         int64_t v_ld, void* o, int64_t o_ld, float* lse, const int32_t* cu_seqlens,
                         const int32_t* row_seg_start, int64_t total_q, int rows, int S, int H, int KV, int hd,
                         int max_segs_per_seq, float softmax_scale, void* stream);
int nm_ref_attention_bwd(int dtype, const void* q, int64_t q_ld, const void* k, int64_t k_ld, const void* v,
                         int64_t v_ld, const void* o, int64_t o_ld, const float* lse, const void* d_o, int64_t do_ld,
                         void* dq, int64_t dq_ld, void* dk, int64_t dk_ld, void* dv, int64_t dv_ld,
                         const int32_t* cu_seqlens, const int32_t* row_seg_start, int64_t total_q, int rows, int S,
                         int H, int KV, int hd, int max_segs_per_seq, float softmax_scale, void* workspace,
                         void* stream);

#ifdef __cplusplus
}
#endif

#endif
