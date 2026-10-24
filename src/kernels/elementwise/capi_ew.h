/* Stage S3 additions to the C ABI of libmarincpp_kernels.so (tests only; capi.h holds the declared core set).
 * Conventions as in capi.h: device pointers, `stream` a cudaStream_t, NM_OK / NM_ERR_* return codes, nm_last_error().
 *
 * Gain gradients in the capi.h norm entry points (nm_rmsnorm_bwd_add dgain, nm_sandwich_tail_bwd dg_next / dg_post,
 * nm_qknorm_rope_bwd dgq / dgk, nm_dq_post_qknorm_rope_bwd dgq): f32 [width], overwritten with this call's gain
 * gradient = the per-part partials reduced in fixed order (reduce_partials_rows), rounded to bf16 when rounding is
 * levanter and the storage is bf16 (levanter's bf16 parameter cotangent). NULL skips it. The entry point allocates the
 * partial buffer with cudaMallocAsync on `stream` (test path; the trainer passes planned partial slots instead).
 * nm_embed_bwd: `workspace` holds nm_embed_bwd_workspace_bytes(T, V) bytes; dtable is accumulated into (+=).     */
#ifndef MARINCPP_CAPI_EW_H
#define MARINCPP_CAPI_EW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* method 0 = host double (DESIGN.md §3.4, the nm_rope_tables method), 1 = device libdevice f32 (XLA:GPU functions) */
int nm_rope_tables_method(int method, int S_max, int hd, double theta, uint16_t* cos_host, uint16_t* sin_host,
                          float* inv_freq_host);
int64_t nm_norm_partial_rows(int64_t T);
/* RoPE (+ scale) alone on normed rows x [T, nheads * hd] (row stride ld) -> out [T, nheads * hd] */
int nm_rope_fwd(int dtype, int rounding, const void* x, int64_t ld, const void* cos_tab, const void* sin_tab,
                const int32_t* pos, void* out, int64_t T, int nheads, int hd, float scale, void* stream);
int nm_reduce_partials_rows(const float* partial, int n_blocks, int width, float* out, int accumulate, int round_bf16,
                            void* stream);
int nm_sum_fixed(const float* x, int64_t n, float* out, void* stream);
/* one tensor through the multi-tensor cast kernel (the device pointer list is built by the entry point) */
int nm_cast_f32_bf16(const float* src, void* dst, int64_t n, void* stream);
/* storage-generic embedding entry points: dtype 0 = f32 storage, 1 = bf16 (nm_embed_fwd / nm_embed_bwd are bf16) */
int nm_embed_fwd_dt(int dtype, const void* table, const int32_t* tokens, void* out, int64_t T, int D, void* stream);
int nm_embed_bwd_dt(int dtype, const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                    int accum, uint8_t* touched_rows, void* workspace, uint64_t workspace_bytes, void* stream);
/* raw partials of the RMSNorm backward ([nm_norm_partial_rows(T), D] f32), for the determinism test */
int nm_rmsnorm_bwd_add_partial(int dtype, int rounding, const void* dy, const void* x, const void* gain,
                               const float* rstd, const void* dres, void* dx, float* dgain_partial, int64_t T, int D,
                               float eps, void* stream);

#ifdef __cplusplus
}
#endif
#endif /* MARINCPP_CAPI_EW_H */
