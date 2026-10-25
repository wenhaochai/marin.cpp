/* Stage S5 additions to the C ABI of libmarincpp_kernels.so (tests and benches only; capi.h declares
 * nm_ce_workspace_bytes and nm_fused_linear_ce). Conventions as in capi.h: device pointers, `stream` a cudaStream_t,
 * NM_OK / NM_ERR_* return codes, nm_last_error(). impl 0 A, 1 B1, 2 B1.5, 3 B2; rounding 0 levanter, 1 accurate;
 * logits_store 0 f32, 1 bf16; dw_round 0 none, 1 per-chunk bf16 (strict).                                         */
#ifndef MARINCPP_CAPI_CE_H
#define MARINCPP_CAPI_CE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel choice of the plain-dz GEMMs (0 cuBLASLt, 1 this module's kernels) and which sm_90a kernels may run on H100:
 * sm90 bit 0 = logits, bit 1 = dX / dW GEMMs (0: the portable kernels everywhere; < 0: unchanged). Process-wide;
 * returns the previous setting packed as dgrad | wgrad << 1 | sm90 << 2. */
int nm_ce_set_backends(int dgrad_own, int wgrad_own, int sm90);
/* the fast statistics exponential (ce_mma.cuh exp_stat; default off): 1 on, 0 off; returns the previous setting */
int nm_ce_set_fast_stats_exp(int on);
/* strict mode: dw += f32(bf16(dw_microbatch)); dw_microbatch = 0 (n = V * D) */
int nm_ce_finish_microbatch(float* dw, float* dw_microbatch, int64_t n, void* stream);
/* *out = mult / sum(w[0..n)) in f32, fixed summation order */
int nm_ce_grad_scale(const float* w, int64_t n, float mult, float* out, void* stream);
/* One readout head (kernels/cross_entropy.hpp ReadoutCall): norm of `state`, CE against w, dW into dw, d(state) added
 * into d_state (NULL: none), gain partials [nm_norm_partial_rows(T), D] into dgain_partial (NULL: none). */
int nm_ce_readout(int impl, int rounding, int logits_store, const void* state, const void* norm_gain,
                  float* norm_rstd_ws, void* norm_out_ws, const void* w, float* dw, float* dgain_partial,
                  const float* grad_scale, void* d_state, int visible_dims, const int32_t* targets,
                  const float* weight, float* loss_tok, float* lse, float* loss_partial, int forward_only, int64_t T,
                  int D, int V, int chunk_rows, float eps, void* workspace, void* stream);
/* B1's dlogits pass alone, in place over a logits store [rows, ldS] (bench: its bandwidth); logits_store 0 f32, 1 bf16 */
int nm_ce_dlogits(int logits_store, void* store, int64_t ldS, const float* lse, const float* gscale,
                  const int32_t* targets, int rows, int V, void* stream);
/* the device's compute capability as the CE dispatch sees it (90 = the sm_90a kernels run) */
int nm_ce_device_arch(void);

#ifdef __cplusplus
}
#endif

#endif
