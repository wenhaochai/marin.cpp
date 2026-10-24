/* C ABI of libmarincpp_kernels.so for tests (pytest + ctypes; torch tensors pass .data_ptr()). DESIGN.md §10.5.
 * No Python is linked into the training binary; this library is the same object code the binary links. The portable
 * kernels are also built for sm_80/86 (vis-node tests); FA3, CE B1/B1.5/B2 and custom WGMMA kernels need sm_90a.
 * All pointers are device pointers unless named *_host. `stream` is a cudaStream_t (0 = legacy default stream).
 * Every int function returns NM_OK (0) on success, else NM_ERR_INVALID (1: bad argument, config or requirement),
 * NM_ERR_DEVICE (2: a CUDA, NCCL or cuBLAS call failed) or NM_ERR_INTERNAL (3: any other failure); nm_last_error()
 * returns the message (thread-local). The library exports only the nm_* symbols (cmake/capi.map).
 * `rounding`: 0 = levanter, 1 = accurate. `dtype`: 0 = f32 storage (f32 compute policy), 1 = bf16 storage.     */
#ifndef MARINCPP_CAPI_H
#define MARINCPP_CAPI_H

#include <stdint.h>

#define NM_OK 0
#define NM_ERR_INVALID 1
#define NM_ERR_DEVICE 2
#define NM_ERR_INTERNAL 3

#ifdef __cplusplus
extern "C" {
#endif

const char* nm_version(void);        /* git sha of the build ("+dirty": uncommitted or untracked sources) */
const char* nm_last_error(void);
int nm_device_arch(void);            /* 80, 86, 90: tests skip sm_90a-only kernels elsewhere; 0 on a CUDA error
                                        (nm_last_error says which; the CUDA error state is cleared) */

/* ---- elementwise (kernels/elementwise.hpp) ---------------------------------------------------------------- */
int nm_doc_meta(const int32_t* tokens, const uint8_t* row_valid, int rows, int S, int32_t eos, int max_segs,
                int32_t* targets, float* weight, int32_t* pos, int32_t* cu_seqlens, int32_t* n_segs,
                int32_t* row_seg_start, uint32_t* overflow, void* stream);
int nm_embed_fwd(const void* table, const int32_t* tokens, void* out, int64_t T, int D, void* stream);
int nm_embed_bwd(const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                 int accum /*0 f32, 1 bf16 sequential*/, uint8_t* touched_rows, void* workspace,
                 uint64_t workspace_bytes, void* stream);
uint64_t nm_embed_bwd_workspace_bytes(int64_t T, int V);
int nm_rmsnorm_fwd(int dtype, int rounding, const void* x, const void* gain, void* y, float* rstd, int64_t T, int D,
                   float eps, void* stream);
int nm_rmsnorm_bwd_add(int dtype, int rounding, const void* dy, const void* x, const void* gain, const float* rstd,
                       const void* dres, void* dx, float* dgain, int64_t T, int D, float eps, void* stream);
int nm_sandwich_tail_fwd(int dtype, int rounding, const void* branch, const void* x_in, const void* g_post,
                         const void* g_next, void* x_out, void* a_next, float* rstd_post, float* rstd_next, int64_t T,
                         int D, float eps, void* stream);
int nm_sandwich_tail_bwd(int dtype, int rounding, const void* dx_out, const void* da_next, const void* x_out,
                         const void* g_next, const float* rstd_next, const void* branch, const void* g_post,
                         const float* rstd_post, void* dx_in, void* dbranch, float* dg_next, float* dg_post,
                         int64_t T, int D, float eps, void* stream);
int nm_qknorm_rope_fwd(int dtype, int rounding, const void* qkv, int64_t qkv_ld, const void* gq, const void* gk,
                       const void* cos_tab, const void* sin_tab, const int32_t* pos, void* q_rot, void* k_rot,
                       float* rstd_q, float* rstd_k, int64_t T, int H, int KV, int hd, float eps, float q_scale,
                       void* stream);
int nm_qknorm_rope_bwd(int dtype, int rounding, const void* dq_rot, const void* dk_rot, const void* qkv,
                       int64_t qkv_ld, const void* gq, const void* gk, const void* cos_tab, const void* sin_tab,
                       const int32_t* pos, const float* rstd_q, const float* rstd_k, void* dqkv, int64_t dqkv_ld,
                       float* dgq, float* dgk, int64_t T, int H, int KV, int hd, float eps, float q_scale,
                       void* stream);
int nm_dq_post_qknorm_rope_bwd(int rounding, const float* dq_accum, const void* qkv, int64_t qkv_ld, const void* gq,
                               const void* cos_tab, const void* sin_tab, const int32_t* pos, const float* rstd_q,
                               void* dqkv, int64_t dqkv_ld, float* dgq, int64_t T, int H, int hd, float eps,
                               float q_scale, void* stream);
int nm_rope_tables(int S_max, int hd, double theta, uint16_t* cos_host, uint16_t* sin_host, float* inv_freq_host);
int nm_swiglu_fwd(int dtype, int rounding, const void* gu, void* h, int64_t T, int F, void* stream);
int nm_swiglu_bwd(int dtype, int rounding, const void* dh, const void* gu, void* dgu, int64_t T, int F, void* stream);

/* ---- GEMM (kernels/gemm.hpp): the exact kernels the trainer chose; `epilogue` = gemm.hpp Epilogue --------------- */
int nm_linear_fwd(int epilogue, const void* x, const void* w, void* y, const void* epilogue_args, int64_t T, int64_t N,
                  int64_t K, void* stream);
int nm_linear_dgrad(int epilogue, const void* dy, const void* w, void* dx, const void* epilogue_args, int64_t T,
                    int64_t N, int64_t K, void* stream);
int nm_linear_wgrad(int rounding, const void* dy, const void* x, float* dw, int64_t T, int64_t N, int64_t K,
                    void* stream);
int nm_gemm_f32(int compute /*1 tf32, 2 f32 pedantic, 3 3xtf32*/, const float* a, const float* b, float* d,
                int64_t m, int64_t n, int64_t k, int symmetric, void* stream);   /* f32 policy and NS tests */

/* ---- attention (kernels/attention.hpp); backend 0 fa3, 1 cudnn, 2 levref, 3 ref_f32; bwd_sched 0 persistent,
 *      1 doc_causal -------------------------------------------------------------------------------------------- */
uint64_t nm_attention_workspace_bytes(int backend, int bwd_sched, int64_t total_q, int rows, int S, int H, int KV,
                                      int hd, int max_segs_per_seq, int training);
int nm_attention_fwd(int backend, int bwd_sched, const void* q, int64_t q_ld, const void* k, int64_t k_ld,
                     const void* v, int64_t v_ld, void* o, int64_t o_ld, float* lse, const int32_t* cu_seqlens,
                     const int32_t* n_segs, const int32_t* row_seg_start, int64_t total_q, int rows, int S, int H,
                     int KV, int hd, int max_segs_per_seq, float softmax_scale, void* workspace, void* stream);
int nm_attention_bwd(int backend, int bwd_sched, const void* q, int64_t q_ld, const void* k, int64_t k_ld,
                     const void* v, int64_t v_ld, const void* o, int64_t o_ld, const float* lse, const void* d_o,
                     int64_t do_ld, const float* softmax_d /*nullable*/, float* dq_accum /*nullable*/, void* dq,
                     int64_t dq_ld, void* dk, int64_t dk_ld, void* dv, int64_t dv_ld, const int32_t* cu_seqlens,
                     const int32_t* n_segs, const int32_t* row_seg_start, int64_t total_q, int rows, int S, int H,
                     int KV, int hd, int max_segs_per_seq, float softmax_scale, int deterministic, void* workspace,
                     void* stream);

/* ---- fused linear cross-entropy (kernels/cross_entropy.hpp); impl 0 A, 1 B1, 2 B1.5, 3 B2; logits_store and io:
 *      0 f32, 1 bf16; dw_round 0 none, 1 per-chunk bf16 (strict) ------------------------------------------------- */
uint64_t nm_ce_workspace_bytes(int impl, int64_t T, int D, int V, int chunk_rows, int logits_store);
int nm_fused_linear_ce(int impl, int rounding, int io, int logits_store, const void* x, const void* w,
                       const int32_t* targets, const float* weight, const float* grad_scale, const float* logit_bias,
                       float* loss_tok, float* lse, float* loss_partial, void* dx /*nullable*/, int accumulate_dx,
                       float* dw /*nullable*/, float* dw_microbatch /*nullable*/, int dw_round, int forward_only,
                       int64_t T, int D, int V, int chunk_rows, void* workspace, void* stream);

/* ---- optimizer (kernels/optim_kernels.hpp), one tensor or a stacked weight at a time for tests --------------- */
int nm_newton_schulz(int precision /*0 f32 pedantic, 1 tf32, 2 3xtf32*/, int symmetric, float* x, int64_t batch,
                     int64_t rows, int64_t cols, int steps, float a, float b, float c, float eps, void* workspace,
                     void* stream);
int nm_muonh_update(float* p, void* p16, const float* g, float* momentum, int64_t layers, int64_t rows, int64_t cols,
                    float mu, float eta, float muon_eps, int ns_precision, float clip_gn, int clip, float max_norm,
                    void* workspace, void* stream);
int nm_adamh_update(float* p, void* p16, const float* g, float* mu, float* nu, int64_t rows, int64_t cols, int64_t t,
                    float eta, float b1, float b2, float eps, float clip_gn, int clip, float max_norm,
                    const int32_t* block_of_row /*nullable*/, const float* block_lr_mult, const uint8_t* block_frozen,
                    int n_blocks, void* stream);
int nm_adam_update(float* p, void* p16, const float* g, float* mu, float* nu, int64_t rows, int64_t cols, int64_t t,
                   float adam_lr, float b1, float b2, float eps, float clip_gn, int clip, float max_norm,
                   const int32_t* block_of_row /*nullable*/, const float* block_lr_mult, const uint8_t* block_frozen,
                   int n_blocks, void* stream);
/* Schedules: the fixture loader and the formula cross-check (host). */
int nm_schedule_tables(const char* config_toml, int64_t n, int from_fixture, float* lr_host, float* adam_lr_host,
                       float* bc1_host, float* bc2_host, int* n_ulp_diffs);

/* ---- data (data.hpp) ----------------------------------------------------------------------------------------- */
int nm_threefry2x32(uint32_t k0, uint32_t k1, uint32_t* x0, uint32_t* x1, int64_t n);   /* host, in place */
int nm_threefry2x32_device(uint32_t k0, uint32_t k1, uint32_t* x0, uint32_t* x1, int64_t n, void* stream);
int nm_levanter_order(uint32_t data_seed, uint64_t n_seqs, int mix_block, uint64_t j0, uint64_t count,
                      uint32_t* out_host);
int nm_plan_seq_of(const char* config_toml, int64_t step, int row, uint64_t* seq_host);
int nm_init_tensor(uint32_t seed, const uint32_t* key_path, int key_path_len, int64_t numel, float scale_f32,
                   float* out_host);  /* truncated normal of model.md §4: threefry + XLA f32 erf/erf_inv port */

/* ---- eval aggregation (eval.hpp), host: T3.1 (i) feeds levanter's per-batch sums ------------------------------- */
int nm_eval_running_mean(const float* batch_sums_host /* [n_batches, 3 + 3*n_tags] */, int n_batches, int n_tags,
                         const char* out_json);

/* ---- whole model (T2/T3 and gates V1-V4 run the real step through these) -------------------------------------- */
typedef struct nm_model nm_model;
nm_model* nm_model_create(const char* config_toml, int rank, int nranks, void* nccl_comm);  /* any policy, incl. f32 */
int nm_model_load_params(nm_model* m, const char* dir /* per-leaf .npy from tools/levref */);
int nm_model_export_params(nm_model* m, const char* dir /* per-leaf .npy, levanter names and layouts */);
int nm_model_set_regime(nm_model* m, int64_t step);   /* selects the regime (and its static flags) of update `step` */
int nm_model_set_rng(nm_model* m, uint32_t k0, uint32_t k1);
int nm_model_loss_and_grads(nm_model* m, const int32_t* tokens_host, int rows, float* loss_out,
                            float* readout_losses_out /* [n_readouts] or NULL */, const char* dump_dir);
int nm_model_dump_activations(nm_model* m, const char* dir);   /* every declared activation of the last forward */
int nm_model_optimizer_step(nm_model* m);
int nm_model_eval(nm_model* m, const char* out_json);
void nm_model_destroy(nm_model* m);

/* ---- variants: one block / head / input layer by registered name, on named buffers ---------------------------- */
/* buffers: n entries of (name, device pointer); names as the variant declares them (params, acts, grads).        */
int nm_block_forward(const char* name, const char* config_toml, int layer, int n, const char* const* buf_names,
                     void* const* buf_ptrs, void* stream);
int nm_block_backward(const char* name, const char* config_toml, int layer, int n, const char* const* buf_names,
                      void* const* buf_ptrs, void* stream);

#ifdef __cplusplus
}
#endif
#endif /* MARINCPP_CAPI_H */
