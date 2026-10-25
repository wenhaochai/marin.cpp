/* Stage S6 additions to the C ABI of libmarincpp_kernels.so (tests only; capi.h holds the declared optimizer entry
 * points nm_newton_schulz, nm_muonh_update, nm_adamh_update, nm_adam_update). Conventions as in capi.h: device
 * pointers unless stated, `stream` a cudaStream_t, NM_OK / NM_ERR_* return codes, nm_last_error().
 *
 * The optimizer engine (src/optim/engine.hpp) over a list of tensors: create, bind tables, prepare, run the phases,
 * read its sums, destroy. Phases: 0 clip sums of squares, 1 update (clip state, Adam, MuonH, AdamH pass 1),
 * 2 AdamH pass 2, 3 AdamH pass 3, 4 AdamH pass 3 + step advance. A sharded engine (ZeRO-1 emulation) needs the
 * caller to sum clip_sumsq after phase 0, slot_pa after phase 1 and slot_w after phase 2 over the ranks.            */
#ifndef MARINCPP_CAPI_OPTIM_H
#define MARINCPP_CAPI_OPTIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nm_opt_group {
  int rule;                 /* 0 MuonH, 1 AdamH, 2 Adam, 3 Frozen */
  int lr_table;             /* 0 main, 1 adam */
  int clip_group;           /* groups with the same index share one clip norm */
  double lr_scale, b1, b2, eps, momentum, max_grad_norm, weight_decay;
} nm_opt_group;

typedef struct nm_opt_tensor {
  int group;
  int64_t rows, cols;       /* the full tensor as [rows, cols] */
  int64_t base_row;         /* full-tensor row addressed by the pointers' row 0 */
  float* p32;
  void* p16;                /* bf16, nullable */
  const float* g;
  float* s0;                /* MuonH momentum buffer, Adam/AdamH mu */
  float* s1;                /* Adam/AdamH nu (nullable for MuonH) */
  float grad_scale, update_scale;
  int64_t n_mats;           /* MuonH: matrix i = rows [i * mat_rows, (i + 1) * mat_rows), or row_map rows */
  int64_t mat_rows;
  const int32_t* row_map;   /* host, [n_mats * mat_rows] storage rows, NULL = contiguous */
  int64_t n_owned;          /* host pairs owned[2 i], owned[2 i + 1] = [begin, end) rows; 0 = every row */
  const int64_t* owned;
  float* u_out;             /* nullable: the applied update u is also stored here (addressed like p32) */
} nm_opt_tensor;

int nm_opt_create(const nm_opt_group* groups, int n_groups, const nm_opt_tensor* tensors, int n_tensors,
                  int ns_precision /* 0 f32, 1 tf32, 2 3xtf32; + 4: symmetric NS blocks; + 8: fold c into alpha */,
                  int ns_steps,
                  double muon_eps,
                  int64_t ns_chunk_bytes, int timed_gemm, int sharded, void** engine);
/* lr_main, lr_adam, bc1, bc2: [n_steps] each; step: the device k (a step at k outside [0, n_steps) traps). */
int nm_opt_set_tables(void* engine, const float* lr_main, const float* lr_adam, const float* bc1, const float* bc2,
                      int32_t* step, int64_t n_steps);
int nm_opt_prepare(void* engine, void* stream);
int nm_opt_phase(void* engine, int phase, void* stream);
int nm_opt_step(void* engine, int advance, void* stream);
/* device pointers of the engine's sums ([n_clip], [n_slots][2], [n_slots]) and clip states ([n_clip] x
 * {float gn, int32 clip, float max_norm, float factor}); also the matrix and NS chunk counts. */
int nm_opt_buffers(void* engine, double** clip_sumsq, int* n_clip, double** slot_pa, double** slot_w, int* n_slots,
                   void** clip_states, int* n_mats, int* n_chunks);
int nm_opt_destroy(void* engine);

/* Raw NS GEMM (probe of BUILD_LOG "Decisions for S4-S6" item 14): D[M, N] = A[M, K] B[K, N], row-major. */
int nm_ns_gemm(int precision, const float* A, const float* B, float* D, int64_t M, int64_t N, int64_t K, void* stream);

#ifdef __cplusplus
}
#endif
#endif /* MARINCPP_CAPI_OPTIM_H */
