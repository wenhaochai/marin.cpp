/* Stage S4 additions to the C ABI of libmarincpp_kernels.so (tests and the GEMM search bench).
 * nm_gemm_search times every deterministic candidate of one linear-layer GEMM at the given shape (GemmCache search,
 * DESIGN.md §3.7) on scratch it allocates, and writes a JSON object (chosen kernel, ms, TFLOP/s, candidate count, the
 * fastest candidates) to json_out (capacity json_cap bytes). orient 0 fwd, 1 dgrad, 2 wgrad; rounding (wgrad) 0
 * levanter, 1 accurate. The C ABI linear entry points of capi.h search a shape on its first use.            */
#ifndef MARINCPP_CAPI_GEMM_H
#define MARINCPP_CAPI_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int nm_gemm_search(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* json_out, int json_cap,
                   void* stream);
/* Every deterministic candidate of a linear GEMM that launched in one search (names, '\n'-terminated, fastest first)
 * and one call of the candidate of that name (wgrad: C = D, beta = 1, in place). Tests: a gate runs every candidate a
 * search may pick, since the timed choice varies between runs.                                                */
int nm_gemm_candidates(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* names_out, int names_cap,
                       void* stream);
int nm_linear_named(const char* kernel, int orient, int rounding, const void* a, const void* b, void* d, int64_t T,
                    int64_t N, int64_t K, void* stream);
/* The kernel the C ABI cache chose for this linear GEMM ("" if the shape has not run yet). */
int nm_gemm_chosen(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* name_out, int name_cap);

/* Custom kernels registered with the GEMM layer (CUTLASS SM90 units on sm_90a builds): count, name, and one call of
 * custom kernel i on a linear GEMM (orient / rounding as above; C = D for wgrad, beta = 1). Returns NM_ERR_INVALID
 * when kernel i does not accept the shape. Workspace from cudaMalloc. Tests only.                          */
int nm_gemm_custom_count(void);
const char* nm_gemm_custom_name(int i);
int nm_linear_custom(int i, int orient, int rounding, const void* a, const void* b, void* d, int64_t T, int64_t N,
                     int64_t K, void* stream);

#ifdef __cplusplus
}
#endif

#endif
