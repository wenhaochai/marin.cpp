// Stage S3 internal API of the elementwise module, beyond include/marincpp/kernels/elementwise.hpp:
//   1. storage-generic entry points (the f32-storage instantiations of the f32 compute policy, DESIGN.md §3.0): the
//      *_dt functions take the public argument structs; with DType::f32 every bf16_t* activation, gain and table
//      pointer in the struct is read as a float* (f32 storage), with DType::bf16 they are bf16 as in the public API.
//      rstd, partials, f32 accumulators and integer arrays keep their types in both.
//   2. the sizes a caller plans for: gain-partial rows, embedding-backward workspace.
//   3. the RoPE table builders of T1.3 (host double method of DESIGN.md §3.4, and the device libdevice method).
#pragma once

#include <cstdint>

#include "marincpp/kernels/elementwise.hpp"
#include "marincpp/kernels/reduce.hpp"

namespace nm::ew {

// Number of gain-partial rows [n, width] f32 that the norm backward kernels (rmsnorm_bwd_add, sandwich_tail_bwd,
// qknorm_rope_bwd, dq_post_qknorm_rope_bwd) write for T rows (tokens) on the current device:
// n = min(1024, ceil(T / 16), 4 x SMs) parts of consecutive rows (ew_common.cuh norm_parts). The partials depend on T
// and the SM count only, so a step is bitwise repeatable on one GPU type (DESIGN.md §11).
int64_t norm_partial_rows(int64_t T);

// Aliasing: rmsnorm_bwd_add may run in place (dres == dx) and sandwich_tail_bwd may too (dx_out == dx_in; pre-LN also
// dbranch == dx_in): those buffers are read with coherent loads, never ld.global.nc. No other output may alias an input.
// q_scale (qknorm_rope_*, dq_post_qknorm_rope_bwd): applied as one multiply by the f32 value passed. Under levanter
// rounding with bf16 storage it must be a bf16 value (numerics.md §3.4, bf16(1/sqrt(hd)), multiplied in bf16; checked);
// otherwise any value: under accurate rounding and under the f32 policy it is one f32 multiply inside the f32 chain. The
// caller chooses it (DESIGN.md §3.4: under accurate, FA3 takes the scale, so q_scale = 1 in qknorm_rope_fwd and the
// whole scale in dq_post_qknorm_rope_bwd).
void rmsnorm_fwd_dt(DType storage, const RmsNormFwd& a, Rounding r, cudaStream_t s);
void rmsnorm_bwd_add_dt(DType storage, const RmsNormBwdAdd& a, Rounding r, cudaStream_t s);
void sandwich_tail_fwd_dt(DType storage, const SandwichTailFwd& a, Rounding r, cudaStream_t s);
void sandwich_tail_bwd_dt(DType storage, const SandwichTailBwd& a, Rounding r, cudaStream_t s);
void qknorm_rope_fwd_dt(DType storage, const QkNormRopeFwd& a, Rounding r, cudaStream_t s);
void qknorm_rope_bwd_dt(DType storage, const QkNormRopeBwd& a, Rounding r, cudaStream_t s);
// RoPE (+ q scale) alone on already-normed rows [T, nheads x hd] at row stride ld; out is [T, nheads x hd]. The
// rounding chain is the RoPE part of qknorm_rope_fwd (T1.3 compares it with levanter's rotated q and k).
void rope_fwd_dt(DType storage, const void* x, int64_t ld, const void* cos_tab, const void* sin_tab, const int32_t* pos,
                 void* out, int64_t T, int nheads, int hd, float scale, Rounding r, cudaStream_t s);
void swiglu_fwd_dt(DType storage, const void* gu, void* h, int64_t T, int F, Rounding r, cudaStream_t s);
void swiglu_bwd_dt(DType storage, const void* dh, const void* gu, void* dgu, int64_t T, int F, Rounding r,
                   cudaStream_t s);
void embed_fwd_dt(DType storage, const void* table, const int32_t* tokens, void* out, int64_t T, int D, cudaStream_t s);
void embed_bwd_sorted_dt(DType storage, const void* dout, const int32_t* tokens, float* dtable, int64_t T, int D, int V,
                         EmbedAccum acc, const EmbedBwdWorkspace& ws, uint8_t* touched_rows, cudaStream_t s);

// Embedding backward workspace: carve one block of embed_bwd_workspace_bytes(T, V) bytes into the struct. The cub_temp
// region also holds the f32 chunk partials of the F32 path (2 x ceil(T / 256) x D floats), so it is sized for
// D <= kEmbedMaxD; embed_bwd_sorted checks the actual need.
inline constexpr int kEmbedMaxD = 4096;
inline constexpr int kEmbedChunk = 256;   // sorted entries per chunk of the F32 path
EmbedBwdWorkspace embed_bwd_carve(void* base, uint64_t bytes, int64_t T, int V);
uint64_t embed_bwd_cub_bytes(int64_t T, int V);   // what cub's radix sort needs (queried once, cached per (T, bits))

// RoPE tables (T1.3). Method::HostDouble = DESIGN.md §3.4 (f32 inv_freq built on the host; cos/sin of the f32 angle in
// double, rounded to f32, then to bf16). Method::DeviceLibdevice evaluates the same f32 expression on the GPU with
// CUDA's libdevice powf / IEEE division / cosf / sinf, the functions XLA:GPU calls for these ops (levanter computes
// inv_freq eagerly on the device and the tables inside the jitted step). Both write [S_max, hd] bf16 bits and
// [hd / 2] f32 to host memory.
enum class RopeMethod : int { HostDouble = 0, DeviceLibdevice = 1 };
void build_rope_tables_method(RopeMethod m, int S_max, int hd, double theta, bf16_t* cos_host, bf16_t* sin_host,
                              float* inv_freq_host);

}  // namespace nm::ew
