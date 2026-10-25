// Device side of the optimizer (stage S6): the tables the kernels read and their launchers. DESIGN.md §4,
// optim.md §3-§5. Internal header of src/kernels/optim; the host engine that builds these tables and records a step
// lives in src/optim/engine.hpp.
//
// Everything here runs inside g_step: no host synchronization, no allocation, the update index k is read from device
// memory, every reduction has a fixed order (per-CTA partials in f64, then a fixed-order segment sum), so a step is
// bitwise repeatable on one GPU type. The update units build with --fmad=false and spell every formula with
// __fmul_rn / __fadd_rn / __fdiv_rn / __fsqrt_rn in the operation order of optim.md (levanter's jaxpr), no contraction.
#pragma once

#include <cstdint>

#include "marincpp/core.hpp"

namespace nm::optk {

// ------------------------------------------------------------------------------------------------ tables
enum : int32_t { kRuleMuonH = 0, kRuleAdamH = 1, kRuleAdam = 2, kRuleFrozen = 3 };

// One optimizer group as the kernels see it. (1 - b) are the f64-then-f32 constants of optim.md §5.3 rule 1 (computed
// on the host from the decimal values, never as 1.0f - f32(b)); lr = lr_tables[lr_table][k] * lr_scale.
struct GroupDev {
  float one_minus_b1, b1, one_minus_b2, b2, eps;
  float momentum;      // MuonH mu
  float lr_scale;
  float weight_decay;  // decoupled (Adam only), groups that list it; 0 for every baseline group
  int32_t lr_table;    // index into Tables::lr
  int32_t clip;        // clip-group index
  int32_t rule;        // kRule*
  int32_t pad;
};

// Clip state of one clip group (optim.md §3.1): gn = sqrt(sum of squares); clip iff !(gn < max_norm); clipped
// g = (g / gn) * max_norm, divide first. `factor` = max_norm / gn when clipping, else 1 (metrics only).
struct ClipDev { float gn; int32_t clip; float max_norm; float factor; };

// Tensors (SoA). A tensor is a row-major [rows, cols] view; the pointers address storage row `base_row` of the full
// tensor (0 when the whole tensor is allocated, the first owned row when only the owned shard is). s0 = momentum
// (MuonH) or mu (Adam, AdamH); s1 = nu.
struct TensorTab {
  float* const* p32;
  bf16_t* const* p16;          // entries may be null (no compute copy)
  const float* const* g;
  float* const* s0;
  float* const* s1;
  const int64_t* cols;
  const int64_t* base_row;
  const int32_t* group;
  const float* grad_scale;     // per tensor (lns): g' = g * grad_scale before the clip sum
  const float* update_scale;   // per tensor (lns): u' = u * update_scale
  const int32_t* rp;           // row partition of the tensor, -1 none
  float* const* u_out;         // entries may be null; else the applied update u (target - p for MuonH / AdamH) is
                               // also stored here, addressed like p32 (tests: T1.7 compares updates, not p + u - p)
};

// Elementwise slices of owned elements (one CTA each, <= kSliceElems elements). `row` is the full-tensor storage row
// of the slice's first element, `col` its column; a slice of a tensor with a row partition holds whole rows of one
// block (`block`), so the block, the lr multiplier and the frozen flag are per slice. `slot` = AdamH norm slot.
struct SliceTab {
  const int32_t* tensor;
  const int64_t* row;
  const int32_t* col;
  const int32_t* len;
  const int32_t* block;        // global row-partition block, -1 none
  const int32_t* slot;         // AdamH norm slot, -1 none
};

// Row partitions (optim.hpp RowPartition), blocks numbered globally.
struct RowPartTab {
  const uint8_t* const* touched;        // [n_rp] -> [rows] lazy updates (InputLayer::touched_rows), entries may be null
  const int32_t* block_rp;              // [all blocks] owning partition
  const int32_t* block_lr_table;        // [all blocks] index into Tables::lr of the lr multiplier, -1 = 1
  const int32_t* block_frozen_from;     // [all blocks] first frozen update index, INT32_MAX = never
};

// MuonH matrices (one layer matrix of one leaf), [OUT, IN] with IN = cols of the tensor. Matrix row r lives at
// full-tensor storage row row_map[r] (row_map null: row0 + r). The NS buffer holds the wide orientation: X = U
// (OUT <= IN) or X = U^T (OUT > IN, trans = 1), at x_off (elements) inside the NS buffers.
struct MatTab {
  const int32_t* tensor;
  const int64_t* row0;
  const int32_t* const* row_map;
  const int32_t* out;
  const int32_t* in;
  const uint8_t* trans;
  const int64_t* x_off;
  const int64_t* tile0;        // [n_mats + 1] prefix of 32 x 32 tiles of [OUT, IN]
};

struct Tables {
  const float* const* lr;      // [n_tables] -> [N] per-k tables (0 main, 1 adam, then custom)
  const float* bc1;            // [N] 1 - b1^(k+1)
  const float* bc2;
  const int32_t* step;         // device k
  int32_t n_steps;             // N: a kernel that reads k outside [0, N) traps (no table read out of bounds)
};

inline constexpr int kSliceElems = 16384;   // 64 KiB of f32 per CTA (DESIGN.md §4.3)
inline constexpr int kTile = 64;            // MuonH tile edge (one CTA; walked as 32 x 32 sub-tiles)
inline constexpr float kNormFloor = 1e-10f; // optim.md §5.3: 0x2EDBE6FF

// ------------------------------------------------------------------------------------------------ reductions
// partial[s - s_begin] = sum over slice s of (g * grad_scale)^2 (f32 squares, f64 sum; frozen blocks give 0).
void sumsq_slices(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const Tables& tb, int s_begin,
                  int s_end, double* partial, cudaStream_t st);
// out[i * stride + c] = sum of partial[idx[j] * stride + c] for j in [seg[i], seg[i+1]) (idx null: j itself), in
// index order, f64. accumulate = 1 adds to out.
void segment_sum(const double* partial, const int32_t* idx, const int64_t* seg, int n_seg, int stride, double* out,
                 int accumulate, cudaStream_t st);
// Clip state of clip groups [0, n) from their f64 sums of squares: gn = sqrt_rn(f32(sum)).
void clip_state(const double* clip_sumsq, const float* max_norm, int n, ClipDev* out, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ Adam, AdamH
void adam_update(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const ClipDev* clip, const Tables& tb, int s_begin, int s_end, cudaStream_t st);
// pass 1: moments, partial2[slice] = [||p||^2, ||a||^2]; pass 2: partial[slice] = ||w||^2 from the slot sums
// slot_pa[slot] = [||p||^2, ||a||^2]; pass 3: the update from slot_pa and slot_w.
void adamh_pass1(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const ClipDev* clip, const Tables& tb, int s_begin, int s_end, double* partial2, cudaStream_t st);
void adamh_pass2(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const Tables& tb, const double* slot_pa, int s_begin, int s_end, double* partial, cudaStream_t st);
void adamh_pass3(const TensorTab& t, const SliceTab& s, const RowPartTab& rp, const GroupDev* groups,
                 const Tables& tb, const double* slot_pa, const double* slot_w, int s_begin, int s_end,
                 cudaStream_t st);

// ------------------------------------------------------------------------------------------------ MuonH
// Kernels over the tiles [tile_begin, tile_end) of the matrices [m_begin, m_end) (one NS chunk). Tile partials are
// indexed by global tile number.
// momentum: buf = mu*buf + g';  u = mu*buf + g'  -> X (wide orientation); partial[tile] = ||u||^2 of the tile.
void muon_momentum(const TensorTab& t, const MatTab& m, const GroupDev* groups, const ClipDev* clip, int m_begin,
                   int m_end, int64_t tile_begin, int64_t tile_end, float* x, double* partial, cudaStream_t st);
// X /= sqrt_rn(f32(u2[mat])) + eps for `count` consecutive matrices of mn elements each (u2 = sums of the chunk).
void ns_normalize(float* x, int count, int64_t mn, const double* u2, float eps, cudaStream_t st);
void ns_scale(float* dst, const float* src, float c, int64_t n, cudaStream_t st);           // dst = c * src
// d[i][j] = d[j][i] for i >= m1 > j, per [m, m] matrix of `count` (symmetric NS blocks: lower-left from upper-right)
void ns_mirror(float* d, int count, int m, int m1, cudaStream_t st);
// 3xTF32 operand split: big = tf32_rna(x) (exact in TF32), small = x - big (exact in f32).
void tf32_split(const float* x, float* big, float* small, int64_t n, cudaStream_t st);
// hyperball (optim.md §4.4), per layer matrix, O read from X (the final NS buffer):
//   norms: partial2[tile] = [||p||^2, ||O||^2];  w: w = p - ((eta*O)*n)/max(m, floor) into X, partial[tile] = ||w||^2;
//   apply: u = (w / max(||w||, floor)) * n - p; u *= update_scale; p = p + u; p16 = bf16(p)
void hyperball_norms(const TensorTab& t, const MatTab& m, int m_begin, int m_end, int64_t tile_begin,
                     int64_t tile_end, const float* x, double* partial2, cudaStream_t st);
void hyperball_w(const TensorTab& t, const MatTab& m, const GroupDev* groups, const Tables& tb, int m_begin,
                 int m_end, int64_t tile_begin, int64_t tile_end, float* x, const double* mat_po, double* partial,
                 cudaStream_t st);
void hyperball_apply(const TensorTab& t, const MatTab& m, int m_begin, int m_end, int64_t tile_begin,
                     int64_t tile_end, const float* x, const double* mat_po, const double* mat_w, cudaStream_t st);

// Standalone NS (C ABI nm_newton_schulz, T1.6): src [batch, rows, cols] -> X in the wide orientation (transposed when
// rows > cols) with partial[b * tiles + t] = ||tile||^2; and back.
void ns_load(const float* src, float* x, int batch, int rows, int cols, double* partial, cudaStream_t st);
void ns_store(const float* x, float* dst, int batch, int rows, int cols, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ step
// (*step)++ at the very end of g_step; ring (nullable) gets lr_main[k], lr_adam[k] of the update just made. Traps
// unless k is in [0, n_steps).
void step_advance(int32_t* step, int32_t n_steps, const float* lr_main, const float* lr_adam, float* ring,
                  cudaStream_t st);

}  // namespace nm::optk
