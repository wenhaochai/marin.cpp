// Device kernels of the optimizer and the deterministic reductions they use. DESIGN.md §4.
// Everything runs inside g_step over the slices THIS rank owns (ZeRO-1, optim.hpp Ownership) and reads the update
// index k from device memory. Formulas and operation order follow optim.md exactly (f32; constants of optim.md §5.3;
// --fmad=false with explicit contractions as the T1.0 PTX shows).
#pragma once

#include <cstdint>

#include "marincpp/core.hpp"

namespace nm {

// Per-group constants and the device tables a group reads (optim.hpp GroupSpec, resolved at plan time).
struct GroupConsts {
  float one_minus_b1, b1, one_minus_b2, b2, eps;   // (1-b) as f64-then-f32 constants (0x3DCCCCCD, 0x3CA3D70A)
  float momentum;                                  // MuonH mu
  float lr_scale;
  const float* lr_table;                           // main, adam or a custom table, indexed by k
  int clip_index;
};

// Per-row control (optim.hpp RowPartition) on device: block of each row, per-block lr multiplier table (or null),
// frozen flag (step-gated by freeze_from), own-norm slot.
struct RowPartitionDev {
  const int32_t* block_of_row = nullptr;   // [rows]
  const float* const* lr_mult = nullptr;   // [n_blocks] tables indexed by k, entries may be null (= 1)
  const int32_t* frozen_from = nullptr;    // [n_blocks] first frozen step, INT32_MAX = never
  const int32_t* norm_slot = nullptr;      // [n_blocks] AdamH norm slot of the block
  const uint8_t* touched = nullptr;        // [rows] lazy updates (InputLayer::touched_rows), or null
  int n_blocks = 0;
};

// A list of owned tensor slices processed by one multi-tensor launch (each CTA handles a 64 KiB slice; slices map to
// (tensor, offset) through a prefix table built at plan time).
struct MultiTensor {
  int n = 0;
  float* const* p32 = nullptr;     // f32 parameters (owner shard)
  bf16_t* const* p16 = nullptr;    // bf16 compute copies (owner rows written here, then all-gathered next step)
  float* const* g = nullptr;       // f32 gradients (reduced: the mean gradient on the owner)
  float* const* s0 = nullptr;      // momentum (MuonH) or mu (Adam/AdamH)
  float* const* s1 = nullptr;      // nu (Adam/AdamH)
  const int64_t* numel = nullptr;
  const int32_t* group = nullptr;  // GroupConsts index of each tensor
  const float* grad_scale = nullptr;    // per tensor (lns), 1 by default
  const float* update_scale = nullptr;  // per tensor (lns), 1 by default
  const int32_t* row_partition = nullptr;  // per tensor index into RowPartitionDev[], -1 none
  const int64_t* slice_prefix = nullptr;   // [n_slices + 1] tensor index and offset per CTA (device)
  int n_slices = 0;
};

// ------------------------------------------------------------------------------------------------ deterministic sums
// Per-slice partial sums of squares of g * grad_scale (frozen rows excluded) -> fixed-order reduction per tensor, per
// logical leaf and per clip group. Run on the comm stream right after each bucket's reduction (comm.hpp). The per-group
// partials of all ranks are then summed by one all-reduce of n_groups floats.
void sumsq_partials(const MultiTensor& t, const RowPartitionDev* rp, float* const* src /* g or p */,
                    float* partial /* [n_slices] */, cudaStream_t s);                       // k_sumsq_partials
void reduce_partials(const float* partial, const int64_t* slice_prefix, int n_tensors, const int32_t* clip_of_tensor,
                     int n_clip, float* tensor_sumsq /* [n] */, float* clip_sumsq /* [n_clip] */,
                     cudaStream_t s);                                                       // k_reduce_partials

// ------------------------------------------------------------------------------------------------ clipping
// Each clip group has its own norm (optim.md §3.1): gn = sqrt(sumsq); clip iff !(gn < max_grad_norm); clipped
// g = (g / gn) * max_grad_norm (divide first). k_clip_state writes {gn, clip} per group; the update kernels apply the
// clip while reading g, so the clipped gradient is never stored.
struct ClipState { float gn; int32_t clip; float max_norm; float pad; };
void clip_state(const float* clip_sumsq, const float* max_norm, int n_clip, ClipState* out, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ MuonH (optim.md §4)
// The matrices this rank owns (whole logical leaves of one layer: q, k, v, o, gate, up, down).
struct MuonMatrices {
  int n_mats;
  const int32_t* tensor;            // owning storage tensor of each matrix
  const int64_t* row_begin;         // first storage row of the leaf (fused Wqkv / Wgu buffers)
  const int64_t* offset;            // element offset of layer l inside a stacked weight
  const int32_t* rows; const int32_t* cols;  // [OUT, IN]
  const uint8_t* transposed;        // OUT > IN: NS iterates on the wide orientation
  float* const* ns_x;               // per matrix: NS buffer X (f32, wide orientation)
};
// k_muon_momentum: buf = mu*buf + g';  u = mu*buf + g'  (Nesterov, no (1-mu)); u into the NS buffer; partial ||u||^2.
void muon_momentum(const MultiTensor& t, const MuonMatrices& m, const GroupConsts* gc, const ClipState* clip,
                   float* u_sumsq_partial, cudaStream_t s);                                  // k_muon_momentum
void ns_normalize(const MuonMatrices& m, const float* u_sumsq, float muon_eps, cudaStream_t s);   // X /= ||u|| + eps
// The 5 iterations (optim.md §4.3, "simple"): A = X X^T; cA = c*A; B = b*A + cA A; X = a*X + B X, per owned same-shape
// batch through GemmCache (TF32, F32_PEDANTIC or F32_3XTF32; symmetric products as upper-triangle tiles).
void ns_scale(float* dst, const float* src, float c, int64_t n, cudaStream_t s);                     // k_ns_scale
void ns_axpby(float* out, float a, const float* x, const float* y, int64_t n, cudaStream_t s);      // out = a*x + y
void ns_mirror_upper(float* x, int batch, int64_t m, cudaStream_t s);   // fills the lower triangle after symmetric GEMMs

// Norm-preserving projection per layer matrix (optim.md §4.4), three passes, f32, exact operation order:
//   pass 1  n = ||p||, m = ||O||;  pass 2  w = p - ((eta * O) * n) / max(m, 1e-10), ||w||^2 (w overwrites O);
//   pass 3  u = (w / max(||w||, 1e-10)) * n - p;  u *= update_scale;  p = p + u;  p16 = bf16(p)
// eta = lr_table[k] * lr_scale. With eta = 0 this still runs (rounding-level renormalization).
void hyperball_norms(const MuonMatrices& m, const MultiTensor& t, float* partial, cudaStream_t s);
void hyperball_w(const MuonMatrices& m, const MultiTensor& t, const GroupConsts* gc, const float* norms,
                 const int32_t* step, float* w_sumsq_partial, cudaStream_t s);
void hyperball_apply(const MuonMatrices& m, const MultiTensor& t, const float* norms, const float* w_sumsq,
                     cudaStream_t s);

// ------------------------------------------------------------------------------------------------ AdamH (optim.md §5.1)
// Whole-matrix (or per row block) projection of the Adam direction at the group's lr. Row-sharded tensors: the norms
// are per-shard partials + record_allreduce_scalars (comm.hpp): same math, class-S order.
//   pass 1  mu = (1-b1)*g + b1*mu;  nu = (1-b2)*g*g + b2*nu;  a = (mu/bc1)/(sqrt(nu/bc2) + eps);  ||p||^2, ||a||^2
//   pass 2  w = p - ((eta*a)*n)/max(m, 1e-10)  (a recomputed from mu, nu);  ||w||^2
//   pass 3  p = p + ((w / ||w||) * n - p)   (2-D branch: no floor in this division);  p16 = bf16(p)
// Frozen rows (RowPartitionDev) keep g = 0 in the norms and get no update; lazy rows update only when touched.
void adamh_pass1(const MultiTensor& t, const RowPartitionDev* rp, const GroupConsts* gc, const ClipState* clip,
                 const float* bc1_table, const float* bc2_table, const int32_t* step, float* partial, cudaStream_t s);
void adamh_pass2(const MultiTensor& t, const RowPartitionDev* rp, const GroupConsts* gc, const float* bc1_table,
                 const float* bc2_table, const int32_t* step, const float* norms, float* partial, cudaStream_t s);
void adamh_pass3(const MultiTensor& t, const RowPartitionDev* rp, const GroupConsts* gc, const float* bc1_table,
                 const float* bc2_table, const int32_t* step, const float* norms, const float* w_sumsq,
                 cudaStream_t s);

// ------------------------------------------------------------------------------------------------ Adam (optim.md §5.2)
// Fused multi-tensor: mu, nu as AdamH; u = (-lr_k * lr_mult[row]) * ((mu/bc1)/(sqrt(nu/bc2) + eps)); p = p + u;
// p16 = bf16(p). weight_decay applies only to groups whose GroupSpec lists it (validate() guards the value).
void adam_update(const MultiTensor& t, const RowPartitionDev* rp, const GroupConsts* gc, const ClipState* clip,
                 const float* bc1_table, const float* bc2_table, const int32_t* step, float weight_decay,
                 cudaStream_t s);                                                            // k_adam_update

// k_step_advance: (*step)++ at the very end of g_step; also copies lr/adam_lr of this step into the metrics ring.
void step_advance(int32_t* step, const float* lr_table, const float* adam_lr_table, float* ring_slot, cudaStream_t s);

}  // namespace nm
