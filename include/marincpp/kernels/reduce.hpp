// Deterministic reductions shared by the norm, embedding, CE, eval and statistics kernels. DESIGN.md §11.
// Rule: no floating-point atomics anywhere in a training step. Every reduction is (1) per-CTA partials written to a
// planned buffer, then (2) one fixed-order pass. The order depends only on the shape and the GPU's SM count (the norm
// partition count, ew::norm_partial_rows), never on the launch timing: bitwise repeatable on one GPU type.
#pragma once

#include <cstdint>

#include "marincpp/core.hpp"

namespace nm {

// out[w] (+)= sum_b partial[b, w] in ascending b, f32. Gain gradients: accumulate=true into the f32 gradient across
// microbatches. round_bf16: under Rounding::Levanter the per-microbatch gain gradient is rounded to bf16 before the
// f32 accumulation (levanter's parameter cotangents are bf16; numerics.md §2.6).
void reduce_partials_rows(const float* partial, int n_blocks, int width, float* out, bool accumulate, bool round_bf16,
                          cudaStream_t s);                                                   // k_reduce_rows

// Sum of n f32 values in a fixed binary-tree order (one CTA); used for loss sums and eval per-batch sums.
void sum_fixed(const float* x, int64_t n, float* out, cudaStream_t s);                       // k_sum_fixed

// Generic statistic at a hook point (model.hpp StatSpec): per-CTA partials over [rows, cols] (optionally a column or
// position range), written to stat partial slots; k_stat_finalize turns them into the slot values. Kinds: sum, sumsq,
// absmax, Dot (<a,b>, ||a||^2, ||b||^2 from a and b), Vector (width values per row group), PosRangeMean (loss_tok
// over positions [pos_begin, pos_end) of every row). The cross-rank reduction (Sum, Max, PerRank) follows in the
// metrics reduction (comm.hpp).
void stat_partials(const void* x, const void* x_b, DType dt, int64_t rows, int64_t cols, int kind, int width,
                   int pos_begin, int pos_end, int S, float* partial, cudaStream_t s);
void stat_finalize(const float* partial, int n_blocks, int kind, int width, int64_t count, float* slot,
                   cudaStream_t s);

}  // namespace nm
