// Internal to src/sched (and its tests): the levanter schedule shape, the host port of the XLA:GPU arithmetic the
// jitted optax schedules run, the fixture format and ulp statistics. The public entry points are load_schedules and
// build_schedules (marincpp/optim.hpp) and nm_schedule_tables (marincpp/capi.h).
//
// Why a port of XLA:GPU and not the literal optax expressions (S2 finding, docs/BUILD_LOG.md): XLA's algebraic
// simplifier rewrites `x / C` into `x * f32(1 / C)` and folds `(pi * c) / T` into `c * f32(f32(pi) * f32(1 / T))`
// (the optimized HLO in cache/levref/<size>/schedules_manifest/vec.opt_hlo.txt); cos and pow are libdevice's
// __nv_cosf and __nv_powf inlined into the fusion. Evaluating optim.md §6 literally in f32 (IEEE division) differs from
// the fixture at 5-25% of the steps whatever cos it uses: 300m main table, 3030 of 11444 entries with a correctly
// rounded cos (math.cos of the f32 argument, rounded to f32), 2786 with numpy's float32 cos; the first at k = 58,
// inside the warmup, where no cos is involved. The port reproduces the fixture at every k of all four sizes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/optim.hpp"

namespace nm::sched {

// One levanter cycle (LV/optim/config.py:286-362, no cooldown, no cycles): [warmup | stable | decay], N = sum.
enum class DecayKind : uint8_t { Linear = 0, Cosine = 1 };
struct Shape {
  int64_t n = 0;          // num_train_steps
  int64_t warmup = 0;     // linear 0 -> peak over `warmup` steps (absent when 0)
  int64_t stable = 0;     // constant peak (absent when 0)
  int64_t decay = 0;      // T_d: linear or cosine peak -> min (absent when 0: a constant segment instead)
  DecayKind kind = DecayKind::Linear;
  double min_lr_ratio = 0.0;
};
// levanter's _convert_frac_or_steps: x <= 1 is the fraction int(x * N), x > 1 must be whole steps (any finite size:
// past N - warmup the decay is clipped, as levanter clips it). optim.decay = -1 is levanter's None (decay =
// N - warmup). optim.warmup = 1 is refused: levanter reads it as the fraction 1.0 (= N).
Shape shape(const Config& c);

// eta(k) for k = 0..n-1 with peak `peak` (a decimal knob value; every scalar constant is made as levanter makes it,
// optim.md §5.3 rule 1, and then folded as XLA folds it).
void lr_table(const Shape& s, double peak, float* out);
// 1 - b^t for t = 1..n (the bias-correction denominator of update t - 1), b = f32(beta); `knob` names beta in errors.
// Refuses f32(beta) outside [0, 1) and n >= 2^24 (the range of xla_powf).
void bc_table(double beta, const char* knob, int64_t n, float* out);

// libdevice __nv_cosf and __nv_powf as XLA:GPU inlines them (the CUDA 13 libdevice of the fixtures, xla_math.cpp; no
// FTZ, no fast math): explicit fused multiply-adds (std::fma), every other operation rounded to f32. Only the ranges
// the schedules reach are supported: |x| < 105615 for cos, 0 <= b < 1 and integer 0 <= t < 2^24 for pow (anything
// else is refused, not approximated).
// powf's one hardware-approximate step, rcp.approx.ftz.f32 of m + 1 (m = the mantissa of b in [0.707, 1.414)), is
// emulated as the correctly rounded 1/x; the A100's rcp differs from it at 627,759 of the 4,613,735 f32 x in
// [1.7, 2.5]. Effect, measured against XLA:GPU's 1 - b**t on the A100 over every f32 b of a binade: b in [0.5, 1)
// differs at 481, 521, 473, 241, 13 of the 2^23 bases for t = 2, 3, 5, 10, 37 and at none for t = 1 or t >= 100
// (100, 300, 1000, 4000 tested); b in [0.25, 0.5) at 112, 68, 15 for t = 2, 3, 5 and none from t = 10. With the
// measured rcp in place of 1/x the port equals XLA at all of them. 16 common betas (0.01 to 0.9999, incl. 0.9, 0.95,
// 0.98, 0.99, 0.999) equal XLA for t = 1..2000. A fixture, when given, is the record (load_schedules).
float xla_cosf(float x);
float xla_powf(float b, float t);
inline constexpr int64_t kMaxPowExponent = int64_t{1} << 24;

// Distance in units in the last place between two finite floats (0 for +0 / -0).
int64_t ulp_distance(float a, float b);
struct UlpStats {
  int64_t n_diff = 0;     // entries whose bits differ
  int64_t max_ulp = 0;
  int64_t first = -1;     // first differing index
};
UlpStats compare(const std::vector<float>& a, const std::vector<float>& b);

// ----------------------------------------------------------------------------------------------- fixture format
// A fixture directory (tests/s2/schedule_fixture.py): schedule.toml + sched_main.f32, sched_adam.f32, bc1.f32, bc2.f32
// (little-endian float32; sched_* indexed by k = 0..N-1, bc* by t = 1..N). schedule.toml:
//   format = 1, size = "<size>", [levanter] num_train_steps, lr_schedule, learning_rate, adam_lr, warmup, decay
//   (-1 = None), min_lr_ratio, beta1, beta2; [files] <table> = "<file>"; [sha256] <table> = "<hex of the file>".
inline constexpr int kFixtureFormat = 1;
inline constexpr const char* kTables[4] = {"sched_main", "sched_adam", "bc1", "bc2"};
// The content hash config resolution records for optim.schedule_fixture ("<path>#sha256=<hex>"), computed by
// resolve() itself, so the loader and the run identity share one implementation.
std::string dir_content_hash(const std::string& dir);
// Reads and checks a fixture against the config (every [levanter] field, table sizes, per-file sha256). The returned
// tables carry source "fixture:<content hash>"; n_ulp_diffs / max_ulp_diff are left 0. load_schedules fills them from
// the formula cross-check, or sets both to -1 when the formula refuses the config (the check is unavailable; the
// fixture still loads; build_schedules(c) states the reason).
ScheduleTables read_fixture(const Config& c, const std::string& dir);
// Per-table comparison (sched_main, sched_adam, bc1, bc2) of two table sets.
struct TableDiffs {
  UlpStats t[4];
};
TableDiffs compare_tables(const ScheduleTables& a, const ScheduleTables& b);

}  // namespace nm::sched
