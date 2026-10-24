// Optimizer plan (host side): groups as data, per-row control, ZeRO-1 ownership, schedules, Newton-Schulz batching.
// DESIGN.md §4, §6.1.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/kernels/gemm.hpp"
#include "marincpp/kernels/optim_kernels.hpp"
#include "marincpp/model.hpp"

namespace nm {

// ------------------------------------------------------------------------------------------------ schedules
// The tables of record come from a fixture (optim.schedule_fixture): tools/levref/schedules.py runs the jitted optax
// schedules of the levanter config on a GPU (XLA:GPU f32 cos and pow; the same code on sm_80 and sm_90) and dumps
// sched_main(k), sched_adam(k) for k = 0..N-1 and 1 - b1^t, 1 - b2^t for t = 1..N. T0.7 checks the fixture against
// every logged W&B value bitwise.
// build_schedules() evaluates optim.md §6 on the host (-ffp-contract=off, exact f32 operation order):
//   linear (130m):  k < stable: peak; else c = clip(k - stable, 0, T_d); frac = 1 - f32(c)/f32(T_d);
//                   eta = (peak - end) * frac + end
//   warmup+cosine:  k < W: c = clip(k, 0, W); eta = (0 - peak) * (1 - f32(c)/f32(W)) + peak
//                   else c = min(k - W, T_d); eta = peak * ((1 - 0) * (0.5f * (1 + cos((f32(pi) * c) / T_d))) + 0)
// With a fixture it is only a cross-check that counts ulp differences (logged); it never overrides a fixture value.
// Without a fixture (non-parity runs) the formula is used and events.jsonl records schedule_source = formula.
struct ScheduleTables {
  std::vector<float> lr;        // main eta(k): MuonH, AdamH (and groups that use the main table)
  std::vector<float> adam_lr;   // Adam eta_a(k)
  std::vector<float> bc1, bc2;  // 1 - b^t at t = k + 1
  std::string source;           // "fixture:<sha256>" or "formula"
  int n_ulp_diffs = 0;          // formula vs fixture
  int max_ulp_diff = 0;
};
ScheduleTables load_schedules(const Config& c);   // fixture if set (required when run.parity), else formula
ScheduleTables build_schedules(const Config& c);  // the formula alone

// ------------------------------------------------------------------------------------------------ groups as data
// One optimizer group (DESIGN.md §4.1). Baseline: muonh, adamh, adam. Variants add groups by GroupSpec file
// (optim.groups) or by declaring ParamSpec::group names their own code registers: e.g. depth_adam (hc/mhc/attnres
// parameters, Adam eps 1e-8, DepthArchMuonHConfig), lns steps (gradient times c_l before the clip, update times c_l
// after: per-tensor grad_scale / update_scale), attnres query clip (post-update kernel).
enum class OptRule : uint8_t { MuonH = 0, AdamH = 1, Adam = 2, Frozen = 3 };
enum class LrTable : uint8_t { Main = 0, Adam = 1, Custom = 2 };
using PostUpdateFn = void (*)(DeviceContext& d, const struct GroupPlan& g, cudaStream_t s);
struct GroupSpec {
  std::string name;
  OptRule rule = OptRule::Adam;
  LrTable lr_table = LrTable::Main;
  int custom_table = -1;        // DeviceSchedules::custom index when lr_table == Custom
  float lr_scale = 1.0f;
  float eps = 1e-20f, b1 = 0.9f, b2 = 0.98f, momentum = 0.95f;
  std::string clip_group;       // groups sharing a name share one clip norm; default = own name
  float max_grad_norm = 1.0f;
  bool weight_decay = false;    // only groups that list it get optim.weight_decay (validate() guards the value)
  PostUpdateFn post_update = nullptr;
};
// Per-tensor scalars a group applies (lns): g' = g * grad_scale before the clip sum; u' = u * update_scale.
struct TensorScales { int param = -1; float grad_scale = 1.0f; float update_scale = 1.0f; };

// Per-row control (DESIGN.md §4.5): a partition of the rows of one tensor into blocks; per block an lr-scale schedule,
// a frozen flag (gradient zeroed before the clip and the AdamH norms; update zeroed) and its own AdamH norm scope.
// freeze_rows_muonh.py block mode: kept rows [0, lo) + the last 256 specials (own norm), rare rows (own norm, lr scale).
struct RowBlock {
  int lr_table = -1;            // DeviceSchedules::custom index of the lr multiplier schedule, or -1 for 1
  bool frozen = false;
  int freeze_from_step = -1;    // step-gated freeze (regime boundary), -1 never
  bool own_norm = true;
};
struct RowPartition {
  int param = -1;
  std::vector<int32_t> block_of_row;   // [rows]
  std::vector<RowBlock> blocks;
  bool lazy = false;            // update only rows touched this step (InputLayer::touched_rows)
};

struct GroupPlan {
  GroupSpec spec;
  int clip_index = -1;          // index into the clip-group arrays
  std::vector<int> params;      // indices into Model::params()
  std::vector<int> buckets;     // gradient buckets that must be reduced before this group starts (DESIGN.md §2.4)
};

// ------------------------------------------------------------------------------------------------ ZeRO-1 ownership
// Every MuonH matrix (one logical leaf of one layer) has one owner rank (whole matrix: NS needs it); every vocab-sized
// tensor is split by rows over all ranks; small tensors have one owner. Assignment by LPT on (NS cost + elementwise
// bytes). With optim.zero1=off every rank owns everything (replicated optimizer, all-reduce).
struct Ownership {
  struct Slice { int param; int64_t row_begin, row_end; int owner; };
  std::vector<Slice> slices;    // covers every parameter exactly once
  std::vector<double> load_per_rank;
};
Ownership plan_ownership(const Model& model, const std::vector<GroupPlan>& groups, int nranks, bool zero1);

// Newton-Schulz work on one rank: the matrices it owns, batched by oriented shape (wide orientation m <= n).
struct NsBatch {
  int64_t m, n;
  std::vector<int> matrices;    // MuonMatrices indices owned by this rank
};
struct NsPlan {
  std::vector<NsBatch> batches;
  Compute compute;              // TF32, F32_PEDANTIC or F32_3XTF32 (optim.ns_precision resolved, O3)
  bool symmetric_products;      // X X^T and A A computed as upper-triangle tiles
  float a, b, c;                // f32 of (3.4445, -4.7750, 2.0315)
  int steps;
};

struct OptimizerPlan {
  std::vector<GroupPlan> groups;
  std::vector<TensorScales> scales;
  std::vector<RowPartition> row_partitions;
  Ownership own;
  MultiTensor muon, adamh, adam;  // device tables over this rank's owned slices, built at bind time
  MuonMatrices mats;
  NsPlan ns;
};

OptimizerPlan build_optimizer_plan(const Config& c, const Model& model, const ExecShape& e, int rank);
void declare_optimizer(MemoryPlan& plan, const OptimizerPlan& p, const Model& model);
// Records the optimizer section of g_step: per clip group (each after its own buckets' events): cross-rank sum of the
// sum-of-squares partials (computed on comm per bucket) -> clip state -> updates of the group's owned slices (MuonH:
// momentum, NS, hyperball; AdamH; Adam) -> post-update hooks. Then the metrics all-reduce and the StepRecord (comm.hpp,
// metrics.hpp), and k++. Reads k from device memory; no host values are baked in.
void record_optimizer_step(DeviceContext& d, const OptimizerPlan& p, const int32_t* step, const DeviceSchedules& s,
                           float* ring_slot);

}  // namespace nm
