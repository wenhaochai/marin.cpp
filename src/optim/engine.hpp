// Optimizer engine (stage S6): one rank's optimizer step on device, built from groups-as-data and the tensors this
// rank owns. DESIGN.md §4 (groups, clipping, MuonH, AdamH, Adam, per-row control), §6.1 (ZeRO-1 ownership);
// optim.md §3-§5 (the math, operation order and constants).
//
// The engine is the S6 core behind optim.hpp: S7 builds the EngTensor list from Model::params() and the Ownership
// (engine_layout() below), fills in the device pointers of its buffers, and records the phases into g_step with its
// cross-rank all-reduces between them. Every phase is graph-capturable after prepare(): no allocation, no host
// synchronization, the update index k is read from device memory (Tables::step), every reduction has a fixed order.
//
// One step, in order (optim.md §3):
//   record_sumsq    per-slice sums of squares of g * grad_scale (f64) -> clip_sumsq()[n_clip]   (local part)
//   -- all-reduce clip_sumsq over ranks when sharded() --
//   record_update   clip state per clip group; Adam (one pass); MuonH per NS chunk (Nesterov momentum into the NS
//                   buffer, X / (||X|| + eps), Newton-Schulz, hyperball in three passes); AdamH pass 1 (moments,
//                   ||p||^2, ||a||^2) -> slot_pa()[n_slots][2]
//   -- all-reduce slot_pa when sharded() --
//   record_adamh_w  AdamH pass 2 (||w||^2) -> slot_w()[n_slots]
//   -- all-reduce slot_w when sharded() --
//   record_finish   AdamH pass 3 (p + u), then optionally k++ (k_step_advance)
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "kernels/optim/ns_gemm.hpp"
#include "kernels/optim/optim_dev.hpp"
#include "marincpp/config.hpp"
#include "marincpp/model.hpp"
#include "marincpp/optim.hpp"

namespace nm::opt {

// A resolved optimizer group. The decimal hyperparameters stay f64 so the kernels' f32 constants are made as levanter
// makes them (optim.md §5.3 rule 1: f32(1 - b) from the f64 difference).
struct EngGroup {
  std::string name;
  OptRule rule = OptRule::Adam;
  int lr_table = 0;            // index into the lr tables: 0 main, 1 adam, then custom
  double lr_scale = 1.0;
  double b1 = 0.9, b2 = 0.98, eps = 1e-20, momentum = 0.95;
  double max_grad_norm = 1.0;
  double weight_decay = 0.0;   // Adam only (decoupled), groups that list it
  std::string clip_group;      // groups sharing a name share one clip norm; empty = own name
};
// muonh, adamh, adam of optim.md §1 from the config (optim.lr / optim.adam_lr tables 0 / 1).
std::vector<EngGroup> baseline_groups(const Config& c);
// The f32 constants the kernels read (optim.md §5.3 rule 1: f32 of each decimal, f32(1 - b) from the f64 difference).
optk::GroupDev group_dev(const EngGroup& g, int clip_index);

// One MuonH layer matrix [out, in = the tensor's cols]. Its rows are full-tensor storage rows row0 .. row0 + out - 1,
// or row_map[r] for fused buffers whose leaves interleave row blocks.
struct EngMatrix {
  std::string name;
  int64_t row0 = 0;
  int64_t out = 0;
  std::vector<int32_t> row_map;
};

// One tensor as the engine sees it. Pointers address full-tensor storage row `base_row` (0 when the whole tensor is
// allocated on this rank, the first owned row when only the owned shard is).
struct EngTensor {
  std::string name;
  int group = -1;                         // index into the EngGroup list
  int64_t rows = 0, cols = 0;             // the full tensor as [rows, cols]
  int64_t base_row = 0;
  float* p32 = nullptr;
  bf16_t* p16 = nullptr;                  // optional compute copy, written as bf16(p) after the update
  const float* g = nullptr;               // mean gradient (on the owner)
  float* s0 = nullptr;                    // MuonH momentum buffer, Adam/AdamH mu
  float* s1 = nullptr;                    // Adam/AdamH nu
  float* u_out = nullptr;                 // optional: the applied update u is stored here too (tests)
  float grad_scale = 1.0f, update_scale = 1.0f;
  int row_partition = -1;                 // index into the RowPartition list (Adam, AdamH)
  std::vector<EngMatrix> mats;            // MuonH: every layer matrix (owned whole by one rank)
  std::vector<std::pair<int64_t, int64_t>> owned;   // row ranges this rank updates (MuonH: whole matrices' rows)
};

struct EngOptions {
  optk::NsPrec ns = optk::NsPrec::TF32;
  int ns_steps = 5;
  double ns_a = 3.4445, ns_b = -4.7750, ns_c = 2.0315;   // "simple": the same triple every iteration
  double muon_eps = 1e-5;
  size_t ns_chunk_bytes = size_t(1) << 30;   // NS buffers (X, X', A, cA, B) of one chunk of same-shape matrices
  bool timed_gemm = false;                   // NS GEMM algorithm by timed search (else cuBLASLt's first heuristic)
  bool ns_symmetric = false;                 // X X^T and cA A as 3 of 4 blocks + mirror (ns_gemm.hpp; not 3xTF32;
                                             // T1.6 / T1.7 hold; slower on the H100 below 1.2B: BUILD_LOG S6)
  bool ns_fold_c = false;                    // B = c (A A) + b A (alpha = c): no cA pass; TF32 operand A, not c A
  bool sharded = false;                      // sums must be all-reduced over ranks (ZeRO-1 with ranks > 1)
};

class Engine {
 public:
  Engine(std::vector<EngGroup> groups, std::vector<EngTensor> tensors, std::vector<RowPartition> row_partitions,
         std::vector<const uint8_t*> touched_rows, EngOptions opt);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Device workspace (sums, clip state, NS buffers, NS GEMM workspace). bind(nullptr) allocates it.
  size_t workspace_bytes() const;
  void bind(void* workspace);
  // Device tables the step reads: lr tables indexed by k (0 main, 1 adam, then custom), 1 - b^(k+1) of the b1 / b2
  // every Adam and AdamH group shares, and k itself; every table has n_steps entries (a step at k >= n_steps traps).
  void set_tables(const std::vector<const float*>& lr_tables, const float* bc1, const float* bc2, int32_t* step,
                  int64_t n_steps);
  // NS GEMM algorithms (timed search when EngOptions::timed_gemm). Before capture.
  void prepare(cudaStream_t s);
  std::string save_gemm_algos() const;
  void load_gemm_algos(const std::string& bytes);

  void record_sumsq(cudaStream_t s);
  // The same in two parts, for the trainer's per-bucket schedule (DESIGN.md §4.1: partials of a bucket's tensors on
  // the comm stream right after its reduction): the slice partials of one tensor, then the per-clip-group sums once
  // every tensor's partials exist. record_sumsq == every record_sumsq_tensor, then record_clip_sums.
  void record_sumsq_tensor(int tensor, cudaStream_t s);
  void record_clip_sums(cudaStream_t s);
  // clip_override (device, [n_clip]) replaces the clip state computed from clip_sumsq (C ABI tests of one tensor).
  void record_update(cudaStream_t s, const optk::ClipDev* clip_override = nullptr);
  void record_adamh_w(cudaStream_t s);
  void record_finish(cudaStream_t s, bool advance_step, float* lr_ring = nullptr);
  using AllReduce = std::function<void(double* buf, int n, cudaStream_t s)>;
  void record_step(cudaStream_t s, const AllReduce& allreduce_sum, bool advance_step, float* lr_ring = nullptr);

  bool sharded() const { return opt_.sharded; }
  double* clip_sumsq() const { return clip_sumsq_; }
  int n_clip() const { return n_clip_; }
  double* slot_pa() const { return slot_pa_; }   // [n_slots][2]
  double* slot_w() const { return slot_w_; }
  int n_slots() const { return n_slots_; }
  const optk::ClipDev* clip_states() const { return clip_dev_; }   // [n_clip], after record_update
  const std::vector<std::string>& clip_names() const { return clip_names_; }
  int n_matrices() const { return static_cast<int>(mat_count_); }
  int n_ns_chunks() const { return static_cast<int>(chunks_.size()); }

 private:
  struct Chunk {
    int m_begin, m_end;        // matrices (MatTab order)
    int64_t tile_begin, tile_end;
    int64_t m, n;              // oriented shape
    optk::NsIterPlan ns;       // the chunk's NS GEMMs
  };
  void build();
  void build_tables_on_device();

  std::vector<EngGroup> groups_;
  std::vector<EngTensor> tensors_;
  std::vector<RowPartition> rps_;
  std::vector<const uint8_t*> touched_;
  EngOptions opt_;
  optk::NsGemm gemm_;

  // host plan
  std::vector<std::string> clip_names_;
  std::vector<int> group_clip_;
  std::vector<float> clip_max_;
  int n_clip_ = 0, n_slots_ = 0;
  struct SliceH { int tensor; int64_t row; int col; int len; int block; int slot; };
  std::vector<SliceH> slices_;
  int sumsq_begin_ = 0, sumsq_end_ = 0, adam_begin_ = 0, adam_end_ = 0, adamh_begin_ = 0, adamh_end_ = 0;
  std::vector<int32_t> clip_idx_, slot_idx_;     // slice order per segment
  std::vector<std::pair<int, int>> sumsq_of_tensor_;   // [b, e) of each tensor's sumsq slices
  std::vector<int64_t> clip_seg_, slot_seg_;
  struct MatH { int tensor; int64_t row0; int map; int32_t out, in; bool trans; int64_t x_off; };
  std::vector<MatH> mats_;
  std::vector<std::vector<int32_t>> row_maps_;
  std::vector<int64_t> tile0_;
  size_t mat_count_ = 0;
  std::vector<Chunk> chunks_;
  int64_t ns_x_elems_ = 0, ns_a_elems_ = 0;

  // device
  void* tables_blob_ = nullptr;   // owned: pointer arrays and index tables
  void* ws_ = nullptr;
  bool own_ws_ = false;
  optk::TensorTab tt_{};
  optk::SliceTab st_{};
  optk::RowPartTab rt_{};
  optk::MatTab mt_{};
  optk::Tables tb_{};
  optk::GroupDev* groups_dev_ = nullptr;
  float* clip_max_dev_ = nullptr;
  int32_t *clip_idx_dev_ = nullptr, *slot_idx_dev_ = nullptr;
  int64_t *clip_seg_dev_ = nullptr, *slot_seg_dev_ = nullptr;
  const float** lr_tab_dev_ = nullptr;
  const float* lr_main_ = nullptr;
  const float* lr_adam_ = nullptr;
  // workspace views
  double* slice_partial_ = nullptr;   // [n_slices * 2]
  double* tile_partial_ = nullptr;    // [n_tiles * 2]
  double* mat_u2_ = nullptr;          // [n_mats]
  double* mat_po_ = nullptr;          // [n_mats * 2]
  double* mat_w_ = nullptr;           // [n_mats]
  double* clip_sumsq_ = nullptr;
  double* slot_pa_ = nullptr;
  double* slot_w_ = nullptr;
  optk::ClipDev* clip_dev_ = nullptr;
  float *ns_x_ = nullptr, *ns_x2_ = nullptr, *ns_a_ = nullptr, *ns_ca_ = nullptr, *ns_b_ = nullptr;
  void* gemm_ws_ = nullptr;
  bool prepared_ = false;
  bool tables_set_ = false;
};

// ------------------------------------------------------------------------------------------------ ownership (ZeRO-1)
// DESIGN.md §6.1: every MuonH layer matrix (one leaf of one layer) has one owner (whole matrix: NS needs it); every
// vocab-sized tensor is split by rows over all ranks (contiguous, as even as possible); every other tensor has one
// owner. Owners by LPT (largest first onto the least-loaded rank, ties to the lower rank) on the estimated H100 time
// of NS (TF32) plus the elementwise bytes. zero1 = false or ranks = 1: every slice has owner -1 (every rank).
// `group_rules[i]` is the rule of ParamSpec group name `group_names[i]`.
Ownership plan_ownership_specs(const std::vector<ParamSpec>& params, const std::vector<EngGroup>& groups, int nranks,
                               bool zero1);
// The EngTensor list of one rank: group indices, shapes, MuonH matrices from the leaves (one per LeafSlice of a
// LayerMatrix-scoped parameter; ParamSpec::cols = in), owned row ranges from the ownership. Pointers are left null
// for the caller (S7 binds its buffers).
std::vector<EngTensor> engine_layout(const std::vector<ParamSpec>& params, const std::vector<EngGroup>& groups,
                                     const Ownership& own, int rank);

}  // namespace nm::opt
