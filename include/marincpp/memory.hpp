// Static memory plan and arena. DESIGN.md §2.
//
// One cudaMalloc per GPU at startup, sized by the plan; nothing is allocated after the first step. The plan is a pure
// function of (Config, ModelShape, ExecShape, DatasetMeta) and runs on the CPU (`marincpp plan`), so a run that does
// not fit is refused before it is queued (reliability rule R1).
#pragma once

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

struct Config;
struct ModelShape;
struct ExecShape;
struct DatasetMeta;

// Lifetime decides placement:
//   Persistent  - owner shards of masters and optimizer state, replicated bf16 copies, gradients, tables, rings.
//   Microbatch  - lives for one microbatch's whole forward and backward but belongs to no single layer: depth sources
//                 (attnres), readout states, the strict-mode per-microbatch head-gradient accumulator.
//   Activation  - kept from a layer's forward to its backward within one microbatch; one copy per layer.
//   Transient   - lives inside one layer's forward or backward; all transients of the same role share one slot.
//   Workspace   - scratch of one kernel family (FA3 accumulators, CE chunk, sort temp, cuBLASLt); reused by all calls.
enum class Lifetime : uint8_t { Persistent, Microbatch, Activation, Transient, Workspace };

// Activation policy: core.hpp Remat. Per tensor: who may skip storing it.
enum class Recompute : uint8_t { Never, Elementwise, Layer };

struct BufferSpec {
  std::string name;          // unique: "act.L3.gu", "grad.f32", "ws.ce.logits" (printed in the plan; bind key)
  std::string group;         // row of the grouped plan (DESIGN.md §2.5): "act.L*.gu"; empty = the name
  Lifetime life;
  DType dtype;
  int ndim = 1;
  int64_t shape[4] = {0, 0, 0, 0};   // checked at bind time
  int layer = -1;            // -1 = not per layer
  Recompute recompute = Recompute::Never;
  bool host_pinned = false;  // true: cudaHostAlloc'd (checkpoint staging, token slots), not in the device arena
  int64_t numel() const;     // saturates at INT64_MAX
  uint64_t bytes() const;    // numel x dtype size, saturating at kMaxBufferBytes
};
// A buffer request above 1 PiB counts as exactly 1 PiB: such a plan never fits, and the sum over a plan's buffers
// cannot overflow (knob ranges allow absurd shapes; the planner must report "does not fit", not wrap around).
inline constexpr uint64_t kMaxBufferBytes = uint64_t(1) << 50;
const char* lifetime_name(Lifetime l);
// The four subtotals of DESIGN.md §2.5: P = Persistent, A = Microbatch + Activation, T = Transient, W = Workspace.
char lifetime_class(Lifetime l);

struct BufferId {
  int32_t idx = -1;
  bool valid() const { return idx >= 0; }
};

// Choice the planner makes when exec.micro_seqs / exec.remat / optim.zero1 / exec.ce_chunk_rows are "auto".
struct PlanChoice {
  int micro_seqs = 0;
  Remat remat = Remat::All;
  bool zero1 = true;            // ZeRO-1 ownership for every optimizer group (DESIGN.md §6.1)
  bool zero2_vocab = false;     // reduce-scatter vocab-sized gradients after every microbatch; not implemented yet:
                                // always false (auto resolves to off, validate() refuses on; build_plan has no ZeRO-2)
  int ce_chunk_rows = 8192;     // smallest chunk with head-grad RMW <= 0.5% of the step, capped by fit (§3.8)
  uint64_t device_bytes = 0;    // arena size: device_bytes() of the finalized plan (guard bands if exec.debug_arena)
  uint64_t pinned_bytes = 0;    // host pinned
};

class MemoryPlan {
 public:
  BufferId add(BufferSpec spec);   // fatal on a duplicate name
  BufferId find(const std::string& name) const;   // invalid id if absent
  // Lays out every buffer: Persistent, Microbatch, one Activation region per layer, then the Transient and Workspace
  // slots (each slot sized by the largest member). Offsets are 256-byte aligned; with canaries, a 4 KiB guard band
  // precedes and another follows every device buffer (an underrun or an overrun lands in a band, never in alignment
  // padding). Fails (fatal) above `budget_bytes`.
  void finalize(uint64_t budget_bytes, bool canaries);
  uint64_t device_bytes() const;   // arena size after finalize: payload + 256-byte alignment (+ guard bands)
  uint64_t pinned_bytes() const;
  uint64_t payload_bytes() const;  // sum of the device buffers' bytes; the number scripts/design_budget.py prints
  uint64_t payload_bytes(char lifetime_class) const;
  int size() const { return static_cast<int>(specs_.size()); }
  bool finalized() const { return finalized_; }
  bool canaries() const { return canaries_; }
  uint64_t offset(BufferId id) const;
  const BufferSpec& spec(BufferId id) const;
  // Grouped rows in first-declaration order: (group, buffers, bytes).
  struct Group { std::string name; Lifetime life; int count; uint64_t bytes; };
  std::vector<Group> groups() const;
  // Prints one line per buffer (name, life, dtype, shape, bytes, offset) and subtotals per lifetime, then
  // "TOTAL <bytes> of budget <bytes>". The training binary prints it at start; `marincpp plan` prints it alone.
  void print(FILE* f) const;
  void print_grouped(FILE* f) const;   // act.L*.x summed over layers, as in DESIGN.md §2.5

 private:
  std::vector<BufferSpec> specs_;
  std::vector<uint64_t> offsets_;
  std::unordered_map<std::string, int32_t> index_;
  uint64_t device_bytes_ = 0, pinned_bytes_ = 0;
  bool finalized_ = false, canaries_ = false;
};
inline constexpr uint64_t kCanaryBytes = 4096;  // guard band before and after every buffer in canary/poison modes

// Builds the full plan for one rank. Ranks differ only in owner-shard sizes (whole MuonH matrices and row shards are
// assigned by LPT at plan time). Every module declares its buffers through its own declare_*(); the plan of a run with
// several step regimes is the maximum over regimes (DESIGN.md §2.4).
MemoryPlan build_plan(const Config& c, const ModelShape& m, const ExecShape& e, const DatasetMeta& d);

// Chooses (micro_seqs, remat, zero1, zero2_vocab, ce_chunk_rows) when "auto": the largest micro_seqs dividing
// seqs_per_rank that fits with remat=all; remat=min if that gives at least twice the tokens of a microbatch below
// 16,384 tokens; remat=layer only when requested or when nothing else fits. Writes rejected candidates to `report`.
// Fit rule: a candidate fits iff device_bytes() of its finalized plan (payload + 256-byte alignment, + guard bands when
// exec.debug_arena is on) is at most exec.mem_budget_gib; `marincpp plan` and the trainer use the same measure. While
// choosing micro_seqs / remat the CE chunk is exec.ce_chunk_rows when set, else ce_rows_floor(micro tokens). The CE
// chunk rule (§3.8) then takes, for a microbatch of at least 8,192 tokens, the smallest 8,192 x 2^i rows (at most the
// microbatch) whose head-gradient read-modify-write is <= 0.5% of the step budget, capped by what fits with 2 GiB
// headroom on the payload (scripts/design_budget.py's cap); a smaller microbatch takes one chunk of its tokens rounded
// up to a multiple of 128. `device_bytes` of the result is the arena of the chosen plan; 0 means nothing fits.
PlanChoice choose_plan(const Config& c, const ModelShape& m, const DatasetMeta& d, std::string* report);
// CE chunk while choosing micro_seqs: min(8192, micro tokens) rounded up to a multiple of 128 (the exec.ce_chunk_rows
// check: a resolved config must pass its own validate() for every seq_len).
int ce_rows_floor(int64_t micro_tokens);
uint64_t mem_budget_bytes(const Config& c);

// ------------------------------------------------------------------------------------------------ step budget
// The per-GPU step-time model of DESIGN.md §12 (port of scripts/design_budget.py perf(), same operation order, so the
// numbers agree bitwise in double). Integer products are exact (__int128, as Python's ints), converted to double where
// Python converts. Used by the CE chunk rule and printed by `marincpp-train plan`.
enum class CeStage : uint8_t { B1 = 0, B1_5 = 1, B2 = 2 };
enum class NsPrecision : uint8_t { TF32 = 0, F32 = 1, X3TF32 = 2 };
struct StepBudget {
  std::vector<std::pair<const char*, double>> rows;   // seconds, in the order of DESIGN.md §12.2
  double step_s = 0;
  double eval_s = 0;          // one Paloma eval (forward only, every readout)
  int n_evals = 0;            // init eval + periodic evals + final eval
  double tokens_per_s = 0;
  double mfu_pct = 0;         // levanter's formula (model.md §6)
  double row(const char* name) const;
};
StepBudget step_budget(const Config& c, const ModelShape& m, const ExecShape& e, CeStage stage, bool fused,
                       NsPrecision ns);

// Debug modes (exec.debug_arena, local): Canary checks the guard bands after every step (fatal with the buffer name);
// Poison also fills every Transient and Workspace buffer with NaN before each use. The recompute slots
// (tr.recompute, tr.layer_recompute) are Transient. Persistent, Microbatch and Activation buffers are never poisoned:
// an Activation buffer is saved state the backward reads, whatever its Recompute tag says (under remat=all the
// Elementwise rows a, qk_rot, m, h are saved tensors).
enum class ArenaDebug : uint8_t { Off = 0, Canary = 1, Poison = 2 };
bool poison_target(const BufferSpec& b);   // the buffers poison_transients() fills

// The arena of one GPU. All device pointers in the step graphs point into it, so graph replay is valid forever.
// Owns its allocations: movable, not copyable; the destructor releases.
class Arena {
 public:
  Arena() = default;
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&& o) noexcept;
  Arena& operator=(Arena&& o) noexcept;
  void allocate(const MemoryPlan& plan, int device, ArenaDebug dbg);  // one cudaMalloc + one cudaHostAlloc
  void release();   // frees everything and resets to the default state; safe to call twice
  // Bind by declared name with dtype and shape checks; a mismatch is fatal at bind time. No positional lookup exists.
  void* bind(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const;
  template <class T> T* bind_as(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const {
    return static_cast<T*>(bind(name, dtype, shape));
  }
  TensorView view(const std::string& name) const;
  void check_canaries(cudaStream_t s) const;     // enqueues the guard-band check (debug modes only)
  // After the stream of check_canaries() completed: the name of the first buffer whose guard band changed, or "".
  std::string canary_failure() const;
  void poison_transients(cudaStream_t s) const;  // Poison mode: NaN fill, recorded before each layer's use
  // After warm-up the trainer asserts cudaMemGetInfo is unchanged at every eval and checkpoint (no hidden allocation).
  uint64_t free_bytes_at_start() const { return free_at_start_; }

 private:
  const MemoryPlan* plan_ = nullptr;
  char* dev_ = nullptr;
  char* host_ = nullptr;
  ArenaDebug dbg_ = ArenaDebug::Off;
  uint64_t free_at_start_ = 0;
  int device_ = -1;
  uint64_t* d_guards_ = nullptr;   // device table of guard-band offsets (canary modes): 2 per device buffer (before,
                                   // after), in declaration order; allocated with the arena
  int n_guards_ = 0;
  int* d_bad_ = nullptr;           // first corrupted buffer index (device), copied to h_bad_ (pinned)
  int* h_bad_ = nullptr;
};

}  // namespace nm
