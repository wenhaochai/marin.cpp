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
  Lifetime life;
  DType dtype;
  int ndim = 1;
  int64_t shape[4] = {0, 0, 0, 0};   // checked at bind time
  int layer = -1;            // -1 = not per layer
  Recompute recompute = Recompute::Never;
  bool host_pinned = false;  // true: cudaHostAlloc'd (checkpoint staging, token slots), not in the device arena
  int64_t numel() const;
};

struct BufferId {
  int32_t idx = -1;
  bool valid() const { return idx >= 0; }
};

// Choice the planner makes when exec.micro_seqs / exec.remat / optim.zero1 / exec.ce_chunk_rows are "auto".
struct PlanChoice {
  int micro_seqs = 0;
  Remat remat = Remat::All;
  bool zero1 = true;            // ZeRO-1 ownership for every optimizer group (DESIGN.md §6.1)
  bool zero2_vocab = false;     // reduce-scatter vocab-sized gradients after every microbatch
  int ce_chunk_rows = 8192;     // smallest chunk with head-grad RMW <= 0.5% of the step, capped by fit (§3.8)
  uint64_t device_bytes = 0;    // arena size
  uint64_t pinned_bytes = 0;    // host pinned
};

class MemoryPlan {
 public:
  BufferId add(BufferSpec spec);   // fatal on a duplicate name
  BufferId find(const std::string& name) const;   // invalid id if absent
  // Lays out every buffer: Persistent, Microbatch, one Activation region per layer, then the Transient and Workspace
  // slots (each slot sized by the largest member). Offsets are 256-byte aligned; with canaries, a 4 KiB guard band
  // follows every buffer. Fails (fatal) above `budget_bytes`.
  void finalize(uint64_t budget_bytes, bool canaries);
  uint64_t device_bytes() const;
  uint64_t pinned_bytes() const;
  uint64_t offset(BufferId id) const;
  const BufferSpec& spec(BufferId id) const;
  // Prints one line per buffer (name, life, dtype, shape, bytes, offset) and subtotals per lifetime, then
  // "TOTAL <bytes> of budget <bytes>". The training binary prints it at start; `marincpp plan` prints it alone.
  void print(FILE* f) const;
  void print_grouped(FILE* f) const;   // act.L*.x summed over layers, as in DESIGN.md §2.5

 private:
  std::vector<BufferSpec> specs_;
  std::vector<uint64_t> offsets_;
  uint64_t device_bytes_ = 0, pinned_bytes_ = 0;
};

// Builds the full plan for one rank. Ranks differ only in owner-shard sizes (whole MuonH matrices and row shards are
// assigned by LPT at plan time). Every module declares its buffers through its own declare_*(); the plan of a run with
// several step regimes is the maximum over regimes (DESIGN.md §2.4).
MemoryPlan build_plan(const Config& c, const ModelShape& m, const ExecShape& e, const DatasetMeta& d);

// Chooses (micro_seqs, remat, zero1, zero2_vocab, ce_chunk_rows) when "auto": the largest micro_seqs dividing
// seqs_per_rank that fits with remat=all; remat=min if that gives at least twice the tokens of a microbatch below
// 16,384 tokens; remat=layer only when requested or when nothing else fits. Writes rejected candidates to `report`.
PlanChoice choose_plan(const Config& c, const ModelShape& m, const DatasetMeta& d, std::string* report);

// Debug modes (exec.debug_arena, local): Canary checks the guard bands after every step (fatal with the buffer name);
// Poison also fills Transient, Workspace and recompute slots with NaN before each use.
enum class ArenaDebug : uint8_t { Off = 0, Canary = 1, Poison = 2 };

// The arena of one GPU. All device pointers in the step graphs point into it, so graph replay is valid forever.
class Arena {
 public:
  void allocate(const MemoryPlan& plan, int device, ArenaDebug dbg);  // one cudaMalloc + one cudaHostAlloc
  void release();
  // Bind by declared name with dtype and shape checks; a mismatch is fatal at bind time. No positional lookup exists.
  void* bind(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const;
  template <class T> T* bind_as(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const {
    return static_cast<T*>(bind(name, dtype, shape));
  }
  TensorView view(const std::string& name) const;
  void check_canaries(cudaStream_t s) const;     // enqueues the guard-band check (debug modes only)
  void poison_transients(cudaStream_t s) const;  // Poison mode: NaN fill, recorded before each layer's use
  // After warm-up the trainer asserts cudaMemGetInfo is unchanged at every eval and checkpoint (no hidden allocation).
  uint64_t free_bytes_at_start() const { return free_at_start_; }

 private:
  const MemoryPlan* plan_ = nullptr;
  char* dev_ = nullptr;
  char* host_ = nullptr;
  ArenaDebug dbg_ = ArenaDebug::Off;
  uint64_t free_at_start_ = 0;
};

}  // namespace nm
