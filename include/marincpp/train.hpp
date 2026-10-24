// The trainer: run directory, step graphs per regime, the host loop, stop handling. DESIGN.md §1, §2.4, §7.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/core.hpp"
#include "marincpp/model.hpp"

namespace nm {

struct DeviceContext;
class Process;

// <run_root>/<label>-<run_id>/
//   config.resolved.json  run.json (id, label, parent, code_sha, created, host history)  plan.txt
//   bin/                  snapshot of the marincpp binary and its shared libraries (resumes run this copy)
//   ckpt/                 checkpoints (checkpoint.hpp)
//   metrics.jsonl events.jsonl eval/  gemm_algos.bin  DONE (written last, after the final eval is durable)
//   lock                  flock'd by the running process: a second process on the same run exits 1
struct RunDir {
  std::string path;
  std::string id;
  bool done = false;              // DONE exists: exit 0 before any GPU initialisation
};
RunDir open_run_dir(const Config& c);

// Structure of one update k (DESIGN.md §2.4):
//   [outside the graph] barrier(k); cudaMemcpyAsync(tokens of step k: pinned slot -> device); event;
//                       checkpoint step: copy-stream D2H of owner slices after g_step(k-1), record ckpt_copy_done;
//                       otherwise record ckpt_copy_done at once
//   g_step[regime(k)]:  comm: bf16 all-gather of step k-1's owner updates, per layer bucket in forward order
//                       compute: per microbatch: doc_meta, input layer, layers (each waits on its all-gather event),
//                       final norm, output head, readouts interleaved with the backward, input-layer backward; during
//                       the last microbatch buckets are reduced to their owners on comm, then sum-of-squares partials
//                       optimizer: [wait ckpt_copy_done] each clip group after its own buckets (AdamH, MuonH, Adam),
//                       metrics all-reduce, StepRecord, k++
// The host never waits inside a step. Host work per step: one barrier, one memcpy, one graph launch, events.
class Trainer {
 public:
  explicit Trainer(const Config& c);
  // main(): resolves config, opens the run dir, plans memory (prints the plan), starts Process with run_rank().
  ExitCode run();
 private:
  ExitCode run_rank(DeviceContext& d, Process& p);
  // Captures one g_step per regime with cudaStreamCaptureModeThreadLocal and instantiates it; runs the capture lint
  // (no step-dependent host scalar recorded as a kernel argument; every such value is a device table or regime flag).
  void build_graphs(DeviceContext& d, const std::vector<StepRegime>& regimes);
  int regime_of(int64_t k) const;
  // Deterministic mode: update k eagerly on a staged copy of the state and as a graph; bitwise-equal checksums.
  void warmup_and_check(DeviceContext& d);
  Config cfg_;
  RunDir run_;
  std::vector<StepRegime> regimes_;
};

// CLI (src/main.cpp):
//   marincpp train  --config <toml> [--preset levanter_strict] [--set k=v ...] [--allow-code-change]
//   marincpp plan   --config <toml> [--set ...] [--create]  CPU only: memory plan + step budget, exit 1 if no fit
//   marincpp knobs                                        the knob table (knobs.def) and the registry names
//   marincpp order  --run <dir> --step k [--row r]        data queries (DataPlan::seq_of / times_seen)
//   marincpp eval   --run <dir> --ckpt step-<k>           eval of a checkpoint with the run's config
//   marincpp profile --config <toml> --steps 20 [--profile-ops]   eager per-op timings vs the budget table

}  // namespace nm
