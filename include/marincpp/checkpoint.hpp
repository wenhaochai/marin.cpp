// Checkpoints: raw binary + JSON manifest, owner slices, async device -> pinned -> file, atomic rename, save on SIGTERM;
// loading into a different model (init.from) and importing levanter checkpoints. DESIGN.md §7.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

struct DeviceContext;
struct Config;

// <run>/ckpt/step-<k:08d>/
//   manifest.json   {format: 2, run_id, config_hash, config: <canonical JSON>, code_sha, step_next: k,
//                    data: {plan_sha, cursor_step: k}, rng: {seed, counter: k}, ranks: R, tensors: [{name, dtype,
//                    shape, row_begin, row_end, rank_file, offset, nbytes, xxh3_64}], bytes_total, wall_seconds_write,
//                    gemm_algos: "gemm_algos.bin"}
//   rank<r>.bin     the slices this rank OWNS (ZeRO-1: whole MuonH matrices, row shards of vocab-sized tensors, small
//                   tensors), back to back, 4 KiB aligned
// Content: f32 masters, MuonH momentum, Adam/AdamH mu and nu, EMA weights if any. Not saved: bf16 copies (recast at
// load), gradients (zero at a step start), data loader state (a function of k). Tensors are read by name and row range,
// so a checkpoint loads on any rank count.
struct TensorEntry {
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
  int64_t row_begin = 0, row_end = 0;
  int rank_file;
  uint64_t offset, nbytes;
  uint64_t xxh3;
};
struct Manifest {
  int format = 2;
  std::string run_id, config_hash, config_json, code_sha;
  int64_t step_next = 0;          // number of completed updates: the next update index
  std::string data_plan_sha;
  int ranks = 1;
  std::vector<TensorEntry> tensors;
  uint64_t bytes_total = 0;
  double write_seconds = 0;       // measured; used to predict whether a TERM-time save fits (DESIGN.md §1.5)
};
Manifest read_manifest(const std::string& ckpt_dir);
// Verifies a manifest against its OWN stored config (hash of config_json == config_hash), so forks and init.from work
// with parents written by older binaries.
bool manifest_self_consistent(const Manifest& m);

// Write protocol (atomic): write into step-<k>.tmp/ -> fsync every rank file -> host barrier -> rank 0 writes
// manifest.json, fsyncs it and the directory -> rename(step-<k>.tmp, step-<k>) -> fsync ckpt/ -> atomic replace of
// the `latest` symlink. On start: delete every *.tmp; resume from `latest`. Retention: the newest run.keep_ckpts, every
// multiple of run.keep_every, and every pass boundary when run.keep_pass_boundaries.
class CheckpointWriter {
 public:
  void init(const std::string& run_dir, int rank, int nranks, uint64_t slice_bytes_max);  // pinned staging
  // Async path (periodic): after g_step(k-1) is enqueued, copies this rank's owned slices D2H on the copy stream (after
  // an event on compute) and records ckpt_copy_done; g_step(k)'s optimizer section waits on that external event. On
  // steps without a checkpoint the host records ckpt_copy_done at once. The writer thread then writes the file.
  void begin_async(DeviceContext& d, int64_t step_next);
  void record_no_copy(DeviceContext& d);
  // Sync path (SIGTERM, final): the same, but the caller waits for durability.
  void save_blocking(DeviceContext& d, int64_t step_next);
  bool busy() const { return busy_.load(); }
  double last_write_gbps() const { return last_gbps_; }
 private:
  std::atomic<bool> busy_{false};
  double last_gbps_ = 0;
  std::thread writer_;
};

// Resume: config_hash must equal the hash of the resolved config (else exit 1: a different run must use a new run
// dir, a continuation with new phases must be a fork), checks xxh3 of every slice it reads, loads the slices this rank
// owns, recasts bf16 copies (owner rows, then the first g_step's all-gather), sets the device step counter.
// A code_sha different from the binary's is allowed only with --allow-code-change (logged as an event).
struct ResumeInfo { bool found = false; int64_t step_next = 0; std::string path; };
ResumeInfo find_latest(const std::string& run_dir);
void load_checkpoint(DeviceContext& d, const std::string& ckpt_dir, const std::string& expected_hash);

// init.from (DESIGN.md §7.6): load tensors of another run into a different model. Rules of init.map are applied in
// order: target regex -> source tensor name (with optional row range / reshape) or "init" (fresh draw). init.freeze puts
// matching tensors in group Frozen; init.opt_state copies optimizer state of mapped tensors or starts fresh. The source
// manifest must be self-consistent. Unmapped target tensors are an error unless a rule maps them to "init".
struct InitMapRule { std::string target_regex; std::string source; };   // source "" = init
void load_init_from(DeviceContext& d, const Config& c, const std::string& source_ckpt,
                    const std::vector<InitMapRule>& rules, bool copy_opt_state);

// Levanter import (tools/levref/import_levanter.py, Python): levanter checkpoint (model, opt_state, step counts) ->
// manifest + rank0.bin. Row map, per layer: Wqkv rows = q rows in (kv, g, d) order from q_proj [L, KV, G, hd, D], then
// k_proj, then v_proj; Wgu = gate and up rows interleaved in 64-row blocks; o_proj [L, D, H, hd] -> [D, H*hd]. Round
// trip: import, load, nm_model_export_params (capi.h), compare every levanter leaf bitwise. T3.2 uses it.

}  // namespace nm
