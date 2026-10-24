// Device-side metric accumulation, rare host copies, local JSONL as the record. DESIGN.md §8.2.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

struct DeviceContext;

// One record per update, written by kernels inside g_step AFTER the metrics all-reduce (comm.hpp
// record_metrics_allreduce), so every value is global (all ranks), never rank 0's rows only. The record is a header
// plus a float payload whose layout is fixed at plan time from the number of clip groups and readouts.
struct StepRecordHeader {
  int32_t step;                 // update index k (the W&B _step of train/loss)
  uint32_t nonfinite;           // bit 0 loss, bit 1 any group norm, bit 2 doc-meta overflow (any rank)
  float loss_sum, weight_sum;   // global; train/loss = loss_sum / weight_sum (the batch's token mean)
  float lr, adam_lr;            // optim/learning_rate, optim/adam_lr: the values update k used
  float grad_norm_total;        // every step; logged as grad/norm/total_every_step
};
struct StepRecordLayout {
  int n_clip = 0;               // payload: group_norm[n_clip], clip[n_clip]
  int n_readouts = 0;           //          readout_loss[n_readouts], readout_weight[n_readouts]
  size_t bytes() const;         // header + payload, 16-byte aligned
};

// Ring of StepRecords plus a stats ring written only by the stats regime (k % 10 == 0 && k > 1): params/norm/<leaf> of
// the f32 parameters BEFORE update k and grad/norm/<leaf> of the pre-clip mean gradient of batch k, per logical leaf
// (13(L+1)+4 keys per prefix incl. transformer.layers.stacked.<X> and total), and the hook statistics.
struct MetricsRing {
  void* dev = nullptr;          // [capacity] records in the arena
  StepRecordLayout layout;
  float* stats_dev = nullptr;   // [stat_capacity, n_stats]
  int capacity = 1024;
  int n_stats = 0;
  int stat_capacity = 128;
};

// Host side: every run.metrics_flush steps the rank-0 worker enqueues cudaMemcpyAsync(ring slice -> pinned) on the
// copy stream with an event; the writer thread waits on that event (not the worker), formats JSON lines and appends
// them to <run>/metrics.jsonl (fsync at every checkpoint). A non-finite flag ends the run with exit 2 after at most
// metrics_flush extra steps.
// Logging contract (eval.md §6-§7): train/loss, global_step, run_progress, optim/*, throughput/* at steps 2..N-1;
// throughput/hook_time and loading_time at 0..N-1; grad/norm/* and params/norm/* at 10, 20, ..., floor((N-1)/10)*10;
// eval/* at {1000k, N-1}; eval_init/* for the init eval; NOTHING at a step >= N. Before an eval row at step s the writer
// forces a flush of every train row <= s (W&B drops rows below the last logged step).
class MetricsWriter {
 public:
  void open(const std::string& run_dir, int64_t resume_step, int64_t n_steps);  // truncates lines with step >= resume
  void enqueue_copy(DeviceContext& d, const MetricsRing& r, int64_t first, int64_t count);
  void flush_through(DeviceContext& d, const MetricsRing& r, int64_t step);   // forced before eval rows
  void write_row(int64_t step, const std::vector<std::pair<std::string, double>>& kv);  // refuses step >= N
  void event(const std::string& kind, const std::string& json);  // events.jsonl: start, resume, ckpt, stop, done
  void flush_and_sync();
  void close();
 private:
  FILE* metrics_ = nullptr;
  FILE* events_ = nullptr;
  int64_t n_steps_ = 0;
  std::thread writer_;
};

}  // namespace nm
