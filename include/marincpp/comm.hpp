// Data-parallel communication inside the captured step graph: gradient buckets reduced to their ZeRO-1 owners and
// overlapped with the backward, per-bucket sum-of-squares partials, bf16 all-gather of updated parameters at the start
// of the next step, the metrics reduction, and (later stage) row-sharded tables. DESIGN.md §6.
#pragma once

#include <cstdint>
#include <vector>

#include "marincpp/core.hpp"
#include "marincpp/memory.hpp"

namespace nm {

struct DeviceContext;
struct Ownership;
struct MultiTensor;

// A bucket is a contiguous byte range of the f32 gradient buffer. The buffer is laid out in reverse backward order
// (heads and readouts, final norm, layer L-1, ..., layer 0, input layer), so bucket i becomes ready before bucket i+1
// during the backward of the LAST microbatch. Target size 64 MiB, never splitting a parameter. The input-layer
// (embedding) bucket is last; only the clip groups that contain it wait for it.
struct Bucket {
  int id;
  uint64_t offset_elems;      // into grad.f32
  uint64_t numel;
  int first_param, last_param;  // indices into Model::params()
  int forward_layer;          // layer whose forward first reads these parameters (all-gather wait point), -1 = input
};

class GradReducer {
 public:
  // Plans buckets over the model's parameter order and the ownership map; declares a bf16 staging buffer when the
  // reduction dtype is bf16, and the per-slice sum-of-squares partial buffer.
  void plan(const std::vector<uint64_t>& param_numel, const Ownership& own, DType reduce_dtype, int nranks,
            MemoryPlan& mp);
  void bind(const Arena& arena, float* grad_base, const MultiTensor& owned_slices);
  // Called by the model while recording the last microbatch, right after the kernel that writes the last gradient of
  // the bucket containing `param`: records an event on compute, makes `comm` wait on it, and enqueues
  //   ncclGroupStart; ncclReduce(sum, root = owner) per whole-owned tensor slice; ncclReduceScatter per row-sharded
  //   tensor; ncclGroupEnd    (bf16 reduction: k_f32_to_bf16 before, k_bf16_to_f32 after; levanter strict preset)
  //   then k_sumsq_partials over this rank's owned slices of the bucket (clip and per-leaf norms; DESIGN.md §4.1)
  //   and records bucket_event(id).
  // With optim.zero1=off it all-reduces in place instead. The per-token scale already divides by B_global * (S-1).
  void param_ready(DeviceContext& d, int param);
  cudaEvent_t bucket_event(int bucket) const;   // optimizer clip groups wait on their buckets' events
  // ZeRO-2 for vocab-sized tensors (optim.zero2_vocab): reduce-scatter those gradients after every microbatch.
  void microbatch_done(DeviceContext& d, int microbatch);
  const std::vector<Bucket>& buckets() const { return buckets_; }

 private:
  std::vector<Bucket> buckets_;
  std::vector<int> bucket_of_param_;
  std::vector<int> remaining_;
  std::vector<cudaEvent_t> events_;
  DType dtype_ = DType::f32;
  float* grad_ = nullptr;
  bf16_t* staging_ = nullptr;
};

// bf16 all-gather of the parameters updated by their owners in step k-1, recorded at the START of g_step(k) on `comm`,
// per bucket in forward order (grouped ncclBroadcast from owners, ncclAllGather for row shards); layer l's forward
// (and readout l's head) waits on layer_event(l). Wire bytes per parameter with the reduction: 6 (N-1)/N vs 8 (N-1)/N.
class ParamGather {
 public:
  void plan(const std::vector<Bucket>& buckets, const Ownership& own, int nranks);
  void bind(const Arena& arena, bf16_t* p16_base);
  void record(DeviceContext& d);
  cudaEvent_t layer_event(int forward_layer) const;
 private:
  std::vector<cudaEvent_t> events_;
};

// Metrics reduction (DESIGN.md §6.3): one f32 vector per rank summed across ranks inside g_step before the StepRecord:
// [loss_sum, weight_sum, readout_loss_sum[n_readouts], readout_weight_sum[n_readouts], nonfinite_loss,
//  nonfinite_grad, docmeta_overflow, per-leaf norm partials (stats regime only)]. Fixed order (Ring/Simple).
void record_metrics_allreduce(DeviceContext& d, float* vec, int n);
void record_allreduce_scalars(DeviceContext& d, float* vals, int n);   // clip-group partials, AdamH shard norms

// Row-sharded table (later stage; hashed n-gram input tables, 12.8M-class OD head): rows [r*R/N, (r+1)*R/N) live on
// rank r with their master, moments and gradient. Lookup: all-gather of the microbatch ids, local gather of owned rows,
// reduce-scatter of the gathered rows to their requesters; backward: sparse gradients on touched rows, lazy Adam.
struct ShardedTable {
  int64_t rows = 0, cols = 0;
  int64_t row_begin = 0, row_end = 0;   // this rank's rows
};
void sharded_lookup(DeviceContext& d, const ShardedTable& t, const int32_t* ids, int64_t n, bf16_t* out);
void sharded_lookup_bwd(DeviceContext& d, const ShardedTable& t, const int32_t* ids, int64_t n, const bf16_t* dout,
                        float* grad_rows_owned, uint8_t* touched_owned);

}  // namespace nm
