// Stage S7: the multi-GPU step loop. One process, one host thread per GPU, NCCL communicators from ncclCommInitAll
// (DESIGN.md §1). Each rank runs its rows of the global batch as microbatches, all-reduces the f32 gradients and the
// per-microbatch loss sums, and applies the same optimizer step (ZeRO-1 follows). With `graph` the whole step of a
// rank (zero grads, every microbatch, the all-reduces, the update) is one captured CUDA graph; the host only copies
// the next step's tokens and launches it (DESIGN.md §2.4).
#pragma once

#include <cstdint>
#include <string>

#include "marincpp/config.hpp"
#include "marincpp/kernels/attention.hpp"
#include "marincpp/kernels/cross_entropy.hpp"

namespace nm::train {

struct TrainOptions {
  std::string size = "130m";
  int gpus = 4;
  int64_t steps = 20;              // updates to run (from step 0)
  bool graph = true;
  int micro_rows = 32;             // sequences per microbatch per rank
  int ce_chunk = 32768;
  AttnBackendKind attn = AttnBackendKind::FA3;
  CeImpl ce = CeImpl::B1;
  EmbedAccum embed = EmbedAccum::F32;
  Rounding rounding = Rounding::Levanter;
  int log_every = 10;
  std::string data_dir;            // converted dataset (tokens.u32, meta.json)
  std::string init_fixture;        // levanter p0 directory (init.mode=fixture)
  std::string gemm_cache;          // file the ranks share their GEMM choices through (rank 0 searches)
  std::string metrics;             // JSONL output: one line per logged step ("" = stdout only)
};

// Runs the loop; returns 0 on success. Prints one line per log interval: step, loss, s/step, tokens/s.
int train(const TrainOptions& o);

}  // namespace nm::train
