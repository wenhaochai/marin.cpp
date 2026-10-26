// Stage S7: the S6 optimizer engine bound to the baseline model's buffers on one rank (no ZeRO-1 yet: every tensor is
// whole on this rank). DESIGN.md §4; optim.md §3-§6. The update index k lives on the device; record_step() is
// graph-capturable after prepare().
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/memory.hpp"
#include "model/baseline.hpp"
#include "optim/engine.hpp"

namespace nm::train {

class Optimizer {
 public:
  Optimizer(const Config& c, const model::Baseline& net);
  ~Optimizer();
  // Buffers: opt.s0 / opt.s1 (momentum, mu / nu; full tensors), sched.* tables [n_steps], opt.step, opt.ws.
  void declare(MemoryPlan& plan) const;
  // Binds the engine to the arena and the model's p32 / p16 / grad, uploads the schedule tables, sets k, zeroes the
  // states, and runs the engine's prepare (NS GEMM algorithms). Synchronous.
  void bind(const Arena& arena, const model::Baseline& net, cudaStream_t s, int32_t k0 = 0);
  void record_step(cudaStream_t s);     // clip, update, p16 = bf16(p32), k++
  int32_t* step() const { return step_; }
  opt::Engine& engine() { return *engine_; }
  int64_t n_steps() const { return n_steps_; }

 private:
  std::vector<opt::EngGroup> groups_;
  std::vector<opt::EngTensor> tensors_;
  std::unique_ptr<opt::Engine> engine_;
  ScheduleTables tables_;
  int64_t n_steps_ = 0, total_ = 0;
  float *s0_ = nullptr, *s1_ = nullptr, *lr_ = nullptr, *alr_ = nullptr, *bc1_ = nullptr, *bc2_ = nullptr;
  int32_t* step_ = nullptr;
  void* dummy_ = nullptr;
};

}  // namespace nm::train
