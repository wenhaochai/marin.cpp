// Process and device model: one process, one host thread per GPU, NCCL from ncclCommInitAll. DESIGN.md §1.
#pragma once

#include <atomic>
#include <barrier>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "marincpp/core.hpp"
#include "marincpp/memory.hpp"

namespace nm {

class GemmCache;  // kernels/gemm.hpp

// Per-GPU state, owned by exactly one worker thread (rank == CUDA device ordinal within CUDA_VISIBLE_DEVICES).
struct DeviceContext {
  int rank = 0;
  int nranks = 1;
  int device = 0;
  cudaStream_t compute = nullptr;   // forward, backward, optimizer
  cudaStream_t comm = nullptr;      // NCCL, per-bucket sum-of-squares, all-gathers; forked from compute in the capture
  cudaStream_t side = nullptr;      // optimizer clip groups that run beside MuonH (forked in the capture)
  cudaStream_t copy = nullptr;      // D2H of metrics rings and checkpoint slices, H2D of token slots (outside graphs)
  ncclComm_t nccl = nullptr;
  cublasLtHandle_t lt = nullptr;
  GemmCache* gemm = nullptr;        // chosen kernel per GEMM shape (persisted in the run dir; DESIGN.md §11)
  Arena arena;
  const MemoryPlan* plan = nullptr;
  int sm_count = 0;
  cudaEvent_t ckpt_copy_done = nullptr;  // external event waited on by the optimizer section of g_step
  // One instantiated g_step per step regime (model.hpp StepRegime); the host picks by k. Eval graphs per batch group.
  std::vector<cudaGraphExec_t> g_step;
  std::vector<cudaGraphExec_t> g_eval;
};

// Stop decision shared by all workers. The signal thread only sets `requested`; the decision for step k is taken
// once, in the completion function of the per-step barrier, so every rank launches exactly the same steps.
struct StopControl {
  std::atomic<int> requested{0};       // signal number (SIGTERM=15, SIGINT=2) or 0
  std::atomic<int64_t> deadline_unix{0};  // optional voluntary stop (exit 99) before a known wall
  int64_t decided_stop_step = -1;      // written by the barrier completion; read after the barrier
};

// The process: parses nothing, owns the threads. main() builds it after config resolution and planning.
class Process {
 public:
  // Blocks SIGTERM/SIGINT/SIGUSR1 in all threads, starts the signal thread (sigwait), creates NCCL communicators with
  // ncclCommInitAll for the visible devices, then starts one worker per device running `body(ctx)`.
  // Returns the exit code chosen by the workers (all ranks agree by construction).
  ExitCode run(int nranks, const std::function<ExitCode(DeviceContext&, Process&)>& body);

  // Per-step rendezvous of the worker threads (~1 us). Its completion function samples StopControl.
  void step_barrier(int64_t step);
  StopControl& stop() { return stop_; }
  // Host barrier for rare collective host work (checkpoint commit, eval key emission).
  void host_barrier();

 private:
  StopControl stop_;
  std::vector<std::thread> workers_;
  std::thread signal_thread_;
};

// Node health probe (reliability rule R12): bf16 GEMM 8192^3 x 20 and an all-reduce of 1 GiB on every rank.
// Returns the slowest rank's fraction of the expected rate (expected: 650 TFLOP/s GEMM, 330 GB/s bus bandwidth).
double health_probe(DeviceContext& ctx);

}  // namespace nm
