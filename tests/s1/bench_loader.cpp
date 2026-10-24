// Sustained throughput of the C++ reader on the converted fineweb-edu-10B set (DESIGN.md §5.4 target: the trainer's
// tokens/s, 2,793,000 at 130m on 4 H100 per §12.2). One consumer drains the loader as fast as it can: acquire(k) ->
// copy_rank_rows to the GPU (when one exists) for every rank -> release; steps are fresh (first_step chosen by the
// caller), so reads come from GPFS, not from this node's cache.
//   nm_bench_s1 [--first K] [--steps N] [--seconds T] [--threads R] [--depth D] [--batch B] [--ranks n] [--mmap]
//               [--host-only] [--pace TOKENS_PER_S]
// --pace: the consumer takes one step every B*4096/pace seconds (a trainer at that speed); the result then reports the
// time acquire() blocked after the first step (0 = the reader keeps up with no stall).
// Prints one JSON line.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "data/internal.hpp"
#include "marincpp/data.hpp"

using namespace nm;

int main(int argc, char** argv) {
  int64_t first = 30000, steps = 2000;
  double seconds = 60;
  int threads = 16, depth = 256, B = 128, ranks = 4;
  bool mmap_mode = false, host_only = false;
  double pace = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--first") first = std::atoll(next());
    else if (a == "--steps") steps = std::atoll(next());
    else if (a == "--seconds") seconds = std::atof(next());
    else if (a == "--threads") threads = std::atoi(next());
    else if (a == "--depth") depth = std::atoi(next());
    else if (a == "--batch") B = std::atoi(next());
    else if (a == "--ranks") ranks = std::atoi(next());
    else if (a == "--mmap") mmap_mode = true;
    else if (a == "--host-only") host_only = true;
    else if (a == "--pace") pace = std::atof(next());
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  try {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
      cudaGetLastError();
      host_only = true;
    }
    const DatasetMeta m = load_dataset_meta("fineweb-edu-10B-marin128k");
    TokenFile f;
    f.open(m);
    DataPlan p;
    Phase ph;
    ph.step_begin = 0;
    ph.step_end = first + steps;
    ph.dataset = m.name;
    ph.global_batch = B;
    ph.seq_len = 4096;
    p.phases.push_back(ph);
    data::validate_plan(p, ranks);
    BatchLoader l;
    data::LoaderOptions o;
    o.host_only = host_only;
    o.read = mmap_mode ? data::ReadMode::Mmap : data::ReadMode::Pread;
    o.hash_every = 0;
    data::set_loader_options(l, o);
    // the vis node has one GPU: every rank's rows go to device 0 through rank-r events created on device 0 when
    // ranks > device count is impossible, so the bench uses as many ranks as devices for the copies
    const int copy_ranks = host_only ? ranks : std::min(ranks, n_dev);
    const int loader_ranks = host_only ? ranks : copy_ranks;
    const auto t0 = std::chrono::steady_clock::now();
    l.start(p, {&f}, loader_ranks, depth, threads, first);
    std::vector<int32_t*> dbuf(copy_ranks, nullptr);
    std::vector<cudaStream_t> streams(copy_ranks, nullptr);
    if (!host_only)
      for (int r = 0; r < copy_ranks; ++r) {
        NM_CUDA(cudaSetDevice(r));
        NM_CUDA(cudaMalloc(&dbuf[r], size_t(B) * 4096 * 4));
        NM_CUDA(cudaStreamCreate(&streams[r]));
      }
    std::vector<double> step_s;
    double stall_s = 0;   // acquire() blocking after the first step
    int64_t stalls = 0;
    int64_t done = 0;
    double first_batch_s = 0;
    auto prev = std::chrono::steady_clock::now();
    for (int64_t k = first; k < first + steps; ++k) {
      if (pace > 0 && k > first) {   // trainer at `pace` tokens/s: the next step is due one step time after the last
        const auto due = prev + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                    std::chrono::duration<double>(double(B) * 4096 / pace));
        while (std::chrono::steady_clock::now() < due) std::this_thread::sleep_until(due);
      }
      const auto a0 = std::chrono::steady_clock::now();
      TokenSlot& s = l.acquire(k);
      if (k > first) {
        const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
        if (w > 1e-3) ++stalls;
        stall_s += w;
      }
      if (!host_only)
        for (int r = 0; r < copy_ranks; ++r) {
          NM_CUDA(cudaSetDevice(r));
          data::copy_rank_rows(l, s, r, dbuf[r], streams[r]);
        }
      for (int r = 0; r < loader_ranks; ++r) l.release(s, r);
      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - prev).count();
      if (k == first) first_batch_s = std::chrono::duration<double>(now - t0).count();
      else step_s.push_back(dt);
      prev = now;
      ++done;
      if (std::chrono::duration<double>(now - t0).count() > seconds) break;
    }
    if (!host_only)
      for (int r = 0; r < copy_ranks; ++r) {
        NM_CUDA(cudaSetDevice(r));
        NM_CUDA(cudaStreamSynchronize(streams[r]));
      }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto st = data::loader_stats(l);
    l.stop();
    for (int r = 0; r < copy_ranks && !host_only; ++r) {
      cudaFree(dbuf[r]);
      cudaStreamDestroy(streams[r]);
    }
    std::vector<double> sorted = step_s;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted[size_t(q * (sorted.size() - 1))]; };
    double steady = 0;
    for (double x : step_s) steady += x;
    const double tok_step = double(B) * 4096;
    std::printf("{\"first_step\": %lld, \"steps\": %lld, \"B\": %d, \"threads\": %d, \"depth\": %d, \"read\": \"%s\", "
                "\"gpu_copies\": %s, \"ranks\": %d, \"seconds\": %.3f, \"first_batch_s\": %.3f, "
                "\"tokens_per_s_total\": %.0f, \"tokens_per_s_after_first\": %.0f, \"step_ms_p50\": %.3f, "
                "\"step_ms_p90\": %.3f, \"step_ms_max\": %.3f, \"acquire_wait_s\": %.3f, \"windows_read\": %lld, "
                "\"read_s_sum\": %.3f, \"mean_read_ms\": %.3f, \"pace\": %.0f, \"stall_s_after_first\": %.4f, "
                "\"stalls_over_1ms\": %lld}\n",
                (long long)first, (long long)done, B, threads, depth, mmap_mode ? "mmap" : "pread",
                host_only ? "false" : "true", loader_ranks, total, first_batch_s, done * tok_step / total,
                step_s.empty() ? 0.0 : step_s.size() * tok_step / steady, 1e3 * pct(0.5), 1e3 * pct(0.9),
                1e3 * pct(1.0), st.wait_seconds, (long long)st.windows_read, st.read_seconds,
                st.windows_read ? 1e3 * st.read_seconds / st.windows_read : 0.0, pace, stall_s, (long long)stalls);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nm_bench_s1: %s\n", e.what());
    return 1;
  }
}
