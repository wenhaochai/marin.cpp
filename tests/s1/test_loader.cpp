// S1 loader: prefetch ring, pinned slots, H2D handoff, restart at any step (resume / fork), error propagation, stop()
// while workers are blocked, the checks of start(). Synthetic converted set (tests write it), so these run anywhere;
// the GPU cases skip without a device.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "s1_util.hpp"

using namespace nm;

namespace {

struct Synth {
  s1::Synthetic d;
  DatasetMeta m;
  TokenFile f;
  explicit Synth(const char* name = "synth-loader", int doc_len = 300) {
    // 997 windows of 512 tokens (+11 tail tokens); LCG constants (a, b) = (101, 7), gcd(101, 997) = 1
    d = s1::make_synthetic(name, 997ull * 512 + 11, 512, doc_len, 42, 101, 7);
    m = load_dataset_meta(d.dir);
    data::register_lcg(42, (997ull * 512 + 11) / 256, 101, 7);   // phase 2 reads 256-token windows: its own N
    f.open(m);
  }
  ~Synth() { s1::remove_dir(d.dir); }
};

DataPlan two_phase_plan(const std::string& ds) {   // B 8 then B 16 at a new length: two step regimes
  DataPlan p;
  p.phases.push_back(s1::levanter_phase(ds, 0, 60, 8, 512));
  Phase r = s1::levanter_phase(ds, 60, 140, 16, 256);
  r.kind = OrderKind::Repeat;
  r.subset = 300;
  p.phases.push_back(r);
  return p;
}

data::LoaderOptions host_opts(data::ReadMode mode = data::ReadMode::Pread) {
  data::LoaderOptions o;
  o.host_only = true;
  o.read = mode;
  return o;
}

bool have_gpu() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    cudaGetLastError();
    return false;
  }
  return true;
}

// Consumes steps [first, end) from a started loader, compares each slot with the direct assembly; returns mismatches.
int drain(BatchLoader& l, const DataPlan& p, TokenFile& f, int64_t first, int64_t end, int nranks) {
  int bad = 0;
  for (int64_t k = first; k < end; ++k) {
    TokenSlot& s = l.acquire(k);
    const auto want = data::assemble_batch(p, {&f}, k);
    bad += s.step != k || std::memcmp(s.host, want.data(), want.size() * 4) != 0;
    for (int r = 0; r < nranks; ++r) l.release(s, r);
  }
  return bad;
}

}  // namespace

NM_TEST(s1_loader, host_ring_all_steps_both_read_modes) {
  Synth s;
  const DataPlan p = two_phase_plan(s.d.dir);
  data::validate_plan(p, 4);
  for (auto mode : {data::ReadMode::Pread, data::ReadMode::Mmap}) {
    for (int depth : {1, 4, 64}) {
      BatchLoader l;
      s1::LoaderGuard guard{l};
      data::LoaderOptions o = host_opts(mode);
      o.hash_every = 50;
      data::set_loader_options(l, o);
      l.start(p, {&s.f}, 2, depth, 8, 0);
      CHECK_EQ(drain(l, p, s.f, 0, 140, 2), 0);
      const auto st = data::loader_stats(l);
      l.stop();
      CHECK_EQ(st.steps_staged, int64_t(140));
      CHECK_EQ(st.windows_read, int64_t(60 * 8 + 80 * 16));
      CHECK_EQ(st.hashes.size(), size_t(3));   // k = 0, 50, 100
      CHECK_EQ(st.hashes[1].second, data::batch_hash(p, {&s.f}, 50));
    }
  }
}

// Reshuffled passes through the loader (the first reader of a step computes its rows outside the lock; the pass
// permutation is shared by all readers): equal to the direct assembly across 4 pass boundaries, at 16 readers.
NM_TEST(s1_loader, reshuffled_passes) {
  Synth s;
  DataPlan p;
  Phase r = s1::levanter_phase(s.d.dir, 0, 150, 8, 512);
  r.kind = OrderKind::Repeat;
  r.subset = 250;   // passes start at m = 250, 500, 750, 1000 (inside steps 31, 62, 93, 125)
  r.reshuffle = true;
  p.phases.push_back(r);
  data::validate_plan(p, 2);
  for (auto mode : {data::ReadMode::Pread, data::ReadMode::Mmap}) {
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::set_loader_options(l, host_opts(mode));
    l.start(p, {&s.f}, 2, 32, 16, 0);
    CHECK_EQ(drain(l, p, s.f, 0, 150, 2), 0);
    l.stop();
  }
}

// Resume / fork: the loader started at any k yields the plan's batch k (no loader state); a forked plan's batches
// before the fork are the parent's.
NM_TEST(s1_loader, restart_and_fork) {
  Synth s;
  const DataPlan p = two_phase_plan(s.d.dir);
  for (int64_t first : {1, 59, 60, 61, 139}) {
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::set_loader_options(l, host_opts());
    l.start(p, {&s.f}, 1, 8, 4, first);
    CHECK_EQ(drain(l, p, s.f, first, std::min<int64_t>(first + 10, 140), 1), 0);
    l.stop();   // stop with steps still staged / in flight
  }
  Phase tail = s1::levanter_phase(s.d.dir, 30, 90, 8, 512);
  tail.seed = 42;
  tail.index_base = 5000;
  const DataPlan fk = data::fork_plan(p, 30, {tail});
  data::validate_plan(fk, 1);
  for (int64_t k = 0; k < 30; ++k) CHECK(data::assemble_batch(fk, {&s.f}, k) == data::assemble_batch(p, {&s.f}, k));
  BatchLoader l;
  s1::LoaderGuard guard{l};
  data::set_loader_options(l, host_opts());
  l.start(fk, {&s.f}, 1, 16, 6, 25);
  CHECK_EQ(drain(l, fk, s.f, 25, 90, 1), 0);
  l.stop();
}

// Consumers that hold slots: the stager never overwrites a slot before every rank released it.
NM_TEST(s1_loader, slow_consumers_hold_slots) {
  Synth s;
  const DataPlan p = two_phase_plan(s.d.dir);
  BatchLoader l;
  s1::LoaderGuard guard{l};
  data::set_loader_options(l, host_opts());
  l.start(p, {&s.f}, 4, 32, 8, 0);
  int bad = 0;
  for (int64_t k = 0; k < 140; ++k) {
    TokenSlot& sl = l.acquire(k);
    const auto want = data::assemble_batch(p, {&s.f}, k);
    std::vector<std::thread> ranks;
    std::atomic<int> local_bad{0};
    for (int r = 0; r < 4; ++r)
      ranks.emplace_back([&, r] {
        std::this_thread::sleep_for(std::chrono::microseconds(200 * r));
        if (std::memcmp(sl.host, want.data(), want.size() * 4) != 0) ++local_bad;   // slot unchanged while held
        l.release(sl, r);
      });
    for (auto& t : ranks) t.join();
    bad += local_bad.load();
  }
  l.stop();
  CHECK_EQ(bad, 0);
}

// A read error reaches acquire() as an exception, not a hang and not a crash: the token file is cut to half its length
// after it was opened, so windows of the second half fail. pread mode: EOF; mmap mode: SIGBUS on the loader's mapping,
// caught by the guard. direct assembly (pread) reports the same error.
NM_TEST(s1_loader, read_error_propagates) {
  for (auto mode : {data::ReadMode::Pread, data::ReadMode::Mmap}) {
    Synth s;
    DataPlan pb;
    pb.phases.push_back(s1::levanter_phase(s.d.dir, 0, 60, 8, 512));
    data::validate_plan(pb, 1);   // the meta is loaded (sizes checked) before the file shrinks underneath the run
    s1::truncate_file(s.d.dir + "/tokens.u32", s.d.n_tokens * 2);
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::set_loader_options(l, host_opts(mode));
    l.start(pb, {&s.f}, 1, 8, 4, 0);
    std::string what;
    try {
      for (int64_t k = 0; k < 60; ++k) {
        TokenSlot& sl = l.acquire(k);
        l.release(sl, 0);
      }
    } catch (const std::exception& e) {
      what = e.what();
    }
    l.stop();
    CHECK(what.find(mode == data::ReadMode::Pread ? "EOF" : "SIGBUS") != std::string::npos);
    CHECK_THROWS([&] { for (int64_t k = 0; k < 60; ++k) data::assemble_batch(pb, {&s.f}, k); }(), "EOF");
  }
}

// stop() while a worker is blocked in acquire() (an abnormal shutdown: rank 1 never releases step 0, so step 3 cannot
// be staged): the blocked worker wakes with "loader stopped" and stop() frees the slots only after it left acquire()
// (heap-use-after-free under ASan before the S1 review fix). Repeated to hit many interleavings.
NM_TEST(s1_loader, stop_while_acquire_blocked) {
  Synth s;
  DataPlan p;
  p.phases.push_back(s1::levanter_phase(s.d.dir, 0, 60, 8, 512));
  int stopped = 0;
  for (int it = 0; it < 200; ++it) {
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::set_loader_options(l, host_opts());
    l.start(p, {&s.f}, 2, 4, 4, 0);
    for (int64_t k = 0; k < 3; ++k) l.release(l.acquire(k), 0);
    std::atomic<bool> entered{false};
    std::string what;
    std::thread t([&] {
      entered = true;
      try {
        (void)l.acquire(3);
      } catch (const std::exception& e) {
        what = e.what();
      }
    });
    while (!entered) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::microseconds(it % 4 == 0 ? 0 : 200));
    l.stop();
    t.join();
    stopped += what.find("loader stopped") != std::string::npos || what.find("not started") != std::string::npos;
  }
  CHECK_EQ(stopped, 200);
}

// release() and acquire() racing stop() from other threads, and every call after stop(): an exception, never a touch
// of freed memory; stop() is idempotent and may run on two threads at once.
NM_TEST(s1_loader, calls_race_stop) {
  Synth s;
  DataPlan p;
  p.phases.push_back(s1::levanter_phase(s.d.dir, 0, 60, 8, 512));
  for (int it = 0; it < 200; ++it) {
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::set_loader_options(l, host_opts());
    l.start(p, {&s.f}, 2, 4, 4, 0);
    TokenSlot& sl = l.acquire(0);
    std::thread rel([&] {
      try {
        l.release(sl, 1);
        l.release(sl, 0);
        l.release(l.acquire(1), 0);
      } catch (const std::exception&) {
      }
    });
    std::thread stop2([&] { l.stop(); });
    l.stop();
    rel.join();
    stop2.join();
    CHECK_THROWS(l.acquire(2), "not started");
    CHECK_THROWS(l.release(sl, 0), "not started");
    l.stop();   // no-op
  }
}

// start() refuses bad inputs before it allocates and keeps the pending options for the next start(): a TokenFile of
// another dataset (it would serve other tokens), a plan with a gap (the readers' phase scan relies on validation), and
// with pinned slots more ranks than CUDA devices.
NM_TEST(s1_loader, start_checks_inputs) {
  Synth s;
  Synth other("synth-other", 250);   // same length, other documents: another tokens_sha256
  const DataPlan p = two_phase_plan(s.d.dir);
  BatchLoader l;
  s1::LoaderGuard guard{l};
  data::LoaderOptions o = host_opts();
  o.hash_every = 7;
  data::set_loader_options(l, o);
  CHECK_THROWS(l.start(p, {&other.f}, 1, 4, 2, 0), "holds dataset synth-other");
  CHECK_THROWS(data::assemble_batch(p, {&other.f}, 0), "holds dataset synth-other");
  DataPlan gap = p;
  gap.phases[1].step_begin = 61;
  CHECK_THROWS(l.start(gap, {&s.f}, 1, 4, 2, 0), "contiguous from step 0");
  l.start(p, {&s.f}, 1, 4, 2, 0);   // the options survived both refusals: hashes at k = 0, 7, 14
  CHECK_EQ(drain(l, p, s.f, 0, 15, 1), 0);
  const auto st = data::loader_stats(l);
  l.stop();
  CHECK(st.hashes.size() == 3 && st.hashes[1].first == 7 && st.hashes[2].first == 14);
  if (have_gpu()) {
    int n_dev = 0;
    NM_CUDA(cudaGetDeviceCount(&n_dev));
    if (n_dev < kMaxGpus) {
      BatchLoader g;
      s1::LoaderGuard gg{g};
      Phase ph = s1::levanter_phase(s.d.dir, 0, 60, 8 * (n_dev + 1), 512);
      DataPlan pg;
      pg.phases.push_back(ph);
      CHECK_THROWS(g.start(pg, {&s.f}, n_dev + 1, 4, 2, 0), "one CUDA device per rank");
    }
  }
}

// GPU path: pinned slots, per-rank events, copy_rank_rows; the device rows equal the plan's rows.
NM_TEST(s1_loader, gpu_h2d_rows) {
  if (!have_gpu()) SKIP("no CUDA device");
  Synth s;
  const DataPlan p = two_phase_plan(s.d.dir);
  BatchLoader l;
  s1::LoaderGuard guard{l};
  l.start(p, {&s.f}, 1, 16, 8, 0);
  int32_t* d = nullptr;
  NM_CUDA(cudaMalloc(&d, 16 * 512 * 4));
  cudaStream_t st;
  NM_CUDA(cudaStreamCreate(&st));
  std::vector<int32_t> back(16 * 512);
  int bad = 0;
  for (int64_t k = 0; k < 139; ++k) {
    TokenSlot& sl = l.acquire(k);
    const auto [B, S] = data::slot_shape(l, k);
    data::copy_rank_rows(l, sl, 0, d, st);
    l.release(sl, 0);   // the stager waits on the copy event before reusing the slot
    NM_CUDA(cudaMemcpyAsync(back.data(), d, size_t(B) * S * 4, cudaMemcpyDeviceToHost, st));
    NM_CUDA(cudaStreamSynchronize(st));
    const auto want = data::assemble_batch(p, {&s.f}, k);
    bad += std::memcmp(back.data(), want.data(), want.size() * 4) != 0;
  }
  // the last step: its copy is still in flight and the slot still held when stop() runs; stop() waits on the slot's
  // copy event before cudaFreeHost, and a copy after stop() is refused
  TokenSlot& last = l.acquire(139);
  data::copy_rank_rows(l, last, 0, d, st);
  l.stop();
  CHECK_THROWS(data::copy_rank_rows(l, last, 0, d, st), "not started");
  NM_CUDA(cudaMemcpyAsync(back.data(), d, size_t(16) * 256 * 4, cudaMemcpyDeviceToHost, st));
  NM_CUDA(cudaStreamSynchronize(st));
  const auto want = data::assemble_batch(p, {&s.f}, 139);
  bad += std::memcmp(back.data(), want.data(), want.size() * 4) != 0;
  cudaStreamDestroy(st);
  cudaFree(d);
  CHECK_EQ(bad, 0);
}
