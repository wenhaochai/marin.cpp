// Prefetch ring + pinned slots + async H2D handoff. DESIGN.md §5.4.
//
//   reader threads (reader_threads): the first reader to reach step k computes the sequence ids of its B rows outside
//                                    the lock (step_seqs; a reshuffled pass's permutation is computed once per
//                                    process), then readers claim (step, row) in step order up to prefetch_steps ahead
//                                    of the stager and read each window into the pageable ring entry of that step
//                                    (pread, or a SIGBUS-guarded copy from a loader-private mapping in ReadMode::Mmap)
//   stager (1 thread):               for k = first_step, first_step+1, ...: waits for ring entry k to be complete and
//                                    for pinned slot k % nslots to be free (every rank released it and its copy event
//                                    completed), copies the entry into the slot (u32 -> i32: same bits, ids < 2^31),
//                                    computes H(B, k) at k = 0 and every hash_every steps, marks the slot ready
//   worker r:                        acquire(k) -> copy_rank_rows (cudaMemcpyAsync of rows [rB/n, (r+1)B/n) + event)
//                                    -> release(slot, r)
//   stop():                          refuses new callers, wakes blocked ones, joins readers and stager, waits until no
//                                    caller is inside acquire / release / copy_rank_rows, then frees slots and ring
// The order is a pure function of (plan, step): a restart (resume, fork) is start(..., first_step = k).
#include <cuda_runtime.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>

#include "data/internal.hpp"
#include "marincpp/data.hpp"

namespace nm {
namespace data {
namespace {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------- SIGBUS guard (ReadMode::Mmap)
// A storage error or a file shorter than its mapping turns a load from the mapping into SIGBUS. A reader arms a jump
// buffer and the page range of its copy; the handler jumps back when a kernel-raised SIGBUS hits the armed range on the
// faulting thread, and otherwise reinstalls the previous action and returns, so the fault repeats under that action
// (default: terminate with a core dump).
// volatile + signal fences: the handler reads these, so the compiler must neither drop the arming stores (a later store
// of nullptr and a memcpy that does not read them would make them dead at -O3) nor move the copy across them.
thread_local sigjmp_buf* volatile t_bus_jmp = nullptr;
thread_local volatile uintptr_t t_bus_lo = 0, t_bus_hi = 0;
struct sigaction g_prev_bus;
std::once_flag g_bus_once;

void bus_handler(int, siginfo_t* si, void*) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(si->si_addr);
  if (t_bus_jmp != nullptr && si->si_code > 0 && a >= t_bus_lo && a < t_bus_hi) {
    sigjmp_buf* j = t_bus_jmp;
    t_bus_jmp = nullptr;
    siglongjmp(*j, 1);
  }
  sigaction(SIGBUS, &g_prev_bus, nullptr);
}

void install_bus_handler() {
  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = bus_handler;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;   // SIGBUS stays unblocked in the handler: siglongjmp needs no saved mask
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGBUS, &sa, &g_prev_bus) != 0)
    fatal(NM_WHERE, std::string("sigaction(SIGBUS) failed: ") + std::strerror(errno));
}

// memcpy from a file mapping; false if the copy raised SIGBUS.
bool guarded_copy(void* dst, const void* src, size_t n) {
  std::call_once(g_bus_once, install_bus_handler);
  sigjmp_buf jb;
  if (sigsetjmp(jb, 0) != 0) return false;   // the handler cleared t_bus_jmp
  t_bus_lo = reinterpret_cast<uintptr_t>(src) & ~uintptr_t(4095);
  t_bus_hi = (reinterpret_cast<uintptr_t>(src) + n + 4095) & ~uintptr_t(4095);
  t_bus_jmp = &jb;
  std::atomic_signal_fence(std::memory_order_seq_cst);
  std::memcpy(dst, src, n);
  std::atomic_signal_fence(std::memory_order_seq_cst);
  t_bus_jmp = nullptr;
  return true;
}

// ---------------------------------------------------------------------------- state
struct FileMap {
  const TokenFile* f = nullptr;
  const char* map = nullptr;   // ReadMode::Mmap: loader-private read-only mapping of the whole file, MADV_RANDOM
  size_t bytes = 0;
};

struct RingEntry {
  int64_t step = -1;      // the step whose rows this entry holds (set once its sequence ids are known)
  int64_t prep = -1;      // the step whose sequence ids a reader is computing outside the lock
  int B = 0, S = 0;
  int rows_done = 0;
  size_t file = 0;        // index into State::files
  std::vector<uint64_t> seq;
};

struct State {
  LoaderOptions opt;
  DataPlan plan;
  std::vector<FileMap> files;                 // distinct token files of the plan
  std::vector<size_t> phase_file;             // per phase: index into files
  int nranks = 0, depth = 0, nslots = 3;
  int64_t first = 0, end = 0;
  size_t max_tokens = 0;                      // max over phases of B * S: ring entry and slot size
  uint32_t* ring = nullptr;                   // BatchLoader::ring_
  TokenSlot* slots = nullptr;                 // BatchLoader::slots_
  std::vector<RingEntry> entries;
  std::mutex mu;                              // guards everything below and each slot's step / consumers handoff
  std::condition_variable cv_claim, cv_ready, cv_slot, cv_idle;
  int64_t claim_step = 0;
  int claim_row = 0;
  int64_t staged_upto = 0;
  bool stop = false;                          // stop() began: callers are refused
  int callers = 0;                            // callers inside acquire / release / copy_rank_rows
  std::exception_ptr error;
  LoaderStats stats;
  bool pinned = false;
  std::mutex stop_mu;                         // one stop() at a time
};

std::mutex g_tab_mu;
std::map<const BatchLoader*, std::shared_ptr<State>>& table() {
  static std::map<const BatchLoader*, std::shared_ptr<State>> t;
  return t;
}
std::map<const BatchLoader*, LoaderOptions>& pending_options() {
  static std::map<const BatchLoader*, LoaderOptions> t;
  return t;
}

std::shared_ptr<State> state_of(const BatchLoader* l) {
  std::lock_guard<std::mutex> g(g_tab_mu);
  auto it = table().find(l);
  if (it == table().end()) fatal(NM_WHERE, "BatchLoader is not started");
  return it->second;
}

bool stopping(const State& st) { return st.stop || st.error; }   // caller holds st.mu

void fail(State* st, std::exception_ptr e) {   // caller holds st->mu
  if (!st->error) st->error = e;
  st->cv_claim.notify_all();
  st->cv_ready.notify_all();
  st->cv_slot.notify_all();
}

// A caller of acquire / release / copy_rank_rows: refused once stop() began; while admitted it holds the state and is
// counted, and stop() frees the slots only when the count is 0, so no caller touches freed memory.
class Caller {
 public:
  explicit Caller(const BatchLoader* l) : st_(state_of(l)) {
    std::lock_guard<std::mutex> g(st_->mu);
    if (st_->stop) fatal(NM_WHERE, "loader stopped");
    ++st_->callers;
  }
  ~Caller() {
    std::lock_guard<std::mutex> g(st_->mu);
    if (--st_->callers == 0) st_->cv_idle.notify_all();
  }
  Caller(const Caller&) = delete;
  Caller& operator=(const Caller&) = delete;
  State& operator*() const { return *st_; }
  State* operator->() const { return st_.get(); }

 private:
  std::shared_ptr<State> st_;
};

bool owns_slot(const State& st, const TokenSlot& s) { return st.slots && &s >= st.slots && &s < st.slots + st.nslots; }

void read_mapped(const FileMap& fm, uint64_t seq, int S, uint32_t* dst) {
  NM_REQUIRE((seq + 1) * static_cast<uint64_t>(S) <= fm.f->n_tokens(), "window past the end of the token file");
  if (!guarded_copy(dst, fm.map + seq * static_cast<uint64_t>(S) * 4, static_cast<size_t>(S) * 4))
    fatal(NM_WHERE, "SIGBUS reading window " + std::to_string(seq) + " of " + token_file_path(*fm.f) +
                        " through its mapping: the file is shorter than its meta.json or its storage failed");
}

// Frees what start() allocated: slot memory and events, loader-private mappings. Safe on a partial allocation.
void free_resources(State& st) {
  if (st.slots)
    for (int i = 0; i < st.nslots; ++i) {
      TokenSlot& s = st.slots[i];
      if (st.pinned) {
        for (int r = 0; r < kMaxGpus; ++r)
          if (s.copied[r]) {
            cudaEventSynchronize(s.copied[r]);   // the H2D copy from s.host has finished before cudaFreeHost
            cudaEventDestroy(s.copied[r]);
            s.copied[r] = nullptr;
          }
        if (s.host) cudaFreeHost(s.host);
      } else {
        std::free(s.host);
      }
      s.host = nullptr;
    }
  for (FileMap& fm : st.files)
    if (fm.map) {
      ::munmap(const_cast<char*>(fm.map), fm.bytes);
      fm.map = nullptr;
    }
}

void reader_main(State* st) {
  std::unique_lock<std::mutex> lk(st->mu);
  for (;;) {
    st->cv_claim.wait(lk, [&] {
      if (stopping(*st) || st->claim_step >= st->end) return true;
      if (st->claim_step - st->staged_upto >= st->depth) return false;
      const RingEntry& e = st->entries[static_cast<size_t>(st->claim_step % st->depth)];
      return e.step == st->claim_step || e.prep != st->claim_step;   // ready to claim, or nobody prepares it
    });
    if (stopping(*st) || st->claim_step >= st->end) return;
    const int64_t k = st->claim_step;
    RingEntry& e = st->entries[static_cast<size_t>(k % st->depth)];
    if (e.step != k) {   // first reader at step k: its rows' sequence ids, computed without holding the lock
      e.prep = k;
      lk.unlock();
      std::vector<uint64_t> seq;
      size_t pi = 0;
      std::exception_ptr err;
      try {
        pi = phase_index(st->plan, k);
        seq.resize(static_cast<size_t>(st->plan.phases[pi].global_batch));
        step_seqs(st->plan, k, seq.data());
      } catch (...) {
        err = std::current_exception();
      }
      lk.lock();
      e.prep = -1;
      if (err) {
        fail(st, err);
        return;
      }
      const Phase& ph = st->plan.phases[pi];
      e.seq.swap(seq);
      e.B = ph.global_batch;
      e.S = ph.seq_len;
      e.file = st->phase_file[pi];
      e.rows_done = 0;
      e.step = k;
      st->cv_claim.notify_all();
      continue;
    }
    const int r = st->claim_row++;
    if (st->claim_row == e.B) {
      ++st->claim_step;
      st->claim_row = 0;
    }
    uint32_t* dst = st->ring + static_cast<size_t>(k % st->depth) * st->max_tokens + static_cast<size_t>(r) * e.S;
    const uint64_t seq = e.seq[static_cast<size_t>(r)];
    const FileMap& fm = st->files[e.file];
    const int S = e.S;
    lk.unlock();
    const auto t0 = Clock::now();
    std::exception_ptr err;
    try {
      if (st->opt.read == ReadMode::Pread) pread_window(*fm.f, seq, S, dst);
      else read_mapped(fm, seq, S, dst);
    } catch (...) {
      err = std::current_exception();
    }
    const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    lk.lock();
    if (err) {
      fail(st, err);
      return;
    }
    st->stats.windows_read += 1;
    st->stats.bytes_read += static_cast<int64_t>(S) * 4;
    st->stats.read_seconds += dt;
    if (++e.rows_done == e.B) st->cv_ready.notify_all();
  }
}

void stager_main(State* st) {
  try {
    for (int64_t k = st->first; k < st->end; ++k) {
      RingEntry& e = st->entries[static_cast<size_t>(k % st->depth)];
      TokenSlot& s = st->slots[k % st->nslots];
      {
        std::unique_lock<std::mutex> lk(st->mu);
        st->cv_ready.wait(lk, [&] { return stopping(*st) || (e.step == k && e.rows_done == e.B); });
        if (stopping(*st)) return;
        st->cv_slot.wait(lk, [&] { return stopping(*st) || s.step < 0 || s.consumers.load() >= st->nranks; });
        if (stopping(*st)) return;
        s.ready.store(0, std::memory_order_release);
      }
      if (st->pinned)   // the slot's previous H2D copies (every rank recorded its event before release)
        for (int r = 0; r < st->nranks; ++r)
          if (s.copied[r]) NM_CUDA(cudaEventSynchronize(s.copied[r]));
      const size_t n = static_cast<size_t>(e.B) * e.S;
      std::memcpy(s.host, st->ring + static_cast<size_t>(k % st->depth) * st->max_tokens, n * 4);
      std::string h;
      if (k == 0 || (st->opt.hash_every > 0 && k % st->opt.hash_every == 0)) h = sha256_hex(s.host, n * 4);
      std::lock_guard<std::mutex> g(st->mu);
      s.step = k;
      s.consumers.store(0);
      s.ready.store(1, std::memory_order_release);
      st->staged_upto = k + 1;
      st->stats.steps_staged += 1;
      if (!h.empty()) st->stats.hashes.emplace_back(k, h);
      st->cv_slot.notify_all();
      st->cv_claim.notify_all();
    }
  } catch (...) {
    std::lock_guard<std::mutex> g(st->mu);
    fail(st, std::current_exception());
  }
}

}  // namespace

void set_loader_options(BatchLoader& l, const LoaderOptions& o) {
  NM_REQUIRE(o.pinned_slots >= 2, "at least 2 pinned slots");
  std::lock_guard<std::mutex> g(g_tab_mu);
  NM_REQUIRE(!table().count(&l), "set_loader_options after start()");
  pending_options()[&l] = o;
}

LoaderStats loader_stats(const BatchLoader& l) {
  auto st = state_of(&l);
  std::lock_guard<std::mutex> g(st->mu);
  return st->stats;
}

std::pair<int, int> slot_shape(const BatchLoader& l, int64_t step) {
  auto st = state_of(&l);
  const Phase& ph = st->plan.phase_at(step);
  return {ph.global_batch, ph.seq_len};
}

void copy_rank_rows(const BatchLoader& l, TokenSlot& slot, int rank, int32_t* dst, cudaStream_t stream) {
  Caller c(&l);
  State& st = *c;
  NM_REQUIRE(rank >= 0 && rank < st.nranks, "rank outside the loader's ranks");
  NM_REQUIRE(owns_slot(st, slot), "copy from a slot of another loader");
  NM_REQUIRE(slot.ready.load(std::memory_order_acquire) == 1, "copy from a slot that is not ready");
  const Phase& ph = st.plan.phase_at(slot.step);
  const size_t rows = static_cast<size_t>(ph.global_batch / st.nranks), S = static_cast<size_t>(ph.seq_len);
  NM_CUDA(cudaMemcpyAsync(dst, slot.host + rank * rows * S, rows * S * 4, cudaMemcpyHostToDevice, stream));
  if (slot.copied[rank]) NM_CUDA(cudaEventRecord(slot.copied[rank], stream));
}

}  // namespace data

void BatchLoader::start(const DataPlan& plan, const std::vector<TokenFile*>& files, int nranks, int prefetch_steps,
                        int reader_threads, int64_t first_step) {
  NM_REQUIRE(!plan.phases.empty(), "empty data plan");
  NM_REQUIRE(files.size() == 1 || files.size() == plan.phases.size(), "one TokenFile, or one per phase");
  NM_REQUIRE(nranks >= 1 && nranks <= kMaxGpus, "nranks in [1, kMaxGpus]");
  NM_REQUIRE(prefetch_steps >= 1 && reader_threads >= 1, "prefetch_steps and reader_threads must be positive");
  // contiguous phases from step 0, batches divisible by nranks, known datasets and orders: the readers rely on it
  data::validate_plan(plan, nranks);
  auto st = std::make_shared<data::State>();
  {
    std::lock_guard<std::mutex> g(data::g_tab_mu);
    NM_REQUIRE(!data::table().count(this), "BatchLoader::start twice without stop()");
    auto it = data::pending_options().find(this);
    if (it != data::pending_options().end()) st->opt = it->second;   // consumed only by a start() that succeeds
  }
  st->plan = plan;
  st->nranks = nranks;
  st->depth = prefetch_steps;
  st->nslots = st->opt.pinned_slots;
  st->first = first_step;
  st->end = plan.phases.back().step_end;
  NM_REQUIRE(first_step >= 0 && first_step <= st->end, "first_step outside the plan");
  for (size_t i = 0; i < plan.phases.size(); ++i) {
    const Phase& ph = plan.phases[i];
    const TokenFile* f = files.size() == 1 ? files[0] : files[i];
    NM_REQUIRE(f != nullptr, "null TokenFile");
    data::check_token_file(*f, ph);   // the file must hold this phase's dataset
    size_t fi = 0;
    while (fi < st->files.size() && st->files[fi].f != f) ++fi;
    if (fi == st->files.size()) st->files.push_back(data::FileMap{f, nullptr, 0});
    st->phase_file.push_back(fi);
    st->max_tokens = std::max(st->max_tokens, static_cast<size_t>(ph.global_batch) * ph.seq_len);
  }
  st->pinned = !st->opt.host_only;
  int dev0 = 0;
  if (st->pinned) {
    int n_dev = 0;
    NM_CUDA(cudaGetDeviceCount(&n_dev));
    NM_REQUIRE(nranks <= n_dev, "pinned slots need one CUDA device per rank (rank r = device r): " +
                                    std::to_string(nranks) + " ranks, " + std::to_string(n_dev) + " devices");
    NM_CUDA(cudaGetDevice(&dev0));
  }
  try {   // a failure frees everything allocated so far (free_resources handles a partial allocation)
    ring_.assign(static_cast<size_t>(prefetch_steps) * st->max_tokens, 0u);
    st->ring = ring_.data();
    st->entries.resize(static_cast<size_t>(prefetch_steps));
    if (st->opt.read == data::ReadMode::Mmap)
      for (data::FileMap& fm : st->files) {   // loader-private: the access-pattern hint stays off other users' mappings
        fm.bytes = static_cast<size_t>(fm.f->n_tokens()) * 4;
        void* p = ::mmap(nullptr, fm.bytes, PROT_READ, MAP_SHARED, data::token_file_fd(*fm.f), 0);
        if (p == MAP_FAILED)
          fatal(NM_WHERE, "mmap of " + data::token_file_path(*fm.f) + " failed: " + std::strerror(errno));
        fm.map = static_cast<const char*>(p);
        if (::madvise(p, fm.bytes, MADV_RANDOM) != 0)   // random 16 KiB windows: no page-fault read-around
          fatal(NM_WHERE, std::string("madvise(MADV_RANDOM) failed: ") + std::strerror(errno));
      }
    slots_.reset(new TokenSlot[st->nslots]);
    st->slots = slots_.get();
    for (int i = 0; i < st->nslots; ++i) {
      TokenSlot& s = slots_[i];
      s.step = -1;
      if (st->pinned) {
        void* p = nullptr;
        NM_CUDA(cudaHostAlloc(&p, st->max_tokens * 4, cudaHostAllocPortable));
        s.host = static_cast<int32_t*>(p);
        for (int r = 0; r < nranks; ++r) {   // an event belongs to the device current at creation: rank r = device r
          NM_CUDA(cudaSetDevice(r));
          NM_CUDA(cudaEventCreateWithFlags(&s.copied[r], cudaEventDisableTiming));
        }
      } else {
        s.host = static_cast<int32_t*>(std::aligned_alloc(4096, (st->max_tokens * 4 + 4095) / 4096 * 4096));
        NM_REQUIRE(s.host != nullptr, "host slot allocation failed");
      }
    }
    if (st->pinned) NM_CUDA(cudaSetDevice(dev0));
  } catch (...) {
    if (st->pinned) cudaSetDevice(dev0);
    data::free_resources(*st);
    st->slots = nullptr;
    slots_.reset();
    ring_.clear();
    ring_.shrink_to_fit();
    throw;
  }
  st->claim_step = st->staged_upto = first_step;
  stop_.store(false);
  {
    std::lock_guard<std::mutex> g(data::g_tab_mu);
    data::table()[this] = st;
    data::pending_options().erase(this);
  }
  try {
    for (int t = 0; t < reader_threads; ++t) readers_.emplace_back(data::reader_main, st.get());
    stager_ = std::thread(data::stager_main, st.get());
  } catch (...) {
    stop();   // joins the threads that did start and frees everything
    std::lock_guard<std::mutex> g(data::g_tab_mu);
    data::pending_options()[this] = st->opt;
    throw;
  }
}

TokenSlot& BatchLoader::acquire(int64_t step) {
  data::Caller c(this);
  data::State& st = *c;
  NM_REQUIRE(step >= st.first && step < st.end, "acquire of a step outside [first_step, plan end)");
  std::unique_lock<std::mutex> lk(st.mu);
  TokenSlot& s = st.slots[step % st.nslots];
  auto ready = [&] { return s.step == step && s.ready.load(std::memory_order_acquire) == 1; };
  if (!data::stopping(st) && !ready()) {
    const auto t0 = data::Clock::now();
    st.cv_slot.wait(lk, [&] { return data::stopping(st) || ready(); });
    st.stats.wait_seconds += std::chrono::duration<double>(data::Clock::now() - t0).count();
  }
  if (st.error) std::rethrow_exception(st.error);
  if (st.stop) fatal(NM_WHERE, "loader stopped");
  return s;
}

void BatchLoader::release(TokenSlot& s, int rank) {
  data::Caller c(this);
  NM_REQUIRE(rank >= 0 && rank < c->nranks, "release by a rank outside the loader's ranks");
  std::lock_guard<std::mutex> g(c->mu);
  NM_REQUIRE(data::owns_slot(*c, s), "release of a slot of another loader");
  s.consumers.fetch_add(1);
  c->cv_slot.notify_all();
}

void BatchLoader::stop() {
  std::shared_ptr<data::State> st;
  {
    std::lock_guard<std::mutex> g(data::g_tab_mu);
    auto it = data::table().find(this);
    if (it == data::table().end()) return;
    st = it->second;
  }
  std::lock_guard<std::mutex> one(st->stop_mu);
  {
    std::lock_guard<std::mutex> g(data::g_tab_mu);
    auto it = data::table().find(this);
    if (it == data::table().end() || it->second != st) return;   // a concurrent stop() finished first
  }
  {
    std::lock_guard<std::mutex> g(st->mu);
    st->stop = true;   // refuses new callers; blocked ones wake and throw "loader stopped"
    stop_.store(true, std::memory_order_release);
    st->cv_claim.notify_all();
    st->cv_ready.notify_all();
    st->cv_slot.notify_all();
  }
  for (auto& t : readers_) t.join();
  readers_.clear();
  if (stager_.joinable()) stager_.join();
  {
    std::unique_lock<std::mutex> lk(st->mu);
    st->cv_idle.wait(lk, [&] { return st->callers == 0; });
  }
  data::free_resources(*st);
  st->slots = nullptr;
  slots_.reset();
  st->ring = nullptr;
  ring_.clear();
  ring_.shrink_to_fit();
  std::lock_guard<std::mutex> g(data::g_tab_mu);
  data::table().erase(this);
}

}  // namespace nm
