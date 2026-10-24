// Training data: converted flat token files, the exact levanter order, data plans (phases, passes, forks), and the
// prefetch ring + pinned slots that feed the devices. DESIGN.md §5.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

// ------------------------------------------------------------------------------------------------ converted files
// Written once by tools/convert/levanter_to_flat.py (Python, vis node) from the levanter zarr3/tensorstore cache:
//   <dir>/tokens.u32        all tokens of the cache in cache order, little-endian uint32 (10,000,000,738 x 4 B)
//   <dir>/segs_per_seq.u8   segments of each S-token window (EOS-derived), for max_segs_per_seq and data queries
//   <dir>/docs.u64          document start offsets (token index), plus <dir>/doc_bytes.u64 [n_docs, 2] source byte
//                           ranges when the source has them (maps resegmented / repacked sequences across tokenizers)
//   <dir>/meta.json         the fields below + sha256 of every file + the source path and its ledger
// Eval sets (eval.hpp) use the same layout with the windows of the 16 domains concatenated in component order.
struct DatasetMeta {
  std::string name, dir;
  uint64_t n_tokens = 0;          // 10,000,000,738
  int seq_len = 0;                // 4096 (window length the counts below refer to)
  uint64_t n_seqs = 0;            // n_tokens / seq_len = 2,441,406 (tail dropped)
  uint64_t n_docs = 0;
  int vocab = 0;                  // 128256
  int32_t bos = 128000, eos = 128001;
  int max_segs_per_seq = 0;       // 17 boundaries-based for fineweb-edu-10B at 4096 (plus EOS-in-text, measured)
  std::string tokens_sha256;
  std::string meta_sha256;        // sha256 of meta.json: the value hashed into the run identity
};
DatasetMeta load_dataset_meta(const std::string& dir);

// mmap of tokens.u32 (MAP_SHARED | read-only; page cache shared by the ranks of the process) and of docs.u64.
class TokenFile {
 public:
  void open(const DatasetMeta& m);
  const uint32_t* window(uint64_t seq, int S) const;  // tokens[seq*S, seq*S + S)
  uint64_t doc_of_token(uint64_t token_index) const;  // binary search in docs.u64
  uint64_t n_tokens() const { return n_; }
 private:
  const uint32_t* base_ = nullptr;
  const uint64_t* docs_ = nullptr;
  uint64_t n_ = 0, n_docs_ = 0;
  size_t map_bytes_ = 0;
};

// ------------------------------------------------------------------------------------------------ JAX threefry port
// Bit-exact port of jax_threefry_partitionable=True semantics (data.md §4.1; model.md §4.2), used by the data order,
// the parameter init and (same code, device build) the in-graph RNG of model.hpp RngKey.
struct Key { uint32_t k0, k1; };
void threefry2x32(Key k, uint32_t& x0, uint32_t& x1);       // 20 rounds, in place
Key prng_key(uint32_t seed);                               // PRNGKey(seed) = (0, seed)
Key split(Key k, uint32_t i);                              // split(k)[i] = T(k, (0, i))
Key fold_in(Key k, uint32_t d);                            // T(k, (0, d))
uint32_t bits32(Key k, uint64_t i);                        // bits32(k, n)[i] = x0 ^ x1 of T(k, (0, i))
void permutation(Key k, uint32_t n, uint32_t* out);        // jax.random.permutation(k, arange(n)): stable-sort rounds

// ------------------------------------------------------------------------------------------------ levanter order
// s(j) = (a * ((blk*B + pi_blk[j mod B]) mod N) + b) mod N,  blk = j div B,  pi_blk = permutation(fold_in(mix_key, blk))
// with data_seed 42: mix_key = (1832780943, 270669613), a = 830293, b = 746237, N = 2,441,406, B = 2048 (data.md §4.4).
// a and b come from levanter's numpy PCG64 draw, keyed by data_seed in a table that tools/convert fills for every seed
// in use; an unknown seed is a ConfigError.
class LevanterOrder {
 public:
  LevanterOrder(uint32_t data_seed, uint64_t n_seqs, int mix_block);
  uint64_t operator()(uint64_t j) const;                   // sequence id of mixture index j (block cache of pi_blk)
  std::string table_sha256(uint64_t count) const;          // T0.1: s(0..5,859,327) -> dde0ca7b...655b
 private:
  Key mix_key_{};
  uint64_t N_, a_, b_;
  int block_;
  mutable std::vector<uint32_t> cache_pi_;
  mutable int64_t cache_blk_ = -1;
};

// ------------------------------------------------------------------------------------------------ data plan
// The order of a run is a pure function of (DataPlan, step): no loader state is ever checkpointed. A plan is a list of
// phases; phase p covers updates [step_begin, step_end) with its own dataset, global batch and sequence length (a
// change of batch or length is a new step regime, model.hpp StepRegime), and maps (step, row) to a sequence:
//   Levanter   j = index_base + (step - step_begin) * B + row;  seq = LevanterOrder(j)      (the baseline: one phase)
//   Repeat     mixture index (j mod subset) of a Levanter order, for a real-valued number of passes (DATA_EPOCHS=2.4
//              repeats without reshuffle, like levanter's max_train_batches restart); pass q optionally reshuffled with
//              permutation(fold_in(Key(seed), q))
//   Explicit   seq ids from a u32 file (pre-materialized pass caches, resegmented / repacked data from tools/convert)
// Forks (run.parent = <id>@<k>) copy the parent's plan up to k and may append phases after k: batches before k are
// bit-identical to the parent's. Gate: seq_of equals the order the JAX runs read (DATA_EPOCHS 2.4, PASS_DATA rs,
// repack), bitwise.
enum class OrderKind : uint8_t { Levanter = 0, Repeat = 1, Explicit = 2 };
struct Phase {
  int64_t step_begin = 0, step_end = 0;
  std::string dataset;            // dataset dir name (its meta sha is hashed)
  int global_batch = 128;
  int seq_len = 4096;
  OrderKind kind = OrderKind::Levanter;
  uint32_t seed = 42;
  uint64_t index_base = 0;        // first mixture index used by this phase
  uint64_t subset = 0;            // Repeat: number of mixture indices in the repeated subset
  double passes = 1.0;            // Repeat: real-valued
  bool reshuffle = false;
  std::string explicit_file;      // Explicit
};
struct DataPlan {
  std::vector<Phase> phases;
  // Query API (also in the C ABI and `marincpp order`): which sequence a row of a step reads; how many times a
  // DOCUMENT was read up to a step (documents, not sequence ids, so the count survives resegmentation and repacking);
  // the steps that start a new pass (checkpoints are kept there when run.keep_pass_boundaries).
  uint64_t seq_of(int64_t step, int row) const;
  uint32_t times_seen(uint64_t doc, int64_t up_to_step) const;
  std::vector<int64_t> pass_boundaries() const;
  const Phase& phase_at(int64_t step) const;
  std::string canonical() const;  // hashed into the run identity
};
DataPlan load_data_plan(const std::string& path_or_empty, const struct Config& c);

// ------------------------------------------------------------------------------------------------ loader
// Data depth is separate from launch depth (DESIGN.md §5.4). `reader_threads` (16-32) fill a pageable ring of
// `prefetch_steps` (256) future steps with pread from the mmap (GPFS cold-batch p90 measured 250-1,000 ms; 256 steps
// cover 48 s at 130m). A stager thread copies ring entries into 3 pinned slots, converting u32 -> i32 and computing
// H(B, k) at k = 0 and every eval step (data.md §7 guard). Each rank's worker issues cudaMemcpyAsync from its slice of a
// pinned slot before launching g_step and records an event; the slot is refilled after every rank's event completed.
// Launch lookahead stays 2 graphs, so the SIGTERM drain does not depend on the ring depth.
struct TokenSlot {
  int64_t step = -1;
  int32_t* host = nullptr;        // pinned, [B, S]
  std::atomic<int> ready{0};
  std::atomic<int> consumers{0};
  cudaEvent_t copied[kMaxGpus] = {};
};
class BatchLoader {
 public:
  void start(const DataPlan& plan, const std::vector<TokenFile*>& files, int nranks, int prefetch_steps,
             int reader_threads, int64_t first_step);
  TokenSlot& acquire(int64_t step);   // blocks until filled (counts throughput/loading_time when it waits)
  void release(TokenSlot& s, int rank);
  void stop();
 private:
  std::unique_ptr<TokenSlot[]> slots_;  // 3 pinned slots; TokenSlot holds atomics: fixed array, never moved
  std::vector<uint32_t> ring_;          // pageable, prefetch_steps * B * S
  std::vector<std::thread> readers_;
  std::thread stager_;
  std::atomic<bool> stop_{false};
};

}  // namespace nm
