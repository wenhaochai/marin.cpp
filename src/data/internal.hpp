// Internal API of src/data (stage S1): what data.hpp does not declare but tests, tools and later stages need.
// DESIGN.md §5. Public types live in include/marincpp/data.hpp.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "marincpp/data.hpp"

namespace nm::data {

// ------------------------------------------------------------------------------------------------ paths
// Root of converted datasets: $NM_DATA, else $NANOMARIN_ROOT/marin_store_big/marincpp (the converter's output).
// Fatal if neither variable is set. A path is an environment fact like run.root, never a config value.
std::string data_root();
// A dataset reference (knob value, Phase::dataset): a name under data_root(), or a path (contains '/'), optionally
// followed by "#sha256=<meta.json sha256>" (the resolved form). Returns the directory; `sha` gets the hash or "".
std::string dataset_dir(const std::string& ref, std::string* sha = nullptr);
std::string strip_hash(const std::string& ref);   // "name#sha256=..." -> "name"
std::string pin_of(const std::string& ref);       // "name#sha256=<hex>" -> "<hex>"; "" when the reference has no pin

// ------------------------------------------------------------------------------------------------ registries
// LCG constants of levanter's linear permutation (data.md §4.2), keyed by (data_seed, N). Built in: (42, 2441406) ->
// (830293, 746237). load_dataset_meta() adds the entries of a dataset's meta.json `lcg` table (converter --seeds and
// --seq-lens; keys "<seed>" or "<seed>@<seq_len>", each with its N: a phase at another seq_len has another N and other
// constants). An unknown (seed, N) is fatal (a config error): the constants come from numpy's PCG64 draw, never recomputed here.
struct Lcg { uint64_t a = 0, b = 0; };
void register_lcg(uint32_t seed, uint64_t n_seqs, uint64_t a, uint64_t b);   // fatal on a conflicting entry
Lcg lcg_for(uint32_t seed, uint64_t n_seqs);

// Dataset metadata by name (DatasetMeta::name, also the dir name): load_dataset_meta() registers; tests may register
// synthetic metas. find_dataset() loads the meta from data_root() on first use when the name is not registered. A
// pinned reference ("#sha256=") must equal the meta.json sha256 of the dataset it names, whether the meta is loaded now
// or was registered before (a registered meta without a sha cannot satisfy a pin): fatal otherwise.
void register_dataset(const DatasetMeta& m);
const DatasetMeta& find_dataset(const std::string& ref);
// One shared read-only mapping per dataset (TokenFile::open once per process), for DataPlan::times_seen and tools.
const TokenFile& shared_token_file(const std::string& ref);

// ------------------------------------------------------------------------------------------------ order
// s(j) = sigma_N((blk*B + pi_blk[j mod B]) mod M), blk = j div B, pi_blk = permutation(fold_in(mix_key, blk), B),
// sigma_N(r) = (a r + b) mod N. M = N is levanter's "restart" wrap over the whole set (data.md §4.4); M = subset < N is
// the same wrap over levanter's max_train_batches slice (datasets.py:858-866 slices the LCG-permuted set to its first
// M sequences before the mixture, so the wrap is mod M and sigma stays over the full N). Thread-safe: block
// permutations are cached per thread.
uint64_t levanter_seq(uint32_t seed, uint64_t n_seqs, int block, uint64_t wrap, uint64_t j);
std::vector<uint32_t> block_permutation(Key mix_key, uint64_t blk, int block);   // uncached
// Start of the mixture block that holds the first index of pass q: the first index whose unwrapped value
// blk*B + pi_blk[j mod B] reaches q*wrap lies in block floor(q*wrap/B), at a position that pi_blk decides. The function
// returns that block's first index, floor(q*wrap/B)*B, not the index itself; when wrap % B != 0 the block mixes the
// end of pass q-1 and the start of pass q.
uint64_t first_index_of_pass(uint64_t q, uint64_t wrap, int block);
// Mixture block size of every Levanter / Repeat phase. data.hpp Phase has no field for it yet, so a plan accepts only
// levanter's 2048 (load_data_plan refuses another data.mix_block).
inline constexpr int kMixBlock = 2048;

// ------------------------------------------------------------------------------------------------ plans
// Phase files (data.plan): TOML sections [phase.0], [phase.1], ... with keys step_begin, step_end, dataset,
// global_batch, seq_len, kind ("levanter" | "repeat" | "explicit"), seed, index_base, subset, passes, reshuffle,
// explicit_file (relative to the plan file). Phases must be contiguous from step 0 to optim.num_steps. Defaults: the
// config's dataset (with its "#sha256=" pin), data.global_batch, model.seq_len, data.data_seed; index_base continues the
// previous phase's stream when it is a Levanter phase of the same seed and dataset, else 0; a Repeat phase with
// passes > 0 and no step_end ends after ceil(passes * subset / B) steps. `dataset` and `explicit_file` may carry
// "#sha256=<hex>" (meta.json sha256, resp. the file's sha256); a pin that does not match the content is fatal.
// Repeat semantics (pass q = m div subset, m = (step - step_begin) * B + row, j = index_base + m): without reshuffle,
// levanter_seq(.., wrap = subset, j) (exactly levanter's DATA_EPOCHS restart); with reshuffle, pass 0 is the same and
// pass q >= 1 reads sigma_N(perm_q[m mod subset]), perm_q = permutation(fold_in(prng_key(seed), q), subset). A reshuffled
// phase needs index_base = 0 (its passes start at mixture index 0; validate_plan refuses other values).
DataPlan parse_data_plan(const std::string& path, const Config& c);
// The phase list must start at 0, be contiguous, have positive lengths, batches divisible by `ranks` (> 0), known
// datasets and orders; explicit files must cover their phase. Fatal with every problem listed.
void validate_plan(const DataPlan& p, int ranks);
// Fork at step k (run.parent = <id>@<k>): the parent's phases cut at k, then `appended` (which must start at k and
// stay contiguous). Batches before k are the parent's, bit for bit.
DataPlan fork_plan(const DataPlan& parent, int64_t k, const std::vector<Phase>& appended);
// Index of the phase that holds `step` (fatal outside the plan).
size_t phase_index(const DataPlan& p, int64_t step);
// Sequence ids of rows 0 .. B-1 of `step` (B = its phase's global batch): DataPlan::seq_of for the whole step with one
// phase and one dataset lookup. Thread-safe; the reshuffle permutation of a pass is computed once per process and
// shared by every thread.
void step_seqs(const DataPlan& p, int64_t step, uint64_t* out);
// times_seen of every sequence of `dataset` (at the phase seq_len) over steps [0, up_to_step): a window counts once per
// (step, row) that reads it. DataPlan::times_seen(doc) is the maximum of this over the windows that overlap the
// document (0 iff no token of it was read; the number of passes when whole passes were read).
std::vector<uint32_t> window_reads(const DataPlan& p, const std::string& dataset, int seq_len, int64_t up_to_step);
uint32_t explicit_id(const std::string& ref, uint64_t i);   // id i of an explicit order file (pin checked; cached mapping)

// ------------------------------------------------------------------------------------------------ batches
// Host reference of data.md §3.1 (training example): seg[0] = 0, seg[i] = #{t < i : tok[t] == eos}; weight 1 at
// positions 0..S-2 and 0 at S-1. The device form is k_doc_meta (S3).
void segment_ids(const int32_t* tok, int rows, int S, int32_t eos, int32_t* seg);
void loss_weights(int rows, int S, float* w);
int segments_in_window(const uint32_t* w, int S, uint32_t eos);   // 1 + #eos at positions 0..S-2
// Batch k of a plan assembled without the loader (one pread per window, so an I/O error is a Fatal, never a SIGBUS):
// int32 [B, S] row-major; and H(B, k) of data.md §7 (sha256 of that buffer, little-endian). `files` holds one TokenFile,
// or one per phase; each must hold its phase's dataset (check_token_file).
std::vector<int32_t> assemble_batch(const DataPlan& p, const std::vector<const TokenFile*>& files, int64_t step);
std::string batch_hash(const DataPlan& p, const std::vector<const TokenFile*>& files, int64_t step);
// pread of one window into `out` (int32/uint32 bytes) from the file behind `f` (one syscall, retried on EINTR / short
// reads); the loader's read path.
void pread_window(const TokenFile& f, uint64_t seq, int S, uint32_t* out);
int token_file_fd(const TokenFile& f);
const std::string& token_file_path(const TokenFile& f);
const std::string& token_file_sha256(const TokenFile& f);   // tokens_sha256 of the meta the file was opened with
// A TokenFile serves a phase only if it holds that phase's dataset: n_tokens and tokens_sha256 equal those of
// find_dataset(ph.dataset). Fatal otherwise (another dataset's file would silently serve other tokens).
void check_token_file(const TokenFile& f, const Phase& ph);

// ------------------------------------------------------------------------------------------------ loader
// data.hpp BatchLoader has no pimpl member, so its state lives in a table keyed by the loader's address (one entry
// per started loader; stop() erases it). Options must be set before start().
//
// start() validates the plan (validate_plan with nranks), checks that every TokenFile holds its phase's dataset and,
// with pinned slots, that nranks CUDA devices exist; a failure frees everything start() allocated and keeps the
// pending options. stop() may run while workers are blocked in acquire() (an abnormal shutdown): it wakes them, joins
// the reader and stager threads, waits until no caller is inside acquire / release / copy_rank_rows, and only then
// frees the slots. Once stop() began, every one of these calls throws "loader stopped" before it touches a slot; after
// stop() returned they throw "not started". A slot reference must not be read after stop() returns. stop() is
// idempotent and may be called from several threads.
enum class ReadMode : uint8_t { Pread = 0, Mmap = 1 };
struct LoaderOptions {
  // Mmap: reader threads copy windows from a loader-private mapping of each token file with madvise(MADV_RANDOM); a
  // SIGBUS of that copy (storage error, truncated file) is caught and reaches acquire() as an error, like a pread error.
  ReadMode read = ReadMode::Pread;
  int hash_every = 1000;     // log H(B, k) at k = 0 and every k % hash_every == 0 (the eval steps); 0 = only k = 0
  bool host_only = false;    // no CUDA events / pinned memory (tests and benches without a GPU)
  int pinned_slots = 3;
};
void set_loader_options(BatchLoader& l, const LoaderOptions& o);
struct LoaderStats {
  int64_t steps_staged = 0, windows_read = 0, bytes_read = 0;
  double wait_seconds = 0;           // time acquire() blocked: throughput/loading_time
  double read_seconds = 0;           // sum over reader threads of time inside window reads
  std::vector<std::pair<int64_t, std::string>> hashes;   // (k, H(B, k)) logged by the stager
};
LoaderStats loader_stats(const BatchLoader& l);
// Worker r's half of the H2D handoff: copies its rows [r*B/n, (r+1)*B/n) of the slot to `dst` (int32 [B/n, S] on the
// worker's device) on `stream` and records slot.copied[r]; the caller then calls release(slot, r). B and S are those of
// the phase of slot.step.
void copy_rank_rows(const BatchLoader& l, TokenSlot& slot, int rank, int32_t* dst, cudaStream_t stream);
std::pair<int, int> slot_shape(const BatchLoader& l, int64_t step);   // (B, S) of the phase of `step`

}  // namespace nm::data
