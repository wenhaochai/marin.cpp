// In-process eval: eval sources and groupings, exact per-position loss, levanter's estimator for the logged keys, exact
// bpb beside it. DESIGN.md §8.1.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "marincpp/core.hpp"
#include "marincpp/data.hpp"

namespace nm {

struct DeviceContext;
class Model;

// ------------------------------------------------------------------------------------------------ sources
// Paloma (tools/convert): the floor(tokens/4096) windows of each of the 16 domains, concatenated in the component order
// of eval.md §2.1 (14,529 windows), one tag per window, the levanter byte table (bytes of the TARGET token, U+FFFD
// rule, sha256 d3be7450...), an exact UTF-8 byte table, and per domain the exact source bytes the scored windows cover
// (each tokenizer drops a different tail; cross-tokenizer comparisons use the byte range every compared tokenizer
// covers: eval_exact_shared/*).
struct EvalDomain {
  std::string name;                 // "c4_en"; W&B key eval/paloma/c4_en-marin-tokenizer/{loss,bpb}
  uint32_t first_seq = 0, n_seqs = 0;
  uint64_t scored_tokens = 0;       // n_seqs * 4095
  uint64_t covered_bytes_begin = 0, covered_bytes_end = 0;   // source byte range of the scored windows
  std::string sha256;               // of its windows (int32 LE), data.md §8.1 table
};
enum class EvalSourceKind : uint8_t { Paloma = 0, DataPlanSlice = 1, SequenceList = 2 };
struct EvalSource {
  EvalSourceKind kind = EvalSourceKind::Paloma;
  std::string name;                 // key prefix: "paloma", "train_subset", ...
  DatasetMeta meta;
  TokenFile tokens;
  std::vector<EvalDomain> domains;  // Paloma: 16; other sources: one pseudo-domain or a tag table
  std::vector<uint16_t> tag;        // per sequence
  std::vector<int32_t> bytes_levanter, bytes_exact;  // [V]
  std::vector<uint64_t> seq_ids;    // DataPlanSlice / SequenceList
  int batch = 128;                  // estimator batch: fixed, independent of the GPU count
  int n_batches = 0;                // Paloma: 114; the last holds 65 real rows + 63 zero rows (weight 0, no tag)
};
EvalSource load_paloma(const std::string& dir, int batch);
std::vector<EvalSource> load_extra_sources(const std::string& spec_file, const struct DataPlan& plan);

// Groupings of scored positions with device per-group sums (fixed-order reductions): domain tags (Paloma), target-token
// group tables (e.g. frequency buckets, pieces), position bins, times_seen bins (DataPlan::times_seen of the document).
enum class GroupingKind : uint8_t { Tag = 0, TargetTokenGroup = 1, PositionBin = 2, TimesSeenBin = 3 };
struct Grouping {
  GroupingKind kind = GroupingKind::Tag;
  std::string name;
  std::vector<int32_t> table;       // token -> group, or bin edges
  int n_groups = 0;
};

// ------------------------------------------------------------------------------------------------ estimator
// Per estimator batch b (128 sequences) and tag k: L_k = sum l*w, L = sum over all rows, accumulated in f64 on device
// and rounded to f32 ONCE; W_k, Y_k (bytes_levanter[target]*w) are exact integers. A cross-rank sum follows. Then ONE
// host thread applies levanter's RunningMean updates in batch order (eval.md §4.1), with the contraction of
// `mean + (x - mean) * r` that XLA used (explicit std::fmaf or none, decided from the eval PTX; host code is
// -ffp-contract=off):
//   RunningMean.add(x, n): total' = total + n; mean += (x - mean) * (n / total' if total' else 0)
//   token_avg_loss.add(L / max(W,1), W);  loss_per_tag[k].add(W_k > 0 ? L_k / W_k : 0, W_k)
//   bpb.add(L / max(Y,1) * log2e, W);     bpb_per_tag[k].add(L_k / max(Y_k,1) * log2e, W_k)
// The np.mean / np.average keys use numpy's exact float32 algorithms (tested against numpy on the fixture).
// T3.1 (i): given levanter's per-batch dumps (tools/levref/paloma.py), this stage matches to 0-1 ulp;
// T3.1 (ii): the per-batch sums satisfy |nano - exact| <= |lev - exact| + 1 ulp.
struct RunningMeanF32 { float mean; float total; };
struct BatchSums { float L, W, Y; std::vector<float> L_k, W_k, Y_k; };   // per estimator batch, after rounding
struct EvalAccum {
  RunningMeanF32 token_avg_loss, bpb;
  std::vector<RunningMeanF32> loss_per_tag, bpb_per_tag;
  std::vector<double> exact_loss, exact_bytes, exact_tokens;            // per tag, f64
  std::vector<double> group_loss, group_tokens;                         // extra groupings
};
void running_mean_add(RunningMeanF32& m, float x, float n);              // the exact f32 update, FMA as decided
EvalAccum accumulate(const std::vector<BatchSums>& batches, int n_tags);

// Runs one source on all ranks. Execution packs several estimator batches into one eval microbatch sized to memory
// (eval.micro_batches; forward only, no saved activations) and one graph per batch group; per-batch sums stay per
// 128-row batch. `readouts` lists the readout layers to evaluate (pls: every layer -> eval/L{k}/...). The host syncs
// once at the end.
struct EvalResult { std::vector<std::pair<std::string, double>> keys; double seconds_total = 0, seconds_loading = 0; };
EvalResult run_eval(DeviceContext& d, Model& model, const EvalSource& src, const std::vector<Grouping>& groupings,
                    const std::vector<int>& readouts, bool per_position_dump, const std::string& dump_path);

// The 42 keys of eval.md §4.2 with their exact formulas (eval/loss, eval/paloma/micro_loss, eval/macro_loss,
// eval/paloma/macro_loss, eval/bpb, eval/paloma/bpb, eval/macro_bpb, eval/paloma/macro_bpb, 16 x loss, 16 x bpb,
// eval/loading_time, eval/total_time), plus eval_exact/paloma/<d>/bpb, eval_exact/macro_bpb and eval_exact_shared/*
// (prefixes that no objective_compare.py pattern matches). The init eval uses prefix "eval_init" instead of "eval",
// so eval/ keys exist only at steps {1000k, N-1}.
std::vector<std::pair<std::string, double>> eval_keys(const EvalAccum& a, const EvalSource& src,
                                                      const std::string& prefix);

}  // namespace nm
