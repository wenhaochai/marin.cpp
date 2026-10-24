// Data plans: the order of a run as a pure function of (plan, step); phases, passes, forks, queries. DESIGN.md §5.5.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#include "config/text.hpp"
#include "data/internal.hpp"
#include "marincpp/config.hpp"
#include "marincpp/data.hpp"

namespace nm {
namespace data {
namespace {

const char* kind_name(OrderKind k) {
  switch (k) {
    case OrderKind::Levanter: return "levanter";
    case OrderKind::Repeat: return "repeat";
    case OrderKind::Explicit: return "explicit";
  }
  return "?";
}

uint64_t n_seqs_of(const Phase& ph) {
  const DatasetMeta& m = find_dataset(ph.dataset);
  return m.n_tokens / static_cast<uint64_t>(ph.seq_len);
}

uint64_t local_index(const Phase& ph, int64_t step, int row) {
  return static_cast<uint64_t>(step - ph.step_begin) * static_cast<uint64_t>(ph.global_batch) +
         static_cast<uint64_t>(row);
}

uint64_t phase_rows(const Phase& ph) {
  return static_cast<uint64_t>(ph.step_end - ph.step_begin) * static_cast<uint64_t>(ph.global_batch);
}

// Two metas name the same windows: same tokens (content), or the same name for metas without a content hash (tests).
bool same_dataset(const DatasetMeta& a, const DatasetMeta& b) {
  if (!a.tokens_sha256.empty() && !b.tokens_sha256.empty())
    return a.tokens_sha256 == b.tokens_sha256 && a.n_tokens == b.n_tokens;
  return a.name == b.name;
}

// ---------------------------------------------------------------------------- explicit order files
// Keyed by path (the reference without its pin); the content hash is computed once, when the file is mapped, and
// checked against every pinned reference.
struct ExplicitFile { const uint32_t* ids = nullptr; uint64_t count = 0; std::string sha256; };
std::mutex g_explicit_mu;
std::map<std::string, ExplicitFile>& explicit_files() {
  static std::map<std::string, ExplicitFile> m;
  return m;
}

const ExplicitFile& explicit_file(const std::string& ref) {
  const std::string path = strip_hash(ref), want = pin_of(ref);
  std::lock_guard<std::mutex> g(g_explicit_mu);
  auto it = explicit_files().find(path);
  if (it == explicit_files().end()) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fatal(NM_WHERE, "cannot open explicit order file " + path + ": " + std::strerror(errno));
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      const int e = errno;
      ::close(fd);
      fatal(NM_WHERE, "fstat of explicit order file " + path + " failed: " + std::strerror(e));
    }
    if (!(st.st_size > 0 && st.st_size % 4 == 0)) {
      ::close(fd);
      fatal(NM_WHERE, path + ": an explicit order file is a non-empty uint32 array");
    }
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    const int e = errno;
    ::close(fd);
    if (p == MAP_FAILED) fatal(NM_WHERE, "mmap of " + path + " failed: " + std::strerror(e));
    ExplicitFile f{static_cast<const uint32_t*>(p), static_cast<uint64_t>(st.st_size) / 4,
                   sha256_hex(p, static_cast<size_t>(st.st_size))};
    it = explicit_files().emplace(path, std::move(f)).first;
  }
  if (!want.empty() && want != it->second.sha256)
    fatal(NM_WHERE, "explicit order file " + path + " has sha256 " + it->second.sha256 + ", the plan pins " + want);
  return it->second;
}

// ---------------------------------------------------------------------------- reshuffled passes
// permutation(fold_in(prng_key(seed), q), subset) of a reshuffled Repeat pass: computed once per process (the first
// caller computes it, concurrent callers of the same pass wait for it) and shared by every thread; the newest
// kKeepPasses passes stay cached, holders keep evicted ones alive through their shared_ptr.
struct PassPerm {
  std::once_flag once;
  std::vector<uint32_t> perm;
  uint64_t serial = 0;
};
constexpr size_t kKeepPasses = 4;
std::mutex g_perm_mu;
std::map<std::tuple<uint32_t, uint64_t, uint64_t>, std::shared_ptr<PassPerm>>& pass_perms() {
  static std::map<std::tuple<uint32_t, uint64_t, uint64_t>, std::shared_ptr<PassPerm>> m;
  return m;
}

std::shared_ptr<const std::vector<uint32_t>> pass_permutation(uint32_t seed, uint64_t subset, uint64_t q) {
  NM_REQUIRE(q <= 0xffffffffull && subset <= 0xffffffffull, "pass or subset exceeds 2^32");
  std::shared_ptr<PassPerm> e;
  {
    static uint64_t serial = 0;
    std::lock_guard<std::mutex> g(g_perm_mu);
    auto& tab = pass_perms();
    auto& slot = tab[{seed, subset, q}];
    if (!slot) {
      slot = std::make_shared<PassPerm>();
      slot->serial = ++serial;
      if (tab.size() > kKeepPasses) {   // evict the oldest pass
        auto oldest = std::min_element(tab.begin(), tab.end(),
                                       [](const auto& a, const auto& b) { return a.second->serial < b.second->serial; });
        tab.erase(oldest);
      }
    }
    e = tab.at({seed, subset, q});
  }
  std::call_once(e->once, [&] {
    e->perm.resize(subset);
    permutation(fold_in(prng_key(seed), static_cast<uint32_t>(q)), static_cast<uint32_t>(subset), e->perm.data());
  });
  return std::shared_ptr<const std::vector<uint32_t>>(e, &e->perm);
}

// sigma_N(perm_q[i]) for Repeat-with-reshuffle passes q >= 1; each thread keeps its current pass and LCG.
uint64_t reshuffled(const Phase& ph, uint64_t n, uint64_t q, uint64_t i) {
  thread_local uint32_t c_seed = ~0u, c_lcg_seed = ~0u;
  thread_local uint64_t c_subset = 0, c_q = ~0ull, c_n = 0;
  thread_local std::shared_ptr<const std::vector<uint32_t>> perm;
  thread_local Lcg lcg;
  if (!perm || c_seed != ph.seed || c_subset != ph.subset || c_q != q) {
    perm = pass_permutation(ph.seed, ph.subset, q);
    c_seed = ph.seed;
    c_subset = ph.subset;
    c_q = q;
  }
  if (c_n != n || c_lcg_seed != ph.seed) {
    lcg = lcg_for(ph.seed, n);
    c_n = n;
    c_lcg_seed = ph.seed;
  }
  return static_cast<uint64_t>((static_cast<unsigned __int128>(lcg.a) * (*perm)[i] + lcg.b) % n);
}

uint64_t seq_in_phase(const Phase& ph, uint64_t n, uint64_t m) {
  switch (ph.kind) {
    case OrderKind::Levanter:
      return levanter_seq(ph.seed, n, kMixBlock, n, ph.index_base + m);
    case OrderKind::Repeat: {
      if (ph.reshuffle && m >= ph.subset) return reshuffled(ph, n, m / ph.subset, m % ph.subset);
      return levanter_seq(ph.seed, n, kMixBlock, ph.subset, ph.index_base + m);
    }
    case OrderKind::Explicit: {
      const uint64_t s = explicit_id(ph.explicit_file, ph.index_base + m);
      NM_REQUIRE(s < n, "explicit sequence id " + std::to_string(s) + " >= N");
      return s;
    }
  }
  fatal(NM_WHERE, "unknown order kind");
}

std::string fmt_double(double v) {
  char b[40];
  std::snprintf(b, sizeof(b), "%.17g", v);
  return b;
}

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

}  // namespace

uint32_t explicit_id(const std::string& ref, uint64_t i) {
  const ExplicitFile& f = explicit_file(ref);
  NM_REQUIRE(i < f.count, strip_hash(ref) + ": index " + std::to_string(i) + " past its " + std::to_string(f.count) +
                              " ids");
  return f.ids[i];
}

size_t phase_index(const DataPlan& p, int64_t step) {
  for (size_t i = 0; i < p.phases.size(); ++i)
    if (step >= p.phases[i].step_begin && step < p.phases[i].step_end) return i;
  fatal(NM_WHERE, "step " + std::to_string(step) + " is outside the data plan");
}

void step_seqs(const DataPlan& p, int64_t step, uint64_t* out) {
  const Phase& ph = p.phases[phase_index(p, step)];
  const uint64_t n = n_seqs_of(ph);
  for (int r = 0; r < ph.global_batch; ++r) out[r] = seq_in_phase(ph, n, local_index(ph, step, r));
}

void validate_plan(const DataPlan& p, int ranks) {
  std::vector<std::string> err;
  if (p.phases.empty()) err.push_back("a data plan needs at least one phase");
  int64_t expect = 0;
  for (size_t i = 0; i < p.phases.size(); ++i) {
    const Phase& ph = p.phases[i];
    const std::string at = "phase " + std::to_string(i) + ": ";
    if (ph.step_begin != expect)
      err.push_back(at + "step_begin " + std::to_string(ph.step_begin) + " != " + std::to_string(expect) +
                    " (phases are contiguous from step 0)");
    if (ph.step_end <= ph.step_begin) err.push_back(at + "step_end must exceed step_begin");
    expect = ph.step_end;
    if (ph.global_batch < 1) err.push_back(at + "global_batch must be positive");
    else if (ranks > 0 && ph.global_batch % ranks) err.push_back(at + "global_batch % exec.gpus != 0");
    if (ph.seq_len < 1) err.push_back(at + "seq_len must be positive");
    if (ph.kind == OrderKind::Repeat && ph.reshuffle && ph.index_base != 0)
      err.push_back(at + "a reshuffled repeat starts its passes at mixture index 0: index_base must be 0, not " +
                    std::to_string(ph.index_base));
    if (!err.empty()) continue;
    uint64_t n = 0;
    try {
      n = n_seqs_of(ph);
    } catch (const Fatal& e) {
      err.push_back(at + e.what());
      continue;
    }
    if (n == 0) {
      err.push_back(at + "dataset holds no window of seq_len " + std::to_string(ph.seq_len));
      continue;
    }
    try {
      if (ph.kind != OrderKind::Explicit) (void)lcg_for(ph.seed, n);
    } catch (const Fatal& e) {
      err.push_back(at + e.what());
    }
    if (ph.kind == OrderKind::Repeat && (ph.subset == 0 || ph.subset > n))
      err.push_back(at + "repeat needs 0 < subset <= N (" + std::to_string(n) + ")");
    if (ph.kind == OrderKind::Explicit) {
      try {
        const ExplicitFile& f = explicit_file(ph.explicit_file);   // checks a "#sha256=" pin
        if (ph.index_base + phase_rows(ph) > f.count)
          err.push_back(at + "explicit file holds " + std::to_string(f.count) + " ids, the phase reads " +
                        std::to_string(ph.index_base + phase_rows(ph)));
      } catch (const Fatal& e) {
        err.push_back(at + e.what());
      }
    }
  }
  if (!err.empty()) {
    std::string msg = "invalid data plan:";
    for (auto& e : err) msg += "\n  " + e;
    fatal(NM_WHERE, msg);
  }
}

DataPlan parse_data_plan(const std::string& ref, const Config& c) {
  const size_t hp = ref.find("#sha256=");
  const std::string path = ref.substr(0, hp);
  if (hp != std::string::npos) {
    const std::string got = sha256_file_hex(path), want = ref.substr(hp + 8);
    if (got != want) fatal(NM_WHERE, "data plan " + path + " has sha256 " + got + ", the config pins " + want);
  }
  const auto entries = parse_toml_file(path);
  std::map<int, std::map<std::string, const TomlEntry*>> by_phase;
  for (const TomlEntry& e : entries) {
    int idx = -1;
    char key[64] = {0};
    int used = 0;
    if (std::sscanf(e.key.c_str(), "phase.%d.%63s%n", &idx, key, &used) != 2 || idx < 0 ||
        used != static_cast<int>(e.key.size()))
      fatal(NM_WHERE, path + ":" + std::to_string(e.line) + ": unknown key '" + e.key + "' (expected phase.<i>.<key>)");
    if (!by_phase[idx].emplace(key, &e).second)
      fatal(NM_WHERE, path + ":" + std::to_string(e.line) + ": repeated key " + e.key);
  }
  DataPlan p;
  const std::string base_dir = std::filesystem::absolute(path).parent_path().string();
  int expect_idx = 0;
  for (auto& [idx, kv] : by_phase) {
    if (idx != expect_idx++) fatal(NM_WHERE, path + ": phases must be numbered 0, 1, 2, ... without gaps");
    Phase ph;
    ph.step_begin = p.phases.empty() ? 0 : p.phases.back().step_end;
    ph.dataset = c.data_dataset;   // with the resolved "#sha256=" pin: the default dataset stays content-pinned
    ph.global_batch = c.data_global_batch;
    ph.seq_len = c.model_seq_len;
    ph.seed = static_cast<uint32_t>(c.data_data_seed);
    bool have_end = false, have_base = false;
    for (auto& [k, e] : kv) {
      const std::string where = path + ":" + std::to_string(e->line) + ": " + e->key;
      auto need = [&](TomlEntry::Kind kd) {
        if (e->kind != kd) fatal(NM_WHERE, where + " has the wrong type");
      };
      auto as_i64 = [&]() {
        need(TomlEntry::Int);
        int64_t v = 0;
        if (!from_text(e->text, v) || v < 0) fatal(NM_WHERE, where + " must be a non-negative integer");
        return v;
      };
      if (k == "step_begin") ph.step_begin = as_i64();
      else if (k == "step_end") { ph.step_end = as_i64(); have_end = true; }
      else if (k == "dataset") { need(TomlEntry::String); ph.dataset = e->text; }
      else if (k == "global_batch") ph.global_batch = static_cast<int>(as_i64());
      else if (k == "seq_len") ph.seq_len = static_cast<int>(as_i64());
      else if (k == "seed") ph.seed = static_cast<uint32_t>(as_i64());
      else if (k == "index_base") { ph.index_base = static_cast<uint64_t>(as_i64()); have_base = true; }
      else if (k == "subset") ph.subset = static_cast<uint64_t>(as_i64());
      else if (k == "passes") {
        if (e->kind == TomlEntry::Int) ph.passes = static_cast<double>(as_i64());
        else {
          need(TomlEntry::Float);
          if (!from_text(e->text, ph.passes) || !(ph.passes > 0)) fatal(NM_WHERE, where + " must be positive");
        }
      } else if (k == "reshuffle") {
        need(TomlEntry::Bool);
        ph.reshuffle = e->text == "true";
      } else if (k == "kind") {
        need(TomlEntry::String);
        if (e->text == "levanter") ph.kind = OrderKind::Levanter;
        else if (e->text == "repeat") ph.kind = OrderKind::Repeat;
        else if (e->text == "explicit") ph.kind = OrderKind::Explicit;
        else fatal(NM_WHERE, where + " must be levanter, repeat or explicit");
      } else if (k == "explicit_file") {   // relative to the plan file; a "#sha256=" pin is kept and checked on load
        need(TomlEntry::String);
        const std::filesystem::path f(strip_hash(e->text));
        const std::string pin = pin_of(e->text);
        ph.explicit_file = (f.is_absolute() ? f.string() : (std::filesystem::path(base_dir) / f).string()) +
                           (pin.empty() ? "" : "#sha256=" + pin);
      } else {
        fatal(NM_WHERE, where + ": unknown phase key");
      }
    }
    if (!have_end) {
      if (ph.kind == OrderKind::Repeat && ph.passes > 0 && ph.subset > 0)
        ph.step_end = ph.step_begin + static_cast<int64_t>(std::ceil(ph.passes * static_cast<double>(ph.subset) /
                                                                      ph.global_batch));
      else
        ph.step_end = c.optim_num_steps;
    }
    if (!have_base && !p.phases.empty()) {
      const Phase& pr = p.phases.back();
      if (pr.kind == OrderKind::Levanter && ph.kind == OrderKind::Levanter && pr.seed == ph.seed &&
          strip_hash(pr.dataset) == strip_hash(ph.dataset) && pr.seq_len == ph.seq_len)
        ph.index_base = pr.index_base + phase_rows(pr);
    }
    if (ph.kind == OrderKind::Explicit && ph.explicit_file.empty())
      fatal(NM_WHERE, path + ": phase " + std::to_string(idx) + " is explicit but has no explicit_file");
    p.phases.push_back(ph);
  }
  return p;
}

DataPlan fork_plan(const DataPlan& parent, int64_t k, const std::vector<Phase>& appended) {
  NM_REQUIRE(!parent.phases.empty(), "fork of an empty plan");
  NM_REQUIRE(k > 0 && k <= parent.phases.back().step_end, "fork step outside the parent's plan");
  DataPlan p;
  for (const Phase& ph : parent.phases) {
    if (ph.step_begin >= k) break;
    p.phases.push_back(ph);
    p.phases.back().step_end = std::min(ph.step_end, k);
  }
  if (!appended.empty())
    NM_REQUIRE(appended.front().step_begin == k, "the first appended phase must start at the fork step");
  for (const Phase& ph : appended) p.phases.push_back(ph);
  return p;
}

std::vector<uint32_t> window_reads(const DataPlan& p, const std::string& dataset, int seq_len, int64_t up_to_step) {
  const DatasetMeta& m = find_dataset(dataset);
  std::vector<uint32_t> reads(m.n_tokens / static_cast<uint64_t>(seq_len), 0);
  for (const Phase& ph : p.phases) {
    if (!same_dataset(find_dataset(ph.dataset), m) || ph.seq_len != seq_len) continue;
    const uint64_t n = reads.size();
    const int64_t e = std::min(ph.step_end, up_to_step);
    for (int64_t s = ph.step_begin; s < e; ++s)
      for (int r = 0; r < ph.global_batch; ++r) ++reads[seq_in_phase(ph, n, local_index(ph, s, r))];
  }
  return reads;
}

}  // namespace data

// ------------------------------------------------------------------------------------------------ DataPlan
const Phase& DataPlan::phase_at(int64_t step) const { return phases[data::phase_index(*this, step)]; }

uint64_t DataPlan::seq_of(int64_t step, int row) const {
  const Phase& ph = phase_at(step);
  NM_REQUIRE(row >= 0 && row < ph.global_batch, "row outside the phase's global batch");
  return data::seq_in_phase(ph, data::n_seqs_of(ph), data::local_index(ph, step, row));
}

// Sum over phases of: the largest number of reads, inside that phase and before up_to_step, of any window that holds a
// token of the document (in that phase's dataset; a resegmented dataset keeps the document index, docs.u64).
uint32_t DataPlan::times_seen(uint64_t doc, int64_t up_to_step) const {
  uint32_t total = 0;
  for (const Phase& ph : phases) {
    const int64_t e = std::min(ph.step_end, up_to_step);
    if (e <= ph.step_begin) continue;
    const TokenFile& f = data::shared_token_file(ph.dataset);
    const DatasetMeta& m = data::find_dataset(ph.dataset);
    NM_REQUIRE(doc < m.n_docs, "document index past the dataset");
    const uint64_t S = static_cast<uint64_t>(ph.seq_len), n = m.n_tokens / S;
    // token range [lo_tok, hi_tok) of the document: TokenFile exposes docs.u64 only through doc_of_token, so the two
    // ends are found by binary search over token indices (first token of doc, first token of doc + 1)
    auto first_token_of = [&](uint64_t d) {
      uint64_t a = 0, b = m.n_tokens;
      while (a < b) {
        const uint64_t mid = a + (b - a) / 2;
        if (f.doc_of_token(mid) < d) a = mid + 1;
        else b = mid;
      }
      return a;
    };
    const uint64_t lo_tok = first_token_of(doc), hi_tok = first_token_of(doc + 1);
    if (lo_tok >= n * S) continue;   // the document lies in the dropped tail
    NM_REQUIRE(hi_tok > lo_tok, "empty document");
    const uint64_t w0 = lo_tok / S, w1 = std::min((hi_tok - 1) / S, n - 1);
    std::vector<uint32_t> cnt(w1 - w0 + 1, 0);
    for (int64_t s = ph.step_begin; s < e; ++s)
      for (int r = 0; r < ph.global_batch; ++r) {
        const uint64_t q = data::seq_in_phase(ph, n, data::local_index(ph, s, r));
        if (q >= w0 && q <= w1) ++cnt[q - w0];
      }
    total += *std::max_element(cnt.begin(), cnt.end());
  }
  return total;
}

// The step that holds the first index of each pass q >= 1 of every phase (the fork points that
// run.keep_pass_boundaries keeps). Levanter and plain Repeat phases: the step of first_index_of_pass, i.e. of the
// START of the mixture block holding pass q's first index; with wrap % 2048 != 0 that step and the following ones up
// to the block end mix passes q-1 and q (Repeat subset 4767 x 128: pass 1 boundary = step 4752). Reshuffled Repeat
// phases: the step of index q * subset (it mixes two passes when subset % B != 0). A checkpoint kept at a boundary
// step is the state before any sequence of pass q was read.
std::vector<int64_t> DataPlan::pass_boundaries() const {
  std::vector<int64_t> out;
  for (const Phase& ph : phases) {
    const uint64_t rows = data::phase_rows(ph), B = static_cast<uint64_t>(ph.global_batch);
    if (ph.kind == OrderKind::Explicit) continue;
    const uint64_t n = data::n_seqs_of(ph);
    if (ph.kind == OrderKind::Repeat && ph.reshuffle) {
      for (uint64_t q = 1; q * ph.subset < rows; ++q) out.push_back(ph.step_begin + static_cast<int64_t>(q * ph.subset / B));
      continue;
    }
    const uint64_t wrap = ph.kind == OrderKind::Repeat ? ph.subset : n;
    for (uint64_t q = 1;; ++q) {
      const uint64_t jf = data::first_index_of_pass(q, wrap, data::kMixBlock);
      if (jf >= ph.index_base + rows) break;
      if (jf >= ph.index_base) out.push_back(ph.step_begin + static_cast<int64_t>((jf - ph.index_base) / B));
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Hashed into the run identity: every input that decides the order, by content. A dataset enters by its meta.json
// sha256 (the pin when the reference carries one, checked; the loaded meta's otherwise), never by name or path; an
// explicit order file by its own sha256. Metas registered without a sha (tests) enter by name.
std::string DataPlan::canonical() const {
  std::string o = "{\"phases\":[";
  for (size_t i = 0; i < phases.size(); ++i) {
    const Phase& ph = phases[i];
    const DatasetMeta& m = data::find_dataset(ph.dataset);
    o += i ? "," : "";
    o += "{\"begin\":" + std::to_string(ph.step_begin) + ",\"end\":" + std::to_string(ph.step_end) +
         ",\"dataset\":" + data::json_str(m.meta_sha256.empty() ? m.name : "sha256:" + m.meta_sha256) +
         ",\"B\":" + std::to_string(ph.global_batch) + ",\"S\":" + std::to_string(ph.seq_len) +
         ",\"kind\":\"" + data::kind_name(ph.kind) + "\",\"seed\":" + std::to_string(ph.seed) +
         ",\"index_base\":" + std::to_string(ph.index_base) + ",\"block\":" + std::to_string(data::kMixBlock);
    if (ph.kind == OrderKind::Repeat)
      o += ",\"subset\":" + std::to_string(ph.subset) + ",\"passes\":" + data::fmt_double(ph.passes) +
           ",\"reshuffle\":" + (ph.reshuffle ? "true" : "false");
    if (ph.kind == OrderKind::Explicit) o += ",\"explicit\":\"sha256:" + data::explicit_file(ph.explicit_file).sha256 + "\"";
    o += "}";
  }
  return o + "]}";
}

DataPlan load_data_plan(const std::string& path_or_empty, const Config& c) {
  if (c.data_mix_block != data::kMixBlock)
    fatal(NM_WHERE, "data.mix_block = " + std::to_string(c.data_mix_block) +
                        ": only 2048 is supported (data.hpp Phase has no mixture-block field yet)");
  DataPlan p;
  if (path_or_empty.empty()) {
    Phase ph;
    ph.step_begin = 0;
    ph.step_end = c.optim_num_steps;
    ph.dataset = c.data_dataset;
    ph.global_batch = c.data_global_batch;
    ph.seq_len = c.model_seq_len;
    ph.kind = OrderKind::Levanter;
    ph.seed = static_cast<uint32_t>(c.data_data_seed);
    p.phases.push_back(ph);
  } else {
    p = data::parse_data_plan(path_or_empty, c);
  }
  data::validate_plan(p, c.exec_gpus);
  if (p.phases.back().step_end != c.optim_num_steps)
    fatal(NM_WHERE, "the data plan ends at step " + std::to_string(p.phases.back().step_end) +
                        ", optim.num_steps is " + std::to_string(c.optim_num_steps));
  return p;
}

}  // namespace nm
