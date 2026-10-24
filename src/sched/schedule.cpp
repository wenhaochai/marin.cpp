// Learning-rate and bias-correction tables (DESIGN.md §4.4, optim.md §6): the fixture loader of record and the host
// formula (a port of the arithmetic XLA:GPU runs for the levanter schedules, sched.hpp).
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "config/text.hpp"
#include "marincpp/core.hpp"
#include "marincpp/optim.hpp"
#include "sched/sched.hpp"

static_assert(std::endian::native == std::endian::little, "schedule fixtures are little-endian float32");

namespace nm {
namespace sched {
namespace {

[[noreturn]] void sched_error(const std::string& msg) { fatal(NM_WHERE, "schedule: " + msg); }

// levanter _convert_frac_or_steps (LV/optim/config.py:441-448) for a value already known to be valid (finite, >= 0,
// whole when > 1): x <= 1 is the fraction int(x * n), else int(x) steps. Both callers clip the result to at most n,
// so a step count >= n returns n; this also keeps x >= 2^63 away from the float -> int64 conversion.
int64_t frac_or_steps(double x, int64_t n) {
  if (x <= 1.0) return static_cast<int64_t>(x * static_cast<double>(n));
  return x >= static_cast<double>(n) ? n : static_cast<int64_t>(x);
}

// One segment of optax.join_schedules evaluated at the count since its boundary.
float warmup_at(int64_t c0, int64_t w, double peak) {
  // optax.linear_schedule(0.0, peak, w) (polynomial power 1): (init - end) * (1 - c / w) + end, init - end folded in
  // f64 (optim.md §5.3), c / w rewritten by XLA as c * f32(1 / w).
  const float c = static_cast<float>(std::clamp<int64_t>(c0, 0, w));
  const float frac = 1.0f - c * (1.0f / static_cast<float>(w));
  return frac * static_cast<float>(0.0 - peak) + static_cast<float>(peak);
}

float linear_decay_at(int64_t c0, int64_t td, double peak, double min_lr) {
  // optax.linear_schedule(peak, min_lr, td), min_lr = peak * ratio in f64 (config.py:299).
  const float c = static_cast<float>(std::clamp<int64_t>(c0, 0, td));
  const float frac = 1.0f - c * (1.0f / static_cast<float>(td));
  return frac * static_cast<float>(peak - min_lr) + static_cast<float>(min_lr);
}

float cosine_decay_at(int64_t c0, int64_t td, double peak, double alpha) {
  // optax.cosine_decay_schedule(peak, td, alpha, 1.0): count = min(c, float(td)); cd = 0.5 * (1 + cos(pi * count /
  // td)); peak * ((1 - alpha) * cd**1.0 + alpha). XLA folds (pi * count) / td into count * f32(f32(pi) * f32(1/td)),
  // drops **1.0, and folds 0.5 into the neighbouring constant (exact: a power of two).
  const float count = std::min(static_cast<float>(c0), static_cast<float>(td));
  const float k = static_cast<float>(3.141592653589793) * (1.0f / static_cast<float>(td));
  const float cd = (xla_cosf(count * k) + 1.0f) * 0.5f;
  const float decayed = static_cast<float>(1.0 - alpha) * cd + static_cast<float>(alpha);
  return static_cast<float>(peak) * decayed;
}

std::map<std::string, TomlEntry> toml_map(const std::string& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) sched_error("fixture file not found: " + path);
  std::map<std::string, TomlEntry> m;
  for (TomlEntry& e : parse_toml_file(path)) m.emplace(e.key, std::move(e));
  return m;
}

std::string read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) sched_error("cannot read " + path);
  std::ostringstream o;
  o << f.rdbuf();
  return o.str();
}

}  // namespace

Shape shape(const Config& c) {
  Shape s;
  s.n = c.optim_num_steps;
  if (s.n < 1) sched_error("optim.num_steps must be >= 1");
  if (c.optim_warmup == 1)
    sched_error("optim.warmup = 1 is ambiguous: levanter reads it as the fraction 1.0 (warmup over all N steps); "
                "write 0 or a step count >= 2");
  if (c.optim_warmup < 0) sched_error("optim.warmup must be >= 0");
  s.warmup = std::min<int64_t>(frac_or_steps(c.optim_warmup, s.n), s.n);
  const int64_t max_decay = std::max<int64_t>(s.n - s.warmup, 0);
  int64_t requested = max_decay;
  if (c.optim_decay != -1.0) {
    if (!(c.optim_decay >= 0.0) || !std::isfinite(c.optim_decay))   // levanter refuses inf and nan too
      sched_error("optim.decay must be -1 (None), a fraction in [0, 1] or a finite whole number of steps");
    if (c.optim_decay > 1.0 && c.optim_decay != std::floor(c.optim_decay))
      sched_error("optim.decay > 1 must be a whole number of steps (levanter _convert_frac_or_steps)");
    requested = frac_or_steps(c.optim_decay, s.n);
  }
  s.decay = std::min(std::max<int64_t>(requested, 0), max_decay);
  s.stable = s.n - s.warmup - s.decay;
  if (c.optim_schedule == "linear")
    s.kind = DecayKind::Linear;
  else if (c.optim_schedule == "cosine")
    s.kind = DecayKind::Cosine;
  else
    sched_error("optim.schedule '" + c.optim_schedule + "' (linear, cosine)");
  s.min_lr_ratio = c.optim_min_lr_ratio;
  return s;
}

void lr_table(const Shape& s, double peak, float* out) {
  const double min_lr = peak * s.min_lr_ratio;
  const float constant = static_cast<float>(peak);
  const int64_t b1 = s.warmup, b2 = s.warmup + s.stable;   // join_schedules boundaries
  for (int64_t k = 0; k < s.n; ++k) {
    float v;
    if (s.warmup > 0 && k < b1) {
      v = warmup_at(k, s.warmup, peak);
    } else if (s.stable > 0 && k < b2) {
      v = constant;
    } else if (s.decay > 0) {
      v = s.kind == DecayKind::Linear ? linear_decay_at(k - b2, s.decay, peak, min_lr)
                                      : cosine_decay_at(k - b2, s.decay, peak, s.min_lr_ratio);
    } else {
      v = constant;                                          // levanter: decay 0 -> constant_schedule(peak)
    }
    out[k] = v;
  }
}

void bc_table(double beta, const char* knob, int64_t n, float* out) {
  const float b = static_cast<float>(beta);
  if (!(b >= 0.0f && b < 1.0f))
    sched_error(std::string(knob) + " = " + to_text(beta) + ": f32(beta) must lie in [0, 1) (a beta that rounds to 1 " +
                "makes 1 - beta^t = 0, a division by zero in Adam)");
  if (n >= kMaxPowExponent)
    sched_error("optim.num_steps = " + std::to_string(n) + ": the bias-correction port covers t < 2^24 only " +
                "(use optim.schedule_fixture)");
  for (int64_t t = 1; t <= n; ++t) out[t - 1] = 1.0f - xla_powf(b, static_cast<float>(t));
}

std::string dir_content_hash(const std::string& dir) {
  Config probe;                             // every other file knob is empty: resolve() hashes this one only
  probe.optim_schedule_fixture = dir;
  resolve(probe, nullptr, nullptr, nullptr);
  const std::string& v = probe.optim_schedule_fixture;
  const size_t at = v.rfind("#sha256=");
  if (at == std::string::npos) sched_error("config resolution recorded no content hash for " + dir);
  return v.substr(at + 8);
}

ScheduleTables read_fixture(const Config& c, const std::string& dir) {
  const std::string meta_path = (std::filesystem::path(dir) / "schedule.toml").string();
  const auto m = toml_map(meta_path);
  auto get = [&](const std::string& key) -> const TomlEntry& {
    auto it = m.find(key);
    if (it == m.end()) sched_error(meta_path + ": missing key " + key);
    return it->second;
  };
  auto number = [&](const std::string& key) {
    const TomlEntry& e = get(key);
    double v = 0;
    if ((e.kind != TomlEntry::Int && e.kind != TomlEntry::Float) || !from_text(e.text, v))
      sched_error(meta_path + ":" + std::to_string(e.line) + ": " + key + " must be a number");
    return v;
  };
  if (number("format") != kFixtureFormat)
    sched_error(meta_path + ": format " + get("format").text + ", this binary reads " + std::to_string(kFixtureFormat));

  std::vector<std::string> bad;
  auto same = [&](const std::string& key, double want, const std::string& knob) {
    const double got = number("levanter." + key);
    if (got != want) bad.push_back(key + " = " + to_text(got) + " but " + knob + " = " + to_text(want));
  };
  same("num_train_steps", c.optim_num_steps, "optim.num_steps");
  same("learning_rate", c.optim_lr, "optim.lr");
  same("adam_lr", c.optim_adam_lr, "optim.adam_lr");
  same("warmup", c.optim_warmup, "optim.warmup");
  same("decay", c.optim_decay, "optim.decay");
  same("min_lr_ratio", c.optim_min_lr_ratio, "optim.min_lr_ratio");
  same("beta1", c.optim_beta1, "optim.beta1");
  same("beta2", c.optim_beta2, "optim.beta2");
  const TomlEntry& kind = get("levanter.lr_schedule");
  if (kind.kind != TomlEntry::String || kind.text != c.optim_schedule)
    bad.push_back("lr_schedule = " + kind.text + " but optim.schedule = " + c.optim_schedule);
  if (!bad.empty()) {
    std::string msg = "fixture " + dir + " was made for another schedule:";
    for (auto& b : bad) msg += "\n  " + b;
    sched_error(msg);
  }

  const int64_t n = c.optim_num_steps;
  ScheduleTables t;
  std::vector<float>* dst[4] = {&t.lr, &t.adam_lr, &t.bc1, &t.bc2};
  for (int i = 0; i < 4; ++i) {
    const std::string name = kTables[i];
    const TomlEntry& file = get("files." + name);
    if (file.kind != TomlEntry::String || file.text.empty() || file.text.find('/') != std::string::npos ||
        file.text == "." || file.text == "..")
      sched_error(meta_path + ":" + std::to_string(file.line) + ": files." + name + " must be a file name in " + dir);
    const std::string path = (std::filesystem::path(dir) / file.text).string();
    const std::string raw = read_bytes(path);
    if (raw.size() != static_cast<size_t>(n) * sizeof(float))
      sched_error(path + ": " + std::to_string(raw.size()) + " bytes, want " + std::to_string(n) + " float32");
    const std::string sum = sha256_hex(raw.data(), raw.size());
    if (sum != get("sha256." + name).text) sched_error(path + ": sha256 " + sum + " differs from " + meta_path);
    dst[i]->resize(n);
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(dst[i]->data()));
    for (float v : *dst[i])
      if (!std::isfinite(v)) sched_error(path + ": non-finite value");
  }
  t.source = "fixture:" + dir_content_hash(dir);
  return t;
}

TableDiffs compare_tables(const ScheduleTables& a, const ScheduleTables& b) {
  TableDiffs d;
  d.t[0] = compare(a.lr, b.lr);
  d.t[1] = compare(a.adam_lr, b.adam_lr);
  d.t[2] = compare(a.bc1, b.bc1);
  d.t[3] = compare(a.bc2, b.bc2);
  return d;
}

}  // namespace sched

ScheduleTables build_schedules(const Config& c) {
  const sched::Shape s = sched::shape(c);
  ScheduleTables t;
  t.bc1.resize(s.n);                        // the bias corrections first: their refusals come before the lr loops
  t.bc2.resize(s.n);
  sched::bc_table(c.optim_beta1, "optim.beta1", s.n, t.bc1.data());
  sched::bc_table(c.optim_beta2, "optim.beta2", s.n, t.bc2.data());
  t.lr.resize(s.n);
  t.adam_lr.resize(s.n);
  sched::lr_table(s, c.optim_lr, t.lr.data());
  sched::lr_table(s, c.optim_adam_lr, t.adam_lr.data());
  t.source = "formula";
  return t;
}

ScheduleTables load_schedules(const Config& c) {
  if (c.optim_schedule_fixture.empty()) {
    if (c.run_parity) sched::sched_error("run.parity needs optim.schedule_fixture (DESIGN.md §4.4)");
    return build_schedules(c);
  }
  std::string dir = c.optim_schedule_fixture, recorded;
  const size_t at = dir.find("#sha256=");
  if (at != std::string::npos) {
    recorded = dir.substr(at + 8);
    dir = dir.substr(0, at);
  }
  ScheduleTables t = sched::read_fixture(c, dir);
  if (!recorded.empty() && t.source != "fixture:" + recorded)
    sched::sched_error("content of " + dir + " changed: " + t.source.substr(8) + ", the config records " + recorded);
  // The cross-check never decides whether the fixture loads: a config the formula refuses (optim.warmup = 1,
  // optim.num_steps >= 2^24, a beta that rounds to 1) keeps the fixture and reports the check as unavailable.
  ScheduleTables f;
  try {
    f = build_schedules(c);
  } catch (const Fatal&) {
    t.n_ulp_diffs = t.max_ulp_diff = -1;
    return t;
  }
  const sched::TableDiffs d = sched::compare_tables(t, f);
  int64_t n = 0, mx = 0;
  for (const sched::UlpStats& s : d.t) {
    n += s.n_diff;
    mx = std::max(mx, s.max_ulp);
  }
  t.n_ulp_diffs = static_cast<int>(std::min<int64_t>(n, INT_MAX));
  t.max_ulp_diff = static_cast<int>(std::min<int64_t>(mx, INT_MAX));
  return t;
}

}  // namespace nm
