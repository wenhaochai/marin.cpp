// Stage S2 tests: schedule shape rules, the host port of the XLA:GPU schedule arithmetic, the fixture loader, and the
// formula against every levanter fixture (the four sizes and the S2 probes). T0.7 itself (fixture vs every logged W&B
// value) is tests/s2/test_t0_7.py.
//   s2_formula   spec values of optim.md §6, levanter shape rules, xla_cosf / xla_powf sanity and edges, bias-correction
//                edges, the folded cosine constant (GPU values), ulp distance, C ABI
//   s2_fixture   loader on synthetic fixtures (checks, errors, a formula refusal keeps the fixture, content hash ==
//                config resolution's)
//   s2_levanter  formula == fixture bitwise at every k, all four tables, all four sizes (skips without fixtures)
//   s2_probe     formula == fixture bitwise for the probe configs of tests/s2/schedule_fixture.py (skips without them)
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>

#include "config/text.hpp"
#include "marincpp/capi.h"
#include "marincpp/config.hpp"
#include "marincpp/core.hpp"
#include "marincpp/optim.hpp"
#include "nmtest.hpp"
#include "sched/sched.hpp"

using namespace nm;
namespace fs = std::filesystem;

namespace {

Config preset(const std::string& size) {
  Config c;
  apply_preset(c, size);
  return c;
}

uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

std::string g7(float x, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*g", digits, static_cast<double>(x));
  return buf;
}

// A scratch directory under the working directory (ctest runs in the build tree), removed at scope exit.
struct TmpDir {
  fs::path p;
  explicit TmpDir(const std::string& tag) : p(fs::current_path() / ("s2_tmp_" + tag + "_" + std::to_string(getpid()))) {
    fs::remove_all(p);
    fs::create_directories(p);
  }
  ~TmpDir() {
    std::error_code ec;
    fs::remove_all(p, ec);
  }
  std::string str() const { return p.string(); }
};

void write_table(const fs::path& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
}

std::string file_sha(const fs::path& p) { return sha256_file_hex(p.string()); }

// The fixture format of tests/s2/schedule_fixture.py, written from given tables.
void write_fixture(const fs::path& dir, const Config& c, const ScheduleTables& t) {
  const std::vector<float>* src[4] = {&t.lr, &t.adam_lr, &t.bc1, &t.bc2};
  std::string toml = "format = 1\nsize = \"" + c.model_size + "\"\n\n[levanter]\n";
  toml += "num_train_steps = " + std::to_string(c.optim_num_steps) + "\n";
  toml += "lr_schedule = \"" + c.optim_schedule + "\"\n";
  toml += "learning_rate = " + to_text(c.optim_lr) + "\nadam_lr = " + to_text(c.optim_adam_lr) + "\n";
  toml += "warmup = " + std::to_string(c.optim_warmup) + "\ndecay = " + to_text(c.optim_decay) + "\n";
  toml += "min_lr_ratio = " + to_text(c.optim_min_lr_ratio) + "\n";
  toml += "beta1 = " + to_text(c.optim_beta1) + "\nbeta2 = " + to_text(c.optim_beta2) + "\n\n[files]\n";
  for (int i = 0; i < 4; ++i) {
    write_table(dir / (std::string(sched::kTables[i]) + ".f32"), *src[i]);
    toml += std::string(sched::kTables[i]) + " = \"" + sched::kTables[i] + ".f32\"\n";
  }
  toml += "\n[sha256]\n";
  for (int i = 0; i < 4; ++i)
    toml += std::string(sched::kTables[i]) + " = \"" + file_sha(dir / (std::string(sched::kTables[i]) + ".f32")) + "\"\n";
  std::ofstream(dir / "schedule.toml") << toml;
}

// Rewrites the sha256 line of one table after the test changed the file.
void refresh_sha(const fs::path& dir, const std::string& table) {
  std::string toml = nmtest::read_file((dir / "schedule.toml").string());
  const std::string key = "\n" + table + " = \"";
  size_t at = toml.rfind(key);
  NM_REQUIRE(at != std::string::npos, "no sha line");
  at += key.size();
  toml.replace(at, 64, file_sha(dir / (table + ".f32")));
  std::ofstream(dir / "schedule.toml") << toml;
}

std::string report(const sched::TableDiffs& d) {
  std::string s;
  for (int i = 0; i < 4; ++i)
    s += std::string(" ") + sched::kTables[i] + ": " + std::to_string(d.t[i].n_diff) + " diffs (max " +
         std::to_string(d.t[i].max_ulp) + " ulp, first k " + std::to_string(d.t[i].first) + ")";
  return s;
}

// Formula vs one fixture directory; the config is made from the fixture's own [levanter] fields.
void check_formula_equals_fixture(const std::string& dir, Config c) {
  std::map<std::string, std::string> v;
  for (const TomlEntry& e : parse_toml_file(dir + "/schedule.toml")) v[e.key] = e.text;
  c.optim_num_steps = std::stoi(v.at("levanter.num_train_steps"));
  c.optim_schedule = v.at("levanter.lr_schedule");
  c.optim_lr = std::stod(v.at("levanter.learning_rate"));
  c.optim_adam_lr = std::stod(v.at("levanter.adam_lr"));
  c.optim_warmup = std::stoi(v.at("levanter.warmup"));
  c.optim_decay = std::stod(v.at("levanter.decay"));
  c.optim_min_lr_ratio = std::stod(v.at("levanter.min_lr_ratio"));
  c.optim_beta1 = std::stod(v.at("levanter.beta1"));
  c.optim_beta2 = std::stod(v.at("levanter.beta2"));
  c.optim_schedule_fixture = dir;
  const ScheduleTables fx = sched::read_fixture(c, dir);
  const ScheduleTables fo = build_schedules(c);
  const sched::TableDiffs d = sched::compare_tables(fx, fo);
  const bool same = d.t[0].n_diff == 0 && d.t[1].n_diff == 0 && d.t[2].n_diff == 0 && d.t[3].n_diff == 0;
  std::printf("    %s: N=%d%s\n", dir.c_str(), c.optim_num_steps, report(d).c_str());
  if (!same) nmtest::fail(__FILE__, __LINE__, dir + ": formula differs from the fixture:" + report(d));
  const ScheduleTables loaded = load_schedules(c);
  CHECK_EQ(loaded.n_ulp_diffs, 0);
  CHECK(loaded.source.rfind("fixture:", 0) == 0);
}

}  // namespace

// ------------------------------------------------------------------------------------------------ s2_formula
NM_TEST(s2_formula, spec_values_130m) {
  const Config c = preset("130m");
  const sched::Shape s = sched::shape(c);
  CHECK_EQ(s.warmup, 0);
  CHECK_EQ(s.stable, 992);
  CHECK_EQ(s.decay, 3967);
  const ScheduleTables t = build_schedules(c);
  CHECK_EQ(t.lr.size(), size_t{4959});
  CHECK_EQ(bits(t.lr[0]), 0x3CA3D70Au);           // peak f32(0.02), constant through k = 991
  CHECK_EQ(bits(t.lr[991]), 0x3CA3D70Au);
  CHECK_EQ(bits(t.adam_lr[0]), 0x3C03126Fu);
  CHECK_EQ(g7(t.lr[4958], 7), std::string("5.041361e-06"));   // optim.md §6.1, as logged
  CHECK_EQ(g7(t.adam_lr[4958], 7), std::string("2.016544e-06"));
  CHECK(t.lr[4958] > 0.0f);
}

NM_TEST(s2_formula, spec_values_cosine_sizes) {
  for (const char* size : {"300m", "520m", "1_2b"}) {
    const Config c = preset(size);
    const sched::Shape s = sched::shape(c);
    CHECK_EQ(s.warmup, 1000);
    CHECK_EQ(s.stable, 0);
    CHECK_EQ(s.decay, c.optim_num_steps - 1000);
    const ScheduleTables t = build_schedules(c);
    CHECK_EQ(bits(t.lr[0]), 0u);                      // eta(0) = 0
    CHECK_EQ(bits(t.lr[1000]), bits(static_cast<float>(c.optim_lr)));
    // eta(1) = f32(-peak) * (1 - f32(1/1000)) + f32(peak) (optim.md §6.2)
    const float frac = 1.0f - 1.0f * (1.0f / 1000.0f);
    CHECK_EQ(bits(t.lr[1]), bits(frac * static_cast<float>(-c.optim_lr) + static_cast<float>(c.optim_lr)));
  }
  const ScheduleTables a = build_schedules(preset("300m"));
  CHECK_EQ(g7(a.lr[2], 8), std::string("2.0000152e-05"));    // optim.md §6.2: logged at k = 2
  for (const char* size : {"300m", "520m"}) {
    const ScheduleTables t = build_schedules(preset(size));
    CHECK_EQ(g7(t.lr.back(), 6), std::string("2.98023e-10"));     // 2^-24 * peak / 2
    CHECK_EQ(g7(t.adam_lr.back(), 6), std::string("5.96046e-11"));
  }
  const ScheduleTables b = build_schedules(preset("1_2b"));
  CHECK_EQ(b.lr.size(), size_t{22888});
  CHECK_EQ(bits(b.lr[22887]), 0u);                    // 1 + cos rounds to 0: the last 1_2b update has eta = 0
  CHECK_EQ(bits(b.adam_lr[22887]), 0u);
  CHECK_EQ(g7(b.lr[22886], 6), std::string("2.98023e-10"));
}

NM_TEST(s2_formula, levanter_shape_rules) {
  Config c = preset("300m");
  c.optim_warmup = 1;
  CHECK_THROWS(sched::shape(c), "ambiguous");
  c = preset("300m");
  c.optim_decay = 1.5;
  CHECK_THROWS(sched::shape(c), "whole number");
  c.optim_decay = -0.5;
  CHECK_THROWS(sched::shape(c), "must be -1");
  c = preset("300m");
  c.optim_decay = 0;                                  // levanter: no decay -> constant after the warmup
  sched::Shape s = sched::shape(c);
  CHECK_EQ(s.decay, 0);
  CHECK_EQ(s.stable, c.optim_num_steps - 1000);
  ScheduleTables t = build_schedules(c);
  CHECK_EQ(bits(t.lr.back()), bits(0.01f));
  c.optim_decay = 1e6;                                // more steps than remain: clipped to N - warmup
  CHECK_EQ(sched::shape(c).decay, c.optim_num_steps - 1000);
  for (double huge : {9.2e18, 1e19, 1e300}) {         // past 2^63 too (levanter: int(x), then the same clip)
    c.optim_decay = huge;
    s = sched::shape(c);
    CHECK_EQ(s.decay, c.optim_num_steps - 1000);
    CHECK_EQ(s.stable, 0);
  }
  c.optim_decay = std::numeric_limits<double>::infinity();   // levanter refuses inf and nan
  CHECK_THROWS(sched::shape(c), "finite");
  c.optim_decay = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(sched::shape(c), "finite");
  // through the tables: decay = 1e19 is the same schedule as None (N - warmup), bitwise
  c = preset("custom");
  c.optim_num_steps = 100;
  c.optim_schedule = "cosine";
  c.optim_warmup = 10;
  c.optim_decay = -1;
  const ScheduleTables none = build_schedules(c);
  c.optim_decay = 1e19;
  const ScheduleTables huge = build_schedules(c);
  CHECK(std::memcmp(none.lr.data(), huge.lr.data(), 100 * sizeof(float)) == 0);
  CHECK(none.lr[99] < 1e-5f);                         // a cosine to ~0, not the constant peak
  c = preset("130m");
  c.optim_warmup = 100000;                            // warmup longer than the run: min(warmup, N), no decay
  s = sched::shape(c);
  CHECK_EQ(s.warmup, 4959);
  CHECK_EQ(s.decay, 0);
  c = preset("130m");
  c.optim_decay = 0.8;                                // int(0.8 * 4959) = int(3967.2000000000003) = 3967
  CHECK_EQ(sched::shape(c).decay, 3967);
}

NM_TEST(s2_formula, xla_cosf_is_a_cosine) {
  CHECK_EQ(sched::xla_cosf(0.0f), 1.0f);
  int64_t worst = 0;
  for (int i = 0; i <= 200000; ++i) {
    const float x = static_cast<float>(i) * (3.14159274f / 200000.0f);
    const float ref = static_cast<float>(std::cos(static_cast<double>(x)));
    worst = std::max(worst, sched::ulp_distance(sched::xla_cosf(x), ref));
  }
  // __nv_cosf has a 2 ulp bound (CUDA math API); this fixed sweep measures 1, so 1 is asserted (a port regression
  // shows up as a larger distance).
  std::printf("    worst ulp vs correctly rounded cos on [0, pi]: %lld\n", static_cast<long long>(worst));
  CHECK(worst <= 1);
  CHECK_THROWS(sched::xla_cosf(2e5f), "Cody-Waite");
}

NM_TEST(s2_formula, xla_powf_is_a_power) {
  int64_t worst = 0;
  for (float b : {0.9f, 0.98f, 0.5f, 0.999f, 0.95f, 1e-3f, 0.8f})
    for (int t = 0; t <= 100000; t += (t < 100 ? 1 : 97)) {
      const double ref = std::pow(static_cast<double>(b), static_cast<double>(t));
      const float got = sched::xla_powf(b, static_cast<float>(t));
      if (ref < 1e-37) continue;                      // past the normal range
      worst = std::max(worst, sched::ulp_distance(got, static_cast<float>(ref)));
    }
  std::printf("    worst ulp vs correctly rounded pow: %lld\n", static_cast<long long>(worst));
  CHECK(worst <= 1);                                  // measured 1 on this fixed sweep
  CHECK_EQ(sched::xla_powf(0.9f, 0.0f), 1.0f);
  // b = 0 (optim.beta = 0 is a valid knob): pow(+-0, t > 0) = +-0 for odd t, +0 for even t; pow(0, 0) = 1
  CHECK_EQ(bits(sched::xla_powf(0.0f, 0.0f)), bits(1.0f));
  CHECK_EQ(bits(sched::xla_powf(0.0f, 1.0f)), 0u);
  CHECK_EQ(bits(sched::xla_powf(0.0f, 2.0f)), 0u);
  CHECK_EQ(bits(sched::xla_powf(-0.0f, 3.0f)), 0x80000000u);
  CHECK_EQ(bits(sched::xla_powf(-0.0f, 4.0f)), 0u);
  // |t log2 b| > 152 returns 0 before any float -> int conversion: a subnormal base with a large t gave
  // |round(t log2 b)| >= 2^31 (undefined behaviour in the f2i step before the fix; UBSan, BUILD_LOG S2 fixes)
  CHECK_EQ(bits(sched::xla_powf(1e-45f, 1.5e7f)), 0u);
  CHECK_EQ(bits(sched::xla_powf(0.5f, 200.0f)), 0u);
  CHECK_EQ(bits(sched::xla_powf(0.5f, 140.0f)), bits(std::ldexp(1.0f, -140)));   // subnormal result, exact
  CHECK_THROWS(sched::xla_powf(1.0f, 3.0f), "0 <= b < 1");
  CHECK_THROWS(sched::xla_powf(0.9f, 2.5f), "integer");
  CHECK_THROWS(sched::xla_powf(0.9f, 16777216.0f), "2^24");
}

NM_TEST(s2_formula, bias_correction_edges) {
  Config c = preset("custom");
  c.optim_num_steps = 50;
  c.optim_beta1 = 0.0;                                // levanter: 1 - 0^t = 1 for every t >= 1
  c.optim_beta2 = -0.0;                               // the knob check (v >= 0) accepts -0
  ScheduleTables t = build_schedules(c);
  for (int k = 0; k < 50; ++k) {
    CHECK_EQ(bits(t.bc1[k]), bits(1.0f));
    CHECK_EQ(bits(t.bc2[k]), bits(1.0f));
  }
  c.optim_beta2 = 0.99999999;                         // a valid knob whose f32 is 1: 1 - b^t = 0
  CHECK_THROWS(build_schedules(c), "f32(beta) must lie in [0, 1)");
  c.optim_beta2 = 0.98;
  c.optim_num_steps = 1 << 24;
  CHECK_THROWS(build_schedules(c), "t < 2^24");
}

NM_TEST(s2_formula, folded_cosine_constant) {
  // XLA folds (pi * c) / T_d into c * f32(f32(pi) * f32(1/T_d)) (optimized HLO at T_d = 1000: constant 0.00314159296 =
  // 0x3B4DE32F), not c * f32(f32(pi) / f32(T_d)) (0x3B4DE32E). The values: the jitted optax schedule on the A100
  // (cache/levref/s2_probe/cos_fold_1000/sched_main.f32) at the first k where the other constant gives the next float.
  Config c = preset("custom");
  c.optim_num_steps = 3000;
  c.optim_schedule = "cosine";
  c.optim_warmup = 500;
  c.optim_decay = 1000;
  c.optim_min_lr_ratio = 0.0;
  c.optim_lr = 0.01;
  const ScheduleTables t = build_schedules(c);
  const std::pair<int, uint32_t> gpu[] = {{2090, 0x3C209656u}, {2101, 0x3C1FC029u}, {2116, 0x3C1E75D1u},
                                          {2118, 0x3C1E4677u}};
  for (const auto& [k, want] : gpu) CHECK_EQ(bits(t.lr[k]), want);
}

NM_TEST(s2_formula, ulp_distance) {
  CHECK_EQ(sched::ulp_distance(1.0f, std::nextafter(1.0f, 2.0f)), 1);
  CHECK_EQ(sched::ulp_distance(0.0f, -0.0f), 0);
  CHECK_EQ(sched::ulp_distance(std::nextafter(0.0f, 1.0f), std::nextafter(0.0f, -1.0f)), 2);
  CHECK_EQ(sched::ulp_distance(-1.0f, -1.0f), 0);
  const sched::UlpStats s = sched::compare({1.0f, 2.0f, 3.0f}, {1.0f, std::nextafter(2.0f, 3.0f), 3.0f});
  CHECK_EQ(s.n_diff, 1);
  CHECK_EQ(s.max_ulp, 1);
  CHECK_EQ(s.first, 1);
}

NM_TEST(s2_formula, capi_formula) {
  const ScheduleTables want = build_schedules(preset("130m"));
  std::vector<float> lr(4959), alr(4959), bc1(4959), bc2(4959);
  int nd = -1;
  CHECK_EQ(nm_schedule_tables(nullptr, 4959, 0, lr.data(), alr.data(), bc1.data(), bc2.data(), &nd), NM_OK);
  CHECK_EQ(nd, 0);
  CHECK(std::memcmp(lr.data(), want.lr.data(), lr.size() * 4) == 0);
  CHECK(std::memcmp(alr.data(), want.adam_lr.data(), alr.size() * 4) == 0);
  CHECK(std::memcmp(bc1.data(), want.bc1.data(), bc1.size() * 4) == 0);
  CHECK(std::memcmp(bc2.data(), want.bc2.data(), bc2.size() * 4) == 0);
  CHECK_EQ(nm_schedule_tables(nullptr, 100, 0, lr.data(), nullptr, nullptr, nullptr, nullptr), NM_ERR_INVALID);
  CHECK(std::string(nm_last_error()).find("optim.num_steps") != std::string::npos);
  CHECK_EQ(nm_schedule_tables("", 4959, 1, lr.data(), nullptr, nullptr, nullptr, nullptr), NM_ERR_INVALID);
  CHECK(std::string(nm_last_error()).find("needs optim.schedule_fixture") != std::string::npos);
  TmpDir d("capi");
  const std::string cfg = d.str() + "/c.toml";
  std::ofstream(cfg) << "[model]\nsize = \"1_2b\"\n";
  std::vector<float> big(22888);
  CHECK_EQ(nm_schedule_tables(cfg.c_str(), 22888, 0, big.data(), nullptr, nullptr, nullptr, nullptr), NM_OK);
  CHECK_EQ(bits(big[22887]), 0u);
}

// ------------------------------------------------------------------------------------------------ s2_fixture
NM_TEST(s2_fixture, load_checks_and_cross_check) {
  TmpDir d("load");
  Config c = preset("300m");
  write_fixture(d.p, c, build_schedules(c));
  c.optim_schedule_fixture = d.str();
  ScheduleTables t = load_schedules(c);
  CHECK_EQ(t.source, "fixture:" + sched::dir_content_hash(d.str()));
  CHECK_EQ(t.n_ulp_diffs, 0);
  CHECK_EQ(t.lr.size(), size_t{11444});
  // one value moved by one ulp (sha updated): loaded as is, counted by the cross-check, never overridden
  std::vector<float> lr = build_schedules(c).lr;
  lr[5000] = std::nextafter(lr[5000], 1.0f);
  write_table(d.p / "sched_main.f32", lr);
  refresh_sha(d.p, "sched_main");
  t = load_schedules(c);
  CHECK_EQ(t.n_ulp_diffs, 1);
  CHECK_EQ(t.max_ulp_diff, 1);
  CHECK_EQ(bits(t.lr[5000]), bits(lr[5000]));
  // a changed file whose sha256 was not updated
  lr[7] = 1.0f;
  write_table(d.p / "sched_main.f32", lr);
  CHECK_THROWS(load_schedules(c), "sha256");
}

NM_TEST(s2_fixture, refuses_another_schedule) {
  TmpDir d("other");
  Config c = preset("300m");
  write_fixture(d.p, c, build_schedules(c));
  c.optim_schedule_fixture = d.str();
  Config x = c;
  x.optim_lr = 0.02;
  CHECK_THROWS(load_schedules(x), "learning_rate");
  x = c;
  x.optim_beta2 = 0.95;
  CHECK_THROWS(load_schedules(x), "beta2");
  x = c;
  x.optim_schedule = "linear";
  CHECK_THROWS(load_schedules(x), "lr_schedule");
  x = c;
  x.optim_num_steps = 11443;
  CHECK_THROWS(load_schedules(x), "num_train_steps");
  std::vector<float> short_table(100, 0.0f);
  write_table(d.p / "bc1.f32", short_table);
  refresh_sha(d.p, "bc1");
  CHECK_THROWS(load_schedules(c), "bytes");
}

NM_TEST(s2_fixture, refuses_bad_meta) {
  TmpDir d("meta");
  Config c = preset("130m");
  write_fixture(d.p, c, build_schedules(c));
  c.optim_schedule_fixture = d.str();
  std::string toml = nmtest::read_file((d.p / "schedule.toml").string());
  std::string bad = toml;
  bad.replace(bad.find("sched_main = \"sched_main.f32\""), 29, "sched_main = \"../x/sched_main.f32\"");
  std::ofstream(d.p / "schedule.toml") << bad;
  CHECK_THROWS(load_schedules(c), "must be a file name");
  bad = toml;
  bad.replace(bad.find("format = 1"), 10, "format = 2");
  std::ofstream(d.p / "schedule.toml") << bad;
  CHECK_THROWS(load_schedules(c), "format 2");
  fs::remove(d.p / "schedule.toml");
  CHECK_THROWS(load_schedules(c), "not found");
  Config p = preset("130m");
  p.run_parity = true;
  CHECK_THROWS(load_schedules(p), "run.parity needs optim.schedule_fixture");
  p.run_parity = false;
  CHECK_EQ(load_schedules(p).source, std::string("formula"));
}

NM_TEST(s2_fixture, formula_refusal_keeps_the_fixture) {
  // optim.warmup = 1: levanter's fraction 1.0 (warmup over all N), which the formula refuses as ambiguous. A fixture
  // made for it is still the record: it loads, and the cross-check reports itself unavailable (-1).
  TmpDir d("refuse");
  Config c = preset("300m");
  c.optim_warmup = 1;
  CHECK_THROWS(build_schedules(c), "ambiguous");
  Config lev = c;
  lev.optim_warmup = c.optim_num_steps;                // what levanter tabulates for warmup = 1
  const ScheduleTables want = build_schedules(lev);
  write_fixture(d.p, c, want);
  c.optim_schedule_fixture = d.str();
  const ScheduleTables t = load_schedules(c);
  CHECK_EQ(t.n_ulp_diffs, -1);
  CHECK_EQ(t.max_ulp_diff, -1);
  CHECK_EQ(t.source, "fixture:" + sched::dir_content_hash(d.str()));
  CHECK(std::memcmp(t.lr.data(), want.lr.data(), want.lr.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(t.bc2.data(), want.bc2.data(), want.bc2.size() * sizeof(float)) == 0);
  // the same through the C ABI
  TmpDir cd("refuse_cfg");
  const std::string cfg = cd.str() + "/c.toml";
  std::ofstream(cfg) << "[model]\nsize = \"300m\"\n\n[optim]\nwarmup = 1\nschedule_fixture = \"" << d.str() << "\"\n";
  std::vector<float> lr(c.optim_num_steps);
  int nd = 0;
  CHECK_EQ(nm_schedule_tables(cfg.c_str(), c.optim_num_steps, 1, lr.data(), nullptr, nullptr, nullptr, &nd), NM_OK);
  CHECK_EQ(nd, -1);
  CHECK(std::memcmp(lr.data(), want.lr.data(), lr.size() * sizeof(float)) == 0);
  // without a fixture the refusal stands
  c.optim_schedule_fixture.clear();
  CHECK_THROWS(load_schedules(c), "ambiguous");
}

NM_TEST(s2_fixture, content_hash_equals_config_resolution) {
  TmpDir d("hash");
  Config c = preset("520m");
  write_fixture(d.p, c, build_schedules(c));
  c.optim_schedule_fixture = d.str();
  resolve(c, nullptr, nullptr, nullptr);
  CHECK_EQ(c.optim_schedule_fixture, d.str() + "#sha256=" + sched::dir_content_hash(d.str()));
  // the identity covers every file below the directory, as resolution's does: a file in a subdirectory changes it
  const std::string before = sched::dir_content_hash(d.str());
  fs::create_directories(d.p / "sub");
  std::ofstream(d.p / "sub" / "note.txt") << "x";
  CHECK(sched::dir_content_hash(d.str()) != before);
  fs::remove_all(d.p / "sub");
  CHECK_EQ(sched::dir_content_hash(d.str()), before);
  const ScheduleTables t = load_schedules(c);           // the recorded hash matches
  CHECK_EQ(t.source, "fixture:" + sched::dir_content_hash(d.str()));
  std::vector<float> a = build_schedules(c).adam_lr;    // the content changes after resolution
  a[3] = 0.5f;
  write_table(d.p / "sched_adam.f32", a);
  refresh_sha(d.p, "sched_adam");
  CHECK_THROWS(load_schedules(c), "changed");
}

// ------------------------------------------------------------------------------------------------ s2_levanter
NM_TEST(s2_levanter, formula_equals_fixture_all_sizes) {
  int found = 0;
  for (const char* size : {"130m", "300m", "520m", "1_2b"}) {
    const std::string dir = nmtest::src_dir() + "/cache/levref/" + size + "/schedules";
    if (!nmtest::file_exists(dir + "/schedule.toml")) continue;
    ++found;
    check_formula_equals_fixture(dir, preset(size));
  }
  if (found == 0)
    SKIP("no fixture under cache/levref/<size>/schedules: run tests/s2/schedule_fixture.py on a GPU (DESIGN.md §4.4)");
  if (found != 4) nmtest::fail(__FILE__, __LINE__, "only " + std::to_string(found) + " of 4 size fixtures exist");
}

// ------------------------------------------------------------------------------------------------ s2_probe
// The names of PROBES in tests/s2/schedule_fixture.py (a probe missing from the cache fails: rerun --probes).
constexpr const char* kProbes[] = {"cos_min", "lin_warm_mid", "cos_stable", "lin_full", "cos_steps", "cos_odd_t",
                                   "lin_min", "warm_all", "decay0", "cos_long", "cos_fold_1000", "cos_fold_4000",
                                   "decay_huge", "beta0"};

NM_TEST(s2_probe, formula_equals_fixture_probes) {
  const std::string root = nmtest::src_dir() + "/cache/levref/s2_probe";
  std::error_code ec;
  if (!fs::is_directory(root, ec))
    SKIP("no cache/levref/s2_probe: run tests/s2/schedule_fixture.py --probes on a GPU");
  for (const char* name : kProbes) {
    const std::string dir = root + "/" + name;
    if (!nmtest::file_exists(dir + "/schedule.toml")) {
      nmtest::fail(__FILE__, __LINE__, dir + " missing: rerun tests/s2/schedule_fixture.py --probes");
      continue;
    }
    check_formula_equals_fixture(dir, preset("custom"));
  }
}
