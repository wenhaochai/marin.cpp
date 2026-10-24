// Gate T0.6 (parity plan): the f32 bit patterns of the optimizer and schedule scalar constants equal the hex column of
// docs/spec/optim.md §5.3 for every size preset. The table is parsed from the spec, so a changed or added row is seen.
#include <bit>
#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <sstream>

#include "marincpp/config.hpp"
#include "nmtest.hpp"

using namespace nm;

namespace {

struct Row {
  std::string label;
  std::vector<uint32_t> hex;   // the f32 hex values of column 2, in order
};

std::vector<std::string> cells(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (size_t i = 1; i < line.size(); ++i) {
    if (line[i] == '|') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += line[i];
    }
  }
  return out;
}

std::vector<Row> spec_table() {
  std::string text = nmtest::read_file(nmtest::src_dir() + "/docs/spec/optim.md");
  if (text.empty()) SKIP("docs/spec/optim.md not found under --src");
  std::istringstream in(text);
  std::string line;
  bool in_section = false, in_table = false;
  std::vector<Row> rows;
  std::regex hex_re("0x([0-9A-Fa-f]{8})");
  while (std::getline(in, line)) {
    if (!in_section) {
      in_section = line.rfind("### 5.3 Scalar constants", 0) == 0;
      continue;
    }
    if (line.rfind("|", 0) != 0) {
      if (in_table) break;
      continue;
    }
    in_table = true;
    auto c = cells(line);
    if (c.size() < 3 || c[1].find("0x") == std::string::npos) continue;   // header and separator rows
    Row r;
    r.label = c[0];
    for (std::sregex_iterator it(c[1].begin(), c[1].end(), hex_re), end; it != end; ++it)
      r.hex.push_back(static_cast<uint32_t>(std::stoul((*it)[1].str(), nullptr, 16)));
    rows.push_back(r);
  }
  return rows;
}

uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }

std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof(b), "0x%08X", v);
  return b;
}

}  // namespace

NM_TEST(t0_6, optimizer_constants_match_spec_hex) {
  std::vector<Row> rows = spec_table();
  CHECK(rows.size() >= 17);
  std::map<std::string, OptimConstants> k;
  for (const char* s : {"130m", "300m", "520m", "1_2b"}) k[s] = optim_constants(load_config({}, {}, {std::string("model.size=") + s}));
  const std::vector<std::string> all = {"130m", "300m", "520m", "1_2b"}, cosine = {"300m", "520m", "1_2b"};
  std::set<std::string> covered;
  int compared = 0;
  // check(field, sizes, hex index)
  auto check = [&](const Row& r, const char* field, float OptimConstants::* f, const std::vector<std::string>& sizes,
                   size_t idx) {
    if (idx >= r.hex.size()) {
      nmtest::fail(__FILE__, __LINE__, "row '" + r.label + "' has no hex value " + std::to_string(idx));
      return;
    }
    for (const std::string& s : sizes) {
      ++compared;
      uint32_t got = bits(k[s].*f);
      if (got != r.hex[idx])
        nmtest::fail(__FILE__, __LINE__, s + " " + field + ": " + hex(got) + ", spec " + hex(r.hex[idx]) + " (" +
                                             r.label + ")");
    }
    covered.insert(field);
  };
  auto has = [](const Row& r, const char* s) { return r.label.find(s) != std::string::npos; };
  for (const Row& r : rows) {
    if (has(r, "`1 - b1`")) check(r, "one_minus_b1", &OptimConstants::one_minus_b1, all, 0);
    else if (has(r, "`1 - b2`")) check(r, "one_minus_b2", &OptimConstants::one_minus_b2, all, 0);
    else if (has(r, "`b1` = 0.9")) check(r, "b1", &OptimConstants::b1, all, 0);
    else if (has(r, "`b2` = 0.98")) {
      check(r, "b2", &OptimConstants::b2, all, 0);
      check(r, "momentum", &OptimConstants::momentum, cosine, 0);   // MuonH mu = 0.98 at 300m, 520m, 1_2b
    } else if (has(r, "= 0.95")) check(r, "momentum", &OptimConstants::momentum, {"130m"}, 0);
    else if (has(r, "`eps` = 1e-20")) check(r, "eps", &OptimConstants::eps, {"130m"}, 0);
    else if (has(r, "`eps` = 1e-15")) check(r, "eps", &OptimConstants::eps, cosine, 0);
    else if (has(r, "`muon_eps`")) check(r, "muon_eps", &OptimConstants::muon_eps, all, 0);
    else if (has(r, "`max_grad_norm`")) {
      check(r, "max_grad_norm", &OptimConstants::max_grad_norm, {"130m", "300m", "520m"}, 0);
      check(r, "max_grad_norm", &OptimConstants::max_grad_norm, {"1_2b"}, 1);
    } else if (has(r, "NS `a`")) check(r, "ns_a", &OptimConstants::ns_a, all, 0);
    else if (has(r, "NS `b`")) check(r, "ns_b", &OptimConstants::ns_b, all, 0);
    else if (has(r, "NS `c`")) check(r, "ns_c", &OptimConstants::ns_c, all, 0);
    else if (has(r, "norm floor")) check(r, "norm_floor", &OptimConstants::norm_floor, all, 0);
    else if (has(r, "peak 0.02 / 0.01")) {
      check(r, "lr_peak", &OptimConstants::lr_peak, {"130m"}, 0);
      check(r, "lr_peak", &OptimConstants::lr_peak, cosine, 1);
    } else if (has(r, "`adam_lr`")) {
      check(r, "adam_lr_peak", &OptimConstants::adam_lr_peak, {"130m"}, 0);
      check(r, "adam_lr_peak", &OptimConstants::adam_lr_peak, {"300m", "520m"}, 1);
      check(r, "adam_lr_peak", &OptimConstants::adam_lr_peak, {"1_2b"}, 2);
    } else if (has(r, "warmup")) check(r, "warmup_init_minus_end", &OptimConstants::warmup_init_minus_end, cosine, 0);
    else if (has(r, "`jnp.pi`")) {
      check(r, "pi", &OptimConstants::pi, all, 0);
      check(r, "half", &OptimConstants::half, all, 1);
    } else {
      nmtest::fail(__FILE__, __LINE__, "spec row not mapped to a constant: " + r.label);
    }
  }
  for (const char* f : {"b1", "b2", "one_minus_b1", "one_minus_b2", "momentum", "eps", "muon_eps", "max_grad_norm",
                        "ns_a", "ns_b", "ns_c", "norm_floor", "lr_peak", "adam_lr_peak", "warmup_init_minus_end",
                        "pi", "half"})
    if (!covered.count(f)) nmtest::fail(__FILE__, __LINE__, std::string("constant not covered by the spec table: ") + f);
  std::printf("    %zu spec rows, %d size x constant comparisons\n", rows.size(), compared);
}

NM_TEST(t0_6, naive_f32_subtraction_differs) {
  // The trap the spec names: f32(1) - f32(b) is not the levanter constant.
  Config c;
  OptimConstants k = optim_constants(c);
  CHECK_EQ(bits(1.0f - k.b1), 0x3DCCCCD0u);
  CHECK(bits(1.0f - k.b1) != bits(k.one_minus_b1));
  CHECK_EQ(bits(1.0f - k.b2), 0x3CA3D700u);
  CHECK(bits(1.0f - k.b2) != bits(k.one_minus_b2));
}
