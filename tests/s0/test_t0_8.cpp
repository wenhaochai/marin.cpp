// Gate T0.8 (parity plan): parameter count and flops_per_example of the four baseline sizes, exact, against (a) the
// hyperparameter table of docs/spec/kaiyue_baseline.md and (b) the logged W&B values (cache/wandb/api/*.json:
// summary parameter_count and throughput/flops_per_example).
#include <algorithm>
#include <cstdlib>
#include <regex>
#include <sstream>

#include "marincpp/config.hpp"
#include "nmtest.hpp"

using namespace nm;

namespace {

const char* kSizes[4] = {"130m", "300m", "520m", "1_2b"};

ModelShape shape_of(const std::string& size) { return model_shape(load_config({}, {}, {"model.size=" + size})); }

// The four value cells of a "| label | 130m | 300m | 520m | 1_2b | source |" row whose label starts with `prefix`.
std::vector<std::string> table_row(const std::string& text, const std::string& prefix) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("| " + prefix, 0) != 0) continue;
    std::vector<std::string> c;
    std::string cur;
    for (size_t i = 1; i < line.size(); ++i) {
      if (line[i] == '|') {
        size_t b = cur.find_first_not_of(' '), e = cur.find_last_not_of(' ');
        c.push_back(b == std::string::npos ? "" : cur.substr(b, e - b + 1));
        cur.clear();
      } else {
        cur += line[i];
      }
    }
    if (c.size() >= 5) return {c[1], c[2], c[3], c[4]};
  }
  return {};
}

int64_t parse_int_commas(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), ','), s.end());
  return std::stoll(s);
}

}  // namespace

NM_TEST(t0_8, spec_table) {
  std::string text = nmtest::read_file(nmtest::src_dir() + "/docs/spec/kaiyue_baseline.md");
  if (text.empty()) SKIP("docs/spec/kaiyue_baseline.md not found under --src");
  auto params = table_row(text, "parameters (total = W&B `parameter_count`)");
  auto nonemb = table_row(text, "non-embedding parameters");
  auto flops = table_row(text, "`flops_per_example`");
  if (params.size() != 4 || nonemb.size() != 4 || flops.size() != 4) {
    nmtest::fail(__FILE__, __LINE__, "hyperparameter table rows not found in kaiyue_baseline.md");
    return;
  }
  for (int i = 0; i < 4; ++i) {
    ModelShape m = shape_of(kSizes[i]);
    CHECK_EQ(m.n_params, parse_int_commas(params[i]));
    CHECK_EQ(m.n_params - 2 * int64_t(m.V) * m.D, parse_int_commas(nonemb[i]));
    double want = std::strtod(flops[i].c_str(), nullptr);
    CHECK_EQ(m.flops_per_example, want);
    std::printf("    %s: %lld parameters, flops_per_example %.13g\n", kSizes[i], static_cast<long long>(m.n_params),
                m.flops_per_example);
  }
}

NM_TEST(t0_8, wandb_logged_values) {
  int found = 0;
  std::regex pc_re("\"parameter_count\":\\s*([0-9]+)");
  std::regex fl_re("\"throughput/flops_per_example\":\\s*([0-9.eE+]+)");
  for (const char* s : kSizes) {
    std::string path = nmtest::src_dir() + "/cache/wandb/api/reself__marin-della__muonh-qwen3-" + s + "-della4xh100.json";
    std::string text = nmtest::read_file(path);
    if (text.empty()) continue;
    ++found;
    std::smatch a, b;
    if (!std::regex_search(text, a, pc_re) || !std::regex_search(text, b, fl_re)) {
      nmtest::fail(__FILE__, __LINE__, std::string("keys not found in ") + path);
      continue;
    }
    ModelShape m = shape_of(s);
    CHECK_EQ(m.n_params, std::stoll(a[1].str()));
    CHECK_EQ(m.flops_per_example, std::strtod(b[1].str().c_str(), nullptr));
  }
  if (found == 0) SKIP("W&B cache cache/wandb/api/ absent (gitignored); spec_table still ran");
  if (found < 4) nmtest::fail(__FILE__, __LINE__, "only " + std::to_string(found) + " of 4 W&B summaries present");
}

NM_TEST(t0_8, pls_d48_and_variants) {
  // DESIGN.md §2.5: pls d48 (width 512, 48 layers, 47 separate readout heads + lm_head) has 3,400,267,776 parameters.
  CHECK_EQ(shape_of("pls_d48").n_params, int64_t(3400267776));
  // levanter's total_trainable_params omits the q/k-norm gains (model.md §5): 130m 154,153,472 = ours - L * 2 hd.
  ModelShape m = shape_of("130m");
  CHECK_EQ(m.n_params - int64_t(m.L) * 2 * m.hd, int64_t(154153472));
  // tied embeddings drop one V x D table
  ModelShape t = model_shape(load_config({}, {}, {"model.tie_embeddings=true"}));
  CHECK_EQ(t.n_params, m.n_params - int64_t(m.V) * m.D);
}
