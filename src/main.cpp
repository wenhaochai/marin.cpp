// marincpp-train: the training binary (train.hpp CLI). Stage S0 implements plan, config and knobs; train exits 1 until
// the trainer exists (S7/S10).
//
//   marincpp-train plan   [--config f.toml]... [--size S] [--preset P]... [--set k=v]... [--json] [--buffers]
//   marincpp-train config [same]  [--json | --toml]
//   marincpp-train knobs  [--defaults-json]
//   marincpp-train train  [same]
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "config/text.hpp"
#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"

using namespace nm;

namespace {

struct Args {
  std::string cmd;
  std::vector<std::string> files, presets, sets;
  bool json = false, toml = false, buffers = false, defaults_json = false;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "%s\nusage: marincpp-train {plan|config|knobs|train} [--config f.toml]... [--size S] [--preset P]... "
               "[--set section.name=value]... [--json] [--toml] [--buffers] [--defaults-json]\n",
               msg);
  std::exit(static_cast<int>(ExitCode::ConfigError));
}

Args parse(int argc, char** argv) {
  if (argc < 2) usage("missing command");
  Args a;
  a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string o = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value after " + o).c_str());
      return argv[++i];
    };
    if (o == "--config") a.files.push_back(val());
    else if (o == "--size") a.sets.insert(a.sets.begin(), "model.size=" + val());
    else if (o == "--preset") a.presets.push_back(val());
    else if (o == "--set") a.sets.push_back(val());
    else if (o == "--json") a.json = true;
    else if (o == "--toml") a.toml = true;
    else if (o == "--buffers") a.buffers = true;
    else if (o == "--defaults-json") a.defaults_json = true;
    else usage(("unknown option " + o).c_str());
  }
  return a;
}

std::string num(double v) {
  char b[40];
  std::snprintf(b, sizeof(b), "%.17g", v);
  return b;
}

struct Resolved {
  Config c;
  ModelShape m;
  DatasetMeta d;
  PlanChoice pc;
  std::string report;
  std::string label;   // from the config as written, before resolve() (DESIGN.md §9.2)
};

// load -> validate -> plan choice -> resolve -> validate. Without a dataset meta (S1) the planner uses the bound
// kDefaultMaxSegsPerSeq and the dataset knob stays unhashed.
bool resolve_all(const Args& a, Resolved& r) {
  r.c = load_config(a.files, a.presets, a.sets);
  validate(r.c);
  r.label = r.c.run_label.empty() ? default_label(r.c) : r.c.run_label;
  r.m = model_shape(r.c);
  r.d.seq_len = r.m.S;
  r.d.vocab = r.m.V;
  r.d.max_segs_per_seq = kDefaultMaxSegsPerSeq;
  r.pc = choose_plan(r.c, r.m, r.d, &r.report);
  if (r.pc.micro_seqs == 0) return false;
  resolve(r.c, nullptr, nullptr, &r.pc);
  validate(r.c);
  return true;
}

int cmd_knobs(const Args& a) {
  if (a.defaults_json) {
    std::fputs(knob_defaults_json().c_str(), stdout);
    return 0;
  }
  Config d;
  std::printf("%-28s %-7s %-6s %-28s %s\n", "knob", "type", "hash", "default", "doc");
  for (const KnobInfo& k : knob_table())
    std::printf("%-28s %-7s %-6s %-28s %s\n", k.key, knob_type_name(k.type), (k.flags & NM_HASH) ? "yes" : "local",
                k.get_text(d).c_str(), k.doc);
  const char* kinds[] = {"block", "input", "head", "loss"};
  for (int i = 0; i < 4; ++i) {
    std::printf("registry %s:", kinds[i]);
    for (auto& n : registry_names(static_cast<RegistryKind>(i)))
      std::printf(" %s%s", n.c_str(), registry_factory(static_cast<RegistryKind>(i), n) ? "" : "(not built)");
    std::printf("\n");
  }
  std::printf("knob table version %d\n", kKnobTableVersion);
  return 0;
}

int cmd_config(const Args& a) {
  Resolved r;
  if (!resolve_all(a, r)) {
    std::fprintf(stderr, "%snothing fits\n", r.report.c_str());
    return static_cast<int>(ExitCode::ConfigError);
  }
  if (a.toml) {
    std::fputs(full_toml(r.c).c_str(), stdout);
  } else if (a.json) {
    std::printf("{\"run_id\": %s, \"label\": %s, \"config_hash\": %s, \"canonical\": %s,\n\"knobs\": %s}\n",
                json_quote(run_id(r.c)).c_str(), json_quote(r.label).c_str(),
                json_quote(config_hash_hex(r.c)).c_str(), json_quote(canonical_json(r.c)).c_str(),
                full_json(r.c).c_str());
  } else {
    std::printf("run_id %s\nlabel %s\ncanonical %s\n\n%s", run_id(r.c).c_str(), r.label.c_str(),
                canonical_json(r.c).c_str(), full_toml(r.c).c_str());
  }
  return 0;
}

const char* stage_name(CeStage s) { return s == CeStage::B1 ? "B1" : s == CeStage::B1_5 ? "B1.5" : "B2"; }

int cmd_plan(const Args& a) {
  Resolved r;
  if (!resolve_all(a, r)) {
    if (a.json) std::printf("{\"fits\": false, \"report\": %s}\n", json_quote(r.report).c_str());
    else std::fprintf(stderr, "%snothing fits\n", r.report.c_str());
    return static_cast<int>(ExitCode::ConfigError);
  }
  const Config& c = r.c;
  const ModelShape& m = r.m;
  ExecShape e = exec_shape(c, m, &r.d);
  MemoryPlan plan = build_plan(c, m, e, r.d);
  plan.finalize(UINT64_MAX, c.exec_debug_arena != "off");
  const uint64_t budget = mem_budget_bytes(c);
  const bool fits = plan.device_bytes() <= budget;
  const CeStage stages[] = {CeStage::B1, CeStage::B1_5, CeStage::B2};
  std::vector<StepBudget> sb;
  for (CeStage s : stages) sb.push_back(step_budget(c, m, e, s, c.exec_fused, NsPrecision::TF32));
  StepBudget unfused = step_budget(c, m, e, CeStage::B1, false, NsPrecision::TF32);
  StepBudget ns_f32 = step_budget(c, m, e, CeStage::B1, c.exec_fused, NsPrecision::F32);
  StepBudget ns_3x = step_budget(c, m, e, CeStage::B1, c.exec_fused, NsPrecision::X3TF32);
  const char* remat = e.remat == Remat::All ? "all" : e.remat == Remat::Min ? "min" : "layer";

  if (a.json) {
    std::string o = "{";
    o += "\"run_id\": " + json_quote(run_id(c)) + ", \"label\": " + json_quote(r.label) + ",\n";
    o += "\"model\": {\"V\": " + std::to_string(m.V) + ", \"D\": " + std::to_string(m.D) + ", \"L\": " +
         std::to_string(m.L) + ", \"H\": " + std::to_string(m.H) + ", \"KV\": " + std::to_string(m.KV) +
         ", \"hd\": " + std::to_string(m.hd) + ", \"F\": " + std::to_string(m.F) + ", \"S\": " + std::to_string(m.S) +
         ", \"n_readouts\": " + std::to_string(m.n_readouts) + ", \"n_ce_heads\": " + std::to_string(m.n_ce_heads) +
         ", \"n_params\": " + std::to_string(m.n_params) + ", \"flops_per_example\": " + num(m.flops_per_example) +
         "},\n";
    o += "\"choice\": {\"micro_seqs\": " + std::to_string(e.micro_seqs) + ", \"remat\": \"" + remat +
         "\", \"zero1\": " + (e.zero1 ? "true" : "false") + ", \"ce_chunk_rows\": " + std::to_string(e.ce_chunk_rows) +
         ", \"n_micro\": " + std::to_string(e.n_micro) + ", \"micro_tokens\": " + std::to_string(e.micro_tokens) +
         "},\n";
    o += "\"budget_bytes\": " + std::to_string(budget) + ", \"payload_bytes\": " + std::to_string(plan.payload_bytes()) +
         ", \"device_bytes\": " + std::to_string(plan.device_bytes()) + ", \"fits\": " + (fits ? "true" : "false") +
         ",\n";
    o += "\"subtotals\": {";
    const char cls[] = {'P', 'A', 'T', 'W'};
    for (int i = 0; i < 4; ++i)
      o += std::string(i ? ", " : "") + "\"" + cls[i] + "\": " + std::to_string(plan.payload_bytes(cls[i]));
    o += "},\n\"groups\": [";
    bool first = true;
    for (auto& g : plan.groups()) {
      o += std::string(first ? "\n" : ",\n") + "  {\"name\": " + json_quote(g.name) + ", \"class\": \"" +
           lifetime_class(g.life) + "\", \"count\": " + std::to_string(g.count) + ", \"bytes\": " +
           std::to_string(g.bytes) + "}";
      first = false;
    }
    o += "],\n";
    if (a.buffers) {
      o += "\"buffers\": [";
      for (int i = 0; i < plan.size(); ++i) {
        const BufferSpec& b = plan.spec(BufferId{i});
        std::string shp = "[";
        for (int k = 0; k < b.ndim; ++k) shp += (k ? ", " : "") + std::to_string(b.shape[k]);
        o += std::string(i ? ",\n" : "\n") + "  {\"name\": " + json_quote(b.name) + ", \"group\": " +
             json_quote(b.group) + ", \"life\": \"" + lifetime_name(b.life) + "\", \"dtype\": \"" +
             dtype_name(b.dtype) + "\", \"shape\": " + shp + "], \"bytes\": " + std::to_string(b.bytes()) +
             ", \"offset\": " + std::to_string(plan.offset(BufferId{i})) + "}";
      }
      o += "],\n";
    }
    o += "\"budget\": {";
    for (size_t s = 0; s < sb.size(); ++s) {
      o += std::string(s ? ",\n  " : "\n  ") + "\"" + stage_name(stages[s]) + "\": {\"step_s\": " + num(sb[s].step_s) +
           ", \"eval_s\": " + num(sb[s].eval_s) + ", \"tokens_per_s\": " + num(sb[s].tokens_per_s) +
           ", \"mfu_pct\": " + num(sb[s].mfu_pct) + ", \"rows\": {";
      for (size_t k = 0; k < sb[s].rows.size(); ++k)
        o += std::string(k ? ", " : "") + "\"" + sb[s].rows[k].first + "\": " + num(sb[s].rows[k].second);
      o += "}}";
    }
    o += ",\n  \"n_evals\": " + std::to_string(sb[0].n_evals) + ", \"unfused_B1_step_s\": " + num(unfused.step_s) +
         ", \"unfused_B1_elementwise_s\": " + num(unfused.row("elementwise")) + ", \"ns_f32_s\": " +
         num(ns_f32.row("newton_schulz")) + ", \"ns_3xtf32_s\": " + num(ns_3x.row("newton_schulz")) + "},\n";
    o += "\"report\": " + json_quote(r.report) + "}\n";
    std::fputs(o.c_str(), stdout);
  } else {
    std::printf("%s  run %s\n", r.label.c_str(), run_id(c).c_str());
    std::printf("%s: %d GPUs, micro %d seqs/GPU (%" PRId64 " tokens) x %d, remat=%s, zero1=%s, CE chunk %d rows\n",
                c.model_size.c_str(), e.ranks, e.micro_seqs, e.micro_tokens, e.n_micro, remat, e.zero1 ? "on" : "off",
                e.ce_chunk_rows);
    std::printf("parameters %" PRId64 ", flops_per_example %.13g\n\n", m.n_params, m.flops_per_example);
    if (a.buffers) plan.print(stdout);
    else plan.print_grouped(stdout);
    std::printf("budget %.2f GiB: %s\n\n", budget / 1073741824.0, fits ? "fits" : "DOES NOT FIT");
    for (size_t s = 0; s < sb.size(); ++s) {
      std::printf("step budget CE %s (ms):", stage_name(stages[s]));
      for (auto& [n, v] : sb[s].rows) std::printf(" %s=%.1f", n, v * 1e3);
      std::printf(" | step %.1f ms, %.0f tok/s, levanter-MFU %.1f%%\n", sb[s].step_s * 1e3, sb[s].tokens_per_s,
                  sb[s].mfu_pct);
    }
    std::printf("unfused elementwise (B1): step %.1f ms; Newton-Schulz f32 %.1f ms, 3xTF32 %.1f ms\n",
                unfused.step_s * 1e3, ns_f32.row("newton_schulz") * 1e3, ns_3x.row("newton_schulz") * 1e3);
    std::printf("eval %.1f s x %d = %.0f s\n", sb[0].eval_s, sb[0].n_evals, sb[0].eval_s * sb[0].n_evals);
  }
  return fits ? 0 : static_cast<int>(ExitCode::ConfigError);
}

int cmd_train(const Args& a) {
  Resolved r;
  if (!resolve_all(a, r)) {
    std::fprintf(stderr, "%snothing fits\n", r.report.c_str());
    return static_cast<int>(ExitCode::ConfigError);
  }
  std::fprintf(stderr, "marincpp-train train: the trainer is not built yet (stages S7, S10); run %s resolved\n",
               run_id(r.c).c_str());
  return static_cast<int>(ExitCode::ConfigError);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    if (a.cmd == "plan") return cmd_plan(a);
    if (a.cmd == "config") return cmd_config(a);
    if (a.cmd == "knobs") return cmd_knobs(a);
    if (a.cmd == "train") return cmd_train(a);
    usage(("unknown command " + a.cmd).c_str());
  } catch (const std::exception& e) {   // nm::Fatal, and std::filesystem_error, bad_alloc, ... (never terminate())
    std::fprintf(stderr, "marincpp-train: %s\n", e.what());
    return static_cast<int>(ExitCode::ConfigError);
  }
}
