// Presets, loading, resolution, validation and run identity (config.hpp; DESIGN.md §9.1-§9.2).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <sstream>

#include "config/text.hpp"
#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"

namespace nm {
namespace {

// ------------------------------------------------------------------------------------------------ presets
// Hyperparameter rows of docs/spec/kaiyue_baseline.md ("Hyperparameters per size"); pls_d48 is the 130m row at 48
// layers with separate per-layer readouts (DESIGN.md §2.5).
struct SizeRow {
  const char* name;
  int D, L, H, KV, F, B, N;
  double lr, adam_lr, momentum, eps, max_grad_norm;
  const char* schedule;
  int warmup;
  double decay;
};
constexpr SizeRow kSizes[] = {
    {"130m", 512, 6, 8, 8, 1792, 128, 4959, 0.02, 0.008, 0.95, 1e-20, 1.0, "linear", 0, 0.8},
    {"300m", 768, 12, 12, 12, 2688, 128, 11444, 0.01, 0.002, 0.98, 1e-15, 1.0, "cosine", 1000, -1},
    {"520m", 1024, 24, 16, 8, 3584, 256, 9918, 0.01, 0.002, 0.98, 1e-15, 1.0, "cosine", 1000, -1},
    {"1_2b", 2048, 16, 16, 8, 7168, 256, 22888, 0.01, 0.0015, 0.98, 1e-15, 2.0, "cosine", 1000, -1},
};

void apply_size(Config& c, const SizeRow& r) {
  c.model_size = r.name;
  c.model_hidden = r.D;
  c.model_layers = r.L;
  c.model_heads = r.H;
  c.model_kv_heads = r.KV;
  c.model_head_dim = 0;
  c.model_mlp = r.F;
  c.data_global_batch = r.B;
  c.optim_num_steps = r.N;
  c.optim_lr = r.lr;
  c.optim_adam_lr = r.adam_lr;
  c.optim_momentum = r.momentum;
  c.optim_eps = r.eps;
  c.optim_max_grad_norm = r.max_grad_norm;
  c.optim_schedule = r.schedule;
  c.optim_warmup = r.warmup;
  c.optim_decay = r.decay;
}

[[noreturn]] void config_error(const std::string& msg) { fatal(NM_WHERE, "config: " + msg); }

void set_from_toml(Config& c, const TomlEntry& e, const std::string& origin) {
  const KnobInfo* k = find_knob(e.key);
  if (!k) config_error(origin + ":" + std::to_string(e.line) + ": unknown knob " + e.key);
  bool ok_kind = false;
  switch (k->type) {
    case KnobType::String: ok_kind = e.kind == TomlEntry::String; break;
    case KnobType::Bool: ok_kind = e.kind == TomlEntry::Bool; break;
    case KnobType::Int:
    case KnobType::Int64: ok_kind = e.kind == TomlEntry::Int; break;
    case KnobType::Double: ok_kind = e.kind == TomlEntry::Int || e.kind == TomlEntry::Float; break;
  }
  if (!ok_kind || !k->set_text(c, e.text))
    config_error(origin + ":" + std::to_string(e.line) + ": " + e.key + " needs a " + knob_type_name(k->type) +
                 " value, got '" + e.text + "'");
}

// ------------------------------------------------------------------------------------------------ file hashing
std::string hash_dir(const std::filesystem::path& dir) {
  std::vector<std::string> rel;
  for (auto& ent : std::filesystem::recursive_directory_iterator(dir))
    if (ent.is_regular_file()) rel.push_back(std::filesystem::relative(ent.path(), dir).generic_string());
  std::sort(rel.begin(), rel.end());
  std::string listing;
  for (auto& r : rel) listing += r + '\0' + sha256_file_hex((dir / r).string()) + '\n';
  return sha256_hex(listing);
}

// "<path>" -> "<path>#sha256=<hex>"; a value that already carries a hash is verified against the file.
std::string with_content_hash(const std::string& key, const std::string& value) {
  if (value.empty()) return value;
  std::string path = value, want;
  size_t at = value.find("#sha256=");
  if (at != std::string::npos) {
    path = value.substr(0, at);
    want = value.substr(at + 8);
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) config_error(key + ": file not found: " + path);
  std::string got;
  try {
    got = std::filesystem::is_directory(path, ec) ? hash_dir(path) : sha256_file_hex(path);
  } catch (const std::filesystem::filesystem_error& e) {
    config_error(key + ": cannot hash " + path + ": " + e.what());
  }
  if (!want.empty() && want != got)
    config_error(key + ": content of " + path + " changed: sha256 " + got + ", the config records " + want);
  return path + "#sha256=" + got;
}

const char* remat_name(Remat r) {
  switch (r) {
    case Remat::All: return "all";
    case Remat::Min: return "min";
    case Remat::Layer: return "layer";
  }
  return "?";
}

bool parse_range(const std::string& s, int64_t& a, int64_t& b) {
  size_t colon = s.find(':');
  if (colon == std::string::npos) return false;
  return from_text(std::string_view(s).substr(0, colon), a) && from_text(std::string_view(s).substr(colon + 1), b) &&
         a >= 0 && a <= b;
}

bool is_size_name(std::string_view n) {
  for (const SizeRow& r : kSizes)
    if (n == r.name) return true;
  return n == "pls_d48" || n == "custom";
}

// --set value text: a string knob may be quoted ("x" or 'x'); the quotes are not part of the value.
std::string_view unquote(std::string_view v) {
  if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\'')))
    return v.substr(1, v.size() - 2);
  return v;
}

// The content hash of an NM_CONTENT value ("<path>#sha256=<hex>" -> "sha256:<hex>"); the value itself if unresolved.
std::string content_identity(const std::string& v) {
  size_t at = v.find("#sha256=");
  return at == std::string::npos ? v : "sha256:" + v.substr(at + 8);
}

// A label tag with its one conversion replaced by `value` (knobs.cpp checks the tags at compile time).
std::string render_tag(const char* tag, const std::string& value) {
  std::string out;
  for (const char* p = tag; *p; ++p) {
    if (*p == '%' && p[1]) {
      out += value;
      ++p;
    } else {
      out += *p;
    }
  }
  return out;
}

std::string label_value(const std::string& v) {
  std::string s = v.substr(0, v.find("#sha256="));
  size_t slash = s.find_last_of('/');
  if (slash != std::string::npos) s = s.substr(slash + 1);
  return s;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ resolution API
void apply_preset(Config& c, std::string_view name) {
  for (const SizeRow& r : kSizes)
    if (name == r.name) {
      apply_size(c, r);
      return;
    }
  if (name == "pls_d48") {
    apply_size(c, kSizes[0]);
    c.model_size = "pls_d48";
    c.model_layers = 48;
    c.readout_kind = "separate";
    return;
  }
  if (name == "custom") {
    c.model_size = "custom";
    return;
  }
  if (name == "levanter_strict") {
    c.exec_attention = "levref";
    c.exec_grad_reduce = "bf16";
    c.exec_embed_accum = "bf16_seq";
    c.exec_ce_dw_round = "chunk";
    c.exec_ce_chunk_rows = 8192;
    if (c.model_size == "520m" || c.model_size == "1_2b") {
      c.exec_micro_seqs = 32;
      c.exec_remat = "layer";
    }
    return;
  }
  config_error("unknown preset '" + std::string(name) + "' (130m, 300m, 520m, 1_2b, pls_d48, custom, levanter_strict)");
}

void load_toml(Config& c, const std::string& path) {
  for (const TomlEntry& e : parse_toml_file(path)) set_from_toml(c, e, path);
}

void apply_override(Config& c, std::string_view kv) {
  size_t eq = kv.find('=');
  if (eq == std::string_view::npos) config_error("--set needs key=value, got '" + std::string(kv) + "'");
  std::string key(kv.substr(0, eq));
  std::string_view val = kv.substr(eq + 1);
  const KnobInfo* k = find_knob(key);
  if (!k) config_error("--set: unknown knob " + key);
  if (k->type == KnobType::String) val = unquote(val);
  if (!k->set_text(c, val))
    config_error("--set " + key + " needs a " + knob_type_name(k->type) + " value, got '" + std::string(val) + "'");
}

Config load_config(const std::vector<std::string>& toml_files, const std::vector<std::string>& presets,
                   const std::vector<std::string>& overrides) {
  std::vector<std::pair<std::string, std::vector<TomlEntry>>> files;
  for (const std::string& f : toml_files) files.emplace_back(f, parse_toml_file(f));
  std::string size = "130m";
  for (auto& [path, entries] : files)
    for (auto& e : entries)
      if (e.key == "model.size") size = e.text;
  for (const std::string& o : overrides)
    if (o.rfind("model.size=", 0) == 0) size = std::string(unquote(std::string_view(o).substr(11)));
  if (!is_size_name(size))
    config_error("model.size = '" + size + "' is not a size (130m, 300m, 520m, 1_2b, pls_d48, custom)");
  Config c;
  apply_preset(c, size);
  for (const std::string& p : presets) {
    if (is_size_name(p))
      config_error("preset '" + p + "' is a model size: pass it as --size " + p + " (or model.size), not --preset");
    apply_preset(c, p);
  }
  for (auto& [path, entries] : files)
    for (auto& e : entries) set_from_toml(c, e, path);
  for (const std::string& o : overrides) apply_override(c, o);
  return c;
}

void resolve(Config& c, const DatasetMeta* data_meta, const DatasetMeta* eval_meta, const PlanChoice* pc) {
  if (pc) {
    c.exec_micro_seqs = pc->micro_seqs;
    c.exec_remat = remat_name(pc->remat);
    c.optim_zero1 = pc->zero1 ? "on" : "off";
    c.optim_zero2_vocab = pc->zero2_vocab ? "on" : "off";
    c.exec_ce_chunk_rows = pc->ce_chunk_rows;
  }
  // T1.0 facts file does not exist yet (tools/levref, S6): non-parity runs take the bf16 store of DESIGN.md §2.5.
  if (c.exec_ce_logits == "auto" && !c.run_parity) c.exec_ce_logits = "bf16";
  c.init_fixture = with_content_hash("init.fixture", c.init_fixture);
  c.optim_schedule_fixture = with_content_hash("optim.schedule_fixture", c.optim_schedule_fixture);
  c.data_plan = with_content_hash("data.plan", c.data_plan);
  c.readout_table = with_content_hash("readout.table", c.readout_table);
  c.optim_groups = with_content_hash("optim.groups", c.optim_groups);
  c.optim_row_partition = with_content_hash("optim.row_partition", c.optim_row_partition);
  c.eval_extra_sources = with_content_hash("eval.extra_sources", c.eval_extra_sources);
  if (data_meta && c.data_dataset.find("#sha256=") == std::string::npos) {
    NM_REQUIRE(!data_meta->meta_sha256.empty(), "dataset meta without its sha256");
    c.data_dataset += "#sha256=" + data_meta->meta_sha256;
  }
  if (eval_meta && c.eval_paloma.find("#sha256=") == std::string::npos) {
    NM_REQUIRE(!eval_meta->meta_sha256.empty(), "eval meta without its sha256");
    c.eval_paloma += "#sha256=" + eval_meta->meta_sha256;
  }
}

void validate(const Config& c) {
  std::vector<std::string> err;
  for (const KnobInfo& k : knob_table())
    if (!k.check(c)) err.push_back(std::string(k.key) + " = '" + k.get_text(c) + "' is not allowed (" + k.doc + ")");
  auto rule = [&](bool ok, const std::string& msg) {
    if (!ok) err.push_back(msg);
  };
  const int hd = c.model_head_dim ? c.model_head_dim : (c.model_heads ? c.model_hidden / c.model_heads : 0);
  rule(c.model_kv_heads > 0 && c.model_heads % c.model_kv_heads == 0,
       "model.heads must be a multiple of model.kv_heads");
  if (c.model_head_dim == 0)
    rule(c.model_heads > 0 && c.model_hidden % c.model_heads == 0 && (hd == 64 || hd == 128),
         "model.head_dim = 0 needs model.hidden / model.heads in {64, 128}");
  rule(c.exec_gpus > 0 && c.data_global_batch % c.exec_gpus == 0, "data.global_batch must be a multiple of exec.gpus");
  rule(c.exec_gpus > 0 && c.eval_batch % c.exec_gpus == 0, "eval.batch must be a multiple of exec.gpus");
  if (c.exec_micro_seqs > 0 && c.exec_gpus > 0)
    rule((c.data_global_batch / c.exec_gpus) % c.exec_micro_seqs == 0,
         "exec.micro_seqs must divide data.global_batch / exec.gpus");
  rule(!c.readout_local || c.readout_kind == "separate", "readout.local needs readout.kind = separate");
  rule(!c.readout_detach_head || c.readout_kind == "shared", "readout.detach_head needs readout.kind = shared");
  rule(!c.readout_detach_backbone || c.readout_kind == "separate",
       "readout.detach_backbone needs readout.kind = separate");
  int64_t a, b;
  rule(c.readout_on.empty() || parse_range(c.readout_on, a, b), "readout.on must be S:E with 0 <= S <= E");
  rule(c.readout_off.empty() || parse_range(c.readout_off, a, b), "readout.off must be S:E with 0 <= S <= E");
  rule(!(c.model_block == "hc" || c.model_block == "mhc") || c.model_hc_streams >= 2,
       "model.block = hc/mhc needs model.hc_streams >= 2");
  rule(c.optim_weight_decay == 0.0 || c.optim_wd_variant,
       "optim.weight_decay > 0 needs optim.wd_variant = true (the muonh_qwen3 baseline logs 0.1 but applies none)");
  rule(c.optim_ns_coeffs == "simple", "optim.ns_coeffs: only 'simple' is implemented");
  rule(!c.model_tie_embeddings || c.readout_kind != "separate",
       "model.tie_embeddings needs readout.kind != separate (separate readouts have untied heads)");
  rule(c.optim_zero2_vocab != "on",
       "optim.zero2_vocab = on is not implemented (no reducer path and no memory plan for it yet); use auto or off");
  if (c.run_parity) {
    rule(!c.optim_schedule_fixture.empty(), "run.parity needs optim.schedule_fixture");
    rule(c.exec_ce_logits != "auto", "run.parity needs exec.ce_logits resolved from the T1.0 facts file");
    rule(c.init_mode != "threefry",
         "run.parity with init.mode = threefry needs T1.1 recorded as passed for this binary; use init.mode = fixture");
  }
  rule(c.init_mode != "fixture" || !c.init_fixture.empty(), "init.mode = fixture needs init.fixture");
  rule(c.init_mode != "from" || !c.init_from.empty(), "init.mode = from needs init.from");
  if (c.exec_policy == "f32") {
    rule(!c.exec_cuda_graph, "exec.policy = f32 needs exec.cuda_graph = false (eager)");
    rule(c.exec_attention == "ref_f32", "exec.policy = f32 needs exec.attention = ref_f32");
    rule(c.exec_ce == "a", "exec.policy = f32 needs exec.ce = a");
  }
  if (!err.empty()) {
    std::string msg = "invalid config (" + std::to_string(err.size()) + " problem" + (err.size() > 1 ? "s" : "") + "):";
    for (auto& e : err) msg += "\n  - " + e;
    fatal(NM_WHERE, msg);
  }
}

// ------------------------------------------------------------------------------------------------ identity
std::string canonical_json(const Config& c) {
  std::map<std::string, std::string> items;  // sorted by key
  for (const KnobInfo& k : knob_table()) {
    if (!(k.flags & NM_HASH) || k.is_default(c)) continue;
    if (k.flags & NM_CONTENT) items[k.key] = json_quote(content_identity(k.get_text(c)));
    else items[k.key] = k.type == KnobType::String ? json_quote(k.get_text(c)) : k.get_text(c);
  }
  std::string s = "{\"format\":2,\"knob_table_version\":" + std::to_string(kKnobTableVersion) + ",\"knobs\":{";
  bool first = true;
  for (auto& [key, val] : items) {
    if (!first) s += ',';
    first = false;
    s += json_quote(key) + ":" + val;
  }
  return s + "}}";
}

std::string config_hash_hex(const Config& c) { return sha256_hex(canonical_json(c)); }

std::string run_id(const Config& c) { return config_hash_hex(c).substr(0, 12); }

std::string default_label(const Config& c) {
  Config base;
  try {
    apply_preset(base, c.model_size);
  } catch (const Fatal&) {
  }
  std::string label = "muonh-qwen3-" + c.model_size + "-della4xh100-nm";
  for (const KnobInfo& k : knob_table()) {
    if (!*k.tag) continue;
    std::string v = k.get_text(c);
    if (v == k.get_text(base)) continue;
    switch (k.type) {
      case KnobType::Bool: break;
      case KnobType::Int:
      case KnobType::Int64: break;   // canonical integer text is %d
      case KnobType::Double: {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%g", std::stod(v));
        v = buf;
        break;
      }
      case KnobType::String: v = label_value(v); break;
    }
    label += render_tag(k.tag, v);
  }
  return label;
}

std::string full_json(const Config& c) {
  std::string s = "{";
  bool first = true;
  for (const KnobInfo& k : knob_table()) {
    if (!first) s += ",";
    first = false;
    s += "\n  " + json_quote(k.key) + ": " + (k.type == KnobType::String ? json_quote(k.get_text(c)) : k.get_text(c));
  }
  return s + "\n}\n";
}

std::string full_toml(const Config& c) {
  std::string s, section;
  for (const KnobInfo& k : knob_table()) {
    std::string key = k.key;
    size_t dot = key.find('.');
    std::string sec = key.substr(0, dot), name = key.substr(dot + 1);
    if (sec != section) {
      s += (section.empty() ? "" : "\n") + std::string("[") + sec + "]\n";
      section = sec;
    }
    std::string v = k.get_text(c);
    if (k.type == KnobType::String) v = json_quote(v);
    if (k.type == KnobType::Double && v.find_first_of(".e") == std::string::npos) v += ".0";
    s += name + " = " + v + "\n";
  }
  return s;
}

std::string knob_defaults_json() {
  Config d;
  std::string s = "{\n  \"knob_table_version\": " + std::to_string(kKnobTableVersion) + ",\n  \"defaults\": {";
  bool first = true;
  for (const KnobInfo& k : knob_table()) {
    s += first ? "\n" : ",\n";
    first = false;
    s += "    " + json_quote(k.key) + ": " + json_quote(k.get_text(d));
  }
  return s + "\n  }\n}\n";
}

}  // namespace nm
