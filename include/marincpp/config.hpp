// Config: one struct generated from knobs.def; resolution, validation, canonical form, run identity.
// DESIGN.md §9.
#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "marincpp/core.hpp"

namespace nm {

// NM_CONTENT marks a file input (with NM_HASH): the identity takes its content hash only (knobs.def).
enum KnobFlags : uint32_t { NM_HASH = 1u << 0, NM_LOCAL = 1u << 1, NM_CONTENT = 1u << 2 };

// Version of the knob table's defaults. Run identity hashes only hashed knobs that differ from their default, plus this
// version (DESIGN.md §9.2): adding a knob whose default keeps the old behaviour leaves every run id unchanged; CHANGING
// ANY DEFAULT MUST BUMP THIS NUMBER (test s0_config compares the default table against
// tests/s0/knob_defaults_v<N>.json).
inline constexpr int kKnobTableVersion = 1;

// helper used by `check` expressions in knobs.def
inline bool one_of(const std::string& v, std::initializer_list<const char*> xs) {
  for (const char* x : xs) if (v == x) return true;
  return false;
}

// Extension registries (model.hpp register_*). Knobs that name an extension check their value here, so a block,
// input layer, head or loss name is declared only by its register_* line; an unregistered value is rejected.
enum class RegistryKind : uint8_t { Block = 0, Input = 1, Head = 2, Loss = 3 };
bool registry_has(RegistryKind k, const std::string& name);
std::vector<std::string> registry_names(RegistryKind k);
bool registry_has_all(RegistryKind k, const std::string& comma_list);   // "" is valid (no aux losses)
// Storage behind model.hpp register_block/input/head/loss (src/core/registry.cpp). The stage-1 baseline names
// (block "sandwich", input "embed", head "full") are registered by core with a null factory so the default config
// validates before src/model exists; src/model installs their factories through the same call (a null entry may be
// replaced once; any other duplicate is fatal). make_block() on a null factory is fatal.
void registry_add(RegistryKind k, const std::string& name, void* factory);
void* registry_factory(RegistryKind k, const std::string& name);       // nullptr if absent or not built yet

// ------------------------------------------------------------------------------------------------ the config
struct Config {
#define NM_KNOB(sec, name, type, def, flags, tag, check, doc) type sec##_##name = def;
#include "marincpp/knobs.def"
#undef NM_KNOB
};

// Knob metadata, generated from the same X-macro (parser, CLI, hash, label and `marincpp knobs` all use it).
enum class KnobType : uint8_t { Bool, Int, Int64, Double, String };
struct KnobInfo {
  const char* key;    // "model.hidden"
  KnobType type;
  uint32_t flags;     // KnobFlags
  const char* tag;    // label fragment: at most one %d (int), %g (double) or %s (string); checked at compile time
  const char* doc;
  std::string (*get_text)(const Config&);            // canonical text of the value (shortest round-trip for doubles)
  bool (*set_text)(Config&, std::string_view text);  // parse; false on a malformed value
  bool (*check)(const Config&);                      // the knob's own `check` expression
  bool (*is_default)(const Config&);
};
const std::vector<KnobInfo>& knob_table();
const KnobInfo* find_knob(std::string_view key);

// ------------------------------------------------------------------------------------------------ derived shape
// Everything downstream (planner, kernels, optimizer) reads these integers, never the raw knobs.
struct ModelShape {
  int V, D, L, H, KV, hd, F, S;
  int G;                 // H / KV query heads per kv head
  int qkv_cols;          // (H + 2 KV) * hd
  int n_readouts;        // entries of the readout table with their own parameters (+1 for the final head)
  int n_ce_heads;        // CE evaluations per token: 1, or L with per-layer readouts (shared or separate)
  int64_t n_params;      // exact; equals W&B parameter_count for the four baseline sizes (model.md §5)
  double flops_per_example;  // levanter formula (model.md §6) with n_ce_heads lm_head terms; the MFU key only
  float attn_scale_bf16;     // bf16(1/sqrt(hd)) as a float: 0.125 (hd 64), 0.08837890625 (hd 128); class M
};

// DESIGN.md §3.4, decided by E1b (O1); Undecided while exec.attn_bwd_sched = auto and O1 is open.
enum class AttnBwdSched : uint8_t { Persistent = 0, DocCausal = 1, Undecided = 255 };

struct ExecShape {
  int ranks;                 // exec.gpus
  int seqs_per_rank;         // global_batch / ranks (of the phase; DataPlan phases may change it, one regime each)
  int micro_seqs;            // resolved (planner choice if 0)
  int n_micro;               // seqs_per_rank / micro_seqs
  int64_t micro_tokens;      // micro_seqs * S
  int max_segs_per_seq;      // from the dataset meta (converter); sizes segment tables, never padding rows
  int ce_chunk_rows;         // resolved (§3.8 rule)
  bool zero1;                // ZeRO-1 ownership for every optimizer group
  bool zero2_vocab;          // reduce-scatter vocab-sized gradients per microbatch
  Remat remat;
  AttnBwdSched attn_bwd;
  Numerics num;              // policy, rounding, strict items (core.hpp)
  bool fused;                // stage-1 epilogue fusions (false: the unfused reference path)
};

// ------------------------------------------------------------------------------------------------ resolution
// apply_preset(size) -> [apply_preset("levanter_strict")] -> parse file(s) -> --set overrides -> resolve() -> validate().
// load_config runs the first four: the size is model.size from the overrides, else from the files, else "130m".
// `presets` takes only non-size presets (levanter_strict); a size name there is an error (sizes come from --size or
// model.size, so model.size always names the shape that was applied).
// Presets: "130m", "300m", "520m", "1_2b" (hyperparameter rows), "pls_d48", and "levanter_strict", which sets
// exec.attention=levref, exec.grad_reduce=bf16, exec.embed_accum=bf16_seq, exec.ce_dw_round=chunk,
// exec.ce_chunk_rows=8192, and at 520m / 1_2b exec.micro_seqs=32 with exec.remat=layer (DESIGN.md §3.0).
void apply_preset(Config& c, std::string_view name);
void load_toml(Config& c, const std::string& path);           // unknown keys are an error (R3: no config from env)
void apply_override(Config& c, std::string_view key_eq_value);  // "--set model.layers=48"
Config load_config(const std::vector<std::string>& toml_files, const std::vector<std::string>& presets,
                   const std::vector<std::string>& overrides);
// resolve: replaces every value that depends on something outside the config by its concrete value: the "auto" plan
// choices (exec.micro_seqs, exec.remat, exec.ce_chunk_rows, optim.zero1, optim.zero2_vocab from `plan_choice`),
// exec.ce_logits (T1.0 item 2 fact; without the facts file "auto" becomes bf16 except in parity runs, where it stays
// "auto" and validate() refuses), and file inputs that change numbers (init.fixture, optim.schedule_fixture,
// data.plan, readout.table, optim.groups, optim.row_partition, eval.extra_sources) by "<path>#sha256=<hex>" (a path
// that already carries a hash is verified; directories hash their sorted file list). data.dataset becomes
// "<name>#sha256=<meta.json sha256>" when `data_meta` is given, eval.paloma likewise from `eval_meta`. These are the
// NM_CONTENT knobs: the identity hashes only the "#sha256=" part, the path or name is local. Values that are pure functions of other hashed knobs
// (head_dim = 0, decay fractions) stay as written: ModelShape and the schedule code derive them, and the identity is
// the same either way. Knobs decided by H100 measurements (exec.ce, exec.attn_bwd_sched, optim.ns_precision) stay
// "auto" until their decision (DESIGN.md Open decisions) is recorded in code.
void resolve(Config& c, const struct DatasetMeta* data_meta, const struct DatasetMeta* eval_meta,
             const struct PlanChoice* plan_choice);
// validate: every knob's check, plus the cross-knob rules (each rule names its knobs in its message), e.g.
//   heads % kv_heads == 0; hidden == heads*head_dim unless head_dim set; global_batch % gpus == 0;
//   readout.local => readout.kind == separate; readout.detach_head => kind == shared; eval.batch % gpus == 0;
//   seq_len == dataset seq_len; vocab == dataset vocab; block in {hc, mhc} => hc_streams >= 2;
//   optim.weight_decay > 0 with a muonh_qwen3 preset => optim.wd_variant (levanter logs 0.1 but applies none);
//   run.parity => optim.schedule_fixture set and exec.ce_logits resolved from the T1.0 facts file;
//   exec.policy == f32 => exec.cuda_graph == false, exec.attention == ref_f32, exec.ce == a;
//   init.mode == threefry && run.parity => T1.1 recorded as passed for this binary (else use init.mode=fixture).
//   model.tie_embeddings => readout.kind != separate; optim.zero2_vocab != on (not implemented).
// Hashing (below) runs on the RESOLVED config, so "auto" values enter the identity as their concrete choice.
void validate(const Config& c);
ModelShape model_shape(const Config& c);
// Needs the resolved plan choices (fatal on "auto" / 0). max_segs_per_seq comes from `data_meta`, else
// kDefaultMaxSegsPerSeq (the planner bound of DESIGN.md §2.5 until S1 reads the dataset meta).
inline constexpr int kDefaultMaxSegsPerSeq = 24;
ExecShape exec_shape(const Config& c, const ModelShape& m, const struct DatasetMeta* data_meta = nullptr);

// f32 scalar constants of the optimizer and schedules, made as levanter makes them (optim.md §5.3 rule 1): a Python
// float expression is evaluated in f64 and rounded to f32 once. Gate T0.6 compares their bit patterns with the hex
// column of optim.md §5.3. The optimizer kernels (S6) and the schedule formula (S2) take their scalars from here.
struct OptimConstants {
  float b1, b2;                    // f32(beta1), f32(beta2)
  float one_minus_b1, one_minus_b2;  // f32(1 - beta) evaluated in f64 (not f32(1) - f32(beta))
  float momentum;                  // MuonH mu
  float eps, muon_eps;             // Adam/AdamH eps (outside the sqrt); NS input normalization eps
  float max_grad_norm;
  float ns_a, ns_b, ns_c;          // Newton-Schulz coefficients (optim.ns_coeffs = simple)
  float norm_floor;                // 1e-10 floor of the norm-preserving projection (optim.md §4.4)
  float lr_peak, adam_lr_peak;     // f32 of the decimal peaks
  float warmup_init_minus_end, adam_warmup_init_minus_end;  // f32(0 - peak): warmup (init - end)
  float cos_one_minus_alpha;       // f32(1 - min_lr_ratio): cosine (1 - alpha)
  float pi, half;                  // f32(jnp.pi), 0.5
};
OptimConstants optim_constants(const Config& c);

// ------------------------------------------------------------------------------------------------ identity
// canonical_json: every NM_HASH knob whose value differs from its default, sorted by key, values in canonical text
// (an NM_CONTENT knob that carries "#sha256=<hex>" contributes "sha256:<hex>" only), plus
// {"knob_table_version": kKnobTableVersion, "format": 2}. run_id: first 12 hex chars of its sha256.
//
// The RESOLVED config is the identity, and resolution depends on the binary: the planner formulas and buffer list
// (micro_seqs, remat, the CE chunk from the §12.1 estimates), and the "auto" decisions still open (exec.ce,
// exec.attn_bwd_sched, optim.ns_precision). The same config file can therefore resolve to another run_id after S1-S12
// add buffers or H100 numbers replace the estimates. Resume and fork must load the stored resolved config
// (config.resolved.json or the manifest) and must never re-resolve the original file.
std::string canonical_json(const Config& c);
std::string config_hash_hex(const Config& c);   // sha256 of canonical_json, 64 hex chars
std::string run_id(const Config& c);
// e.g. muonh-qwen3-130m-della4xh100-nm-s1: tags of the knobs that differ from the size preset. Call it on the config as
// written (before resolve), so automatic choices never appear in a name.
std::string default_label(const Config& c);
std::string full_json(const Config& c);         // every knob (hashed and local), for config.resolved.json and manifests
std::string full_toml(const Config& c);         // every knob as TOML sections; load_config(full_toml(c)) == c
std::string knob_defaults_json();               // key -> canonical default text, the kKnobTableVersion guard

}  // namespace nm
