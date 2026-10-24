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

enum KnobFlags : uint32_t { NM_HASH = 1u << 0, NM_LOCAL = 1u << 1 };

// Version of the knob table's defaults. Run identity hashes only hashed knobs that differ from their default, plus this
// version (DESIGN.md §9.2): adding a knob whose default keeps the old behaviour leaves every run id unchanged; CHANGING
// ANY DEFAULT MUST BUMP THIS NUMBER (a test compares the default table against tests/knob_defaults_v<N>.json).
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
  const char* tag;    // label fragment, printf-style
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
  int64_t n_params;      // exact; equals W&B parameter_count for the four baseline sizes (model.md §5)
  double flops_per_example;  // levanter formula (model.md §6), for the MFU key only
  float attn_scale_bf16;     // bf16(1/sqrt(hd)) as a float: 0.125 (hd 64), 0.08837890625 (hd 128); class M
};

enum class AttnBwdSched : uint8_t { Persistent = 0, DocCausal = 1 };   // DESIGN.md §3.4, decided by E1b (O1)

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
// Presets: "130m", "300m", "520m", "1_2b" (hyperparameter rows), "pls_d48", and "levanter_strict", which sets
// exec.attention=levref, exec.grad_reduce=bf16, exec.embed_accum=bf16_seq, exec.ce_dw_round=chunk,
// exec.ce_chunk_rows=8192, and at 520m / 1_2b exec.micro_seqs=32 with exec.remat=layer (DESIGN.md §3.0).
void apply_preset(Config& c, std::string_view name);
void load_toml(Config& c, const std::string& path);           // unknown keys are an error (R3: no config from env)
void apply_override(Config& c, std::string_view key_eq_value);  // "--set model.layers=48"
// resolve: fills derived values (head_dim, decay steps, "auto" plan choices, exec.ce_logits from the T1.0 facts file),
// replaces file paths that affect numbers (init.fixture, init.from, optim.schedule_fixture, data.plan, data.dataset,
// eval.paloma) by "<path>#sha256=<hex>".
void resolve(Config& c, const struct DatasetMeta* data_meta, const struct PlanChoice* plan_choice);
// validate: every knob's check, plus the cross-knob rules (each rule names its knobs in its message), e.g.
//   heads % kv_heads == 0; hidden == heads*head_dim unless head_dim set; global_batch % gpus == 0;
//   readout.local => readout.kind == separate; readout.detach_head => kind == shared; eval.batch % gpus == 0;
//   seq_len == dataset seq_len; vocab == dataset vocab; block in {hc, mhc} => hc_streams >= 2;
//   optim.weight_decay > 0 with a muonh_qwen3 preset => optim.wd_variant (levanter logs 0.1 but applies none);
//   run.parity => optim.schedule_fixture set and exec.ce_logits resolved from the T1.0 facts file;
//   exec.policy == f32 => exec.cuda_graph == false, exec.attention == ref_f32, exec.ce == a;
//   init.mode == threefry && run.parity => T1.1 recorded as passed for this binary (else use init.mode=fixture).
// Hashing (below) runs on the RESOLVED config, so "auto" values enter the identity as their concrete choice.
void validate(const Config& c);
ModelShape model_shape(const Config& c);
ExecShape exec_shape(const Config& c, const ModelShape& m);

// ------------------------------------------------------------------------------------------------ identity
// canonical_json: every NM_HASH knob whose value differs from its default, sorted by key, values in canonical text,
// plus {"knob_table_version": kKnobTableVersion, "format": 2}. run_id: first 12 hex chars of its sha256.
std::string canonical_json(const Config& c);
std::string config_hash_hex(const Config& c);   // sha256 of canonical_json, 64 hex chars
std::string run_id(const Config& c);
std::string default_label(const Config& c);     // e.g. muonh-qwen3-130m-della4xh100-nm (+ tags of non-default knobs)
std::string full_json(const Config& c);         // every knob (hashed and local), for config.resolved.json and manifests

}  // namespace nm
