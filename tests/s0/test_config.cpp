// s0_config: knob table, TOML subset, load order, validation rules, resolution, run identity, labels, defaults guard.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>

#include "config/text.hpp"
#include "marincpp/config.hpp"
#include "marincpp/data.hpp"
#include "marincpp/memory.hpp"
#include "nmtest.hpp"

using namespace nm;
namespace fs = std::filesystem;

namespace {

const char* kSizes[] = {"130m", "300m", "520m", "1_2b", "pls_d48"};

std::string write_tmp(const std::string& name, const std::string& text) {
  fs::path p = fs::current_path() / ("s0_tmp_" + name);
  std::ofstream(p, std::ios::binary) << text;
  return p.string();
}

// The CLI's resolution flow (src/main.cpp resolve_all).
Config resolved(const std::vector<std::string>& sets, const std::vector<std::string>& presets = {}) {
  Config c = load_config({}, presets, sets);
  validate(c);
  ModelShape m = model_shape(c);
  DatasetMeta d;
  d.seq_len = m.S;
  d.vocab = m.V;
  d.max_segs_per_seq = kDefaultMaxSegsPerSeq;
  std::string rep;
  PlanChoice pc = choose_plan(c, m, d, &rep);
  NM_REQUIRE(pc.micro_seqs > 0, rep);
  resolve(c, nullptr, nullptr, &pc);
  validate(c);
  return c;
}

bool same_knobs(const Config& a, const Config& b, std::string* diff) {
  for (const KnobInfo& k : knob_table())
    if (k.get_text(a) != k.get_text(b)) {
      *diff = std::string(k.key) + ": " + k.get_text(a) + " vs " + k.get_text(b);
      return false;
    }
  return true;
}

}  // namespace

NM_TEST(s0_config, knob_table_is_well_formed) {
  std::set<std::string> keys;
  Config d;
  for (const KnobInfo& k : knob_table()) {
    std::string key = k.key;
    CHECK(keys.insert(key).second);
    CHECK(std::count(key.begin(), key.end(), '.') == 1);
    CHECK(((k.flags & NM_HASH) != 0) != ((k.flags & NM_LOCAL) != 0));
    CHECK(k.is_default(d));
    if (!k.check(d)) nmtest::fail(__FILE__, __LINE__, "default of " + key + " fails its own check");
    CHECK(find_knob(key) == &k);
    if (k.flags & NM_CONTENT) CHECK((k.flags & NM_HASH) && k.type == KnobType::String);   // file inputs are hashed
  }
  CHECK(find_knob("model.nonexistent") == nullptr);
  validate(d);
  for (const char* s : kSizes) validate(load_config({}, {}, {std::string("model.size=") + s}));
}

NM_TEST(s0_config, knob_defaults_guard) {
  // CHANGING A DEFAULT MUST BUMP kKnobTableVersion and add tests/s0/knob_defaults_v<N>.json (config.hpp).
  std::string path = nmtest::src_dir() + "/tests/s0/knob_defaults_v" + std::to_string(kKnobTableVersion) + ".json";
  std::string want = nmtest::read_file(path);
  if (want.empty()) nmtest::fail(__FILE__, __LINE__, "missing " + path + " (marincpp-train knobs --defaults-json)");
  else if (want != knob_defaults_json())
    nmtest::fail(__FILE__, __LINE__, "knob defaults differ from " + path +
                                         ": a default changed without bumping kKnobTableVersion");
}

NM_TEST(s0_config, toml_subset) {
  auto e = parse_toml(
      "# comment\nmodel.layers = 48   # trailing\n[optim]\nlr = 1e-3\nnum_steps = 1_000\nschedule = \"cosine\"\n"
      "[run]\nlabel = 'lit \"q\"'\nparity = false\n",
      "t");
  CHECK_EQ(e.size(), size_t(6));
  CHECK_EQ(e[0].key, std::string("model.layers"));
  CHECK(e[0].kind == TomlEntry::Int);
  CHECK_EQ(e[1].key, std::string("optim.lr"));
  CHECK(e[1].kind == TomlEntry::Float);
  CHECK_EQ(e[2].text, std::string("1000"));
  CHECK_EQ(e[3].text, std::string("cosine"));
  CHECK_EQ(e[4].text, std::string("lit \"q\""));
  CHECK(e[5].kind == TomlEntry::Bool);
  CHECK_THROWS(parse_toml("a.b = 1\na.b = 2\n", "t"), "t:2: duplicate key a.b");
  CHECK_THROWS(parse_toml("a.b = [1, 2]\n", "t"), "not supported");
  CHECK_THROWS(parse_toml("a.b = \"open\n", "t"), "unterminated string");
  CHECK_THROWS(parse_toml("a.b = inf\n", "t"), "non-finite");
  CHECK_THROWS(parse_toml("a.b = 1 2\n", "t"), "unexpected text");
  CHECK_THROWS(parse_toml("[[x]]\n", "t"), "arrays of tables");
  // TOML 1.0 numbers: '_' only between digits, no leading zeros (a typo must not become another value)
  CHECK_THROWS(parse_toml("model.layers = 0_1_2\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("optim.lr = 1e_2\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 012\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 1__0\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 10_\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = _10\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 1.\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = .5\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 1._5\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = 0x10\n", "t"), "malformed value");
  CHECK_THROWS(parse_toml("a.b = -nan\n", "t"), "non-finite");
  CHECK_THROWS(parse_toml("a.b = 99999999999999999999\n", "t"), "out of range");
  auto n = parse_toml("a.i = -0\na.j = +1_000_000\na.f = 6.626e-34\na.g = 1e06\na.h = -0.0\na.k = 3.141_592\n", "t");
  CHECK_EQ(n[1].text, std::string("+1000000"));
  CHECK(n[2].kind == TomlEntry::Float);
  CHECK_EQ(n[3].text, std::string("1e06"));   // exponent digits may lead with 0
  CHECK_EQ(n[5].text, std::string("3.141592"));
  // every table and key defined once
  CHECK_THROWS(parse_toml("[optim]\nlr = 1.0\n[optim]\nwarmup = 1\n", "t"), "t:3: duplicate table [optim]");
  CHECK_THROWS(parse_toml("model.layers = 1\n[model]\nhidden = 512\n", "t"), "already defined by dotted keys");
  CHECK_THROWS(parse_toml("a = 1\na.b = 2\n", "t"), "extends the value a");
  // strings: every TOML escape; no raw control characters
  auto e2 = parse_toml("a.b = \"x\\b\\f\\u0001\\u00e9\\U0001F600\\u007f\"\n", "t");
  CHECK_EQ(e2[0].text, std::string("x\b\f\x01\xc3\xa9\xf0\x9f\x98\x80\x7f"));
  CHECK_THROWS(parse_toml("a.b = \"\\ud800\"\n", "t"), "not a Unicode scalar value");
  CHECK_THROWS(parse_toml("a.b = \"\\u12zz\"\n", "t"), "bad hex digit");
  CHECK_THROWS(parse_toml("a.b = \"\\u12\"\n", "t"), "short");
  CHECK_THROWS(parse_toml("a.b = \"\\x41\"\n", "t"), "unsupported escape");
  CHECK_THROWS(parse_toml(std::string("a.b = \"x\x01y\"\n"), "t"), "control character");
  CHECK_THROWS(parse_toml(std::string("a.b = 'x\x7fy'\n"), "t"), "control character");
  CHECK_EQ(parse_toml("a.b = \"tab\there\"\n", "t")[0].text, std::string("tab\there"));   // raw tab is allowed
}

NM_TEST(s0_config, load_order_and_type_errors) {
  std::string f = write_tmp("a.toml", "[model]\nsize = \"300m\"\nlayers = 10\n[optim]\nlr = 0.5\n");
  Config c = load_config({f}, {}, {});
  CHECK_EQ(c.model_hidden, 768);   // the size preset applied first
  CHECK_EQ(c.model_layers, 10);    // then the file
  CHECK_EQ(c.optim_lr, 0.5);
  Config o = load_config({f}, {}, {"model.layers=7", "optim.schedule='linear'", "optim.warmup=0"});
  CHECK_EQ(o.model_layers, 7);     // then --set
  CHECK_EQ(o.optim_schedule, std::string("linear"));
  Config s = load_config({}, {"levanter_strict"}, {"model.size=520m"});
  CHECK_EQ(s.exec_remat, std::string("layer"));
  CHECK_EQ(s.exec_micro_seqs, 32);
  CHECK_EQ(s.exec_attention, std::string("levref"));
  CHECK_EQ(s.exec_ce_chunk_rows, 8192);
  Config s130 = load_config({}, {"levanter_strict"}, {});
  CHECK_EQ(s130.exec_remat, std::string("auto"));   // 130m keeps 1 x 32 (levanter's split)
  CHECK_THROWS(load_config({write_tmp("b.toml", "model.layers = \"six\"\n")}, {}, {}), "needs a int value");
  CHECK_THROWS(load_config({write_tmp("c.toml", "model.color = 1\n")}, {}, {}), "unknown knob model.color");
  CHECK_THROWS(load_config({}, {}, {"model.layers=6.5"}), "needs a int value");
  CHECK_THROWS(load_config({}, {}, {"model.layers"}), "key=value");
  CHECK_THROWS(load_config({}, {"nope"}, {}), "unknown preset");
  // sizes: quoted --set values, and never through --preset (model.size must name the applied shape)
  CHECK_EQ(load_config({}, {}, {"model.size=\"300m\""}).model_hidden, 768);
  CHECK_EQ(load_config({}, {}, {"model.size='520m'"}).model_hidden, 1024);
  CHECK_THROWS(load_config({}, {"300m"}, {"model.size=520m"}), "is a model size");
  CHECK_THROWS(load_config({}, {"pls_d48"}, {}), "is a model size");
  CHECK_THROWS(load_config({}, {}, {"model.size=3000m"}), "is not a size");
  Config d = load_config({write_tmp("d.toml", "optim.lr = 1\n")}, {}, {});   // an int is a valid double
  CHECK_EQ(d.optim_lr, 1.0);
}

NM_TEST(s0_config, toml_round_trip) {
  for (const char* s : kSizes)
    for (bool strict : {false, true}) {
      Config c = strict ? load_config({}, {"levanter_strict"}, {std::string("model.size=") + s})
                        : resolved({std::string("model.size=") + s});
      c.run_label = "with \"quotes\" and \\ backslash";
      c.run_wandb_name = std::string("ctl \x01\x02\b\f\x1f del \x7f tab\t nl\n cr\r utf8 \xc3\xa9 end");
      c.optim_eps = 1e-15;
      std::string f = write_tmp("rt.toml", full_toml(c));
      Config back = load_config({f}, {}, {});
      std::string diff;
      if (!same_knobs(c, back, &diff)) nmtest::fail(__FILE__, __LINE__, std::string(s) + " round trip: " + diff);
    }
}

NM_TEST(s0_config, validation_rules) {
  auto bad = [](std::vector<std::string> sets, const char* expect) {
    Config c = load_config({}, {}, sets);
    CHECK_THROWS(validate(c), expect);
  };
  bad({"model.kv_heads=3"}, "model.heads must be a multiple of model.kv_heads");
  bad({"model.heads=16"}, "model.head_dim = 0 needs");                      // 512 / 16 = 32
  bad({"data.global_batch=130"}, "data.global_batch must be a multiple of exec.gpus");
  bad({"eval.batch=126"}, "eval.batch must be a multiple of exec.gpus");
  bad({"exec.micro_seqs=5"}, "exec.micro_seqs must divide");
  bad({"readout.local=true"}, "readout.local needs readout.kind = separate");
  bad({"readout.detach_head=true"}, "readout.detach_head needs readout.kind = shared");
  bad({"readout.detach_backbone=true"}, "readout.detach_backbone needs");
  bad({"readout.on=10"}, "readout.on must be S:E");
  bad({"optim.weight_decay=0.1"}, "optim.wd_variant");
  bad({"run.parity=true"}, "run.parity needs optim.schedule_fixture");
  bad({"run.parity=true"}, "init.mode = threefry needs T1.1");
  bad({"init.mode=fixture"}, "init.mode = fixture needs init.fixture");
  bad({"exec.policy=f32"}, "exec.policy = f32 needs exec.cuda_graph = false");
  bad({"model.block=moda"}, "model.block = 'moda' is not allowed");      // registry: no kernel yet
  bad({"model.losses=zloss"}, "model.losses");
  bad({"exec.gpus=9"}, "exec.gpus");
  bad({"model.head_dim=96"}, "model.head_dim");
  bad({"optim.ns_coeffs=quintic"}, "only 'simple' is implemented");
  bad({"model.size=pls_d48", "model.tie_embeddings=true"}, "model.tie_embeddings needs readout.kind != separate");
  bad({"optim.zero2_vocab=on"}, "optim.zero2_vocab = on is not implemented");
  bad({"model.hidden=0"}, "model.hidden");                       // ranges keep planner products exact
  bad({"model.mlp=2097152"}, "model.mlp");
  bad({"data.global_batch=2097152"}, "data.global_batch");
  validate(load_config({}, {}, {"model.tie_embeddings=true", "readout.kind=shared"}));
  // several problems are reported together
  Config c = load_config({}, {}, {"model.kv_heads=3", "optim.weight_decay=0.1"});
  CHECK_THROWS(validate(c), "2 problems");
  validate(load_config({}, {}, {"optim.weight_decay=0.1", "optim.wd_variant=true"}));
  validate(load_config({}, {}, {"exec.policy=f32", "exec.cuda_graph=false", "exec.attention=ref_f32", "exec.ce=a"}));
}

NM_TEST(s0_config, identity) {
  Config d;
  CHECK_EQ(canonical_json(d), std::string("{\"format\":2,\"knob_table_version\":1,\"knobs\":{}}"));
  Config local = d;
  local.run_label = "x";
  local.exec_mem_budget_gib = 60;
  local.data_reader_threads = 32;
  CHECK_EQ(run_id(local), run_id(d));                    // local knobs never enter the identity
  Config seed = d;
  seed.init_seed = 1;
  CHECK(run_id(seed) != run_id(d));
  CHECK_EQ(canonical_json(seed), std::string("{\"format\":2,\"knob_table_version\":1,\"knobs\":{\"init.seed\":1}}"));
  Config two = d;
  two.optim_lr = 0.01;
  two.model_layers = 12;   // keys sorted: model.* before optim.*
  CHECK_EQ(canonical_json(two),
           std::string("{\"format\":2,\"knob_table_version\":1,\"knobs\":{\"model.layers\":12,\"optim.lr\":0.01}}"));
  CHECK_EQ(config_hash_hex(two).size(), size_t(64));
  CHECK_EQ(run_id(two), config_hash_hex(two).substr(0, 12));
  // The resolved 130m baseline: pinned so an accidental change of the canonical form or of the planner shows up.
  Config r = resolved({});
  CHECK_EQ(canonical_json(r),
           std::string("{\"format\":2,\"knob_table_version\":1,\"knobs\":{\"exec.ce_chunk_rows\":32768,"
                       "\"exec.ce_logits\":\"bf16\",\"exec.micro_seqs\":32,\"exec.remat\":\"all\","
                       "\"optim.zero1\":\"on\",\"optim.zero2_vocab\":\"off\"}}"));
  CHECK_EQ(run_id(r), sha256_hex(canonical_json(r)).substr(0, 12));
  // file inputs enter by content: the path spelling is local
  std::string f = write_tmp("ident_plan.toml", "phase = 1\n");
  Config rel = load_config({}, {}, {"data.plan=s0_tmp_ident_plan.toml"});    // tests run in the binary dir
  Config abs = load_config({}, {}, {"data.plan=" + f});
  resolve(rel, nullptr, nullptr, nullptr);
  resolve(abs, nullptr, nullptr, nullptr);
  CHECK(rel.data_plan != abs.data_plan);                     // the stored values keep each path
  CHECK_EQ(run_id(rel), run_id(abs));
  CHECK(canonical_json(abs).find("\"data.plan\":\"sha256:" + sha256_file_hex(f) + "\"") != std::string::npos);
  CHECK(canonical_json(abs).find(f) == std::string::npos);
  std::ofstream(f, std::ios::binary) << "phase = 2\n";
  Config other = load_config({}, {}, {"data.plan=" + f});
  resolve(other, nullptr, nullptr, nullptr);
  CHECK(run_id(other) != run_id(abs));                       // other content, other identity
}

NM_TEST(s0_config, labels) {
  CHECK_EQ(default_label(Config{}), std::string("muonh-qwen3-130m-della4xh100-nm"));
  CHECK_EQ(default_label(load_config({}, {}, {"init.seed=1"})), std::string("muonh-qwen3-130m-della4xh100-nm-s1"));
  CHECK_EQ(default_label(load_config({}, {}, {"model.size=300m"})), std::string("muonh-qwen3-300m-della4xh100-nm"));
  CHECK_EQ(default_label(load_config({}, {}, {"model.size=1_2b", "exec.policy=f32", "optim.lr=0.02"})),
           std::string("muonh-qwen3-1_2b-della4xh100-nm-lr0.02-f32"));
  CHECK_EQ(default_label(load_config({}, {}, {"model.tie_embeddings=true"})),
           std::string("muonh-qwen3-130m-della4xh100-nm-tie"));
  CHECK_EQ(default_label(load_config({}, {}, {"model.size=pls_d48"})), std::string("muonh-qwen3-pls_d48-della4xh100-nm"));
}

NM_TEST(s0_config, resolve_hashes_file_inputs) {
  std::string f = write_tmp("plan.toml", "phase = 1\n");
  Config c = load_config({}, {}, {"data.plan=" + f});
  resolve(c, nullptr, nullptr, nullptr);
  CHECK_EQ(c.data_plan, f + "#sha256=" + sha256_file_hex(f));
  Config again = c;
  resolve(again, nullptr, nullptr, nullptr);                        // a recorded hash is verified and kept
  CHECK_EQ(again.data_plan, c.data_plan);
  std::ofstream(f, std::ios::binary) << "phase = 2\n";
  Config changed = c;
  CHECK_THROWS(resolve(changed, nullptr, nullptr, nullptr), "changed");
  Config missing = load_config({}, {}, {"init.fixture=/nonexistent/fixture.npz"});
  CHECK_THROWS(resolve(missing, nullptr, nullptr, nullptr), "file not found");
  // directories hash their sorted file list; the result does not depend on creation order
  fs::path dir = fs::current_path() / "s0_tmp_sched";
  fs::remove_all(dir);
  fs::create_directories(dir / "sub");
  std::ofstream(dir / "b.f32") << "B";
  std::ofstream(dir / "sub" / "a.f32") << "A";
  Config s = load_config({}, {}, {"optim.schedule_fixture=" + dir.string()});
  resolve(s, nullptr, nullptr, nullptr);
  std::string listing = std::string("b.f32") + '\0' + sha256_hex("B") + "\n" + "sub/a.f32" + '\0' + sha256_hex("A") + "\n";
  CHECK_EQ(s.optim_schedule_fixture, dir.string() + "#sha256=" + sha256_hex(listing));
  // dataset identity from its meta.json sha256
  DatasetMeta meta;
  meta.meta_sha256 = std::string(64, 'a');
  Config dm;
  DatasetMeta pal;
  pal.meta_sha256 = std::string(64, 'b');
  resolve(dm, &meta, &pal, nullptr);
  CHECK_EQ(dm.data_dataset, std::string("fineweb-edu-10B-marin128k#sha256=") + std::string(64, 'a'));
  CHECK_EQ(dm.eval_paloma, std::string("paloma-marin128k#sha256=") + std::string(64, 'b'));
  CHECK(canonical_json(dm).find("\"eval.paloma\":\"sha256:" + std::string(64, 'b') + "\"") != std::string::npos);
  // ce_logits: auto -> bf16 outside parity runs; parity runs keep auto (validate refuses until the T1.0 facts exist)
  Config par = load_config({}, {}, {"run.parity=true"});
  resolve(par, nullptr, nullptr, nullptr);
  CHECK_EQ(par.exec_ce_logits, std::string("auto"));
}

NM_TEST(s0_config, shapes) {
  ModelShape m = model_shape(Config{});
  CHECK_EQ(m.hd, 64);
  CHECK_EQ(m.attn_scale_bf16, 0.125f);
  CHECK_EQ(m.qkv_cols, 1536);
  CHECK_EQ(m.n_readouts, 1);
  ModelShape b = model_shape(load_config({}, {}, {"model.size=1_2b"}));
  CHECK_EQ(b.hd, 128);
  CHECK_EQ(b.G, 2);
  CHECK_EQ(b.attn_scale_bf16, 0.08837890625f);
  ModelShape p = model_shape(load_config({}, {}, {"model.size=pls_d48"}));
  CHECK_EQ(p.n_readouts, 48);
  CHECK_EQ(p.n_ce_heads, 48);
  CHECK_THROWS(exec_shape(Config{}, m), "unresolved");
  Config r = resolved({});
  ExecShape e = exec_shape(r, m);
  CHECK_EQ(e.micro_seqs, 32);
  CHECK_EQ(e.n_micro, 1);
  CHECK_EQ(e.micro_tokens, int64_t(131072));
  CHECK(e.remat == Remat::All);
  CHECK(e.attn_bwd == AttnBwdSched::Undecided);
  CHECK_EQ(e.max_segs_per_seq, kDefaultMaxSegsPerSeq);
}
