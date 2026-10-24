// S1 data plans: default plan, Repeat (levanter's max_train_batches restart), reshuffle, Explicit, pass boundaries, fork,
// plan files, canonical form, times_seen. Real N = 2,441,406 through a registered meta (no token file needed).
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#include "marincpp/config.hpp"
#include "s1_util.hpp"

using namespace nm;

namespace {
const char* kFake = "fake-fineweb";   // N = 2,441,406 windows of 4096, like the real set
void fake() { s1::register_fake(kFake, 10000000738ull, 4096); }

Config cfg(int64_t steps = 4959, int B = 128) {
  Config c;
  c.data_dataset = kFake;
  c.optim_num_steps = static_cast<int>(steps);
  c.data_global_batch = B;
  return c;
}
}  // namespace

NM_TEST(s1_plan, default_plan_is_levanter_order) {
  fake();
  const Config c = cfg();
  const DataPlan p = load_data_plan("", c);
  CHECK_EQ(p.phases.size(), size_t(1));
  LevanterOrder o(42, 2441406, 2048);
  for (int64_t k : {0, 1, 2, 1000, 4958})
    for (int r = 0; r < 128; r += 7) CHECK_EQ(p.seq_of(k, r), o(uint64_t(k) * 128 + r));
  CHECK_THROWS(p.seq_of(4959, 0), "outside the data plan");
  CHECK_THROWS(p.seq_of(0, 128), "row outside");
  // 130m reads 0.26 passes: no pass boundary; 1_2b (B 256, 22888 steps): passes 2 and 3 start at 9536 and 19072
  CHECK(p.pass_boundaries().empty());
  const DataPlan p12 = load_data_plan("", cfg(22888, 256));
  CHECK(p12.pass_boundaries() == std::vector<int64_t>({9536, 19072}));
  CHECK(load_data_plan("", cfg(9918, 256)).pass_boundaries() == std::vector<int64_t>({9536}));
  Config bad = cfg();
  bad.data_mix_block = 1024;
  CHECK_THROWS(load_data_plan("", bad), "only 2048");
  Config odd = cfg(100, 130);
  CHECK_THROWS(load_data_plan("", odd), "global_batch % exec.gpus");
}

// Repeat without reshuffle = levanter with max_train_batches = subset / B: the LCG-permuted set is sliced to its first
// `subset` sequences, then the mixture wraps mod subset (datasets.py:858-866, mixture.py restart).
NM_TEST(s1_plan, repeat_is_levanter_slice_restart) {
  fake();
  const uint64_t N = 2441406, subset = 4768 * 128;   // DATA_EPOCHS = 2.4 at 300m: 11444 / 2.4 = 4768 batches
  DataPlan p;
  Phase ph = s1::levanter_phase(kFake, 0, 11444, 128);
  ph.kind = OrderKind::Repeat;
  ph.subset = subset;
  ph.passes = 11444.0 / 4768.0;
  p.phases.push_back(ph);
  data::validate_plan(p, 4);
  const data::Lcg l = data::lcg_for(42, N);
  const auto pi_of = [&](uint64_t blk) { return data::block_permutation(split(prng_key(42), 0), blk, 2048); };
  for (int64_t k = 0; k < 11444; k += 13)
    for (int r = 0; r < 128; r += 31) {
      const uint64_t j = uint64_t(k) * 128 + r, blk = j / 2048;
      const uint64_t rr = (blk * 2048 + pi_of(blk)[j % 2048]) % subset;
      CHECK_EQ(p.seq_of(k, r), (l.a * rr + l.b) % N);
    }
  // inside the first subset (blocks below the wrap) it equals the plain levanter order
  LevanterOrder o(42, N, 2048);
  for (uint64_t j = 0; j < (subset / 2048) * 2048; j += 4099) CHECK_EQ(p.seq_of(int64_t(j / 128), int(j % 128)), o(j));
  // pass q starts in block floor(q * subset / 2048)
  const auto pb = p.pass_boundaries();
  CHECK(pb == std::vector<int64_t>({int64_t(subset / 2048 * 2048 / 128), int64_t(2 * subset / 2048 * 2048 / 128)}));
  // subset = N reproduces the Levanter phase exactly
  DataPlan full = p;
  full.phases[0].subset = N;
  full.phases[0].step_end = 5859328 / 128;
  for (uint64_t j = 0; j < 5859328; j += 104729) CHECK_EQ(full.seq_of(int64_t(j / 128), int(j % 128)), o(j));
}

NM_TEST(s1_plan, repeat_reshuffle_passes) {
  fake();
  const uint64_t N = 2441406, subset = 1024 * 10;
  DataPlan p;
  Phase ph = s1::levanter_phase(kFake, 0, 480, 64);
  ph.kind = OrderKind::Repeat;
  ph.subset = subset;
  ph.reshuffle = true;
  p.phases.push_back(ph);
  data::validate_plan(p, 4);
  const data::Lcg l = data::lcg_for(42, N);
  std::set<uint64_t> base;
  for (uint64_t i = 0; i < subset; ++i) base.insert((l.a * i + l.b) % N);
  // passes 1 and 2 each read every sequence of the subset exactly once, in different orders
  std::vector<uint64_t> p1, p2;
  for (uint64_t m = subset; m < 3 * subset; ++m) {
    const uint64_t s = p.seq_of(int64_t(m / 64), int(m % 64));
    (m < 2 * subset ? p1 : p2).push_back(s);
  }
  CHECK(std::set<uint64_t>(p1.begin(), p1.end()) == base);
  CHECK(std::set<uint64_t>(p2.begin(), p2.end()) == base);
  CHECK(p1 != p2);
  // pass 0 is the non-reshuffled order
  Phase plain = ph;
  plain.reshuffle = false;
  DataPlan q;
  q.phases.push_back(plain);
  for (uint64_t m = 0; m < subset; m += 17) CHECK_EQ(p.seq_of(int64_t(m / 64), int(m % 64)), q.seq_of(int64_t(m / 64), int(m % 64)));
  CHECK(p.pass_boundaries() == std::vector<int64_t>({160, 320}));
}

NM_TEST(s1_plan, explicit_and_plan_file_and_fork) {
  fake();
  const std::string dir = s1::temp_dir("plan");
  std::vector<uint32_t> ids(64 * 100);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<uint32_t>((i * 7919) % 2441406);
  s1::write_vec(dir + "/ids.u32", ids);
  {
    std::ofstream f(dir + "/plan.toml");
    f << "# three phases: the levanter stream, a batch change that continues it, then an explicit list\n"
         "[phase.0]\nstep_end = 50\nglobal_batch = 128\n"
         "[phase.1]\nstep_end = 80\nglobal_batch = 256\n"
         "[phase.2]\nkind = \"explicit\"\nexplicit_file = \"ids.u32\"\nglobal_batch = 64\nstep_end = 180\n";
  }
  Config c = cfg(180);
  const DataPlan p = load_data_plan(dir + "/plan.toml", c);
  CHECK_EQ(p.phases.size(), size_t(3));
  CHECK_EQ(p.phases[1].index_base, uint64_t(50 * 128));   // continues the stream
  LevanterOrder o(42, 2441406, 2048);
  CHECK_EQ(p.seq_of(49, 127), o(49 * 128 + 127));
  CHECK_EQ(p.seq_of(50, 0), o(50 * 128));
  CHECK_EQ(p.seq_of(79, 255), o(50 * 128 + 29 * 256 + 255));
  CHECK_EQ(p.seq_of(80, 0), uint64_t(ids[0]));
  CHECK_EQ(p.seq_of(179, 63), uint64_t(ids[99 * 64 + 63]));
  // the resolved form carries the content hash; a changed file is refused
  const std::string pinned = dir + "/plan.toml#sha256=" + sha256_file_hex(dir + "/plan.toml");
  CHECK_EQ(load_data_plan(pinned, c).canonical(), p.canonical());
  CHECK_THROWS(load_data_plan(dir + "/plan.toml#sha256=00", c), "the config pins");
  // canonical: stable, sensitive to the order inputs, independent of the explicit file's path
  CHECK(p.canonical().find("\"explicit\":\"sha256:") != std::string::npos);
  DataPlan p2 = p;
  p2.phases[1].seed = 43;
  CHECK(p2.canonical() != p.canonical());
  // fork at 60: batches before 60 are the parent's; the appended phase starts at 60
  Phase tail = s1::levanter_phase(kFake, 60, 180, 128);
  tail.kind = OrderKind::Repeat;
  tail.subset = 4096;
  tail.index_base = 0;
  const DataPlan fk = data::fork_plan(p, 60, {tail});
  data::validate_plan(fk, 4);
  for (int64_t k = 0; k < 60; ++k) CHECK_EQ(fk.seq_of(k, 0), p.seq_of(k, 0));
  CHECK_EQ(fk.phases.size(), size_t(3));
  CHECK_EQ(fk.phases[1].step_end, int64_t(60));
  CHECK_THROWS(data::fork_plan(p, 60, {s1::levanter_phase(kFake, 61, 180, 128)}), "must start at the fork step");
  // malformed files: gaps, unknown keys, short explicit file
  auto bad_plan = [&](const char* text) {
    std::ofstream(dir + "/bad.toml") << text;
    return load_data_plan(dir + "/bad.toml", c);
  };
  CHECK_THROWS(bad_plan("[phase.0]\nstep_end = 50\n[phase.2]\nstep_end = 180\n"), "without gaps");
  CHECK_THROWS(bad_plan("[phase.0]\nstep_end = 180\nbatch = 3\n"), "unknown phase key");
  CHECK_THROWS(bad_plan("[phase.0]\nstep_end = 170\n"), "ends at step 170");
  CHECK_THROWS(bad_plan("[phase.0]\nstep_begin = 5\nstep_end = 180\n"), "contiguous from step 0");
  CHECK_THROWS(bad_plan("[phase.0]\nkind = \"explicit\"\nexplicit_file = \"ids.u32\"\nstep_end = 180\n"),
               "explicit file holds 6400 ids");
  CHECK_THROWS(bad_plan("[phase.0]\nkind = \"repeat\"\nstep_end = 180\n"), "0 < subset <= N");
  CHECK_THROWS(bad_plan("[phase.0]\nseed = 7\nstep_end = 180\n"), "no LCG constants for data_seed 7");
  s1::remove_dir(dir);
}

// times_seen counts documents through docs.u64; window_reads counts windows. Synthetic set: 64 windows of 256 tokens,
// documents of 100 tokens; a Repeat plan over 16 windows for 3 passes.
NM_TEST(s1_plan, times_seen_documents) {
  const uint64_t n_tok = 64 * 256 + 37;
  // N = 64 windows; LCG (a, b) = (5, 3) (gcd(5, 64) = 1)
  s1::Synthetic d = s1::make_synthetic("synth-ts", n_tok, 256, 100, 42, 5, 3);
  DatasetMeta m = load_dataset_meta(d.dir);
  CHECK_EQ(m.name, std::string("synth-ts"));
  DataPlan p;
  Phase ph = s1::levanter_phase(d.dir, 0, 12, 4, 256);
  ph.kind = OrderKind::Repeat;
  ph.subset = 16;
  p.phases.push_back(ph);
  data::validate_plan(p, 4);
  const auto reads = data::window_reads(p, d.dir, 256, 12);
  uint64_t total = 0;
  for (uint32_t r : reads) total += r;
  CHECK_EQ(total, uint64_t(48));
  // brute force: a document's count = max over its windows of the window's reads
  for (uint64_t doc = 0; doc + 1 < d.docs.size(); ++doc) {
    const uint64_t a = d.docs[doc], b = d.docs[doc + 1];
    uint32_t want = 0;
    if (a < 64 * 256)
      for (uint64_t w = a / 256; w <= std::min<uint64_t>((b - 1) / 256, 63); ++w) want = std::max(want, reads[w]);
    CHECK_EQ(p.times_seen(doc, 12), want);
  }
  CHECK_EQ(p.times_seen(0, 0), 0u);
  s1::remove_dir(d.dir);
}

// Content identity (DESIGN.md §9.2): the default phase keeps the config's "#sha256=" dataset pin, a wrong pin is
// refused even when the dataset was loaded before, canonical() names every dataset by its meta.json sha256 (pinned or
// not, under any path), and an explicit file's pin is checked against its content.
NM_TEST(s1_plan, datasets_and_explicit_files_enter_by_content) {
  const uint64_t n_tok = 64 * 256 + 37;
  s1::Synthetic d = s1::make_synthetic("synth-pin", n_tok, 256, 100, 42, 5, 3);   // N = 64, LCG (5, 3)
  const std::string sha = sha256_file_hex(d.dir + "/meta.json");
  const std::string dir2 = s1::temp_dir("pin_copy");
  for (const char* f : {"meta.json", "tokens.u32", "docs.u64"})
    std::filesystem::copy_file(d.dir + "/" + f, dir2 + "/" + f, std::filesystem::copy_options::overwrite_existing);
  std::vector<uint32_t> ids(4 * 20);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<uint32_t>((i * 7) % 64);
  s1::write_vec(dir2 + "/ids.u32", ids);
  const std::string ids_sha = sha256_file_hex(dir2 + "/ids.u32");
  Config c;
  c.model_seq_len = 256;
  c.data_global_batch = 4;
  c.optim_num_steps = 40;
  c.data_dataset = d.dir + "#sha256=" + sha;   // the resolved form
  auto write_plan = [&](const std::string& text) {
    std::ofstream(dir2 + "/plan.toml") << text;
    return dir2 + "/plan.toml";
  };
  // default phase: pinned dataset of the config; canonical names it by content
  const DataPlan p = load_data_plan(write_plan("[phase.0]\nstep_end = 40\n"), c);
  CHECK_EQ(p.phases[0].dataset, c.data_dataset);
  CHECK(p.canonical().find("\"dataset\":\"sha256:" + sha + "\"") != std::string::npos);
  // a wrong pin is refused, although the dataset is registered by now
  Config wrong = c;
  wrong.data_dataset = d.dir + "#sha256=" + std::string(64, '0');
  CHECK_THROWS(load_data_plan(write_plan("[phase.0]\nstep_end = 40\n"), wrong), "pins meta.json sha256");
  CHECK_THROWS(load_data_plan("", wrong), "pins meta.json sha256");
  // a plan-file dataset without a pin, under another path: the same canonical form (content, not name or path)
  const DataPlan q = load_data_plan(write_plan("[phase.0]\nstep_end = 40\ndataset = \"" + dir2 + "\"\n"), c);
  CHECK_EQ(q.canonical(), p.canonical());
  CHECK_THROWS(load_data_plan(write_plan("[phase.0]\nstep_end = 40\ndataset = \"" + dir2 + "#sha256=" +
                                         std::string(64, 'f') + "\"\n"), c), "pins");
  // explicit file pins: right pin loads (canonical = file content), wrong pin is refused
  const std::string ex = "[phase.0]\nstep_end = 20\n[phase.1]\nkind = \"explicit\"\nstep_end = 40\nexplicit_file = ";
  const DataPlan e = load_data_plan(write_plan(ex + "\"ids.u32#sha256=" + ids_sha + "\"\n"), c);
  CHECK_EQ(e.seq_of(20, 3), uint64_t(ids[3]));
  CHECK(e.canonical().find("\"explicit\":\"sha256:" + ids_sha + "\"") != std::string::npos);
  CHECK_THROWS(load_data_plan(write_plan(ex + "\"ids.u32#sha256=" + std::string(64, '0') + "\"\n"), c),
               "the plan pins");
  s1::remove_dir(d.dir);
  s1::remove_dir(dir2);
}

// A reshuffled Repeat starts its passes at mixture index 0: index_base != 0 is refused (pass 0 would read
// levanter_seq(index_base + m) and not cover the subset).
NM_TEST(s1_plan, reshuffle_needs_index_base_zero) {
  fake();
  DataPlan p;
  Phase ph = s1::levanter_phase(kFake, 0, 64, 64);
  ph.kind = OrderKind::Repeat;
  ph.subset = 2048;
  ph.reshuffle = true;
  ph.index_base = 1000;
  p.phases.push_back(ph);
  CHECK_THROWS(data::validate_plan(p, 4), "index_base must be 0");
  p.phases[0].index_base = 0;
  data::validate_plan(p, 4);
  std::set<uint64_t> pass0;
  for (uint64_t m = 0; m < 2048; ++m) pass0.insert(p.seq_of(int64_t(m / 64), int(m % 64)));
  CHECK_EQ(pass0.size(), size_t(2048));
}

// pass_boundaries reports the step of the START of the mixture block that holds a pass's first index: with
// max_train_batches 4767 (subset % 2048 = 1920) pass 1's block 297 starts at index 608,256 = step 4752, and steps
// 4752-4767 mix passes 0 and 1. With 4768 (subset % 2048 = 0) the boundary is exactly the pass start, step 4768.
NM_TEST(s1_plan, pass_boundary_is_block_start) {
  fake();
  for (uint64_t mtb : {4767ull, 4768ull}) {
    DataPlan p;
    Phase ph = s1::levanter_phase(kFake, 0, 11444, 128);
    ph.kind = OrderKind::Repeat;
    ph.subset = mtb * 128;
    p.phases.push_back(ph);
    const auto pb = p.pass_boundaries();
    CHECK(!pb.empty());
    CHECK_EQ(pb[0], int64_t(mtb == 4767 ? 4752 : 4768));
    CHECK_EQ(data::first_index_of_pass(1, mtb * 128, 2048), uint64_t(mtb == 4767 ? 608256 : 610304));
    // the boundary step's block holds the first wrapped index; no earlier step reads a wrapped one
    const uint64_t N = 2441406;
    LevanterOrder o(42, N, 2048);
    const uint64_t j0 = uint64_t(pb[0]) * 128;
    for (uint64_t j = 0; j < j0; j += 61) CHECK_EQ(p.seq_of(int64_t(j / 128), int(j % 128)), o(j));
    bool wrapped = false;
    for (uint64_t j = j0; j < j0 + 2048; ++j) wrapped = wrapped || p.seq_of(int64_t(j / 128), int(j % 128)) != o(j);
    CHECK(wrapped);
  }
}

// step_seqs (the loader's per-step path) equals seq_of row by row for every order kind, and threads that read the
// same reshuffled pass at once agree (one shared permutation per pass).
NM_TEST(s1_plan, step_seqs_equals_seq_of) {
  fake();
  const std::string dir = s1::temp_dir("steps");
  std::vector<uint32_t> ids(64 * 50);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<uint32_t>((i * 104729) % 2441406);
  s1::write_vec(dir + "/ids.u32", ids);
  DataPlan p;
  p.phases.push_back(s1::levanter_phase(kFake, 0, 30, 64));
  Phase rp = s1::levanter_phase(kFake, 30, 90, 64);
  rp.kind = OrderKind::Repeat;
  rp.subset = 1000;
  p.phases.push_back(rp);
  Phase rs = rp;
  rs.step_begin = 90;
  rs.step_end = 150;
  rs.reshuffle = true;
  p.phases.push_back(rs);
  Phase ex = s1::levanter_phase(kFake, 150, 200, 64);
  ex.kind = OrderKind::Explicit;
  ex.explicit_file = dir + "/ids.u32";
  p.phases.push_back(ex);
  data::validate_plan(p, 4);
  std::vector<uint64_t> got(64);
  int bad = 0;
  for (int64_t k = 0; k < 200; ++k) {
    data::step_seqs(p, k, got.data());
    for (int r = 0; r < 64; ++r) bad += got[size_t(r)] != p.seq_of(k, r);
  }
  CHECK_EQ(bad, 0);
  std::vector<std::vector<uint64_t>> per_thread(8, std::vector<uint64_t>(60 * 64));
  std::vector<std::thread> th;
  for (int t = 0; t < 8; ++t)
    th.emplace_back([&, t] {
      for (int64_t k = 90; k < 150; ++k) data::step_seqs(p, k, per_thread[size_t(t)].data() + (k - 90) * 64);
    });
  for (auto& t : th) t.join();
  for (int t = 1; t < 8; ++t) CHECK(per_thread[size_t(t)] == per_thread[0]);
  s1::remove_dir(dir);
}
