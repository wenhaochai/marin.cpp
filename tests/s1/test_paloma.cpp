// T0.4 (Paloma eval set: 16 per-domain sha256, 14,529 windows, 114 batches of 128, 63 zero rows in batch 113), the
// exact segment counts of every window, and T0.5 (bytes-per-token table) on the converted files, read through the C++
// data module. The comparison with levanter's own eval batches is the separate suite t0_4_levanter, which skips visibly
// (ctest "Skipped") when bench/n1_loader/levanter_ref.json is absent.
#include <algorithm>
#include <cstring>

#include "s1_util.hpp"

using namespace nm;

namespace {

const char* kDomains[16] = {"4chan", "c4_100_domains", "c4_en", "dolma-v1_5", "dolma_100_programing_languages",
                            "dolma_100_subreddits", "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit",
                            "m2d2_wikipedia_unsplit", "manosphere_meta_sep", "mc4", "ptb", "redpajama",
                            "twitterAAE_HELM_fixed", "wikitext_103"};
// data.md §8.1 table (levanter-made, [verified]): windows and sha256 of each domain's windows (int32 LE)
const struct { uint32_t seqs; const char* sha; } kRef[16] = {
    {226, "418421ca41b128b5a6e15b6d0f620d9a4e4215c0b26a6712acf04b12802caee0"},
    {2310, "167b9ea6e7dbb4b47d7691ba9931c76a3af1641d5a5d0130d2008326ef9767f4"},
    {236, "5ef2fc5286e6a1fac552f34e9e48b3b664ddec8e3550dacba889de051d433b69"},
    {703, "ef6c458e93f48c42e725ee9d3f3807ac77aa3c52b2de1c6979179ae8e0e04080"},
    {2075, "8bd793b5b43e0a1319ef70ea2a0567ee73cae27053b9fd6e5e4508f644272a8f"},
    {2303, "f0548824ea12008b094e09c55aead5a826843982561cd7d179dbe0dd940a2abc"},
    {236, "0f26d6b4aa56892d8194623578dae32260c1b3459e2dddf7867934c52a37b0c9"},
    {234, "7ebf2640b9eba8524cc34631b5b4f5e3b67f9dbf21d4eb9e8de73458af00f850"},
    {4112, "f9e4608406c0bb372a6d93c94b52ec14162f6f22b3b5f344fff2ecff05f55df0"},
    {1181, "4d8d5225bc27e7859e0ee980579ff431314803937acb6bfc9a8ae66c38d4ac77"},
    {235, "09b6b8efed1a3847107a353540f51e501eb6b65922f3a130606983249de23f1e"},
    {236, "081f5f67541a337845e3fd9f89b1589112f20eb90871ead87f5049e75030a6e3"},
    {21, "ffb45ee0e9a0b5f9f6ed16c51175cf6f7a7ca89756128727a704233dd7b374e7"},
    {156, "8bc20349c6e0191412591b9f940c5b49d35f723bb70b8c0eac4e73ec99696e7e"},
    {205, "9d110d797b7a233ab5b82fdcdd3c7202f4f5bb93ffd14b5701b28774ccf32d93"},
    {60, "a3cadf50589690a770299069a1a7f9f81f0bd532c180713f276f20e6dc91002a"},
};

// Estimator batch b of the eval set: rows b*128 .. b*128+127 in index order; rows past the end are zeros (tokens 0,
// weight 0, no tag), as levanter's DataLoader pads (data.md §8.2).
std::vector<int32_t> eval_batch(const TokenFile& f, uint64_t n_seqs, int b, int* real_rows) {
  std::vector<int32_t> out(size_t(128) * 4096, 0);
  *real_rows = 0;
  for (int r = 0; r < 128; ++r) {
    const uint64_t i = uint64_t(b) * 128 + r;
    if (i >= n_seqs) break;
    std::memcpy(out.data() + size_t(r) * 4096, f.window(i, 4096), 4096 * 4);
    ++*real_rows;
  }
  return out;
}

}  // namespace

NM_TEST(t0_4_paloma, domains_and_batches) {
  const std::string d = s1::converted_or_skip(s1::kPaloma);
  const DatasetMeta m = load_dataset_meta(s1::kPaloma);
  CHECK_EQ(m.n_seqs, uint64_t(14529));
  CHECK_EQ(m.seq_len, 4096);
  TokenFile f;
  f.open(m);
  const json::Value meta = json::parse_file(d + "/meta.json");
  const json::Value& doms = meta.at("domains");
  CHECK_EQ(doms.arr.size(), size_t(16));
  const auto tags = s1::read_vec<uint16_t>(d + "/tags.u16");
  CHECK_EQ(tags.size(), size_t(14529));
  uint64_t first = 0;
  for (int i = 0; i < 16 && i < int(doms.arr.size()); ++i) {
    const json::Value& e = doms.at(size_t(i));
    CHECK_EQ(e.at("name").as_str(), std::string(kDomains[i]));
    CHECK_EQ(e.at("first_seq").as_u64(), first);
    CHECK_EQ(e.at("n_seqs").as_u64(), uint64_t(kRef[i].seqs));
    CHECK_EQ(e.at("scored_tokens").as_u64(), uint64_t(kRef[i].seqs) * 4095);
    const std::string h = sha256_hex(f.window(first, 4096), size_t(kRef[i].seqs) * 4096 * 4);   // C++ read
    CHECK_EQ(h, std::string(kRef[i].sha));
    CHECK_EQ(e.at("sha256").as_str(), std::string(kRef[i].sha));
    for (uint64_t s = first; s < first + kRef[i].seqs; ++s) CHECK_EQ(int(tags[s]), i);
    first += kRef[i].seqs;
  }
  CHECK_EQ(first, uint64_t(14529));
  CHECK_EQ(meta.at("eval_batch").as_i64(), int64_t(128));
  CHECK_EQ(meta.at("n_batches").as_i64(), int64_t(114));
  CHECK_EQ(meta.at("last_batch_real_rows").as_i64(), int64_t(65));
  int real = 0;
  (void)eval_batch(f, m.n_seqs, 113, &real);
  CHECK_EQ(real, 65);   // batch 113: rows 14,464..14,528 real, 63 padding rows (their zeros are eval_batch's fill)
}

// Levanter's DataLoader(DomainTaggedDataset) batches (bench/n1_loader/levanter_ref.json): tokens and tags of batches 19,
// 20, 21 (c4_en spans them), tokens of the padded batch 113 (the reference holds no tags for it; its real rows' tags
// are the domain layout that t0_4_paloma checks), and c4_en's segment ids and loss weights.
NM_TEST(t0_4_levanter, eval_batches) {
  const std::string d = s1::converted_or_skip(s1::kPaloma);
  const json::Value lev = json::parse_file(s1::repo_file_or_skip("bench/n1_loader/levanter_ref.json")).at("paloma");
  const DatasetMeta m = load_dataset_meta(s1::kPaloma);
  TokenFile f;
  f.open(m);
  const auto tags = s1::read_vec<uint16_t>(d + "/tags.u16");
  int real = 0;
  const auto last = eval_batch(f, m.n_seqs, 113, &real);
  CHECK_EQ(int64_t(real), lev.at("last_batch").at("real_rows").as_i64());
  for (const char* b : {"19", "20", "21"}) {
    const auto x = eval_batch(f, m.n_seqs, std::stoi(b), &real);
    CHECK_EQ(sha256_hex(x.data(), x.size() * 4), lev.at("eval_batches").at(b).at("tokens_all").as_str());
    const json::Value& tg = lev.at("eval_batches").at(b).at("tag");
    for (int r = 0; r < 128; ++r) CHECK_EQ(int64_t(tags[size_t(std::stoi(b)) * 128 + r]), tg.at(size_t(r)).as_i64());
  }
  CHECK_EQ(sha256_hex(last.data(), last.size() * 4), lev.at("last_batch").at("tokens_all").as_str());
  const uint64_t c4 = 2536;
  std::vector<int32_t> tok(size_t(236) * 4096), seg(tok.size());
  std::vector<float> w(tok.size());
  std::memcpy(tok.data(), f.window(c4, 4096), tok.size() * 4);
  data::segment_ids(tok.data(), 236, 4096, m.eos, seg.data());
  data::loss_weights(236, 4096, w.data());
  CHECK_EQ(lev.at("domain").as_str(), std::string("c4_en"));
  CHECK_EQ(sha256_hex(tok.data(), tok.size() * 4), lev.at("tokens").as_str());
  CHECK_EQ(sha256_hex(seg.data(), seg.size() * 4), lev.at("segment_ids").as_str());
  CHECK_EQ(sha256_hex(w.data(), w.size() * 4), lev.at("loss_weight").as_str());
}

// Exact segment counts (1 + #EOS at positions 0..4094) of all 14,529 windows, recounted here from tokens.u32: equal to
// segs_per_seq.u16 for every window (no stored count wraps), and the maximum, 291, equals meta.json max_segs_per_seq
// (format-1 converters stored count mod 256 and reported 255). The 85 windows above 255 all lie in twitterAAE_HELM_fixed
// (tag 14, about 17 tokens per document); per-domain maxima twitterAAE 291, gab 122, dolma_100_programing_languages 47,
// every other domain 41 or less.
NM_TEST(t0_4_paloma, segment_counts) {
  const std::string d = s1::converted_or_skip(s1::kPaloma);
  const DatasetMeta m = load_dataset_meta(s1::kPaloma);
  TokenFile f;
  f.open(m);
  const auto stored = s1::read_vec<uint16_t>(d + "/segs_per_seq.u16");
  const auto tags = s1::read_vec<uint16_t>(d + "/tags.u16");
  CHECK_EQ(stored.size(), size_t(m.n_seqs));
  CHECK_EQ(tags.size(), size_t(m.n_seqs));
  CHECK(!nmtest::file_exists(d + "/segs_per_seq.u8"));   // the wrapped format-1 file is gone
  int differ = 0, above255 = 0, above255_other = 0, mx = 0;
  int dom_max[16] = {0};
  for (uint64_t s = 0; s < m.n_seqs && s < stored.size(); ++s) {
    const int n = data::segments_in_window(f.window(s, 4096), 4096, uint32_t(m.eos));
    differ += n != int(stored[s]);
    mx = std::max(mx, n);
    if (n > 255) {
      ++above255;
      above255_other += tags[s] != 14;
    }
    if (tags[s] < 16) dom_max[tags[s]] = std::max(dom_max[tags[s]], n);
  }
  CHECK_EQ(differ, 0);
  CHECK_EQ(mx, 291);
  CHECK_EQ(m.max_segs_per_seq, 291);
  CHECK_EQ(above255, 85);
  CHECK_EQ(above255_other, 0);
  CHECK_EQ(dom_max[14], 291);
  CHECK_EQ(dom_max[7], 122);
  CHECK_EQ(dom_max[4], 47);
  for (int i = 0; i < 16; ++i)
    if (i != 14 && i != 7 && i != 4) CHECK(dom_max[i] <= 41);
}

// Levanter target bytes per domain (sum of bytes_levanter[next token] over scored positions) recomputed in C++ from the
// converted windows and table, against data.md §8.1's "target bytes" column (from data_ref.json).
NM_TEST(t0_4_paloma, target_bytes) {
  const std::string d = s1::converted_or_skip(s1::kPaloma);
  const std::string rp = s1::repo_file_or_skip("cache/data/data_ref.json");
  const DatasetMeta m = load_dataset_meta(s1::kPaloma);
  TokenFile f;
  f.open(m);
  const auto bl = s1::read_vec<int32_t>(d + "/bytes_levanter.i32");
  const json::Value ref = json::parse_file(rp).at("paloma");
  uint64_t first = 0;
  for (int i = 0; i < 16; ++i) {
    uint64_t y = 0;
    for (uint64_t s = first; s < first + kRef[i].seqs; ++s) {
      const uint32_t* t = f.window(s, 4096);
      for (int p = 0; p < 4095; ++p) y += uint64_t(bl[t[p + 1]]);
    }
    CHECK_EQ(y, ref.at(kDomains[i]).at("target_bytes").as_u64());
    first += kRef[i].seqs;
  }
}

// T0.5: the levanter byte table in both converted sets: sha256 d3be7450...4e4d over all 128,256 entries, the 256 special
// ids (BOS, EOS included) at 0; the converter reran levanter's byte_length_of_token and recorded equality. The exact
// UTF-8 table differs only where the U+FFFD rule applies (data.md §8.4): counted and printed.
NM_TEST(t0_5_bytes, tables) {
  int seen = 0;
  for (const char* name : {s1::kFineweb, s1::kPaloma}) {
    std::string d;
    try {
      d = s1::converted_or_skip(name);
    } catch (const nmtest::Skip&) {
      continue;
    }
    ++seen;
    CHECK_EQ(sha256_file_hex(d + "/bytes_levanter.i32"),
             std::string("d3be74509077b5ba84a2822f0348c7574b685404178786b7e0d473c64a334e4d"));
    const auto bl = s1::read_vec<int32_t>(d + "/bytes_levanter.i32");
    const auto bx = s1::read_vec<int32_t>(d + "/bytes_exact.i32");
    CHECK_EQ(bl.size(), size_t(128256));
    CHECK_EQ(bx.size(), size_t(128256));
    int zeros = 0, differ = 0, fffd_like = 0;
    for (size_t i = 0; i < bl.size() && i < bx.size(); ++i) {
      zeros += bl[i] == 0;
      if (bl[i] != bx[i]) {
        ++differ;
        fffd_like += bl[i] >= 3 && bl[i] % 3 == 0 && bx[i] < bl[i];   // U+FFFD is 3 bytes in place of 1-3 raw bytes
      }
    }
    CHECK_EQ(zeros, 256);
    CHECK_EQ(bl[128000], 0);
    CHECK_EQ(bl[128001], 0);
    CHECK_EQ(bx[128000], 0);
    std::printf("    t0_5 %s: exact table differs at %d ids, %d of them U+FFFD-shaped\n", name, differ, fffd_like);
    const json::Value meta = json::parse_file(d + "/meta.json");
    const json::Value& chk = meta.at("checks").at("bytes_levanter");
    CHECK(chk.has("levanter_recheck") && chk.at("levanter_recheck").at("equal").as_bool());
    const std::string ref = nmtest::src_dir() + "/cache/data/bytes_per_token_marin.i32";
    if (nmtest::file_exists(ref)) CHECK(s1::read_vec<int32_t>(ref) == bl);
  }
  if (!seen) SKIP("no converted dataset");
}
