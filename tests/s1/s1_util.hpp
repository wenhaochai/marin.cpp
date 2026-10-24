// Helpers of the S1 tests: converted-data lookup (skip when absent), synthetic datasets, small file IO.
#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../s0/nmtest.hpp"
#include "data/internal.hpp"
#include "data/json.hpp"
#include "marincpp/data.hpp"

namespace s1 {

inline const char* kFineweb = "fineweb-edu-10B-marin128k";
inline const char* kPaloma = "paloma-marin128k";

// Directory of a converted dataset, or SKIP with the command that makes it.
inline std::string converted_or_skip(const char* name) {
  std::string root;
  try {
    root = nm::data::data_root();
  } catch (const std::exception&) {
    SKIP("NM_DATA / NANOMARIN_ROOT not set: converted datasets unavailable");
  }
  const std::string d = root + "/" + name;
  if (!nmtest::file_exists(d + "/meta.json"))
    SKIP(d + "/meta.json absent: run tools/convert/levanter_to_flat.py " +
         std::string(std::string(name) == kPaloma ? "paloma" : "fineweb") + " on a vis node");
  return d;
}

// A reference file of the repo (cache/ is not committed): SKIP when absent.
inline std::string repo_file_or_skip(const std::string& rel) {
  const std::string p = nmtest::src_dir() + "/" + rel;
  if (!nmtest::file_exists(p)) SKIP(rel + " absent (cache/ is produced by the levref / data tools, not committed)");
  return p;
}

inline std::string temp_dir(const std::string& tag) {
  const char* t = std::getenv("TMPDIR");
  std::string tmpl = std::string(t && *t ? t : "/tmp") + "/nm_s1_" + tag + "_XXXXXX";
  if (!::mkdtemp(tmpl.data())) throw std::runtime_error("mkdtemp failed");
  return tmpl;
}

template <class T>
void write_vec(const std::string& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <class T>
std::vector<T> read_vec(const std::string& path) {
  const std::string s = nmtest::read_file(path);
  std::vector<T> v(s.size() / sizeof(T));
  std::memcpy(v.data(), s.data(), v.size() * sizeof(T));
  return v;
}

// .npy (version 1/2, little-endian, C order) payload bytes; empty if absent or not that layout.
inline std::string npy_payload(const std::string& path) {
  const std::string s = nmtest::read_file(path);
  if (s.size() < 12 || s.compare(1, 5, "NUMPY") != 0) return "";
  const size_t hl = s[6] == 1 ? (uint8_t(s[8]) | (uint8_t(s[9]) << 8)) : (uint8_t(s[8]) | (uint8_t(s[9]) << 8) |
                                                                            (uint8_t(s[10]) << 16) | (uint8_t(s[11]) << 24));
  const size_t off = (s[6] == 1 ? 10 : 12) + hl;
  if (s.find("'fortran_order': False", 0) == std::string::npos) return "";
  return s.substr(off);
}

// Synthetic converted dataset: n_tokens tokens t -> f(t) (EOS every `doc_len` tokens closes a document), docs.u64,
// meta.json with the fields load_dataset_meta() reads, and its LCG constants registered for `seed`.
struct Synthetic {
  std::string dir, name;
  uint64_t n_tokens = 0;
  int S = 0;
  std::vector<uint32_t> tok;
  std::vector<uint64_t> docs;
};

inline uint32_t synth_token(uint64_t t, int doc_len) {
  if (t % doc_len == static_cast<uint64_t>(doc_len) - 1) return 128001u;
  if (t % doc_len == 0) return 128000u;
  uint64_t x = t * 0x9E3779B97F4A7C15ull;
  x ^= x >> 29;
  return static_cast<uint32_t>(x % 128000u);
}

inline Synthetic make_synthetic(const std::string& name, uint64_t n_tokens, int S, int doc_len, uint32_t seed,
                                uint64_t lcg_a, uint64_t lcg_b) {
  Synthetic d;
  d.dir = temp_dir(name);
  d.name = name;
  d.n_tokens = n_tokens;
  d.S = S;
  d.tok.resize(n_tokens);
  for (uint64_t t = 0; t < n_tokens; ++t) d.tok[t] = synth_token(t, doc_len);
  for (uint64_t t = 0; t < n_tokens; t += static_cast<uint64_t>(doc_len)) d.docs.push_back(t);
  d.docs.push_back(n_tokens);
  write_vec(d.dir + "/tokens.u32", d.tok);
  write_vec(d.dir + "/docs.u64", d.docs);
  const std::string tsha = nm::sha256_hex(d.tok.data(), d.tok.size() * 4);
  const uint64_t n_seqs = n_tokens / static_cast<uint64_t>(S);
  char buf[2048];
  std::snprintf(buf, sizeof(buf),
                "{\"format\": 2, \"name\": \"%s\", \"kind\": \"train\", \"n_tokens\": %llu, \"seq_len\": %d, "
                "\"n_seqs\": %llu, \"n_docs\": %llu, \"vocab\": 128256, \"bos\": 128000, \"eos\": 128001, "
                "\"max_segs_per_seq\": %d, \"tokens_sha256\": \"%s\", \"files\": {\"tokens.u32\": {\"bytes\": %llu, "
                "\"sha256\": \"%s\"}, \"docs.u64\": {\"bytes\": %llu, \"sha256\": \"x\"}}, \"lcg\": {\"%u\": {\"N\": "
                "%llu, \"a\": %llu, \"b\": %llu}}}\n",
                name.c_str(), (unsigned long long)n_tokens, S, (unsigned long long)n_seqs,
                (unsigned long long)(d.docs.size() - 1), S / doc_len + 2, tsha.c_str(),
                (unsigned long long)(n_tokens * 4), tsha.c_str(), (unsigned long long)(d.docs.size() * 8), seed,
                (unsigned long long)n_seqs, (unsigned long long)lcg_a, (unsigned long long)lcg_b);
  std::ofstream(d.dir + "/meta.json") << buf;
  return d;
}

inline void remove_dir(const std::string& d) {
  if (d.empty() || d.find("nm_s1_") == std::string::npos) return;
  std::string cmd = "rm -rf '" + d + "'";
  if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "could not remove %s\n", d.c_str());
}

// Shortens a file on disk (a TokenFile opened before keeps its full-length mapping: reads past the new end fail).
inline void truncate_file(const std::string& path, uint64_t bytes) {
  if (::truncate(path.c_str(), static_cast<off_t>(bytes)) != 0) throw std::runtime_error("truncate failed: " + path);
}

// A fake registered meta (no files): enough for DataPlan::seq_of on the real N.
inline void register_fake(const std::string& name, uint64_t n_tokens, int S) {
  nm::DatasetMeta m;
  m.name = name;
  m.dir = "/nonexistent/" + name;
  m.n_tokens = n_tokens;
  m.seq_len = S;
  m.n_seqs = n_tokens / static_cast<uint64_t>(S);
  m.vocab = 128256;
  m.max_segs_per_seq = 24;
  nm::data::register_dataset(m);
}

inline nm::Phase levanter_phase(const std::string& ds, int64_t b, int64_t e, int B, int S = 4096, uint32_t seed = 42) {
  nm::Phase p;
  p.step_begin = b;
  p.step_end = e;
  p.dataset = ds;
  p.global_batch = B;
  p.seq_len = S;
  p.kind = nm::OrderKind::Levanter;
  p.seed = seed;
  return p;
}

// data.hpp BatchLoader has no destructor: a test that throws past a started loader would destroy joinable threads
// (std::terminate). Every test holds its loader through this guard.
struct LoaderGuard {
  nm::BatchLoader& l;
  ~LoaderGuard() { l.stop(); }
};

}  // namespace s1
