// Converted datasets: meta.json, the read-only token mapping, registries. DESIGN.md §5.1-§5.2.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "data/internal.hpp"
#include "data/json.hpp"
#include "marincpp/data.hpp"

namespace nm {
namespace data {
namespace {

std::mutex g_mu;   // guards every table below

std::map<std::string, DatasetMeta>& datasets() {   // by DatasetMeta::name: register_dataset, load_dataset_meta
  static std::map<std::string, DatasetMeta> m;
  return m;
}

// By reference (a path, or a name whose meta.json carries another name): the meta find_dataset loaded for exactly that
// reference, so two directories whose metas share a name never answer for each other.
std::map<std::string, DatasetMeta>& loaded_refs() {
  static std::map<std::string, DatasetMeta> m;
  return m;
}

// TokenFile has no fd member (data.hpp): what open() knows about the file, keyed by its mapping address.
struct FileInfo {
  int fd;
  std::string path, dataset, tokens_sha256;
  uint64_t n_tokens;
};
std::map<const void*, FileInfo>& files_by_base() {
  static std::map<const void*, FileInfo> m;
  return m;
}

std::map<std::string, std::unique_ptr<TokenFile>>& shared_files() {
  static std::map<std::string, std::unique_ptr<TokenFile>> m;
  return m;
}

const FileInfo& file_info(const TokenFile& f) {
  NM_REQUIRE(f.n_tokens() > 0, "TokenFile is not open");
  std::lock_guard<std::mutex> g(g_mu);
  auto it = files_by_base().find(f.window(0, 1));
  if (it == files_by_base().end()) fatal(NM_WHERE, "TokenFile is not open");
  return it->second;   // map nodes are stable and never erased (TokenFile never closes)
}

size_t file_size(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) fatal(NM_WHERE, "missing data file " + path + ": " + std::strerror(errno));
  return static_cast<size_t>(st.st_size);
}

size_t fd_size(int fd, const std::string& path) {
  struct stat st;
  if (::fstat(fd, &st) != 0) fatal(NM_WHERE, "fstat of " + path + " failed: " + std::strerror(errno));
  return static_cast<size_t>(st.st_size);
}

}  // namespace

std::string pin_of(const std::string& ref) {
  const size_t p = ref.find("#sha256=");
  return p == std::string::npos ? "" : ref.substr(p + 8);
}

std::string data_root() {
  if (const char* d = std::getenv("NM_DATA"); d && *d) return d;
  if (const char* r = std::getenv("NANOMARIN_ROOT"); r && *r) return std::string(r) + "/marin_store_big/marincpp";
  fatal(NM_WHERE, "set NM_DATA or NANOMARIN_ROOT: converted datasets live under $NANOMARIN_ROOT/marin_store_big/marincpp");
}

std::string strip_hash(const std::string& ref) {
  const size_t p = ref.find("#sha256=");
  return p == std::string::npos ? ref : ref.substr(0, p);
}

std::string dataset_dir(const std::string& ref, std::string* sha) {
  if (sha) *sha = pin_of(ref);
  const std::string name = strip_hash(ref);
  NM_REQUIRE(!name.empty(), "empty dataset reference");
  if (name.find('/') != std::string::npos) return name;
  return data_root() + "/" + name;
}

void register_dataset(const DatasetMeta& m) {
  NM_REQUIRE(!m.name.empty(), "dataset meta without a name");
  std::lock_guard<std::mutex> g(g_mu);
  datasets()[m.name] = m;
}

const DatasetMeta& find_dataset(const std::string& ref) {
  const std::string key = strip_hash(ref), want = pin_of(ref);   // key: a name or a path
  const DatasetMeta* m = nullptr;
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = datasets().find(key);   // registered names (never contain '/')
    if (it != datasets().end()) m = &it->second;
    else if (auto l = loaded_refs().find(key); l != loaded_refs().end()) m = &l->second;
  }
  if (m == nullptr) {
    const DatasetMeta loaded = load_dataset_meta(ref);   // also registers it under loaded.name; checks the pin
    std::lock_guard<std::mutex> g(g_mu);
    m = &(loaded_refs()[key] = loaded);
  }
  // a registered meta answers the reference only if it is the pinned content (a pin is never skipped because the
  // dataset was loaded earlier, e.g. by another phase or by resolve())
  if (!want.empty() && want != m->meta_sha256)
    fatal(NM_WHERE, "dataset " + key + ": the reference pins meta.json sha256 " + want + ", the registered meta has " +
                        (m->meta_sha256.empty() ? std::string("no sha256") : m->meta_sha256));
  return *m;
}

const TokenFile& shared_token_file(const std::string& ref) {
  const DatasetMeta& m = find_dataset(ref);
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = shared_files().find(m.dir);   // by directory: the files themselves
    if (it != shared_files().end()) return *it->second;
  }
  auto f = std::make_unique<TokenFile>();
  f->open(m);   // outside the lock (open registers its fd under it)
  std::lock_guard<std::mutex> g(g_mu);
  auto& slot = shared_files()[m.dir];
  if (!slot) slot = std::move(f);   // a racing opener's TokenFile is dropped; TokenFile never unmaps, so it stays valid
  return *slot;
}

int token_file_fd(const TokenFile& f) { return file_info(f).fd; }
const std::string& token_file_path(const TokenFile& f) { return file_info(f).path; }
const std::string& token_file_sha256(const TokenFile& f) { return file_info(f).tokens_sha256; }

void check_token_file(const TokenFile& f, const Phase& ph) {
  const FileInfo& fi = file_info(f);
  const DatasetMeta& m = find_dataset(ph.dataset);
  if (fi.n_tokens != m.n_tokens || fi.tokens_sha256 != m.tokens_sha256)
    fatal(NM_WHERE, "TokenFile " + fi.path + " holds dataset " + fi.dataset + " (" + std::to_string(fi.n_tokens) +
                        " tokens, sha256 " + fi.tokens_sha256 + "), the phase reads " + m.name + " (" +
                        std::to_string(m.n_tokens) + " tokens, sha256 " + m.tokens_sha256 + ")");
}

void pread_window(const TokenFile& f, uint64_t seq, int S, uint32_t* out) {
  NM_REQUIRE(S > 0 && (seq + 1) * static_cast<uint64_t>(S) <= f.n_tokens(), "window past the end of the token file");
  const int fd = token_file_fd(f);
  size_t want = static_cast<size_t>(S) * 4, done = 0;
  const off_t off = static_cast<off_t>(seq * static_cast<uint64_t>(S) * 4);
  char* dst = reinterpret_cast<char*>(out);
  while (done < want) {
    const ssize_t n = ::pread(fd, dst + done, want - done, off + static_cast<off_t>(done));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0)
      fatal(NM_WHERE, "pread of window " + std::to_string(seq) + " of " + token_file_path(f) + " failed: " +
                          std::string(n < 0 ? std::strerror(errno) : "EOF (file shorter than its meta.json)"));
    done += static_cast<size_t>(n);
  }
}

}  // namespace data

// ------------------------------------------------------------------------------------------------ DatasetMeta
DatasetMeta load_dataset_meta(const std::string& ref) {
  std::string want_sha;
  const std::string dir = data::dataset_dir(ref, &want_sha);
  const std::string path = dir + "/meta.json";
  DatasetMeta m;
  m.meta_sha256 = sha256_file_hex(path);
  if (!want_sha.empty() && want_sha != m.meta_sha256)
    fatal(NM_WHERE, path + " has sha256 " + m.meta_sha256 + " but the config pins " + want_sha +
                        " (the dataset was reconverted or another directory is mounted)");
  const json::Value v = json::parse_file(path);
  // format 2: exact segs_per_seq.u16 (format 1 stored segs_per_seq.u8 with counts wrapped mod 256): reconvert format 1
  NM_REQUIRE(v.at("format").as_i64() == 2,
             path + ": meta format " + v.at("format").text + ", expected 2 (rerun tools/convert/levanter_to_flat.py)");
  m.name = v.at("name").as_str();
  m.dir = dir;
  m.n_tokens = v.at("n_tokens").as_u64();
  m.seq_len = static_cast<int>(v.at("seq_len").as_i64());
  m.n_seqs = v.at("n_seqs").as_u64();
  m.n_docs = v.at("n_docs").as_u64();
  m.vocab = static_cast<int>(v.at("vocab").as_i64());
  m.bos = static_cast<int32_t>(v.at("bos").as_i64());
  m.eos = static_cast<int32_t>(v.at("eos").as_i64());
  m.max_segs_per_seq = static_cast<int>(v.at("max_segs_per_seq").as_i64());
  m.tokens_sha256 = v.at("tokens_sha256").as_str();
  NM_REQUIRE(m.seq_len > 0 && m.n_seqs == m.n_tokens / static_cast<uint64_t>(m.seq_len), path + ": n_seqs");
  NM_REQUIRE(m.vocab > 0 && m.max_segs_per_seq >= 1 && m.max_segs_per_seq <= m.seq_len,
             path + ": vocab / max_segs_per_seq");
  // sizes only: the converter checks the content before it commits (and `levanter_to_flat.py verify` re-hashes); a full
  // hash of tokens.u32 (40 GB) at every start would cost about a minute
  const json::Value& files = v.at("files");
  for (const auto& [f, info] : files.obj) {
    const size_t sz = data::file_size(dir + "/" + f);
    if (sz != info.at("bytes").as_u64())
      fatal(NM_WHERE, dir + "/" + f + ": size " + std::to_string(sz) + " != meta.json " + info.at("bytes").text);
  }
  NM_REQUIRE(files.at("tokens.u32").at("bytes").as_u64() == 4 * m.n_tokens, path + ": tokens.u32 size");
  NM_REQUIRE(files.at("tokens.u32").at("sha256").as_str() == m.tokens_sha256, path + ": tokens sha256");
  if (v.has("lcg"))
    for (const auto& [seed, e] : v.at("lcg").obj)
      data::register_lcg(static_cast<uint32_t>(std::stoul(seed.substr(0, seed.find('@')))), e.at("N").as_u64(),
                         e.at("a").as_u64(), e.at("b").as_u64());   // key "<seed>" or "<seed>@<seq_len>"
  data::register_dataset(m);
  return m;
}

// ------------------------------------------------------------------------------------------------ TokenFile
void TokenFile::open(const DatasetMeta& m) {
  NM_REQUIRE(base_ == nullptr, "TokenFile::open twice");
  const std::string tpath = m.dir + "/tokens.u32", dpath = m.dir + "/docs.u64";
  const int fd = ::open(tpath.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fatal(NM_WHERE, "cannot open " + tpath + ": " + std::strerror(errno));
  void* p = MAP_FAILED;
  void* dp = MAP_FAILED;
  size_t sz = 0, dsz = 0;
  try {   // every failure below releases what was opened or mapped: no half-open TokenFile, no leaked fd
    sz = data::fd_size(fd, tpath);
    if (sz != m.n_tokens * 4 || sz == 0)
      fatal(NM_WHERE, tpath + ": size " + std::to_string(sz) + " != 4 x n_tokens (" + std::to_string(m.n_tokens) + ")");
    p = ::mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) fatal(NM_WHERE, "mmap of " + tpath + " failed: " + std::strerror(errno));
    if (::access(dpath.c_str(), R_OK) == 0) {
      const int dfd = ::open(dpath.c_str(), O_RDONLY | O_CLOEXEC);
      if (dfd < 0) fatal(NM_WHERE, "cannot open " + dpath + ": " + std::strerror(errno));
      try {
        dsz = data::fd_size(dfd, dpath);
        NM_REQUIRE(dsz == (m.n_docs + 1) * 8, dpath + ": size != 8 x (n_docs + 1)");
        dp = ::mmap(nullptr, dsz, PROT_READ, MAP_SHARED, dfd, 0);
        if (dp == MAP_FAILED) fatal(NM_WHERE, "mmap of " + dpath + " failed: " + std::strerror(errno));
      } catch (...) {
        ::close(dfd);
        throw;
      }
      ::close(dfd);
      const uint64_t* d = static_cast<const uint64_t*>(dp);
      NM_REQUIRE(d[0] == 0 && d[m.n_docs] == m.n_tokens, dpath + ": first offset must be 0 and the last n_tokens");
    }
  } catch (...) {
    if (dp != MAP_FAILED) ::munmap(dp, dsz);
    if (p != MAP_FAILED) ::munmap(p, sz);
    ::close(fd);
    throw;
  }
  base_ = static_cast<const uint32_t*>(p);
  map_bytes_ = sz;
  n_ = m.n_tokens;
  if (dp != MAP_FAILED) {
    docs_ = static_cast<const uint64_t*>(dp);
    n_docs_ = m.n_docs;
  }
  std::lock_guard<std::mutex> g(data::g_mu);
  data::files_by_base()[base_] = data::FileInfo{fd, tpath, m.name, m.tokens_sha256, m.n_tokens};
}

const uint32_t* TokenFile::window(uint64_t seq, int S) const {
  NM_REQUIRE(base_ != nullptr, "TokenFile is not open");
  NM_REQUIRE(S > 0 && (seq + 1) * static_cast<uint64_t>(S) <= n_, "window past the end of the token file");
  return base_ + seq * static_cast<uint64_t>(S);
}

uint64_t TokenFile::doc_of_token(uint64_t t) const {
  NM_REQUIRE(docs_ != nullptr, "dataset has no docs.u64");
  NM_REQUIRE(t < n_, "token index past the end");
  return static_cast<uint64_t>(std::upper_bound(docs_, docs_ + n_docs_ + 1, t) - docs_) - 1;
}

}  // namespace nm
