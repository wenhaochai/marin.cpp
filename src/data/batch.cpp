// Batch assembly without the loader, H(B, k), and the host reference of the training example. data.md §3.1, §7.
#include <cstring>

#include "data/internal.hpp"
#include "marincpp/data.hpp"

namespace nm::data {

void segment_ids(const int32_t* tok, int rows, int S, int32_t eos, int32_t* seg) {
  for (int r = 0; r < rows; ++r) {
    const int32_t* t = tok + static_cast<size_t>(r) * S;
    int32_t* g = seg + static_cast<size_t>(r) * S;
    int32_t c = 0;
    g[0] = 0;
    for (int i = 1; i < S; ++i) {
      c += t[i - 1] == eos;
      g[i] = c;
    }
  }
}

void loss_weights(int rows, int S, float* w) {
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < S; ++i) w[static_cast<size_t>(r) * S + i] = i == S - 1 ? 0.0f : 1.0f;
}

int segments_in_window(const uint32_t* w, int S, uint32_t eos) {
  int n = 1;
  for (int i = 0; i + 1 < S; ++i) n += w[i] == eos;
  return n;
}

std::vector<int32_t> assemble_batch(const DataPlan& p, const std::vector<const TokenFile*>& files, int64_t step) {
  NM_REQUIRE(files.size() == 1 || files.size() == p.phases.size(), "one TokenFile, or one per phase");
  const size_t pi = phase_index(p, step);
  const Phase& ph = p.phases[pi];
  const TokenFile& f = *files[files.size() == 1 ? 0 : pi];
  check_token_file(f, ph);
  const size_t S = static_cast<size_t>(ph.seq_len);
  std::vector<uint64_t> seq(static_cast<size_t>(ph.global_batch));
  step_seqs(p, step, seq.data());
  std::vector<int32_t> out(static_cast<size_t>(ph.global_batch) * S);
  for (int r = 0; r < ph.global_batch; ++r)   // ids < 2^31: the uint32 bytes are the int32 bytes
    pread_window(f, seq[static_cast<size_t>(r)], ph.seq_len, reinterpret_cast<uint32_t*>(out.data() + r * S));
  return out;
}

std::string batch_hash(const DataPlan& p, const std::vector<const TokenFile*>& files, int64_t step) {
  const std::vector<int32_t> b = assemble_batch(p, files, step);
  return sha256_hex(b.data(), b.size() * 4);   // little-endian host
}

}  // namespace nm::data
