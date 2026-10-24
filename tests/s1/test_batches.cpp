// T0.2 (12 batch hashes through the C++ reader), T0.3 (segment ids and loss weights of the same batches), on the
// converted fineweb-edu-10B set. Skips with a message when the converted set is absent. The comparisons with levanter's
// own arrays are the separate suite t0_3_levanter, which skips visibly (ctest "Skipped") when a reference is absent.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "s1_util.hpp"

using namespace nm;

namespace {

struct Ref { int B; int64_t k; const char* sha; };
// data.md §7 (levanter mix.get_batch and the closed form agree) [verified]
const Ref kRefs[] = {
    {128, 0, "4626638f4644adcaf66a9084c34ea3d4d423262a6473d03f6b95123f6bc9e449"},
    {128, 1, "691e9e1ad524af5b1e3a772db5aceadc58bcf9c6e93506eaef508f37e7aaf50c"},
    {128, 2, "ffd0b0e8f29bece94a5888c79fba44a12e7bb6dbff34a6a7c4e45e227e2dc64f"},
    {128, 1000, "d1d0427ec3b89db162f09b57dd8ecea5c7053117c50c115fc1aea2a9c48104bf"},
    {128, 4958, "f8ba38d570bd33a16887fa5927f6ffe0550d9589b3ed44b9240228da2437bab1"},
    {128, 11443, "6bc7e8d141df40d3f5c210d123e9bd81c2f3b45291fec053874c00d058378e2c"},
    {256, 0, "813ff307c4b074a6ec62e51efa1cc95085a5b7c45dd31841d0fbd46895f1624f"},
    {256, 1, "0778706834532ed1b912f18c990a7e72c323b26fe7734dc32c37050eb5282c0e"},
    {256, 4767, "154a347fbd2ab2800744e25b3ca92d383cb5bb34c30831bb9873484d6e2f8025"},
    {256, 9536, "3fc488330822e8a49e96ba54aa722f50f467a7fd3d4ac622661585c0751188ec"},
    {256, 9917, "38211b426b26c5c7c1ce1092fe24eb329cc96b53d2a698180dcf969ea57730c4"},
    {256, 22887, "53942bac0515978ee19cdb0c606636d7e722900d4fadbfb74409cf773309a0e7"},
};

DataPlan plan_for(int B) {   // the baseline plans: 130m/300m (B 128) and 520m/1_2b (B 256), long enough for every k
  DataPlan p;
  p.phases.push_back(s1::levanter_phase(s1::kFineweb, 0, B == 128 ? 11444 : 22888, B));
  return p;
}

std::string hash_bytes(const void* p, size_t n) { return sha256_hex(p, n); }

bool have_gpu() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    cudaGetLastError();
    return false;
  }
  return true;
}

}  // namespace

// The 12 hashes, each through the full loader path: start(first_step = k) -> reader threads pread the windows into the
// ring -> stager copies into a slot -> acquire(k). With a GPU (della-vis1's A100) the slots are the pinned ones and
// each batch also goes to the device through copy_rank_rows and back, hashed again; without one, host slots. Also the
// direct assembly and the loader's own H(B, k) log.
NM_TEST(t0_2_batches, twelve_hashes_through_loader) {
  s1::converted_or_skip(s1::kFineweb);
  DatasetMeta m = load_dataset_meta(s1::kFineweb);
  TokenFile f;
  f.open(m);
  const bool gpu = have_gpu();
  int32_t* d = nullptr;
  cudaStream_t stream = nullptr;
  if (gpu) {
    NM_CUDA(cudaMalloc(&d, size_t(256) * 4096 * 4));
    NM_CUDA(cudaStreamCreate(&stream));
  }
  std::vector<int32_t> back(size_t(256) * 4096);
  for (const Ref& r : kRefs) {
    DataPlan p = plan_for(r.B);
    BatchLoader l;
    s1::LoaderGuard guard{l};
    data::LoaderOptions o;
    o.host_only = !gpu;
    o.hash_every = 1;
    data::set_loader_options(l, o);
    l.start(p, {&f}, 1, 4, 16, r.k);
    TokenSlot& s = l.acquire(r.k);
    const size_t bytes = size_t(r.B) * 4096 * 4;
    const std::string h = hash_bytes(s.host, bytes);
    if (gpu) {
      data::copy_rank_rows(l, s, 0, d, stream);
      NM_CUDA(cudaMemcpyAsync(back.data(), d, bytes, cudaMemcpyDeviceToHost, stream));
      NM_CUDA(cudaStreamSynchronize(stream));
      CHECK_EQ(hash_bytes(back.data(), bytes), std::string(r.sha));
    }
    l.release(s, 0);
    const auto st = data::loader_stats(l);
    l.stop();
    CHECK_EQ(h, std::string(r.sha));
    CHECK(!st.hashes.empty() && st.hashes[0].first == r.k && st.hashes[0].second == r.sha);
    CHECK_EQ(data::batch_hash(p, {&f}, r.k), std::string(r.sha));
  }
  std::printf("    t0_2: 12 hashes through %s slots%s\n", gpu ? "pinned" : "host",
              gpu ? " and copy_rank_rows -> device -> host" : " (no GPU)");
  if (gpu) {
    cudaStreamDestroy(stream);
    cudaFree(d);
  }
}

// The B = 256 batch k is the concatenation of B = 128 batches 2k and 2k + 1 (data.md §4.5).
NM_TEST(t0_2_batches, b256_is_two_b128) {
  s1::converted_or_skip(s1::kFineweb);
  load_dataset_meta(s1::kFineweb);
  DataPlan p128 = plan_for(128), p256 = plan_for(256);
  for (int64_t k : {0, 1, 4767, 5721}) {
    for (int r = 0; r < 256; ++r) CHECK_EQ(p256.seq_of(k, r), p128.seq_of(2 * k + r / 128, r % 128));
  }
}

// T0.3: segment ids and loss weights of the 12 batches, element by element against the definition of data.md §3.1
// (seg[0] = 0, seg[i] - seg[i-1] = [tok[i-1] == 128001]; weight 0 only at position 4095), and each row's segment count
// against the converter's numpy count (segs_per_seq.u16) and the dataset bound max_segs_per_seq (= 17, the maximum
// of the converter's histogram over all 2,441,406 windows).
NM_TEST(t0_3_segments, twelve_batches) {
  const std::string d = s1::converted_or_skip(s1::kFineweb);
  DatasetMeta m = load_dataset_meta(s1::kFineweb);
  TokenFile f;
  f.open(m);
  const auto segs_file = s1::read_vec<uint16_t>(d + "/segs_per_seq.u16");
  CHECK_EQ(segs_file.size(), size_t(m.n_seqs));
  CHECK_EQ(m.max_segs_per_seq, 17);
  CHECK_EQ(int(*std::max_element(segs_file.begin(), segs_file.end())), m.max_segs_per_seq);
  CHECK_EQ(int(*std::min_element(segs_file.begin(), segs_file.end())), 1);
  const json::Value meta = json::parse_file(d + "/meta.json");
  uint64_t hist_total = 0;
  int hist_max = 0;
  for (const auto& [k, v] : meta.at("segs_per_seq_hist").obj) {
    hist_total += v.as_u64();
    hist_max = std::max(hist_max, std::stoi(k));
  }
  CHECK_EQ(hist_total, m.n_seqs);
  CHECK_EQ(hist_max, m.max_segs_per_seq);
  int64_t rows = 0, bad_seg = 0, bad_w = 0, bad_count = 0;
  for (const Ref& r : kRefs) {
    DataPlan p = plan_for(r.B);
    const std::vector<int32_t> tok = data::assemble_batch(p, {&f}, r.k);
    const int S = 4096, B = r.B;
    std::vector<int32_t> seg(tok.size());
    std::vector<float> w(tok.size());
    data::segment_ids(tok.data(), B, S, m.eos, seg.data());
    data::loss_weights(B, S, w.data());
    for (int row = 0; row < B; ++row) {
      const int32_t* t = tok.data() + size_t(row) * S;
      const int32_t* g = seg.data() + size_t(row) * S;
      const float* ww = w.data() + size_t(row) * S;
      bad_seg += g[0] != 0;
      for (int i = 1; i < S; ++i) bad_seg += g[i] - g[i - 1] != (t[i - 1] == 128001 ? 1 : 0);
      for (int i = 0; i < S; ++i) bad_w += ww[i] != (i == S - 1 ? 0.0f : 1.0f);
      // seg[S-1] = #EOS at positions 0..S-2, so the row has seg[S-1] + 1 segments: the converter's numpy count
      bad_count += int(g[S - 1]) + 1 != int(segs_file[p.seq_of(r.k, row)]) || g[S - 1] + 1 > m.max_segs_per_seq;
      ++rows;
    }
  }
  CHECK_EQ(rows, int64_t(6 * 128 + 6 * 256));
  CHECK_EQ(bad_seg, int64_t(0));
  CHECK_EQ(bad_w, int64_t(0));
  CHECK_EQ(bad_count, int64_t(0));
}

// T0.3 against levanter's own arrays (bench/n1_loader/levanter_ref.json: hashes of tokens, segment ids, loss weights
// and of each of the 4 devices' rows, for B = 128 batches 0, 1, 1000, 4958 of the 4xH100 layout). The other 8 batches
// have no levanter array reference; t0_3_segments checks them against the formula.
NM_TEST(t0_3_levanter, levanter_ref_json_batches) {
  s1::converted_or_skip(s1::kFineweb);
  const json::Value lev = json::parse_file(s1::repo_file_or_skip("bench/n1_loader/levanter_ref.json"));
  DatasetMeta m = load_dataset_meta(s1::kFineweb);
  TokenFile f;
  f.open(m);
  const DataPlan p = plan_for(128);
  const int S = 4096, B = 128;
  int checked = 0;
  for (int64_t k : {0, 1, 1000, 4958}) {
    const json::Value& e = lev.at("train_B128").at(std::to_string(k));
    const std::vector<int32_t> tok = data::assemble_batch(p, {&f}, k);
    std::vector<int32_t> seg(tok.size());
    std::vector<float> w(tok.size());
    data::segment_ids(tok.data(), B, S, m.eos, seg.data());
    data::loss_weights(B, S, w.data());
    CHECK_EQ(hash_bytes(tok.data(), tok.size() * 4), e.at("tokens").as_str());
    CHECK_EQ(hash_bytes(seg.data(), seg.size() * 4), e.at("segment_ids").as_str());
    CHECK_EQ(hash_bytes(w.data(), w.size() * 4), e.at("loss_weight").as_str());
    CHECK_EQ(e.at("devices").arr.size(), size_t(4));
    for (const json::Value& dv : e.at("devices").arr) {   // 4-device split of the 4xH100 runs
      const int r0 = int(dv.at("rows").at(0).as_i64()), r1 = int(dv.at("rows").at(1).as_i64());
      const size_t o = size_t(r0) * S, n = size_t(r1 - r0) * S;
      CHECK_EQ(hash_bytes(tok.data() + o, n * 4), dv.at("tokens").as_str());
      CHECK_EQ(hash_bytes(seg.data() + o, n * 4), dv.at("segment_ids").as_str());
      CHECK_EQ(hash_bytes(w.data() + o, n * 4), dv.at("loss_weight").as_str());
    }
    ++checked;
  }
  CHECK_EQ(checked, 4);
}

// T0.3 against the levanter batch-0 fixture (tools/levref/batch.py: tokens, segment ids, loss weights of B = 128
// batch 0 as levanter's trainer builds them), bitwise. Skips until cache/levref/130m/batch0 exists.
NM_TEST(t0_3_levanter, batch0_fixture) {
  s1::converted_or_skip(s1::kFineweb);
  const std::string fx = "cache/levref/130m/batch0/";
  const std::string t = s1::npy_payload(s1::repo_file_or_skip(fx + "tokens.npy")),
                    sg = s1::npy_payload(s1::repo_file_or_skip(fx + "segment_ids.npy")),
                    lw = s1::npy_payload(s1::repo_file_or_skip(fx + "loss_weight.npy"));
  DatasetMeta m = load_dataset_meta(s1::kFineweb);
  TokenFile f;
  f.open(m);
  const std::vector<int32_t> tok = data::assemble_batch(plan_for(128), {&f}, 0);
  std::vector<int32_t> seg(tok.size());
  std::vector<float> w(tok.size());
  data::segment_ids(tok.data(), 128, 4096, m.eos, seg.data());
  data::loss_weights(128, 4096, w.data());
  CHECK(t.size() == tok.size() * 4 && std::memcmp(t.data(), tok.data(), t.size()) == 0);
  CHECK(sg.size() == seg.size() * 4 && std::memcmp(sg.data(), seg.data(), sg.size()) == 0);
  CHECK(lw.size() == w.size() * 4 && std::memcmp(lw.data(), w.data(), lw.size()) == 0);
}
