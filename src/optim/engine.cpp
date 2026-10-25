// Optimizer engine (engine.hpp): host plan, device tables, and the recorded phases of one step.
#include "optim/engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <numeric>

namespace nm::opt {

namespace {

constexpr size_t kAlign = 256;
size_t align_up(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

// Host image of a device blob: aligned appends, one copy, then device addresses by offset.
class Blob {
 public:
  template <class T>
  size_t add(const std::vector<T>& v) {
    const size_t off = align_up(bytes_.size());
    bytes_.resize(off + v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(bytes_.data() + off, v.data(), v.size() * sizeof(T));
    return off;
  }
  size_t size() const { return std::max<size_t>(bytes_.size(), 1); }
  const uint8_t* data() const { return bytes_.data(); }
  std::vector<uint8_t>& raw() { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

template <class T>
T at(void* base, size_t off) {
  return reinterpret_cast<T>(static_cast<uint8_t*>(base) + off);
}

bool rows_owned(const EngTensor& t, int64_t row) {
  for (const auto& [b, e] : t.owned)
    if (row >= b && row < e) return true;
  return false;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ groups
std::vector<EngGroup> baseline_groups(const Config& c) {
  EngGroup mu;
  mu.name = "muonh";
  mu.rule = OptRule::MuonH;
  mu.lr_table = 0;
  mu.momentum = c.optim_momentum;
  mu.max_grad_norm = c.optim_max_grad_norm;
  EngGroup ah;
  ah.name = "adamh";
  ah.rule = OptRule::AdamH;
  ah.lr_table = 0;   // AdamH uses the main learning rate (optim.md §5.1)
  ah.b1 = c.optim_beta1;
  ah.b2 = c.optim_beta2;
  ah.eps = c.optim_eps;
  ah.max_grad_norm = c.optim_max_grad_norm;
  EngGroup ad = ah;
  ad.name = "adam";
  ad.rule = OptRule::Adam;
  ad.lr_table = 1;
  ad.weight_decay = 0.0;   // optim.md §7: no weight decay anywhere in the baseline
  return {mu, ah, ad};
}

optk::GroupDev group_dev(const EngGroup& e, int clip_index) {
  optk::GroupDev d{};
  d.one_minus_b1 = static_cast<float>(1.0 - e.b1);   // optim.md §5.3 rule 1
  d.b1 = static_cast<float>(e.b1);
  d.one_minus_b2 = static_cast<float>(1.0 - e.b2);
  d.b2 = static_cast<float>(e.b2);
  d.eps = static_cast<float>(e.eps);
  d.momentum = static_cast<float>(e.momentum);
  d.lr_scale = static_cast<float>(e.lr_scale);
  d.weight_decay = static_cast<float>(e.weight_decay);
  d.lr_table = e.lr_table;
  d.clip = clip_index;
  d.rule = static_cast<int32_t>(e.rule);
  d.pad = 0;
  return d;
}

// ------------------------------------------------------------------------------------------------ engine
Engine::Engine(std::vector<EngGroup> groups, std::vector<EngTensor> tensors, std::vector<RowPartition> row_partitions,
               std::vector<const uint8_t*> touched_rows, EngOptions opt)
    : groups_(std::move(groups)),
      tensors_(std::move(tensors)),
      rps_(std::move(row_partitions)),
      touched_(std::move(touched_rows)),
      opt_(opt),
      gemm_(opt.ns) {
  touched_.resize(rps_.size(), nullptr);
  build();
  build_tables_on_device();
}

Engine::~Engine() {
  if (tables_blob_) cudaFree(tables_blob_);
  if (lr_tab_dev_) cudaFree(lr_tab_dev_);
  if (own_ws_ && ws_) cudaFree(ws_);
}

void Engine::build() {
  NM_REQUIRE(opt_.ns_steps >= 1, "ns_steps must be >= 1");
  // clip groups
  std::map<std::string, int> clip_of;
  for (const EngGroup& g : groups_) {
    const std::string cn = g.clip_group.empty() ? g.name : g.clip_group;
    auto it = clip_of.find(cn);
    if (it == clip_of.end()) {
      it = clip_of.emplace(cn, n_clip_++).first;
      clip_names_.push_back(cn);
      clip_max_.push_back(static_cast<float>(g.max_grad_norm));
    } else {
      NM_REQUIRE(clip_max_[it->second] == static_cast<float>(g.max_grad_norm),
                 "groups sharing clip group " + cn + " have different max_grad_norm");
    }
    group_clip_.push_back(it->second);
  }
  // One bias-correction table pair (set_tables) serves every Adam and AdamH group: their betas must agree.
  const EngGroup* beta_ref = nullptr;
  for (const EngGroup& g : groups_) {
    if (g.rule != OptRule::Adam && g.rule != OptRule::AdamH) continue;
    if (!beta_ref) beta_ref = &g;
    NM_REQUIRE(g.b1 == beta_ref->b1 && g.b2 == beta_ref->b2,
               "groups " + beta_ref->name + " and " + g.name +
                   " have different b1 / b2, but the engine has one bias-correction table pair");
  }
  // tensors
  for (size_t i = 0; i < tensors_.size(); ++i) {
    const EngTensor& t = tensors_[i];
    NM_REQUIRE(t.group >= 0 && t.group < static_cast<int>(groups_.size()), "tensor " + t.name + ": bad group");
    NM_REQUIRE(t.rows > 0 && t.cols > 0 && t.cols <= INT32_MAX, "tensor " + t.name + ": bad shape");
    NM_REQUIRE(t.row_partition < static_cast<int>(rps_.size()), "tensor " + t.name + ": bad row partition");
    for (const auto& [b, e] : t.owned)
      NM_REQUIRE(b >= 0 && b < e && e <= t.rows && b >= t.base_row, "tensor " + t.name + ": bad owned range");
    const OptRule r = groups_[t.group].rule;
    if (t.owned.empty() || r == OptRule::Frozen) continue;
    NM_REQUIRE(t.p32 && t.g && t.s0, "tensor " + t.name + ": p32, g and s0 are required");
    if (r != OptRule::MuonH) NM_REQUIRE(t.s1, "tensor " + t.name + ": s1 (nu) is required");
    if (r == OptRule::MuonH) NM_REQUIRE(!t.mats.empty() && t.row_partition < 0, "MuonH tensor " + t.name +
                                                                                    " needs matrices, no row partition");
    // AdamH renormalizes over the whole tensor (or block): a lazy update would drop the untouched rows from ||p||
    // and ||a||, which levanter's whole-matrix norm keeps. Not defined, so refused.
    if (r == OptRule::AdamH && t.row_partition >= 0)
      NM_REQUIRE(touched_[t.row_partition] == nullptr,
                 "AdamH tensor " + t.name + ": lazy (touched-row) updates are not supported (the norms are whole-tensor)");
  }

  // ---- slices
  auto block_of = [&](const EngTensor& t, int64_t row) -> int {
    if (t.row_partition < 0) return -1;
    const RowPartition& rp = rps_[t.row_partition];
    int base = 0;
    for (int i = 0; i < t.row_partition; ++i) base += static_cast<int>(rps_[i].blocks.size());
    return base + rp.block_of_row.at(row);
  };
  auto add_rows = [&](int ti, int64_t rb, int64_t re, const std::function<int(int)>& slot_of) {
    const EngTensor& t = tensors_[ti];
    if (t.row_partition >= 0) {
      NM_REQUIRE(t.cols <= optk::kSliceElems, "row-partitioned tensor " + t.name + " wider than a slice");
      const int64_t per = optk::kSliceElems / t.cols;
      int64_t r = rb;
      while (r < re) {
        const int blk = block_of(t, r);
        int64_t e = r + 1;
        while (e < re && e - r < per && block_of(t, e) == blk) ++e;
        slices_.push_back({ti, r, 0, static_cast<int>((e - r) * t.cols), blk, slot_of(blk)});
        r = e;
      }
      return;
    }
    for (int64_t x = rb * t.cols; x < re * t.cols; x += optk::kSliceElems) {
      const int64_t len = std::min<int64_t>(optk::kSliceElems, re * t.cols - x);
      slices_.push_back({ti, x / t.cols, static_cast<int>(x % t.cols), static_cast<int>(len), -1, slot_of(-1)});
    }
  };
  auto no_slot = [](int) { return -1; };
  // clip sums: slices ordered by clip group
  sumsq_begin_ = 0;
  clip_seg_.push_back(0);
  for (int ci = 0; ci < n_clip_; ++ci) {
    for (size_t ti = 0; ti < tensors_.size(); ++ti) {
      const EngTensor& t = tensors_[ti];
      if (group_clip_[t.group] != ci || groups_[t.group].rule == OptRule::Frozen) continue;
      for (const auto& [b, e] : t.owned) add_rows(static_cast<int>(ti), b, e, no_slot);
    }
    clip_seg_.push_back(static_cast<int64_t>(slices_.size()) - sumsq_begin_);
  }
  sumsq_end_ = static_cast<int>(slices_.size());
  sumsq_of_tensor_.assign(tensors_.size(), {0, 0});
  for (int i = sumsq_begin_; i < sumsq_end_;) {
    int j = i;
    while (j < sumsq_end_ && slices_[j].tensor == slices_[i].tensor) ++j;
    sumsq_of_tensor_[slices_[i].tensor] = {i, j};   // a tensor's slices are consecutive (one clip group)
    i = j;
  }
  // Adam
  adam_begin_ = sumsq_end_;
  for (size_t ti = 0; ti < tensors_.size(); ++ti) {
    const EngTensor& t = tensors_[ti];
    if (groups_[t.group].rule != OptRule::Adam) continue;
    for (const auto& [b, e] : t.owned) add_rows(static_cast<int>(ti), b, e, no_slot);
  }
  adam_end_ = static_cast<int>(slices_.size());
  // AdamH: one norm slot per tensor, plus one per own-norm row block
  adamh_begin_ = adam_end_;
  for (size_t ti = 0; ti < tensors_.size(); ++ti) {
    const EngTensor& t = tensors_[ti];
    if (groups_[t.group].rule != OptRule::AdamH) continue;
    const int tensor_slot = n_slots_++;
    std::map<int, int> block_slot;
    if (t.row_partition >= 0) {
      int base = 0;
      for (int i = 0; i < t.row_partition; ++i) base += static_cast<int>(rps_[i].blocks.size());
      const auto& blocks = rps_[t.row_partition].blocks;
      for (size_t b = 0; b < blocks.size(); ++b)
        if (blocks[b].own_norm) block_slot[base + static_cast<int>(b)] = n_slots_++;
    }
    auto slot_of = [&](int blk) {
      auto it = block_slot.find(blk);
      return it == block_slot.end() ? tensor_slot : it->second;
    };
    for (const auto& [b, e] : t.owned) add_rows(static_cast<int>(ti), b, e, slot_of);
  }
  adamh_end_ = static_cast<int>(slices_.size());
  {
    std::vector<int32_t> order(adamh_end_ - adamh_begin_);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return slices_[adamh_begin_ + a].slot < slices_[adamh_begin_ + b].slot;
    });
    slot_idx_ = order;
    slot_seg_.assign(n_slots_ + 1, 0);
    for (int32_t j : order) ++slot_seg_[slices_[adamh_begin_ + j].slot + 1];
    for (int s = 0; s < n_slots_; ++s) slot_seg_[s + 1] += slot_seg_[s];
  }

  // ---- MuonH matrices, grouped by oriented shape, chunked
  struct Pending { int tensor; const EngMatrix* mat; int64_t m, n; };
  std::vector<std::pair<std::pair<int64_t, int64_t>, std::vector<Pending>>> shapes;
  for (size_t ti = 0; ti < tensors_.size(); ++ti) {
    const EngTensor& t = tensors_[ti];
    if (groups_[t.group].rule != OptRule::MuonH) continue;
    for (const EngMatrix& mx : t.mats) {
      NM_REQUIRE(mx.out > 0 && (mx.row_map.empty() || static_cast<int64_t>(mx.row_map.size()) == mx.out),
                 "matrix " + mx.name + ": bad rows");
      auto row = [&](int64_t r) { return mx.row_map.empty() ? mx.row0 + r : int64_t(mx.row_map[r]); };
      if (!rows_owned(t, row(0))) continue;
      for (int64_t r = 0; r < mx.out; ++r)
        NM_REQUIRE(row(r) < t.rows && rows_owned(t, row(r)),
                   "matrix " + mx.name + " is not owned whole by this rank (ZeRO-1 owns whole MuonH matrices)");
      const int64_t m = std::min(mx.out, t.cols), n = std::max(mx.out, t.cols);
      auto it = std::find_if(shapes.begin(), shapes.end(), [&](const auto& s) { return s.first == std::pair{m, n}; });
      if (it == shapes.end()) shapes.push_back({{m, n}, {}}), it = shapes.end() - 1;
      it->second.push_back({static_cast<int>(ti), &mx, m, n});
    }
  }
  tile0_.push_back(0);
  for (const auto& [mn, list] : shapes) {
    const auto [m, n] = mn;
    const size_t per = size_t(2 * m * n + 3 * m * m) * sizeof(float);
    const int cap = static_cast<int>(std::max<size_t>(1, opt_.ns_chunk_bytes / per));
    for (size_t c0 = 0; c0 < list.size(); c0 += cap) {
      const int cnt = static_cast<int>(std::min<size_t>(cap, list.size() - c0));
      Chunk ch;
      ch.m_begin = static_cast<int>(mats_.size());
      ch.tile_begin = tile0_.back();
      ch.m = m, ch.n = n;
      for (int j = 0; j < cnt; ++j) {
        const Pending& pm = list[c0 + j];
        const EngTensor& t = tensors_[pm.tensor];
        MatH h;
        h.tensor = pm.tensor;
        h.row0 = pm.mat->row0;
        h.map = -1;
        if (!pm.mat->row_map.empty()) {
          h.map = static_cast<int>(row_maps_.size());
          row_maps_.push_back(pm.mat->row_map);
        }
        h.out = static_cast<int32_t>(pm.mat->out);
        h.in = static_cast<int32_t>(t.cols);
        h.trans = pm.mat->out > t.cols;
        h.x_off = int64_t(j) * m * n;
        mats_.push_back(h);
        tile0_.push_back(tile0_.back() + ((h.out + optk::kTile - 1) / optk::kTile) *
                                             ((h.in + optk::kTile - 1) / optk::kTile));
      }
      ch.m_end = static_cast<int>(mats_.size());
      ch.tile_end = tile0_.back();
      ch.ns = optk::plan_ns_iteration(gemm_, m, n, cnt, static_cast<float>(opt_.ns_a), static_cast<float>(opt_.ns_b),
                                      static_cast<float>(opt_.ns_c), opt_.ns_symmetric, opt_.ns_fold_c);
      ns_x_elems_ = std::max(ns_x_elems_, int64_t(cnt) * m * n);
      ns_a_elems_ = std::max(ns_a_elems_, int64_t(cnt) * m * m);
      chunks_.push_back(ch);
    }
  }
  mat_count_ = mats_.size();
}

void Engine::build_tables_on_device() {
  const size_t nt = tensors_.size();
  Blob blob;
  std::vector<float*> p32(nt), s0(nt), s1(nt), uo(nt);
  std::vector<bf16_t*> p16(nt);
  std::vector<const float*> g(nt);
  std::vector<int64_t> cols(nt), base(nt);
  std::vector<int32_t> grp(nt), rp(nt);
  std::vector<float> gs(nt), us(nt);
  for (size_t i = 0; i < nt; ++i) {
    const EngTensor& t = tensors_[i];
    p32[i] = t.p32, p16[i] = t.p16, g[i] = t.g, s0[i] = t.s0, s1[i] = t.s1;
    cols[i] = t.cols, base[i] = t.base_row, grp[i] = t.group, rp[i] = t.row_partition;
    gs[i] = t.grad_scale, us[i] = t.update_scale;
    uo[i] = t.u_out;
  }
  const size_t o_p32 = blob.add(p32), o_p16 = blob.add(p16), o_g = blob.add(g), o_s0 = blob.add(s0),
               o_s1 = blob.add(s1), o_cols = blob.add(cols), o_base = blob.add(base), o_grp = blob.add(grp),
               o_gs = blob.add(gs), o_us = blob.add(us), o_rp = blob.add(rp), o_uo = blob.add(uo);
  // slices
  const size_t ns = slices_.size();
  std::vector<int32_t> s_t(ns), s_col(ns), s_len(ns), s_blk(ns), s_slot(ns);
  std::vector<int64_t> s_row(ns);
  for (size_t i = 0; i < ns; ++i) {
    s_t[i] = slices_[i].tensor, s_row[i] = slices_[i].row, s_col[i] = slices_[i].col, s_len[i] = slices_[i].len;
    s_blk[i] = slices_[i].block, s_slot[i] = slices_[i].slot;
  }
  const size_t o_st = blob.add(s_t), o_srow = blob.add(s_row), o_scol = blob.add(s_col), o_slen = blob.add(s_len),
               o_sblk = blob.add(s_blk), o_sslot = blob.add(s_slot);
  // row partitions
  std::vector<int32_t> b_rp, b_lr, b_frozen;
  for (size_t r = 0; r < rps_.size(); ++r)
    for (const RowBlock& b : rps_[r].blocks) {
      b_rp.push_back(static_cast<int32_t>(r));
      b_lr.push_back(b.lr_table);
      b_frozen.push_back(b.frozen ? 0 : (b.freeze_from_step >= 0 ? b.freeze_from_step : INT32_MAX));
    }
  std::vector<const uint8_t*> touched = touched_;
  const size_t o_touch = blob.add(touched), o_brp = blob.add(b_rp), o_blr = blob.add(b_lr),
               o_bfr = blob.add(b_frozen);
  // matrices
  const size_t nm = mats_.size();
  std::vector<int32_t> m_t(nm), m_out(nm), m_in(nm);
  std::vector<int64_t> m_row0(nm), m_xoff(nm);
  std::vector<uint8_t> m_tr(nm);
  for (size_t i = 0; i < nm; ++i) {
    m_t[i] = mats_[i].tensor, m_out[i] = mats_[i].out, m_in[i] = mats_[i].in, m_row0[i] = mats_[i].row0;
    m_xoff[i] = mats_[i].x_off, m_tr[i] = mats_[i].trans;
  }
  std::vector<size_t> o_maps;
  for (const auto& rm : row_maps_) o_maps.push_back(blob.add(rm));
  const size_t o_mt = blob.add(m_t), o_mout = blob.add(m_out), o_min = blob.add(m_in), o_mrow0 = blob.add(m_row0),
               o_mx = blob.add(m_xoff), o_mtr = blob.add(m_tr), o_tile0 = blob.add(tile0_);
  const size_t o_mapptr = blob.add(std::vector<const int32_t*>(nm, nullptr));
  // groups and segments
  std::vector<optk::GroupDev> gd(groups_.size());
  for (size_t i = 0; i < groups_.size(); ++i) gd[i] = group_dev(groups_[i], group_clip_[i]);
  const size_t o_gd = blob.add(gd), o_cmax = blob.add(clip_max_), o_cseg = blob.add(clip_seg_),
               o_sidx = blob.add(slot_idx_), o_sseg = blob.add(slot_seg_);

  NM_CUDA(cudaMalloc(&tables_blob_, blob.size()));
  // pointer fix-up of the row-map pointer array before the copy
  for (size_t i = 0; i < nm; ++i) {
    const int32_t* p = mats_[i].map >= 0 ? at<const int32_t*>(tables_blob_, o_maps[mats_[i].map]) : nullptr;
    std::memcpy(blob.raw().data() + o_mapptr + i * sizeof(p), &p, sizeof(p));
  }
  NM_CUDA(cudaMemcpy(tables_blob_, blob.data(), blob.size(), cudaMemcpyHostToDevice));
  void* B = tables_blob_;
  tt_ = optk::TensorTab{at<float* const*>(B, o_p32), at<bf16_t* const*>(B, o_p16), at<const float* const*>(B, o_g),
                        at<float* const*>(B, o_s0),  at<float* const*>(B, o_s1),   at<const int64_t*>(B, o_cols),
                        at<const int64_t*>(B, o_base), at<const int32_t*>(B, o_grp), at<const float*>(B, o_gs),
                        at<const float*>(B, o_us),   at<const int32_t*>(B, o_rp),  at<float* const*>(B, o_uo)};
  st_ = optk::SliceTab{at<const int32_t*>(B, o_st),   at<const int64_t*>(B, o_srow), at<const int32_t*>(B, o_scol),
                       at<const int32_t*>(B, o_slen), at<const int32_t*>(B, o_sblk), at<const int32_t*>(B, o_sslot)};
  rt_ = optk::RowPartTab{at<const uint8_t* const*>(B, o_touch), at<const int32_t*>(B, o_brp),
                         at<const int32_t*>(B, o_blr), at<const int32_t*>(B, o_bfr)};
  mt_ = optk::MatTab{at<const int32_t*>(B, o_mt),   at<const int64_t*>(B, o_mrow0), at<const int32_t* const*>(B, o_mapptr),
                     at<const int32_t*>(B, o_mout), at<const int32_t*>(B, o_min),   at<const uint8_t*>(B, o_mtr),
                     at<const int64_t*>(B, o_mx),   at<const int64_t*>(B, o_tile0)};
  groups_dev_ = at<optk::GroupDev*>(B, o_gd);
  clip_max_dev_ = at<float*>(B, o_cmax);
  clip_seg_dev_ = at<int64_t*>(B, o_cseg);
  slot_idx_dev_ = at<int32_t*>(B, o_sidx);
  slot_seg_dev_ = at<int64_t*>(B, o_sseg);
}

size_t Engine::workspace_bytes() const {
  const size_t ns = slices_.size(), ntile = static_cast<size_t>(tile0_.back()), nm = mats_.size();
  size_t b = 0;
  b += align_up(2 * ns * sizeof(double));
  b += align_up(2 * ntile * sizeof(double));
  b += align_up(nm * sizeof(double)) + align_up(2 * nm * sizeof(double)) + align_up(nm * sizeof(double));
  b += align_up(n_clip_ * sizeof(double)) + align_up(2 * n_slots_ * sizeof(double)) + align_up(n_slots_ * sizeof(double));
  b += align_up(n_clip_ * sizeof(optk::ClipDev));
  b += 2 * align_up(ns_x_elems_ * sizeof(float)) + 3 * align_up(ns_a_elems_ * sizeof(float));
  b += align_up(gemm_.workspace_bytes());
  return b;
}

void Engine::bind(void* workspace) {
  const size_t bytes = workspace_bytes();
  if (!workspace) {
    NM_CUDA(cudaMalloc(&ws_, bytes));
    own_ws_ = true;
  } else {
    ws_ = workspace;
  }
  uint8_t* p = static_cast<uint8_t*>(ws_);
  auto take = [&](size_t n) {
    uint8_t* r = p;
    p += align_up(n);
    return r;
  };
  const size_t ns = slices_.size(), ntile = static_cast<size_t>(tile0_.back()), nm = mats_.size();
  slice_partial_ = reinterpret_cast<double*>(take(2 * ns * sizeof(double)));
  tile_partial_ = reinterpret_cast<double*>(take(2 * ntile * sizeof(double)));
  mat_u2_ = reinterpret_cast<double*>(take(nm * sizeof(double)));
  mat_po_ = reinterpret_cast<double*>(take(2 * nm * sizeof(double)));
  mat_w_ = reinterpret_cast<double*>(take(nm * sizeof(double)));
  clip_sumsq_ = reinterpret_cast<double*>(take(n_clip_ * sizeof(double)));
  slot_pa_ = reinterpret_cast<double*>(take(2 * n_slots_ * sizeof(double)));
  slot_w_ = reinterpret_cast<double*>(take(n_slots_ * sizeof(double)));
  clip_dev_ = reinterpret_cast<optk::ClipDev*>(take(n_clip_ * sizeof(optk::ClipDev)));
  ns_x_ = reinterpret_cast<float*>(take(ns_x_elems_ * sizeof(float)));
  ns_x2_ = reinterpret_cast<float*>(take(ns_x_elems_ * sizeof(float)));
  ns_a_ = reinterpret_cast<float*>(take(ns_a_elems_ * sizeof(float)));
  ns_ca_ = reinterpret_cast<float*>(take(ns_a_elems_ * sizeof(float)));
  ns_b_ = reinterpret_cast<float*>(take(ns_a_elems_ * sizeof(float)));
  gemm_ws_ = take(gemm_.workspace_bytes());
  NM_REQUIRE(static_cast<size_t>(p - static_cast<uint8_t*>(ws_)) <= bytes, "optimizer workspace overflow");
  gemm_.bind_workspace(gemm_ws_);
}

void Engine::set_tables(const std::vector<const float*>& lr_tables, const float* bc1, const float* bc2,
                        int32_t* step, int64_t n_steps) {
  NM_REQUIRE(n_steps >= 1 && n_steps <= INT32_MAX, "optimizer: tables need 1 .. 2^31 - 1 entries");
  int need = 0;
  for (const EngGroup& g : groups_) need = std::max(need, g.lr_table + 1);
  for (const RowPartition& r : rps_)
    for (const RowBlock& b : r.blocks) need = std::max(need, b.lr_table + 1);
  NM_REQUIRE(static_cast<int>(lr_tables.size()) >= need, "optimizer: missing lr tables");
  NM_REQUIRE(bc1 && bc2 && step, "optimizer: bias-correction tables and the step counter are required");
  if (lr_tab_dev_) cudaFree(lr_tab_dev_);
  NM_CUDA(cudaMalloc(&lr_tab_dev_, std::max<size_t>(1, lr_tables.size()) * sizeof(float*)));
  NM_CUDA(cudaMemcpy(lr_tab_dev_, lr_tables.data(), lr_tables.size() * sizeof(float*), cudaMemcpyHostToDevice));
  tb_ = optk::Tables{lr_tab_dev_, bc1, bc2, step, static_cast<int32_t>(n_steps)};
  lr_main_ = lr_tables.empty() ? nullptr : lr_tables[0];
  lr_adam_ = lr_tables.size() > 1 ? lr_tables[1] : nullptr;
  tables_set_ = true;
}

void Engine::prepare(cudaStream_t s) {
  NM_REQUIRE(ws_, "Engine::prepare before bind()");
  gemm_.choose(opt_.timed_gemm, s);
  gemm_.bind_workspace(gemm_ws_);
  prepared_ = true;
}

std::string Engine::save_gemm_algos() const { return gemm_.save(); }
void Engine::load_gemm_algos(const std::string& bytes) {
  gemm_.load(bytes);
  prepared_ = ws_ != nullptr;
}

void Engine::record_sumsq(cudaStream_t s) {
  NM_REQUIRE(prepared_ && tables_set_, "optimizer engine: bind(), set_tables() and prepare() come first");
  optk::sumsq_slices(tt_, st_, rt_, tb_, sumsq_begin_, sumsq_end_, slice_partial_, s);
  record_clip_sums(s);
}

void Engine::record_sumsq_tensor(int tensor, cudaStream_t s) {
  NM_REQUIRE(prepared_ && tables_set_, "optimizer engine: bind(), set_tables() and prepare() come first");
  const auto [b, e] = sumsq_of_tensor_.at(tensor);
  optk::sumsq_slices(tt_, st_, rt_, tb_, b, e, slice_partial_ + (b - sumsq_begin_), s);
}

void Engine::record_clip_sums(cudaStream_t s) {
  optk::segment_sum(slice_partial_, nullptr, clip_seg_dev_, n_clip_, 1, clip_sumsq_, 0, s);
}

void Engine::record_update(cudaStream_t s, const optk::ClipDev* clip_override) {
  const optk::ClipDev* clip = clip_override ? clip_override : clip_dev_;
  if (!clip_override) optk::clip_state(clip_sumsq_, clip_max_dev_, n_clip_, clip_dev_, s);
  optk::adam_update(tt_, st_, rt_, groups_dev_, clip, tb_, adam_begin_, adam_end_, s);
  const float eps = static_cast<float>(opt_.muon_eps);
  for (const Chunk& ch : chunks_) {
    const int cnt = ch.m_end - ch.m_begin;
    float *x = ns_x_, *x2 = ns_x2_;
    optk::muon_momentum(tt_, mt_, groups_dev_, clip, ch.m_begin, ch.m_end, ch.tile_begin, ch.tile_end, x,
                        tile_partial_, s);
    // per-matrix ||u||^2: tile partials of matrix i are [tile0[i], tile0[i+1])
    optk::segment_sum(tile_partial_, nullptr, mt_.tile0 + ch.m_begin, cnt, 1, mat_u2_ + ch.m_begin, 0, s);
    optk::ns_normalize(x, cnt, ch.m * ch.n, mat_u2_ + ch.m_begin, eps, s);
    x = optk::run_ns_iteration(gemm_, ch.ns, opt_.ns_steps, x, x2, ns_a_, ns_ca_, ns_b_, s);
    optk::hyperball_norms(tt_, mt_, ch.m_begin, ch.m_end, ch.tile_begin, ch.tile_end, x, tile_partial_, s);
    optk::segment_sum(tile_partial_, nullptr, mt_.tile0 + ch.m_begin, cnt, 2, mat_po_ + 2 * ch.m_begin, 0, s);
    optk::hyperball_w(tt_, mt_, groups_dev_, tb_, ch.m_begin, ch.m_end, ch.tile_begin, ch.tile_end, x, mat_po_,
                      tile_partial_, s);
    optk::segment_sum(tile_partial_, nullptr, mt_.tile0 + ch.m_begin, cnt, 1, mat_w_ + ch.m_begin, 0, s);
    optk::hyperball_apply(tt_, mt_, ch.m_begin, ch.m_end, ch.tile_begin, ch.tile_end, x, mat_po_, mat_w_, s);
  }
  optk::adamh_pass1(tt_, st_, rt_, groups_dev_, clip, tb_, adamh_begin_, adamh_end_, slice_partial_, s);
  optk::segment_sum(slice_partial_, slot_idx_dev_, slot_seg_dev_, n_slots_, 2, slot_pa_, 0, s);
}

void Engine::record_adamh_w(cudaStream_t s) {
  optk::adamh_pass2(tt_, st_, rt_, groups_dev_, tb_, slot_pa_, adamh_begin_, adamh_end_, slice_partial_, s);
  optk::segment_sum(slice_partial_, slot_idx_dev_, slot_seg_dev_, n_slots_, 1, slot_w_, 0, s);
}

void Engine::record_finish(cudaStream_t s, bool advance_step, float* lr_ring) {
  optk::adamh_pass3(tt_, st_, rt_, groups_dev_, tb_, slot_pa_, slot_w_, adamh_begin_, adamh_end_, s);
  if (advance_step) optk::step_advance(const_cast<int32_t*>(tb_.step), tb_.n_steps, lr_main_, lr_adam_, lr_ring, s);
}

void Engine::record_step(cudaStream_t s, const AllReduce& allreduce_sum, bool advance_step, float* lr_ring) {
  const bool ar = opt_.sharded && allreduce_sum;
  NM_REQUIRE(!opt_.sharded || allreduce_sum, "sharded optimizer step without an all-reduce");
  record_sumsq(s);
  if (ar) allreduce_sum(clip_sumsq_, n_clip_, s);
  record_update(s);
  if (ar && n_slots_) allreduce_sum(slot_pa_, 2 * n_slots_, s);
  record_adamh_w(s);
  if (ar && n_slots_) allreduce_sum(slot_w_, n_slots_, s);
  record_finish(s, advance_step, lr_ring);
}

// ------------------------------------------------------------------------------------------------ ownership
namespace {
int64_t spec_rows(const ParamSpec& p) {
  if (p.rows > 0) return p.rows;
  int64_t r = 1;
  for (size_t i = 0; i + 1 < p.shape.size(); ++i) r *= p.shape[i];
  return r;
}
int64_t spec_cols(const ParamSpec& p) { return p.cols > 0 ? p.cols : (p.shape.empty() ? 1 : p.shape.back()); }
int group_index(const std::vector<EngGroup>& groups, const std::string& name) {
  for (size_t i = 0; i < groups.size(); ++i)
    if (groups[i].name == name) return static_cast<int>(i);
  fatal(NM_WHERE, "parameter group '" + name + "' has no optimizer group");
}
// Estimated H100 seconds: NS at ~400 TFLOP/s TF32 (5 iterations: 4 m^2 n + 2 m^3 each), elementwise at ~3 TB/s.
double ns_seconds(int64_t out, int64_t in) {
  const double m = double(std::min(out, in)), n = double(std::max(out, in));
  return 5.0 * (4.0 * m * m * n + 2.0 * m * m * m) / 4e14;
}
double ew_seconds(int64_t numel, double bytes_per_elem) { return double(numel) * bytes_per_elem / 3e12; }
}  // namespace

Ownership plan_ownership_specs(const std::vector<ParamSpec>& params, const std::vector<EngGroup>& groups, int nranks,
                               bool zero1) {
  NM_REQUIRE(nranks >= 1, "nranks must be >= 1");
  Ownership own;
  own.load_per_rank.assign(nranks, 0.0);
  const bool replicated = !zero1 || nranks == 1;
  struct Unit { int param; std::vector<std::pair<int64_t, int64_t>> rows; double cost; };
  std::vector<Unit> units;
  for (size_t i = 0; i < params.size(); ++i) {
    const ParamSpec& p = params[i];
    const int64_t R = spec_rows(p), C = spec_cols(p);
    const OptRule rule = groups[group_index(groups, p.group)].rule;
    if (replicated) {
      own.slices.push_back({static_cast<int>(i), 0, R, -1});
      continue;
    }
    if (rule == OptRule::MuonH) {
      if (p.leaves.empty()) {
        units.push_back({static_cast<int>(i), {{0, R}}, ns_seconds(R, C) + ew_seconds(R * C, 40)});
      } else {
        for (const LeafSlice& l : p.leaves) {
          Unit u{static_cast<int>(i), {}, 0.0};
          int64_t out = 0;
          for (size_t k = 0; k + 1 < l.rows.size(); k += 2) {
            u.rows.push_back({l.rows[k], l.rows[k + 1]});
            out += l.rows[k + 1] - l.rows[k];
          }
          u.cost = ns_seconds(out, C) + ew_seconds(out * C, 40);
          units.push_back(u);
        }
      }
    } else if (p.vocab_sized) {
      const int64_t per = (R + nranks - 1) / nranks;
      for (int r = 0; r < nranks; ++r) {
        const int64_t b = std::min(R, int64_t(r) * per), e = std::min(R, int64_t(r + 1) * per);
        if (b < e) own.slices.push_back({static_cast<int>(i), b, e, r});
        own.load_per_rank[r] += ew_seconds((e - b) * C, rule == OptRule::AdamH ? 44 : 28);
      }
    } else {
      units.push_back({static_cast<int>(i), {{0, R}}, ew_seconds(R * C, 28) + 2e-6});
    }
  }
  std::vector<size_t> order(units.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return units[a].cost > units[b].cost; });
  for (size_t k : order) {
    const int r = static_cast<int>(std::min_element(own.load_per_rank.begin(), own.load_per_rank.end()) -
                                   own.load_per_rank.begin());
    own.load_per_rank[r] += units[k].cost;
    for (const auto& [b, e] : units[k].rows) own.slices.push_back({units[k].param, b, e, r});
  }
  std::stable_sort(own.slices.begin(), own.slices.end(), [](const Ownership::Slice& a, const Ownership::Slice& b) {
    return a.param != b.param ? a.param < b.param : a.row_begin < b.row_begin;
  });
  return own;
}

std::vector<EngTensor> engine_layout(const std::vector<ParamSpec>& params, const std::vector<EngGroup>& groups,
                                     const Ownership& own, int rank) {
  std::vector<EngTensor> out(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    const ParamSpec& p = params[i];
    EngTensor& t = out[i];
    t.name = p.name;
    t.group = group_index(groups, p.group);
    t.rows = spec_rows(p);
    t.cols = spec_cols(p);
    t.row_partition = p.row_partition;
    if (groups[t.group].rule == OptRule::MuonH) {
      if (p.leaves.empty()) {
        t.mats.push_back({p.name, 0, t.rows, {}});
      } else {
        for (const LeafSlice& l : p.leaves) {
          EngMatrix m;
          m.name = l.name;
          if (l.rows.size() == 2) {
            m.row0 = l.rows[0];
            m.out = l.rows[1] - l.rows[0];
          } else {
            for (size_t k = 0; k + 1 < l.rows.size(); k += 2)
              for (int64_t r = l.rows[k]; r < l.rows[k + 1]; ++r) m.row_map.push_back(static_cast<int32_t>(r));
            m.out = static_cast<int64_t>(m.row_map.size());
          }
          t.mats.push_back(std::move(m));
        }
      }
    }
  }
  for (const Ownership::Slice& s : own.slices)
    if (s.owner == rank || s.owner == -1) out[s.param].owned.push_back({s.row_begin, s.row_end});
  return out;
}

}  // namespace nm::opt
