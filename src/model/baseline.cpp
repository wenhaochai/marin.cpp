#include "model/baseline.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "kernels/elementwise/ew_api.hpp"
#include "marincpp/device.hpp"
#include "marincpp/kernels/elementwise.hpp"
#include "marincpp/kernels/reduce.hpp"

namespace nm::model {

// ------------------------------------------------------------------------------------------------ parameter table
std::vector<ParamSpec> baseline_params(const ModelShape& m) {
  std::vector<ParamSpec> ps;
  const int64_t D = m.D, H = m.H, KV = m.KV, hd = m.hd, F = m.F, V = m.V;
  auto add = [&](std::string name, std::vector<int64_t> shape, const char* group, int layer,
                 std::vector<LeafSlice> leaves) -> ParamSpec& {
    ParamSpec p;
    p.name = std::move(name);
    p.shape = std::move(shape);
    p.group = group;
    p.layer = layer;
    p.rows = 1;
    for (size_t i = 0; i + 1 < p.shape.size(); ++i) p.rows *= p.shape[i];
    p.cols = p.shape.back();
    p.scope = std::string(group) == "muonh" ? NormScope::LayerMatrix : NormScope::WholeTensor;
    p.leaves = std::move(leaves);
    ps.push_back(std::move(p));
    return ps.back();
  };
  auto whole = [](const std::string& leaf, int64_t rows) { return std::vector<LeafSlice>{{leaf, {0, rows}}}; };
  for (int l = 0; l < m.L; ++l) {
    const std::string ls = "L" + std::to_string(l) + ".";
    const std::string lv = "transformer.layers." + std::to_string(l) + ".";
    add(ls + "Wqkv", {(H + 2 * KV) * hd, D}, "muonh", l,
        {{lv + "self_attn.q_proj.weight", {0, H * hd}},
         {lv + "self_attn.k_proj.weight", {H * hd, (H + KV) * hd}},
         {lv + "self_attn.v_proj.weight", {(H + KV) * hd, (H + 2 * KV) * hd}}});
    add(ls + "Wo", {D, H * hd}, "muonh", l, whole(lv + "self_attn.o_proj.weight", D));
    {
      LeafSlice gate{lv + "mlp.gate_proj.weight", {}}, up{lv + "mlp.up_proj.weight", {}};
      for (int64_t b = 0; b < F / 64; ++b) {
        gate.rows.push_back(128 * b), gate.rows.push_back(128 * b + 64);
        up.rows.push_back(128 * b + 64), up.rows.push_back(128 * b + 128);
      }
      add(ls + "Wgu", {2 * F, D}, "muonh", l, {gate, up});
    }
    add(ls + "Wdown", {D, F}, "muonh", l, whole(lv + "mlp.down_proj.weight", D));
    for (const char* g : {"input_layernorm", "post_attn_layernorm", "post_attention_layernorm", "post_mlp_layernorm"})
      add(ls + g, {D}, "adam", l, whole(lv + g + ".weight", 1));
    add(ls + "q_norm", {hd}, "adam", l, whole(lv + "self_attn.q_norm.weight", 1));
    add(ls + "k_norm", {hd}, "adam", l, whole(lv + "self_attn.k_norm.weight", 1));
  }
  add("final_norm", {D}, "adam", -1, whole("transformer.norm.weight", 1));
  add("embedding", {V, D}, "adam", -1, whole("embeddings.token_embeddings.weight", V)).vocab_sized = true;
  add("lm_head", {V, D}, "adamh", -1, whole("lm_head.weight", V)).vocab_sized = true;
  return ps;
}

// ------------------------------------------------------------------------------------------------ per-layer state
struct Baseline::L {
  int t_qkv, t_o, t_gu, t_down, t_gin, t_gpa, t_gpre, t_gpm, t_gq, t_gk;   // tensor indices
  bf16_t *x_in, *a, *qkv, *q_rot, *k_rot, *attn_o, *oproj, *x_mid, *m, *gu, *h, *y;
  float *rstd_in, *rstd_pa, *rstd_pre, *rstd_pm, *rstd_q, *rstd_k, *lse;
};

Baseline::Baseline(const BaselineSpec& s) : s_(s) {
  NM_REQUIRE(s_.m.F % 64 == 0, "MLP width must be a multiple of 64 (interleaved gate/up blocks)");
  NM_REQUIRE(s_.global_rows > 0 && s_.micro_rows > 0, "rows must be positive");
  params_ = baseline_params(s_.m);
  offsets_.reserve(params_.size());
  for (const ParamSpec& p : params_) {
    offsets_.push_back(total_);
    total_ += p.rows * p.cols;
  }
  layers_.resize(s_.m.L);
  for (int l = 0; l < s_.m.L; ++l) {
    const std::string ls = "L" + std::to_string(l) + ".";
    L& y = layers_[l];
    y.t_qkv = find(ls + "Wqkv"), y.t_o = find(ls + "Wo"), y.t_gu = find(ls + "Wgu"), y.t_down = find(ls + "Wdown");
    y.t_gin = find(ls + "input_layernorm"), y.t_gpa = find(ls + "post_attn_layernorm");
    y.t_gpre = find(ls + "post_attention_layernorm"), y.t_gpm = find(ls + "post_mlp_layernorm");
    y.t_gq = find(ls + "q_norm"), y.t_gk = find(ls + "k_norm");
  }
  t_final_ = find("final_norm"), t_embed_ = find("embedding"), t_head_ = find("lm_head");

  DeviceContext dummy;   // make_attention ignores the context
  attn_ = make_attention(s_.attn, dummy);
  attn_shape_.total_q = T();
  attn_shape_.rows = s_.micro_rows;
  attn_shape_.S = s_.m.S;
  attn_shape_.H = s_.m.H;
  attn_shape_.KV = s_.m.KV;
  attn_shape_.hd = s_.m.hd;
  attn_shape_.max_segs_per_seq = s_.max_segs;
  attn_shape_.bwd_sched = s_.attn_sched;
  attn_shape_.deterministic = s_.deterministic;
  attn_shape_.io = DType::bf16;
  ce_shape_.T = T();
  ce_shape_.D = s_.m.D;
  ce_shape_.V = s_.m.V;
  ce_shape_.chunk_rows = s_.ce_chunk;
  ce_shape_.logits_store = s_.num.ce_logits_store;
  ce_shape_.io = DType::bf16;
  n_part_ = ew::norm_partial_rows(T());
}

Baseline::~Baseline() = default;

int64_t Baseline::numel(int t) const { return params_[t].rows * params_[t].cols; }

int Baseline::find(const std::string& name) const {
  for (size_t i = 0; i < params_.size(); ++i)
    if (params_[i].name == name) return static_cast<int>(i);
  fatal(NM_WHERE, "no parameter named " + name);
}

float Baseline::loss_scale() const { return 1.0f / (float(s_.global_rows) * float(s_.m.S - 1)); }

// ------------------------------------------------------------------------------------------------ GEMMs
void Baseline::plan_gemms(GemmCache& gemm) {
  const int64_t t = T(), D = s_.m.D, Hd = int64_t(s_.m.H) * s_.m.hd, Q = s_.m.qkv_cols, F = s_.m.F;
  const int64_t NK[4][2] = {{Q, D}, {D, Hd}, {2 * F, D}, {D, F}};   // qkv, o, gu, down: Y[T, N] = X[T, K] W[N, K]^T
  const char* tags[4][3] = {{"L*.qkv.fwd", "L*.qkv.dgrad", "L*.qkv.wgrad"},
                            {"L*.o.fwd", "L*.o.dgrad", "L*.o.wgrad"},
                            {"L*.gu.fwd", "L*.gu.dgrad", "L*.gu.wgrad"},
                            {"L*.down.fwd", "L*.down.dgrad", "L*.down.wgrad"}};
  for (int i = 0; i < 4; ++i) {
    const GemmDesc d[3] = {linear_fwd_desc(t, NK[i][0], NK[i][1], Epilogue::None, tags[i][0]),
                           linear_dgrad_desc(t, NK[i][0], NK[i][1], Epilogue::None, tags[i][1]),
                           linear_wgrad_desc(t, NK[i][0], NK[i][1], s_.num.rounding, tags[i][2])};
    id_fwd_[i] = gemm.plan(d[0]);
    id_dgrad_[i] = gemm.plan(d[1]);
    id_wgrad_[i] = gemm.plan(d[2]);
    for (const GemmDesc& x : d) gemm_scratch_ = std::max(gemm_scratch_, gemm_search_scratch_bytes(x));
  }
  gemm_ = &gemm;
}

// ------------------------------------------------------------------------------------------------ buffers
void Baseline::declare(MemoryPlan& plan, uint64_t gemm_ws) const {
  auto add = [&](const std::string& name, Lifetime life, DType t, std::initializer_list<int64_t> shape,
                 int layer = -1) {
    BufferSpec b;
    b.name = name;
    b.group = name;
    b.life = life;
    b.dtype = t;
    b.ndim = static_cast<int>(shape.size());
    int i = 0;
    for (int64_t v : shape) b.shape[i++] = v;
    b.layer = layer;
    plan.add(std::move(b));
  };
  const auto P = Lifetime::Persistent, MB = Lifetime::Microbatch, A = Lifetime::Activation,
             TR = Lifetime::Transient, W = Lifetime::Workspace;
  const DType f32 = DType::f32, bf = DType::bf16, i32 = DType::i32, u8 = DType::u8, u32 = DType::u32;
  const int64_t t = T(), D = s_.m.D, H = s_.m.H, KV = s_.m.KV, hd = s_.m.hd, F = s_.m.F, S = s_.m.S;
  const int64_t Q = s_.m.qkv_cols, rows = s_.micro_rows, ms = s_.max_segs;
  add("param.f32", P, f32, {total_});
  add("param.bf16", P, bf, {total_});
  add("grad.f32", P, f32, {total_});
  add("rope.cos", P, bf, {S, hd});
  add("rope.sin", P, bf, {S, hd});
  add("loss.mb", P, f32, {kMaxMicro});
  add("loss.gscale", P, f32, {1});
  add("mb.tokens", MB, i32, {rows, S});
  add("mb.targets", MB, i32, {t});
  add("mb.pos", MB, i32, {t});
  add("mb.weight", MB, f32, {t});
  add("mb.cu_seqlens", MB, i32, {rows * ms + 1});
  add("mb.n_segs", MB, i32, {1});
  add("mb.row_seg_start", MB, i32, {rows, ms + 1});
  add("mb.overflow", MB, u32, {1});
  add("mb.loss_tok", MB, f32, {t});
  add("mb.ce_lse", MB, f32, {t});
  add("mb.loss_part", MB, f32, {(t + s_.ce_chunk - 1) / s_.ce_chunk});
  for (int l = 0; l < s_.m.L; ++l) {
    const std::string p = "act.L" + std::to_string(l) + ".";
    for (const char* n : {"x_in", "a", "oproj", "x_mid", "m", "y"}) add(p + n, A, bf, {t, D}, l);
    add(p + "qkv", A, bf, {t, Q}, l);
    add(p + "q_rot", A, bf, {t, H * hd}, l);
    add(p + "k_rot", A, bf, {t, KV * hd}, l);
    add(p + "attn_o", A, bf, {t, H * hd}, l);
    add(p + "gu", A, bf, {t, 2 * F}, l);
    add(p + "h", A, bf, {t, F}, l);
    for (const char* n : {"rstd_in", "rstd_pa", "rstd_pre", "rstd_pm"}) add(p + n, A, f32, {t}, l);
    add(p + "rstd_q", A, f32, {t, H}, l);
    add(p + "rstd_k", A, f32, {t, KV}, l);
    add(p + "lse", A, f32, {H, t}, l);
  }
  add("act.final.x", A, bf, {t, D}, s_.m.L);
  add("act.final.xf", A, bf, {t, D}, s_.m.L);
  add("act.final.rstd", A, f32, {t}, s_.m.L);
  for (const char* n : {"tr.dx", "tr.dA", "tr.dxf", "tr.dy", "tr.dm", "tr.doproj", "tr.zeros"}) add(n, TR, bf, {t, D});
  add("tr.dh", TR, bf, {t, F});
  add("tr.dgu", TR, bf, {t, 2 * F});
  add("tr.dattn", TR, bf, {t, H * hd});
  add("tr.dq", TR, bf, {t, H * hd});
  add("tr.dk", TR, bf, {t, KV * hd});
  add("tr.dqkv", TR, bf, {t, Q});
  add("tr.part_a", TR, f32, {n_part_, D});
  add("tr.part_b", TR, f32, {n_part_, D});
  add("tr.part_q", TR, f32, {n_part_, hd});
  add("tr.part_k", TR, f32, {n_part_, hd});
  add("ws.attn", W, u8, {int64_t(std::max<uint64_t>(1, attn_->workspace_bytes(attn_shape_, true)))});
  add("ws.ce", W, u8, {int64_t(std::max<uint64_t>(1, ce_workspace_bytes(s_.ce, ce_shape_)))});
  add("ws.embed", W, u8, {int64_t(std::max<uint64_t>(1, embed_bwd_workspace_bytes(t, s_.m.V)))});
  add("ws.gemm", W, u8, {int64_t(std::max<uint64_t>(1, gemm_ws))});
}

void* Baseline::bind_buf(const Arena& a, const std::string& name, DType t, std::initializer_list<int64_t> shape) const {
  return a.bind(name, t, shape);
}

void Baseline::bind(const Arena& a, GemmCache& gemm, cudaStream_t s) {
  NM_REQUIRE(gemm_ == &gemm, "plan_gemms() must run on the same GemmCache before bind()");
  const DType f32 = DType::f32, bf = DType::bf16, i32 = DType::i32, u8 = DType::u8, u32 = DType::u32;
  const int64_t t = T(), D = s_.m.D, H = s_.m.H, KV = s_.m.KV, hd = s_.m.hd, F = s_.m.F, S = s_.m.S;
  const int64_t Q = s_.m.qkv_cols, rows = s_.micro_rows, ms = s_.max_segs;
  auto B16 = [&](const std::string& n, std::initializer_list<int64_t> sh) {
    return static_cast<bf16_t*>(a.bind(n, bf, sh));
  };
  auto B32 = [&](const std::string& n, std::initializer_list<int64_t> sh) {
    return static_cast<float*>(a.bind(n, f32, sh));
  };
  auto BI = [&](const std::string& n, std::initializer_list<int64_t> sh) {
    return static_cast<int32_t*>(a.bind(n, i32, sh));
  };
  p32_ = B32("param.f32", {total_});
  p16_ = B16("param.bf16", {total_});
  grad_ = B32("grad.f32", {total_});
  rope_cos_ = B16("rope.cos", {S, hd});
  rope_sin_ = B16("rope.sin", {S, hd});
  loss_mb_ = B32("loss.mb", {kMaxMicro});
  gscale_ = B32("loss.gscale", {1});
  tokens_ = BI("mb.tokens", {rows, S});
  targets_ = BI("mb.targets", {t});
  pos_ = BI("mb.pos", {t});
  weight_ = B32("mb.weight", {t});
  cu_seqlens_ = BI("mb.cu_seqlens", {rows * ms + 1});
  n_segs_ = BI("mb.n_segs", {1});
  row_seg_ = BI("mb.row_seg_start", {rows, ms + 1});
  overflow_ = static_cast<uint32_t*>(a.bind("mb.overflow", u32, {1}));
  loss_tok_ = B32("mb.loss_tok", {t});
  ce_lse_ = B32("mb.ce_lse", {t});
  loss_part_ = B32("mb.loss_part", {(t + s_.ce_chunk - 1) / s_.ce_chunk});
  for (int l = 0; l < s_.m.L; ++l) {
    const std::string p = "act.L" + std::to_string(l) + ".";
    L& y = layers_[l];
    y.x_in = B16(p + "x_in", {t, D}), y.a = B16(p + "a", {t, D}), y.oproj = B16(p + "oproj", {t, D});
    y.x_mid = B16(p + "x_mid", {t, D}), y.m = B16(p + "m", {t, D}), y.y = B16(p + "y", {t, D});
    y.qkv = B16(p + "qkv", {t, Q});
    y.q_rot = B16(p + "q_rot", {t, H * hd});
    y.k_rot = B16(p + "k_rot", {t, KV * hd});
    y.attn_o = B16(p + "attn_o", {t, H * hd});
    y.gu = B16(p + "gu", {t, 2 * F});
    y.h = B16(p + "h", {t, F});
    y.rstd_in = B32(p + "rstd_in", {t}), y.rstd_pa = B32(p + "rstd_pa", {t});
    y.rstd_pre = B32(p + "rstd_pre", {t}), y.rstd_pm = B32(p + "rstd_pm", {t});
    y.rstd_q = B32(p + "rstd_q", {t, H});
    y.rstd_k = B32(p + "rstd_k", {t, KV});
    y.lse = B32(p + "lse", {H, t});
  }
  fin_x_ = B16("act.final.x", {t, D});
  fin_xf_ = B16("act.final.xf", {t, D});
  fin_rstd_ = B32("act.final.rstd", {t});
  dx_ = B16("tr.dx", {t, D}), dA_ = B16("tr.dA", {t, D}), dxf_ = B16("tr.dxf", {t, D}), dy_ = B16("tr.dy", {t, D});
  dm_ = B16("tr.dm", {t, D}), doproj_ = B16("tr.doproj", {t, D}), zeros_ = B16("tr.zeros", {t, D});
  dh_ = B16("tr.dh", {t, F});
  dgu_ = B16("tr.dgu", {t, 2 * F});
  dattn_ = B16("tr.dattn", {t, H * hd});
  dq_ = B16("tr.dq", {t, H * hd});
  dk_ = B16("tr.dk", {t, KV * hd});
  dqkv_ = B16("tr.dqkv", {t, Q});
  part_a_ = B32("tr.part_a", {n_part_, D});
  part_b_ = B32("tr.part_b", {n_part_, D});
  part_q_ = B32("tr.part_q", {n_part_, hd});
  part_k_ = B32("tr.part_k", {n_part_, hd});
  const TensorView va = a.view("ws.attn"), vc = a.view("ws.ce"), ve = a.view("ws.embed"), vg = a.view("ws.gemm");
  ws_attn_ = va.data, ws_ce_ = vc.data, ws_embed_ = ve.data, ws_gemm_ = vg.data;
  ws_embed_bytes_ = ve.nbytes();
  (void)u8;
  gemm.bind_workspace(ws_gemm_);
  attn_->prepare(attn_shape_, true);

  // RoPE tables (DESIGN.md §3.4, host double method) and the loss scale; synchronous, before any capture.
  std::vector<bf16_t> cos_h(S * hd), sin_h(S * hd);
  std::vector<float> inv(hd / 2);
  build_rope_tables(static_cast<int>(S), static_cast<int>(hd), 10000.0, cos_h.data(), sin_h.data(), inv.data());
  NM_CUDA(cudaMemcpyAsync(rope_cos_, cos_h.data(), cos_h.size() * 2, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemcpyAsync(rope_sin_, sin_h.data(), sin_h.size() * 2, cudaMemcpyHostToDevice, s));
  const float gs = loss_scale();
  NM_CUDA(cudaMemcpyAsync(gscale_, &gs, 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemsetAsync(zeros_, 0, size_t(t) * D * 2, s));
  NM_CUDA(cudaStreamSynchronize(s));
}

void Baseline::zero_grads(cudaStream_t s) const {
  NM_CUDA(cudaMemsetAsync(grad_, 0, size_t(total_) * 4, s));
  NM_CUDA(cudaMemsetAsync(loss_mb_, 0, size_t(kMaxMicro) * 4, s));
}

void Baseline::reduce_gain(const float* partial, int tensor, int width, cudaStream_t s) const {
  reduce_partials_rows(partial, static_cast<int>(n_part_), width, g32(tensor), true,
                       s_.num.rounding == Rounding::Levanter, s);
}

// ------------------------------------------------------------------------------------------------ one microbatch
void Baseline::record_microbatch(int index, const int32_t* tokens, cudaStream_t s) {
  NM_REQUIRE(index >= 0 && index < kMaxMicro, "microbatch index out of range");
  const int64_t t = T();
  const int D = s_.m.D, H = s_.m.H, KV = s_.m.KV, hd = s_.m.hd, F = s_.m.F, S = s_.m.S, V = s_.m.V;
  const int64_t Q = s_.m.qkv_cols;
  const Rounding r = s_.num.rounding;
  const float eps = 1e-5f;
  // levanter: q pre-scaled in bf16 by bf16(1/sqrt(hd)) and FA3 gets scale 1; accurate: FA3 takes the scale
  const bool lev = r == Rounding::Levanter;
  const float q_scale = lev ? s_.m.attn_scale_bf16 : 1.0f;
  const float sm_scale = lev ? 1.0f : s_.m.attn_scale_bf16;
  const GemmCache& g = *gemm_;

  // document metadata: targets, weights, positions, segment tables
  DocMetaArgs dm{tokens, nullptr, s_.micro_rows, S, s_.eos_id, s_.max_segs, targets_, weight_, pos_,
                 cu_seqlens_, n_segs_, row_seg_, overflow_};
  doc_meta(dm, s);

  // ---------------------------------------------------------------- forward
  embed_fwd(w16(t_embed_), tokens, layers_[0].x_in, t, D, s);
  {
    RmsNormFwd a{layers_[0].x_in, w16(layers_[0].t_gin), layers_[0].a, layers_[0].rstd_in, t, D, eps};
    rmsnorm_fwd(a, r, s);
  }
  for (int l = 0; l < s_.m.L; ++l) {
    L& y = layers_[l];
    const bool last = l == s_.m.L - 1;
    g.run(id_fwd_[0], y.a, w16(y.t_qkv), nullptr, y.qkv, s);
    QkNormRopeFwd qf{y.qkv, Q, w16(y.t_gq), w16(y.t_gk), rope_cos_, rope_sin_, pos_, y.q_rot, y.k_rot,
                     y.rstd_q, y.rstd_k, t, H, KV, hd, eps, q_scale};
    qknorm_rope_fwd(qf, r, s);
    AttnFwdArgs af{};
    af.q = y.q_rot, af.q_ld = int64_t(H) * hd;
    af.k = y.k_rot, af.k_ld = int64_t(KV) * hd;
    af.v = y.qkv + int64_t(H + KV) * hd, af.v_ld = Q;
    af.o = y.attn_o, af.o_ld = int64_t(H) * hd;
    af.lse = y.lse;
    af.cu_seqlens = cu_seqlens_, af.n_segs = n_segs_, af.row_seg_start = row_seg_;
    af.softmax_scale = sm_scale;
    attn_->fwd(attn_shape_, af, ws_attn_, s);
    g.run(id_fwd_[1], y.attn_o, w16(y.t_o), nullptr, y.oproj, s);
    SandwichTailFwd ta{y.oproj, y.x_in, w16(y.t_gpa), w16(y.t_gpre), y.x_mid, y.m, y.rstd_pa, y.rstd_pre, t, D, eps};
    sandwich_tail_fwd(ta, r, s);
    g.run(id_fwd_[2], y.m, w16(y.t_gu), nullptr, y.gu, s);
    swiglu_fwd(y.gu, y.h, t, F, r, s);
    g.run(id_fwd_[3], y.h, w16(y.t_down), nullptr, y.y, s);
    SandwichTailFwd tm{y.y, y.x_mid, w16(y.t_gpm),
                       last ? w16(t_final_) : w16(layers_[l + 1].t_gin),
                       last ? fin_x_ : layers_[l + 1].x_in, last ? fin_xf_ : layers_[l + 1].a,
                       y.rstd_pm, last ? fin_rstd_ : layers_[l + 1].rstd_in, t, D, eps};
    sandwich_tail_fwd(tm, r, s);
  }

  // ---------------------------------------------------------------- head: CE forward + its gradient
  CeArgs ca{};
  ca.x = fin_xf_, ca.w = w16(t_head_), ca.targets = targets_, ca.weight = weight_, ca.grad_scale = gscale_;
  ca.logit_bias = nullptr, ca.loss_tok = loss_tok_, ca.lse = ce_lse_, ca.loss_partial = loss_part_;
  ca.dx = dxf_, ca.accumulate_dx = false, ca.dw = g32(t_head_), ca.dw_microbatch = nullptr;
  ca.dw_round = s_.num.ce_dw_round, ca.forward_only = false;
  fused_linear_ce(s_.ce, r, ce_shape_, ca, ws_ce_, *gemm_, s);
  sum_fixed(loss_part_, (t + s_.ce_chunk - 1) / s_.ce_chunk, loss_mb_ + index, s);

  // ---------------------------------------------------------------- backward
  // dres: residual gradient w.r.t. the layer's output stream; dan: gradient w.r.t. the next pre-norm's output
  for (int l = s_.m.L - 1; l >= 0; --l) {
    L& y = layers_[l];
    const bool last = l == s_.m.L - 1;
    // MLP tail: through the next pre-norm (or the final norm) and the post-MLP norm
    SandwichTailBwd bm{};
    bm.dx_out = last ? zeros_ : dx_;
    bm.da_next = last ? dxf_ : dA_;
    bm.x_out = last ? fin_x_ : layers_[l + 1].x_in;
    bm.g_next = last ? w16(t_final_) : w16(layers_[l + 1].t_gin);
    bm.rstd_next = last ? fin_rstd_ : layers_[l + 1].rstd_in;
    bm.branch = y.y, bm.g_post = w16(y.t_gpm), bm.rstd_post = y.rstd_pm;
    bm.dx_in = dx_, bm.dbranch = dy_;
    bm.dg_next_partial = part_a_, bm.dg_post_partial = part_b_;
    bm.T = t, bm.D = D, bm.eps = eps;
    sandwich_tail_bwd(bm, r, s);
    reduce_gain(part_a_, last ? t_final_ : layers_[l + 1].t_gin, D, s);
    reduce_gain(part_b_, y.t_gpm, D, s);
    // down projection, SwiGLU, gate/up projection
    g.run(id_dgrad_[3], dy_, w16(y.t_down), nullptr, dh_, s);
    g.run(id_wgrad_[3], dy_, y.h, g32(y.t_down), g32(y.t_down), s);
    swiglu_bwd(dh_, y.gu, dgu_, t, F, r, s);
    g.run(id_dgrad_[2], dgu_, w16(y.t_gu), nullptr, dm_, s);
    g.run(id_wgrad_[2], dgu_, y.m, g32(y.t_gu), g32(y.t_gu), s);
    // attention tail: through the pre-MLP norm and the post-attention norm
    SandwichTailBwd ba{};
    ba.dx_out = dx_, ba.da_next = dm_;
    ba.x_out = y.x_mid, ba.g_next = w16(y.t_gpre), ba.rstd_next = y.rstd_pre;
    ba.branch = y.oproj, ba.g_post = w16(y.t_gpa), ba.rstd_post = y.rstd_pa;
    ba.dx_in = dx_, ba.dbranch = doproj_;
    ba.dg_next_partial = part_a_, ba.dg_post_partial = part_b_;
    ba.T = t, ba.D = D, ba.eps = eps;
    sandwich_tail_bwd(ba, r, s);
    reduce_gain(part_a_, y.t_gpre, D, s);
    reduce_gain(part_b_, y.t_gpa, D, s);
    // o projection
    g.run(id_dgrad_[1], doproj_, w16(y.t_o), nullptr, dattn_, s);
    g.run(id_wgrad_[1], doproj_, y.attn_o, g32(y.t_o), g32(y.t_o), s);
    // attention: dq, dk into their own buffers, dv straight into the v columns of dqkv
    AttnBwdArgs ab{};
    ab.fwd.q = y.q_rot, ab.fwd.q_ld = int64_t(H) * hd;
    ab.fwd.k = y.k_rot, ab.fwd.k_ld = int64_t(KV) * hd;
    ab.fwd.v = y.qkv + int64_t(H + KV) * hd, ab.fwd.v_ld = Q;
    ab.fwd.o = y.attn_o, ab.fwd.o_ld = int64_t(H) * hd;
    ab.fwd.lse = y.lse;
    ab.fwd.cu_seqlens = cu_seqlens_, ab.fwd.n_segs = n_segs_, ab.fwd.row_seg_start = row_seg_;
    ab.fwd.softmax_scale = sm_scale;
    ab.d_o = dattn_, ab.do_ld = int64_t(H) * hd;
    ab.softmax_d = nullptr, ab.dq_accum = nullptr;
    ab.dq = dq_, ab.dq_ld = int64_t(H) * hd;
    ab.dk = dk_, ab.dk_ld = int64_t(KV) * hd;
    ab.dv = dqkv_ + int64_t(H + KV) * hd, ab.dv_ld = Q;
    attn_->bwd(attn_shape_, ab, ws_attn_, s);
    // QK-norm + RoPE backward into the q, k columns of dqkv
    QkNormRopeBwd qb{dq_, dk_, y.qkv, Q, w16(y.t_gq), w16(y.t_gk), rope_cos_, rope_sin_, pos_, y.rstd_q, y.rstd_k,
                     dqkv_, Q, part_q_, part_k_, t, H, KV, hd, eps, q_scale};
    qknorm_rope_bwd(qb, r, s);
    reduce_gain(part_q_, y.t_gq, hd, s);
    reduce_gain(part_k_, y.t_gk, hd, s);
    // qkv projection: da (the input-norm output gradient) and the weight gradient
    g.run(id_dgrad_[0], dqkv_, w16(y.t_qkv), nullptr, dA_, s);
    g.run(id_wgrad_[0], dqkv_, y.a, g32(y.t_qkv), g32(y.t_qkv), s);
  }
  // layer 0 input norm (layers above fold theirs into the MLP tail of the layer below), then the embedding
  {
    RmsNormBwdAdd b{dA_, layers_[0].x_in, w16(layers_[0].t_gin), layers_[0].rstd_in, dx_, dx_, part_a_, t, D, eps};
    rmsnorm_bwd_add(b, r, s);
    reduce_gain(part_a_, layers_[0].t_gin, D, s);
  }
  const EmbedBwdWorkspace ew_ws = ew::embed_bwd_carve(ws_embed_, ws_embed_bytes_, t, V);
  embed_bwd_sorted(dx_, tokens, g32(t_embed_), t, D, V, s_.num.embed_accum, ew_ws, nullptr, s);
}

}  // namespace nm::model
