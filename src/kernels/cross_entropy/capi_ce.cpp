// C ABI of the fused CE (capi.h block and capi_ce.h).
#include <cuda_runtime.h>

#include <string>

#include "core/capi_util.hpp"
#include "kernels/cross_entropy/capi_ce.h"
#include "kernels/cross_entropy/ce_internal.hpp"
#include "marincpp/capi.h"

namespace {

cudaStream_t S(void* p) { return static_cast<cudaStream_t>(p); }

nm::CeImpl impl_of(int i) {
  if (i < 0 || i > 3) nm::fatal(NM_WHERE, "impl must be 0 (A), 1 (B1), 2 (B1.5) or 3 (B2), got " + std::to_string(i));
  return static_cast<nm::CeImpl>(i);
}
nm::Rounding rounding_of(int r) {
  if (r != 0 && r != 1) nm::fatal(NM_WHERE, "rounding must be 0 (levanter) or 1 (accurate), got " + std::to_string(r));
  return r == 0 ? nm::Rounding::Levanter : nm::Rounding::Accurate;
}
nm::DType dtype_of(int d, const char* what) {
  if (d != 0 && d != 1) nm::fatal(NM_WHERE, std::string(what) + " must be 0 (f32) or 1 (bf16), got " + std::to_string(d));
  return d == 0 ? nm::DType::f32 : nm::DType::bf16;
}
nm::CeShape shape(int impl, int io, int logits_store, int64_t T, int D, int V, int chunk_rows) {
  nm::CeShape s{T, D, V, chunk_rows, dtype_of(logits_store, "logits_store"), dtype_of(io, "io")};
  (void)impl;
  return s;
}

}  // namespace

extern "C" {

uint64_t nm_ce_workspace_bytes(int impl, int64_t T, int D, int V, int chunk_rows, int logits_store) {
  uint64_t r = 0;
  const int st = nm::capi_call([&] { r = nm::ce_workspace_bytes(impl_of(impl), shape(impl, 1, logits_store, T, D, V,
                                                                                     chunk_rows)); });
  return st == NM_OK ? r : 0;
}

int nm_fused_linear_ce(int impl, int rounding, int io, int logits_store, const void* x, const void* w,
                       const int32_t* targets, const float* weight, const float* grad_scale, const float* logit_bias,
                       float* loss_tok, float* lse, float* loss_partial, void* dx, int accumulate_dx, float* dw,
                       float* dw_microbatch, int dw_round, int forward_only, int64_t T, int D, int V, int chunk_rows,
                       void* workspace, void* stream) {
  return nm::capi_call([&] {
    if (dw_round != 0 && dw_round != 1) nm::fatal(NM_WHERE, "dw_round must be 0 or 1");
    nm::CeArgs a{x, w, targets, weight, grad_scale, logit_bias, loss_tok, lse, loss_partial, dx, accumulate_dx != 0,
                 dw, dw_microbatch, dw_round ? nm::CeDwRound::PerChunkBf16 : nm::CeDwRound::None, forward_only != 0};
    nm::ce::run(impl_of(impl), rounding_of(rounding), shape(impl, io, logits_store, T, D, V, chunk_rows), a, workspace,
                nullptr, S(stream));
  });
}

int nm_ce_set_backends(int dgrad_own, int wgrad_own, int sm90) {
  auto& b = nm::ce::backends();
  const int prev = int(b.dgrad == nm::ce::GemmBackend::Own) | int(b.wgrad == nm::ce::GemmBackend::Own) << 1 |
                   b.sm90 << 2;
  b.dgrad = dgrad_own ? nm::ce::GemmBackend::Own : nm::ce::GemmBackend::Lt;
  b.wgrad = wgrad_own ? nm::ce::GemmBackend::Own : nm::ce::GemmBackend::Lt;
  if (sm90 >= 0) b.sm90 = sm90 & (nm::ce::kSm90Logits | nm::ce::kSm90Gemm);
  return prev;
}

int nm_ce_set_fast_stats_exp(int on) {
  const int prev = nm::ce::backends().fast_stats_exp;
  nm::ce::backends().fast_stats_exp = on != 0;
  return prev;
}

int nm_ce_finish_microbatch(float* dw, float* dw_microbatch, int64_t n, void* stream) {
  return nm::capi_call([&] { nm::ce::launch_finish_microbatch(dw, dw_microbatch, n, S(stream)); });
}

int nm_ce_grad_scale(const float* w, int64_t n, float mult, float* out, void* stream) {
  return nm::capi_call([&] { nm::ce::grad_scale(w, n, mult, out, S(stream)); });
}

int nm_ce_readout(int impl, int rounding, int logits_store, const void* state, const void* norm_gain,
                  float* norm_rstd_ws, void* norm_out_ws, const void* w, float* dw, float* dgain_partial,
                  const float* grad_scale, void* d_state, int visible_dims, const int32_t* targets,
                  const float* weight, float* loss_tok, float* lse, float* loss_partial, int forward_only, int64_t T,
                  int D, int V, int chunk_rows, float eps, void* workspace, void* stream) {
  return nm::capi_call([&] {
    nm::ReadoutCall c{0, static_cast<const nm::bf16_t*>(state), static_cast<const nm::bf16_t*>(norm_gain), norm_rstd_ws,
                      static_cast<nm::bf16_t*>(norm_out_ws), static_cast<const nm::bf16_t*>(w), dw, dgain_partial,
                      grad_scale, static_cast<nm::bf16_t*>(d_state), visible_dims};
    nm::CeArgs base{};
    base.targets = targets;
    base.weight = weight;
    base.loss_tok = loss_tok;
    base.lse = lse;
    base.loss_partial = loss_partial;
    base.dw_round = nm::CeDwRound::None;
    base.forward_only = forward_only != 0;
    nm::ce::readout(impl_of(impl), rounding_of(rounding), shape(impl, 1, logits_store, T, D, V, chunk_rows), c, base,
                    eps, workspace, nullptr, S(stream));
  });
}

int nm_ce_dlogits(int logits_store, void* store, int64_t ldS, const float* lse, const float* gscale,
                  const int32_t* targets, int rows, int V, void* stream) {
  return nm::capi_call([&] {
    nm::ce::DzArgs a{store, ldS, lse, gscale, targets, rows, V};
    nm::ce::launch_dlogits_inplace(dtype_of(logits_store, "logits_store"), a, S(stream));
  });
}

int nm_ce_device_arch(void) {
  int a = 0;
  return nm::capi_call([&] { a = nm::ce::device_arch(); }) == NM_OK ? a : 0;
}

}  // extern "C"
