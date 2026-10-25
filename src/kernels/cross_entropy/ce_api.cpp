// Host side of the fused linear CE (DESIGN.md §3.8): workspace layout, the per-chunk schedule of implementations A, B1,
// B1.5 and B2, the readout heads, the public entry points of kernels/cross_entropy.hpp, and the kernel dispatch
// (sm_90a kernels on H100, the portable mma.sync kernels elsewhere).
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>

#include "kernels/cross_entropy/ce_internal.hpp"
#include "kernels/elementwise/ew_api.hpp"
#include "marincpp/kernels/gemm.hpp"

namespace nm {
namespace ce {
namespace {

int dev_attr(cudaDeviceAttr at) {
  static std::mutex mu;
  static std::map<std::pair<int, int>, int> cache;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_pair(dev, int(at));
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int v = 0;
  NM_CUDA(cudaDeviceGetAttribute(&v, at, dev));
  cache[key] = v;
  return v;
}

const char* impl_name(CeImpl i) {
  switch (i) {
    case CeImpl::A: return "A";
    case CeImpl::B1: return "B1";
    case CeImpl::B1_5: return "B1.5";
    case CeImpl::B2: return "B2";
  }
  return "?";
}

template <class T>
T* offset(T* p, int64_t elems, DType dt) {
  using B = std::conditional_t<std::is_const_v<T>, const uint8_t, uint8_t>;
  return p ? reinterpret_cast<T*>(reinterpret_cast<B*>(p) + elems * int64_t(dtype_size(dt))) : nullptr;
}

void check_shape(CeImpl impl, const CeShape& s) {
  NM_REQUIRE(s.T >= 0 && s.D > 0 && s.V > 0, "fused CE: T >= 0, D > 0, V > 0");
  NM_REQUIRE(s.D % 8 == 0, "fused CE: D must be a multiple of 8");
  NM_REQUIRE(s.chunk_rows > 0, "fused CE: chunk_rows must be positive");
  NM_REQUIRE(s.logits_store == DType::bf16 || s.logits_store == DType::f32, "fused CE: logits store bf16 or f32");
  NM_REQUIRE(s.io == DType::bf16 || (s.io == DType::f32 && impl == CeImpl::A),
             std::string("fused CE: io f32 (the f32 policy) runs only implementation A, not ") + impl_name(impl));
}

}  // namespace

int device_arch() {
  return dev_attr(cudaDevAttrComputeCapabilityMajor) * 10 + dev_attr(cudaDevAttrComputeCapabilityMinor);
}
int sm_count() { return dev_attr(cudaDevAttrMultiProcessorCount); }

Backends& backends() {
  static Backends b;
  return b;
}

Ws carve(CeImpl impl, const CeShape& s, void* base) {
  Ws w;
  const int64_t rows = s.chunk_rows;
  uint64_t off = 0;
  auto take = [&](uint64_t bytes) -> void* {
    off = uint64_t(round_up(int64_t(off), 256));
    void* p = base ? static_cast<uint8_t*>(base) + off : nullptr;
    off += bytes;
    return p;
  };
  w.ldS = store_ld(s.V);
  if (impl == CeImpl::A) {
    w.store = take(uint64_t(rows) * w.ldS * 4);
  } else {
    w.store = take(uint64_t(rows) * w.ldS * dtype_size(s.logits_store));
    const int NTp = stats_tiles(s.V);          // allocated tiles (the planner's V / 128 + 1)
    w.NT = int((s.V + kTile - 1) / kTile);     // tiles the logits kernel writes and the finalize merges
    w.ldT = rows;
    w.tmax = static_cast<float*>(take(uint64_t(NTp) * rows * 4));
    w.tsum = static_cast<float*>(take(uint64_t(NTp) * rows * 4));
  }
  w.lse = static_cast<float*>(take(uint64_t(rows) * 4));
  w.gscale = static_cast<float*>(take(uint64_t(rows) * 4));
  // dxtmp is bf16: the f32 policy (A, io f32) calls cuBLASLt with beta directly and needs none, so the bf16-io layout
  // (the one the C ABI sizes) also bounds the f32-io one
  const uint64_t dxb = uint64_t(rows) * s.D * 2;
  if (s.io == DType::f32) {
    w.dxtmp = nullptr;
  } else if (impl != CeImpl::A && uint64_t(stats_tiles(s.V)) * rows * 4 * 2 >= dxb) {
    w.dxtmp = w.tmax;   // the span tmax .. end of tsum (free after the finalize)
  } else {
    w.dxtmp = take(dxb);
  }
  w.ltws = take(kLtWsBytes);
  w.bytes = off;
  return w;
}

// ------------------------------------------------------------------------------------------------ dispatch
void launch_logits_stats(DType store_dt, const LogitsArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  if ((backends().sm90 & kSm90Logits) && device_arch() == 90 && launch_logits_stats_sm90(store_dt, a, st)) return;
  launch_logits_stats_portable(store_dt, a, st);
}
void launch_dgrad(DType store_dt, const DgradArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  if ((backends().sm90 & kSm90Gemm) && device_arch() == 90 && launch_dgrad_sm90(store_dt, a, st)) return;
  launch_dgrad_portable(store_dt, a, st);
}
void launch_wgrad(DType store_dt, const WgradArgs& a, cudaStream_t st) {
  if (a.rows <= 0) return;
  if ((backends().sm90 & kSm90Gemm) && device_arch() == 90 && launch_wgrad_sm90(store_dt, a, st)) return;
  launch_wgrad_portable(store_dt, a, st);
}

void grad_scale(const float* w, int64_t n, float mult, float* out, cudaStream_t st) {
  launch_sum_scale(w, n, mult, out, st);
}

// ------------------------------------------------------------------------------------------------ one call
void run(CeImpl impl, Rounding r, const CeShape& s, const CeArgs& a, void* ws, cublasLtHandle_t lt, cudaStream_t st) {
  check_shape(impl, s);
  if (s.T == 0) return;
  NM_REQUIRE(a.x && a.w && a.targets && a.weight && a.loss_tok && ws, "fused CE: x, w, targets, weight, loss_tok and "
                                                                       "the workspace are required");
  const bool grads = !a.forward_only && (a.dx || a.dw);
  const bool strict_dw = a.dw && a.dw_round == CeDwRound::PerChunkBf16;
  if (grads) NM_REQUIRE(a.grad_scale, "fused CE: gradients need grad_scale");
  if (strict_dw) NM_REQUIRE(a.dw_microbatch && s.io == DType::bf16,
                            "fused CE: CeDwRound::PerChunkBf16 needs dw_microbatch (and bf16 io)");
  NM_REQUIRE(!(a.dx && a.dx == a.x && a.accumulate_dx), "fused CE: dx == x cannot accumulate");
  if (impl != CeImpl::A) NM_REQUIRE(!a.logit_bias, "fused CE: logit_bias is implemented only by implementation A");
  if (!lt && (impl == CeImpl::A || backends().dgrad == GemmBackend::Lt || backends().wgrad == GemmBackend::Lt))
    lt = lt_handle();
  const Ws w = carve(impl, s, ws);
  const DType io = s.io;
  const DType sd = s.logits_store;
  float* dw_target = strict_dw ? a.dw_microbatch : a.dw;
  const int D = s.D, V = s.V;
  const int64_t n_chunks = (s.T + s.chunk_rows - 1) / s.chunk_rows;

  for (int64_t c = 0; c < n_chunks; ++c) {
    const int64_t r0 = c * s.chunk_rows;
    const int n = int(std::min<int64_t>(s.chunk_rows, s.T - r0));
    const void* x_c = offset(a.x, r0 * D, io);
    const int32_t* tg_c = a.targets + r0;
    const float* wt_c = a.weight + r0;
    float* loss_c = a.loss_tok + r0;
    float* lse_c = a.lse ? a.lse + r0 : w.lse;
    void* dx_c = grads ? offset(a.dx, r0 * D, io) : nullptr;
    const bool in_place = dx_c && dx_c == x_c;
    const bool acc_dx = a.accumulate_dx;

    // plain-dz GEMMs (B1, B1.5's wgrad, A)
    auto wgrad_plain = [&](const void* dz, int64_t ldz) {
      if (!dw_target) return;
      if (strict_dw || backends().wgrad == GemmBackend::Own) {
        WgradArgs g{dz, ldz, x_c, nullptr, nullptr, nullptr, n, D, V, dw_target, strict_dw};
        launch_wgrad(DType::u8, g, st);
      } else {
        lt_wgrad(lt, DType::bf16, dz, ldz, x_c, dw_target, n, D, V, w.ltws, st);
      }
    };
    auto dgrad_plain = [&](const void* dz, int64_t ldz) {
      if (!dx_c) return;
      // the accurate accumulation bf16(old + acc) runs on the own kernel: Hopper cuBLASLt with beta = 1 and a bf16
      // C / D returned bf16(old + bf16(acc)), the levanter form, to the bit (t72, tests/s5 accumulate_dx)
      if (backends().dgrad == GemmBackend::Own || (acc_dx && r == Rounding::Accurate)) {
        DgradArgs g{dz, ldz, a.w, nullptr, nullptr, nullptr, n, D, V, acc_dx ? dx_c : nullptr, dx_c, false, r};
        launch_dgrad(DType::u8, g, st);
      } else if (acc_dx && r == Rounding::Levanter) {   // bf16(old + f32(bf16(acc))): GEMM to dxtmp, then add
        lt_dgrad(lt, DType::bf16, dz, ldz, a.w, w.dxtmp, 0.f, n, D, V, w.ltws, st);
        launch_dx_combine(w.dxtmp, dx_c, dx_c, int64_t(n) * D, DType::bf16, true, st);
      } else {
        lt_dgrad(lt, DType::bf16, dz, ldz, a.w, dx_c, 0.f, n, D, V, w.ltws, st);
      }
    };

    if (impl == CeImpl::A) {
      float* logits = static_cast<float*>(w.store);
      lt_logits(lt, io, x_c, a.w, logits, w.ldS, n, D, V, w.ltws, st);
      RowArgs ra{logits, w.ldS, a.logit_bias, tg_c, wt_c, a.grad_scale, n, V, lse_c, loss_c, w.gscale, grads,
                 sd == DType::bf16 && io == DType::bf16, io};
      launch_row_a(ra, st);
      if (grads) {
        if (io == DType::bf16) {
          const int64_t ldz = 2 * w.ldS;   // bf16 dz over the first half of each f32 row
          wgrad_plain(logits, ldz);        // before dgrad: dx may overwrite x (dx == x)
          dgrad_plain(logits, ldz);
        } else {
          NM_REQUIRE(!strict_dw, "fused CE: f32 io has no per-chunk bf16 dW rounding");
          if (dw_target) lt_wgrad(lt, DType::f32, logits, w.ldS, x_c, dw_target, n, D, V, w.ltws, st);
          if (dx_c) lt_dgrad(lt, DType::f32, logits, w.ldS, a.w, dx_c, acc_dx ? 1.f : 0.f, n, D, V, w.ltws, st);
        }
      }
    } else {
      LogitsArgs la{x_c, a.w, nullptr, tg_c, n, D, V, grads ? w.store : nullptr, w.ldS, w.tmax, w.tsum, w.ldT, loss_c,
                    backends().fast_stats_exp};
      launch_logits_stats(sd, la, st);
      FinalizeArgs fa{w.tmax, w.tsum, w.ldT, w.NT, loss_c, tg_c, wt_c, a.grad_scale, n, V, lse_c, loss_c, w.gscale};
      launch_finalize(fa, st);
      if (grads) {
        const int64_t ldz = sd == DType::bf16 ? w.ldS : 2 * w.ldS;   // plain dz written over the store
        if (impl == CeImpl::B1 || (impl == CeImpl::B1_5 && !dx_c)) {
          DzArgs za{w.store, w.ldS, lse_c, w.gscale, tg_c, n, V};
          launch_dlogits_inplace(sd, za, st);
          wgrad_plain(w.store, ldz);
          dgrad_plain(w.store, ldz);
        } else if (impl == CeImpl::B1_5) {
          void* out = in_place ? w.dxtmp : dx_c;
          DgradArgs g{w.store, w.ldS, a.w, lse_c, w.gscale, tg_c, n, D, V, acc_dx ? dx_c : nullptr, out, true, r};
          launch_dgrad(sd, g, st);
          wgrad_plain(w.store, ldz);
          if (in_place) launch_dx_combine(w.dxtmp, nullptr, dx_c, int64_t(n) * D, io, false, st);
        } else {   // B2: both GEMMs from the store; wgrad first (dx may overwrite x)
          if (dw_target) {
            WgradArgs g{w.store, w.ldS, x_c, lse_c, w.gscale, tg_c, n, D, V, dw_target, strict_dw};
            launch_wgrad(sd, g, st);
          }
          if (dx_c) {
            DgradArgs g{w.store, w.ldS, a.w, lse_c, w.gscale, tg_c, n, D, V, acc_dx ? dx_c : nullptr, dx_c, false, r};
            launch_dgrad(sd, g, st);
          }
        }
      }
    }
    if (a.loss_partial) launch_chunk_loss(loss_c, wt_c, n, a.loss_partial + c, st);
  }
}

// ------------------------------------------------------------------------------------------------ readout
void readout(CeImpl impl, Rounding r, const CeShape& s, const ReadoutCall& c, const CeArgs& base, float eps, void* ws,
             cublasLtHandle_t lt, cudaStream_t st) {
  check_shape(impl, s);
  NM_REQUIRE(s.io == DType::bf16, "readout: bf16 io only");
  NM_REQUIRE(c.state && c.norm_gain && c.norm_rstd_ws && c.norm_out_ws && c.w, "readout: state, norm_gain, "
                                                                                "norm_rstd_ws, norm_out_ws, w required");
  NM_REQUIRE(c.visible_dims > 0 && c.visible_dims <= s.D, "readout: visible_dims in (0, D]");
  const int64_t T = s.T;
  const int D = s.D;
  if (T == 0) return;
  ew::rmsnorm_fwd_dt(DType::bf16, RmsNormFwd{c.state, c.norm_gain, c.norm_out_ws, c.norm_rstd_ws, T, D, eps}, r, st);
  if (c.visible_dims < D) launch_zero_cols(c.norm_out_ws, T, D, c.visible_dims, DType::bf16, st);
  const bool need_dy = !base.forward_only && (c.d_state || c.dgain_partial);
  CeArgs a = base;
  a.x = c.norm_out_ws;
  a.w = c.w;
  a.dw = c.dw;
  a.grad_scale = c.grad_scale;
  a.dx = need_dy ? c.norm_out_ws : nullptr;   // dy over the normed input: every GEMM that reads x runs before
  a.accumulate_dx = false;
  run(impl, r, s, a, ws, lt, st);
  if (!need_dy) return;
  if (c.visible_dims < D) launch_zero_cols(c.norm_out_ws, T, D, c.visible_dims, DType::bf16, st);
  bf16_t* dxo = c.d_state;
  if (!dxo) {   // gain gradient only: the norm's dX goes to the (now free) logits store
    const Ws w = carve(impl, s, ws);
    const uint64_t store_bytes = uint64_t(s.chunk_rows) * w.ldS * (impl == CeImpl::A ? 4 : dtype_size(s.logits_store));
    NM_REQUIRE(uint64_t(T) * D * 2 <= store_bytes, "readout: the logits store is too small to hold the discarded norm dX");
    dxo = static_cast<bf16_t*>(w.store);
  }
  RmsNormBwdAdd nb{c.norm_out_ws, c.state, c.norm_gain, c.norm_rstd_ws, c.d_state, dxo, c.dgain_partial, T, D, eps};
  ew::rmsnorm_bwd_add_dt(DType::bf16, nb, r, st);
}

}  // namespace ce

// ------------------------------------------------------------------------------------------------ public API
uint64_t ce_workspace_bytes(CeImpl impl, const CeShape& s) { return ce::carve(impl, s, nullptr).bytes; }

void fused_linear_ce(CeImpl impl, Rounding r, const CeShape& s, const CeArgs& a, void* ws, GemmCache& gemm,
                     cudaStream_t st) {
  (void)gemm;   // the CE GEMMs are this module's kernels and its own cuBLASLt plans (BUILD_LOG S5)
  ce::run(impl, r, s, a, ws, nullptr, st);
}

void ce_finish_microbatch(const CeShape& s, const CeArgs& a, cudaStream_t st) {
  if (a.dw_round != CeDwRound::PerChunkBf16 || !a.dw) return;
  NM_REQUIRE(a.dw_microbatch, "ce_finish_microbatch: dw_microbatch required");
  ce::launch_finish_microbatch(a.dw, a.dw_microbatch, int64_t(s.V) * s.D, st);
}

void run_readout(CeImpl impl, Rounding r, const CeShape& s, const ReadoutCall& c, const CeArgs& base, void* ws,
                 GemmCache& gemm, cudaStream_t st) {
  (void)gemm;
  // ReadoutCall carries no eps yet (BUILD_LOG S5 request); every stage-1 model uses model.norm_eps = 1e-5
  ce::readout(impl, r, s, c, base, 1e-5f, ws, nullptr, st);
}

}  // namespace nm
