"""Shared helpers of the S4 gates and benches: the S3 helper module (ctypes Lib, runner, error metrics) extended with
the S4 entry points of libmarincpp_kernels.so (capi.h attention and GEMM blocks, capi_attn.h, capi_gemm.h), segment
tables, the f64 attention oracle and an attention harness over every backend / FA3 mode.

Layouts (attention.hpp): q [T, H*hd], k / v [T, KV*hd], o like q, lse [H, T] (natural log); head h reads kv head
h // (H // KV). Segments: causal inside each segment; a segment starts at every row start and after every EOS.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "s3"))
import nmk  # noqa: E402  (S3 helpers: Lib, Runner, rel, row_report, ...)
from nmk import C, F32, I32, I64, U64, VP  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

nmk.SIGS.update({
    "nm_attention_workspace_bytes": (U64, [I32, I32, I64, I32, I32, I32, I32, I32, I32, I32]),
    "nm_attention_fwd": (I32, [I32, I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, VP, VP, I64, I32, I32, I32, I32,
                               I32, I32, F32, VP, VP]),
    "nm_attention_bwd": (I32, [I32, I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, I64, VP, VP, VP, I64, VP, I64,
                               VP, I64, VP, VP, VP, I64, I32, I32, I32, I32, I32, I32, F32, I32, VP, VP]),
    "nm_fa3_mode_workspace_bytes": (U64, [I32, I64, I32, I32, I32, I32, I32, I32, I32]),
    "nm_fa3_mode_fwd": (I32, [I32, I32, I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, VP, VP, I64, I32, I32, I32,
                              I32, I32, I32, F32, VP, VP]),
    "nm_fa3_mode_bwd": (I32, [I32, I32, I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, I64, VP, I64, VP, I64, VP,
                              I64, VP, VP, VP, I64, I32, I32, I32, I32, I32, I32, F32, I32, VP, VP]),
    "nm_ref_attention_fwd": (I32, [I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, VP, I64, I32, I32, I32, I32, I32,
                                   I32, F32, VP]),
    "nm_ref_attention_bwd": (I32, [I32, VP, I64, VP, I64, VP, I64, VP, I64, VP, VP, I64, VP, I64, VP, I64, VP, I64,
                                   VP, VP, I64, I32, I32, I32, I32, I32, I32, F32, VP, VP]),
    "nm_linear_fwd": (I32, [I32, VP, VP, VP, VP, I64, I64, I64, VP]),
    "nm_linear_dgrad": (I32, [I32, VP, VP, VP, VP, I64, I64, I64, VP]),
    "nm_linear_wgrad": (I32, [I32, VP, VP, VP, I64, I64, I64, VP]),
    "nm_gemm_f32": (I32, [I32, VP, VP, VP, I64, I64, I64, I32, VP]),
    "nm_gemm_search": (I32, [I32, I32, I64, I64, I64, C.c_char_p, I32, VP]),
    "nm_gemm_chosen": (I32, [I32, I32, I64, I64, I64, C.c_char_p, I32]),
    "nm_gemm_custom_count": (I32, []),
    "nm_gemm_custom_name": (C.c_char_p, [I32]),
    "nm_linear_custom": (I32, [I32, I32, I32, VP, VP, VP, I64, I64, I64, VP]),
    "nm_gemm_candidates": (I32, [I32, I32, I64, I64, I64, C.c_char_p, I32, VP]),
    "nm_linear_named": (I32, [C.c_char_p, I32, I32, VP, VP, VP, I64, I64, I64, VP]),
})

ptr = nmk.ptr
rel = nmk.rel
init = nmk.init
Skip = nmk.Skip

# FA3 launch modes (attn_internal.hpp Fa3Mode): persistent = what exec.attn_bwd_sched=persistent runs (the work list,
# one CTA per list slot), doc_causal = exec.attn_bwd_sched=doc_causal; E1b arms: sm_grid (the same list on a grid of SM
# count), stock_padded, stock_exact.
FA3_MODES = {"sm_grid": 0, "doc_causal": 1, "stock_padded": 2, "stock_exact": 3, "persistent": 4}
BACKENDS = {"fa3": 0, "cudnn": 1, "levref": 2, "ref_f32": 3}          # capi.h nm_attention_* backend ids
SCHEDS = {"persistent": 0, "doc_causal": 1}                            # capi.h bwd_sched
SCALE_BF16 = {64: 0.125, 128: 0.08837890625}          # bf16(1/sqrt(hd)): 0x3E00, 0x3DB5
ATTN_SHAPES = {                                       # (H, KV, hd) per size, DESIGN.md §3.4
    "130m": (8, 8, 64), "300m": (12, 12, 64), "520m": (16, 8, 64), "1_2b": (16, 8, 128),
}


# ------------------------------------------------------------------------------------------------ segments
class Segs:
    """Segment tables of a microbatch from per-row segment ids [rows, S] (equal id = same segment; ids change at
    segment starts). cu: packed starts [rows * max_segs + 1] (empty trailing slots repeat T), rss: per-row starts
    [rows, max_segs + 1] (then S), n_segs: device count."""

    def __init__(self, seg_ids, max_segs=None):
        seg_ids = np.asarray(seg_ids)
        self.rows, self.S = seg_ids.shape
        starts = []
        for r in range(self.rows):
            s = seg_ids[r]
            st = [0] + [int(i) for i in np.nonzero(s[1:] != s[:-1])[0] + 1]
            starts.append(st)
        self.starts = starts
        self.max_in_data = max(len(s) for s in starts)
        self.max_segs = max(max_segs or 0, self.max_in_data)
        T = self.rows * self.S
        self.T = T
        cu = [r * self.S + p for r, st in enumerate(starts) for p in st]
        self.n = len(cu)
        self.lengths = np.diff(np.array(cu + [T]))
        self.max_len = int(self.lengths.max())
        cu = cu + [T] * (self.rows * self.max_segs + 1 - len(cu))
        rss = np.full((self.rows, self.max_segs + 1), self.S, np.int32)
        for r, st in enumerate(starts):
            rss[r, :len(st)] = st
        self.cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
        self.rss = torch.tensor(rss, dtype=torch.int32, device="cuda")
        self.n_segs = torch.tensor([self.n], dtype=torch.int32, device="cuda")


def real_segments(src, S=4096, rows=32):
    """Real permuted fineweb-edu segment ids of bench/e1_attention/segments (gitignored; None if absent)."""
    p = os.path.join(src, "bench", "e1_attention", "segments", f"seg_llama128k_S{S}_B{rows}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["seg"]


def synthetic_segments(rows, S, gen_np, edge_cases=True):
    """Segment ids with length-1 segments, a segment ending exactly on a 128 tile edge, and random lengths."""
    seg = np.zeros((rows, S), np.int32)
    for r in range(rows):
        cuts = set()
        if edge_cases:
            cuts |= {1, 2, 128, 129, 256, S - 1}       # length-1 segments, tile-edge ends
        n = int(gen_np.integers(2, 8))
        cuts |= set(int(x) for x in gen_np.integers(1, S, n))
        cuts = sorted(c for c in cuts if 0 < c < S)
        sid = 0
        prev = 0
        for c in cuts + [S]:
            seg[r, prev:c] = sid
            sid += 1
            prev = c
    return seg


# ------------------------------------------------------------------------------------------------ oracle
def oracle_f64(q, k, v, dO, segs, H, KV, hd, scale, rows=None):
    """f64 causal same-segment attention per segment (DESIGN.md §10.2: chunked by segment). q, dO [T, H*hd], k, v
    [T, KV*hd] (any float dtype). Returns O, L [H, T], dq, dk, dv in float64 for the token rows of `rows` (row indices of
    the microbatch; None = all)."""
    T = q.shape[0]
    G = H // KV
    out = {n: torch.zeros(T, H * hd if n in ("O", "dq") else KV * hd, dtype=torch.float64, device="cuda")
           for n in ("O", "dq", "dk", "dv")}
    L = torch.zeros(H, T, dtype=torch.float64, device="cuda")
    rows = range(segs.rows) if rows is None else rows
    for r in rows:
        st = segs.starts[r] + [segs.S]
        for a, b in zip(st[:-1], st[1:]):
            a0, b0 = r * segs.S + a, r * segs.S + b
            n = b0 - a0
            qs = q[a0:b0].double().view(n, H, hd).transpose(0, 1)        # [H, n, hd]
            ks = k[a0:b0].double().view(n, KV, hd).transpose(0, 1).repeat_interleave(G, 0)
            vs = v[a0:b0].double().view(n, KV, hd).transpose(0, 1).repeat_interleave(G, 0)
            ds = dO[a0:b0].double().view(n, H, hd).transpose(0, 1)
            s = torch.matmul(qs, ks.transpose(1, 2)) * scale
            mask = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
            s = s.masked_fill(~mask, float("-inf"))
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            ls = p.sum(-1, keepdim=True)
            p = p / ls
            o = torch.matmul(p, vs)
            L[:, a0:b0] = (m + torch.log(ls))[..., 0]
            dp = torch.matmul(ds, vs.transpose(1, 2))
            Dr = (ds * o).sum(-1, keepdim=True)
            dS = p * (dp - Dr)
            dq = torch.matmul(dS, ks) * scale
            dk = (torch.matmul(dS.transpose(1, 2), qs) * scale).view(KV, G, n, hd).sum(1)
            dv = torch.matmul(p.transpose(1, 2), ds).view(KV, G, n, hd).sum(1)
            out["O"][a0:b0] = o.transpose(0, 1).reshape(n, H * hd)
            out["dq"][a0:b0] = dq.transpose(0, 1).reshape(n, H * hd)
            out["dk"][a0:b0] = dk.transpose(0, 1).reshape(n, KV * hd)
            out["dv"][a0:b0] = dv.transpose(0, 1).reshape(n, KV * hd)
    out["L"] = L
    return out


def row_report(name, a, b, width, tensor_rel, factor=10.0, show=8):
    """DESIGN.md §10.2 per-row report: per-row error = ||a_r - b_r|| / max(||b_r||, median row norm). Rows smaller than
    the median row are measured on the median row's scale: the gradients of short segments are small and cancelling
    (dS = P (dP - D) with dP ~ D; exactly 0 for a segment's first query and for length-1 segments), so dividing by
    their own norm measures f32 cancellation, not a bug. Passes when every row <= factor x tensor rel."""
    a2 = a.double().reshape(-1, width)
    b2 = b.double().reshape(-1, width)
    nb = torch.linalg.vector_norm(b2, dim=1)
    nd = torch.linalg.vector_norm(a2 - b2, dim=1)
    rr = nd / torch.clamp(nb, min=max(nb.median().item(), 1e-300))
    bound = factor * max(tensor_rel, 1e-7)
    worst = torch.topk(rr, min(show, rr.numel()))
    rows = ", ".join(f"{int(i)}:{v:.2e}" for v, i in zip(worst.values.tolist(), worst.indices.tolist()))
    print(f"    rows {name}: max row rel {worst.values[0].item():.3e} (bound {bound:.3e}) worst [{rows}]")
    return bool((rr <= bound).all()), worst.values[0].item()


def token_rows(segs, rows):
    """Token index tensor of the microbatch rows `rows`."""
    return torch.cat([torch.arange(r * segs.S, (r + 1) * segs.S, device="cuda") for r in rows])


# ------------------------------------------------------------------------------------------------ harness
class Attn:
    """One shape: inputs, outputs and calls of every backend through the C ABI."""

    def __init__(self, lib, segs, H, KV, hd, gen, dtype=torch.bfloat16, scale=None, inputs=None):
        self.lib, self.segs, self.H, self.KV, self.hd = lib, segs, H, KV, hd
        T = segs.T
        self.T = T
        self.scale = SCALE_BF16[hd] if scale is None else scale
        if inputs is None:
            q = torch.randn(T, H * hd, generator=gen, device="cuda")
            k = torch.randn(T, KV * hd, generator=gen, device="cuda")
            v = torch.randn(T, KV * hd, generator=gen, device="cuda")
            dO = torch.randn(T, H * hd, generator=gen, device="cuda")
            inputs = (q, k, v, dO)
        self.q, self.k, self.v, self.dO = (x.to(dtype).contiguous() for x in inputs)
        self.dtype = dtype

    def new_out(self):
        T, H, KV, hd = self.T, self.H, self.KV, self.hd
        e = lambda n: torch.empty(T, n, dtype=self.dtype, device="cuda")  # noqa: E731
        return dict(O=e(H * hd), L=torch.empty(H, T, dtype=torch.float32, device="cuda"), dq=e(H * hd),
                    dk=e(KV * hd), dv=e(KV * hd))

    def shape_args(self):
        s = self.segs
        return (self.T, s.rows, s.S, self.H, self.KV, self.hd, s.max_segs)

    # FA3 through capi_attn.h (any mode)
    def fa3_ws(self, mode, training=True):
        n = self.lib.nm_fa3_mode_workspace_bytes(FA3_MODES[mode], *self.shape_args(), int(training))
        if n == 2 ** 64 - 1:
            raise RuntimeError(self.lib.dll.nm_last_error().decode())
        return torch.empty(max(int(n), 256), dtype=torch.uint8, device="cuda")

    def fa3_fwd(self, mode, out, ws, stream=None):
        s = self.segs
        hd = self.hd
        self.lib.nm_fa3_mode_fwd(FA3_MODES[mode], s.n, s.max_len, ptr(self.q), self.H * hd, ptr(self.k), self.KV * hd,
                                 ptr(self.v), self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]), ptr(s.cu),
                                 ptr(s.n_segs), ptr(s.rss), *self.shape_args(), self.scale, ptr(ws), stream)

    def fa3_bwd(self, mode, out, ws, det, stream=None):
        s = self.segs
        hd = self.hd
        self.lib.nm_fa3_mode_bwd(FA3_MODES[mode], s.n, s.max_len, ptr(self.q), self.H * hd, ptr(self.k), self.KV * hd,
                                 ptr(self.v), self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]), ptr(self.dO),
                                 self.H * hd, ptr(out["dq"]), self.H * hd, ptr(out["dk"]), self.KV * hd, ptr(out["dv"]),
                                 self.KV * hd, ptr(s.cu), ptr(s.n_segs), ptr(s.rss), *self.shape_args(), self.scale,
                                 int(det), ptr(ws), stream)

    # production entry points (capi.h nm_attention_*: make_attention, fa3::fwd / bwd, fa3_mode_of)
    def api_ws(self, backend, sched, training=True):
        n = self.lib.nm_attention_workspace_bytes(BACKENDS[backend], SCHEDS[sched], *self.shape_args(), int(training))
        if n == 2 ** 64 - 1:
            raise RuntimeError(self.lib.dll.nm_last_error().decode())
        return torch.empty(max(int(n), 256), dtype=torch.uint8, device="cuda")

    def api_fwd(self, backend, sched, out, ws, stream=None):
        s = self.segs
        hd = self.hd
        self.lib.nm_attention_fwd(BACKENDS[backend], SCHEDS[sched], ptr(self.q), self.H * hd, ptr(self.k),
                                  self.KV * hd, ptr(self.v), self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]),
                                  ptr(s.cu), ptr(s.n_segs), ptr(s.rss), *self.shape_args(), self.scale, ptr(ws), stream)

    def api_bwd(self, backend, sched, out, ws, det, stream=None):
        s = self.segs
        hd = self.hd
        self.lib.nm_attention_bwd(BACKENDS[backend], SCHEDS[sched], ptr(self.q), self.H * hd, ptr(self.k),
                                  self.KV * hd, ptr(self.v), self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]),
                                  ptr(self.dO), self.H * hd, None, None, ptr(out["dq"]), self.H * hd, ptr(out["dk"]),
                                  self.KV * hd, ptr(out["dv"]), self.KV * hd, ptr(s.cu), ptr(s.n_segs), ptr(s.rss),
                                  *self.shape_args(), self.scale, int(det), ptr(ws), stream)

    # reference attention (capi_attn.h nm_ref_attention_*), cu_seqlens or row_seg_start
    def ref(self, out, use_cu=True, stream=None):
        s = self.segs
        hd = self.hd
        dt = 0 if self.dtype == torch.float32 else 1
        cu = ptr(s.cu) if use_cu else None
        rss = None if use_cu else ptr(s.rss)
        self.lib.nm_ref_attention_fwd(dt, ptr(self.q), self.H * hd, ptr(self.k), self.KV * hd, ptr(self.v),
                                      self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]), cu, rss,
                                      *self.shape_args(), self.scale, stream)
        ws = torch.empty(self.H * self.T, dtype=torch.float32, device="cuda")
        self.lib.nm_ref_attention_bwd(dt, ptr(self.q), self.H * hd, ptr(self.k), self.KV * hd, ptr(self.v),
                                      self.KV * hd, ptr(out["O"]), self.H * hd, ptr(out["L"]), ptr(self.dO),
                                      self.H * hd, ptr(out["dq"]), self.H * hd, ptr(out["dk"]), self.KV * hd,
                                      ptr(out["dv"]), self.KV * hd, cu, rss, *self.shape_args(), self.scale, ptr(ws),
                                      stream)

    def fwd_flops(self):
        """Document FLOPs of the forward (4 hd H sum_i n_i (n_i + 1) / 2), E1's convention (fwd+bwd = 3.5 x fwd)."""
        n = self.segs.lengths.astype(np.float64)
        return 4.0 * self.hd * self.H * float((n * (n + 1) / 2).sum())


def time_ms(fn, warm=5, reps=20):
    """BUILD_LOG decision 2: 5 warm-up and 20 timed calls, median, CUDA events."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
