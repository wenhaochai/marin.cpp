"""S3 bandwidth report: achieved GB/s of each elementwise kernel at the per-GPU microbatch shapes of 130m (T = 131,072,
D = 512) and 1_2b (T = 32,768, D = 2048), bf16 storage, levanter rounding.

GB/s = the bytes a kernel must move (each input read once, each output written once; listed per kernel) / the median
time of 20 calls (CUDA events, after 3 warm-up calls). Reference points: the device's peak DRAM bandwidth (from its
memory clock and bus width) and a device-to-device copy measured in the same process (what the GPU delivers right now;
the vis node's GPU is shared). No pass/fail threshold: the H100 profile (DESIGN.md §12.3) owns the 0.8-of-HBM budget.
"""
import ctypes
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # the vis GPU is shared and fragmented

import numpy as np
import torch

from nmk import BF16S, EPS, LEV, init, ptr

lib, R = init("s3_bench")


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e-3)
    return float(np.median(ts)), float(np.min(ts))


def peak_gbs():
    dev = torch.cuda.current_device()
    cudart = ctypes.CDLL("libcudart.so.12")
    clk, bus = ctypes.c_int(), ctypes.c_int()
    cudart.cudaDeviceGetAttribute(ctypes.byref(clk), 36, dev)   # cudaDevAttrMemoryClockRate (kHz)
    cudart.cudaDeviceGetAttribute(ctypes.byref(bus), 37, dev)   # cudaDevAttrGlobalMemoryBusWidth (bits)
    return 2.0 * clk.value * 1e3 * bus.value / 8 / 1e9


ROWS = []


def report(name, nbytes, fn, ref_gbs, peak):
    med, best = timeit(fn)
    g = nbytes / med / 1e9
    ROWS.append((name, nbytes, med, g, best))
    print(f"    {name:<34s} {nbytes / 1e6:9.1f} MB  {med * 1e6:8.1f} us  {g:7.1f} GB/s  "
          f"{100 * g / peak:5.1f}% of peak  {100 * g / ref_gbs:5.1f}% of copy  (best {nbytes / best / 1e9:7.1f} GB/s)",
          flush=True)


def sec_norm(T, D, ref, peak, g):
    x, y, dy, dres, dx = g(T, D), g(T, D), g(T, D), g(T, D), g(T, D)
    gain = g(D)
    rstd = torch.empty(T, device="cuda")
    part = torch.empty(lib.nm_norm_partial_rows(T), D, device="cuda")
    report("rmsnorm_fwd", 4 * T * D + 4 * T,
           lambda: lib.nm_rmsnorm_fwd(BF16S, LEV, ptr(x), ptr(gain), ptr(y), ptr(rstd), T, D, EPS, None), ref, peak)
    report("rmsnorm_bwd_add", 8 * T * D + 4 * T,
           lambda: lib.nm_rmsnorm_bwd_add_partial(BF16S, LEV, ptr(dy), ptr(x), ptr(gain), ptr(rstd), ptr(dres), ptr(dx),
                                                  ptr(part), T, D, EPS, None), ref, peak)
    xo, an, dxi, dbr = g(T, D), g(T, D), g(T, D), g(T, D)
    rp, rn = torch.empty(T, device="cuda"), torch.empty(T, device="cuda")
    report("sandwich_tail_fwd (post-norm)", 8 * T * D + 8 * T,
           lambda: lib.nm_sandwich_tail_fwd(BF16S, LEV, ptr(x), ptr(dres), ptr(gain), ptr(gain), ptr(xo), ptr(an),
                                            ptr(rp), ptr(rn), T, D, EPS, None), ref, peak)
    dgn, dgp = torch.empty(D, device="cuda"), torch.empty(D, device="cuda")
    report("sandwich_tail_bwd (post-norm) +red", 12 * T * D + 8 * T,
           lambda: lib.nm_sandwich_tail_bwd(BF16S, LEV, ptr(dy), ptr(dres), ptr(xo), ptr(gain), ptr(rn), ptr(x),
                                            ptr(gain), ptr(rp), ptr(dxi), ptr(dbr), ptr(dgn), ptr(dgp), T, D, EPS,
                                            None), ref, peak)


def sec_qk(T, H, KV, hd, ref, peak, g):
    bf = torch.bfloat16
    W = (H + 2 * KV) * hd
    qkv = g(T, W)
    c = np.empty((4096, hd), np.uint16)
    s = np.empty((4096, hd), np.uint16)
    lib.nm_rope_tables(4096, hd, 10000.0, c.ctypes.data_as(ctypes.c_void_p), s.ctypes.data_as(ctypes.c_void_p), None)
    ct = torch.from_numpy(c.view(np.int16)).cuda().view(bf)
    st = torch.from_numpy(s.view(np.int16)).cuda().view(bf)
    pos = (torch.arange(T, device="cuda", dtype=torch.int32) % 4096)
    qr, kr = g(T, H * hd), g(T, KV * hd)
    rq, rk = torch.empty(T, H, device="cuda"), torch.empty(T, KV, device="cuda")
    gq = g(hd)
    nqk = T * (H + KV) * hd
    sc = float(torch.tensor(1.0 / math.sqrt(hd)).to(bf).float())   # bf16(1/sqrt(hd))
    report("qknorm_rope_fwd", 4 * nqk + 4 * T * (H + KV) + 4 * T,
           lambda: lib.nm_qknorm_rope_fwd(BF16S, LEV, ptr(qkv), W, ptr(gq), ptr(gq), ptr(ct), ptr(st), ptr(pos), ptr(qr),
                                          ptr(kr), ptr(rq), ptr(rk), T, H, KV, hd, EPS, sc, None), ref, peak)
    dqkv = g(T, W)
    dgq, dgk = torch.empty(hd, device="cuda"), torch.empty(hd, device="cuda")
    report("qknorm_rope_bwd +red", 6 * nqk + 4 * T * (H + KV) + 4 * T,
           lambda: lib.nm_qknorm_rope_bwd(BF16S, LEV, ptr(qr), ptr(kr), ptr(qkv), W, ptr(gq), ptr(gq), ptr(ct), ptr(st),
                                          ptr(pos), ptr(rq), ptr(rk), ptr(dqkv), W, ptr(dgq), ptr(dgk), T, H, KV, hd,
                                          EPS, sc, None), ref, peak)
    acc = torch.randn(H, T, hd, device="cuda")
    report("dq_post_qknorm_rope_bwd +red", 8 * T * H * hd + 4 * T * H + 4 * T,
           lambda: lib.nm_dq_post_qknorm_rope_bwd(LEV, ptr(acc), ptr(qkv), W, ptr(gq), ptr(ct), ptr(st), ptr(pos),
                                                  ptr(rq), ptr(dqkv), W, ptr(dgq), T, H, hd, EPS, sc, None), ref, peak)


def sec_swiglu(T, F, ref, peak, g):
    gu, h, dh = g(T, 2 * F), g(T, F), g(T, F)
    dgu = g(T, 2 * F)
    report("swiglu_fwd", 6 * T * F, lambda: lib.nm_swiglu_fwd(BF16S, LEV, ptr(gu), ptr(h), T, F, None), ref, peak)
    report("swiglu_bwd", 10 * T * F, lambda: lib.nm_swiglu_bwd(BF16S, LEV, ptr(dh), ptr(gu), ptr(dgu), T, F, None),
           ref, peak)


def sec_embed(T, D, V, ref, peak, g):
    rng = np.random.default_rng(0)   # fineweb-like token frequencies: Zipf 1.1
    t = rng.zipf(1.1, size=T) - 1
    t = np.where(t < V, t, rng.integers(0, V, size=T))
    tok = torch.from_numpy(rng.permutation(V)[t].astype(np.int32)).cuda()
    distinct = int(torch.unique(tok).numel())
    table = g(V, D)
    out = g(T, D)
    report("embed_fwd", 4 * T * D + 4 * T, lambda: lib.nm_embed_fwd(ptr(table), ptr(tok), ptr(out), T, D, None), ref,
           peak)
    del table
    torch.cuda.empty_cache()
    dtable = torch.zeros(V, D, device="cuda")
    nb = lib.nm_embed_bwd_workspace_bytes(T, V)
    ws = torch.empty(nb, dtype=torch.uint8, device="cuda")
    report(f"embed_bwd f32 ({distinct} rows)", 2 * T * D + 8 * distinct * D + 24 * T,
           lambda: lib.nm_embed_bwd(ptr(out), ptr(tok), ptr(dtable), T, D, V, 0, None, ptr(ws), nb, None), ref, peak)
    report(f"embed_bwd bf16 sequential ({distinct} rows)", 2 * T * D + 8 * distinct * D + 24 * T,
           lambda: lib.nm_embed_bwd(ptr(out), ptr(tok), ptr(dtable), T, D, V, 1, None, ptr(ws), nb, None), ref, peak)


def sec_doc_meta(T, ref, peak, g):
    rows, S = max(1, T // 4096), 4096
    T = rows * S
    toks = torch.randint(0, 128000, (rows, S), dtype=torch.int32, device="cuda")
    toks[toks % 200 == 0] = 128001
    tg = torch.empty(T, dtype=torch.int32, device="cuda")
    wt = torch.empty(T, device="cuda")
    ps = torch.empty(T, dtype=torch.int32, device="cuda")
    cu = torch.empty(rows * 64 + 1, dtype=torch.int32, device="cuda")
    ns = torch.empty(1, dtype=torch.int32, device="cuda")
    rss = torch.empty(rows, 65, dtype=torch.int32, device="cuda")
    ov = torch.zeros(1, dtype=torch.int32, device="cuda")
    report("doc_meta", 16 * T, lambda: lib.nm_doc_meta(ptr(toks), None, rows, S, 128001, 64, ptr(tg), ptr(wt), ptr(ps),
                                                       ptr(cu), ptr(ns), ptr(rss), ptr(ov), None), ref, peak)


def with_retry(label, fn, T):
    """Run a section at T tokens; on a CUDA out-of-memory error (other jobs share the vis GPU) retry at T/2, T/4."""
    for k in range(3):
        try:
            fn(T >> k)
            if k:
                print(f"    ({label} ran at T = {T >> k}: GPU memory held by other processes)")
            return
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    print(f"    {label}: skipped, out of GPU memory at T = {T >> 2}")


def bench_shape(name, T, D, H, KV, hd, F, V=128256):
    print(f"  {name}: T={T} D={D} H={H} KV={KV} hd={hd} F={F}")
    g = lambda *s: torch.randn(*s, device="cuda").to(torch.bfloat16)
    ref, nbytes = None, T * D * 4   # device copy reference at this size (smaller if other jobs hold the memory)
    while ref is None:
        try:
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            cmed, _ = timeit(lambda: dst.copy_(src))
            ref = 2 * nbytes / cmed / 1e9
            del src, dst
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            nbytes //= 4
            if nbytes < (1 << 20):
                raise
    peak = peak_gbs()
    print(f"    device copy {nbytes / 1e6:.0f} MB: {ref:.1f} GB/s; peak DRAM {peak:.1f} GB/s")
    torch.cuda.empty_cache()
    for label, fn in (("norms", lambda t: sec_norm(t, D, ref, peak, g)),
                      ("qk", lambda t: sec_qk(t, H, KV, hd, ref, peak, g)),
                      ("swiglu", lambda t: sec_swiglu(t, F, ref, peak, g)),
                      ("embed", lambda t: sec_embed(t, D, V, ref, peak, g)),
                      ("doc_meta", lambda t: sec_doc_meta(t, ref, peak, g))):
        with_retry(label, fn, T)
        torch.cuda.empty_cache()


@R.case
def bandwidth():
    print(f"  device {torch.cuda.get_device_name()}  (nm_device_arch {lib.nm_device_arch()})")
    bench_shape("130m / pls width 512", 131072, 512, 8, 8, 64, 1792)
    bench_shape("1_2b", 32768, 2048, 16, 8, 128, 7168)


R.run(__import__("nmk").ARGS.cases)
