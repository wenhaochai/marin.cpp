"""S5 bench (report only): ms per fused-CE call (forward + dX + dW, bf16 store, f32 dW, levanter rounding) per
implementation and GEMM backend at the E2 shapes (docs/bench/RESULTS.md §3; BUILD_LOG "Decisions for S4-S6" items
11-12). One GPU; 5 warm-up and 20 timed calls, median of CUDA-event times. Also the forward-only time (logits + lse,
the eval path) and B1's split: forward, + dW, + dX.

    python tests/s5/bench_ce.py --lib build/libmarincpp_kernels.so --src . [quick|full] [--jsonl out.jsonl]

quick (default, ctest): T = 16,384, D = 512, V = 128,256 only. full: T = 16,384 and 131,072 x D = 512 / 768 / 1024 /
2048 at V = 128,256, and T = 131,072 at V = 64,256 and 8,256. focus: the H100 candidates of each stage at the E2 shapes
(B1 with cuBLASLt and with the sm_90a GEMMs, B1.5, B2; B1's split), chunk 16,384.
"""
import json
import os
import statistics
import sys

import nmce
from nmce import torch, ce_call, make_case, IMPLS, BF16S

# E2 bars (ms): Liger with f32 dW, the fastest measured candidate per shape (RESULTS.md §3)
BARS = {(16384, 512, 128256): 13.57, (16384, 768, 128256): 18.1, (16384, 1024, 128256): 22.6,
        (16384, 2048, 128256): 40.8, (131072, 512, 128256): 109.0, (131072, 768, 128256): 145.2,
        (131072, 1024, 128256): 179.3, (131072, 2048, 128256): 320.0, (131072, 512, 64256): 53.7,
        (131072, 768, 64256): 72.8, (131072, 1024, 64256): 89.1, (131072, 2048, 64256): 162.4,
        (131072, 512, 8256): 8.2, (131072, 768, 8256): 10.6, (131072, 1024, 8256): 12.7, (131072, 2048, 8256): 21.8}

mode = "quick"
jsonl = None
argv = []
it = iter(sys.argv[1:])
for a in it:
    if a == "--jsonl":
        jsonl = next(it)
    elif a in ("quick", "full", "focus"):
        mode = a
    else:
        argv.append(a)
sys.argv = [sys.argv[0]] + argv
lib, R = nmce.init("s5_bench")


def time_calls(fn, warm=5, reps=20):
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
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts)


def bench_shape(T, D, V, chunks, variants, out):
    x, W, tg, w = make_case(T, D, V, seed=1)
    gs = torch.tensor([1.0 / 524160], dtype=torch.float32, device="cuda")
    dx = torch.empty_like(x)
    dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
    loss = torch.empty(T, dtype=torch.float32, device="cuda")
    lse = torch.empty(T, dtype=torch.float32, device="cuda")
    bar = BARS.get((T, D, V))
    for chunk in chunks:
        if chunk > T and chunk != chunks[0]:
            continue
        ws = nmce.Ws.get(max(lib.nm_ce_workspace_bytes(IMPLS[i], T, D, V, chunk, BF16S) for i, *_ in variants))
        for impl, own_d, own_w, sm90, part in variants:
            lib.nm_ce_set_backends(own_d, own_w, sm90 & 3)
            lib.nm_ce_set_fast_stats_exp(int(sm90 >> 3 & 1))
            kw = dict(dx=None if part in ("fwd", "fwd+dw") else dx, dw=None if part in ("fwd", "fwd+dx") else dw,
                      forward_only=part == "fwd")

            def call():
                lib.nm_fused_linear_ce(IMPLS[impl], 0, 1, BF16S, nmce.ptr(x), nmce.ptr(W), nmce.ptr(tg), nmce.ptr(w),
                                       nmce.ptr(gs), None, nmce.ptr(loss), nmce.ptr(lse), None, nmce.ptr(kw["dx"]), 0,
                                       nmce.ptr(kw["dw"]), None, 0, int(kw["forward_only"]), T, D, V, chunk,
                                       nmce.ptr(ws), None)
            try:
                med, mn = time_calls(call)
            except RuntimeError as e:
                print(f"    {impl} {part}: {e}")
                continue
            flops = (2 if part == "fwd" else 4 if part in ("fwd+dw", "fwd+dx") else 6) * T * D * V
            tag = f"{impl}/{'own' if own_d else 'lt'}d/{'own' if own_w else 'lt'}w/sm90:{sm90 & 3}" \
                  f"{'/fastexp' if sm90 & 8 else ''}"
            ratio = f"  bar {bar:.2f} ms: x{med / bar:.3f}" if (bar and part == "all") else ""
            print(f"  T {T:6d} D {D:4d} V {V:6d} chunk {chunk:5d} {tag:28s} {part:7s} {med:9.3f} ms (min {mn:8.3f})"
                  f"  {flops / med / 1e9:6.1f} TFLOP/s{ratio}", flush=True)
            out.append(dict(T=T, D=D, V=V, chunk=chunk, impl=impl, dgrad="own" if own_d else "lt",
                            wgrad="own" if own_w else "lt", sm90=sm90, part=part, ms=med, ms_min=mn,
                            tflops=flops / med / 1e9, bar_ms=bar, arch=lib.nm_ce_device_arch()))
    lib.nm_ce_set_backends(0, 0, 3)   # the library default
    lib.nm_ce_set_fast_stats_exp(0)
    del x, W, dx, dw
    torch.cuda.empty_cache()


def bench_dlogits(T, V, out):
    """B1's in-place dlogits pass alone: bytes = 2 x rows x V x 2 (bf16 read + write)."""
    g = torch.Generator(device="cuda").manual_seed(2)
    ld = (V + 7) // 8 * 8
    store = (torch.randn(T, ld, generator=g, device="cuda") * 3).to(torch.bfloat16)
    lse = torch.full((T,), 12.0, device="cuda")
    gs = torch.full((T,), 1.0 / 524160, device="cuda")
    tg = torch.randint(0, V, (T,), generator=g, device="cuda", dtype=torch.int32)
    call = lambda: lib.nm_ce_dlogits(1, nmce.ptr(store), ld, nmce.ptr(lse), nmce.ptr(gs), nmce.ptr(tg), T, V, None)
    med, mn = time_calls(call)
    gbs = 4.0 * T * V / med / 1e6
    print(f"  dlogits alone T {T} V {V}: {med:.3f} ms (min {mn:.3f})  {gbs:.0f} GB/s", flush=True)
    out.append(dict(T=T, V=V, part="dlogits", ms=med, ms_min=mn, gbs=gbs, arch=lib.nm_ce_device_arch()))
    del store


@R.case
def bench():
    out = []
    print(f"  device {torch.cuda.get_device_name()} (CE dispatch arch {lib.nm_ce_device_arch()})")
    # (impl, own dgrad, own wgrad, sm_90a kernels: bit 0 logits, bit 1 GEMMs, + 8: the fast statistics exp, part)
    full_variants = [("B1", 0, 0, 1, "all"), ("B1", 1, 1, 1, "all"), ("B1.5", 0, 0, 1, "all"), ("B1.5", 1, 1, 1, "all"),
                     ("B2", 1, 1, 1, "all"), ("B1", 0, 0, 1, "fwd"), ("B1", 0, 0, 1, "fwd+dw"), ("B1", 0, 0, 1, "fwd+dx")]
    if lib.nm_ce_device_arch() == 90:
        full_variants += [("B1", 0, 0, 0, "all"), ("B1", 0, 0, 0, "fwd"), ("B1", 1, 1, 3, "all"), ("B1", 0, 1, 3, "all"),
                          ("B1", 1, 0, 3, "all"), ("B1.5", 1, 1, 3, "all"), ("B1.5", 0, 1, 3, "all"),
                          ("B2", 1, 1, 3, "all")]
    bench_dlogits(16384, 128256, out)
    if mode == "focus":
        fv = [("B1", 0, 0, 3, "all"), ("B1", 0, 0, 3, "fwd"), ("B1", 0, 0, 3, "fwd+dw"), ("B1", 0, 0, 3, "fwd+dx"),
              ("B1", 0, 0, 11, "all"), ("B1", 0, 0, 11, "fwd"),
              ("B1", 1, 1, 3, "all"), ("B1.5", 0, 0, 3, "all"), ("B1.5", 1, 1, 3, "all"), ("B2", 1, 1, 3, "all")]
        for D in (512, 768, 1024, 2048):
            bench_shape(16384, D, 128256, [16384], fv, out)
        for D in (512, 768, 1024, 2048):
            bench_shape(131072, D, 128256, [16384], fv, out)
        for V in (64256, 8256):
            for D in (512, 2048):
                bench_shape(131072, D, V, [16384], [v for v in fv if v[4] == "all"], out)
    elif mode == "quick":
        bench_shape(16384, 512, 128256, [16384], full_variants + [("A", 0, 0, 1, "all")], out)
    else:
        for D in (512, 768, 1024, 2048):
            bench_shape(16384, D, 128256, [8192, 16384], full_variants + [("A", 0, 0, 1, "all")], out)
        for D in (512, 768, 1024, 2048):
            bench_shape(131072, D, 128256, [16384, 32768], full_variants, out)
        for V in (64256, 8256):
            for D in (512, 768, 1024, 2048):
                bench_shape(131072, D, V, [16384, 32768], [v for v in full_variants if v[4] == "all"], out)
    if jsonl:
        with open(jsonl, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    R.run(nmce.ARGS.cases)
