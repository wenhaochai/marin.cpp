"""S4 GEMM gates (DESIGN.md §3.7, §11): the linear-layer GEMMs of the C ABI (GemmCache search on first use) against
torch f64, both wgrad roundings, bitwise repeatability, the f32 GEMMs.

    python tests/s4/test_gemm.py --lib build/libmarincpp_kernels.so --src . [case ...]

Bounds
  fwd, dgrad (bf16 out, f32 accumulate, one rounding): rel(k, f64) <= 1.05 x the single-rounding floor.
  wgrad accurate (f32 out, beta = 1 in place): rel(D - C, f64 product) <= 1e-4 (tensor-core f32 accumulation over
  T = 16,384 terms; measured up to 2.4e-5 on the A100, 1/40 of a bf16 ulp).
  wgrad levanter, C = 0: every element of D is a bf16 value; D = bf16 of the product within 1.05 x the floor; and
  D(C) == C + D(0) bitwise (D = C + f32(bf16(acc)): the f32 add is the only operation after the rounding).
  Repeatability: two calls give the same bits (deterministic split-K schemes only).
  every_candidate: the bounds above hold for EVERY deterministic candidate a search may pick (the timed choice varies
  between runs by up to 8%), each run through the cache's own code path by name (nm_linear_named), with a random
  nonzero C for both wgrads.
  nm_gemm_f32: pedantic rel <= 1e-6; TF32 rel <= 2e-3.
"""
import ctypes
import time

import torch

import nmk4
from nmk4 import ptr, rel

lib, R = nmk4.init("s4_gemm")
ARGS = nmk4.nmk.ARGS

# (name, K = in features, N = out features) of the owner's linear layers (DESIGN.md §2.5 sizes)
LAYERS = [
    ("130m.qkv", 512, 1536), ("130m.o_proj", 512, 512), ("130m.gate_up", 512, 3584), ("130m.down", 1792, 512),
    ("300m.qkv", 768, 2304), ("300m.down", 2688, 768),
    ("520m.qkv", 1024, 2048), ("520m.gate_up", 1024, 7168),
    ("1_2b.qkv", 2048, 4096), ("1_2b.down", 7168, 2048),
    ("head.512", 512, 128256),
]


def tokens_for(N):
    return 2048 if N > 100000 else 8192


@R.case
def linear_fwd_dgrad():
    gen = torch.Generator(device="cuda").manual_seed(1)
    for name, K, N in LAYERS:
        T = tokens_for(N)
        x = torch.randn(T, K, generator=gen, device="cuda").to(torch.bfloat16)
        w = (torch.randn(N, K, generator=gen, device="cuda") / K ** 0.5).to(torch.bfloat16)
        dy = torch.randn(T, N, generator=gen, device="cuda").to(torch.bfloat16)
        y = torch.empty(T, N, dtype=torch.bfloat16, device="cuda")
        dx = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
        lib.nm_linear_fwd(0, ptr(x), ptr(w), ptr(y), None, T, N, K, None)
        lib.nm_linear_dgrad(0, ptr(dy), ptr(w), ptr(dx), None, T, N, K, None)
        torch.cuda.synchronize()
        y64 = x.double() @ w.double().t()
        dx64 = dy.double() @ w.double()
        print(f"  {name}: T {T} K {K} N {N}")
        nmk4.nmk.accurate_check(R, f"{name}.fwd", y, y64)
        nmk4.nmk.accurate_check(R, f"{name}.dgrad", dx, dx64)
        y2 = torch.empty_like(y)
        lib.nm_linear_fwd(0, ptr(x), ptr(w), ptr(y2), None, T, N, K, None)
        torch.cuda.synchronize()
        R.check(torch.equal(y, y2), f"{name}.fwd: two calls differ")


@R.case
def linear_wgrad_accurate():
    gen = torch.Generator(device="cuda").manual_seed(2)
    for name, K, N in LAYERS:
        if N > 100000:
            continue   # the CE head's wgrad is S5's (fused CE)
        T = 16384
        x = torch.randn(T, K, generator=gen, device="cuda").to(torch.bfloat16)
        dy = torch.randn(T, N, generator=gen, device="cuda").to(torch.bfloat16)
        c = torch.randn(N, K, generator=gen, device="cuda")
        dw = c.clone()
        lib.nm_linear_wgrad(1, ptr(dy), ptr(x), ptr(dw), T, N, K, None)
        torch.cuda.synchronize()
        p64 = dy.double().t() @ x.double()
        r = rel(dw.double() - c.double(), p64)
        print(f"  {name}: T {T}: rel(D - C, f64) {r:.3e}")
        R.check(r <= 1e-4, f"{name}: wgrad accurate rel {r:.3e} > 1e-4")
        dw2 = c.clone()
        lib.nm_linear_wgrad(1, ptr(dy), ptr(x), ptr(dw2), T, N, K, None)
        torch.cuda.synchronize()
        R.check(torch.equal(dw, dw2), f"{name}: wgrad two calls differ")


@R.case
def linear_wgrad_levanter():
    gen = torch.Generator(device="cuda").manual_seed(3)
    for name, K, N in LAYERS:
        if N > 100000:
            continue
        T = 16384
        x = torch.randn(T, K, generator=gen, device="cuda").to(torch.bfloat16)
        dy = torch.randn(T, N, generator=gen, device="cuda").to(torch.bfloat16)
        d0 = torch.zeros(N, K, device="cuda")
        lib.nm_linear_wgrad(0, ptr(dy), ptr(x), ptr(d0), T, N, K, None)
        c = torch.randn(N, K, generator=gen, device="cuda")
        dc = c.clone()
        lib.nm_linear_wgrad(0, ptr(dy), ptr(x), ptr(dc), T, N, K, None)
        torch.cuda.synchronize()
        is_bf16 = torch.equal(d0, d0.to(torch.bfloat16).float())
        R.check(is_bf16, f"{name}: levanter wgrad with C = 0 is not bf16-valued")
        p64 = dy.double().t() @ x.double()
        print(f"  {name}: T {T}: D(0) bf16-valued {is_bf16}")
        nmk4.nmk.accurate_check(R, f"{name}.wgrad_lev", d0.to(torch.bfloat16), p64)
        same = torch.equal(dc, c + d0)
        print(f"    {name}: D(C) == C + D(0) bitwise: {same}")
        R.check(same, f"{name}: levanter wgrad D(C) != C + f32(bf16(acc))")


def floor_ratio(k, orc):
    return rel(k, orc) / rel(orc.float().to(torch.bfloat16), orc)


def named(kn, orient, rnd, a_, b_, out, T, N, K):
    r = lib.dll.nm_linear_named(kn.encode(), orient, rnd, ptr(a_), ptr(b_), ptr(out), T, N, K, None)
    if r != 0:
        raise RuntimeError(f"nm_linear_named {kn}: {lib.dll.nm_last_error().decode()}")
    return out


def candidate_check(tag, kn, orient, rnd, a_, b_, orc, c, T, N, K, attempts=8, wait_s=30):
    """Two calls of candidate kn, the bounds of its orientation; (all checks passed, error measure). A CUDA
    out-of-memory error (the vis node's GPU is shared) retries this candidate only."""
    slack = nmk4.nmk.ACC_SLACK
    for k in range(attempts):
        try:
            def call(init):
                out = init.clone() if init is not None else torch.empty(
                    orc.shape[0], orc.shape[1], dtype=torch.bfloat16, device="cuda")
                return named(kn, orient, rnd, a_, b_, out, T, N, K)
            d1 = call(c)
            d2 = call(c)
            d0 = call(torch.zeros_like(c)) if (orient == 2 and rnd == 0) else None
            torch.cuda.synchronize()
            break
        except Exception as e:  # noqa: BLE001 (re-raised unless it is an OOM)
            if not nmk4.nmk.is_oom(e) or k == attempts - 1:
                raise
            print(f"    ({tag} {kn}: CUDA out of memory; retry {k + 1} in {wait_s} s)", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait_s)
    tag = f"{tag} {kn}"
    ok = R.check(torch.equal(d1, d2), f"{tag}: two calls differ")
    if orient < 2:
        q = floor_ratio(d1, orc)
        ok &= R.check(q <= slack, f"{tag}: {q:.4f} x floor > {slack}")
    elif rnd == 1:
        q = rel(d1.double() - c.double(), orc)
        ok &= R.check(q <= 1e-4, f"{tag}: rel(D - C, f64) {q:.3e} > 1e-4")
    else:
        ok &= R.check(torch.equal(d0, d0.to(torch.bfloat16).float()), f"{tag}: D(0) not bf16")
        q = floor_ratio(d0.to(torch.bfloat16), orc)
        ok &= R.check(q <= slack, f"{tag}: D(0) {q:.4f} x floor > {slack}")
        ok &= R.check(torch.equal(d1, c + d0), f"{tag}: D(C) != C + D(0)")
    return ok, q


@R.case
def every_candidate():
    """Every candidate of every owner layer and orientation (fwd, dgrad, wgrad accurate, wgrad levanter) against f64,
    called twice: the gate covers whatever kernel a later timed search picks. On sm_90 (the training GPU) every layer;
    elsewhere (the shared A100, ~200 candidates per wgrad shape) one layer per size and the head."""
    gen = torch.Generator(device="cuda").manual_seed(6)
    buf = ctypes.create_string_buffer(1 << 16)
    total = 0
    layers = LAYERS if lib.nm_device_arch() == 90 else [
        ly for ly in LAYERS if ly[0] in ("130m.qkv", "300m.down", "520m.gate_up", "1_2b.down", "head.512")]
    for name, K, N in layers:
        T = tokens_for(N)
        x = torch.randn(T, K, generator=gen, device="cuda").to(torch.bfloat16)
        w = (torch.randn(N, K, generator=gen, device="cuda") / K ** 0.5).to(torch.bfloat16)
        dy = torch.randn(T, N, generator=gen, device="cuda").to(torch.bfloat16)
        orients = [(0, 1, "fwd"), (1, 1, "dgrad")]
        if N < 100000:   # the CE head's wgrad is S5's
            orients += [(2, 1, "wgrad_acc"), (2, 0, "wgrad_lev")]
        for orient, rnd, oname in orients:
            Tg = 16384 if orient == 2 else T
            if orient == 2 and Tg != T:
                x_, dy_ = torch.randn(Tg, K, generator=gen, device="cuda").to(torch.bfloat16), \
                    torch.randn(Tg, N, generator=gen, device="cuda").to(torch.bfloat16)
            else:
                x_, dy_ = x, dy
            for k in range(8):   # the search allocates scratch: retry a CUDA out-of-memory error (shared GPU)
                rc = lib.dll.nm_gemm_candidates(orient, rnd, Tg, N, K, buf, len(buf), None)
                err = lib.dll.nm_last_error().decode() if rc else ""
                if rc == 0 or "out of memory" not in err or k == 7:
                    break
                print(f"    ({name}.{oname}: CUDA out of memory in the search; retry {k + 1} in 30 s)", flush=True)
                torch.cuda.empty_cache()
                time.sleep(30)
            R.check(rc == 0, f"{name}.{oname}: nm_gemm_candidates failed: {err}")
            if rc != 0:
                continue
            names = [n for n in buf.value.decode().split("\n") if n]
            if orient == 0:
                a_, b_, orc = x_, w, x_.double() @ w.double().t()
            elif orient == 1:
                a_, b_, orc = dy_, w, dy_.double() @ w.double()
            else:
                a_, b_, orc = dy_, x_, dy_.double().t() @ x_.double()
            c = torch.randn(orc.shape[0], orc.shape[1], generator=gen, device="cuda") if orient == 2 else None
            worst, worst_name, bad = 0.0, "", 0
            for kn in names:
                ok, q = candidate_check(f"{name}.{oname}", kn, orient, rnd, a_, b_, orc, c, Tg, N, K)
                bad += not ok
                if q > worst:
                    worst, worst_name = q, kn
            total += len(names)
            unit = "rel(D - C, f64)" if (orient == 2 and rnd == 1) else "x floor"
            wq = f"{worst:.3e}" if unit.startswith("rel") else f"{worst:.4f}"
            print(f"  {name}.{oname}: T {Tg}: {len(names)} candidates, {bad} failing, worst {wq} {unit} "
                  f"({worst_name})")
    print(f"  {total} candidates checked")


@R.case
def custom_kernels():
    """Every registered custom kernel (CUTLASS SM90) on every linear orientation it accepts: the same bounds as the
    cuBLASLt path, two calls bitwise equal (stream-K with the deterministic fixup included). wgrad runs in place on a
    random nonzero C (beta = 1, as in training): accurate rel(D - C, f64); levanter D(C) == C + D(0) bitwise."""
    n = lib.dll.nm_gemm_custom_count()
    if n == 0:
        raise nmk4.Skip("no custom GEMM kernels in this build (sm_90a units)")
    if lib.nm_device_arch() != 90:
        raise nmk4.Skip("custom kernels are sm_90a")
    gen = torch.Generator(device="cuda").manual_seed(5)
    names = [lib.dll.nm_gemm_custom_name(i).decode() for i in range(n)]
    print(f"  {n} custom kernels")
    T = 16384
    for lname, K, N in [("130m.qkv", 512, 1536), ("520m.gate_up", 1024, 7168), ("1_2b.down", 7168, 2048)]:
        x = torch.randn(T, K, generator=gen, device="cuda").to(torch.bfloat16)
        w = (torch.randn(N, K, generator=gen, device="cuda") / K ** 0.5).to(torch.bfloat16)
        dy = torch.randn(T, N, generator=gen, device="cuda").to(torch.bfloat16)
        y64 = x.double() @ w.double().t()
        dx64 = dy.double() @ w.double()
        p64 = dy.double().t() @ x.double()
        c = torch.randn(N, K, generator=gen, device="cuda")
        for i, nm in enumerate(names):
            for orient, rnd in ((0, 1), (1, 1), (2, 1), (2, 0)):
                if orient == 0:
                    a_, b_, out = x, w, torch.empty(T, N, dtype=torch.bfloat16, device="cuda")
                elif orient == 1:
                    a_, b_, out = dy, w, torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
                else:
                    a_, b_, out = dy, x, c.clone()
                rc = lib.dll.nm_linear_custom(i, orient, rnd, ptr(a_), ptr(b_), ptr(out), T, N, K, None)
                if rc != 0:
                    continue   # the kernel does not take this orientation
                out2 = c.clone() if orient == 2 else torch.zeros_like(out)
                lib.nm_linear_custom(i, orient, rnd, ptr(a_), ptr(b_), ptr(out2), T, N, K, None)
                torch.cuda.synchronize()
                tag = f"{lname} {nm} orient {orient} rounding {rnd}"
                R.check(torch.equal(out, out2), f"{tag}: two calls differ")
                if orient == 0:
                    nmk4.nmk.accurate_check(R, tag, out, y64)
                elif orient == 1:
                    nmk4.nmk.accurate_check(R, tag, out, dx64)
                elif rnd == 1:
                    r = rel(out.double() - c.double(), p64)
                    print(f"    {tag}: rel(D - C, f64) {r:.3e}")
                    R.check(r <= 1e-4, f"{tag}: rel(D - C, f64) {r:.3e} > 1e-4")
                else:
                    d0 = torch.zeros(N, K, device="cuda")
                    lib.dll.nm_linear_custom(i, orient, rnd, ptr(a_), ptr(b_), ptr(d0), T, N, K, None)
                    torch.cuda.synchronize()
                    R.check(torch.equal(d0, d0.to(torch.bfloat16).float()), f"{tag}: D(0) not bf16-valued")
                    nmk4.nmk.accurate_check(R, tag, d0.to(torch.bfloat16), p64)
                    same = torch.equal(out, c + d0)
                    print(f"    {tag}: D(C) == C + D(0) bitwise: {same}")
                    R.check(same, f"{tag}: D(C) != C + f32(bf16(acc))")


@R.case
def gemm_f32():
    gen = torch.Generator(device="cuda").manual_seed(4)
    for (m, n, k) in [(512, 512, 512), (768, 2304, 768), (1024, 1024, 3584)]:
        a = torch.randn(m, k, generator=gen, device="cuda")
        b = torch.randn(k, n, generator=gen, device="cuda")
        p64 = a.double() @ b.double()
        for comp, tol, nm in [(2, 1e-6, "pedantic"), (1, 2e-3, "tf32")]:
            d = torch.empty(m, n, device="cuda")
            lib.nm_gemm_f32(comp, ptr(a), ptr(b), ptr(d), m, n, k, 0, None)
            torch.cuda.synchronize()
            r = rel(d, p64)
            print(f"  {m}x{n}x{k} {nm}: rel {r:.3e} (bound {tol:.0e})")
            R.check(r <= tol, f"{nm} {m}x{n}x{k}: rel {r:.3e} > {tol}")


R.run(ARGS.cases)
