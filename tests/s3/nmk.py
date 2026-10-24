"""Shared helpers of the S3 gate tests: the C ABI of libmarincpp_kernels.so through ctypes, torch tensors on the GPU,
bf16 helpers, error metrics (DESIGN.md §10.2) and a minimal case runner.

Every test script takes --lib <libmarincpp_kernels.so> --src <repo root>. Exit 0 = all cases passed, 77 = all skipped
(no GPU), 1 = a failure.
"""
import argparse
import ctypes
import math
import os
import sys
import traceback

import numpy as np

try:
    import torch
except ImportError:  # the gate must not vanish: report it as a failure that says why
    print("FAIL: torch is not importable in this interpreter (configure with -DNM_TORCH_PYTHON=<python with torch>)")
    sys.exit(1)

C = ctypes
VP, I32, I64, U64, F32, F64 = C.c_void_p, C.c_int, C.c_int64, C.c_uint64, C.c_float, C.c_double

SIGS = {
    "nm_last_error": (C.c_char_p, []),
    "nm_device_arch": (I32, []),
    "nm_doc_meta": (I32, [VP, VP, I32, I32, I32, I32, VP, VP, VP, VP, VP, VP, VP, VP]),
    "nm_embed_fwd": (I32, [VP, VP, VP, I64, I32, VP]),
    "nm_embed_fwd_dt": (I32, [I32, VP, VP, VP, I64, I32, VP]),
    "nm_embed_bwd": (I32, [VP, VP, VP, I64, I32, I32, I32, VP, VP, U64, VP]),
    "nm_embed_bwd_dt": (I32, [I32, VP, VP, VP, I64, I32, I32, I32, VP, VP, U64, VP]),
    "nm_embed_bwd_workspace_bytes": (U64, [I64, I32]),
    "nm_rmsnorm_fwd": (I32, [I32, I32, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_rmsnorm_bwd_add": (I32, [I32, I32, VP, VP, VP, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_rmsnorm_bwd_add_partial": (I32, [I32, I32, VP, VP, VP, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_sandwich_tail_fwd": (I32, [I32, I32, VP, VP, VP, VP, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_sandwich_tail_bwd": (I32, [I32, I32, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_qknorm_rope_fwd": (I32, [I32, I32, VP, I64, VP, VP, VP, VP, VP, VP, VP, VP, VP, I64, I32, I32, I32, F32,
                                 F32, VP]),
    "nm_qknorm_rope_bwd": (I32, [I32, I32, VP, VP, VP, I64, VP, VP, VP, VP, VP, VP, VP, VP, I64, VP, VP, I64, I32,
                                 I32, I32, F32, F32, VP]),
    "nm_dq_post_qknorm_rope_bwd": (I32, [I32, VP, VP, I64, VP, VP, VP, VP, VP, VP, I64, VP, I64, I32, I32, F32, F32,
                                         VP]),
    "nm_rope_tables": (I32, [I32, I32, F64, VP, VP, VP]),
    "nm_rope_tables_method": (I32, [I32, I32, I32, F64, VP, VP, VP]),
    "nm_rope_fwd": (I32, [I32, I32, VP, I64, VP, VP, VP, VP, I64, I32, I32, F32, VP]),
    "nm_swiglu_fwd": (I32, [I32, I32, VP, VP, I64, I32, VP]),
    "nm_swiglu_bwd": (I32, [I32, I32, VP, VP, VP, I64, I32, VP]),
    "nm_norm_partial_rows": (I64, [I64]),
    "nm_reduce_partials_rows": (I32, [VP, I32, I32, VP, I32, I32, VP]),
    "nm_sum_fixed": (I32, [VP, I64, VP, VP]),
    "nm_cast_f32_bf16": (I32, [VP, VP, I64, VP]),
}

LEV, ACC = 0, 1          # rounding
F32S, BF16S = 0, 1       # storage dtype


class Lib:
    def __init__(self, path):
        self.dll = C.CDLL(path)
        for name, (res, args) in SIGS.items():
            f = getattr(self.dll, name)
            f.restype, f.argtypes = res, args

    def __getattr__(self, name):
        f = getattr(self.dll, name)

        def call(*a):
            r = f(*a)
            if f.restype is I32 and name not in ("nm_device_arch",) and r != 0:
                raise RuntimeError(f"{name} returned {r}: {self.dll.nm_last_error().decode()}")
            return r
        return call


def ptr(t):
    return None if t is None else VP(t.data_ptr())


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("cases", nargs="*")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------ bf16 helpers
def rb(x):
    """Round an f32 tensor to bf16 precision (RNE), keep f32."""
    return x.to(torch.bfloat16).to(torch.float32)


def bits16(t):
    """int32 tensor of the bf16 bit patterns of a bf16 tensor."""
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def ulp_diff_bf16(a, b):
    """|a - b| in bf16 ulps (ordered-integer distance), both bf16 tensors; NaN-free inputs."""
    def ordered(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (ordered(a) - ordered(b)).abs()


def ulp_diff_f32(a, b):
    def ordered(t):
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    return (ordered(a) - ordered(b)).abs()


def bitwise_check(R, name, k, emu, orc=None, min_frac=0.9999):
    """Kernel k (bf16) against the levanter emulation emu (bf16) of the same op:
      (1) bitwise-equal fraction >= min_frac (1.0 = every element; used where the inputs make every sum exact);
      (2) with the f64 oracle orc: rel(k, emu) <= rel(emu, orc) (oracle rule (ii) of the parity plan: the port differs
          from levanter by less than levanter's own rounding error)."""
    eq = (bits16(k) == bits16(emu)).double().mean().item()
    raw = ulp_diff_bf16(k, emu).max().item()
    msg = f"    {name}: bitwise {100 * eq:.4f}%  max ulp {raw}"
    if orc is not None:
        r_ke, r_eo = rel(k, emu), rel(emu, orc)
        msg += f"  rel(k, emu) {r_ke:.2e} <= rel(emu, f64) {r_eo:.2e}"
        R.check(r_ke <= r_eo, f"{name}: rel(k, emu) {r_ke:.3e} > rel(emu, f64) {r_eo:.3e}")
    print(msg)
    R.check(eq >= min_frac, f"{name}: bitwise fraction {eq:.6f} < {min_frac}")
    return eq


def exact_sum_rows(T, D, gen, vals=(-3, -2, -1, 1, 2, 3), q=0.25):
    """bf16 [T, D] with entries in q * vals: every product with another such tensor (or its bf16 multiple by a gain
    1 + j/16) and every row sum of such products is exact in f32 at D <= 4096, so a levanter-mode kernel must equal the
    emulation bitwise whatever its summation order."""
    v = torch.tensor(vals, dtype=torch.float32, device="cuda")
    idx = torch.randint(0, len(vals), (T, D), generator=gen, device="cuda")
    return (v[idx] * q).to(torch.bfloat16)


def exact_gain(n, gen):
    return (1.0 + torch.randint(-4, 5, (n,), generator=gen, device="cuda").float() / 16).to(torch.bfloat16)


def rel(a, b):
    a64, b64 = a.double(), b.double()
    nb = torch.linalg.vector_norm(b64).item()
    d = torch.linalg.vector_norm(a64 - b64).item()
    return d / nb if nb > 0 else d


ACC_SLACK = 1.05   # accurate mode: rel(kernel, f64) <= 1.05 x the single-rounding floor rel(bf16(f64), f64)
F32_TOL = 1e-6     # an f32 output (rstd, gain gradients) against f64


def accurate_check(R, name, k, orc, lev=None, slack=ACC_SLACK):
    """DESIGN.md §3.0 contract of Rounding::Accurate for a bf16 output: f32 inside, one rounding at the store. Its error
    against the f64 oracle of the same stored inputs can then exceed the floor rel(bf16(orc), orc) of rounding the exact
    result once only by the rare f32-induced rounding flips: rel(k, orc) <= slack x floor. With `lev` (the levanter
    emulation of the same output) the gate also shows it has power on these inputs: a kernel that rounded like levanter
    would fail it (rel(lev, orc) > slack x floor). Returns rel(k, orc)."""
    floor = rel(orc.float().to(torch.bfloat16), orc)
    rk = rel(k, orc)
    msg = f"    {name}: rel(kernel, f64) {rk:.4e}  single-rounding floor {floor:.4e}  ratio {rk / floor:.4f}"
    if lev is not None:
        rl = rel(lev, orc)
        msg += f"  (levanter rounding: {rl / floor:.3f} x floor)"
        R.check(rl > slack * floor, f"{name}: the gate has no power here: levanter rounding {rl:.4e} passes "
                                    f"{slack} x floor {floor:.4e}")
    print(msg)
    R.check(rk <= slack * floor, f"{name}: accurate-mode rel {rk:.4e} > {slack} x single-rounding floor {floor:.4e}")
    return rk


def f32_bound(R, name, k, orc, tol=F32_TOL):
    r = rel(k, orc)
    print(f"    {name}: rel(f32 output, f64) {r:.3e}")
    R.check(r <= tol, f"{name}: rel {r:.3e} > {tol}")
    return r


def row_report(name, a, b, width, tensor_rel, factor=10.0, floor=1e-7, show=10):
    """DESIGN.md §10.2 per-row report: rows of `width` elements; per-row rel = ||a_r - b_r|| / ||b_r||. Passes when
    every row's rel <= factor x max(tensor_rel, floor). Prints the `show` worst rows. Returns (ok, worst)."""
    a2 = a.double().reshape(-1, width)
    b2 = b.double().reshape(-1, width)
    nb = torch.linalg.vector_norm(b2, dim=1)
    nd = torch.linalg.vector_norm(a2 - b2, dim=1)
    rr = torch.where(nb > 0, nd / nb.clamp_min(1e-300), nd)
    bound = factor * max(tensor_rel, floor)
    worst = torch.topk(rr, min(show, rr.numel()))
    ok = bool((rr <= bound).all())
    rows = ", ".join(f"{int(i)}:{v:.2e}" for v, i in zip(worst.values.tolist(), worst.indices.tolist()))
    print(f"    rows {name}: max row rel {worst.values[0].item():.3e} (bound {bound:.3e}) worst [{rows}]")
    return ok, worst.values[0].item()


# ------------------------------------------------------------------------------------------------ runner
class Skip(Exception):
    pass


class XFail(Exception):
    """A known failure against the binding spec text that is not this step's to waive (the owner decides). The case
    is reported as XFAIL, never as passed; any other check failure in the same case is still a FAIL."""


def is_oom(e):
    """A CUDA out-of-memory error (torch's allocator, or a cudaMalloc* inside the C ABI)."""
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e)


def run_with_oom_retry(fn, runner, attempts=4, wait_s=15):
    """The vis node's GPU is shared: other jobs sometimes hold all but a few MB of it. A case that dies with a CUDA
    out-of-memory error before any check failed is rerun (up to `attempts` times, after `wait_s` s); a check failure or
    any other error is never retried."""
    import time
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (re-raised below unless it is an OOM)
            if not is_oom(e) or runner.failed_now or k == attempts - 1:
                raise
            print(f"    (CUDA out of memory: other processes hold the GPU; retry {k + 1} in {wait_s} s)", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait_s)


class Runner:
    def __init__(self, suite):
        self.suite, self.cases, self.failed, self.passed, self.skipped = suite, [], [], 0, 0
        self.xfailed = []

    def case(self, fn):
        self.cases.append(fn)
        return fn

    def check(self, cond, msg):
        if not cond:
            self.failed_now = True
            print(f"    CHECK FAILED: {msg}")
        return cond

    def run(self, only=()):
        if not torch.cuda.is_available():
            print(f"SKIP {self.suite}: no CUDA device")
            sys.exit(77)
        for fn in self.cases:
            if only and fn.__name__ not in only:
                continue
            self.failed_now = False
            name = f"{self.suite}.{fn.__name__}"
            print(f"RUN  {name}", flush=True)
            try:
                run_with_oom_retry(fn, self)
            except XFail as e:
                if self.failed_now:
                    print(f"FAIL {name}")
                    self.failed.append(name)
                else:
                    print(f"XFAIL {name} (NOT PASSED; known failure against the spec text): {e}")
                    self.xfailed.append(name)
                continue
            except Skip as e:
                if self.failed_now:
                    print(f"FAIL {name} (failed, then skipped: {e})")
                    self.failed.append(name)
                else:
                    print(f"SKIP {name}: {e}")
                    self.skipped += 1
                continue
            except Exception:
                traceback.print_exc()
                self.failed_now = True
            if self.failed_now:
                print(f"FAIL {name}", flush=True)
                self.failed.append(name)
            else:
                print(f"PASS {name}", flush=True)
                self.passed += 1
        xf = f", {len(self.xfailed)} XFAIL (not passed: {', '.join(self.xfailed)})" if self.xfailed else ""
        print(f"{self.suite}: {self.passed} passed, {len(self.failed)} failed, {self.skipped} skipped{xf}")
        if self.failed:
            sys.exit(1)
        sys.exit(77 if self.passed == 0 and self.skipped else 0)


# ------------------------------------------------------------------------------------------------ shapes
# The owner's shapes (config presets): 130m, 1_2b, the pls width-512 trunk (pls_d48), plus 300m / 520m widths so every
# row-group geometry (G = 64, 96, 128, 256) runs.
SHAPES = {
    "130m": dict(D=512, H=8, KV=8, hd=64, F=1792),
    "300m": dict(D=768, H=12, KV=12, hd=64, F=2688),
    "520m": dict(D=1024, H=16, KV=8, hd=64, F=3584),
    "1_2b": dict(D=2048, H=16, KV=8, hd=128, F=7168),
    "pls_w512": dict(D=512, H=8, KV=8, hd=64, F=1792),
}
EPS = float(np.float32(1e-5))


def rand_rows(T, D, gen, scale_spread=True):
    """bf16 [T, D] activations with per-row scales spanning 1e-3 .. 1e2 (a row of each extreme included)."""
    x = torch.randn(T, D, generator=gen, device="cuda")
    if scale_spread:
        s = torch.exp(torch.empty(T, 1, device="cuda").uniform_(math.log(0.05), math.log(20.0), generator=gen))
        s[0] = 1e-3
        s[-1] = 100.0
        x = x * s
    return x.to(torch.bfloat16)


def rand_gain(n, gen):
    return (1.0 + 0.2 * torch.randn(n, generator=gen, device="cuda")).to(torch.bfloat16)


def src_path(*p):
    return os.path.join(ARGS.src, *p)


ARGS = None


def init(suite):
    global ARGS
    ARGS = args_parse()
    lib = Lib(ARGS.lib)
    torch.backends.cuda.matmul.allow_tf32 = False
    return lib, Runner(suite)
