"""Shared helpers of the S6 gate tests: the optimizer C ABI of libmarincpp_kernels.so through ctypes (capi.h optimizer
block and src/optim/capi_optim.h), torch tensors on the GPU, f64 metrics (kaiyue_baseline.md "Conventions"), the
levanter fixtures of cache/levref, and a minimal case runner.

Every test script takes --lib <libmarincpp_kernels.so> --src <repo root>. Exit 0 = all cases passed, 77 = all skipped
(no GPU or no fixture), 1 = a failure (FAIL, ERROR, or SPECFAIL: misses a binding bar of the spec but holds a proposed
rule that awaits a decision; reported with both numbers, never counted as a pass).
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

try:
    import torch
except ImportError:  # the gate must not vanish: report it as a failure that says why
    print("FAIL: torch is not importable in this interpreter (configure with -DNM_TORCH_PYTHON=<python with torch>)")
    sys.exit(1)

C = ctypes
VP, I32, I64, F32, F64 = C.c_void_p, C.c_int, C.c_int64, C.c_float, C.c_double

RULE = {"muonh": 0, "adamh": 1, "adam": 2, "frozen": 3}
# NS precision codes of nm_opt_create: + 4 (suffix s) symmetric NS blocks, + 8 (suffix c) c folded into alpha
PREC = {"f32": 0, "tf32": 1, "3xtf32": 2, "f32s": 4, "tf32s": 5, "f32c": 8, "tf32c": 9}


class OptGroup(C.Structure):
    _fields_ = [("rule", I32), ("lr_table", I32), ("clip_group", I32), ("lr_scale", F64), ("b1", F64), ("b2", F64),
                ("eps", F64), ("momentum", F64), ("max_grad_norm", F64), ("weight_decay", F64)]


class OptTensor(C.Structure):
    _fields_ = [("group", I32), ("rows", I64), ("cols", I64), ("base_row", I64), ("p32", VP), ("p16", VP), ("g", VP),
                ("s0", VP), ("s1", VP), ("grad_scale", F32), ("update_scale", F32), ("n_mats", I64),
                ("mat_rows", I64), ("row_map", VP), ("n_owned", I64), ("owned", VP), ("u_out", VP)]


SIGS = {
    "nm_last_error": (C.c_char_p, []),
    "nm_newton_schulz": (I32, [I32, I32, VP, I64, I64, I64, I32, F32, F32, F32, F32, VP, VP]),
    "nm_muonh_update": (I32, [VP, VP, VP, VP, I64, I64, I64, F32, F32, F32, I32, F32, I32, F32, VP, VP]),
    "nm_adamh_update": (I32, [VP, VP, VP, VP, VP, I64, I64, I64, F32, F32, F32, F32, F32, I32, F32, VP, VP, VP, I32,
                              VP]),
    "nm_adam_update": (I32, [VP, VP, VP, VP, VP, I64, I64, I64, F32, F32, F32, F32, F32, I32, F32, VP, VP, VP, I32,
                             VP]),
    "nm_opt_create": (I32, [C.POINTER(OptGroup), I32, C.POINTER(OptTensor), I32, I32, I32, F64, I64, I32, I32,
                            C.POINTER(VP)]),
    "nm_opt_set_tables": (I32, [VP, VP, VP, VP, VP, VP, I64]),
    "nm_opt_prepare": (I32, [VP, VP]),
    "nm_opt_phase": (I32, [VP, I32, VP]),
    "nm_opt_step": (I32, [VP, I32, VP]),
    "nm_opt_buffers": (I32, [VP, C.POINTER(VP), C.POINTER(I32), C.POINTER(VP), C.POINTER(VP), C.POINTER(I32),
                             C.POINTER(VP), C.POINTER(I32), C.POINTER(I32)]),
    "nm_opt_destroy": (I32, [VP]),
    "nm_ns_gemm": (I32, [I32, VP, VP, VP, I64, I64, I64, VP]),
    "nm_schedule_tables": (I32, [C.c_char_p, I64, I32, VP, VP, VP, VP, C.POINTER(I32)]),
}


class Lib:
    def __init__(self, path):
        self.lib = C.CDLL(path)
        for name, (res, args) in SIGS.items():
            f = getattr(self.lib, name)
            f.restype, f.argtypes = res, args

    def __getattr__(self, name):
        f = getattr(self.lib, name)

        def call(*a):
            rc = f(*a)
            if name != "nm_last_error" and SIGS[name][0] is I32 and rc != 0:
                raise RuntimeError(f"{name} -> {rc}: {self.lib.nm_last_error().decode()}")
            return rc

        return call


def ptr(t):
    return None if t is None else C.c_void_p(t.data_ptr())


_cudart = None


def d2h(dptr, nbytes):
    """Bytes at a device address (synchronous cudaMemcpy through the cudart the library links)."""
    global _cudart
    if _cudart is None:
        _cudart = C.CDLL("libcudart.so.12")
        _cudart.cudaMemcpy.argtypes = [VP, VP, C.c_size_t, I32]
    out = np.empty(nbytes, np.uint8)
    rc = _cudart.cudaMemcpy(out.ctypes.data, dptr, nbytes, 2)
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy -> {rc}")
    return out


def stream():
    return C.c_void_p(torch.cuda.current_stream().cuda_stream)


def rel(a, b):
    """||a - b|| / ||b|| in f64 (0 when both are 0)."""
    a = torch.as_tensor(a).double().cpu() if not isinstance(a, torch.Tensor) else a.double()
    b = torch.as_tensor(b).double().cpu() if not isinstance(b, torch.Tensor) else b.double()
    b = b.to(a.device)
    nb = torch.linalg.vector_norm(b).item()
    nd = torch.linalg.vector_norm(a - b).item()
    return nd / nb if nb > 0 else (0.0 if nd == 0 else float("inf"))


def f32_bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def ulp_diff_max(a, b):
    """max |a - b| in f32 ulps of b (element-wise, as integers on the ordered bit patterns)."""
    def ordered(x):
        u = x.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(u < 0, -(u & 0x7FFFFFFF), u)
    return int(np.max(np.abs(ordered(np.asarray(a)) - ordered(np.asarray(b))))) if np.size(a) else 0


def schedule_tables(lib, src, size, n):
    """lr_main, lr_adam, bc1, bc2 [n] f32 from the schedule fixture of record (cache/levref/<size>/schedules), loaded
    by marincpp's own loader (nm_schedule_tables)."""
    fdir = os.path.abspath(os.path.join(src, "cache", "levref", size, "schedules"))
    if not os.path.isdir(fdir):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, f"{size}.toml")
        with open(cfg, "w") as f:
            f.write(f'[model]\nsize = "{size}"\n\n[optim]\nschedule_fixture = "{fdir}"\n')
        out = [np.zeros(n, np.float32) for _ in range(4)]
        nd = C.c_int(0)
        lib.nm_schedule_tables(cfg.encode(), n, 1, *[o.ctypes.data_as(VP) for o in out], C.byref(nd))
    return out


# ------------------------------------------------------------------------------------------------ fixtures
def fixture_dir(src, *parts):
    return os.path.join(src, "cache", "levref", *parts)


def load_tree(d, names=None):
    """{name: np.ndarray} of a fixture directory (index.json + <name>.npy)."""
    idx = json.load(open(os.path.join(d, "index.json")))
    out = {}
    for name, e in idx.items():
        if names is not None and name not in names:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("npy"):
            out[name] = np.load(os.path.join(d, name + ".npy"))
        elif "value" in e:
            out[name] = np.asarray(e["value"])
    return out


def index(d):
    return json.load(open(os.path.join(d, "index.json")))


# ------------------------------------------------------------------------------------------------ runner
class Runner:
    def __init__(self, name):
        self.name = name
        self.fail = 0
        self.ok = 0
        self.skip = 0
        self.specfail = 0
        self.rows = []

    def case(self, title, fn, *a, **k):
        try:
            r = fn(*a, **k)
            if r == "skip":
                self.skip += 1
                print(f"SKIP {title}")
            elif isinstance(r, tuple) and r[0] == "specfail":
                self.specfail += 1
                print(f"SPECFAIL {title}: {r[1]}")
            else:
                self.ok += 1
                print(f"PASS {title}")
        except AssertionError as e:
            self.fail += 1
            print(f"FAIL {title}: {e}")
        except Exception:
            self.fail += 1
            print(f"ERROR {title}")
            traceback.print_exc()
        sys.stdout.flush()

    def finish(self):
        print(f"{self.name}: {self.ok} passed, {self.specfail} specfail, {self.fail} failed, {self.skip} skipped")
        if self.fail or self.specfail:
            return 1
        return 77 if self.ok == 0 else 0


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--json", default=None, help="write the measured numbers here")
    ap.add_argument("--only", default=None, help="comma-separated case filter")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        sys.exit(77)
    return a


def retry_oom(fn, *a, tries=8, wait=60, **k):
    """fn(*a, **k), retried after a CUDA out-of-memory error: on the shared vis-node GPU an OOM is the state of other
    jobs, not a verdict on the code (any other error, and the last OOM, propagate)."""
    for i in range(tries):
        try:
            return fn(*a, **k)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or i == tries - 1:
                raise
            print(f"  CUDA out of memory; retry {i + 1} of {tries - 1} in {wait} s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(wait)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
