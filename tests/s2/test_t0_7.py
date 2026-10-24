#!/usr/bin/env python3
"""Gate T0.7 (docs/spec/kaiyue_baseline.md, parity plan): the schedule tables eta(k), eta_a(k), k = 0..N-1, of all four
sizes are f32-bitwise equal to the logged `optim/learning_rate` and `optim/adam_lr` at every logged k, and at the holes
and k = 0, 1 equal to marincpp's formula and inside their logged neighbours; includes 130m k = 4958
(5.041361e-06 / 2.016544e-06) and 1_2b k = 22887 (exactly 0).

The tables under test are the ones marincpp uses: libmarincpp_kernels.so's nm_schedule_tables loads each fixture
(cache/levref/<size>/schedules, written by tests/s2/schedule_fixture.py from levanter's jitted schedules; DESIGN.md
§4.4) through the C++ loader, which checks it against the config. "The formula" is marincpp's build_schedules (the
same call with from_fixture = 0): optim.md §6 evaluated with the arithmetic XLA:GPU runs for it (src/sched/sched.hpp),
not in the literal f32 operation order optim.md §6 writes down. That literal order differs from the log at thousands
of steps and from the fixture at some holes (printed per size; BUILD_LOG S2 lists the spec amendment). The loader's own
cross-check (formula vs fixture over all four tables, n_ulp_diffs) must be 0 as well.

W&B histories: cache/wandb/run-2026091[4-7]_*-muonh-qwen3-<size>-della4xh100.jsonl.gz, one file per resumed segment,
deduplicated by _step (a step logged twice must carry the same values).

Usage: test_t0_7.py --lib <libmarincpp_kernels.so> --src <repo root>       (plain Python 3, no dependencies)
Exit 0 pass, 1 fail, 77 skip (no fixture yet).
"""
import argparse
import array
import ctypes
import glob
import gzip
import json
import math
import os
import struct
import sys
import tempfile

SIZES = {  # size: (N, peak lr, peak adam_lr, logged steps in k = 2..N-1 per optim.md §6.2)
    "130m": (4959, 0.02, 0.008, 4957),
    "300m": (11444, 0.01, 0.002, 11435),
    "520m": (9918, 0.01, 0.002, 9900),
    "1_2b": (22888, 0.01, 0.0015, 22822),
}
FAILS = []


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("    FAIL " + msg)


def load_lib(path):
    lib = ctypes.CDLL(path)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.nm_schedule_tables.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, fp, fp, fp, fp,
                                       ctypes.POINTER(ctypes.c_int)]
    lib.nm_schedule_tables.restype = ctypes.c_int
    lib.nm_last_error.restype = ctypes.c_char_p
    return lib


def tables(lib, cfg_path, n, from_fixture):
    bufs = [(ctypes.c_float * n)() for _ in range(4)]
    nd = ctypes.c_int(-1)
    rc = lib.nm_schedule_tables(cfg_path.encode(), n, from_fixture, *bufs, ctypes.byref(nd))
    if rc != 0:
        raise SystemExit(f"nm_schedule_tables({cfg_path}, from_fixture={from_fixture}) = {rc}: "
                         f"{lib.nm_last_error().decode()}")
    return [list(b) for b in bufs], nd.value


def read_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return list(a)


def logged(src, size):
    pat = os.path.join(src, "cache", "wandb", f"run-2026091[4-7]_*-muonh-qwen3-{size}-della4xh100.jsonl.gz")
    files = sorted(glob.glob(pat))
    out, dups = {}, 0
    for fn in files:
        with gzip.open(fn, "rt") as f:
            for line in f:
                r = json.loads(line)
                if "optim/learning_rate" not in r and "optim/adam_lr" not in r:
                    continue
                k = r["_step"]
                v = (r.get("optim/learning_rate"), r.get("optim/adam_lr"))
                if k in out:
                    dups += 1
                    check(out[k] == v, f"{size}: _step {k} logged twice with different values {out[k]} vs {v}")
                out[k] = v
    return files, out, dups


def spec_formula(size, k, peak):
    """optim.md §6 literally in f32 (IEEE division; math.cos of the f32 argument, rounded to f32): informational."""
    n = SIZES[size][0]
    if size == "130m":
        td, st = int(0.8 * n), n - int(0.8 * n)
        if k < st:
            return f32(peak)
        c = float(min(max(k - st, 0), td))
        return f32(f32(peak) * f32(1.0 - f32(c / td)))
    w, td = 1000, n - 1000
    if k < w:
        c = float(min(k, w))
        return f32(f32(f32(-peak) * f32(1.0 - f32(c / w))) + f32(peak))
    c = float(min(k - w, td))
    x = f32(f32(f32(math.pi) * c) / td)
    return f32(f32(peak) * f32(0.5 * f32(1.0 + f32(math.cos(x)))))


def check_size(lib, src, tmp, size):
    n, peak, apeak, n_logged_spec = SIZES[size]
    print(f"== {size} (N = {n})")
    fdir = os.path.abspath(os.path.join(src, "cache", "levref", size, "schedules"))
    cfg = os.path.join(tmp, f"{size}.toml")
    with open(cfg, "w") as f:
        f.write(f'[model]\nsize = "{size}"\n\n[optim]\nschedule_fixture = "{fdir}"\n')
    (lr, alr, bc1, bc2), nd = tables(lib, cfg, n, 1)
    (flr, falr, fbc1, fbc2), _ = tables(lib, cfg, n, 0)
    check(nd == 0, f"{size}: loader cross-check counts {nd} formula-vs-fixture differences (want 0)" if nd >= 0 else
                   f"{size}: loader cross-check unavailable (n_ulp_diffs = {nd}: the formula refuses the config)")
    for name, got in (("sched_main", lr), ("sched_adam", alr), ("bc1", bc1), ("bc2", bc2)):
        raw = read_f32(os.path.join(fdir, name + ".f32"))
        check([bits(x) for x in raw] == [bits(x) for x in got], f"{size}: C ABI {name} differs from the file")

    files, log, dups = logged(src, size)
    ks = sorted(k for k in log if 0 <= k < n)
    if ks:
        print(f"   {len(files)} W&B segments, {len(log)} logged steps ({dups} duplicates), k in [{ks[0]}, {ks[-1]}]")
        check(len(ks) == len(log), f"{size}: {len(log) - len(ks)} logged steps outside [0, N)")
        check(len(ks) == n_logged_spec, f"{size}: {len(ks)} logged steps, optim.md §6.2 says {n_logged_spec}")
        bad = {"main": [], "adam": []}
        for k in ks:
            for j, (key, tab) in enumerate((("main", lr), ("adam", alr))):
                v = log[k][j]
                if v is None or f32(v) != v:
                    check(False, f"{size}: k={k} {key} logged value {v!r} is not an f32")
                    continue
                if bits(tab[k]) != bits(v):
                    bad[key].append((k, v, tab[k]))
        for key in bad:
            check(not bad[key], f"{size}: {key} table differs from the log at {len(bad[key])} steps, first "
                                f"{bad[key][:3]}")
        print(f"   logged: {2 * len(ks)} values compared bitwise, {len(bad['main']) + len(bad['adam'])} differ")

        holes = sorted(set(range(n)) - set(ks))
        hole_bad, bracket_bad, spec_diff = [], [], 0
        for k in holes:
            for key, tab, form, p in (("main", lr, flr, peak), ("adam", alr, falr, apeak)):
                if bits(tab[k]) != bits(form[k]):
                    hole_bad.append((k, key, tab[k], form[k]))
                if bits(spec_formula(size, k, p)) != bits(tab[k]):
                    spec_diff += 1
                lo = max((j for j in ks if j < k), default=None)
                hi = min((j for j in ks if j > k), default=None)
                if lo is not None and hi is not None:
                    j = 0 if key == "main" else 1
                    a_, b_ = log[lo][j], log[hi][j]
                    if not (min(a_, b_) <= tab[k] <= max(a_, b_)):
                        bracket_bad.append((k, key, tab[k], a_, b_))
        check(not hole_bad, f"{size}: fixture differs from the formula at holes {hole_bad[:4]}")
        check(not bracket_bad, f"{size}: hole values outside their logged neighbours {bracket_bad[:4]}")
        print(f"   holes ({len(holes)}: {holes[:12]}{' ...' if len(holes) > 12 else ''}): fixture == formula at "
              f"{2 * len(holes) - len(hole_bad)} of {2 * len(holes)}; literal f32 optim.md §6 (math.cos) differs at "
              f"{spec_diff}")
        check(0 in holes and 1 in holes, f"{size}: k = 0, 1 are expected to be unlogged")
    else:
        check(False, f"{size}: no W&B history in cache/wandb ({len(files)} files match "
                     f"run-2026091[4-7]_*-muonh-qwen3-{size}-della4xh100.jsonl.gz, 0 logged steps in [0, N))")

    if size == "130m":
        check(bits(lr[0]) == bits(f32(peak)) and bits(lr[1]) == bits(f32(peak)), "130m: eta(0), eta(1) != peak")
        check("%.7g" % lr[4958] == "5.041361e-06", f"130m: eta(4958) = {lr[4958]!r}")
        check("%.7g" % alr[4958] == "2.016544e-06", f"130m: eta_a(4958) = {alr[4958]!r}")
        print(f"   k=4958: {lr[4958]:.7g} / {alr[4958]:.7g}")
    else:
        for key, tab, p in (("main", lr, peak), ("adam", alr, apeak)):
            check(bits(tab[0]) == 0, f"{size}: {key} eta(0) = {tab[0]!r}, want 0")
            e1 = f32(f32(f32(1.0 - f32(1.0 * f32(1.0 / 1000.0))) * f32(-p)) + f32(p))
            check(bits(tab[1]) == bits(e1), f"{size}: {key} eta(1) = {tab[1]!r}, want {e1!r}")
    if size == "1_2b":
        check(bits(lr[22887]) == 0 and bits(alr[22887]) == 0, f"1_2b: k=22887 = {lr[22887]!r} / {alr[22887]!r}")
        print(f"   k=22887: {lr[22887]!r} / {alr[22887]!r}")
    bcd = sum(bits(x) != bits(y) for x, y in zip(bc1 + bc2, fbc1 + fbc2))
    print(f"   bias correction 1-b^t (no logged values; T1/S6 use them): formula differs at {bcd} of {2 * n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    missing = [s for s in SIZES if not os.path.exists(os.path.join(a.src, "cache", "levref", s, "schedules",
                                                                    "schedule.toml"))]
    if len(missing) == len(SIZES):
        print("SKIP t0_7: no schedule fixture under cache/levref/<size>/schedules; make them with "
              "tests/s2/schedule_fixture.py on a GPU (DESIGN.md §4.4)")
        return 77
    for s in missing:
        check(False, f"{s}: fixture cache/levref/{s}/schedules missing (the other sizes have one)")
    lib = load_lib(a.lib)
    with tempfile.TemporaryDirectory(prefix="t0_7_", dir=os.getcwd()) as tmp:   # removed on every exit path
        for size in SIZES:
            if size not in missing:
                check_size(lib, a.src, tmp, size)
    print(f"t0_7: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
