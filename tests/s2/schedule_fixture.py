#!/usr/bin/env python3
"""S2 schedule fixture (DESIGN.md §4.4; the design names this tool tools/levref/schedules.py).

Runs levanter's own schedules, jitted, on a GPU and writes the tables of record for one or more sizes:

    cache/levref/<size>/schedules/
        schedule.toml       the levanter optimizer fields the tables depend on + sha256 of every table (no timestamps,
                            hosts or versions, so the directory's content hash is the identity of the tables)
        sched_main.f32      eta(k),   k = 0..N-1   (MuonH, AdamH; peak learning_rate)    little-endian float32
        sched_adam.f32      eta_a(k), k = 0..N-1   (Adam; peak adam_lr)
        bc1.f32, bc2.f32    1 - b1^t, 1 - b2^t,  t = 1..N   (bias-correction denominators of update k = t - 1)
    cache/levref/<size>/schedules_manifest/
        manifest.json       provenance: command, host, jax/jaxlib/optax, device
        vec.stablehlo.txt   the lowered (pre-optimization) StableHLO of the vectorized evaluation
        vec.opt_hlo.txt     the optimized HLO XLA:GPU compiled from it: the evidence for the rewrites the C++ formula
                            ports (`c / T` -> `c * f32(1/T)`, the folded cosine constant), readable without a GPU

How the values are made (nothing here re-derives a formula):
  - The optimizer config is the launcher's own (tools.levref.config.build: the plain baseline as the marin runner
    builds it), and the schedules are `opt.lr_scheduler(N)` and `opt.lr_scheduler(N, override_lr=opt.adam_lr)`, the
    two calls of MuonHConfig.build (LV/optim/muonh.py:58-59).
  - Both are evaluated under jax.jit on the GPU with the update index as a traced int32 (as inject_hyperparams passes
    its int32 count), so XLA:GPU's f32 cos, divisions and any fusion it chooses are the ones the run used. Two
    evaluations must agree bitwise: one vectorized over k = 0..N-1 and one scalar per k through lax.map (the shape
    of the training step, which evaluates one count).
  - The bias correction mirrors optax.tree_utils.tree_bias_correction (`jax.jit(inline=True)`, `1 - decay**count`
    with the Python float decay and the int32 count; _tree_math.py:400-411) on count = 1..N.
  - The XLA_FLAGS of the run environment (tools.levref.config.RUN_ENV) are applied before the backend starts. A value
    already set in the shell that differs from the run's (or is set where the run had none) is refused, not used.

Run (vis-node GPU or an H100 through tools/devq), from the repository root, with the marin venv's Python:
    NANOMARIN_ROOT=<root> <root>/project/marin/.venv/bin/python tests/s2/schedule_fixture.py [--size 130m ...]
    ... tests/s2/schedule_fixture.py --probes     # the S2 probe fixtures, cache/levref/s2_probe/<name>/ (see PROBES)
The T0.7 gate (tests/s2/test_t0_7.py) checks the result against every logged W&B value.
"""
import argparse
import functools
import hashlib
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from tools.levref import common  # noqa: E402  (sets the levref process defaults: no bytecode, W&B disabled)
from tools.levref import config as lcfg  # noqa: E402

FORMAT = 1
TABLES = ("sched_main", "sched_adam", "bc1", "bc2")


def f32_hex(x: float) -> str:
    return "0x%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


def toml_value(v) -> str:
    if v is None:
        raise SystemExit("toml_value(None)")
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if ("." in r or "e" in r or "n" in r) else r + ".0"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def optimizer_fields(opt, n: int) -> dict:
    """The fields of the levanter OptimizerConfig the tables depend on. marincpp's schedule supports one cycle, no
    cooldown and no rewarmup; anything else is refused here rather than silently tabulated."""
    for name, ok in (("cooldown", opt.cooldown is None), ("cycles", opt.cycles is None),
                     ("cycle_length", opt.cycle_length is None), ("haps", getattr(opt, "haps", None) is None),
                     ("rewarmup", opt.rewarmup == 0)):
        if not ok:
            raise SystemExit(f"optimizer.{name} = {getattr(opt, name)!r}: marincpp schedules have one cycle, no cooldown")
    sched = opt.lr_schedule if isinstance(opt.lr_schedule, str) else type(opt.lr_schedule).__name__
    if sched not in ("linear", "cosine"):
        raise SystemExit(f"lr_schedule {sched!r}: marincpp has linear and cosine")
    if sched == "cosine" and not isinstance(opt.lr_schedule, str) and getattr(opt.lr_schedule, "exponent", 1.0) != 1.0:
        raise SystemExit("cosine exponent != 1")
    return dict(
        num_train_steps=int(n),
        lr_schedule=sched,
        learning_rate=float(opt.learning_rate),
        adam_lr=float(opt.adam_lr),
        warmup=opt.warmup,                    # int steps or float fraction, as levanter reads it
        decay=-1 if opt.decay is None else opt.decay,   # marincpp optim.decay: -1 = None (N - warmup)
        min_lr_ratio=float(opt.min_lr_ratio),
        beta1=float(opt.beta1),
        beta2=float(opt.beta2),
    )


def make_tables(opt, n: int):
    import jax
    import jax.numpy as jnp
    import numpy as np

    sched_main = opt.lr_scheduler(n)
    sched_adam = opt.lr_scheduler(n, override_lr=opt.adam_lr)

    @functools.partial(jax.jit, inline=True)
    def bias_correction(decay, count):     # optax/tree_utils/_tree_math.py:400-411, the denominator only
        return 1 - decay ** count

    b1, b2 = float(opt.beta1), float(opt.beta2)

    @jax.jit
    def vec(k, t):
        return sched_main(k), sched_adam(k), bias_correction(b1, t), bias_correction(b2, t)

    @jax.jit
    def scalar(k, t):
        return jax.lax.map(lambda kt: (sched_main(kt[0]), sched_adam(kt[0]), bias_correction(b1, kt[1]),
                                       bias_correction(b2, kt[1])), (k, t))

    dev = jax.devices()[0]
    k = jax.device_put(np.arange(n, dtype=np.int32), dev)
    t = jax.device_put(np.arange(1, n + 1, dtype=np.int32), dev)
    a = [np.asarray(x) for x in vec(k, t)]
    b = [np.asarray(x) for x in scalar(k, t)]
    for name, x, y in zip(TABLES, a, b):
        if x.dtype != np.float32 or x.shape != (n,):
            raise SystemExit(f"{name}: dtype {x.dtype} shape {x.shape}, want float32 [{n}]")
        bad = np.nonzero(x.view(np.uint32) != y.view(np.uint32))[0]
        if bad.size:
            raise SystemExit(f"{name}: vectorized and lax.map evaluations differ at {bad.size} k (first {bad[:8].tolist()})")
    lowered = vec.lower(k, t)
    hlo = {"vec.stablehlo.txt": lowered.as_text(), "vec.opt_hlo.txt": lowered.compile().as_text()}
    return dict(zip(TABLES, a)), hlo


def write_fixture(group: str, name: str, label: str, fields: dict, tables: dict, hlo: dict, env: dict) -> str:
    out = common.fixture_dir(group, name)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    sums = {}
    for tab in TABLES:
        raw = tables[tab].astype("<f4").tobytes()
        with open(os.path.join(out, tab + ".f32"), "wb") as f:
            f.write(raw)
        sums[tab] = hashlib.sha256(raw).hexdigest()
    lines = [
        "# marincpp schedule fixture (tests/s2/schedule_fixture.py; DESIGN.md §4.4). Tables: little-endian float32.",
        "# sched_main/sched_adam index k = 0..N-1; bc1/bc2 index t = 1..N (1 - beta^t of update k = t - 1).",
        f"format = {FORMAT}",
        f"size = {toml_value(label)}",
        "",
        "[levanter]",
    ]
    lines += [f"{k} = {toml_value(v)}" for k, v in fields.items()]
    lines += ["", "[f32]"]
    lines += [f"{k} = {toml_value(f32_hex(fields[k]))}" for k in ("learning_rate", "adam_lr", "beta1", "beta2")]
    lines += ["", "[files]"]
    lines += [f"{tab} = {toml_value(tab + '.f32')}" for tab in TABLES]
    lines += ["", "[sha256]"]
    lines += [f"{tab} = {toml_value(sums[tab])}" for tab in TABLES]
    with open(os.path.join(out, "schedule.toml"), "w") as f:
        f.write("\n".join(lines) + "\n")

    man_dir = common.fixture_dir(group, name + "_manifest")
    for fname, text in hlo.items():
        with open(os.path.join(man_dir, fname), "w") as f:
            f.write(text)
    common.write_manifest(man_dir, checks=["T0.7"], env=env,
                          extra=dict(fixture_dir=os.path.relpath(os.path.realpath(out), os.path.realpath(REPO)), tables_sha256=sums, levanter=fields))
    return out


# S2 probes: synthetic levanter optimizer configs that reach the schedule paths the baseline does not (min_lr_ratio > 0,
# warmup before a linear decay, a stable phase before a cosine, decay given as a fraction or in steps, decay 0, warmup
# over all N, other T_d for the folded cos constant, other betas for pow). The C++ formula must equal each bitwise
# (tests/s2 suite s2_probe, which lists the same names). Fields: N, lr_schedule, warmup, decay (None = N - warmup),
# min_lr_ratio, lr, adam_lr, b1, b2.
# cos_fold_1000 and cos_fold_4000 have a T_d where the two candidate folded cosine constants differ,
# f32(f32(pi) * f32(1/T_d)) (XLA's, the port's) and f32(f32(pi) / f32(T_d)); at every other probe T_d and at the three
# cosine baseline T_d they are equal, so only these two tell them apart. decay_huge gives the decay as whole steps past
# 2^63 (levanter clips it to N - warmup); beta0 has beta1 = beta2 = 0 (1 - 0^t = 1).
PROBES = {
    "cos_min":      (3000, "cosine", 300, None, 0.1, 0.02, 0.004, 0.9, 0.95),
    "lin_warm_mid": (2500, "linear", 200, 0.5, 0.05, 0.003, 0.0007, 0.8, 0.999),
    "cos_stable":   (5000, "cosine", 0, 0.3, 0.0, 0.05, 0.01, 0.95, 0.99),
    "lin_full":     (777, "linear", 0, None, 0.0, 0.0123, 0.00456, 0.9, 0.98),
    "cos_steps":    (4000, "cosine", 500, 1500, 0.0, 0.007, 0.0011, 0.85, 0.97),
    "cos_odd_t":    (12345, "cosine", 1234, None, 0.0, 0.01, 0.002, 0.9, 0.98),
    "lin_min":      (6007, "linear", 0, 0.8, 0.2, 0.02, 0.008, 0.9, 0.98),
    "warm_all":     (1500, "cosine", 1500, None, 0.0, 0.01, 0.002, 0.9, 0.98),
    "decay0":       (2000, "cosine", 100, 0, 0.0, 0.01, 0.002, 0.9, 0.98),
    "cos_long":     (30011, "cosine", 2000, None, 0.0, 0.003, 0.0004, 0.9, 0.98),
    "cos_fold_1000": (3000, "cosine", 500, 1000, 0.0, 0.01, 0.002, 0.9, 0.98),
    "cos_fold_4000": (4500, "cosine", 500, None, 0.1, 0.006, 0.003, 0.9, 0.98),
    "decay_huge":   (2000, "cosine", 200, 1e19, 0.0, 0.01, 0.002, 0.9, 0.98),
    "beta0":        (1200, "linear", 100, 0.5, 0.0, 0.01, 0.002, 0.0, 0.0),
}


def probe_optimizer(spec):
    from levanter.optim.muonh import MuonHConfig

    n, sched, warmup, decay, ratio, lr, alr, b1, b2 = spec
    return n, MuonHConfig(learning_rate=lr, adam_lr=alr, lr_schedule=sched, warmup=warmup, decay=decay,
                          min_lr_ratio=ratio, beta1=b1, beta2=b2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", action="append", choices=lcfg.SIZES, help="repeatable; default: all four")
    ap.add_argument("--probes", action="store_true", help="write the S2 probe fixtures (cache/levref/s2_probe/<name>/)")
    args = ap.parse_args()
    sizes = args.size or (["130m"] if args.probes else list(lcfg.SIZES))   # probes: 130m only for the run env

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    shell = {k: os.environ.get(k) for k in lcfg.APPLY_RUN_ENV}   # read before build(), which edits os.environ
    built = {}
    for s in sizes:                       # build every config before the JAX backend starts (RUN_ENV, XLA_FLAGS)
        cfg, _pod = lcfg.build(s)
        built[s] = cfg
    flags = {s: {k: lcfg.RUN_ENV[s][k] for k in lcfg.APPLY_RUN_ENV if k in lcfg.RUN_ENV[s]} for s in sizes}
    if len({tuple(sorted(v.items())) for v in flags.values()}) != 1:
        raise SystemExit(f"run environments differ between sizes; run one size per process: {flags}")
    run_env = flags[sizes[0]]
    for k in lcfg.APPLY_RUN_ENV:            # the shell must not change what XLA compiles
        have, want = shell[k], run_env.get(k)
        if have is not None and have != want:
            raise SystemExit(f"{k}={have!r} is set in this shell, but the run used "
                             f"{'no ' + k if want is None else repr(want)}; unset it")
        if want is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = want

    import jax

    if jax.default_backend() != "gpu":
        raise SystemExit(f"backend {jax.default_backend()}: the fixture must come from XLA:GPU (DESIGN.md §4.4)")
    env = common.environment()
    env["compute_capability"] = getattr(jax.devices()[0], "compute_capability", None)
    jobs = []
    if args.probes:
        for name, spec in PROBES.items():
            n, opt = probe_optimizer(spec)
            jobs.append(("s2_probe", name, "probe:" + name, n, opt))
    else:
        for s in sizes:
            jobs.append((s, "schedules", s, built[s].trainer.num_train_steps, built[s].optimizer))
    for group, name, label, n, opt in jobs:
        fields = optimizer_fields(opt, n)
        tables, hlo = make_tables(opt, n)
        out = os.path.realpath(write_fixture(group, name, label, fields, tables, hlo, env))
        print(f"{label}: N={n} -> {os.path.relpath(out, os.path.realpath(REPO))}  eta(0)={tables['sched_main'][0]:.9g} "
              f"eta(N-1)={tables['sched_main'][-1]:.9g} eta_a(N-1)={tables['sched_adam'][-1]:.9g}", flush=True)


if __name__ == "__main__":
    main()
