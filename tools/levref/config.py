"""The plain-baseline TrainLmConfig per size, built by the launcher itself, and a diff against the logged W&B config.

build(size) imports experiments/references/della_muonh_qwen3_scaling.py (read-only, every launcher env knob removed so
VARIANT=baseline, SEED=0, TIE=0, SEQ_LEN=4096, NUM_GPUS=4, no SMOKE_STEPS), resolves muonh_qwen3_run(size) the way the
marin runner does (materialized_config against MARIN_PREFIX; read-only), and applies the same run-time changes as
marin.training.run_levanter_train_lm (_prepare_training_run: output path, run id). num_train_steps is the full N.

for_harness(cfg, scratch) is the copy the fixture modules run: same math, but no tracker, checkpoints under `scratch`
(never the run directory) and no checkpoint loading.

Self-test (CPU, 4 forced host devices so per_device_parallelism resolves as on 4 GPUs):
    python -m tools.levref.config --check [130m 300m 520m 1_2b]
It prints every leaf that differs from cache/wandb/muonh-qwen3-<size>-della4xh100.config.json, marks the known
run-environment differences as EXPECTED, writes cache/levref/<size>/config/{config.json,diff.json,manifest.json}, and
exits 1 on any UNEXPECTED difference.
"""
import os
import sys

if __name__ == "__main__" and "--check" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

import dataclasses
import enum
import importlib
import json
import re
from datetime import timedelta
from pathlib import Path

from tools.levref import common

SIZES = ("130m", "300m", "520m", "1_2b")
LAUNCHER = "experiments.references.della_muonh_qwen3_scaling"
LAUNCHER_FILE = os.path.join("experiments", "references", "della_muonh_qwen3_scaling.py")
NUM_GPUS = 4
_KEEP_ENV = {"MARIN_PREFIX"}
# Set by the run's sbatch (marin scripts/della/muonh_qwen3_h100x4.sbatch:28; kaiyue_baseline.md, Other environment),
# which overrides any inherited value. harness() does the same after the launcher scrub: the launcher reads
# XLA_PYTHON_CLIENT_MEM_FRACTION (1_2b OV-variant branch only), so the scrub removes an inherited value, e.g. the .50
# that login shells on this cluster export, and harness() then sets 0.9 before the JAX backend starts.
RUN_SBATCH_ENV = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}
_launcher = None
_scrubbed: dict[str, str] = {}
RUN_ENV: dict[str, dict[str, str]] = {}  # size -> the env run_levanter_train_lm would merge into os.environ (build fills it)
# Keys of that env the harness applies (setdefault, before the JAX backend starts): XLA flags and NCCL logging. It never
# applies JAX_COMPILATION_CACHE_DIR, which points into the marin store, or any other path-valued key.
APPLY_RUN_ENV = ("XLA_FLAGS", "NCCL_DEBUG")
POLICIES = ("bf16", "f32")


def _scrub_launcher_env() -> dict[str, str]:
    """Remove every environment variable the launcher reads (except MARIN_PREFIX), so it builds the plain baseline."""
    src = open(os.path.join(common.marin_dir(), LAUNCHER_FILE)).read()
    names = set(re.findall(r'os\.environ(?:\.get)?\(\s*"([A-Za-z0-9_]+)"', src)) | set(re.findall(r'os\.environ\["([A-Za-z0-9_]+)"\]', src))
    removed = {n: os.environ.pop(n) for n in sorted(names - _KEEP_ENV) if n in os.environ}
    return removed


def launcher():
    global _launcher
    if _launcher is None:
        _scrubbed.update(_scrub_launcher_env())
        os.environ.setdefault("MARIN_PREFIX", common.marin_prefix())
        _launcher = importlib.import_module(LAUNCHER)
        assert os.path.realpath(_launcher.__file__) == os.path.realpath(os.path.join(common.marin_dir(), LAUNCHER_FILE)), _launcher.__file__
        assert _launcher.VARIANT == "baseline" and _launcher.SMOKE_STEPS == 0 and _launcher.SEED == 0 and not _launcher.TIE
        assert _launcher.SEQ_LEN == 4096 and _launcher.NUM_GPUS == NUM_GPUS and _launcher.TPP == 20 and _launcher.PDP is None
    return _launcher


def build(size: str):
    """(TrainLmConfig, TrainLmOnPodConfig) of the plain baseline at `size`, as run_levanter_train_lm receives them."""
    from marin.execution.lazy import materialized_config
    from marin.training import training as _training

    if size not in SIZES:
        raise SystemExit(f"size {size!r}: one of {SIZES}")
    L = launcher()
    step = L.muonh_qwen3_run(size)
    pod = materialized_config(step, os.environ["MARIN_PREFIX"])
    pod, train_config, env = _training._prepare_training_run(pod)
    RUN_ENV[size] = dict(env)
    t = train_config.trainer
    assert t.num_train_steps == L.SIZES[size]["steps"], (t.num_train_steps, L.SIZES[size]["steps"])
    assert t.max_eval_batches is None and t.seed == 0 and train_config.data_seed == 42
    return train_config, pod


def for_harness(cfg, scratch: str):
    """The config the fixture modules run: identical math, no tracker, checkpoints only under `scratch`, nothing loaded."""
    from levanter.tracker.tracker import NoopConfig

    ck = dataclasses.replace(cfg.trainer.checkpointer, base_path=os.path.join(scratch, "checkpoints"),
                             temporary_base_path=os.path.join(scratch, "checkpoints-temp"))
    trainer = dataclasses.replace(cfg.trainer, tracker=NoopConfig(), checkpointer=ck, load_checkpoint=False,
                                  log_dir=Path(scratch) / "logs", initialize_from=None)
    return dataclasses.replace(cfg, trainer=trainer, hf_save_path=None, hf_save_steps=None)


def open_trainer(cfg):
    """levanter.trainer.initialize, the optimizer and the Trainer exactly as levanter/main/train_lm.py:197-211; use as `with`."""
    import levanter
    from levanter.trainer import Trainer

    levanter.trainer.initialize(cfg)
    optimizer = cfg.optimizer.build(cfg.trainer.num_train_steps)

    def loss_function(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=cfg.z_loss_weight)

    return Trainer(cfg.trainer, optimizer, loss_function)


def with_policy(cfg, policy: str):
    """bf16: the logged policy p=f32,c=bfloat16 (unchanged). f32: the same config with compute_dtype=f32 (targets.md §4)."""
    import jax.numpy as jnp
    import jmp

    mp = cfg.trainer.mp
    assert (mp.param_dtype, mp.compute_dtype) == (jnp.float32, jnp.bfloat16), mp
    if policy == "bf16":
        return cfg
    if policy != "f32":
        raise SystemExit(f"policy {policy!r}: one of {POLICIES}")
    return dataclasses.replace(cfg, trainer=dataclasses.replace(cfg.trainer, mp=jmp.get_policy("p=f32,c=f32")))


def harness(size: str, tag: str, policy: str = "bf16"):
    """(cfg0, cfg, scratch, applied_env): the built baseline, the harness copy under `policy`, its scratch dir, and the
    run-env keys applied to os.environ. Call before the JAX backend starts."""
    scratch = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"levref-{tag}-{size}-{os.getpid()}")
    cfg0, _pod = build(size)
    applied = {}
    for k in APPLY_RUN_ENV:
        if k in RUN_ENV[size] and k not in os.environ:
            os.environ[k] = applied[k] = RUN_ENV[size][k]
    for k, v in RUN_SBATCH_ENV.items():
        os.environ[k] = applied[k] = v
        if _scrubbed.get(k, v) != v:
            applied[f"{k}_inherited_and_replaced"] = _scrubbed[k]
    try:  # every env setting above must reach the JAX backend, which reads it once, when it starts
        from jax._src import xla_bridge

        assert not xla_bridge._backends, "JAX backend started before config.harness() set its environment"
    except (ImportError, AttributeError):
        pass
    cfg = with_policy(for_harness(cfg0, scratch), policy)
    if os.environ.get("JAX_PLATFORMS") == "cpu":  # CPU debugging only: TrainerConfig.initialize would refuse a CPU backend
        cfg = dataclasses.replace(cfg, trainer=dataclasses.replace(cfg.trainer, require_accelerator=False))
    return cfg0, cfg, scratch, applied


# ------------------------------------------------------------------------------------------------ diff vs W&B
def _plain(x):
    """dataclasses.asdict output to the JSON shapes W&B stores."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, enum.Enum):
        return x.name if isinstance(x.value, int) or not isinstance(x.value, str) else x.value
    if isinstance(x, type):
        return f"{x.__module__}.{x.__qualname__}"
    if isinstance(x, timedelta):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def _walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{path}.{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<absent>"))
            else:
                _walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
    else:
        same = a == b
        if not same and isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            same = float(a) == float(b)
        if not same and isinstance(a, str) and isinstance(b, str):
            same = a.lower() == b.lower()  # enum name vs value spelling
        if not same:
            out.append((path, a, b))


# Leaves that differ for a reason outside the training math. Each regex maps to its reason.
EXPECTED = {
    r"^git_commit$": "logged by the tracker, not a config field; the run commit differs from the marin HEAD only in trainer.py (model.md §0), no math",
    r"^trainer\.tracker\.entity$": "W&B entity came from the run environment (WANDB_ENTITY), unset here",
    r"^trainer\.tracker\.tags": "the launcher at HEAD adds the PRECISION tag (bf16) after the run commit; tags carry no math",
}


def _classify(path, built, logged):
    for pat, why in EXPECTED.items():
        if re.search(pat, path):
            return why
    if built == {} and logged == "<absent>":
        return "W&B drops empty dicts from a run config"
    return None


def check(size: str) -> int:
    import jax

    cfg, pod = build(size)
    t = cfg.trainer
    t._validate_and_set_defaults()  # what Trainer.initialize resolves at run time (per_device_parallelism on 4 devices)
    if t.require_accelerator is None:  # TrainerConfig.initialize resolves it the same way
        t.require_accelerator = not sys.platform.startswith("darwin")
    built = _plain(dataclasses.asdict(cfg))
    logged_rec = json.load(open(os.path.join(common.nano_dir(), "cache", "wandb", f"muonh-qwen3-{size}-della4xh100.config.json")))
    logged = logged_rec["config"]
    diffs = []
    _walk(built, logged, "", diffs)
    rows = []
    bad = 0
    for path, a, b in diffs:
        path = path.lstrip(".")
        why = _classify(path, a, b)
        bad += why is None
        rows.append(dict(path=path, built=a, logged=b, verdict="EXPECTED" if why else "UNEXPECTED", reason=why))
    print(f"== {size}: {len(diffs)} differing leaves, {bad} unexpected  (devices: {jax.device_count()} {jax.default_backend()})")
    for r in rows:
        print(f"  [{r['verdict']}] {r['path']}: built={json.dumps(r['built'], default=str)[:200]} logged={json.dumps(r['logged'], default=str)[:200]}"
              + (f"  ({r['reason']})" if r["reason"] else ""))
    out = common.fixture_dir(size, "config")
    common.write_json(os.path.join(out, "config.json"), built)
    common.write_json(os.path.join(out, "diff.json"), dict(size=size, logged_run=logged_rec["path"], logged_git_commit=logged.get("git_commit"),
                                                            n_diff=len(diffs), n_unexpected=bad, launcher_env_removed=sorted(_scrubbed), rows=rows))
    common.write_manifest(out, checks=["T0.6", "T0.7", "T0.8", "config parity with W&B (Levanter fixtures convention)"], env=common.environment(),
                          extra=dict(size=size, output_path=pod.output_path, run_id=t.id, num_train_steps=t.num_train_steps))
    return bad


def main():
    if "--check" not in sys.argv:
        print(__doc__)
        return
    sizes = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SIZES)
    bad = sum(check(s) for s in sizes)
    print("OK: no unexpected differences" if bad == 0 else f"FAIL: {bad} unexpected differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
