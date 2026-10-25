"""T1.7 and T2.1-T2.2, T2.4-T2.7 fixtures: step-0 loss and gradients, the three pre-clip group norms, optimizer
updates 0 and 1, and the step-1 / step-2 losses, under one mixed-precision policy.

    python -m tools.levref.step --size 130m --policy bf16              # -> cache/levref/<size>/step0_bf16/
    python -m tools.levref.step --size 130m --policy f32               # -> step0_f32/ (also T1.7 cases A, B and, at 130m, C)
    python -m tools.levref.step --size 130m --policy f32 --repeat-only # -> rep_f32/ (a second process, for e_floor)
    python -m tools.levref.floor 130m                                  # -> floor/e_floor.json (CPU)

Everything runs inside the baseline Trainer (tools.levref.config.open_trainer) from the seed-0 initial state
(tools.levref.init.build_initial_state, train_lm.py:268) on batches 0-2 of the train loader (tools.levref.batch).

Reference step (`ref_step`, one named_jit with the train step's axis resources, no donation): a line-by-line copy of
Trainer._train_step_body (trainer.py:750-781) that also returns the gradient, the update and the three pre-clip group
norms. The gradient is trainer._compute_gradients_microbatched(trainer.loss_fn, ...) and the update is
state.take_step(...) (MuonHConfig.build(N): clip per group -> MuonH / AdamH / Adam -> p + u). The group norms are
optax's own computation: labels = MuonHConfig.create_mask(grads), the other groups' leaves replaced by MaskedNode
(optax/transforms/_combining.py:278, _masking.py:90-92), then optax linear_algebra.global_norm (_clipping.py:92).
`ref_update` is the same take_step on a given gradient (T1.7 cases B and C).

The losses are also taken from levanter's own program: trainer.train_step on batches 0, 1, 2 from a copy of the initial
state (step 0 runs the jit-hook program, steps 1-2 the no-hook program, as in the real run). Its parameters after
updates 0 and 1 are compared with ref_step's, tensor by tensor.

Matmul precision (docs/DESIGN.md §3.0, par-2). bf16 policy: JAX's DEFAULT precision everywhere, as the run. f32 policy:
every call of the main chain (ref_step: forward, backward and optimizer; ref_update for cases B and C;
trainer.train_step) runs under jax.default_matmul_precision("highest"), so every f32 dot is a true f32 GEMM (DEFAULT
runs f32 dots as TF32 on A100 and H100). JAX reads the precision at trace time and keys the jit cache on it, so each
call compiles its own program; checks.json `precision` counts the stablehlo.dot_general ops of the lowered programs by
precision and operand types, and the f32 policy asserts that every dot of ref_step is HIGHEST. The run itself ran its
optimizer (the MuonH Newton-Schulz) at DEFAULT, so for T1.7 the f32 policy also writes caseA_default/, caseB_default/
and caseC_default/: ref_update at DEFAULT on the same inputs (T1.7 oracle rule: levanter at DEFAULT vs the HIGHEST
oracle when T1.0 shows TF32 dots; rel <= 1e-5 against HIGHEST when it shows f32 dots).

Cross entropy: checks.json `ce` records the fused-CE implementation levanter selected (api._SELECTED_IMPL_LOGGED), its
fallback warnings, and every H100 B-tiled decision (batched_xla._h100_full_vocab_b_tiled_block_size, wrapped in this
process by a pass-through recorder that returns the original value). On an H100 the bf16 policy asserts the run's path:
batched_xla with 8192-row B tiles.

Files in cache/levref/<size>/step0_<policy>/ (npy = float32 arrays in levanter's stored axis order, file per tensor;
every directory has index.json: name -> axes, shape, dtype, sha256 of the bytes, f64 L2 norm, per-layer L2 norms of
stacked tensors, scalar values with their f32 bit pattern):
  losses.json        L0 (each in-process repeat), L1, L2 from ref_step and from trainer.train_step, f32 hex
  group_norms.json   per gradient (g0 each repeat, g1, g2; caseB/caseC on G0, and the *_default copies): muonh/adamh/adam
                     pre-clip norms, the total norm, max_grad_norm, whether the clip triggers, the clip factor
                     min(1, max/gn) in f64
  p0/                initial parameters of this process (the init on this mesh), npy               T2.x start, T1.7 A
  g0/                step-0 gradient (repeat 0), npy                                      T2.1/T2.2 inputs, T2.4, T2.5
  u0/  s1/           update 0 (= target - p for MuonH/AdamH) and the optimizer state after it, npy     T2.7, T1.7 case A
  p1/                parameters after update 0: index; npy only for tensors where the host f32 sum p0 + u0 differs
                     from levanter's p1 (XLA may fuse the add); every other tensor is p0 + u0 bitwise (checks.json p1)
  g1/  s2/           index only;  u1/  update 1 (on g1, batch 1), npy;  p2/  as p1/ (p1 + u1)          T2.7
  train_step_p1/ train_step_p2/   parameters after updates 0 and 1 from trainer.train_step: index; npy only for tensors
                     whose bytes differ from ref_step's p1 / p2
  caseB/{u,s,p}      f32 only: update on G0 again at k = 1 from the case-A state                        T1.7 case B
  caseC/{u,s,p,pin}  f32, 130m only: update on G0 at k = 4958 from the step-4958 checkpoint             T1.7 case C
                     (the checkpoint stores step 4959 and counts 4959; every count, incl. inject_hyperparams'
                     schedule counters, is set to 4958, so the update uses lr(4958) and t = 4959)
  caseA_default/ caseB_default/ caseC_default/  {u,s,p}: f32 only, ref_update at DEFAULT precision on the inputs of
                     cases A (p0, zero state, G0), B, C. u/ and s/ hold npy only for tensors whose bytes differ from the
                     HIGHEST result (u0/ s1/, caseB/, caseC/; index: same_as_base, rel_to_base); p/ as p1/
  rep_inproc.json    repeat 1 vs repeat 0 of ref_step on the same inputs: per-tensor rel and sha equality of g0, u0, s1
  checks.json        batch hashes vs data_ref (T0.2), init digest, p0 + u0 == p1, train_step vs ref_step parameters,
                     norm preservation of MuonH layers and lm_head (T1.7), case C inputs, precision, ce, default copies
  manifest.json      also: matmul_precision, ce_implementation, device_memory (bytes_limit, peak), the env it ran with
--repeat-only writes rep_<policy>/ with losses.json, group_norms.json, g0/, u0/, s1/, checks.json (no further steps).
"""
import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import time

from tools.levref import batch as lbatch, common, config as lconfig, init as linit

GROUPS = ("muonh", "adamh", "adam")
CKPT_130M = os.path.join("speedrun", "muonh-qwen3-130m-della4xh100-restore", "2026.09.13", "checkpoints", "step-4958")
CASE_C_K = 4958
# Matmul precision of each policy's main chain (DESIGN.md §3.0): None = JAX DEFAULT (the run).
POLICY_PRECISION = {"bf16": None, "f32": "highest"}
H100_B_TILE = 8192  # batched_xla._H100_FULL_VOCAB_B_TILED_BLOCK_SIZE: the CE path of the run (spec T1.5)

_t0 = time.time()


def log(*a):
    print(f"[levref.step {time.time() - _t0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ host helpers
def f32hex(x) -> str:
    import numpy as np

    return "0x%08X" % int(np.asarray(x, dtype=np.float32).reshape(()).view(np.uint32))


def host(leaf):
    import jax
    import numpy as np

    return np.asarray(jax.device_get(leaf.array if hasattr(leaf, "array") else leaf))


def rel(a, b) -> float:
    """||a - b|| / ||b|| in float64 (0 when both are equal, inf when b = 0 and a != b)."""
    import numpy as np

    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    num = float(np.sqrt(((a64 - b64) ** 2).sum()))
    den = float(np.sqrt((b64 ** 2).sum()))
    if num == 0.0:
        return 0.0
    return num / den if den > 0 else math.inf


def max_ulp(a, b) -> int:
    from tools.levref.compare import ulp_distance

    return int(ulp_distance(a, b).max()) if a.size else 0


def opt_name(path: str) -> str:
    """inner_state.inner_states.muonh.inner_state.1.momentum_buffer.<param> -> muonh.momentum_buffer.<param>."""
    return ".".join(t for t in path.split(".") if t not in ("inner_state", "inner_states") and not t.isdigit())


def named(tree, kind: str):
    """[(name, leaf)] of a model-shaped tree (kind 'model') or of an optimizer state (kind 'opt', names via opt_name)."""
    if kind == "model":
        return linit.named_leaves(tree)
    out = [(opt_name(n), l) for n, l in linit.named_leaves(tree, arrays=True)]
    names = [n for n, _ in out]
    assert len(set(names)) == len(names), "optimizer-state names collide"
    return out


def save_tree(out: str, sub: str, items, full) -> dict:
    """Write <out>/<sub>/{<name>.npy (if full: True, or a set of names), index.json}; returns the index."""
    import numpy as np

    d = os.path.join(out, sub)
    os.makedirs(d, exist_ok=True)
    idx = {}
    for name, leaf in items:
        save = full if isinstance(full, bool) else name in full
        w = host(leaf)
        x = w.astype(np.float64) if w.dtype.kind == "f" else w
        e = dict(axes=[a.name for a in leaf.axes] if hasattr(leaf, "axes") else None, shape=list(w.shape), dtype=str(w.dtype),
                 sha256=common.sha256_array(w))
        if w.size == 1:
            e["value"] = w.reshape(()).item()
            if w.dtype == np.float32:
                e["f32_hex"] = f32hex(w)
        else:
            e["l2"] = float(np.sqrt((x * x).sum())) if w.dtype.kind == "f" else None
            if e["axes"] and e["axes"][0] == "layer":
                per = x.reshape(w.shape[0], -1)
                e["per_layer_l2"] = [float(v) for v in np.sqrt((per * per).sum(1))]
            if save:
                np.save(os.path.join(d, f"{name}.npy"), w)
        e["npy"] = bool(save and w.size > 1)
        idx[name] = e
    common.write_json(os.path.join(d, "index.json"), idx)
    log(f"wrote {sub}/ ({len(idx)} leaves, npy={full if isinstance(full, bool) else len(full)})")
    return idx


def save_params_after(out: str, sub: str, before, upd, after) -> tuple[dict, dict]:
    """Index of the parameters after an update. A tensor gets an npy only when the host f32 sum before + update differs
    from levanter's result (XLA may fuse p + u, e.g. into an FMA with the Adam scale), so p_after = p_before + u
    reconstructs every other tensor bitwise. Returns (index, per-tensor check)."""
    import numpy as np

    chk, need = {}, set()
    for (n, p), (m, uu), (k, q) in zip(before, upd, after):
        assert n == m == k, (n, m, k)
        hp, hu, hq = host(p), host(uu), host(q)
        eq = bool(np.array_equal(hp + hu, hq))
        chk[n] = dict(before_plus_update_equals_after=eq, frac_after_equal_before=float(np.mean(hq == hp)),
                      max_ulp_after_vs_before=max_ulp(hq, hp))
        if not eq:
            d = np.abs((hp + hu).astype(np.float64) - hq.astype(np.float64))
            chk[n].update(n_mismatch=int((hp + hu != hq).sum()), max_abs_mismatch=float(d.max()))
            need.add(n)
    return save_tree(out, sub, after, full=need), chk


def norms_record(norms: dict, max_grad_norm: float) -> dict:
    out = {}
    for g in (*GROUPS, "total"):
        gn = host(norms[g]).astype("float32").reshape(())
        r = dict(norm=float(gn), f32_hex=f32hex(gn))
        if g != "total":
            trig = bool(gn < max_grad_norm)  # optax: trigger = g_norm < max_norm; equality also clips
            r.update(clip_triggered=not trig, clip_factor=1.0 if trig else float(max_grad_norm) / float(gn))
        out[g] = r
    out["max_grad_norm"] = max_grad_norm
    return out


def compare_trees(a_items, b_items) -> dict:
    out = {}
    for (n, a), (m, b) in zip(a_items, b_items):
        assert n == m, (n, m)
        xa, xb = host(a), host(b)
        out[n] = dict(sha_equal=common.sha256_array(xa) == common.sha256_array(xb),
                      rel=rel(xa, xb) if xa.dtype.kind == "f" else (0.0 if (xa == xb).all() else math.inf),
                      max_ulp=max_ulp(xa, xb) if xa.dtype == "float32" else None)
    return out


def cmp_summary(c: dict) -> dict:
    worst = max(c.items(), key=lambda kv: kv[1]["rel"])
    return dict(all_bitwise=all(v["sha_equal"] for v in c.values()), n_bitwise=sum(v["sha_equal"] for v in c.values()),
                n=len(c), max_rel=worst[1]["rel"], worst=worst[0])


def save_variant(out: str, sub: str, items, base_items, base_sub: str) -> dict:
    """<sub>/ for a variant of the tree in <base_sub>/ (same names): index of every tensor, npy only for tensors whose
    bytes differ from the base; each index entry says same_as_base, rel_to_base, max_ulp_to_base. Returns the comparison."""
    c = compare_trees(items, base_items)
    idx = save_tree(out, sub, items, full={n for n, v in c.items() if not v["sha_equal"]})
    for n, v in c.items():
        idx[n].update(base=base_sub, same_as_base=v["sha_equal"], rel_to_base=v["rel"], max_ulp_to_base=v["max_ulp"])
    common.write_json(os.path.join(out, sub, "index.json"), idx)
    return c


def norm_preservation(p_in: dict, p_out: dict, labels: dict) -> dict:
    """Per-layer (MuonH, stacked) and whole-matrix (lm_head) L2 norm change |n_out / n_in - 1| from two indexes."""
    out = {}
    for n, lab in labels.items():
        if lab not in ("muonh", "adamh"):
            continue
        a, b = p_in[n], p_out[n]
        if "per_layer_l2" in a:
            d = [abs(y / x - 1) for x, y in zip(a["per_layer_l2"], b["per_layer_l2"])]
        else:
            d = [abs(b["l2"] / a["l2"] - 1)]
        out[n] = dict(group=lab, max_rel_change=max(d), per_layer=d)
    return out


# ------------------------------------------------------------------------------------------------ precision, CE, memory
def mm_precision(prec):
    """jax.default_matmul_precision(prec), or no context for None (DEFAULT). Read at trace time; part of the jit key."""
    import jax

    return jax.default_matmul_precision(prec) if prec else contextlib.nullcontext()


def dot_audit(lowered) -> dict:
    """stablehlo.dot_general ops of a lowered program, counted by '<precision>:<lhs type>x<rhs type>'."""
    cnt: dict[str, int] = {}
    for line in lowered.as_text().splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        m = re.search(r"precision = \[(\w+), (\w+)\]", line)
        prec = (m.group(1) if m.group(1) == m.group(2) else f"{m.group(1)},{m.group(2)}") if m else "none"
        t = re.search(r":\s*\(tensor<(?:[^>]*x)?(\w+)>, tensor<(?:[^>]*x)?(\w+)>\)", line)
        key = f"{prec}:{t.group(1)}x{t.group(2)}" if t else f"{prec}:?"
        cnt[key] = cnt.get(key, 0) + 1
    return dict(n_dot_general=sum(cnt.values()), by_precision_and_types=dict(sorted(cnt.items())))


class CeRecorder:
    """Which fused-CE implementation levanter selects in this process, its fallback warnings, and every H100 B-tiled
    decision: batched_xla._h100_full_vocab_b_tiled_block_size is wrapped by a pass-through that records its inputs and
    returns the original value (install before the first trace of the loss)."""

    def __init__(self):
        from levanter.kernels.pallas.fused_cross_entropy_loss import api, batched_xla

        self.api, self.calls = api, []
        orig = batched_xla._h100_full_vocab_b_tiled_block_size

        def recorded(x, w, *, return_argmax=False):
            r = orig(x, w, return_argmax=return_argmax)
            self.calls.append(dict(x=f"{tuple(x.shape)} {x.dtype}", w=f"{tuple(w.shape)} {w.dtype}",
                                   return_argmax=return_argmax, b_tile=r))
            return r

        batched_xla._h100_full_vocab_b_tiled_block_size = recorded

    def record(self) -> dict:
        uniq = sorted({json.dumps(c, sort_keys=True) for c in self.calls})
        return dict(selected=sorted(self.api._SELECTED_IMPL_LOGGED),
                    fallback_warnings=sorted(self.api._IMPLEMENTATION_FALLBACK_WARNINGS_EMITTED),
                    b_tiled_decisions=[json.loads(u) for u in uniq],
                    b_tiled_used={c["b_tile"] for c in self.calls} == {H100_B_TILE},
                    autotune_on_miss=os.environ.get("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS"))


def device_memory() -> list:
    import jax

    out = []
    for d in jax.local_devices():
        try:
            st = d.memory_stats() or {}
        except Exception as e:  # CPU devices have none
            st = dict(error=repr(e))
        out.append(dict(device=str(d), **{k: st.get(k) for k in ("bytes_limit", "peak_bytes_in_use", "bytes_in_use")}))
    return out


# ------------------------------------------------------------------------------------------------ jitted levanter calls
def make_fns(trainer, opt_cfg):
    """(ref_step, ref_update, group_labels) bound to this trainer; see the module docstring."""
    import equinox as eqx
    import haliax as hax
    import jax
    from haliax.partitioning import named_jit
    from haliax.quantization import partition_for_grad_overwrite
    from levanter.trainer_state import trainables_only
    from levanter.utils.tree_utils import inference_mode
    from optax._src import linear_algebra
    from optax.transforms._masking import MaskedNode

    pam = trainer.parameter_axis_mapping

    def _train_grads(state, grads):  # take_train_step's view of the gradient (trainer_state.py:236-238)
        g = trainables_only(grads, state.is_trainable)
        _, g = partition_for_grad_overwrite(g)
        return g

    def group_norms(state, grads):
        g = _train_grads(state, grads)
        labels = opt_cfg.create_mask(g)
        out = {}
        for group in GROUPS:
            mask = jax.tree.map(lambda x: x == group, labels)  # optax _combining.make_mask
            masked = jax.tree.map(lambda m, p: p if m else MaskedNode(), mask, g)  # optax _masking.mask_pytree
            out[group] = linear_algebra.global_norm(masked)
        out["total"] = linear_algebra.global_norm(g)
        return out

    def _step(state, batch, batch_kwargs):  # trainer.py:750-781, plus grads / updates / norms as outputs
        key, new_key = jax.random.split(state.training_key)
        model = inference_mode(state.model, False)
        loss, grads, _metrics = trainer._compute_gradients_microbatched(trainer.loss_fn, model, *batch, **batch_kwargs, key=key)

        def obj_fun(trainable_model):
            model = eqx.combine(trainable_model, state.model)
            with hax.axis_mapping(trainer.compute_axis_mapping):
                model = trainer.mp.cast_to_compute(model)
                loss_for_opt, _m = trainer._raw_loss_function(model, *batch, **batch_kwargs, key=key)
                return loss_for_opt.scalar()

        new_state, updates = state.take_step(grads, obj_fun=obj_fun, loss=loss, key=new_key)
        new_state = hax.shard(new_state, pam)
        return hax.shard_with_axis_mapping((loss, grads, updates, new_state, group_norms(state, grads)), pam)

    def _update(state, grads):
        _key, new_key = jax.random.split(state.training_key)
        new_state, updates = state.take_step(grads, obj_fun=None, loss=None, key=new_key)
        new_state = hax.shard(new_state, pam)
        return hax.shard_with_axis_mapping((updates, new_state, group_norms(state, grads)), pam)

    def labels_of(state):
        g = _train_grads(state, state.model)
        lab = opt_cfg.create_mask(g)
        flat = jax.tree.leaves(lab)
        names = [n for n, _ in linit.named_leaves(g)]
        assert len(flat) == len(names), (len(flat), len(names))
        return dict(zip(names, flat))

    ref_step = named_jit(_step, axis_resources=pam, out_axis_resources=pam)
    ref_update = named_jit(_update, axis_resources=pam, out_axis_resources=pam)
    return ref_step, ref_update, labels_of


def slice_rows(example, rows: int, trainer):
    """DEBUG ONLY (--debug-rows): the first `rows` sequences of a batch, resharded."""
    import haliax as hax

    name = trainer.config.batch_axis_name

    def f(x):
        if isinstance(x, hax.NamedArray) and name in x.axis_names:
            return x[{name: slice(0, rows)}]
        return x

    ex = hax.tree_util.tree_map(f, example, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    return hax.shard(ex, trainer.compute_axis_mapping)


def copy_tree(tree):
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, tree)


def set_counts(opt_state, k: int):
    """Every int32 `count` leaf of the optimizer state (inject_hyperparams, AdamH, Adam) set to k; returns (state, old)."""
    import jax
    import jax.numpy as jnp

    old = {}

    def f(path, x):
        name = opt_name(".".join(str(getattr(p, "name", getattr(p, "key", getattr(p, "idx", p)))) for p in path))
        if name.split(".")[-1] == "count":
            old[name] = int(jax.device_get(x))
            return jnp.full_like(x, k)
        return x

    new = jax.tree_util.tree_map_with_path(f, opt_state)
    return new, old


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=lconfig.SIZES)
    ap.add_argument("--policy", required=True, choices=lconfig.POLICIES)
    ap.add_argument("--name", default=None, help="fixture prefix (default step0, or rep with --repeat-only)")
    ap.add_argument("--repeat-only", action="store_true", help="only ref_step on batch 0 (a second process for e_floor)")
    ap.add_argument("--reps", type=int, default=2, help="in-process repeats of ref_step on batch 0")
    ap.add_argument("--case-c", choices=("auto", "on", "off"), default="auto", help="T1.7 case C (auto: 130m f32 full run)")
    ap.add_argument("--ckpt", default=CKPT_130M, help="case C checkpoint, relative to MARIN_PREFIX (read only)")
    ap.add_argument("--debug-rows", type=int, default=0, help="DEBUG: keep only the first R rows of each batch")
    args = ap.parse_args()
    prefix = args.name or ("rep" if args.repeat_only else "step0")
    name = f"{prefix}_{args.policy}" + (f"_debug{args.debug_rows}" if args.debug_rows else "")
    out = common.fixture_dir(args.size, name)
    case_c = args.case_c == "on" or (args.case_c == "auto" and args.size == "130m" and args.policy == "f32" and not args.repeat_only)
    prec = POLICY_PRECISION[args.policy]
    P = lambda: mm_precision(prec)  # the policy's precision context for every main-chain call

    cfg0, cfg, scratch, applied_env = lconfig.harness(args.size, f"step-{args.policy}", args.policy)
    import jax
    import numpy as np

    mgn = float(cfg.optimizer.max_grad_norm)
    checks: dict = dict(size=args.size, policy=args.policy, mp=str(cfg.trainer.mp), debug_rows=args.debug_rows or None)
    losses: dict = dict(policy=args.policy)
    gnorms: dict = dict(policy=args.policy)
    with lconfig.open_trainer(cfg) as trainer:
        kind = jax.devices()[0].device_kind
        log("mesh", dict(trainer.device_mesh.shape), jax.device_count(), kind, "mp", cfg.trainer.mp,
            "matmul precision", prec or "default", "XLA_PYTHON_CLIENT_MEM_FRACTION", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))
        ce_rec = CeRecorder()
        state0, Vocab, _mk, keys = linit.build_initial_state(cfg, trainer)
        combined, _per = linit.model_digests(state0.model)
        init_idx = os.path.join(common.nano_dir(), "cache", "levref", args.size, "init", "index.json")
        ref_combined = json.load(open(init_idx))["summary"]["combined_sha256"] if os.path.exists(init_idx) else None
        checks["init"] = dict(combined_sha256=combined, init_fixture_combined_sha256=ref_combined,
                              equal_to_init_fixture=None if ref_combined is None else combined == ref_combined)
        log("init digest", combined[:16], "init fixture", str(ref_combined)[:16])

        n_batches = 1 if args.repeat_only else 3
        it = lbatch.train_iterator(cfg, trainer, 0)
        batches = [next(it) for _ in range(n_batches)]
        lbatch.stop_iterator(it)
        ref = json.load(open(os.path.join(common.nano_dir(), "cache", "data", "data_ref.json")))["batch_hashes"]
        B = cfg.trainer.train_batch_size
        checks["batches"] = {}
        for k, b in enumerate(batches):
            h = lbatch.tokens_sha256(b)
            want = ref.get(f"B{B}_step{k}", {}).get("sha256")
            checks["batches"][str(k)] = dict(sha256=h, ref_sha256=want, equal=None if want is None else h == want)
            assert want is None or h == want, f"batch {k} hash {h} != data_ref {want}"
        if args.debug_rows:
            batches = [slice_rows(b, args.debug_rows, trainer) for b in batches]
        log("batches", {k: v["equal"] for k, v in checks["batches"].items()})

        ref_step, ref_update, labels_of = make_fns(trainer, cfg.optimizer)
        labels = labels_of(state0)
        checks["groups"] = labels
        p0_items = named(state0.model, "model")
        p0_idx = None if args.repeat_only else save_tree(out, "p0", p0_items, full=True)

        # ---- step 0: ref_step twice on the same inputs (in-process floor)
        reps = []
        for r in range(max(1, args.reps if not args.repeat_only else 1)):
            with P():
                loss, g, u, s1, nrm = ref_step(state0, (batches[0],), {})
            loss = host(loss).astype(np.float32).reshape(())
            reps.append((loss, g, u, s1, nrm))
            log(f"ref_step batch 0 rep {r}: loss {float(loss):.7f} ({f32hex(loss)}) norms",
                {k: float(host(v)) for k, v in nrm.items()})

        # ---- what ran: dot precision of the lowered program, CE implementation (fail fast before the long part)
        with P():
            audit = dot_audit(ref_step.lower(state0, (batches[0],), {}))
        not_highest = {k: v for k, v in audit["by_precision_and_types"].items() if not k.startswith("HIGHEST:")}
        checks["precision"] = dict(main_chain=prec or "default", jax_config_default=str(jax.config.jax_default_matmul_precision),
                                   ref_step=audit, ref_step_all_highest=not not_highest)
        ce = ce_rec.record()
        b_tiles = {d["b_tile"] for d in ce["b_tiled_decisions"]}
        expect_b_tiled = args.policy == "bf16" and "h100" in kind.lower()
        ce.update(device_kind=kind, expected="batched_xla, B tile 8192 (the run's H100 path, spec T1.5)" if expect_b_tiled else None)
        checks["ce"] = ce
        log("dot_general by precision", audit["by_precision_and_types"], "CE", ce["selected"], "B tiles", sorted(b_tiles, key=str))
        if prec == "highest":
            assert audit["n_dot_general"] > 0 and not not_highest, f"f32 policy: dots not at HIGHEST: {not_highest}"
        if expect_b_tiled:
            assert ce["selected"] == ["batched_xla"] and b_tiles == {H100_B_TILE}, f"bf16 on H100: not the run's CE path: {ce}"

        loss0, g0, u0, state1, nrm0 = reps[0]
        losses["L0_ref"] = [dict(value=float(r[0]), f32_hex=f32hex(r[0])) for r in reps]
        gnorms["g0"] = [norms_record(r[4], mgn) for r in reps]
        g0_items, u0_items, s1_items = named(g0, "model"), named(u0, "model"), named(state1.opt_state, "opt")
        if len(reps) > 1:
            rep = dict(g0=compare_trees(g0_items, named(reps[1][1], "model")),
                       u0=compare_trees(u0_items, named(reps[1][2], "model")),
                       s1=compare_trees(s1_items, named(reps[1][3].opt_state, "opt")),
                       loss0_equal=f32hex(reps[0][0]) == f32hex(reps[1][0]),
                       loss0_abs_diff=abs(float(reps[0][0]) - float(reps[1][0])))
            common.write_json(os.path.join(out, "rep_inproc.json"), rep)
            log("in-process repeat: g0 bitwise", sum(v["sha_equal"] for v in rep["g0"].values()), "/", len(rep["g0"]),
                "max rel", max(v["rel"] for v in rep["g0"].values()))
        reps = reps[:1]
        save_tree(out, "g0", g0_items, full=True)
        save_tree(out, "u0", u0_items, full=True)
        save_tree(out, "s1", s1_items, full=True)

        if not args.repeat_only:
            p1_items = named(state1.model, "model")
            p1_idx, checks["p1"] = save_params_after(out, "p1", p0_items, u0_items, p1_items)
            checks["norm_preservation_update0"] = norm_preservation(p0_idx, p1_idx, labels)

            # ---- update 0 -> step 1 -> update 1 -> step 2
            with P():
                loss1, g1, u1, state2, nrm1 = ref_step(state1, (batches[1],), {})
            losses["L1_ref"] = dict(value=float(host(loss1)), f32_hex=f32hex(host(loss1)))
            gnorms["g1"] = norms_record(nrm1, mgn)
            save_tree(out, "g1", named(g1, "model"), full=False)
            u1_items = named(u1, "model")
            save_tree(out, "u1", u1_items, full=True)
            p2_idx, checks["p2"] = save_params_after(out, "p2", p1_items, u1_items, named(state2.model, "model"))
            save_tree(out, "s2", named(state2.opt_state, "opt"), full=False)
            checks["norm_preservation_update1"] = norm_preservation(p1_idx, p2_idx, labels)
            with P():
                loss2, _g2, _u2, _s3, nrm2 = ref_step(state2, (batches[2],), {})
            losses["L2_ref"] = dict(value=float(host(loss2)), f32_hex=f32hex(host(loss2)))
            gnorms["g2"] = norms_record(nrm2, mgn)
            del g1, _g2, _u2, _s3
            log("ref losses", losses["L0_ref"][0]["value"], losses["L1_ref"]["value"], losses["L2_ref"]["value"])

            # ---- the same three steps through levanter's own train_step program (same precision context)
            st = copy_tree(state0)
            tsl, tcmp = [], {}
            for k, b in enumerate(batches):
                with P():
                    info = trainer.train_step(st, b)
                st = info.state
                tsl.append(dict(value=float(info.loss), f32_hex=f32hex(info.loss)))
                if k in (0, 1):
                    want = state1 if k == 0 else state2
                    ts_items = named(st.model, "model")
                    c = compare_trees(ts_items, named(want.model, "model"))
                    # npy where train_step and ref_step differ: the update-1 differences can then be read element by element
                    save_tree(out, f"train_step_p{k + 1}", ts_items, full={n for n, v in c.items() if not v["sha_equal"]})
                    tcmp[f"params_after_update{k}"] = dict(all_bitwise=all(v["sha_equal"] for v in c.values()),
                                                           max_rel=max(v["rel"] for v in c.values()), tensors=c)
                log(f"train_step {k}: loss {info.loss:.7f}")
            losses["L_train_step"] = tsl
            losses["train_step_equals_ref"] = [tsl[0]["f32_hex"] == losses["L0_ref"][0]["f32_hex"],
                                               tsl[1]["f32_hex"] == losses["L1_ref"]["f32_hex"],
                                               tsl[2]["f32_hex"] == losses["L2_ref"]["f32_hex"]]
            checks["train_step_vs_ref_step"] = tcmp
            del st, info, state2, want, ts_items, c

            if args.policy == "f32":
                # ---- T1.7 case B: G0 again at k = 1 from the case-A state (f32 policy: G0 = f32 gradient)
                with P():
                    uB, sB, nB = ref_update(state1, g0)
                gnorms["caseB"] = norms_record(nB, mgn)
                uB_items, sB_items = named(uB, "model"), named(sB.opt_state, "opt")
                save_tree(out, "caseB/u", uB_items, full=True)
                save_tree(out, "caseB/s", sB_items, full=True)
                pB_idx, checks["caseB_p"] = save_params_after(out, "caseB/p", p1_items, uB_items, named(sB.model, "model"))
                checks["norm_preservation_caseB"] = norm_preservation(p1_idx, pB_idx, labels)
                hypB = {n: host(l).item() for n, l in sB_items if n.startswith("hyperparams")}
                checks["caseB"] = dict(k=1, gradient="G0 of this fixture (g0/)", hyperparams_used=hypB,
                                       hyperparams_f32_hex={n: f32hex(v) for n, v in hypB.items()})

                # ---- T1.7 at DEFAULT precision (the run's optimizer precision), cases A and B
                with P():
                    uAh, _sAh, _nAh = ref_update(state0, g0)  # case A through ref_update, as B and C
                    au_hi = dot_audit(ref_update.lower(state0, g0))
                checks["caseA_ref_update_vs_ref_step"] = cmp_summary(compare_trees(named(uAh, "model"), u0_items))
                del uAh, _sAh, _nAh
                checks["precision"]["ref_update"] = au_hi
                checks["precision"]["ref_update_default"] = dot_audit(ref_update.lower(state0, g0))
                checks["default_precision"] = dict(
                    caseA_default=default_copy(out, "caseA_default", ref_update, state0, g0, p0_items, p0_idx,
                                               (u0_items, "u0"), (s1_items, "s1"), labels, gnorms, mgn),
                    caseB_default=default_copy(out, "caseB_default", ref_update, state1, g0, p1_items, p1_idx,
                                               (uB_items, "caseB/u"), (sB_items, "caseB/s"), labels, gnorms, mgn))
                del uB, sB, uB_items, sB_items

        # ---- T1.7 case C: the step-4958 checkpoint, gradient G0, k = 4958
        if case_c:
            import equinox as eqx
            from levanter.checkpoint import load_checkpoint

            src = os.path.join(common.marin_prefix(), args.ckpt)
            listing = lambda: sorted((os.path.relpath(os.path.join(dp, f), src), os.path.getsize(os.path.join(dp, f)),
                                      os.path.getmtime(os.path.join(dp, f))) for dp, _, fs in os.walk(src) for f in fs)
            before = listing()
            template = eqx.filter_eval_shape(lambda s: s, state1)
            stateC = load_checkpoint(template, src, axis_mapping=trainer.parameter_axis_mapping, mesh=trainer.device_mesh)
            stored_step = int(jax.device_get(stateC.step))
            optC, old_counts = set_counts(stateC.opt_state, CASE_C_K)
            stateC = dataclasses.replace(stateC, opt_state=optC, step=jax.numpy.asarray(CASE_C_K, dtype=jax.numpy.int32))
            pin_items = named(stateC.model, "model")
            pin_idx = save_tree(out, "caseC/pin", pin_items, full=False)
            with P():
                uC, sC, nC = ref_update(stateC, g0)
            gnorms["caseC"] = norms_record(nC, mgn)
            uC_items, sC_items = named(uC, "model"), named(sC.opt_state, "opt")
            save_tree(out, "caseC/u", uC_items, full=True)
            save_tree(out, "caseC/s", sC_items, full=True)
            pC_idx, checks["caseC_p"] = save_params_after(out, "caseC/p", pin_items, uC_items, named(sC.model, "model"))
            after = listing()
            hyp = {n: host(l).item() for n, l in sC_items if n.startswith("hyperparams")}
            checks["caseC"] = dict(checkpoint=os.path.relpath(src, common.marin_prefix()), stored_step=stored_step,
                                   stored_counts=old_counts, k=CASE_C_K, counts_set_to=CASE_C_K,
                                   hyperparams_used=hyp, hyperparams_f32_hex={n: f32hex(v) for n, v in hyp.items()},
                                   gradient="G0 of this fixture (g0/)", source_unchanged=before == after,
                                   source_files=len(before))
            checks["norm_preservation_caseC"] = norm_preservation(pin_idx, pC_idx, labels)
            log("case C: stored step", stored_step, "counts", old_counts, "lr", hyp)
            assert before == after, "the checkpoint directory changed"
            if args.policy == "f32":
                checks.setdefault("default_precision", {})["caseC_default"] = default_copy(
                    out, "caseC_default", ref_update, stateC, g0, pin_items, pin_idx, (uC_items, "caseC/u"), (sC_items, "caseC/s"),
                    labels, gnorms, mgn)
            del uC, sC, stateC

        checks["ce_end"] = ce_rec.record()  # after every trace (train_step, ref_update) of this process
        mem = device_memory()

    common.write_json(os.path.join(out, "losses.json"), losses)
    common.write_json(os.path.join(out, "group_norms.json"), gnorms)
    common.write_json(os.path.join(out, "checks.json"), checks)
    if args.repeat_only:
        served = [f"e_floor input (second process) for T2.4/T2.5/T2.7, {args.policy} policy"]
    elif args.policy == "bf16":
        served = ["T2.1 (L0 bf16)", "T2.5 (g0 bf16)", "T2.6 (group norms, clip factors)", "T2.7 (updates 0-1, L1, L2, bf16)",
                  "T2.4/T2.5 e_floor input (in-process repeat)", "init tensors (p0/) of this mesh"]
    else:
        served = ["T2.2 (L0 f32)", "T2.4 (g0 f32)", "T2.5 (f32 reference)", "T1.7 case A (u0, s1, group norms on G0)",
                  "T1.7 case B (caseB/)", "T1.7 at DEFAULT precision (case*_default/)", "T2.7 bf16-vs-f32 floor input",
                  "init tensors (p0/) of this mesh"] + (["T1.7 case C (caseC/)"] if case_c else [])
    if args.debug_rows:
        served = [f"DEBUG ONLY ({args.debug_rows} rows per batch): not a reference"]
    mp_rec = dict(main_chain=prec or "default", jax_config_default=str(jax.config.jax_default_matmul_precision),
                  ref_step_all_highest=checks["precision"]["ref_step_all_highest"],
                  default_precision_copies=sorted(checks.get("default_precision", {})) or None)
    common.write_manifest(out, checks=served, env=common.environment(),
                          extra=dict(size=args.size, policy=args.policy, mp=str(cfg.trainer.mp), run_id=cfg0.trainer.id,
                                     num_train_steps=cfg0.trainer.num_train_steps, applied_run_env=applied_env,
                                     run_env_keys=sorted(lconfig.RUN_ENV.get(args.size, {})), init_keys=keys,
                                     init_combined_sha256=combined, max_grad_norm=mgn, matmul_precision=mp_rec,
                                     ce_implementation=checks["ce_end"], device_memory=mem,
                                     compilation_cache=dict(dir=getattr(jax.config, "jax_compilation_cache_dir", None),
                                                            enabled=getattr(jax.config, "jax_enable_compilation_cache", None))))
    log("wrote", out)


def default_copy(out, tag, ref_update, state_in, grads, pin_items, pin_idx, base_u, base_s, labels, gnorms, mgn) -> dict:
    """ref_update at DEFAULT precision (no context) on one T1.7 case's inputs; u/ and s/ against the HIGHEST result."""
    u, s, nrm = ref_update(state_in, grads)
    u_items, s_items = named(u, "model"), named(s.opt_state, "opt")
    cu = save_variant(out, f"{tag}/u", u_items, base_u[0], base_u[1])
    cs = save_variant(out, f"{tag}/s", s_items, base_s[0], base_s[1])
    p_idx, cp = save_params_after(out, f"{tag}/p", pin_items, u_items, named(s.model, "model"))
    gnorms[tag] = norms_record(nrm, mgn)
    differ = {n: dict(rel_default_vs_highest=v["rel"], max_ulp=v["max_ulp"], group=labels.get(n))
              for n, v in cu.items() if not v["sha_equal"]}
    log(f"{tag}: update tensors that differ from HIGHEST", {n: f"{v['rel_default_vs_highest']:.2e}" for n, v in differ.items()})
    return dict(precision="default", base_u=base_u[1], base_s=base_s[1], u=cmp_summary(cu), u_differ=differ,
                s=cmp_summary(cs), p=cp, norm_preservation=norm_preservation(pin_idx, p_idx, labels))


if __name__ == "__main__":
    import sys

    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown: the loader's background thread can abort it after all files are written
