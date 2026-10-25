"""T1.1 fixture: the levanter seed-0 init of the plain baseline, built exactly as levanter/main/train_lm.py:268.

    python -m tools.levref.init --size 130m [--save full|stats] [--name init]

Runs train_lm.main's path up to the initial state (trainer.initialize, Trainer, split(PRNGKey(seed), 4),
round_axis_for_partitioning, trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key)))
with jax_threefry_partitionable=True from the config's jax_config. Writes to cache/levref/<size>/<name>/:
  checkpoint/          state.model as a levanter tensorstore checkpoint (tree {"model": model}, step 0); reloaded with
                       load_checkpoint(..., subpath="model"), the initialize_model_from_checkpoint_path loader, and
                       compared bitwise (--save full only)
  npy/<name>.npy       every parameter, float32, levanter's stored axis order (--save full only)
  index.json           name -> axes, shape, dtype, sha256 of the f32 bytes; combined digest
  stats.json           per tensor: std / (0.98658 scale), mean, max |w|, the T1.1 bounds, per-layer Frobenius norms;
                       the lm_head / embedding ratio check
  manifest.json
"""
import argparse
import dataclasses
import hashlib
import math
import os
import shutil
import sys
import time

from tools.levref import common, config as lconfig

TN_STD = 0.98658  # std of the (-3, 3) truncated standard normal, as the spec writes it
TN_STD_EXACT = None  # filled in main() from the closed form


def _tn_std_exact() -> float:
    phi3 = math.exp(-4.5) / math.sqrt(2 * math.pi)
    z = math.erf(3 / math.sqrt(2))
    return math.sqrt(1 - 2 * 3 * phi3 / z)


def named_leaves(model, arrays: bool = False):
    """[(name, NamedArray)] in tree order; names are dataclass field paths, without the Stacked wrapper.
    arrays=True also yields plain jax arrays (optimizer counts, injected hyperparameters) as (name, jax.Array)."""
    import haliax as hax
    import jax

    flat = jax.tree_util.tree_flatten_with_path(model, is_leaf=lambda x: isinstance(x, hax.NamedArray))[0]
    out = []
    for path, leaf in flat:
        if not (isinstance(leaf, hax.NamedArray) or (arrays and isinstance(leaf, jax.Array))):
            continue
        parts = []
        for k in path:
            n = getattr(k, "name", None)
            n = str(n if n is not None else getattr(k, "key", getattr(k, "idx", k)))
            if n in ("stacked",):
                continue
            parts.append(n)
        out.append((".".join(parts), leaf))
    return out


def init_scale(name: str, m) -> float | None:
    """The init scale of model.md §4.1 (None for norm gains, which are ones)."""
    D, F, H, hd = m.hidden_dim, m.intermediate_dim, m.num_heads, m.hidden_dim // m.num_heads if m.head_dim is None else m.head_dim
    if "norm" in name.split(".")[-2]:
        return None
    if name.startswith("embeddings.token_embeddings"):
        return 1.0 / D
    if name.startswith("lm_head"):
        return 1.0 / math.sqrt(D)
    if ".down_proj." in name:
        return 1.0 / math.sqrt(F)
    if ".o_proj." in name:
        return 1.0 / math.sqrt(H * hd)
    if name.endswith("_proj.weight"):
        return 1.0 / math.sqrt(D)
    raise KeyError(f"no init scale for {name}")


def tensor_stats(name, axes, w, scale, num_layers):
    import numpy as np

    x = w.astype(np.float64)
    n = x.size
    st = dict(name=name, axes=list(axes), shape=list(w.shape), numel=int(n), mean=float(x.mean()), std=float(x.std()),
              max_abs=float(np.abs(x).max()), min=float(x.min()), max=float(x.max()))
    if scale is None:
        st.update(init="ones", all_ones=bool(np.all(w == 1.0)), ok=bool(np.all(w == 1.0)))
        return st
    std_ratio = st["std"] / (TN_STD * scale)
    b_std = 5 / math.sqrt(2 * n)
    b_mean = 5 * TN_STD * scale / math.sqrt(n)
    st.update(init="truncated_normal(-3,3)*scale", scale=scale, std_over_0p98658_scale=std_ratio,
              std_over_exact_tn_std_scale=st["std"] / (TN_STD_EXACT * scale), max_abs_over_scale=st["max_abs"] / scale,
              bound_std_ratio=b_std, bound_abs_mean=b_mean, bound_max_abs=3 * scale,
              ok_std=abs(std_ratio - 1) <= b_std, ok_mean=abs(st["mean"]) <= b_mean, ok_max_abs=st["max_abs"] < 3 * scale,
              frobenius=float(np.sqrt((x * x).sum())), expected_frobenius=TN_STD * math.sqrt(n) * scale)
    st["ok"] = bool(st["ok_std"] and st["ok_mean"] and st["ok_max_abs"])
    if axes and axes[0] == "layer" and w.shape[0] == num_layers:
        per = x.reshape(num_layers, -1)
        st["per_layer_frobenius"] = [float(v) for v in np.sqrt((per * per).sum(1))]
        st["per_layer_std_over_0p98658_scale"] = [float(v) for v in per.std(1) / (TN_STD * scale)]
        st["expected_per_layer_frobenius"] = TN_STD * math.sqrt(per.shape[1]) * scale
    return st


def head_embedding_ratio(emb, emb_axes, head, head_axes, D):
    """lm_head / embedding = sqrt(D) elementwise (one shared draw, two f32 scalings), model.md §4.2."""
    import numpy as np

    perm = [head_axes.index(a) for a in emb_axes]
    h = np.transpose(head, perm).astype(np.float64)
    e = emb.astype(np.float64)
    nz = e != 0
    r = h[nz] / e[nz] / math.sqrt(D)
    corr = float(np.corrcoef(h.ravel(), e.ravel())[0, 1])
    # the f32 identity behind it: t = emb / f32(1/D) recovered, then head == f32(t * f32(1/sqrt(D)))?
    f32 = np.float32
    t = (emb.astype(f32) / f32(1.0 / D)).astype(f32)
    head_from_t = (t * f32(1.0 / math.sqrt(D))).astype(f32)
    exact = float(np.mean(np.transpose(head, perm) == head_from_t))
    out = dict(axes_head=list(head_axes), axes_embedding=list(emb_axes), zero_embedding_entries=int((~nz).sum()),
               max_rel_dev_from_sqrtD=float(np.abs(r - 1).max()), mean_rel_dev=float(np.abs(r - 1).mean()), correlation=corr,
               bound=2.4e-7, frac_head_equal_f32_recompute=exact)
    out["ok"] = bool(out["max_rel_dev_from_sqrtD"] <= out["bound"] and corr > 0.999999)
    return out


def vocab_axis(cfg, trainer):
    """Vocab exactly as levanter/main/train_lm.py:250-251 (rounded for partitioning; 128256 needs no rounding)."""
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning

    vocab_size = len(cfg.data.the_tokenizer)
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer.parameter_axis_mapping)
    assert Vocab.size == vocab_size == 128256, (Vocab.size, vocab_size)
    return Vocab


def split_keys(cfg):
    """data_key, loader_key, model_key, training_key = split(PRNGKey(seed), 4) (levanter/main/train_lm.py:217-218)."""
    import jax.random as jrandom

    return jrandom.split(jrandom.PRNGKey(cfg.trainer.seed), 4)


def build_initial_state(cfg, trainer):
    """levanter/main/train_lm.py:217-268 inside an open Trainer: (state, Vocab, model_key, keys as lists)."""
    import numpy as np

    _data_key, _loader_key, model_key, training_key = split_keys(cfg)
    keys = dict(model_key=np.asarray(model_key).tolist(), training_key=np.asarray(training_key).tolist())  # initial_state donates training_key
    Vocab = vocab_axis(cfg, trainer)
    state = trainer.initial_state(training_key, model_init=lambda: cfg.model.build(Vocab, key=model_key))
    assert int(state.step) == 0
    return state, Vocab, model_key, keys


def model_digests(model) -> tuple[str, dict]:
    """(combined sha256, {name: sha256 of the f32 bytes}): the same digests main() writes to index.json."""
    import jax
    import numpy as np

    per, combined = {}, hashlib.sha256()
    for name, arr in named_leaves(model):
        w = np.asarray(jax.device_get(arr.array))
        assert w.dtype == np.float32, (name, w.dtype)
        per[name] = common.sha256_array(w)
        combined.update(f"{name}:{per[name]}\n".encode())
    return combined.hexdigest(), per


def main():
    global TN_STD_EXACT
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=lconfig.SIZES)
    ap.add_argument("--save", choices=("full", "stats"), default="full")
    ap.add_argument("--name", default="init", help="fixture subdirectory (e.g. init_1gpu for a device-count probe)")
    args = ap.parse_args()
    TN_STD_EXACT = _tn_std_exact()
    t0 = time.time()

    def log(*a):
        print(f"[levref.init {time.time() - t0:7.1f}s]", *a, flush=True)

    out = common.fixture_dir(args.size, args.name)
    for sub in ("npy", "checkpoint"):
        if os.path.exists(os.path.join(out, sub)):
            shutil.rmtree(os.path.join(out, sub))
    cfg0, cfg, scratch, _applied = lconfig.harness(args.size, "init")

    with lconfig.open_trainer(cfg) as trainer:
        import jax
        import numpy as np

        assert jax.config.jax_threefry_partitionable, "jax_threefry_partitionable must be True (trainer.py DEFAULT_JAX_CONFIG)"
        state, Vocab, model_key, keys = build_initial_state(cfg, trainer)
        model, seed, parameter_axis_mapping = state.model, cfg.trainer.seed, trainer.parameter_axis_mapping
        log("initial state built on", jax.device_count(), jax.devices()[0].device_kind, "mesh", dict(trainer.device_mesh.shape))

        leaves = named_leaves(model)
        m = cfg.model
        index, stats, host = {}, [], {}
        combined = hashlib.sha256()
        for name, arr in leaves:
            w = np.asarray(jax.device_get(arr.array))
            assert w.dtype == np.float32, (name, w.dtype)
            axes = [a.name for a in arr.axes]
            digest = common.sha256_array(w)
            combined.update(f"{name}:{digest}\n".encode())
            index[name] = dict(axes=axes, shape=list(w.shape), dtype=str(w.dtype), sha256=digest,
                               sharding=str(arr.array.sharding.spec) if hasattr(arr.array.sharding, "spec") else str(arr.array.sharding))
            stats.append(tensor_stats(name, axes, w, init_scale(name, m), m.num_layers))
            if name.startswith(("embeddings.token_embeddings", "lm_head")):
                host[name] = (w, axes)
            if args.save == "full":
                os.makedirs(os.path.join(out, "npy"), exist_ok=True)
                np.save(os.path.join(out, "npy", f"{name}.npy"), w)
            log(f"{name:55s} {str(axes):55s} sha {digest[:12]}")
        (emb, emb_axes), = [v for k, v in host.items() if k.startswith("embeddings")]
        (head, head_axes), = [v for k, v in host.items() if k.startswith("lm_head")]
        ratio = head_embedding_ratio(emb, emb_axes, head, head_axes, m.hidden_dim)
        n_params = sum(int(np.prod(v["shape"])) for v in index.values())
        summary = dict(size=args.size, seed=seed, n_tensors=len(index), n_params=n_params, combined_sha256=combined.hexdigest(),
                       all_tensors_pass=all(s["ok"] for s in stats), head_embedding_ratio_pass=ratio["ok"],
                       devices=jax.device_count(), device_kind=jax.devices()[0].device_kind, backend=jax.default_backend(),
                       **keys)
        log("summary", summary)
        common.write_json(os.path.join(out, "index.json"), dict(summary=summary, tensors=index))
        common.write_json(os.path.join(out, "stats.json"), dict(summary=summary, tn_std_spec=TN_STD, tn_std_exact=TN_STD_EXACT,
                                                                 head_embedding_ratio=ratio, tensors=stats))

        roundtrip = None
        if args.save == "full":
            import equinox as eqx
            from levanter.checkpoint import load_checkpoint, save_checkpoint

            ck = os.path.join(out, "checkpoint")
            save_checkpoint({"model": model}, step=0, checkpoint_path=ck, is_temporary=False)
            template = eqx.filter_eval_shape(lambda: cfg.model.build(Vocab, key=model_key))
            loaded = load_checkpoint(template, ck, subpath="model", axis_mapping=parameter_axis_mapping, mesh=trainer.device_mesh)
            bad = []
            for (n1, a1), (n2, a2) in zip(leaves, named_leaves(loaded)):
                assert n1 == n2, (n1, n2)
                if not np.array_equal(np.asarray(jax.device_get(a1.array)), np.asarray(jax.device_get(a2.array))):
                    bad.append(n1)
            roundtrip = dict(loader="levanter.checkpoint.load_checkpoint(subpath='model') as initialize_model_from_checkpoint_path",
                             tensors=len(leaves), bitwise_equal=not bad, mismatched=bad)
            log("checkpoint round trip", roundtrip)
            assert not bad, bad

    # scratch (logs) stays until the job wrapper removes TMPDIR: the levanter log handler still writes there at exit
    checks = ["T1.1", "T2.1-T2.7 input (targets.md §4(a) fixture 1)", "T4 start point"]
    common.write_manifest(out, checks=checks, env=common.environment(),
                          extra=dict(size=args.size, save=args.save, run_id=cfg0.trainer.id, num_train_steps=cfg0.trainer.num_train_steps,
                                     summary=summary, checkpoint_roundtrip=roundtrip,
                                     reference="trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key)), levanter/main/train_lm.py:268"))
    log("wrote", out)
    if not (summary["all_tensors_pass"] and ratio["ok"]):
        print("WARNING: a T1.1 statistical bound failed; see stats.json", file=sys.stderr)


if __name__ == "__main__":
    main()
