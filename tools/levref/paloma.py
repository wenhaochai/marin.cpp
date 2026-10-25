"""T2.3 and T3.1 fixtures: levanter's own full Paloma eval (cb_tagged_lm_evaluate) of the init model and of a trained checkpoint.

    python -m tools.levref.paloma --size 130m --run init:bf16 --run init:f32 --run ckpt:bf16:dump

Each --run is <model>:<policy>[:dump].
  model   init  the seed-0 initial state, built in process exactly as tools.levref.init (train_lm.py:268); its combined
                sha256 is compared with cache/levref/<size>/init/index.json when that fixture exists
          ckpt  the `model` subtree of a levanter checkpoint (--ckpt, relative to MARIN_PREFIX; default the 130m
                step-4958 checkpoint of muonh-qwen3-130m-della4xh100-restore). It is copied to $TMPDIR first and loaded
                from the copy, so nothing touches the run directory; a listing of the source (names, sizes, mtimes) is
                taken before and after and must be unchanged
  policy  bf16  the logged trainer.mp (p=f32, c=bfloat16, out=bfloat16)
          f32   config.with_policy(cfg, "f32").trainer.mp: compute_dtype float32 (targets.md §4 "f32 policy")
  dump    also dump, for every eval batch, levanter's per-position outputs (_default_lm_eval_loss_fn:
          compute_next_token_loss(reduction=None), the loss weight, roll(tokens, -1)) and the batch's tag array

The eval itself is train_lm.py:362-372: cb_tagged_lm_evaluate(EvalBatch, config.data.tagged_eval_sets(Pos), tokenizer,
trainer.device_mesh, compute_axis_mapping, max_eval_batches * eval_batch_size (None), mp=<policy>, checkpoint_path=<fixture dir>),
called once with a StepInfo whose state.step = <logged step> + 1. levanter itself writes <fixture dir>/eval_metrics.jsonl
in the format of the run's checkpoints/eval_metrics.jsonl. Fixture dir: cache/levref/<size>/eval_<model>[_step<k>]_<policy>/.

Files per fixture:
  eval_metrics.jsonl  levanter's record (step, the 42 keys of eval.md §4.2)
  metrics.json        the same values with their float32 bit patterns; for ckpt runs, the per-key ulp distance to the
                      reference record (<ckpt>/../eval_metrics.jsonl, row of the same step)
  dump/               (dump only) loss.npy f32 [114*128, 4096], weight.npy f32, target.npy int32, tags.npy int32 [rows, 16];
                      row r is row r % 128 of eval batch r // 128; tags.json (tag order = tag index)
  reagg.json          (dump only) the dump fed back to levanter's own aggregation (TaggedEvaluator.evaluate with an identity
                      loss_fn, same jitted accum_for_batch) and to two numpy aggregations (float64 exact; float32 running
                      means over float64-exact batch sums), each key as an ulp distance to the evaluator's own value
  checks.json         dump checks: weights per eval.md §2.2 item 3, one-hot tags, 14529 real rows and 63 padding rows,
                      per-domain token sha256 / rows / target bytes against cache/data/data_ref.json (T0.4), byte table
                      sha256 (T0.5), all losses finite
  manifest.json
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
import types
import warnings

from tools.levref import common, config as lconfig, init as linit

DEFAULT_CKPT = os.path.join("speedrun", "muonh-qwen3-130m-della4xh100-restore", "2026.09.13", "checkpoints", "step-4958")
TIME_KEYS = ("eval/loading_time", "eval/total_time")
_t0 = time.time()


def log(*a):
    print(f"[levref.paloma {time.time() - _t0:7.1f}s]", *a, flush=True)


def f32_bits(x: float) -> str:
    import numpy as np

    return f"0x{int(np.float32(x).view(np.uint32)):08x}"


def ulp_f32(a: float, b: float) -> int | None:
    """Distance in float32 ulps between two values that are both exactly float32 (None if either is not)."""
    import numpy as np

    if a is None or b is None or float(np.float32(a)) != a or float(np.float32(b)) != b:
        return None
    from tools.levref.compare import ulp_distance

    return int(ulp_distance(np.array([a], np.float32), np.array([b], np.float32))[0])


def _closure_get(fn, cls):
    """The one object of type `cls` that the closure `fn` captured (cb_tagged_lm_evaluate's TaggedEvaluator)."""
    found = [c.cell_contents for c in (fn.__closure__ or ()) if isinstance(c.cell_contents, cls)]
    assert len(found) == 1, (fn, cls, found)
    return found[0]


def _dir_listing(path: str) -> dict:
    out = {}
    for dirpath, _, names in os.walk(path):
        for n in names:
            p = os.path.join(dirpath, n)
            st = os.stat(p)
            out[os.path.relpath(p, path)] = [st.st_size, st.st_mtime_ns]
    return out


def compare_keys(got: dict, ref: dict) -> dict:
    rows = {}
    for k in sorted(set(got) | set(ref)):
        if k == "step":
            continue
        a, b = got.get(k), ref.get(k)
        r = dict(got=a, ref=b)
        if k in TIME_KEYS:
            r["note"] = "wall time, not compared"
        elif a is None or b is None:
            r["missing"] = "got" if a is None else "ref"
        else:
            r.update(ulp=ulp_f32(a, b), abs_diff=abs(a - b), got_bits=f32_bits(a), ref_bits=f32_bits(b))
        rows[k] = r
    cmp = [r for k, r in rows.items() if k not in TIME_KEYS]
    ulps = [r.get("ulp") for r in cmp]
    return dict(rows=rows, n_keys=len(rows), n_compared=len(cmp), n_missing=sum("missing" in r for r in cmp),
                n_bitwise_equal=sum(u == 0 for u in ulps), max_ulp=max((u for u in ulps if u is not None), default=None),
                max_abs_diff=max((r.get("abs_diff", 0.0) for r in cmp), default=None))


# ---------------------------------------------------------------------------------------------- numpy aggregations
def numpy_aggregate(loss, weight, target, tags, bytes_per_token, eval_batch: int, tag_names: list[str], dtype: str) -> dict:
    """eval.md §4.1-4.2 in numpy. dtype "f64": every sum and running mean in float64. "f32": per-batch sums exact (float64,
    then rounded to float32), the four RunningMeans updated in float32 in batch order exactly as stat_utils.RunningMean."""
    import numpy as np

    f = np.float64 if dtype == "f64" else np.float32
    nb = loss.shape[0] // eval_batch
    K = tags.shape[1]
    log2e = f(np.log2(np.e)) if dtype == "f64" else np.log2(np.float32(np.e))

    def rm_add(mean, total, x, w):
        x, w = f(x) if np.ndim(x) == 0 else x.astype(f), f(w) if np.ndim(w) == 0 else w.astype(f)
        new_total = (total + w).astype(f) if np.ndim(total) else f(total + w)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(new_total != 0, w / new_total, f(0)).astype(f)
        return (mean + (x - mean) * ratio).astype(f), new_total

    tok = [f(0), f(0)]
    bpb = [f(0), f(0)]
    tag_l = [np.zeros(K, f), np.zeros(K, f)]
    tag_b = [np.zeros(K, f), np.zeros(K, f)]
    bpt = bytes_per_token.astype(np.float64)
    for b in range(nb):
        sl = slice(b * eval_batch, (b + 1) * eval_batch)
        l, w, y, G = loss[sl].astype(np.float64), weight[sl].astype(np.float64), target[sl], tags[sl].astype(np.float64)
        lw = l * w
        beta = bpt[y]
        L, W, B = f(lw.sum()), f(w.sum()), f((beta * w).sum())
        Lk, Wk, Bk = (lw.sum(1) @ G).astype(f), (w.sum(1) @ G).astype(f), ((beta * w).sum(1) @ G).astype(f)
        tok = list(rm_add(tok[0], tok[1], L / max(W, f(1)), W))
        with np.errstate(invalid="ignore", divide="ignore"):
            safe = np.where(Wk > 0, Lk / Wk, f(0)).astype(f)
        tag_l = list(rm_add(tag_l[0], tag_l[1], safe, Wk))
        bpb = list(rm_add(bpb[0], bpb[1], f(L / max(B, f(1)) * log2e), W))
        tag_b = list(rm_add(tag_b[0], tag_b[1], (Lk / np.maximum(Bk, f(1)) * log2e).astype(f), Wk))
    mask = tag_l[1] > 0
    out = {"eval/loss": float(tok[0]), "eval/bpb": float(bpb[0]),
           "eval/macro_loss": float(np.mean(tag_l[0])), "eval/macro_bpb": float(np.mean(tag_b[0])),
           "eval/paloma/macro_loss": float(np.mean(tag_l[0], where=mask)),
           "eval/paloma/micro_loss": float(np.average(tag_l[0], weights=tag_l[1] * mask)),
           "eval/paloma/macro_bpb": float(np.mean(tag_b[0], where=mask)),
           "eval/paloma/bpb": float(np.average(tag_b[0], weights=tag_b[1] * mask))}
    for i, t in enumerate(tag_names):
        out[f"eval/{t}/loss"] = float(tag_l[0][i])
        out[f"eval/{t}/bpb"] = float(tag_b[0][i])
    return out


def ulp_table(got: dict, ref: dict) -> dict:
    import numpy as np

    rows = {}
    for k, b in ref.items():
        if k in TIME_KEYS or k == "step" or k not in got:
            continue
        a = got[k]
        rows[k] = dict(value=a, ref=b, ulp=ulp_f32(float(np.float32(a)), b), abs_diff=abs(a - b))
    ul = [r["ulp"] for r in rows.values() if r["ulp"] is not None]
    return dict(rows=rows, max_ulp=max(ul, default=None), n_bitwise_equal=sum(u == 0 for u in ul), n=len(rows),
                max_abs_diff=max((r["abs_diff"] for r in rows.values()), default=None))


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=lconfig.SIZES)
    ap.add_argument("--run", action="append", required=True, help="<init|ckpt>:<bf16|f32>[:dump]; repeatable")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint directory relative to MARIN_PREFIX")
    ap.add_argument("--debug-examples-per-dataset", type=int, default=None,
                    help="DEBUG ONLY: cb_tagged_lm_evaluate max_examples_per_dataset (a partial eval); fixture name gets _debug")
    ap.add_argument("--debug-eval-pdp", type=int, default=None, help="DEBUG ONLY: per_device_eval_parallelism override")
    args = ap.parse_args()
    runs = []
    for r in args.run:
        parts = r.split(":")
        assert parts[0] in ("init", "ckpt") and parts[1] in ("bf16", "f32") and set(parts[2:]) <= {"dump"}, r
        runs.append(dict(model=parts[0], policy=parts[1], dump="dump" in parts[2:]))
    debug = args.debug_examples_per_dataset is not None or args.debug_eval_pdp is not None

    cfg0, cfg, scratch, applied_env = lconfig.harness(args.size, "paloma")  # bf16 policy; the f32 policy below via with_policy
    if args.debug_eval_pdp is not None:
        cfg = dataclasses.replace(cfg, trainer=dataclasses.replace(cfg.trainer, per_device_eval_parallelism=args.debug_eval_pdp))

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import levanter.eval as leval
    from levanter.callbacks import StepInfo
    from levanter.checkpoint import load_checkpoint

    data_ref = json.load(open(os.path.join(common.nano_dir(), "cache", "data", "data_ref.json")))
    ckpt_src = os.path.join(common.marin_prefix(), args.ckpt)

    with lconfig.open_trainer(cfg) as trainer:
        assert jax.config.jax_threefry_partitionable
        compute_axis_mapping = trainer.compute_axis_mapping
        EvalBatch = cfg.trainer.EvalBatch
        Pos = cfg.model.max_Pos.resize(cfg.train_seq_len)
        tokenizer = cfg.data.the_tokenizer
        tagged_eval_datasets = cfg.data.tagged_eval_sets(Pos)
        max_eval_examples_per_ds = cfg.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= cfg.trainer.eval_batch_size
        if args.debug_examples_per_dataset is not None:
            max_eval_examples_per_ds = args.debug_examples_per_dataset
        Vocab = linit.vocab_axis(cfg, trainer)
        mp_bf16 = cfg.trainer.mp
        assert (mp_bf16.param_dtype, mp_bf16.compute_dtype, mp_bf16.output_dtype) == (jnp.float32, jnp.bfloat16, jnp.bfloat16), mp_bf16
        policies = {p: lconfig.with_policy(cfg, p).trainer.mp for p in lconfig.POLICIES}
        log("devices", jax.device_count(), jax.devices()[0].device_kind, "mesh", dict(trainer.device_mesh.shape), "EvalBatch", EvalBatch.size,
            "eval sets", len(tagged_eval_datasets), "max_examples_per_dataset", max_eval_examples_per_ds)

        models: dict = {}

        def get_model(kind):
            if kind in models:
                return models[kind]
            info: dict = {}
            if kind == "init":
                state, _V, _mk, keys = linit.build_initial_state(cfg, trainer)
                model, step = state.model, 0
                info.update(source="trainer.initial_state (levanter/main/train_lm.py:268), as tools.levref.init", keys=keys)
            else:
                before = _dir_listing(ckpt_src)
                copy = os.path.join(scratch, "ckpt_copy")
                shutil.copytree(ckpt_src, copy)
                meta = json.load(open(os.path.join(copy, "metadata.json")))
                step = int(meta["step"]) + 1  # state.step after the last update; StepInfo.step = state.step - 1
                _dk, _lk, model_key, _tk = linit.split_keys(cfg)
                template = eqx.filter_eval_shape(lambda: cfg.model.build(Vocab, key=model_key))
                model = load_checkpoint(template, copy, subpath="model", axis_mapping=trainer.parameter_axis_mapping, mesh=trainer.device_mesh)
                after = _dir_listing(ckpt_src)
                assert before == after, "checkpoint source changed while reading"
                jax.block_until_ready(model)
                shutil.rmtree(copy)  # the tensors are on device now
                info.update(source=f"load_checkpoint(subpath='model') of a $TMPDIR copy of $MARIN_PREFIX/{args.ckpt}", checkpoint_metadata=meta,
                            source_files=len(before), source_bytes=sum(v[0] for v in before.values()), source_unchanged=True)
                art = os.path.join(os.path.dirname(os.path.dirname(ckpt_src)), ".artifact.json")  # <run>/<version>/.artifact.json
                if os.path.exists(art):
                    a = json.load(open(art))
                    info["run_provenance"] = a.get("provenance")
                    built = lconfig._plain(dataclasses.asdict(cfg0))
                    diffs = []
                    lconfig._walk(built, a["config"]["train_config"], "", diffs)
                    info["run_config_diff_vs_built"] = [dict(path=p.lstrip("."), built=x, run=y) for p, x, y in diffs]
            combined, per = linit.model_digests(model)
            info.update(state_step=step, combined_sha256=combined, n_tensors=len(per))
            if kind == "init":
                idx = os.path.join(common.nano_dir(), "cache", "levref", args.size, "init", "index.json")
                if os.path.exists(idx):
                    ref = json.load(open(idx))["summary"]
                    info["init_fixture_combined_sha256"] = ref["combined_sha256"]
                    info["init_fixture_device_kind"] = ref.get("device_kind")
                    info["equals_init_fixture"] = ref["combined_sha256"] == combined
            sh = {n: str(getattr(a.array.sharding, "spec", a.array.sharding)) for n, a in linit.named_leaves(model)}
            info["param_sharding"] = sh
            log(kind, "model ready: state.step", step, "combined sha", combined[:16], {k: v for k, v in info.items() if k.startswith("equals")})
            models[kind] = (model, step, info)
            return models[kind]

        for run in runs:
            model, state_step, minfo = get_model(run["model"])
            name = f"eval_{run['model']}" + (f"_step{state_step - 1}" if run["model"] == "ckpt" else "") + f"_{run['policy']}" + ("_debug" if debug else "")
            out = common.fixture_dir(args.size, name)
            for stale in ("eval_metrics.jsonl", "dump"):
                p = os.path.join(out, stale)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
            mp = policies[run["policy"]]
            log(name, "policy", mp)
            # ---- train_lm.py:362-372, with this run's policy and the fixture dir as checkpoint_path
            cb = leval.cb_tagged_lm_evaluate(EvalBatch, tagged_eval_datasets, tokenizer, trainer.device_mesh, compute_axis_mapping,
                                             max_eval_examples_per_ds, mp=mp, checkpoint_path=out)
            evaluator = _closure_get(cb, leval.TaggedEvaluator)
            st = types.SimpleNamespace(model=model, step=jnp.array(state_step, jnp.int32), model_averaging=None, eval_model=model)
            t = time.time()
            with warnings.catch_warnings(record=True) as caught:  # e.g. the fused-CE fallback warning (eval.md §3 item 5)
                warnings.simplefilter("always")
                cb(StepInfo(state=st, loss=float("nan"), step_duration=0.0))
            caught = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
            log(name, f"evaluator done in {time.time() - t:.1f}s; warnings: {caught}")
            lines = open(os.path.join(out, "eval_metrics.jsonl")).read().splitlines()
            assert len(lines) == 1, lines
            rec = json.loads(lines[0])
            eval_keys = sorted(k for k in rec if k != "step")
            metrics = dict(step=rec["step"], n_keys=len(eval_keys), values={k: rec[k] for k in eval_keys},
                           float32_bits={k: f32_bits(rec[k]) for k in eval_keys if k not in TIME_KEYS},
                           policy=dict(param=str(mp.param_dtype), compute=str(mp.compute_dtype), output=str(mp.output_dtype)),
                           eval_batch=EvalBatch.size, n_batches=len(evaluator.loader), tag_to_index=evaluator.dataset.tag_to_index)
            checks = ["T2.3 reference (targets.md §4(a) fixture 4)"] if run["model"] == "init" else []
            ref_rec = None
            if run["model"] == "ckpt":
                ref_path = os.path.join(os.path.dirname(ckpt_src), "eval_metrics.jsonl")
                ref_rows = [json.loads(x) for x in open(ref_path).read().splitlines() if x.strip()]
                ref_rec = [r for r in ref_rows if r["step"] == rec["step"]][-1]
                metrics["reference"] = dict(path=f"$MARIN_PREFIX/{os.path.relpath(ref_path, common.marin_prefix())}", step=ref_rec["step"])
                metrics["vs_reference"] = compare_keys(rec, ref_rec)
                v = metrics["vs_reference"]
                log(name, f"vs reference: {v['n_bitwise_equal']}/{v['n_compared']} keys bitwise equal, max ulp {v['max_ulp']}, missing {v['n_missing']}")
                checks += ["T3.1 reference: levanter evaluator output (42 keys) and its agreement with the run's eval_metrics.jsonl"]
            common.write_json(os.path.join(out, "metrics.json"), metrics)

            extra = dict(size=args.size, run=run, debug=debug, model=minfo, n_eval_keys=len(eval_keys), run_env_applied=applied_env,
                         eval_warnings=caught, fused_ce_fallback=any("Fused cross-entropy implementation unavailable" in w for w in caught))
            if run["dump"]:
                extra.update(dump_and_check(out, model, evaluator, rec, data_ref, compute_axis_mapping, debug))
                checks += ["T3.1 input: per-position loss, weight, target ids, tags of every eval batch (eval.md §8 test 2)", "T0.4", "T0.5"]
            common.write_manifest(out, checks=checks or ["debug"], env=common.environment(), extra=extra)
            log(name, "wrote", out)
    return 0


def dump_and_check(out, model, evaluator, rec, data_ref, compute_axis_mapping, debug) -> dict:
    import haliax as hax
    import jax
    import numpy as np
    import levanter.eval as leval

    loss_fn = evaluator.loss_fn  # cb_tagged_lm_evaluate's closure: _default_lm_eval_loss_fn with this run's EvalBatch and mp
    dump_fn = hax.named_jit(lambda m, b: loss_fn(m, b), axis_resources=compute_axis_mapping)
    dev, host = [], dict(loss=[], weight=[], target=[], tags=[])
    shardings = None
    t = time.time()
    for k, (batch, tags) in enumerate(evaluator.loader):
        l, w, y = dump_fn(model, batch)
        dev.append(((l, w, y), tags))
        if shardings is None:
            shardings = dict(loss=str(l.sharding), weight=str(w.sharding), target=str(y.sharding), tags=str(tags.sharding))
        host["loss"].append(np.asarray(jax.device_get(l)))
        host["weight"].append(np.asarray(jax.device_get(w)))
        host["target"].append(np.asarray(jax.device_get(y)))
        host["tags"].append(np.asarray(jax.device_get(tags)))
    log(f"dumped {len(dev)} batches in {time.time() - t:.1f}s")
    loss, weight, target, tags = (np.concatenate(host[n]) for n in ("loss", "weight", "target", "tags"))
    EB = host["loss"][0].shape[0]
    tag_names = [t for t, _ in sorted(evaluator.dataset.tag_to_index.items(), key=lambda kv: kv[1])]
    d = os.path.join(out, "dump")
    os.makedirs(d, exist_ok=True)
    for n, a in (("loss", loss), ("weight", weight), ("target", target), ("tags", tags)):
        np.save(os.path.join(d, f"{n}.npy"), a)
    common.write_json(os.path.join(d, "tags.json"), dict(tag_names=tag_names, eval_batch=EB, n_batches=len(dev),
                                                          dtypes={n: str(a.dtype) for n, a in (("loss", loss), ("weight", weight), ("target", target), ("tags", tags))},
                                                          shapes={n: list(a.shape) for n, a in (("loss", loss), ("weight", weight), ("target", target), ("tags", tags))},
                                                          device_shardings=shardings))

    # ---- levanter's own aggregation of the dumped arrays: same TaggedEvaluator state, identity loss_fn, fresh accum_for_batch
    agg = object.__new__(leval.TaggedEvaluator)
    agg.__dict__.update(evaluator.__dict__)
    agg.loss_fn = lambda _model, b: b
    agg.loader = dev
    agg.accum_for_batch = agg._make_accum_for_batch()
    res = agg.evaluate(None)
    lev_reagg = leval.construct_log_dict(agg, res, 0.0, "eval")
    lev_reagg = {k: float(v) for k, v in lev_reagg.items()}
    bpt = np.asarray(jax.device_get(evaluator.bytes_per_token)).astype(np.int32)
    ev = {k: v for k, v in rec.items() if k != "step"}
    reagg = dict(levanter_aggregation_of_dump=ulp_table(lev_reagg, ev))
    for dt in ("f64", "f32"):
        reagg[f"numpy_{dt}"] = ulp_table(numpy_aggregate(loss, weight, target, tags, bpt, EB, tag_names, dt), ev)
    for n, r in reagg.items():
        log(f"re-aggregation {n}: max ulp {r['max_ulp']} vs evaluator, bitwise equal {r['n_bitwise_equal']}/{r['n']}")
    common.write_json(os.path.join(out, "reagg.json"), reagg)

    # ---- checks
    real = tags.sum(1) > 0
    want_w = np.ones(loss.shape[1], np.float32)
    want_w[-1] = 0
    c: dict = dict(rows=int(loss.shape[0]), real_rows=int(real.sum()), padding_rows=int((~real).sum()), eval_batch=EB, n_batches=len(dev))
    c["tags_one_hot_on_real_rows"] = bool(np.all(tags[real].sum(1) == 1) and np.all((tags == 0) | (tags == 1)))
    c["weights_real_rows_formula"] = bool(np.all(weight[real] == want_w))
    c["weights_padding_rows_zero"] = bool(np.all(weight[~real] == 0))
    c["padding_rows_target_zero"] = bool(np.all(target[~real] == 0))
    c["all_losses_finite"] = bool(np.all(np.isfinite(loss)))
    c["loss_zero_where_weight_zero"] = bool(np.all(loss[weight == 0] == 0))
    c["bytes_per_token_sha256"] = common.sha256_array(bpt.astype("<i4"))
    c["T0.5_bytes_table_equal"] = c["bytes_per_token_sha256"] == data_ref["bytes_per_token"]["sha256"]
    pal = data_ref["paloma"]
    dom = {}
    for i, t in enumerate(tag_names):
        rows = np.where(tags[:, i] == 1)[0]
        sub = t.split("/", 1)[1].removesuffix("-marin-tokenizer")
        toks = np.roll(target[rows], 1, axis=1).astype("<i4")  # target = roll(tokens, -1), so tokens = roll(target, 1)
        r = dict(tag=t, rows=int(len(rows)), first_row=int(rows[0]) if len(rows) else None, last_row=int(rows[-1]) if len(rows) else None,
                 contiguous=bool(len(rows) == 0 or rows[-1] - rows[0] + 1 == len(rows)), token_sha256=common.sha256_array(toks),
                 target_bytes=int((bpt[target[rows]].astype(np.int64) * weight[rows].astype(np.int64)).sum()),
                 scored=int(weight[rows].astype(np.int64).sum()))
        ref = pal.get(sub)
        if ref is not None:
            r.update(ref_rows=ref["seqs"], ref_sha256=ref["sha256_seqs"], ref_target_bytes=ref["target_bytes"], ref_scored=ref["scored_positions"],
                     ok=bool(r["rows"] == ref["seqs"] and r["token_sha256"] == ref["sha256_seqs"] and r["target_bytes"] == ref["target_bytes"]
                             and r["scored"] == ref["scored_positions"]))
        dom[sub] = r
    c["domains"] = dom
    c["domain_order"] = [t.split("/", 1)[1].removesuffix("-marin-tokenizer") for t in tag_names]
    c["domain_order_is_A_byte_sorted"] = c["domain_order"] == sorted(c["domain_order"])
    c["rows_in_domain_order"] = bool(all(dom[a]["last_row"] < dom[b]["first_row"] for a, b in zip(c["domain_order"], c["domain_order"][1:])
                                         if dom[a]["rows"] and dom[b]["rows"]))
    if not debug:
        c["T0.4_all_domains_ok"] = all(r.get("ok") for r in dom.values())
        c["T0.4_counts_ok"] = c["real_rows"] == data_ref["paloma_total_seqs"] and c["n_batches"] == data_ref["paloma_eval_batches_of_128"] \
            and c["padding_rows"] == EB * c["n_batches"] - data_ref["paloma_total_seqs"] and EB == 128
    c["dump_reproduces_evaluator_bitwise"] = reagg["levanter_aggregation_of_dump"]["max_ulp"] == 0
    common.write_json(os.path.join(out, "checks.json"), c)
    log("checks", {k: v for k, v in c.items() if k not in ("domains",)})
    return dict(dump_checks={k: v for k, v in c.items() if k != "domains"},
                reagg_max_ulp={n: r["max_ulp"] for n, r in reagg.items()})


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip interpreter teardown: the loaders' background threads can abort it after all files are written
