"""Batch-0 fixture: tokens, segment ids and loss weights exactly as the levanter train loader hands them to step 0.

    python -m tools.levref.batch --size 130m [--verify-steps 3]

Builds config.data.train_set(Pos, batch_schedule, key=PRNGKey(data_seed=42)) and trainer.data_loader(...).iter_from_step(0)
inside the baseline Trainer (levanter/main/train_lm.py:256-260, :455-460), so the batch carries the trainer's mesh sharding
(4 GPUs: batch axis on `data`). Checks against cache/data/data_ref.json and data.md §3.1/§7:
  T0.2  sha256 of the int32 little-endian [B, 4096] tokens equals H(B, k) for every verified step k that data_ref has
  T0.3  segment ids = cumsum(roll(tokens, 1) == 128001) with position 0 forced to 0; q and kv ids equal;
        loss weight = 1 except position 4095 (= 0); per-microbatch weight sum = 4095 x rows
Writes cache/levref/<size>/batch0/{tokens.npy, segment_ids.npy, loss_weight.npy, layout.json, checks.json, manifest.json}.
layout.json holds every leaf of the example (path, shape, dtype, sharding spec) and, per device, the rows it holds, their
token sha256 and loss-weight sum, plus the microbatch split (grad_accum.py:170-179: microbatch j = rows [j*mb, (j+1)*mb)).
"""
import argparse
import dataclasses
import json
import os
import time

from tools.levref import common, config as lconfig

EOS = 128001


def _leaves(example):
    import jax

    flat = jax.tree_util.tree_flatten_with_path(example)[0]
    out = []
    for path, leaf in flat:
        name = ".".join(getattr(k, "name", None) or str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        out.append((name, leaf))
    return out


def train_iterator(cfg, trainer, start_step: int = 0):
    """trainer.data_loader(config.data.train_set(Pos, batch_schedule, key=PRNGKey(data_seed))).iter_from_step(start_step),
    exactly as levanter/main/train_lm.py:256-260, :455-460 (the batches carry the trainer's mesh sharding)."""
    import jax.random as jrandom

    Pos = cfg.model.max_Pos.resize(cfg.train_seq_len)
    train_dataset = cfg.data.train_set(Pos, cfg.trainer.batch_schedule, key=jrandom.PRNGKey(cfg.data_seed))
    return trainer.data_loader(train_dataset).iter_from_step(start_step)


def stop_iterator(it) -> None:
    """Stop the loader's prefetch thread (it aborts the interpreter at exit otherwise)."""
    if hasattr(getattr(it, "_batches", None), "stop"):
        it._batches.stop()


def tokens_sha256(example) -> str:
    """sha256 of the int32 little-endian [B, S] tokens of a batch: H(B, k) of data.md §7."""
    import jax
    import numpy as np

    (tok,) = [l for n, l in _leaves(example) if n.startswith("tokens")]
    tok = np.asarray(jax.device_get(tok.array if hasattr(tok, "array") else tok))
    return common.sha256_array(tok.astype("<i4"))


def _segment_formula(tokens):
    import numpy as np

    prev_eos = np.zeros_like(tokens, dtype=np.int32)
    prev_eos[:, 1:] = (tokens[:, :-1] == EOS).astype(np.int32)
    return np.cumsum(prev_eos, axis=1, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=lconfig.SIZES)
    ap.add_argument("--verify-steps", type=int, default=3, help="also hash-check batches 1..n-1 against data_ref.json")
    args = ap.parse_args()
    t0 = time.time()

    def log(*a):
        print(f"[levref.batch {time.time() - t0:7.1f}s]", *a, flush=True)

    out = common.fixture_dir(args.size, "batch0")
    scratch = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"levref-batch-{args.size}-{os.getpid()}")
    cfg0, _ = lconfig.build(args.size)
    cfg = lconfig.for_harness(cfg0, scratch)
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        cfg = dataclasses.replace(cfg, trainer=dataclasses.replace(cfg.trainer, require_accelerator=False))

    import jax
    import jax.random as jrandom
    import numpy as np
    import levanter
    from levanter.trainer import Trainer

    levanter.trainer.initialize(cfg)
    optimizer = cfg.optimizer.build(cfg.trainer.num_train_steps)

    def loss_function(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=cfg.z_loss_weight)

    ref = json.load(open(os.path.join(common.nano_dir(), "cache", "data", "data_ref.json")))["batch_hashes"]
    B = cfg.trainer.train_batch_size
    S = cfg.train_seq_len
    checks: dict = dict(size=args.size, B=B, S=S, steps={})
    with Trainer(cfg.trainer, optimizer, loss_function) as trainer:
        it = train_iterator(cfg, trainer, 0)
        n_dev = trainer.device_mesh.devices.size
        pdp = cfg.trainer.per_device_parallelism
        mb = pdp * cfg.trainer.data_axis_size
        log("mesh", dict(trainer.device_mesh.shape), "B", B, "per_device_parallelism", pdp, "microbatch", mb)
        for k in range(max(1, args.verify_steps)):
            ex = next(it)
            leaves = _leaves(ex)
            host = {n: np.asarray(jax.device_get(l.array if hasattr(l, "array") else l)) for n, l in leaves}
            tok_name = [n for n in host if n.startswith("tokens")][0]
            lw_name = [n for n in host if n.startswith("loss_weight")][0]
            seg_names = [n for n in host if "segment_ids" in n]
            tokens = host[tok_name]
            assert tokens.shape == (B, S) and tokens.dtype == np.int32, (tokens.shape, tokens.dtype)
            h = common.sha256_array(tokens.astype("<i4"))
            key = f"B{B}_step{k}"
            st = dict(sha256=h, ref_sha256=ref.get(key, {}).get("sha256"))
            st["T0.2_hash_equal"] = None if st["ref_sha256"] is None else h == st["ref_sha256"]
            seg = _segment_formula(tokens)
            st["segment_leaves"] = seg_names
            st["T0.3_segment_ids_equal_formula"] = all(np.array_equal(host[n], seg) for n in seg_names) and bool(seg_names)
            lw = host[lw_name]
            want = np.ones((B, S), np.float32)
            want[:, -1] = 0
            st["loss_weight_dtype"] = str(lw.dtype)
            st["T0.3_loss_weight_equal_formula"] = bool(np.array_equal(lw, want))
            st["loss_weight_sum"] = float(lw.astype(np.float64).sum())
            st["loss_weight_sum_per_microbatch"] = [float(lw[j * mb:(j + 1) * mb].astype(np.float64).sum()) for j in range(B // mb)]
            checks["steps"][str(k)] = st
            log(f"batch {k}: sha {h[:16]} ref {str(st['ref_sha256'])[:16]} hash_ok={st['T0.2_hash_equal']} "
                f"seg_ok={st['T0.3_segment_ids_equal_formula']} lw_ok={st['T0.3_loss_weight_equal_formula']} lw_sum={st['loss_weight_sum']}")
            if k == 0:
                np.save(os.path.join(out, "tokens.npy"), tokens.astype(np.int32))
                np.save(os.path.join(out, "segment_ids.npy"), host[seg_names[0]].astype(np.int32))
                np.save(os.path.join(out, "loss_weight.npy"), lw)
                leaf_info, per_device = [], {}
                for n, l in leaves:
                    arr = l.array if hasattr(l, "array") else l
                    info = dict(path=n, shape=list(arr.shape), dtype=str(arr.dtype),
                                axes=[a.name for a in l.axes] if hasattr(l, "axes") else None,
                                sharding=str(getattr(arr.sharding, "spec", arr.sharding)))
                    if hasattr(arr, "addressable_shards") and arr.ndim >= 1:
                        info["shards"] = [dict(device=s.device.id, device_kind=s.device.device_kind,
                                               index=[[sl.start, sl.stop] if isinstance(sl, slice) else sl for sl in s.index])
                                          for s in sorted(arr.addressable_shards, key=lambda s: s.device.id)]
                    leaf_info.append(info)
                tok_arr = [l for n, l in leaves if n == tok_name][0]
                tok_arr = tok_arr.array if hasattr(tok_arr, "array") else tok_arr
                for s in sorted(tok_arr.addressable_shards, key=lambda s: s.device.id):
                    r0, r1 = s.index[0].start or 0, s.index[0].stop if s.index[0].stop is not None else B
                    loc = np.asarray(s.data)
                    assert np.array_equal(loc, tokens[r0:r1]), "device shard != global rows"
                    per_device[str(s.device.id)] = dict(rows=[int(r0), int(r1)], token_sha256=common.sha256_array(loc.astype("<i4")),
                                                        loss_weight_sum=float(lw[r0:r1].astype(np.float64).sum()),
                                                        microbatch_rows={str(j): [max(r0, j * mb), min(r1, (j + 1) * mb)]
                                                                         for j in range(B // mb) if max(r0, j * mb) < min(r1, (j + 1) * mb)})
                common.write_json(os.path.join(out, "layout.json"), dict(
                    size=args.size, B=B, S=S, devices=n_dev, mesh=dict(trainer.device_mesh.shape), per_device_parallelism=pdp,
                    microbatch_size=mb, microbatches=B // mb,
                    microbatch_rows={str(j): [j * mb, (j + 1) * mb] for j in range(B // mb)},
                    per_device=per_device, leaves=leaf_info,
                    files=dict(tokens="tokens.npy int32 [B, S]", segment_ids=f"segment_ids.npy int32 [B, S] (leaf {seg_names[0]}; all segment leaves equal)",
                               loss_weight=f"loss_weight.npy {lw.dtype} [B, S]")))
        stop_iterator(it)
    ok = all(st["T0.2_hash_equal"] is not False and st["T0.3_segment_ids_equal_formula"] and st["T0.3_loss_weight_equal_formula"]
             for st in checks["steps"].values()) and checks["steps"]["0"]["T0.2_hash_equal"] is True
    checks["all_pass"] = ok
    common.write_json(os.path.join(out, "checks.json"), checks)
    common.write_manifest(out, checks=["T0.2", "T0.3", "T2 input (targets.md §4(a) fixture 2)", "T1 input rows 0-1"], env=common.environment(),
                          extra=dict(size=args.size, run_id=cfg0.trainer.id, all_pass=ok))
    log("all_pass", ok, "wrote", out)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip interpreter teardown: the loader's background thread can abort it after all files are written
