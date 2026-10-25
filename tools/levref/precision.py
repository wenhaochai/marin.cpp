"""T1.0 precision facts and T1.6 Newton-Schulz fixtures, from one full-N train step of the plain baseline.

    python -m tools.levref.precision --size 130m [--dump] [--ld-debug] [--name t1_precision] [--no-step]

On the baseline's 4 GPUs (any device count works; the facts are for 4):
  1. config.harness -> open_trainer -> init.build_initial_state (train_lm.py:268) -> batch 0 from batch.train_iterator
     (token sha256 checked against H(B, 0) of cache/data/data_ref.json). The MuonH values come from the built config and
     must equal the logged W&B config of the size (momentum 0.95 at 130m, 0.98 above; max_grad_norm 2.0 at 1_2b).
  2. G0 and the MuonH Nesterov update u of update 0: the step body of trainer.py:749-787 (grads from
     trainer._compute_gradients_microbatched, then take_train_step with the real optimizer) traced while
     levanter.optim.muonh.map_flattened_linear_layers and .norm_preserving_update are replaced (in memory only) by
     identities, so the optimizer returns u = momentum * buf + g (g clipped by the MuonH group's own global norm to
     max_grad_norm, buf = g at update 0) for every MuonH leaf. u is flattened with levanter.optim.util.flatten_linear_layers
     (the [OUT, IN] layout NS receives) and layer 0 of each projection is saved. That function is a separate jit; the real
     step below is traced with the real optimizer.
  3. The real step: trainer.train_step(state, batch 0) (the program of trainer.py:527-568; step 0 runs the jit-hook program
     when watch hooks are on). With --dump, XLA_FLAGS gets --xla_dump_to=<out>/xla_dump --xla_dump_hlo_as_text (appended
     after the run env of config.harness), and tools.levref.hlofacts reads the optimized HLO afterwards. The fused-CE
     implementation messages of levanter (selected implementation, fallback warnings) are captured into step.json.
  4. Outside the Trainer (no mesh, one device): levanter.optim.util.zeropower_via_newtonschulz5(X, steps, eps, coefficient_type)
     with the baseline's optimizer values (5, 1e-5, "simple"; asserted) on each saved u, compiled per GEMM emitter
     (autotuned = XLA's own choice, as in the step; triton = xla_gpu_cublas_fallback off; cublas = xla_gpu_enable_triton_gemm
     off) and per matmul precision (DEFAULT; jax.default_matmul_precision("tensorfloat32"); "highest" for the autotuned
     emitter). Recorded per matrix and emitter: bitwise DEFAULT == tensorfloat32, DEFAULT == highest, rel between them,
     and the GEMMs of each compiled module (implementation, operand precision, algorithm; hlofacts.gemm_table). The
     in-step NS dots are Triton nested GEMM fusions at DEFAULT precision (T1.0 dump): the triton emitter row measures
     exactly that kind.
  5. f32 GEMM probes x @ y with x, y normalised inside the jit (as NS feeds its dots) and returned: 512^3 with the Triton
     and with the cuBLAS emitter pinned, 2048^3 autotuned (XLA keeps that one on cuBLASLt). Per probe: DEFAULT vs highest
     vs tensorfloat32 vs float64 numpy, the compiled GEMM implementation, and a rounding fingerprint: rel(output, f64
     product of x, y rounded to tf32 (round-to-nearest-even or toward zero), bf16, or kept f32). The input format the GEMM
     used gives the smallest rel, at least 10x below the others. Then the shared libraries mapped into this process
     (/proc/self/maps) and, with --ld-debug, the dynamic loader's own log (the module re-executes itself once with
     LD_DEBUG=libs,files, LD_DEBUG_OUTPUT=<out>/ld_debug/ld; the child drops both variables so subprocesses do not log).
Writes cache/levref/<size>/<name>/: ns/ (TensorDir: u.<proj>, g.<proj>, ns_default.<proj>, ns_highest.<proj>,
ns_tensorfloat32.<proj>, ns_<prec>_<emitter>.<proj>), ns_hlo/ (compiled NS modules), step.json, libs.json, facts.json
(with --dump, from hlofacts), xla_dump/ (with --dump), ld_debug/ (with --ld-debug), manifest.json.
"""
import argparse
import os
import sys

_CHILD = "_LEVREF_LD_CHILD"
PROJ = (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"))
# per-compile XLA options that pin the GEMM emitter (None = XLA's autotuned choice, as in the train step)
EMITTERS = {"autotuned": None, "triton": {"xla_gpu_cublas_fallback": False}, "cublas": {"xla_gpu_enable_triton_gemm": False}}


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--dump", action="store_true", help="XLA dump of every module (T1.0)")
    ap.add_argument("--ld-debug", action="store_true", help="re-exec once under LD_DEBUG=libs,files")
    ap.add_argument("--name", default="t1_precision")
    ap.add_argument("--no-step", action="store_true", help="skip the real train step (u and NS only)")
    ap.add_argument("--debug-batch", type=int, default=0, help="DEBUG ONLY: shrink train_batch_size (fixture is then not a reference)")
    return ap.parse_args()


ARGS = _args() if __name__ == "__main__" else None

if ARGS is not None and ARGS.ld_debug and _CHILD not in os.environ:
    from tools.levref import common as _c

    _out = _c.fixture_dir(ARGS.size, ARGS.name)
    _ld = os.path.join(_out, "ld_debug")
    os.makedirs(_ld, exist_ok=True)
    for f in os.listdir(_ld):
        os.remove(os.path.join(_ld, f))
    env = dict(os.environ, LD_DEBUG="libs,files", LD_DEBUG_OUTPUT=os.path.join(_ld, "ld"), **{_CHILD: "1"})
    os.execve(sys.executable, [sys.executable, "-m", "tools.levref.precision", *sys.argv[1:]], env)
if os.environ.get(_CHILD):
    os.environ.pop("LD_DEBUG", None)  # ld.so read them at startup; children must not inherit them
    os.environ.pop("LD_DEBUG_OUTPUT", None)

import contextlib
import functools
import json
import logging
import re
import time

from tools.levref import batch as lbatch, common, config as lconfig, init as linit


def _rel(a, b) -> float:
    import numpy as np

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def _bits_equal(a, b) -> bool:
    import numpy as np

    return a.shape == b.shape and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32)))


def round_mantissa(x, keep_bits: int, mode: str):
    """float32 values -> float64 values rounded to `keep_bits` explicit mantissa bits (tf32: 10, bf16: 7), with
    round-to-nearest-even ("rne") or toward zero ("rz"). Finite inputs far from overflow only (true for the probes)."""
    import numpy as np

    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    drop = np.uint64(23 - keep_bits)
    mask = (np.uint64(1) << drop) - np.uint64(1)
    if mode == "rne":
        u = u + (mask >> np.uint64(1)) + ((u >> drop) & np.uint64(1))
    elif mode != "rz":
        raise ValueError(mode)
    u = (u & ~mask) & np.uint64(0xFFFFFFFF)
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def format_products(A, B) -> dict:
    """f64 products of A and B after rounding both inputs to each candidate GEMM input format."""
    import numpy as np

    fmt = {"f32": lambda x: np.asarray(x, np.float64), "tf32_rne": lambda x: round_mantissa(x, 10, "rne"),
           "tf32_rz": lambda x: round_mantissa(x, 10, "rz"), "bf16_rne": lambda x: round_mantissa(x, 7, "rne")}
    return {k: f(A) @ f(B) for k, f in fmt.items()}


def rounding_fingerprint(products: dict, out) -> dict:
    """rel(out, each format's product): the input format the GEMM used gives rel ~1e-7 (f32 accumulation only)."""
    return {k: _rel(out, p) for k, p in products.items()}


class _LogCapture(logging.Handler):
    def __init__(self, needles):
        super().__init__()
        self.needles, self.lines = needles, []

    def emit(self, record):
        msg = record.getMessage()
        if any(n in msg for n in self.needles):
            line = f"{record.name}: {msg}"[:400]
            if line not in self.lines:
                self.lines.append(line)


def ld_debug_summary(ld_dir: str) -> dict:
    """Per process log: every library the loader initialised whose name matches cuBLAS / cuDNN / NCCL / CUDA / the XLA plugin."""
    pat = re.compile(r"calling init: (\S+)")
    who = re.compile(r"file=(\S+) \[\d+\];\s+(needed by|dynamically loaded by) (\S+)")
    keep = re.compile(r"libcublas|libcudnn|libnccl|libcuda\.|libcudart|libnvJitLink|xla_cuda|libcusolver|libcufft|libcusparse|libnvrtc")
    out = {}
    for f in sorted(os.listdir(ld_dir)) if os.path.isdir(ld_dir) else []:
        libs, by = [], {}
        with open(os.path.join(ld_dir, f), errors="replace") as fh:
            for line in fh:
                m = pat.search(line)
                if m and keep.search(m.group(1)) and m.group(1) not in libs:
                    libs.append(m.group(1))
                w = who.search(line)
                if w and keep.search(w.group(1)):
                    by.setdefault(os.path.basename(w.group(1)), set()).add(f"{w.group(2)} {w.group(3)}")
        out[f] = [dict(path=p, realpath=os.path.realpath(p), bytes=os.path.getsize(p) if os.path.exists(p) else None,
                       loaded_because=sorted(by.get(os.path.basename(p), set()))) for p in libs]
    return out


def _logged_optimizer(size: str):
    """The optimizer block of the logged W&B config of the baseline run (cache/wandb/...config.json), or None."""
    p = os.path.join(common.nano_dir(), "cache", "wandb", f"muonh-qwen3-{size}-della4xh100.config.json")
    return json.load(open(p))["config"]["optimizer"] if os.path.exists(p) else None


def main():
    args = ARGS
    t0 = time.time()

    def log(*a):
        print(f"[levref.precision {time.time() - t0:7.1f}s]", *a, flush=True)

    out = common.fixture_dir(args.size, args.name)
    cap = _LogCapture(("Fused cross-entropy", "falling back", "batched_xla", "B-tiled", "b_tiled"))
    logging.captureWarnings(True)  # the fallback is a RuntimeWarning: route it through logging (py.warnings) to `cap`
    logging.getLogger().addHandler(cap)
    cfg0, cfg, scratch, applied = lconfig.harness(args.size, "precision")
    if args.debug_batch:
        import dataclasses

        cfg = dataclasses.replace(cfg, trainer=dataclasses.replace(cfg.trainer, train_batch_size=args.debug_batch, per_device_parallelism=-1))
    dump_dir = os.path.join(out, "xla_dump")
    if args.dump:
        if os.path.isdir(dump_dir):
            import shutil

            shutil.rmtree(dump_dir)
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={dump_dir} --xla_dump_hlo_as_text").strip()
        # a persistent-cache hit skips compilation, so the dump would lack config.pbtxt (flags) and autotuning: compile fresh
        cache_env = os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        log("JAX persistent compilation cache disabled for the dump (was", cache_env, ")")
    log("XLA_FLAGS", os.environ.get("XLA_FLAGS"), "applied run env", applied)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import levanter.optim.muonh as muonh_mod
    from haliax.partitioning import named_jit
    from levanter.optim.util import flatten_linear_layers, zeropower_via_newtonschulz5
    from levanter.trainer_state import take_train_step
    from levanter.utils.tree_utils import inference_mode
    from tools.levref import hlofacts

    if args.dump:  # config.harness imported jax already, so the env var alone does not reach jax.config
        jax.config.update("jax_enable_compilation_cache", False)
        jax.config.update("jax_compilation_cache_dir", None)
    opt = cfg.optimizer
    # the NS call of the baseline at every size (asserted); momentum and max_grad_norm differ per size and must equal the
    # logged run's values (130m: 0.95 / 1.0; 300m, 520m: 0.98 / 1.0; 1_2b: 0.98 / 2.0)
    assert (opt.backend_steps, opt.muon_epsilon, opt.coefficient_type, opt.nesterov) == (5, 1e-5, "simple", True), opt
    logged = _logged_optimizer(args.size)
    if logged is not None:
        assert (float(opt.momentum), float(opt.max_grad_norm)) == (float(logged["momentum"]), float(logged["max_grad_norm"])), \
            (opt.momentum, opt.max_grad_norm, logged["momentum"], logged["max_grad_norm"])
    step: dict = dict(size=args.size, devices=None, debug_batch=args.debug_batch or None, momentum=float(opt.momentum),
                      max_grad_norm=float(opt.max_grad_norm), logged_optimizer_checked=logged is not None)
    td = common.TensorDir(os.path.join(out, "ns"))
    with lconfig.open_trainer(cfg) as trainer:
        for lname in ("", "levanter", "py.warnings", "levanter.kernels.pallas.fused_cross_entropy_loss.api"):
            lg = logging.getLogger(lname)  # trainer setup may reset the root logger's handlers: attach again
            if cap not in lg.handlers:
                lg.addHandler(cap)
        state, Vocab, model_key, keys = linit.build_initial_state(cfg, trainer)
        digest, _ = linit.model_digests(state.model)
        ref_idx = os.path.join(common.nano_dir(), "cache", "levref", args.size, "init", "index.json")
        ref_digest = json.load(open(ref_idx))["summary"]["combined_sha256"] if os.path.exists(ref_idx) else None
        step.update(devices=jax.device_count(), device_kind=jax.devices()[0].device_kind, mesh=dict(trainer.device_mesh.shape),
                    per_device_parallelism=cfg.trainer.per_device_parallelism, microbatch_size=cfg.trainer.microbatch_size,
                    train_batch_size=cfg.trainer.train_batch_size, num_train_steps=cfg.trainer.num_train_steps,
                    init_combined_sha256=digest, init_fixture_combined_sha256=ref_digest,
                    init_equal_to_fixture=None if ref_digest is None else digest == ref_digest)
        log("init", step)
        it = lbatch.train_iterator(cfg, trainer, 0)
        ex = next(it)
        lbatch.stop_iterator(it)
        B = cfg.trainer.train_batch_size
        sha = lbatch.tokens_sha256(ex)
        ref = json.load(open(os.path.join(common.nano_dir(), "cache", "data", "data_ref.json")))["batch_hashes"].get(f"B{B}_step0", {}).get("sha256")
        step["batch0"] = dict(sha256=sha, ref_sha256=ref, equal=sha == ref)
        assert sha == ref or args.debug_batch, (sha, ref)

        # ---- 2. G0 -> u of update 0 (trainer.py:749-760 + trainer_state.take_train_step), MuonH NS and projection bypassed
        def g_and_u(state, batch):
            key, _new_key = jax.random.split(state.training_key)
            model = inference_mode(state.model, False)
            loss, grads, _metrics = trainer._compute_gradients_microbatched(trainer.loss_fn, model, batch, key=key)
            _m, _o, updates = take_train_step(state.optimizer, state.model, state.opt_state, grads, obj_fun=None, loss=loss,
                                              is_trainable=state.is_trainable)
            flat = flatten_linear_layers(updates.transformer)
            stacked = flat.layers.stacked
            mats = {}
            for mod, name in PROJ:
                w = getattr(getattr(stacked, mod), name).weight
                mats[name] = w.array[0]
            gflat = flatten_linear_layers(grads.transformer).layers.stacked
            gmats = {name: getattr(getattr(gflat, mod), name).weight.array[0] for mod, name in PROJ}
            return loss, mats, gmats

        real = (muonh_mod.map_flattened_linear_layers, muonh_mod.norm_preserving_update)
        muonh_mod.map_flattened_linear_layers = lambda f, updates, **kw: updates
        muonh_mod.norm_preserving_update = lambda p, u, lr: u
        try:
            loss_u, mats, gmats = named_jit(g_and_u, axis_resources=trainer.parameter_axis_mapping)(state, ex)
            mats = {k: np.asarray(jax.device_get(v)) for k, v in mats.items()}
            gmats = {k: np.asarray(jax.device_get(v)) for k, v in gmats.items()}
        finally:
            muonh_mod.map_flattened_linear_layers, muonh_mod.norm_preserving_update = real
        step["loss_g_and_u"] = float(loss_u)
        # u = f32(momentum * g_clipped) + g_clipped at update 0: the implied MuonH-group clip factor is |u| / ((1 + momentum) |g|)
        step["u_over_1pm_g_layer0"] = {k: float(np.linalg.norm(mats[k].astype(np.float64))
                                                / ((1.0 + float(opt.momentum)) * np.linalg.norm(gmats[k].astype(np.float64)))) for k in mats}
        log("u captured", {k: v.shape for k, v in mats.items()}, "loss", step["loss_g_and_u"])

        # ---- 3. the real step (T1.0 program)
        if not args.no_step:
            hooks_this_time = any(state.step % h.every == 0 for h in trainer.hooks.jit_hooks)
            info = trainer.train_step(state, ex)
            step["train_step"] = dict(loss=float(info.loss), program="train_step (jit hooks)" if hooks_this_time else "train_step_hooks (no jit hooks)",
                                      jit_hooks=len(trainer.hooks.jit_hooks), new_step=int(info.state.step))
            log("train step", step["train_step"])
            del info
    step["ce_implementation_log"] = list(cap.lines)
    log("CE implementation log", cap.lines)

    # ---- 4. Newton-Schulz on one device, no mesh: per emitter x matmul precision
    dev0 = jax.devices()[0]
    ns = functools.partial(zeropower_via_newtonschulz5, steps=opt.backend_steps, eps=opt.muon_epsilon, coefficient_type=opt.coefficient_type)
    hlo_dir = os.path.join(out, "ns_hlo")
    os.makedirs(hlo_dir, exist_ok=True)
    applied_opts: dict = {}

    def compiled(fn, args_, prec, emitter):
        """(Compiled, options used) of jit(fn) lowered under matmul precision `prec` (None = DEFAULT) for `emitter`."""
        with (jax.default_matmul_precision(prec) if prec else contextlib.nullcontext()):
            lowered = jax.jit(fn).lower(*args_)
        opts = dict(EMITTERS[emitter] or {})
        while True:
            try:
                return (lowered.compile(compiler_options=opts) if opts else lowered.compile()), opts
            except Exception as e:  # an option this XLA does not know: drop it, record what was applied
                bad = re.search(r"No such compile option: '?(\w+)", str(e))
                if not bad or bad.group(1) not in opts:
                    raise
                opts.pop(bad.group(1))

    gpu = jax.default_backend() == "gpu"
    emitters = list(EMITTERS) if gpu else ["autotuned"]
    ns_res = {}
    for name, X in mats.items():
        Xd = jax.device_put(jnp.asarray(X), dev0)
        res = dict(shape=list(X.shape), transposed_inside=X.shape[0] > X.shape[1], emitters={})
        outs = {}
        for em in emitters:
            precs = ("default", "tensorfloat32", "highest") if em == "autotuned" else ("default", "tensorfloat32")
            e: dict = dict(gemm={})
            for p in precs:
                comp, opts = compiled(ns, (Xd,), None if p == "default" else p, em)
                applied_opts[em] = opts
                text = comp.as_text()
                with open(os.path.join(hlo_dir, f"{name}.{em}.{p}.txt"), "w") as fh:
                    fh.write(text)
                e["gemm"][p] = hlofacts.gemm_summary(hlofacts.gemm_table(text))
                outs[(em, p)] = np.asarray(jax.device_get(comp(Xd)))
            d = outs[(em, "default")]
            e.update(default_bitwise_equal_tensorfloat32=_bits_equal(d, outs[(em, "tensorfloat32")]),
                     rel_default_vs_tensorfloat32=_rel(d, outs[(em, "tensorfloat32")]))
            if (em, "highest") in outs:
                e.update(default_bitwise_equal_highest=_bits_equal(d, outs[(em, "highest")]), rel_default_vs_highest=_rel(d, outs[(em, "highest")]))
            if em != "autotuned":
                e["default_bitwise_equal_autotuned_default"] = _bits_equal(d, outs[("autotuned", "default")])
            res["emitters"][em] = e
        a = res["emitters"]["autotuned"]
        res.update(rel_default_vs_highest=a["rel_default_vs_highest"], bitwise_equal=a["default_bitwise_equal_highest"])
        td.save(f"u.{name}", X, axes=["__OUT__", "__IN__"])
        td.save(f"g.{name}", gmats[name], axes=["__OUT__", "__IN__"])
        for (em, p), o in outs.items():
            td.save(f"ns_{p}.{name}" if em == "autotuned" else f"ns_{p}_{em}.{name}", o, axes=["__OUT__", "__IN__"])
        ns_res[name] = res
        log("NS", name, {em: {k: v for k, v in e.items() if k != "gemm"} for em, e in res["emitters"].items()})
    td.write_index(part="T1.6 Newton-Schulz", size=args.size, call=f"zeropower_via_newtonschulz5(X, steps={opt.backend_steps}, "
                   f"eps={opt.muon_epsilon}, coefficient_type={opt.coefficient_type!r})", layer=0,
                   momentum=float(opt.momentum), max_grad_norm=float(opt.max_grad_norm), emitter_compile_options=applied_opts,
                   u=f"momentum * buf + g with buf = g (update 0), momentum = {opt.momentum}, g = clip_by_global_norm({opt.max_grad_norm}) "
                     "of the muonh group (each group clipped by its own norm); g.* is the raw G0",
                   outputs="ns_default / ns_highest / ns_tensorfloat32: XLA's autotuned emitter (as in the step); "
                           "ns_<prec>_triton / ns_<prec>_cublas: emitter pinned by compile options")
    step["newton_schulz"] = ns_res
    step["emitter_compile_options"] = applied_opts

    # ---- 5. f32 GEMM probes per emitter, libraries. The operands come out of a fusion (normalised, as in Newton-Schulz) and
    # are returned, so the fingerprint uses their exact values; XLA makes the 512^3 dot a Triton nested GEMM fusion (the kind
    # of the in-step NS and flash dV dots) and keeps a 2048^3 dot on cuBLASLt (A100 debug, 2026-10-24).
    def probe_fn(a, b):
        x = a / (jnp.linalg.norm(a) + 1e-5)
        y = b / (jnp.linalg.norm(b) + 1e-5)
        return x, y, x @ y

    probes = {}
    plan = ({"512_triton": (512, "triton"), "512_cublas": (512, "cublas"), "2048_autotuned": (2048, "autotuned")} if gpu
            else {"2048_autotuned": (2048, "autotuned")})
    for tag, (n, em) in plan.items():
        ka, kb = jax.random.split(jax.random.PRNGKey(7))
        A = jax.random.normal(ka, (n, n), jnp.float32)
        Bm = jax.random.normal(kb, (n, n), jnp.float32)
        o, gem, products = {}, {}, None
        for p in ("default", "highest", "tensorfloat32"):
            comp, _ = compiled(probe_fn, (A, Bm), None if p == "default" else p, em)
            x, y, o[p] = (np.asarray(t) for t in comp(A, Bm))
            gem[p] = hlofacts.gemm_summary(hlofacts.gemm_table(comp.as_text()))
            products = products or format_products(x, y)
        ref64 = products["f32"]
        probes[tag] = dict(shape=[n, n, n], emitter=em, rel_default_vs_f64=_rel(o["default"], ref64),
                           rel_highest_vs_f64=_rel(o["highest"], ref64), rel_tensorfloat32_vs_f64=_rel(o["tensorfloat32"], ref64),
                           default_bitwise_equal_tensorfloat32=_bits_equal(o["default"], o["tensorfloat32"]),
                           default_bitwise_equal_highest=_bits_equal(o["default"], o["highest"]),
                           rounding_fingerprint_default=rounding_fingerprint(products, o["default"]),
                           rounding_fingerprint_tensorfloat32=rounding_fingerprint(products, o["tensorfloat32"]),
                           rounding_fingerprint_highest=rounding_fingerprint(products, o["highest"]),
                           gemm=gem["default"], gemm_by_precision=gem,
                           note="fingerprint: rel(output, f64 product of the operands rounded to that format); the format the GEMM "
                                "used gives the smallest rel, >= 10x below the others (A100: tf32_rne 7e-6..1e-5 vs f32 2.9e-4)")
        log("gemm probe", tag, {k: v for k, v in probes[tag].items() if k not in ("gemm_by_precision",)})
    step["f32_gemm_probe"] = probes["2048_autotuned"]
    step["f32_gemm_probe_by_emitter"] = probes
    step["default_matmul_precision"] = str(jax.config.jax_default_matmul_precision)
    common.write_json(os.path.join(out, "step.json"), step)
    libs = dict(proc_self_maps=common.loaded_libraries())
    if os.environ.get(_CHILD):
        libs["ld_debug"] = ld_debug_summary(os.path.join(out, "ld_debug"))
    for d in libs["proc_self_maps"]:  # the sonames the XLA GPU plugin itself links against (which cuBLAS/cuDNN/NCCL XLA calls)
        if "xla_cuda" in os.path.basename(d.get("path", "")):
            try:
                import subprocess

                dyn = subprocess.run(["readelf", "-d", d["path"]], capture_output=True, text=True, check=True).stdout
                libs["xla_plugin_needed"] = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
                libs["xla_plugin_runpath"] = re.findall(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]+)\]", dyn)
            except Exception as e:
                libs["xla_plugin_needed"] = f"ERROR {e!r}"
    common.write_json(os.path.join(out, "libs.json"), libs)
    log("libs", [os.path.basename(d.get("path", "")) for d in libs["proc_self_maps"]])

    checks = ["T1.6"]
    if args.dump:
        facts = hlofacts.analyze(dump_dir, hidden=cfg.model.hidden_dim, head_dim=cfg.model.attention_config().HeadSize.size,
                                 vocab=Vocab.size, libs=libs, probe=step["f32_gemm_probe_by_emitter"], ns=ns_res,
                                 ce_log=step["ce_implementation_log"])
        common.write_json(os.path.join(out, "facts.json"), facts)
        checks.insert(0, "T1.0")
        log("facts", json.dumps(facts.get("summary", {}), indent=1)[:6000])
    common.write_manifest(out, checks=checks, env=common.environment(),
                          extra=dict(size=args.size, run_id=cfg0.trainer.id, num_train_steps=cfg0.trainer.num_train_steps,
                                     applied_run_env=applied, step=step))
    log("wrote", out)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)
