"""T1.2-T1.5 fixtures: levanter's RMSNorm, RoPE, flash attention and fused cross-entropy on the tensors that the
levanter bf16 forward of the seed-0 init feeds them, with fixed bf16 cotangents.

    python -m tools.levref.units --size 130m [--parts norm,rope,attn,ce] [--name t1_units]

Runs on ONE device (T1 is a 1-GPU check). In process: config.harness -> open_trainer -> init.build_initial_state
(train_lm.py:268, its combined sha256 is compared with cache/levref/<size>/init/index.json when that exists) ->
batch 0 from batch.train_iterator (its token sha256 must equal H(B, 0) of cache/data/data_ref.json) ->
model_c = trainer.mp.cast_to_compute(model) (the bf16 copy the loss function sees, trainer.py:253-254).

Inputs (all from levanter's own modules, under trainer.compute_axis_mapping):
  rows 0-1 of batch 0, layer 0:  x0 = embeddings.embed(tokens); h = input_layernorm(x0); q, k, v = q_proj/k_proj/v_proj(h);
                                 qn, kn = q_norm(q), k_norm(k); qr, kr = rot_embs(qn|kn, arange(4096)).astype(bf16)
                                 (Attention._compute_qkv, attention.py:1422-1444); then the layout Attention.__call__ gives the
                                 kernel (kv_head, q_heads_per_group, position, head_size), keys renamed key_position
  rows 0-31 of batch 0 (GPU 0's rows of microbatch 0 on 4 GPUs, see batch0/layout.json):
                                 x = model_c.activations(tokens, attn_mask) (final-normed, bf16), W = model_c.get_lm_head()
Cotangents: jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(1234), i), shape, jnp.bfloat16), i fixed per use (COT).

Parts (each a TensorDir under cache/levref/<size>/<name>/<part>/ with index.json and results.json):
  norm  T1.2  input_layernorm on x0 (over D) and q_norm / k_norm on q / k (over hd), each under eqx.filter_vjp:
              out, dx, dg (cotangent of the bf16 gain), with x, g (bf16 gain), dy
  rope  T1.3  inv_freq (f32: levanter's expression run eagerly on this device, checked against the constants closed over in
              the traced call), cos / sin tables [4096, hd] bf16 recovered from DefaultRotaryEmbeddings.__call__ itself with two
              one-half-ones probes, and the rotated q / k of layer 0 (qn, kn -> qr, kr)
  attn  T1.4  dot_product_attention exactly as Attention.__call__ calls it (JAX_FLASH, block 1024, upcast_attn=False, segment
              ids of rows 0-1) under eqx.filter_vjp: O, dq, dk, dv; plus the kernel level: q_scaled = q * (1/sqrt(hd)) in bf16,
              the log-normalizer L from _flash_attention_forward, and dq_scaled from the vjp of _flash_attention
  ce    T1.5  maybe_fused_next_token_loss(reduction=None, loss_weight) on x, W (H100: batched_xla B-tiled, 16 tiles of 8192 rows)
              under eqx.filter_vjp of sum(loss_tok * w) / 524160 (the cotangent each token gets in the real 4-GPU step):
              per-token loss (f32, weighted: position 4095 is 0), grad_x, grad_w, with x, W, labels, weights
f64 oracles and rel(lev, oracle) come from tools.levref.oracle (numpy) and go to cache/levref/<size>/t1_oracle/.
"""
import argparse
import json
import logging
import math
import os
import sys
import time

from tools.levref import batch as lbatch, common, config as lconfig, init as linit

DEFAULT_PARTS = {"130m": ("norm", "rope", "attn", "ce"), "300m": ("norm", "rope", "attn", "ce"), "520m": ("attn",),
                 "1_2b": ("norm", "rope", "attn", "ce")}
COT = dict(norm_block=0, norm_q=1, norm_k=2, attn_dO=3)  # fold_in index of each cotangent draw
COT_SEED = 1234
MICROBATCH_WEIGHT_SUM = 128 * 4095  # sum of loss weights over one 128-sequence microbatch: the global CE denominator


class _LogCapture(logging.Handler):
    def __init__(self, needle):
        super().__init__()
        self.needle, self.lines = needle, []

    def emit(self, record):
        msg = record.getMessage()
        if any(n in msg for n in self.needle):
            line = f"{record.name}: {msg}"
            if line not in self.lines:
                self.lines.append(line)


def _named(arr):
    return list(a.name for a in arr.axes)


def _bf16_hex(x: float) -> str:
    import jax.numpy as jnp
    import numpy as np

    return "0x%04X" % int(np.asarray(jnp.asarray(x, dtype=jnp.bfloat16)).view(np.uint16))


def _cot(i, like):
    import jax
    import jax.numpy as jnp
    import haliax as hax

    a = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(COT_SEED), i), like.array.shape, jnp.bfloat16)
    return hax.named(a, like.axes)


def _slice_rows(tree, Batch, r0, r1):
    import haliax as hax
    import jax

    def cut(a):
        if isinstance(a, hax.NamedArray) and Batch in [ax.name for ax in a.axes]:
            return a[Batch, r0:r1]
        return a

    out = jax.tree.map(cut, tree, is_leaf=lambda a: isinstance(a, hax.NamedArray))
    sizes = {ax.size for a in jax.tree.leaves(out, is_leaf=lambda a: isinstance(a, hax.NamedArray))
             if isinstance(a, hax.NamedArray) for ax in a.axes if ax.name == Batch}
    assert sizes == {r1 - r0}, (sizes, r0, r1)
    return out


def _eq(a, b) -> bool:
    import jax
    import numpy as np

    a = np.asarray(jax.device_get(getattr(a, "array", a)))
    b = np.asarray(jax.device_get(getattr(b, "array", b)))
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=lconfig.SIZES)
    ap.add_argument("--parts", default=None, help="comma list of norm,rope,attn,ce (default per size: %s)" % DEFAULT_PARTS)
    ap.add_argument("--name", default="t1_units")
    ap.add_argument("--ce-rows", type=int, default=32, help="rows of batch 0 for the CE fixture (32 = one GPU's 131072 tokens)")
    args = ap.parse_args()
    parts = tuple(args.parts.split(",")) if args.parts else DEFAULT_PARTS[args.size]
    t0 = time.time()

    def log(*a):
        print(f"[levref.units {time.time() - t0:7.1f}s]", *a, flush=True)

    out = common.fixture_dir(args.size, args.name)
    cap = _LogCapture(("Fused cross-entropy", "falling back", "batched_xla", "B-tiled", "b_tiled"))
    logging.getLogger().addHandler(cap)

    cfg0, cfg, scratch, applied = lconfig.harness(args.size, "units")

    import equinox as eqx
    import haliax as hax
    import jax
    import jax.numpy as jnp
    import numpy as np

    results: dict = dict(size=args.size, parts=list(parts), applied_run_env=applied)
    with lconfig.open_trainer(cfg) as trainer:
        assert jax.device_count() == 1, f"T1 units run on one device; this process sees {jax.device_count()}"
        state, Vocab, model_key, keys = linit.build_initial_state(cfg, trainer)
        digest, _ = linit.model_digests(state.model)
        ref_idx = os.path.join(common.nano_dir(), "cache", "levref", args.size, "init", "index.json")
        ref_digest = json.load(open(ref_idx))["summary"]["combined_sha256"] if os.path.exists(ref_idx) else None
        results["init"] = dict(combined_sha256=digest, init_fixture_combined_sha256=ref_digest,
                               equal_to_init_fixture=None if ref_digest is None else digest == ref_digest)
        log("init built; combined sha", digest[:16], "init fixture", (ref_digest or "absent")[:16])

        it = lbatch.train_iterator(cfg, trainer, 0)
        ex = next(it)
        lbatch.stop_iterator(it)
        B = cfg.trainer.train_batch_size
        sha = lbatch.tokens_sha256(ex)
        ref = json.load(open(os.path.join(common.nano_dir(), "cache", "data", "data_ref.json")))["batch_hashes"].get(f"B{B}_step0", {}).get("sha256")
        results["batch0"] = dict(B=B, sha256=sha, ref_sha256=ref, equal=sha == ref)
        assert sha == ref, (sha, ref)
        Batch = trainer.config.batch_axis_name or "batch"
        ex2 = _slice_rows(ex, Batch, 0, 2)
        exce = _slice_rows(ex, Batch, 0, args.ce_rows)

        mapping = trainer.compute_axis_mapping
        with hax.axis_mapping(mapping):
            model_c = trainer.mp.cast_to_compute(state.model)
        del state
        m = cfg.model
        attn_cfg = m.attention_config()
        hd = attn_cfg.HeadSize.size

        @eqx.filter_jit
        def layer0(model_c, ex2):
            with hax.axis_mapping(mapping):
                x0 = model_c.embeddings.embed(ex2.tokens)
                layer = model_c.transformer.layers.get_layer(0)
                h = layer.input_layernorm(x0)
                attn = layer.self_attn
                q, k, v = attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)
                qn, kn = attn.q_norm(q), attn.k_norm(k)
                pos = hax.arange(x0.resolve_axis("position"))
                qr = attn.rot_embs(qn, pos).astype(qn.dtype)
                kr = attn.rot_embs(kn, pos).astype(kn.dtype)
                qkv_ref = attn._compute_qkv(h, key=None, pos_ids=None)
                return dict(x0=x0, h=h, q=q, k=k, v=v, qn=qn, kn=kn, qr=qr, kr=kr, qkv_ref=qkv_ref)

        cap0 = layer0(model_c, ex2)
        layer = model_c.transformer.layers.get_layer(0)
        attn_mod = layer.self_attn
        results["layer0"] = dict(
            dtypes={k: str(v.dtype) for k, v in cap0.items() if k != "qkv_ref"},
            stepwise_equals_compute_qkv=dict(q=_eq(cap0["qr"], cap0["qkv_ref"][0]), k=_eq(cap0["kr"], cap0["qkv_ref"][1]),
                                             v=_eq(cap0["v"], cap0["qkv_ref"][2])))
        log("layer 0 captured", results["layer0"])

        # ------------------------------------------------------------------ T1.2 RMSNorm
        if "norm" in parts:
            td = common.TensorDir(os.path.join(out, "norm"))
            res = {}

            @eqx.filter_jit
            def norm_vjp(norm, x, dy):
                with hax.axis_mapping(mapping):
                    y, f = eqx.filter_vjp(lambda n, x: n(x), norm, x)
                    dnorm, dx = f(dy)
                    return y, dx, dnorm.weight

            for tag, mod, x, ci in (("block", layer.input_layernorm, cap0["x0"], COT["norm_block"]),
                                    ("q", attn_mod.q_norm, cap0["q"], COT["norm_q"]),
                                    ("k", attn_mod.k_norm, cap0["k"], COT["norm_k"])):
                dy = _cot(ci, x)
                y, dx, dg = norm_vjp(mod, x, dy)
                ax = mod.axis if isinstance(mod.axis, (tuple, list)) else (mod.axis,)
                axis = [getattr(a, "name", a) for a in ax]
                for n, a in (("x", x), ("g", mod.weight), ("dy", dy), ("out", y), ("dx", dx), ("dg", dg)):
                    td.save(f"{tag}.{n}", np.asarray(jax.device_get(a.array)), axes=_named(a))
                res[tag] = dict(module=type(mod).__name__, norm_axis=axis, eps=float(mod.eps), compute_dtype=str(mod.dtype),
                                gain_dtype=str(mod.weight.dtype), cotangent=f"fold_in(PRNGKey({COT_SEED}), {ci})",
                                n_positions=int(np.prod(x.array.shape)) // int(mod.weight.array.size))
            td.write_index(part="T1.2 RMSNorm", size=args.size)
            common.write_json(os.path.join(td.path, "results.json"), res)
            log("norm done", res)

        # ------------------------------------------------------------------ T1.3 RoPE
        if "rope" in parts:
            td = common.TensorDir(os.path.join(out, "rope"))
            rot = attn_mod.rot_embs
            HeadDim, rcfg = rot.HeadDim, rot.config
            S = cap0["x0"].resolve_axis("position").size
            with jax.ensure_compile_time_eval():  # the expression of DefaultRotaryEmbeddings.__call__ (rotary.py:35-38)
                HeadHalfSize = HeadDim.resize(HeadDim.size // 2)
                inv_freq = 1.0 / (rcfg.theta ** (hax.arange(HeadHalfSize, step=2) / HeadDim.size))
                inv_freq = inv_freq / rcfg.factor
            inv = np.asarray(jax.device_get(inv_freq.array))
            Pos = hax.Axis("position", S)
            pos = hax.arange(Pos)
            probe_shape = hax.zeros((Pos, HeadDim), jnp.bfloat16)
            closed = jax.make_jaxpr(lambda q, p: rot(q, p).array)(probe_shape, pos)
            consts = [np.asarray(c) for c in closed.consts if np.ndim(c) == 1 and np.shape(c)[0] == HeadDim.size // 2]
            half = HeadDim.size // 2
            ones_lo = hax.named(jnp.concatenate([jnp.ones((S, half), jnp.bfloat16), jnp.zeros((S, half), jnp.bfloat16)], 1), (Pos, HeadDim))
            ones_hi = hax.named(jnp.concatenate([jnp.zeros((S, half), jnp.bfloat16), jnp.ones((S, half), jnp.bfloat16)], 1), (Pos, HeadDim))
            rot_j = eqx.filter_jit(lambda q, p: rot(q, p))
            A = np.asarray(jax.device_get(rot_j(ones_lo, pos).array))
            Bp = np.asarray(jax.device_get(rot_j(ones_hi, pos).array))
            # A = [cos_lo, sin_hi], Bp = [-sin_lo, cos_hi] (q*cos + rotate_half(q)*sin with q one-half ones; all adds exact)
            cos = np.concatenate([A[:, :half], Bp[:, half:]], 1)
            sin = np.concatenate([-Bp[:, :half], A[:, half:]], 1)
            sin = np.where(sin == 0, np.zeros_like(sin), sin)  # sin(0) is +0; the probe's -(+0 + -0) can give -0
            # the same tables by the direct expression (rotary.py:40-43), as a cross-check
            @eqx.filter_jit
            def tables(p):
                freqs = inv_freq.broadcast_axis(p.axes) * p
                emb = hax.concatenate(HeadDim, (freqs, freqs))
                return hax.cos(emb).astype(jnp.bfloat16).array, hax.sin(emb).astype(jnp.bfloat16).array, emb.array

            cos_d, sin_d, emb = (np.asarray(jax.device_get(a)) for a in tables(pos))
            td.save("inv_freq", inv, axes=["head_size_half"])
            td.save("cos", cos, axes=["position", "head_size"])
            td.save("sin", sin, axes=["position", "head_size"])
            td.save("angles_f32", emb, axes=["position", "head_size"])
            for n in ("qn", "kn", "qr", "kr"):
                td.save(n, np.asarray(jax.device_get(cap0[n].array)), axes=_named(cap0[n]))
            # numpy f32 build of numerics.md §3.3, for information (inv_freq must match levanter's bitwise in the port)
            f32 = np.float32
            e = (np.arange(0, HeadDim.size, 2, dtype=np.int32).astype(f32) / f32(HeadDim.size)).astype(f32)
            inv_np = (f32(1.0) / (f32(rcfg.theta) ** e).astype(f32)).astype(f32)
            res = dict(theta=rcfg.theta, factor=rcfg.factor, head_dim=HeadDim.size, positions=S,
                       inv_freq_dtype=str(inv.dtype), closed_over_inv_freq_consts=len(consts),
                       closed_over_consts_equal=[bool(np.array_equal(c.astype(np.float32).view(np.uint32), inv.view(np.uint32))) for c in consts],
                       probe_tables_equal_direct=dict(cos=bool(np.array_equal(cos.view(np.uint16), cos_d.view(np.uint16))),
                                                      sin=bool(np.array_equal(sin.view(np.uint16), sin_d.view(np.uint16)))),
                       numpy_f32_inv_freq_bitwise_equal=bool(np.array_equal(inv_np.view(np.uint32), inv.view(np.uint32))),
                       numpy_f32_inv_freq_max_ulp=int(np.abs(inv_np.view(np.int32).astype(np.int64) - inv.view(np.int32).astype(np.int64)).max()),
                       rotated_from="layer 0, rows 0-1: qr = rot_embs(qn, arange(S)).astype(bf16), same for k")
            td.write_index(part="T1.3 RoPE", size=args.size)
            common.write_json(os.path.join(td.path, "results.json"), res)
            log("rope done", res)

        # ------------------------------------------------------------------ T1.4 attention
        if "attn" in parts:
            from levanter.layers.attention import AttentionMask, dot_product_attention
            from levanter.models import flash_attention as FA

            td = common.TensorDir(os.path.join(out, "attn"))
            q = cap0["qr"].rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
            k = cap0["kr"].rearrange((..., "kv_head", "position", "head_size")).rename({"position": "key_position"})
            v = cap0["v"].rearrange((..., "kv_head", "position", "head_size")).rename({"position": "key_position"})
            mask = ex2.attn_mask
            if attn_cfg.sliding_window is not None and isinstance(mask, AttentionMask):
                mask = mask.with_sliding_window(attn_cfg.sliding_window)
            adtype = jnp.float32 if attn_cfg.upcast_attn else cap0["h"].dtype
            scale = attn_cfg.scaling_factor if attn_cfg.scaling_factor is not None else 1 / math.sqrt(hd)
            block = attn_cfg.flash_attention_block_size or FA.BLOCK_SIZE

            def attn_call(q, k, v):  # attention.py, Attention.__call__
                return dot_product_attention("position", "key_position", "head_size", q, k, v, mask, attention_dtype=adtype,
                                             attn_backend=attn_cfg.attn_backend, flash_block_size=attn_cfg.flash_attention_block_size,
                                             scaling_factor=attn_cfg.scaling_factor, logits_soft_cap=attn_cfg.logits_soft_cap,
                                             inference=True, prng=None)

            QPos, KPos = q.resolve_axis("position"), k.resolve_axis("key_position")

            def kernel(qs, k, v):  # flash_attention.py:98-111, after `q = q * scaling_factor`
                return FA._flash_attention((qs, k, v, None), QPos, KPos, "head_size", mask, None, 0.0, inference=True, key=None,
                                           block_size=block, precision=None, logits_soft_cap=attn_cfg.logits_soft_cap)

            @eqx.filter_jit
            def attn_all(q, k, v, dO):
                with hax.axis_mapping(mapping):
                    o, f = eqx.filter_vjp(attn_call, q, k, v)
                    dq, dk, dv = f(dO)
                    qs = q * scale
                    o2, (_, ell) = FA._flash_attention_forward(None, (qs, k, v, None), QPos, KPos, "head_size", mask, None, 0.0,
                                                               inference=True, key=None, block_size=block, precision=None,
                                                               logits_soft_cap=attn_cfg.logits_soft_cap)
                    o3, g = eqx.filter_vjp(kernel, qs, k, v)
                    dqs, dk3, dv3 = g(dO)
                    return dict(O=o, dq=dq, dk=dk, dv=dv, q_scaled=qs, L=ell, O_fwd=o2, O_kernel=o3, dq_scaled=dqs, dk_kernel=dk3,
                                dv_kernel=dv3, dq_from_scaled=dqs * scale)

            dO = _cot(COT["attn_dO"], cap0["qr"].rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size")))
            r = attn_all(q, k, v, dO)
            seg = mask.segment_ids if isinstance(mask, AttentionMask) else None
            seg_q, seg_kv = (seg if isinstance(seg, tuple) else (seg, seg)) if seg is not None else (None, None)
            for n, a in (("q", q), ("k", k), ("v", v), ("dO", dO), ("O", r["O"]), ("L", r["L"]), ("dq", r["dq"]), ("dk", r["dk"]),
                         ("dv", r["dv"]), ("q_scaled", r["q_scaled"]), ("dq_scaled", r["dq_scaled"])):
                td.save(n, np.asarray(jax.device_get(a.array)), axes=_named(a))
            td.save("segment_ids_q", np.asarray(jax.device_get(seg_q.array)).astype(np.int32), axes=_named(seg_q))
            td.save("segment_ids_kv", np.asarray(jax.device_get(seg_kv.array)).astype(np.int32), axes=_named(seg_kv))
            sq = np.asarray(jax.device_get(seg_q.array))
            res = dict(backend=str(attn_cfg.attn_backend), block_size=block, upcast_attn=attn_cfg.upcast_attn, attention_dtype=str(jnp.dtype(adtype)),
                       head_dim=hd, kv_heads=attn_cfg.KVHeads.size, q_heads_per_group=attn_cfg.QHeadsPerGroup.size,
                       scaling_factor=scale, scale_bf16_hex=_bf16_hex(scale), scale_bf16_value=float(jnp.asarray(scale, jnp.bfloat16)),
                       mask=dict(is_causal=bool(getattr(mask, "is_causal", False)), segments_per_row=[int(s.max()) + 1 for s in sq],
                                 sliding_window=attn_cfg.sliding_window),
                       L_dtype=str(r["L"].dtype), cotangent=f"fold_in(PRNGKey({COT_SEED}), {COT['attn_dO']})",
                       checks=dict(O_equals_kernel_forward=_eq(r["O"], r["O_fwd"]), O_equals_kernel_vjp_primal=_eq(r["O"], r["O_kernel"]),
                                   dk_equals_kernel=_eq(r["dk"], r["dk_kernel"]), dv_equals_kernel=_eq(r["dv"], r["dv_kernel"]),
                                   dq_equals_bf16_dq_scaled_times_scale=_eq(r["dq"], r["dq_from_scaled"]),
                                   q_scaled_dtype=str(r["q_scaled"].dtype)))
            td.write_index(part="T1.4 attention", size=args.size)
            common.write_json(os.path.join(td.path, "results.json"), res)
            log("attn done", res)

        # ------------------------------------------------------------------ T1.5 fused cross-entropy
        if "ce" in parts:
            from levanter.models.lm_model import split_activations
            from levanter.models.loss import maybe_fused_next_token_loss

            td = common.TensorDir(os.path.join(out, "ce"))
            # trainer setup may reset the root logger's handlers: attach the capture to the emitting loggers again
            for lname in ("", "levanter", "levanter.kernels.pallas.fused_cross_entropy_loss.api"):
                lg = logging.getLogger(lname)
                if cap not in lg.handlers:
                    lg.addHandler(cap)

            @eqx.filter_jit
            def hidden(model_c, exce):
                with hax.axis_mapping(mapping):
                    x, _aux = split_activations(model_c.activations(exce.tokens, exce.attn_mask))
                    return x

            x = hidden(model_c, exce)
            W = model_c.get_lm_head()
            Pos = x.resolve_axis("position")
            denom = jnp.float32(MICROBATCH_WEIGHT_SUM)

            def ce_loss(x, W):  # lm_model.py:340-354 with reduction=None, then haliax's weighted mean with the global denominator
                loss_tok = maybe_fused_next_token_loss(Pos, m.Embed, Vocab, x, W, exce.tokens, loss_weight=exce.loss_weight,
                                                       reduction=None, reduction_axis=None, logsumexp_weight=cfg.z_loss_weight,
                                                       dtype=jnp.float32, logit_soft_cap=None)
                return hax.sum(loss_tok).scalar() / denom, loss_tok

            @eqx.filter_jit
            def ce_all(x, W):
                with hax.axis_mapping(mapping):
                    total, f, loss_tok = eqx.filter_vjp(ce_loss, x, W, has_aux=True)
                    gx, gw = f(jnp.ones((), jnp.float32))
                    local_mean = hax.sum(loss_tok).scalar() / hax.sum(exce.loss_weight).scalar()
                    return total, local_mean, loss_tok, gx, gw

            total, local_mean, loss_tok, gx, gw = ce_all(x, W)
            labels = np.roll(np.asarray(jax.device_get(exce.tokens.array)), -1, axis=1).astype(np.int32)
            lw = np.asarray(jax.device_get(exce.loss_weight.array))
            for n, a in (("x", x), ("W", W), ("loss_tok", loss_tok), ("grad_x", gx), ("grad_w", gw)):
                td.save(n, np.asarray(jax.device_get(a.array)), axes=_named(a))
            td.save("labels", labels, axes=_named(exce.tokens))
            td.save("loss_weight", lw, axes=_named(exce.loss_weight))
            lt = np.asarray(jax.device_get(loss_tok.array)).astype(np.float64)
            res = dict(rows=args.ce_rows, tokens=int(lt.size), D=m.hidden_dim, V=Vocab.size, denominator=MICROBATCH_WEIGHT_SUM,
                       weighted_sum_over_denominator_f32=float(total), weighted_mean_these_rows_f32=float(local_mean),
                       weighted_mean_these_rows_f64=float(lt.sum() / lw.astype(np.float64).sum()), loss_weight_sum=float(lw.sum()),
                       dtypes=dict(x=str(x.dtype), W=str(W.dtype), loss_tok=str(loss_tok.dtype), grad_x=str(gx.dtype), grad_w=str(gw.dtype)),
                       device_kind=jax.devices()[0].device_kind, implementation_log=list(cap.lines),
                       cotangent="d/d loss_tok of sum(loss_tok * w) / 524160 (f32), as each GPU's tokens get in the 4-GPU step")
            td.write_index(part="T1.5 fused cross-entropy", size=args.size)
            common.write_json(os.path.join(td.path, "results.json"), res)
            log("ce done", {k: v for k, v in res.items() if k != "implementation_log"}, cap.lines[-3:])

    results["log_lines"] = cap.lines
    checks = [c for p, c in (("norm", "T1.2"), ("rope", "T1.3"), ("attn", "T1.4"), ("ce", "T1.5")) if p in parts]
    common.write_manifest(out, checks=checks, env=common.environment(),
                          extra=dict(size=args.size, run_id=cfg0.trainer.id, num_train_steps=cfg0.trainer.num_train_steps,
                                     cotangent_seed=COT_SEED, cotangent_fold_in=COT, results=results))
    log("wrote", out)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)  # the loader's background thread can abort interpreter teardown
