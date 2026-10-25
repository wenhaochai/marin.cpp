"""float64 oracles for the T1 unit fixtures (kaiyue_baseline.md, "Oracle rule"), in numpy only.

    python -m tools.levref.oracle --size 130m [--parts norm,attn,ce,ns] [--name t1_oracle]

Each oracle takes the fixture's own inputs upcast to float64 and computes the op in exact-ish arithmetic. A nanomarin test
imports these functions (or reads the saved outputs) and checks, per output tensor T:
    rel(nano, oracle) <= 1.25 * rel(lev, oracle)        (and rel(nano, lev) <= rel(lev, oracle) for a bit-faithful port)
with rel(a, b) = ||a - b||_2 / ||b||_2 in float64 over the whole tensor.

Inputs: cache/levref/<size>/t1_units/{norm,attn,ce}/ (tools.levref.units) and cache/levref/<size>/t1_precision/ns/
(tools.levref.precision). Outputs: cache/levref/<size>/<name>/<part>/ (TensorDir of float64 .npy, index.json) and
rel.json there: for every output tensor rel(lev, oracle), max |lev - oracle| and the oracle's norm. RoPE (T1.3) is checked
bitwise against levanter, so it has no oracle.

Formulas (bf16 values are exact in float64; constants are the ones levanter uses):
  rmsnorm   r = (mean_axis(x^2) + eps)^-1/2, eps = float32(1e-5); out = g x r; with a = g dy:
            dx = r a - (r^3 / n) x sum_axis(a x); dg = sum over all other positions of dy x r
  attention s = the bf16 scale constant (0.125 at hd 64, 0.08837890625 at hd 128); S = s q k^T over keys j <= i with
            seg_q[i] == seg_kv[j]; P = softmax(S); O = P v; L = logsumexp(S) (natural log); D = rowsum(dO O);
            dS = P (dO v^T - D); dq = s dS k; dk = s sum_g dS^T q; dv = sum_g P^T dO
  ce        z = x W^T; loss = logsumexp(z) - z[label]; c = w / 524160; dz = (softmax(z) - onehot(label)) c;
            grad_x = dz W; grad_w = dz^T x (W stored [V, D], as levanter's lm_head)
  ns        X / (||X||_F + eps); transpose if rows > cols; 5 x: A = X X^T, B = b A + c A A, X = a X + B X;
            (a, b, c) = (3.4445, -4.7750, 2.0315); eps = 1e-5
"""
import argparse
import json
import os
import time

import numpy as np

from tools.levref import common

EPS_F32 = float(np.float32(1e-5))
NS_SIMPLE = (3.4445, -4.7750, 2.0315)


def rel(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    nb = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / nb) if nb > 0 else float(np.linalg.norm(a.ravel()))


# ------------------------------------------------------------------------------------------------ oracles
def rmsnorm(x, g, dy, axis: int, eps: float = EPS_F32):
    """out, dx, dg in float64. x, dy: same shape; g: 1-D gain along `axis`."""
    x = np.asarray(x, np.float64)
    dy = np.asarray(dy, np.float64)
    shape = [1] * x.ndim
    shape[axis] = -1
    g = np.asarray(g, np.float64).reshape(shape)
    n = x.shape[axis]
    r = 1.0 / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + eps)
    y = x * r
    out = g * y
    a = g * dy
    dx = r * a - (r ** 3 / n) * x * np.sum(a * x, axis=axis, keepdims=True)
    other = tuple(i for i in range(x.ndim) if i != axis)
    dg = np.sum(dy * y, axis=other)
    return out, dx, dg


def attention(q, k, v, seg_q, seg_kv, dO, scale: float):
    """q [B, KV, G, S, hd], k and v [B, KV, T, hd], seg_q [B, S], seg_kv [B, T], dO like q.
    Returns O, L [B, KV, G, S], dq (w.r.t. the unscaled q), dk, dv, all float64."""
    q, k, v, dO = (np.asarray(a, np.float64) for a in (q, k, v, dO))
    Bn, KV, G, S, hd = q.shape
    T = k.shape[2]
    O = np.zeros_like(q)
    L = np.zeros(q.shape[:-1])
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)
    causal = np.arange(T)[None, :] <= np.arange(S)[:, None]
    for b in range(Bn):
        allowed = causal & (np.asarray(seg_q[b])[:, None] == np.asarray(seg_kv[b])[None, :])
        for h in range(KV):
            kb, vb = k[b, h], v[b, h]
            for gi in range(G):
                qs = q[b, h, gi] * scale
                Sm = np.where(allowed, qs @ kb.T, -np.inf)
                m = Sm.max(axis=1, keepdims=True)
                P = np.exp(Sm - m)
                lsum = P.sum(axis=1, keepdims=True)
                P /= lsum
                o = P @ vb
                O[b, h, gi] = o
                L[b, h, gi] = (m + np.log(lsum))[:, 0]
                do = dO[b, h, gi]
                dP = do @ vb.T
                Dr = np.sum(do * o, axis=1, keepdims=True)
                dS = P * (dP - Dr)
                dq[b, h, gi] = scale * (dS @ kb)
                dk[b, h] += dS.T @ qs
                dv[b, h] += P.T @ do
    return O, L, dq, dk, dv


def cross_entropy(x, W, labels, weights, denom: float, tile: int = 2048):
    """x [N, D], W [V, D], labels [N], weights [N] -> loss [N] (unweighted), grad_x [N, D], grad_w [V, D], float64."""
    x = np.asarray(x, np.float64)
    W = np.asarray(W, np.float64)
    labels = np.asarray(labels).astype(np.int64)
    c_all = np.asarray(weights, np.float64) / float(denom)
    N = x.shape[0]
    loss = np.zeros(N)
    gx = np.zeros_like(x)
    gw = np.zeros_like(W)
    for r0 in range(0, N, tile):
        r1 = min(N, r0 + tile)
        xt = x[r0:r1]
        z = xt @ W.T
        m = z.max(axis=1, keepdims=True)
        p = np.exp(z - m)
        s = p.sum(axis=1, keepdims=True)
        lse = (m + np.log(s))[:, 0]
        rows = np.arange(r1 - r0)
        loss[r0:r1] = lse - z[rows, labels[r0:r1]]
        p /= s
        p[rows, labels[r0:r1]] -= 1.0
        p *= c_all[r0:r1, None]
        gx[r0:r1] = p @ W
        gw += p.T @ xt
        del z, p
    return loss, gx, gw


def newton_schulz(X, steps: int = 5, eps: float = 1e-5, coeffs=NS_SIMPLE):
    X = np.asarray(X, np.float64)
    X = X / (np.linalg.norm(X) + eps)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    a, b, c = coeffs
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


# ------------------------------------------------------------------------------------------------ CLI
def _perm(axes, want):
    return [axes.index(a) for a in want]


def _load(d, name):
    idx = json.load(open(os.path.join(d, "index.json")))["tensors"][name]
    return common.load_tensor(d, name), idx["axes"]


def _record(td, rels, name, lev, orc, axes=None):
    td.save(name, np.asarray(orc, np.float64), axes=axes)
    lev = np.asarray(lev, np.float64)
    rels[name] = dict(rel_lev_oracle=rel(lev, orc), max_abs_lev_minus_oracle=float(np.abs(lev - orc).max()),
                      oracle_norm=float(np.linalg.norm(np.asarray(orc).ravel())))


def run_norm(src, dst):
    td, rels = common.TensorDir(dst), {}
    idx = json.load(open(os.path.join(src, "index.json")))["tensors"]
    for tag in sorted({k.split(".")[0] for k in idx}):
        x, axes = _load(src, f"{tag}.x")
        g, gaxes = _load(src, f"{tag}.g")
        dy, _ = _load(src, f"{tag}.dy")
        axis = axes.index(gaxes[0])
        out, dx, dg = rmsnorm(x, g, dy, axis)
        for n, o in (("out", out), ("dx", dx), ("dg", dg)):
            lev, la = _load(src, f"{tag}.{n}")
            _record(td, rels, f"{tag}.{n}", lev, o, la)
    td.write_index(part="T1.2 oracle", source=src)
    return rels


def run_attn(src, dst, scale):
    td, rels = common.TensorDir(dst), {}
    q, qa = _load(src, "q")
    k, ka = _load(src, "k")
    v, va = _load(src, "v")
    dO, da = _load(src, "dO")
    sq, _ = _load(src, "segment_ids_q")
    skv, _ = _load(src, "segment_ids_kv")
    want_q = ["batch", "kv_head", "q_heads_per_group", "position", "head_size"]
    want_k = ["batch", "kv_head", "key_position", "head_size"]
    pq, pk = _perm(qa, want_q), _perm(ka, want_k)
    O, L, dq, dk, dv = attention(q.transpose(pq), k.transpose(pk), v.transpose(_perm(va, want_k)), sq, skv,
                                 dO.transpose(_perm(da, want_q)), scale)
    inv = lambda p: list(np.argsort(p))
    for n, o, p, axes in (("O", O, pq, qa), ("dq", dq, pq, qa), ("dk", dk, pk, ka), ("dv", dv, _perm(va, want_k), va)):
        lev, la = _load(src, n)
        o = o.transpose(inv(p))
        _record(td, rels, n, lev, o, la)
    lev, la = _load(src, "L")
    _record(td, rels, "L", lev, L.transpose(inv(_perm(la, want_q[:-1]))), la)
    td.write_index(part="T1.4 oracle", source=src, scale=scale)
    return rels


def run_ce(src, dst, denom):
    td, rels = common.TensorDir(dst), {}
    x, xa = _load(src, "x")
    W, Wa = _load(src, "W")
    labels, _ = _load(src, "labels")
    w, _ = _load(src, "loss_weight")
    D = x.shape[-1]
    assert xa[-1] == "embed" and Wa == ["vocab", "embed"], (xa, Wa)
    loss, gx, gw = cross_entropy(x.reshape(-1, D), W, labels.reshape(-1), w.reshape(-1), denom)
    lev, la = _load(src, "loss_tok")
    lw = w.reshape(lev.shape)
    _record(td, rels, "loss_tok", lev, (loss.reshape(lev.shape) * lw), la)  # levanter's loss_tok is weighted (pos 4095 = 0)
    td.save("loss_unweighted", loss.reshape(lev.shape), axes=la)
    lev_mean = float(np.asarray(lev, np.float64).sum() / lw.sum())
    orc_mean = float((loss.reshape(lev.shape) * lw).sum() / lw.sum())
    rels["weighted_mean"] = dict(lev_f64_of_f32=lev_mean, oracle=orc_mean, abs_rel_diff=abs(lev_mean - orc_mean) / abs(orc_mean))
    for n, o in (("grad_x", gx.reshape(x.shape)), ("grad_w", gw)):
        lev, la = _load(src, n)
        _record(td, rels, n, lev, o, la)
    td.write_index(part="T1.5 oracle", source=src, denominator=denom)
    return rels


def run_ns(src, dst, steps, eps):
    td, rels = common.TensorDir(dst), {}
    idx = json.load(open(os.path.join(src, "index.json")))["tensors"]
    for name in sorted(k[2:] for k in idx if k.startswith("u.")):
        X, _ = _load(src, f"u.{name}")
        o = newton_schulz(X, steps, eps)
        td.save(name, o, axes=["__OUT__", "__IN__"])
        hi, _ = _load(src, f"ns_highest.{name}")
        de, _ = _load(src, f"ns_default.{name}")
        rels[name] = dict(rel_default_oracle=rel(de, o), rel_highest_oracle=rel(hi, o), rel_default_highest=rel(de, hi),
                          oracle_norm=float(np.linalg.norm(o)))
        # every other saved variant (ns_tensorfloat32, ns_<prec>_<emitter> of precision.py) against the same oracle
        for k in sorted(idx):
            if k.startswith("ns_") and k.endswith(f".{name}") and k not in (f"ns_highest.{name}", f"ns_default.{name}"):
                v, _ = _load(src, k)
                rels[name][f"rel_{k[3:-len(name) - 1]}_oracle"] = rel(v, o)
    td.write_index(part="T1.6 oracle", source=src, steps=steps, eps=eps, coefficients=NS_SIMPLE)
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--parts", default="norm,attn,ce,ns")
    ap.add_argument("--units", default="t1_units")
    ap.add_argument("--precision", default="t1_precision")
    ap.add_argument("--name", default="t1_oracle")
    args = ap.parse_args()
    t0 = time.time()
    base = os.path.join(common.nano_dir(), "cache", "levref", args.size)
    out = common.fixture_dir(args.size, args.name)
    allrels, sources = {}, {}
    for part in args.parts.split(","):
        src = os.path.join(base, args.precision if part == "ns" else args.units, part)
        if not os.path.exists(os.path.join(src, "index.json")):
            print(f"[oracle] {part}: no fixture at {src}, skipped", flush=True)
            continue
        man = os.path.join(os.path.dirname(src), "manifest.json")
        sources[part] = dict(dir=os.path.relpath(src, base), manifest_sha256=common.sha256_file(man) if os.path.exists(man) else None)
        dst = os.path.join(out, part)
        if part == "norm":
            r = run_norm(src, dst)
        elif part == "attn":
            res = json.load(open(os.path.join(src, "results.json")))
            r = run_attn(src, dst, res["scale_bf16_value"])
        elif part == "ce":
            res = json.load(open(os.path.join(src, "results.json")))
            r = run_ce(src, dst, res["denominator"])
        elif part == "ns":
            r = run_ns(src, dst, 5, 1e-5)
        else:
            raise SystemExit(f"unknown part {part}")
        common.write_json(os.path.join(dst, "rel.json"), r)
        allrels[part] = r
        print(f"[oracle {time.time() - t0:7.1f}s] {part}: " + json.dumps(r)[:1500], flush=True)
    common.write_manifest(out, checks=["T1.2", "T1.4", "T1.5", "T1.6"], env=dict(numpy=np.__version__, numpy_config=_blas()),
                          extra=dict(size=args.size, sources=sources, rel=allrels))
    print("wrote", out)


def _blas():
    try:
        cfg = np.show_config(mode="dicts")
        return cfg.get("Build Dependencies", {}).get("blas", {})
    except Exception as e:
        return repr(e)


if __name__ == "__main__":
    main()
