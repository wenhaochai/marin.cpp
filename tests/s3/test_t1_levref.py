"""Parity gates T1.2 (RMSNorm) and T1.3 (RoPE) against the levanter fixtures of tools/levref (kaiyue_baseline.md, T1).

Fixtures: every cache/levref/<size>/<name>/{norm,rope}/index.json under --src (tools.levref.units: levanter's own
RMSNorm, q_norm / k_norm and DefaultRotaryEmbeddings on the tensors of layer 0 of the seed-0 init, batch 0 rows 0-1,
cotangents from PRNGKey(1234)). The backend that made a fixture (manifest.json: cpu or gpu) is printed; the run's
platform is the GPU, so a table gate on an XLA:CPU fixture is informational (XLA:CPU has its own pow / cos / sin).
Each case skips with a message when no fixture of its kind exists.

T1.2 (spec): out and dx: >= 99% of elements bitwise equal and every element within 1 bf16 ulp of levanter; dg (bf16
  reduction over the positions): oracle rule rel(nano, f64) <= 1.25 x rel(levanter, f64). The case enforces that text.
  Elements of dx beyond 1 ulp that are cancelling sums (|nano - levanter| <= 4 f32 ulps of the addends |a r| + |c 2x|:
  a different f32 order of the row sum, class S) are not in the spec: when they are the only violations the case is
  reported as XFAIL (T1.2 not passed) until the owner amends T1.2 or records the failure; any other violation fails
  it. The block norm runs
  rmsnorm_fwd / rmsnorm_bwd_add over D; q_norm / k_norm run both rmsnorm over hd and the fused QK kernel with identity
  RoPE tables (cos 1, sin 0, scale 1: the RoPE stage is then exact, so the QK kernel's norm is what is compared).
T1.3 (spec): f32 inv_freq bitwise; bf16 cos / sin tables [4096, hd] bitwise; rotated q and k bitwise. The rotation
  runs rope_fwd (the RoPE stage of the QK kernel) on levanter's normed qn / kn with levanter's tables.
"""
import ctypes
import glob
import json
import os

import numpy as np
import torch

from nmk import BF16S, EPS, LEV, Skip, XFail, bits16, init, ptr, rel, ulp_diff_bf16

lib, R = init("s3_t1_levref")


PART_TITLE = {"norm": "T1.2 RMSNorm", "rope": "T1.3 RoPE"}   # index.json "part" of a tools.levref.units fixture


def fixtures(part):
    root = os.path.join(__import__("nmk").ARGS.src, "cache", "levref")
    out = []
    for idx in sorted(glob.glob(os.path.join(root, "*", "*", part, "index.json"))):
        d = os.path.dirname(os.path.dirname(idx))
        if os.path.basename(d).startswith("_"):
            continue
        try:
            if json.load(open(idx)).get("part") != PART_TITLE[part]:
                continue   # e.g. an oracle directory (tools.levref.oracle) with the same layout
        except (OSError, ValueError):
            continue
        backend = "?"
        try:
            env = json.load(open(os.path.join(d, "manifest.json"))).get("environment", {})
            backend = env.get("backend", "?")
            devs = env.get("devices", [])
            if devs:
                backend += ":" + str(devs[0])
        except (OSError, ValueError):
            pass
        out.append((os.path.relpath(d, root), os.path.join(d, part), backend))
    return out


def load(path, name):
    a = np.load(os.path.join(path, name))
    t = torch.from_numpy(a.view(np.int16) if a.dtype == np.uint16 else a).cuda()
    return t.view(torch.bfloat16) if a.dtype == np.uint16 else t


def ulp_report(name, k, ref, min_frac=0.99, max_ulp=1, addend=None):
    """Spec: >= min_frac bitwise and every element within max_ulp bf16 ulp of levanter. `addend` (dx only): the
    magnitude |a r| + |c 2x| of the two f32 terms whose sum is rounded to give dx. Where they cancel (|dx| many bf16
    ulps below them), any other f32 summation order of the row sum s moves dx by a few f32 ulps of the addends, which
    can be many bf16 ulps of dx. An element beyond max_ulp with |nano - levanter| <= 4 x 2^-24 x addend (4 f32 ulps of
    the addends) is a cancelling-sum violation: returned (count, description) for the caller's XFAIL, not a check
    failure; every other element beyond max_ulp is a check failure."""
    eq = (bits16(k) == bits16(ref)).double().mean().item()
    u = ulp_diff_bf16(k, ref)
    over = u > max_ulp
    n_over = int(over.sum())
    msg = f"    {name}: bitwise {100 * eq:.4f}%  max ulp {int(u.max())}  elements > {max_ulp} ulp: {n_over}"
    bad, n_canc, desc = over, 0, ""
    if addend is not None and n_over:
        diff = (k.double() - ref.double()).abs()
        canc = over & (diff <= 4 * 2.0 ** -24 * addend)
        bad = over & ~canc
        n_canc = int(canc.sum())
        ratio = (diff[over] / addend[over]).max().item() * 2 ** 24
        msg += f" (cancelling sums: {n_canc}, max |nano - lev| = {ratio:.2f} f32 ulps of the addends)"
        for i in torch.nonzero(canc.reshape(-1)).squeeze(1).tolist():
            kv, rv = k.reshape(-1)[i].float().item(), ref.reshape(-1)[i].float().item()
            msg += (f"\n      element {i}: nano {kv:.6e} levanter {rv:.6e} ({int(u.reshape(-1)[i])} ulp), addends "
                    f"{addend.reshape(-1)[i].item():.3e}")
        desc = f"{name}: {n_canc} elements {int(u[canc].min())}-{int(u[canc].max())} ulp from levanter"
    print(msg)
    R.check(eq >= min_frac, f"{name}: bitwise {eq:.6f} < {min_frac}")
    R.check(int(bad.sum()) == 0, f"{name}: {int(bad.sum())} elements beyond {max_ulp} ulp")
    return n_canc, desc


def dx_addends(x2, g, dy2):
    """|a r| + |c 2x| of the levanter RMSNorm backward, in f64 from the bf16 inputs."""
    x = x2.double()
    a = (g.float() * dy2.float()).to(torch.bfloat16).double()
    ms = (x * x).mean(-1, keepdim=True)
    r = torch.rsqrt(ms + EPS)
    s = (x * a).sum(-1, keepdim=True)
    c = s * (-0.5 * r / (ms + EPS)) / x.shape[-1]
    return (a * r).abs() + (c * 2 * x).abs()


def f64_norm_vjp(x, g, dy):
    x64 = x.double().requires_grad_(True)
    g64 = g.double().requires_grad_(True)
    o = g64 * x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + EPS)
    o.backward(dy.double())
    return o.detach(), x64.grad, g64.grad


def run_rmsnorm(x2, g, dy2):
    T, D = x2.shape
    out = torch.empty_like(x2)
    dx = torch.empty_like(x2)
    rstd = torch.empty(T, device="cuda")
    dg = torch.empty(D, device="cuda")
    lib.nm_rmsnorm_fwd(BF16S, LEV, ptr(x2), ptr(g), ptr(out), ptr(rstd), T, D, EPS, None)
    lib.nm_rmsnorm_bwd_add(BF16S, LEV, ptr(dy2), ptr(x2), ptr(g), ptr(rstd), None, ptr(dx), ptr(dg), T, D, EPS, None)
    torch.cuda.synchronize()
    return out, dx, dg


def run_qk_identity(x3, g, dy3, as_q):
    """QK kernel on heads x3 [T, nh, hd] (as the q or the k block), identity RoPE, scale 1."""
    T, nh, hd = x3.shape
    H, KV = (nh, nh) if as_q else (nh, nh)
    W = (H + 2 * KV) * hd
    qkv = torch.zeros(T, W, dtype=torch.bfloat16, device="cuda")
    off = 0 if as_q else H * hd
    qkv[:, off:off + nh * hd] = x3.reshape(T, nh * hd)
    ones = torch.ones(1, hd, dtype=torch.bfloat16, device="cuda")
    cos = ones.expand(1, hd).contiguous()
    sin = torch.zeros(1, hd, dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros(T, dtype=torch.int32, device="cuda")
    q_rot = torch.empty(T, H, hd, dtype=torch.bfloat16, device="cuda")
    k_rot = torch.empty(T, KV, hd, dtype=torch.bfloat16, device="cuda")
    rq = torch.empty(T, H, device="cuda")
    rk = torch.empty(T, KV, device="cuda")
    lib.nm_qknorm_rope_fwd(BF16S, LEV, ptr(qkv), W, ptr(g), ptr(g), ptr(cos), ptr(sin), ptr(pos), ptr(q_rot),
                           ptr(k_rot), ptr(rq), ptr(rk), T, H, KV, hd, EPS, 1.0, None)
    d = dy3.reshape(T, nh, hd).contiguous()
    dqkv = torch.zeros(T, W, dtype=torch.bfloat16, device="cuda")
    dgq = torch.empty(hd, device="cuda")
    dgk = torch.empty(hd, device="cuda")
    lib.nm_qknorm_rope_bwd(BF16S, LEV, ptr(d) if as_q else None, None if as_q else ptr(d), ptr(qkv), W, ptr(g),
                           ptr(g), ptr(cos), ptr(sin), ptr(pos), ptr(rq), ptr(rk), ptr(dqkv), W, ptr(dgq), ptr(dgk),
                           T, H, KV, hd, EPS, 1.0, None)
    torch.cuda.synchronize()
    out = q_rot if as_q else k_rot
    dx = dqkv[:, off:off + nh * hd].reshape(T, nh, hd)
    return out, dx, dgq if as_q else dgk


@R.case
def t1_2_rmsnorm():
    fx = fixtures("norm")
    if not fx:
        raise Skip("no T1.2 fixture (cache/levref/<size>/<name>/norm); run tools/levref/units.py")
    spec_violations = []
    for name, path, backend in fx:
        print(f"  fixture {name} (backend {backend})")
        for tag in ("block", "q", "k"):
            x, g, dy = (load(path, f"{tag}.{t}.bf16.npy") for t in ("x", "g", "dy"))
            lo, ldx, ldg = (load(path, f"{tag}.{t}.bf16.npy") for t in ("out", "dx", "dg"))
            w = x.shape[-1]
            x2, dy2 = x.reshape(-1, w).contiguous(), dy.reshape(-1, w).contiguous()
            o64, dx64, dg64 = f64_norm_vjp(x2, g, dy2)
            print(f"   {tag}: rows {x2.shape[0]} x {w}, gain values {torch.unique(g.float())[:4].tolist()}")
            runs = [("rmsnorm", run_rmsnorm(x2, g, dy2))]
            if tag != "block":
                nh = x2.shape[0] // (x.shape[0] * x.shape[1])
                T = x.shape[0] * x.shape[1]
                o3, d3, g3 = run_qk_identity(x2.reshape(T, nh, w), g, dy2.reshape(T, nh, w), tag == "q")
                runs.append(("qk kernel", (o3.reshape(-1, w), d3.reshape(-1, w), g3)))
            add = dx_addends(x2, g, dy2)
            for kname, (out, dx, dg) in runs:
                ulp_report(f"{tag} out [{kname}]", out, lo.reshape(-1, w))
                n_canc, desc = ulp_report(f"{tag} dx  [{kname}]", dx, ldx.reshape(-1, w), addend=add)
                if n_canc:
                    spec_violations.append(f"{name} {desc}")
                rn, rl = rel(dg, dg64), rel(ldg.float(), dg64)
                print(f"    {tag} dg  [{kname}]: rel(nano, f64) {rn:.4e} <= 1.25 x rel(levanter, f64) {rl:.4e}")
                R.check(rn <= 1.25 * rl, f"{tag} dg [{kname}]: oracle rule")
    if spec_violations:
        raise XFail("T1.2 requires every element within 1 bf16 ulp; beyond it, all cancelling sums (within 4 f32 ulps "
                    "of the addends): " + "; ".join(spec_violations) + ". Owner: amend T1.2 or record it as failing")


def rope_fixture_tensors(path):
    res = json.load(open(os.path.join(path, "results.json")))
    t = {n: load(path, f"{n}.bf16.npy") for n in ("cos", "sin", "qn", "kn", "qr", "kr")}
    t["inv_freq"] = load(path, "inv_freq.npy")
    return res, t


@R.case
def t1_3_rotation():
    fx = fixtures("rope")
    if not fx:
        raise Skip("no T1.3 fixture (cache/levref/<size>/<name>/rope); run tools/levref/units.py")
    for name, path, backend in fx:
        res, t = rope_fixture_tensors(path)
        hd, S = res["head_dim"], res["positions"]
        print(f"  fixture {name} (backend {backend}) hd={hd} positions={S}")
        for x, ref, lab in ((t["qn"], t["qr"], "q"), (t["kn"], t["kr"], "k")):
            B, Sx = x.shape[0], x.shape[1]
            T = B * Sx
            nh = x.numel() // (T * hd)
            x2 = x.reshape(T, nh * hd).contiguous()
            pos = (torch.arange(T, device="cuda", dtype=torch.int32) % Sx).contiguous()
            out = torch.empty_like(x2)
            lib.nm_rope_fwd(BF16S, LEV, ptr(x2), nh * hd, ptr(t["cos"]), ptr(t["sin"]), ptr(pos), ptr(out), T, nh, hd,
                            1.0, None)
            torch.cuda.synchronize()
            eq = torch.equal(bits16(out), bits16(ref.reshape(T, nh * hd)))
            frac = (bits16(out) == bits16(ref.reshape(T, nh * hd))).double().mean().item()
            print(f"    rotated {lab}: bitwise {100 * frac:.4f}% ({T} tokens x {nh} heads)")
            R.check(eq, f"rotated {lab} not bitwise equal")


def host_tables(method, S, hd):
    c = np.empty((S, hd), np.uint16)
    s = np.empty((S, hd), np.uint16)
    inv = np.empty(hd // 2, np.float32)
    vp = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.nm_rope_tables_method(method, S, hd, 10000.0, vp(c), vp(s), vp(inv))
    return inv, c, s


def bf16_bits_np(x):
    """f32 array -> bf16 bit patterns (RNE)."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def mismatch(a, b, hd):
    """'n entries (k distinct angles)': the tables repeat each half, so entries (p, i) and (p, i + hd/2) share one
    angle."""
    m = (a != b).reshape(-1, hd)
    distinct = int((m[:, :hd // 2] | m[:, hd // 2:]).sum())
    return f"{int(m.sum())} entries ({distinct} distinct angles)"


def bf16_ulp_np(a, b):
    def ordered(u):
        i = u.astype(np.int32)
        return np.where(i >= 0x8000, -(i & 0x7FFF), i)
    return np.abs(ordered(a) - ordered(b))


def table_report(t, S, hd):
    """Bitwise agreement of each build method with the fixture, plus the checks of the library's builders themselves:
    the host-double method equals its own definition (cos / sin of the f32 angle in double, rounded to f32, then bf16)
    computed in numpy from its own inv_freq, on every entry; the libdevice method's inv_freq is within 4 f32 ulp of it
    (powf) and its tables within 1 bf16 ulp of cos / sin in double of its own f32 angle (cosf / sinf: <= 2 f32 ulp). Returns ({method: all equal to the fixture}, cos and sin fractions of levanter's angle)."""
    fi = t["inv_freq"].cpu().numpy().view(np.uint32)
    fc = t["cos"].view(torch.int16).cpu().numpy().view(np.uint16)
    fs = t["sin"].view(torch.int16).cpu().numpy().view(np.uint16)
    ok, built = {}, {}
    for m, mname in ((0, "host double (DESIGN.md)"), (1, "device libdevice (XLA:GPU functions)")):
        inv, c, s = host_tables(m, S, hd)
        built[m] = (inv, c, s)
        ei = (inv.view(np.uint32) == fi)
        ulp = np.abs(inv.view(np.int32).astype(np.int64) - fi.view(np.int32)).max()
        print(f"    {mname} vs fixture: inv_freq {ei.sum()}/{ei.size} bitwise (max {ulp} ulp), cos differs at "
              f"{mismatch(c, fc, hd)}, sin at {mismatch(s, fs, hd)}")
        ok[m] = bool(ei.all() and (c == fc).all() and (s == fs).all())
    # the library's builders against their own definitions
    inv0, c0, s0 = built[0]
    ang = (np.arange(S, dtype=np.float32)[:, None] * inv0[None, :]).astype(np.float32)
    ang = np.concatenate([ang, ang], 1).astype(np.float64)
    rc, rs = bf16_bits_np(np.cos(ang).astype(np.float32)), bf16_bits_np(np.sin(ang).astype(np.float32))
    print(f"    host double vs its definition (numpy, own inv_freq): cos differs at {mismatch(c0, rc, hd)}, sin at "
          f"{mismatch(s0, rs, hd)}")
    R.check(bool((c0 == rc).all() and (s0 == rs).all()), "host-double tables differ from their definition")
    inv1, c1, s1 = built[1]
    iu = int(np.abs(inv1.view(np.int32).astype(np.int64) - inv0.view(np.int32)).max())
    ang1 = (np.arange(S, dtype=np.float32)[:, None] * inv1[None, :]).astype(np.float32)
    ang1 = np.concatenate([ang1, ang1], 1).astype(np.float64)
    r1c, r1s = bf16_bits_np(np.cos(ang1).astype(np.float32)), bf16_bits_np(np.sin(ang1).astype(np.float32))
    cu, su = int(bf16_ulp_np(c1, r1c).max()), int(bf16_ulp_np(s1, r1s).max())
    print(f"    libdevice: inv_freq max {iu} f32 ulp from host double (powf); tables vs cos / sin in double of its own "
          f"f32 angle: cos max {cu} bf16 ulp ({mismatch(c1, r1c, hd)}), sin max {su} ({mismatch(s1, r1s, hd)})")
    R.check(iu <= 4 and cu <= 1 and su <= 1, "libdevice method: inv_freq > 4 f32 ulp from host double, or a table "
                                             "entry > 1 bf16 ulp from its own angle's cos / sin")
    # cos / sin evaluated from levanter's own inv_freq (isolates the cos / sin evaluation from pow)
    inv = t["inv_freq"].double().cpu().numpy().astype(np.float32)
    ang = (np.arange(S, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    ang = np.concatenate([ang, ang], 1).astype(np.float64)
    cd, sd = bf16_bits_np(np.cos(ang).astype(np.float32)), bf16_bits_np(np.sin(ang).astype(np.float32))
    print(f"    cos / sin of levanter's own f32 angle (double, f32, bf16) vs fixture: cos differs at "
          f"{mismatch(cd, fc, hd)}, sin at {mismatch(sd, fs, hd)}")
    return ok, (cd == fc).mean(), (sd == fs).mean()


@R.case
def t1_3_tables_gpu():
    fx = [f for f in fixtures("rope") if f[2].startswith("gpu") or "cuda" in f[2].lower()]
    if not fx:
        raise Skip("no GPU-made T1.3 fixture yet (the training run computed inv_freq and the tables on the GPU; "
                   "run tools/levref/t1_gpu.sh units)")
    for name, path, backend in fx:
        res, t = rope_fixture_tensors(path)
        print(f"  fixture {name} (backend {backend})")
        ok, _, _ = table_report(t, res["positions"], res["head_dim"])
        R.check(ok[0] or ok[1], "no build method reproduces the GPU tables bitwise: O4, load inv_freq and the tables "
                                "from the fixture")
        print(f"    bitwise method: {[m for m in ok if ok[m]]} (0 host double, 1 device libdevice)")


@R.case
def t1_3_tables_cpu_info():
    fx = [f for f in fixtures("rope") if f[2].startswith("cpu")]
    if not fx:
        raise Skip("no XLA:CPU T1.3 fixture")
    for name, path, backend in fx:
        res, t = rope_fixture_tensors(path)
        print(f"  fixture {name} (backend {backend}): informational, XLA:CPU's pow / cos / sin are not the run's")
        _, fc, fs = table_report(t, res["positions"], res["head_dim"])
        R.check(fc >= 0.9999 and fs >= 0.9999, "cos / sin of levanter's own angle differ from the f32 evaluation")


R.run(__import__("nmk").ARGS.cases)
