"""S3 gate: RMSNorm over D (rmsnorm_fwd, rmsnorm_bwd_add) and the sandwich tails (sandwich placement with a post-norm;
pre-LN / hybrid placement without), every storage x rounding instantiation, at the owner's widths.

Checks per tensor (DESIGN.md §10.2):
  f32 storage    rel(kernel, f64 autograd) <= 1e-6 (math bugs), both rounding modes
  bf16 levanter  (a) exact-sum inputs (nmk.exact_sum_rows): bitwise equal to the levanter emulation (levemu.py) on
                     every element: the rounding chain itself is levanter's;
                 (b) realistic inputs: bitwise equal on >= 99.99% (a different f32 summation order flips rare bf16
                     roundings), rel(kernel, emulation) <= rel(emulation, f64), oracle rule rel(kernel, f64) <= 1.25 x
                     rel(emulation, f64); the forward driven by the kernel's own rstd is bitwise equal everywhere
  bf16 accurate  DESIGN.md §3.0 (f32 inside, one rounding per stored output): every bf16 output rel(kernel, f64) <=
                 1.05 x rel(bf16(f64), f64), the floor of a single rounding, against the f64 oracle of the kernel's own
                 stored inputs (the sandwich backward reads the stored bf16 x_out); the levanter emulation must fail
                 that bound (the gate has power); rstd and the f32 gain gradients rel <= 1e-6
  per-row report every row's rel <= 10 x the tensor rel, the 10 worst rows printed
"""
import torch

import levemu as E
from nmk import (ACC, BF16S, EPS, F32S, LEV, SHAPES, accurate_check, bits16, bitwise_check, exact_gain, exact_sum_rows,
                 f32_bound, init, ptr, rand_gain, rand_rows, rb, rel, row_report)

lib, R = init("s3_rmsnorm")
T_DEFAULT = 8192


def oracle_rule(name, k, emu, orc, slack=1.25):
    rk, re = rel(k, orc), rel(emu, orc)
    print(f"    {name}: rel(kernel, f64) {rk:.4e}  rel(levanter emulation, f64) {re:.4e}")
    R.check(rk <= slack * re + 1e-12, f"{name}: oracle rule {rk:.4e} > {slack} x {re:.4e}")
    return rk


def f32_check(name, k, orc, tol=1e-6):
    r = rel(k, orc)
    print(f"    {name}: rel(f32 kernel, f64) {r:.3e}")
    R.check(r <= tol, f"{name}: rel {r:.3e} > {tol}")


def rows_ok(name, k, orc, width, rk):
    R.check(row_report(name, k, orc, width, rk)[0], f"{name}: per-row bound")


def run_norm(T, D, storage, rounding, x, g, dy, dres):
    """fwd + bwd_add; returns out, rstd, dx, dg."""
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    xs, gs, dys, dress = (t.to(dt).contiguous() for t in (x, g, dy, dres))
    out = torch.empty(T, D, dtype=dt, device="cuda")
    rstd = torch.empty(T, dtype=torch.float32, device="cuda")
    dx = torch.empty(T, D, dtype=dt, device="cuda")
    dg = torch.empty(D, dtype=torch.float32, device="cuda")
    lib.nm_rmsnorm_fwd(storage, rounding, ptr(xs), ptr(gs), ptr(out), ptr(rstd), T, D, EPS, None)
    lib.nm_rmsnorm_bwd_add(storage, rounding, ptr(dys), ptr(xs), ptr(gs), ptr(rstd), ptr(dress), ptr(dx), ptr(dg), T, D,
                           EPS, None)
    torch.cuda.synchronize()
    return out, rstd, dx, dg


def lev_norm_emu(x, g, dy, dres, r=None):
    x32, g32, dy32 = x.float(), g.float(), dy.float()
    ms, r_t = E.norm_stats(x32, EPS)
    r = r_t if r is None else r.reshape(-1, 1)
    oe, _ = E.lev_norm_fwd(x32, g32, r)
    dxe, dgc = E.lev_norm_bwd(x32, g32, dy32, r, ms, EPS)
    return oe, rb(dres.float() + dxe), rb(dgc.sum(0))


def t_for(D):
    """8192 rows, 4096 at D = 2048 (the f64 oracles of a [T, 2048] case take ~1.5 GB on a shared 40 GB GPU)."""
    return T_DEFAULT if D < 2048 else T_DEFAULT // 2


def norm_case(shape, T=None, seed=0):
    D = SHAPES[shape]["D"]
    T = T or t_for(D)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = rand_rows(T, D, gen)
    g = rand_gain(D, gen)
    dy = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    dres = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    x64 = x.double().requires_grad_(True)
    g64 = g.double().requires_grad_(True)
    o64 = E.f64_norm(x64, g64, EPS)
    o64.backward(dy.double())
    o64 = o64.detach()
    dx64 = x64.grad + dres.double()
    dg64 = g64.grad
    r64 = torch.rsqrt((x.double() ** 2).mean(-1) + EPS)
    oe, dxe, dge = lev_norm_emu(x, g, dy, dres)
    print(f"  shape {shape} T={T} D={D}")
    for rnd in (LEV, ACC):
        out, rstd, dx, dg = run_norm(T, D, F32S, rnd, x, g, dy, dres)
        tag = "f32/" + ("lev" if rnd == LEV else "acc")
        f32_check(tag + " out", out, o64)
        f32_check(tag + " rstd", rstd, r64)
        f32_check(tag + " dx", dx, dx64)
        f32_check(tag + " dg", dg, dg64)
    out, rstd, dx, dg = run_norm(T, D, BF16S, LEV, x, g, dy, dres)
    oe_k, _, _ = lev_norm_emu(x, g, dy, dres, r=rstd)
    bitwise_check(R, "bf16/lev out vs emulation (kernel rstd)", out, oe_k.to(torch.bfloat16), min_frac=1.0)
    bitwise_check(R, "bf16/lev out vs emulation", out, oe.to(torch.bfloat16), o64)
    bitwise_check(R, "bf16/lev dx vs emulation", dx, dxe.to(torch.bfloat16), dx64)
    bitwise_check(R, "bf16/lev dg vs emulation", dg.to(torch.bfloat16), dge.to(torch.bfloat16), dg64, min_frac=0.99)
    rows_ok("bf16/lev out", out, o64, D, oracle_rule("bf16/lev out", out, oe, o64))
    rows_ok("bf16/lev dx", dx, dx64, D, oracle_rule("bf16/lev dx", dx, dxe, dx64))
    oracle_rule("bf16/lev dg", dg, dge, dg64)
    out, rstd, dx, dg = run_norm(T, D, BF16S, ACC, x, g, dy, dres)
    rows_ok("bf16/acc out", out, o64, D, accurate_check(R, "bf16/acc out", out, o64, lev=oe))
    rows_ok("bf16/acc dx", dx, dx64, D, accurate_check(R, "bf16/acc dx", dx, dx64, lev=dxe))
    f32_bound(R, "bf16/acc rstd", rstd, r64)
    f32_bound(R, "bf16/acc dg", dg, dg64)


def exact_case(shape, T=4096, seed=5):
    """Exact-sum inputs: the levanter-mode kernel equals the emulation on every element of out and dx."""
    D = SHAPES[shape]["D"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x, dy, dres = (exact_sum_rows(T, D, gen) for _ in range(3))
    g = exact_gain(D, gen)
    oe, dxe, _ = lev_norm_emu(x, g, dy, dres)
    out, rstd, dx, dg = run_norm(T, D, BF16S, LEV, x, g, dy, dres)
    print(f"  shape {shape} T={T} D={D} exact-sum inputs")
    bitwise_check(R, "bf16/lev out", out, oe.to(torch.bfloat16), min_frac=1.0)
    bitwise_check(R, "bf16/lev dx", dx, dxe.to(torch.bfloat16), min_frac=1.0)


for _s in ("130m", "300m", "520m", "1_2b", "pls_w512"):
    def _f(s=_s):
        norm_case(s)
    _f.__name__ = f"grad_rmsnorm_{_s}"
    R.case(_f)

    def _g(s=_s):
        exact_case(s)
    _g.__name__ = f"levanter_chain_rmsnorm_{_s}"
    R.case(_g)


def run_sandwich(T, D, storage, rounding, b, xi, gp, gn, dxo, dan):
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    cv = lambda t: None if t is None else t.to(dt).contiguous()
    b, xi, gp, gn, dxo, dan = map(cv, (b, xi, gp, gn, dxo, dan))
    e = lambda: torch.empty(T, D, dtype=dt, device="cuda")
    x_out, a_next, dx_in, dbr = e(), e(), e(), e()
    rp = torch.empty(T, device="cuda")
    rn = torch.empty(T, device="cuda")
    dgn = torch.empty(D, device="cuda")
    dgp = torch.empty(D, device="cuda")
    lib.nm_sandwich_tail_fwd(storage, rounding, ptr(b), ptr(xi), ptr(gp), ptr(gn), ptr(x_out), ptr(a_next), ptr(rp),
                             ptr(rn), T, D, EPS, None)
    lib.nm_sandwich_tail_bwd(storage, rounding, ptr(dxo), ptr(dan), ptr(x_out), ptr(gn), ptr(rn), ptr(b), ptr(gp),
                             ptr(rp), ptr(dx_in), ptr(dbr), ptr(dgn), ptr(dgp), T, D, EPS, None)
    torch.cuda.synchronize()
    return dict(x_out=x_out, a_next=a_next, dx_in=dx_in, dbranch=dbr, dgn=dgn, dgp=dgp, rp=rp, rn=rn)


def sandwich_emu(b, xi, gp, gn, dxo, dan, post, rp_k=None, rn_k=None):
    b32, x32 = b.float(), xi.float()
    if post:
        msp, rp = E.norm_stats(b32, EPS)
        rp = rp if rp_k is None else rp_k.reshape(-1, 1)
        y, _ = E.lev_norm_fwd(b32, gp.float(), rp)
    else:
        y = b32
    xo = rb(x32 + y)
    msn, rn = E.norm_stats(xo, EPS)
    rn = rn if rn_k is None else rn_k.reshape(-1, 1)
    an, _ = E.lev_norm_fwd(xo, gn.float(), rn)
    dxn, dgcn = E.lev_norm_bwd(xo, gn.float(), dan.float(), rn, msn, EPS)
    dxm = rb(dxo.float() + dxn)
    emu = dict(x_out=xo, a_next=an, dx_in=dxm, dgn=rb(dgcn.sum(0)), dbranch=dxm)
    if post:
        dbe, dgcp = E.lev_norm_bwd(b32, gp.float(), dxm, rp, msp, EPS)
        emu.update(dbranch=dbe, dgp=rb(dgcp.sum(0)))
    return emu


def sandwich_stage_oracle(b, xi, gp, gn, dxo, dan, post, x_out_k):
    """f64 math of the sandwich tails from the stored tensors: x_out = x_in + norm_post(branch); a_next, rstd_next and
    the backward from the kernel's stored bf16 x_out (what the training step keeps); dx_in = dx_out + vjp of the next
    norm; dbranch = vjp of the post-norm at dx_in (unrounded)."""
    b64 = b.double().requires_grad_(True)
    gp64 = gp.double().requires_grad_(True) if post else None
    y = E.f64_norm(b64, gp64, EPS) if post else b64
    xk = x_out_k.double().requires_grad_(True)
    gn64 = gn.double().requires_grad_(True)
    a = E.f64_norm(xk, gn64, EPS)
    a.backward(dan.double())
    dx_in = dxo.double() + xk.grad
    o = dict(x_out=(xi.double() + y).detach(), a_next=a.detach(), dx_in=dx_in, dgn=gn64.grad,
             rn=torch.rsqrt((x_out_k.double() ** 2).mean(-1) + EPS))
    if post:
        y.backward(dx_in)
        o.update(dbranch=b64.grad, dgp=gp64.grad, rp=torch.rsqrt((b.double() ** 2).mean(-1) + EPS))
    else:
        o.update(dbranch=dx_in)
    return o


def sandwich_case(shape, post, T=None, seed=1):
    D = SHAPES[shape]["D"]
    T = T or t_for(D)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = rand_rows(T, D, gen)
    xi = rand_rows(T, D, gen)
    gp = rand_gain(D, gen) if post else None
    gn = rand_gain(D, gen)
    dxo = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    dan = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    print(f"  shape {shape} T={T} D={D} placement={'sandwich' if post else 'pre-LN'}")
    b64 = b.double().requires_grad_(True)
    x64 = xi.double().requires_grad_(True)
    gp64 = gp.double().requires_grad_(True) if post else None
    gn64 = gn.double().requires_grad_(True)
    y = E.f64_norm(b64, gp64, EPS) if post else b64
    xo = x64 + y
    a = E.f64_norm(xo, gn64, EPS)
    (a * dan.double()).sum().add_((xo * dxo.double()).sum()).backward()
    o = dict(x_out=xo.detach(), a_next=a.detach(), dx_in=x64.grad, dbranch=b64.grad, dgn=gn64.grad,
             dgp=gp64.grad if post else None)
    emu = sandwich_emu(b, xi, gp, gn, dxo, dan, post)
    names = ["x_out", "a_next", "dx_in", "dbranch", "dgn"] + (["dgp"] if post else [])
    for rnd in (LEV, ACC):
        got = run_sandwich(T, D, F32S, rnd, b, xi, gp, gn, dxo, dan)
        for n in names:
            f32_check(f"f32/{'lev' if rnd == LEV else 'acc'} {n}", got[n], o[n])
    tag = "bf16/lev"
    got = run_sandwich(T, D, BF16S, LEV, b, xi, gp, gn, dxo, dan)
    ek = sandwich_emu(b, xi, gp, gn, dxo, dan, post, got["rp"], got["rn"])
    for n in ("x_out", "a_next"):
        bitwise_check(R, f"{tag} {n} vs emulation (kernel rstd)", got[n], ek[n].to(torch.bfloat16), min_frac=1.0)
    for n in names:
        bitwise_check(R, f"{tag} {n} vs emulation", got[n].to(torch.bfloat16), emu[n].to(torch.bfloat16), o[n],
                      min_frac=0.99 if n in ("dgn", "dgp") else 0.9999)
        rk = oracle_rule(f"{tag} {n}", got[n], emu[n], o[n])
        if n not in ("dgn", "dgp"):
            rows_ok(f"{tag} {n}", got[n], o[n], D, rk)
    # accurate: each output against the f64 math of the kernel's stored inputs (x_out is stored in bf16, and a_next and
    # the whole backward read it): one rounding per bf16 output, f32 gain gradients and rstd
    tag = "bf16/acc"
    got = run_sandwich(T, D, BF16S, ACC, b, xi, gp, gn, dxo, dan)
    so = sandwich_stage_oracle(b, xi, gp, gn, dxo, dan, post, got["x_out"])
    for n in names:
        if n in ("dgn", "dgp"):
            f32_bound(R, f"{tag} {n}", got[n], so[n])
        else:
            # x_out = x_in + y: pre-LN is one rounding in both modes (no power check); with the post-norm, levanter's
            # extra roundings of y show only where y dominates x_in, so the power check runs on those rows
            lev = None if n == "x_out" else emu[n]
            rows_ok(f"{tag} {n}", got[n], so[n], D, accurate_check(R, f"{tag} {n}", got[n], so[n], lev=lev))
            if n == "x_out" and post:
                y_norm = torch.linalg.vector_norm(so[n] - xi.double(), dim=1)
                m = torch.linalg.vector_norm(xi.double(), dim=1) < 0.25 * y_norm
                accurate_check(R, f"{tag} x_out, {int(m.sum())} rows with |x_in| < |y| / 4", got[n][m], so[n][m],
                               lev=emu[n][m])
    f32_bound(R, f"{tag} rstd_next", got["rn"], so["rn"])
    if post:
        f32_bound(R, f"{tag} rstd_post", got["rp"], so["rp"])


for _s in ("130m", "1_2b", "pls_w512"):
    for _post in (True, False):
        def _f(s=_s, p=_post):
            sandwich_case(s, p)
        _f.__name__ = f"grad_sandwich_{_s}_{'post' if _post else 'preln'}"
        R.case(_f)


@R.case
def in_place_residual_gradients():
    """Aliasing contract (ew_api.hpp): rmsnorm_bwd_add with dres == dx and sandwich_tail_bwd with dx_out == dx_in (and,
    pre-LN, dbranch == dx_in) give the same bits as separate buffers, both rounding modes."""
    T, D = 4111, 512
    gen = torch.Generator(device="cuda").manual_seed(21)
    b, xi = rand_rows(T, D, gen), rand_rows(T, D, gen)
    gp, gn = rand_gain(D, gen), rand_gain(D, gen)
    dxo = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    dan = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
    for rnd in (LEV, ACC):
        ref = run_sandwich(T, D, BF16S, rnd, b, xi, gp, gn, dxo, dan)
        for post in (True, False):
            if not post:
                ref = run_sandwich(T, D, BF16S, rnd, b, xi, None, gn, dxo, dan)
            buf = dxo.clone()
            dbr = buf if not post else torch.empty_like(buf)
            dgn, dgp = torch.empty(D, device="cuda"), torch.empty(D, device="cuda")
            lib.nm_sandwich_tail_bwd(BF16S, rnd, ptr(buf), ptr(dan), ptr(ref["x_out"]), ptr(gn), ptr(ref["rn"]),
                                     ptr(b), ptr(gp) if post else None, ptr(ref["rp"]) if post else None, ptr(buf),
                                     ptr(dbr), ptr(dgn), ptr(dgp) if post else None, T, D, EPS, None)
            torch.cuda.synchronize()
            R.check(torch.equal(bits16(buf), bits16(ref["dx_in"])), f"rnd={rnd} post={post}: in-place dx_in differs")
            R.check(torch.equal(bits16(dbr), bits16(ref["dbranch"])), f"rnd={rnd} post={post}: dbranch differs")
            R.check(torch.equal(dgn.view(torch.int32), ref["dgn"].view(torch.int32)), f"rnd={rnd}: dgn differs")
        x, g = rand_rows(T, D, gen), rand_gain(D, gen)
        dy, dres = (torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(2))
        _, rstd, dx, dg = run_norm(T, D, BF16S, rnd, x, g, dy, dres)
        buf = dres.clone()
        dg2 = torch.empty(D, device="cuda")
        lib.nm_rmsnorm_bwd_add(BF16S, rnd, ptr(dy), ptr(x), ptr(g), ptr(rstd), ptr(buf), ptr(buf), ptr(dg2), T, D,
                               EPS, None)
        torch.cuda.synchronize()
        R.check(torch.equal(bits16(buf), bits16(dx)), f"rnd={rnd}: in-place rmsnorm_bwd_add dx differs")
        print(f"    rounding {'lev' if rnd == LEV else 'acc'}: in-place outputs bitwise equal to separate buffers")


@R.case
def determinism_and_partials():
    """Two runs give bitwise-equal outputs and partials; the partial row count is nm_norm_partial_rows(T)."""
    for T in (1, 17, 4111, 131072):
        D = 512
        gen = torch.Generator(device="cuda").manual_seed(T)
        x, dy = rand_rows(T, D, gen), torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
        g = rand_gain(D, gen)
        P = lib.nm_norm_partial_rows(T)
        R.check(1 <= P <= 1024, f"partial rows {P} out of [1, 1024] at T={T}")
        outs = []
        for _ in range(2):
            dx = torch.empty_like(x)
            part = torch.full((P + 1, D), float("nan"), device="cuda")
            lib.nm_rmsnorm_bwd_add_partial(BF16S, LEV, ptr(dy), ptr(x), ptr(g), None, None, ptr(dx), ptr(part), T, D,
                                           EPS, None)
            torch.cuda.synchronize()
            outs.append((dx, part))
        R.check(torch.equal(bits16(outs[0][0]), bits16(outs[1][0])), f"dx differs between runs at T={T}")
        R.check(torch.equal(outs[0][1][:P].view(torch.int32), outs[1][1][:P].view(torch.int32)),
                f"partials differ between runs at T={T}")
        R.check(bool(torch.isnan(outs[0][1][P]).all()), f"kernel wrote past partial row {P} at T={T}")
        R.check(bool(torch.isfinite(outs[0][1][:P]).all()), f"unwritten partial rows at T={T}")
        print(f"    T={T}: {P} partial rows, bitwise repeatable")


@R.case
def edge_widths():
    """D = 64 (8-lane groups) up to D = 4096 (512-thread groups), odd T; f32 storage against f64."""
    for D, T in ((64, 37), (128, 5), (256, 1), (4096, 33)):
        gen = torch.Generator(device="cuda").manual_seed(D)
        x, g = rand_rows(T, D, gen), rand_gain(D, gen)
        dy = torch.randn(T, D, generator=gen, device="cuda").to(torch.bfloat16)
        dres = torch.zeros(T, D, device="cuda").to(torch.bfloat16)
        x64 = x.double().requires_grad_(True)
        g64 = g.double().requires_grad_(True)
        o64 = E.f64_norm(x64, g64, EPS)
        o64.backward(dy.double())
        out, rstd, dx, dg = run_norm(T, D, F32S, ACC, x, g, dy, dres)
        print(f"  D={D} T={T}")
        f32_check("out", out, o64.detach())
        f32_check("dx", dx, x64.grad)
        f32_check("dg", dg, g64.grad)


@R.case
def rejects_bad_width():
    x = torch.zeros(4, 100, dtype=torch.bfloat16, device="cuda")
    try:
        lib.nm_rmsnorm_fwd(BF16S, LEV, ptr(x), ptr(x), ptr(x), None, 4, 100, EPS, None)
        R.check(False, "width 100 accepted")
    except RuntimeError as e:
        R.check("multiple of 8" in str(e), f"unexpected message: {e}")


R.run(__import__("nmk").ARGS.cases)
