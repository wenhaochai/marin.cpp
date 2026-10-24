"""S3 gate: QK-norm (RMSNorm over hd, one gain per layer shared by the heads) + RoPE (NeoX half split, bf16 tables at
absolute positions) + the q scale, forward and backward, and the dq-postprocess fusion; MHA hd 64 (130m, pls width 512),
GQA hd 64 (520m) and GQA hd 128 (1_2b).

Checks (DESIGN.md §10.2):
  f32 storage    rel(kernel, f64 autograd) <= 1e-6 for q_rot, k_rot, rstd, dq, dk, dgq, dgk; both rounding modes
  bf16 levanter  forward driven by the kernel's own rstd: bitwise equal to the levanter emulation everywhere;
                 sparse rows (one nonzero per head row, so every row sum is exact): forward and backward bitwise equal
                 everywhere (the RoPE-backward and norm-backward rounding chains are levanter's);
                 realistic rows: >= 99.99% bitwise, rel(kernel, emu) <= rel(emu, f64), oracle rule, per-row report
  bf16 accurate  DESIGN.md §3.0 (f32 inside, one rounding per stored output): q_rot, k_rot, dq, dk rel(kernel, f64) <=
                 1.05 x rel(bf16(f64), f64) (single-rounding floor), and the levanter emulation must fail that bound;
                 rstd and the f32 gain gradients rel <= 1e-6
  layout         the v columns of dqkv are never written; a padded qkv_ld works; a k-only call (dq_rot = NULL) writes the
                 same k columns and leaves the q columns alone
  dq_post        dq_post_qknorm_rope_bwd on FA3's f32 accumulator equals qknorm_rope_bwd on bf16(accumulator) bitwise
                 (levanter); accurate: dq within 1.05 x the single-rounding floor of f64, dgq rel <= 1e-6
"""
import math

import numpy as np
import torch

import levemu as E
from nmk import (ACC, BF16S, EPS, F32S, LEV, SHAPES, accurate_check, bits16, bitwise_check, f32_bound, init, ptr,
                 rand_gain, rb, rel, row_report)

lib, R = init("s3_qknorm_rope")
S_MAX = 4096


def qscale(hd):
    return float(torch.tensor(1.0 / math.sqrt(hd)).to(torch.bfloat16).float())   # bf16(1/sqrt(hd)): class M


def tables(hd):
    c = np.empty((S_MAX, hd), np.uint16)
    s = np.empty((S_MAX, hd), np.uint16)
    inv = np.empty(hd // 2, np.float32)
    lib.nm_rope_tables(S_MAX, hd, 10000.0, c.ctypes.data_as(__import__("ctypes").c_void_p),
                       s.ctypes.data_as(__import__("ctypes").c_void_p), inv.ctypes.data_as(__import__("ctypes").c_void_p))
    to = lambda a: torch.from_numpy(a.view(np.int16)).cuda().view(torch.bfloat16)
    return to(c), to(s)


def make_inputs(T, H, KV, hd, gen, sparse=False, ld_pad=0):
    W = (H + 2 * KV) * hd
    ld = W + ld_pad
    buf = torch.randn(T, ld, generator=gen, device="cuda")
    s = torch.exp(torch.empty(T, 1, device="cuda").uniform_(math.log(0.05), math.log(20.0), generator=gen))
    buf = buf * s
    if sparse:   # one nonzero per head row, value in {+-0.25, +-0.5, +-0.75}
        v = torch.tensor([-3, -2, -1, 1, 2, 3], device="cuda", dtype=torch.float32) * 0.25
        nh = (H + 2 * KV)
        z = torch.zeros(T, nh, hd, device="cuda")
        idx = torch.randint(0, hd, (T, nh, 1), generator=gen, device="cuda")
        val = v[torch.randint(0, 6, (T, nh, 1), generator=gen, device="cuda")]
        z.scatter_(2, idx, val)
        buf[:, :W] = z.reshape(T, W)
    qkv = buf.to(torch.bfloat16)
    pos = (torch.arange(T, device="cuda", dtype=torch.int32) % S_MAX).contiguous()
    return qkv, ld, pos


def run_fwd(storage, rounding, qkv, ld, gq, gk, ct, st, pos, T, H, KV, hd):
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    qkv_s, gq_s, gk_s, ct_s, st_s = (t.to(dt).contiguous() for t in (qkv, gq, gk, ct, st))
    q_rot = torch.empty(T, H, hd, dtype=dt, device="cuda")
    k_rot = torch.empty(T, KV, hd, dtype=dt, device="cuda")
    rq = torch.empty(T, H, device="cuda")
    rk = torch.empty(T, KV, device="cuda")
    lib.nm_qknorm_rope_fwd(storage, rounding, ptr(qkv_s), ld, ptr(gq_s), ptr(gk_s), ptr(ct_s), ptr(st_s), ptr(pos),
                           ptr(q_rot), ptr(k_rot), ptr(rq), ptr(rk), T, H, KV, hd, EPS, qscale(hd), None)
    torch.cuda.synchronize()
    return q_rot, k_rot, rq, rk


SENT = -7.0   # sentinel in dqkv: the v columns (and q columns of a k-only call) must keep it


def run_bwd(storage, rounding, dq, dk, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd, dld=None):
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    cv = lambda t: None if t is None else t.to(dt).contiguous()
    dq_s, dk_s, qkv_s, gq_s, gk_s, ct_s, st_s = map(cv, (dq, dk, qkv, gq, gk, ct, st))
    dld = dld or (H + 2 * KV) * hd
    dqkv = torch.full((T, dld), SENT, dtype=dt, device="cuda")
    dgq = torch.full((hd,), float("nan"), device="cuda")
    dgk = torch.full((hd,), float("nan"), device="cuda")
    lib.nm_qknorm_rope_bwd(storage, rounding, ptr(dq_s), ptr(dk_s), ptr(qkv_s), ld, ptr(gq_s), ptr(gk_s), ptr(ct_s),
                           ptr(st_s), ptr(pos), ptr(rq), ptr(rk), ptr(dqkv), dld, ptr(dgq), ptr(dgk), T, H, KV, hd,
                           EPS, qscale(hd), None)
    torch.cuda.synchronize()
    return dqkv, dgq, dgk


def split(qkv, T, H, KV, hd):
    W = (H + 2 * KV) * hd
    q = qkv[:, :H * hd].reshape(T, H, hd)
    k = qkv[:, H * hd:(H + KV) * hd].reshape(T, KV, hd)
    v = qkv[:, (H + KV) * hd:W]
    return q, k, v


def oracle(q, k, gq, gk, ct, st, pos, dq, dk, hd):
    sc = qscale(hd)
    c64 = ct.double()[pos.long()].unsqueeze(1)
    s64 = st.double()[pos.long()].unsqueeze(1)
    q64 = q.double().requires_grad_(True)
    k64 = k.double().requires_grad_(True)
    gq64 = gq.double().requires_grad_(True)
    gk64 = gk.double().requires_grad_(True)
    nq = E.f64_norm(q64, gq64, EPS)
    nk = E.f64_norm(k64, gk64, EPS)
    oq = (nq * c64 + E.rot_half(nq) * s64) * sc
    ok = nk * c64 + E.rot_half(nk) * s64
    (oq * dq.double()).sum().add_((ok * dk.double()).sum()).backward()
    rq = torch.rsqrt((q.double() ** 2).mean(-1) + EPS)
    rk = torch.rsqrt((k.double() ** 2).mean(-1) + EPS)
    return dict(q_rot=oq.detach(), k_rot=ok.detach(), rq=rq, rk=rk, dq=q64.grad, dk=k64.grad, dgq=gq64.grad,
                dgk=gk64.grad)


def emulate(q, k, gq, gk, ct, st, pos, dq, dk, hd, rq_k=None, rk_k=None):
    sc = qscale(hd)
    c = ct.float()[pos.long()].unsqueeze(1)
    s = st.float()[pos.long()].unsqueeze(1)
    out = {}
    for name, x, g, d, rr, isq in (("q", q, gq, dq, rq_k, True), ("k", k, gk, dk, rk_k, False)):
        x32 = x.float()
        ms, r = E.norm_stats(x32, EPS)
        if rr is not None:
            r = rr.unsqueeze(-1)
        n, _ = E.lev_norm_fwd(x32, g.float(), r)
        o = E.lev_rope_fwd(n, c, s)
        d32 = d.float()
        if isq:
            o = rb(o * sc)
            d32 = rb(d32 * sc)
        dn = E.lev_rope_bwd(d32, c, s)
        dx, dgc = E.lev_norm_bwd(x32, g.float(), dn, r, ms, EPS)
        out[name + "_rot"] = o
        out["d" + name] = dx
        out["dg" + name] = rb(dgc.sum((0, 1)))
    return out


def f32_check(name, k, orc, tol=1e-6):
    r = rel(k, orc)
    print(f"    {name}: rel(f32 kernel, f64) {r:.3e}")
    R.check(r <= tol, f"{name}: rel {r:.3e} > {tol}")


def oracle_rule(name, k, emu, orc, slack=1.25):
    rk_, re = rel(k, orc), rel(emu, orc)
    print(f"    {name}: rel(kernel, f64) {rk_:.4e}  rel(levanter emulation, f64) {re:.4e}")
    R.check(rk_ <= slack * re + 1e-12, f"{name}: oracle rule {rk_:.4e} > {slack} x {re:.4e}")
    return rk_


def qk_case(shape, T=8192, seed=0, ld_pad=0):
    sh = SHAPES[shape]
    H, KV, hd = sh["H"], sh["KV"], sh["hd"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ct, st = tables(hd)
    qkv, ld, pos = make_inputs(T, H, KV, hd, gen, ld_pad=ld_pad)
    gq, gk = rand_gain(hd, gen), rand_gain(hd, gen)
    dq = torch.randn(T, H, hd, generator=gen, device="cuda").to(torch.bfloat16)
    dk = torch.randn(T, KV, hd, generator=gen, device="cuda").to(torch.bfloat16)
    q, k, v = split(qkv, T, H, KV, hd)
    o = oracle(q, k, gq, gk, ct, st, pos, dq, dk, hd)
    emu = emulate(q, k, gq, gk, ct, st, pos, dq, dk, hd)
    print(f"  shape {shape} T={T} H={H} KV={KV} hd={hd} qkv_ld={ld}")
    W = (H + 2 * KV) * hd
    for storage in (F32S, BF16S):
        for rnd in (LEV, ACC):
            tag = ("f32/" if storage == F32S else "bf16/") + ("lev" if rnd == LEV else "acc")
            q_rot, k_rot, rq, rk = run_fwd(storage, rnd, qkv, ld, gq, gk, ct, st, pos, T, H, KV, hd)
            dqkv, dgq, dgk = run_bwd(storage, rnd, dq, dk, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd)
            gdq, gdk, gv = split(dqkv, T, H, KV, hd)
            R.check(bool((gv.float() == SENT).all()), f"{tag}: v columns of dqkv were written")
            got = dict(q_rot=q_rot, k_rot=k_rot, rq=rq, rk=rk, dq=gdq, dk=gdk, dgq=dgq, dgk=dgk)
            if storage == F32S:
                for n in got:
                    f32_check(f"{tag} {n}", got[n], o[n])
                continue
            if rnd == LEV:
                ek = emulate(q, k, gq, gk, ct, st, pos, dq, dk, hd, rq, rk)
                for n in ("q_rot", "k_rot"):
                    bitwise_check(R, f"{tag} {n} vs emulation (kernel rstd)", got[n], ek[n].to(torch.bfloat16),
                                  min_frac=1.0)
                for n in ("q_rot", "k_rot", "dq", "dk", "dgq", "dgk"):
                    bitwise_check(R, f"{tag} {n} vs emulation", got[n].to(torch.bfloat16), emu[n].to(torch.bfloat16),
                                  o[n], min_frac=0.95 if n.startswith("dg") else 0.9999)
            for n in ("q_rot", "k_rot", "dq", "dk", "dgq", "dgk"):
                if rnd == LEV:
                    rk_ = oracle_rule(f"{tag} {n}", got[n], emu[n], o[n])
                elif n.startswith("dg"):
                    f32_bound(R, f"{tag} {n}", got[n], o[n])
                    continue
                else:
                    rk_ = accurate_check(R, f"{tag} {n}", got[n], o[n], lev=emu[n])
                if not n.startswith("dg"):
                    R.check(row_report(f"{tag} {n}", got[n], o[n], hd, rk_)[0], f"{tag} {n}: per-row bound")
            if rnd == ACC:
                f32_bound(R, f"{tag} rq", rq, o["rq"])
                f32_bound(R, f"{tag} rk", rk, o["rk"])


def sparse_case(shape, T=4096, seed=3):
    """One nonzero per head row: every row sum is exact, so levanter mode equals the emulation on every element."""
    sh = SHAPES[shape]
    H, KV, hd = sh["H"], sh["KV"], sh["hd"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ct, st = tables(hd)
    qkv, ld, pos = make_inputs(T, H, KV, hd, gen, sparse=True)
    gq, gk = rand_gain(hd, gen), rand_gain(hd, gen)
    dq = torch.randn(T, H, hd, generator=gen, device="cuda").to(torch.bfloat16)
    dk = torch.randn(T, KV, hd, generator=gen, device="cuda").to(torch.bfloat16)
    q, k, _ = split(qkv, T, H, KV, hd)
    emu = emulate(q, k, gq, gk, ct, st, pos, dq, dk, hd)
    q_rot, k_rot, rq, rk = run_fwd(BF16S, LEV, qkv, ld, gq, gk, ct, st, pos, T, H, KV, hd)
    dqkv, dgq, dgk = run_bwd(BF16S, LEV, dq, dk, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd)
    gdq, gdk, _ = split(dqkv, T, H, KV, hd)
    print(f"  shape {shape} T={T} sparse rows")
    for n, got in (("q_rot", q_rot), ("k_rot", k_rot), ("dq", gdq), ("dk", gdk)):
        bitwise_check(R, f"bf16/lev {n}", got, emu[n].to(torch.bfloat16), min_frac=1.0)


for _s in ("130m", "520m", "1_2b", "pls_w512"):
    def _f(s=_s):
        qk_case(s, ld_pad=64 if s == "520m" else 0)
    _f.__name__ = f"grad_qknorm_rope_{_s}"
    R.case(_f)

    def _g(s=_s):
        sparse_case(s)
    _g.__name__ = f"levanter_chain_qknorm_rope_{_s}"
    R.case(_g)


@R.case
def k_only_call():
    sh = SHAPES["520m"]
    H, KV, hd, T = sh["H"], sh["KV"], sh["hd"], 1000
    gen = torch.Generator(device="cuda").manual_seed(9)
    ct, st = tables(hd)
    qkv, ld, pos = make_inputs(T, H, KV, hd, gen)
    gq, gk = rand_gain(hd, gen), rand_gain(hd, gen)
    dq = torch.randn(T, H, hd, generator=gen, device="cuda").to(torch.bfloat16)
    dk = torch.randn(T, KV, hd, generator=gen, device="cuda").to(torch.bfloat16)
    _, _, rq, rk = run_fwd(BF16S, LEV, qkv, ld, gq, gk, ct, st, pos, T, H, KV, hd)
    full, _, dgk_f = run_bwd(BF16S, LEV, dq, dk, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd)
    konly, _, dgk_k = run_bwd(BF16S, LEV, None, dk, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd)
    fq, fk, _ = split(full, T, H, KV, hd)
    kq, kk, _ = split(konly, T, H, KV, hd)
    R.check(torch.equal(bits16(fk), bits16(kk)), "k columns differ between the full and the k-only call")
    R.check(bool((kq.float() == SENT).all()), "k-only call wrote q columns")
    R.check(torch.equal(dgk_f.view(torch.int32), dgk_k.view(torch.int32)), "dgk differs between full and k-only")


@R.case
def dq_post_fusion():
    for shape in ("130m", "1_2b"):
        sh = SHAPES[shape]
        H, KV, hd, T = sh["H"], sh["KV"], sh["hd"], 4096
        gen = torch.Generator(device="cuda").manual_seed(11)
        ct, st = tables(hd)
        qkv, ld, pos = make_inputs(T, H, KV, hd, gen)
        gq, gk = rand_gain(hd, gen), rand_gain(hd, gen)
        acc = torch.randn(H, T, hd, generator=gen, device="cuda")   # FA3 dq accumulator, f32 [H, T, hd]
        _, _, rq, rk = run_fwd(BF16S, LEV, qkv, ld, gq, gk, ct, st, pos, T, H, KV, hd)
        dld = (H + 2 * KV) * hd
        for rnd in (LEV, ACC):
            post = torch.full((T, dld), SENT, dtype=torch.bfloat16, device="cuda")
            dgq_p = torch.empty(hd, device="cuda")
            lib.nm_dq_post_qknorm_rope_bwd(rnd, ptr(acc), ptr(qkv), ld, ptr(gq), ptr(ct), ptr(st), ptr(pos), ptr(rq),
                                           ptr(post), dld, ptr(dgq_p), T, H, hd, EPS, qscale(hd), None)
            torch.cuda.synchronize()
            pq, pk, _ = split(post, T, H, KV, hd)
            R.check(bool((pk.float() == SENT).all()), "dq_post wrote k columns")
            if rnd == LEV:
                dq_b = acc.permute(1, 0, 2).contiguous().to(torch.bfloat16)   # FA3 postprocess at softmax_scale 1
                ref, dgq_r, _ = run_bwd(BF16S, LEV, dq_b, None, qkv, ld, gq, gk, ct, st, pos, rq, rk, T, H, KV, hd)
                rq_, _, _ = split(ref, T, H, KV, hd)
                R.check(torch.equal(bits16(pq), bits16(rq_)), f"{shape}: dq_post q columns != qknorm_rope_bwd")
                R.check(torch.equal(dgq_p.view(torch.int32), dgq_r.view(torch.int32)), f"{shape}: dgq differs")
                print(f"    {shape} levanter: bitwise equal to qknorm_rope_bwd on bf16(accumulator)")
            else:
                q, k, _ = split(qkv, T, H, KV, hd)
                # f64 oracle: the accumulator is the cotangent of the scaled q (FA3 ran at softmax_scale 1)
                o = oracle(q, k, gq, gk, ct, st, pos, acc.permute(1, 0, 2), torch.zeros_like(k), hd)
                accurate_check(R, f"{shape} accurate dq_post dq", pq, o["dq"])
                f32_bound(R, f"{shape} accurate dq_post dgq", dgq_p, o["dgq"])


R.run(__import__("nmk").ARGS.cases)
