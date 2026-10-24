"""S3 gate: SwiGLU forward and backward on the interleaved gate/up layout (64-column blocks), at the owner's MLP widths.

Checks (DESIGN.md §10.2): f32 storage rel(kernel, f64 autograd) <= 1e-6 (both rounding modes); bf16 levanter bitwise
equal to the levanter emulation on every element (SwiGLU has no reduction, so nothing may differ); bf16 accurate
(DESIGN.md §3.0: f32 inside, one rounding per output) rel(kernel, f64) <= 1.05 x rel(bf16(f64), f64), the levanter
emulation failing that bound, and per-row report; gate values up to |g| = 100 (exp overflow and
saturation) included.
"""
import torch

import levemu as E
from nmk import ACC, BF16S, F32S, LEV, SHAPES, accurate_check, bitwise_check, init, ptr, rel, row_report

lib, R = init("s3_swiglu")


def run(storage, rounding, gu, dh, T, F):
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    gu_s, dh_s = gu.to(dt).contiguous(), dh.to(dt).contiguous()
    h = torch.empty(T, F, dtype=dt, device="cuda")
    dgu = torch.empty(T, 2 * F, dtype=dt, device="cuda")
    lib.nm_swiglu_fwd(storage, rounding, ptr(gu_s), ptr(h), T, F, None)
    lib.nm_swiglu_bwd(storage, rounding, ptr(dh_s), ptr(gu_s), ptr(dgu), T, F, None)
    torch.cuda.synchronize()
    return h, dgu


def inputs(T, F, gen):
    g = 2.0 * torch.randn(T, F, generator=gen, device="cuda")
    g.view(-1)[:: 997] = torch.empty(g.view(-1)[:: 997].shape, device="cuda").uniform_(-100, 100, generator=gen)
    u = torch.randn(T, F, generator=gen, device="cuda")
    dh = torch.randn(T, F, generator=gen, device="cuda")
    g, u, dh = (t.to(torch.bfloat16) for t in (g, u, dh))
    return g, u, dh, E.interleave_gu(g, u)


def case(shape, T):
    F = SHAPES[shape]["F"]
    gen = torch.Generator(device="cuda").manual_seed(F)
    g, u, dh, gu = inputs(T, F, gen)
    g64 = g.double().requires_grad_(True)
    u64 = u.double().requires_grad_(True)
    h64 = torch.nn.functional.silu(g64) * u64
    h64.backward(dh.double())
    o = dict(h=h64.detach(), dg=g64.grad, du=u64.grad)
    he = E.lev_swiglu_fwd(g.float(), u.float())
    dge, due = E.lev_swiglu_bwd(g.float(), u.float(), dh.float())
    emu = dict(h=he, dg=dge, du=due)
    print(f"  shape {shape} T={T} F={F}")
    for storage in (F32S, BF16S):
        for rnd in (LEV, ACC):
            tag = ("f32/" if storage == F32S else "bf16/") + ("lev" if rnd == LEV else "acc")
            h, dgu = run(storage, rnd, gu, dh, T, F)
            dg, du = E.split_gu(dgu)
            got = dict(h=h, dg=dg, du=du)
            for n in got:
                if storage == F32S:
                    r = rel(got[n], o[n])
                    print(f"    {tag} {n}: rel(f32 kernel, f64) {r:.3e}")
                    R.check(r <= 1e-6, f"{tag} {n}: rel {r:.3e} > 1e-6")
                    continue
                if rnd == LEV:
                    bitwise_check(R, f"{tag} {n} vs emulation", got[n], emu[n].to(torch.bfloat16), o[n], min_frac=1.0)
                    rk, re = rel(got[n], o[n]), rel(emu[n], o[n])
                    print(f"    {tag} {n}: rel(kernel, f64) {rk:.4e}  rel(levanter emulation, f64) {re:.4e}")
                    R.check(rk <= 1.25 * re, f"{tag} {n}: oracle rule")
                else:
                    rk = accurate_check(R, f"{tag} {n}", got[n], o[n], lev=emu[n])
                R.check(row_report(f"{tag} {n}", got[n], o[n], F, rk)[0], f"{tag} {n}: per-row bound")


for _s, _T in (("130m", 8192), ("300m", 4096), ("520m", 4096), ("1_2b", 2048), ("pls_w512", 8192)):
    def _f(s=_s, T=_T):
        case(s, T)
    _f.__name__ = f"grad_swiglu_{_s}"
    R.case(_f)


@R.case
def rejects_bad_F():
    x = torch.zeros(4, 200, dtype=torch.bfloat16, device="cuda")
    try:
        lib.nm_swiglu_fwd(BF16S, LEV, ptr(x), ptr(x), 4, 100, None)
        R.check(False, "F = 100 accepted")
    except RuntimeError as e:
        R.check("multiple of 64" in str(e), f"unexpected message: {e}")


R.run(__import__("nmk").ARGS.cases)
