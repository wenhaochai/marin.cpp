"""S6 Newton-Schulz gates: the TF32 rounding probe, NS against an f64 reference, and T1.6 against the levanter
fixtures (kaiyue_baseline.md T1.6; BUILD_LOG "Decisions for S4-S6" items 14, 15).

  probe     512^3 f32 GEMM through the NS GEMM path: TF32 operands rounded to nearest, i.e. rel to the f64 product of
            TF32-RNE-rounded operands <= 1e-5 (levanter 1.2e-6 to 8.5e-6); truncation would give ~8.7e-4. f32: rel
            to the f64 product of the f32 operands <= 1e-6; 3xTF32 (small parts rounded to TF32 again, f32 tensor-core
            accumulation): <= 1e-5, the f32-dot bound of T1.6.
  ns_ref    nm_newton_schulz (f32, 3xtf32, tf32) on random matrices (square, wide, tall, odd sizes, batched) against
            the same iteration in f64: f32 rel <= 1e-5, 3xtf32 <= 5e-5; tf32 <= 1.5 x the error of an f64 emulation
            with TF32-RNA-rounded GEMM operands (the iteration amplifies single operand-rounding flips, so two TF32
            computations are independent samples of the same error, not near-copies); a wrong formula or layout
            gives O(1); the symmetric-block products (f32s, tf32s) to the same bounds. Two runs bitwise equal; a
            matrix's result does not depend on the batch it is in.
  t1_6      per leaf of 130m and 1_2b: u (levanter's Nesterov update after update 0) -> NS TF32 (full products and
            symmetric blocks); oracle rule with
            levanter at HIGHEST as the oracle: rel(nano, HIGHEST) <= 1.25 x rel(levanter DEFAULT, HIGHEST). Also
            reported: rel(nano, DEFAULT). The f32-policy paths against HIGHEST: true f32 must hold the f32-dot bound
            rel <= 1e-5; 3xTF32 is reported (DESIGN.md §4.2: the f32 policy uses 3xTF32 only where T1.6 shows it
            within 1e-5, else true f32).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s6lib import PREC, Lib, Runner, args, check, ptr, rel, stream, torch  # noqa: E402

import numpy as np  # noqa: E402

A, B, Cc, EPS = np.float32(3.4445), np.float32(-4.7750), np.float32(2.0315), np.float32(1e-5)
LEAVES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OUT = {}


def tf32_round(x, mode):
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if mode == "rne":
        u = u + 0xFFF + ((u >> 13) & 1)
    elif mode == "rna":
        u = u + 0x1000
    u = (u & 0xFFFFE000) & 0xFFFFFFFF
    u = torch.where(u >= 2**31, u - 2**32, u).to(torch.int32)
    return u.view(torch.float32)


def probe(lib):
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(512, 512, generator=g).cuda()
    b = torch.randn(512, 512, generator=g).cuda()
    res = {}
    for prec in ("tf32", "f32", "3xtf32"):
        d = torch.empty(512, 512, device="cuda")
        lib.nm_ns_gemm(PREC[prec], ptr(a), ptr(b), ptr(d), 512, 512, 512, stream())
        r = {"f32_operands": rel(d, a.double() @ b.double())}
        for mode in ("rne", "rna", "rz"):
            r["tf32_" + mode] = rel(d, tf32_round(a, mode).double() @ tf32_round(b, mode).double())
        res[prec] = r
        print(f"  probe {prec}: " + ", ".join(f"{k} {v:.3e}" for k, v in r.items()))
    OUT["probe"] = res
    check(res["tf32"]["tf32_rne"] <= 1e-5, f"TF32 rel to RNE-rounded operands {res['tf32']['tf32_rne']:.3e} > 1e-5")
    check(res["tf32"]["tf32_rz"] > 1e-4, "TF32 product looks truncated")
    check(res["f32"]["f32_operands"] <= 1e-6, f"f32 rel {res['f32']['f32_operands']:.3e} > 1e-6")
    check(res["3xtf32"]["f32_operands"] <= 1e-5, f"3xTF32 rel {res['3xtf32']['f32_operands']:.3e} > 1e-5")


def ns_reference(x, steps, tf32):
    """levanter's zeropower_via_newtonschulz5 ("simple") in f64; tf32: GEMM operands TF32-RNA-rounded from f32."""
    a, b, c = float(A), float(B), float(Cc)
    out = []
    for m in x:
        X = m.double()
        X = X / (torch.linalg.vector_norm(X) + float(EPS))
        tr = X.shape[0] > X.shape[1]
        if tr:
            X = X.T
        for _ in range(steps):
            if tf32:
                r = lambda t: tf32_round(t.float(), "rna").double()  # noqa: E731
                Am = r(X) @ r(X).T
                Bm = b * Am + r(c * Am) @ r(Am)
                X = a * X + r(Bm) @ r(X)
            else:
                Am = X @ X.T
                Bm = b * Am + (c * Am) @ Am
                X = a * X + Bm @ X
        out.append(X.T if tr else X)
    return torch.stack(out)


def ns_run(lib, x, prec, steps=5):
    """prec f32 / tf32 / 3xtf32; suffix s: the symmetric-block products, suffix c: c folded into GEMM2's alpha
    (ns_gemm.hpp; nm_newton_schulz's `symmetric` argument carries them as bits 1 and 2)."""
    y = x.clone().contiguous()
    n, r, c = y.shape
    base = prec.rstrip("sc")
    flags = (1 if "s" in prec[len(base):] else 0) | (2 if "c" in prec[len(base):] else 0)
    lib.nm_newton_schulz(PREC[base], flags, ptr(y), n, r, c, steps, A, B, Cc, EPS, None, stream())
    return y


def ns_ref(lib):
    g = torch.Generator(device="cpu").manual_seed(1)
    res = {}
    for (n, r, c) in [(3, 512, 512), (2, 1792, 512), (2, 512, 1792), (1, 100, 37), (2, 64, 300), (1, 33, 65)]:
        x = torch.randn(n, r, c, generator=g).cuda()
        ref = ns_reference(x, 5, False)
        emu = rel(ns_reference(x, 5, True), ref)
        res[f"{n}x{r}x{c}/tf32_emulation"] = emu
        for prec in ("f32", "3xtf32", "tf32", "f32s", "tf32s", "f32c", "tf32c"):
            e = rel(ns_run(lib, x, prec), ref)
            res[f"{n}x{r}x{c}/{prec}"] = e
            bound = 5e-5 if prec == "3xtf32" else (1e-5 if prec.startswith("f32") else 1.5 * emu)
            check(e <= bound, f"{n}x{r}x{c} {prec}: rel to the f64 reference {e:.3e} > {bound:.3e}")
        for prec in ("tf32", "tf32s", "tf32c"):
            y1 = ns_run(lib, x, prec)
            check(torch.equal(y1, ns_run(lib, x, prec)), f"{n}x{r}x{c} {prec}: two runs differ")
            if n > 1:   # a matrix gets the same bits in any batch
                check(torch.equal(y1[1], ns_run(lib, x[1:2], prec)[0]), f"{n}x{r}x{c} {prec}: batch-dependent bits")
    for k, v in res.items():
        print(f"  {k}: {v:.3e}")
    OUT["ns_ref"] = res


def t1_6(lib, src, size):
    d = os.path.join(src, "cache", "levref", size, "t1_precision", "ns")
    if not os.path.isdir(d):
        return "skip"
    rows = {}
    bad = []
    for leaf in LEAVES:
        u = torch.from_numpy(np.load(os.path.join(d, f"u.{leaf}.npy"))).cuda()
        hi = torch.from_numpy(np.load(os.path.join(d, f"ns_highest.{leaf}.npy"))).cuda()
        de = torch.from_numpy(np.load(os.path.join(d, f"ns_default.{leaf}.npy"))).cuda()
        row = {"shape": list(u.shape), "lev_default_vs_highest": rel(de, hi)}
        for prec in ("tf32", "tf32s", "tf32c", "3xtf32", "f32", "f32s", "f32c"):
            y = ns_run(lib, u[None], prec)[0]
            row[f"{prec}_vs_highest"] = rel(y, hi)
            row[f"{prec}_vs_default"] = rel(y, de)
        row["ratio"] = row["tf32_vs_highest"] / row["lev_default_vs_highest"]
        row["ratio_sym"] = row["tf32s_vs_highest"] / row["lev_default_vs_highest"]
        row["ratio_fold"] = row["tf32c_vs_highest"] / row["lev_default_vs_highest"]
        rows[leaf] = row
        print(f"  {size} {leaf:9s} {tuple(u.shape)}: lev DEFAULT {row['lev_default_vs_highest']:.3e}  "
              f"nano TF32 {row['tf32_vs_highest']:.3e} (x{row['ratio']:.3f}; vs DEFAULT {row['tf32_vs_default']:.3e})"
              f"  symmetric x{row['ratio_sym']:.3f}  fold-c x{row['ratio_fold']:.3f}  3xTF32 {row['3xtf32_vs_highest']:.3e}  f32 {row['f32_vs_highest']:.3e}"
              f" (symmetric {row['f32s_vs_highest']:.3e})")
        for key in ("ratio", "ratio_sym", "ratio_fold"):
            if not row[key] <= 1.25:
                bad.append(f"{leaf} TF32 {key} {row[key]:.3f} > 1.25")
        for prec in ("f32", "f32s", "f32c"):
            if not row[f"{prec}_vs_highest"] <= 1e-5:
                bad.append(f"{leaf} {prec} {row[prec + '_vs_highest']:.3e} > 1e-5")
    x3 = max(r["3xtf32_vs_highest"] for r in rows.values())
    print(f"  {size}: 3xTF32 max rel to HIGHEST {x3:.3e}: " +
          ("within 1e-5" if x3 <= 1e-5 else "outside 1e-5, so the f32 policy runs true f32 (DESIGN.md §4.2)"))
    rows["3xtf32_within_1e-5"] = x3 <= 1e-5
    OUT["t1_6_" + size] = rows
    check(not bad, "; ".join(bad))


def main():
    a = args()
    lib = Lib(a.lib)
    r = Runner("s6_ns")
    only = set(a.only.split(",")) if a.only else None
    cases = [("probe", lambda: probe(lib)), ("ns_ref", lambda: ns_ref(lib)),
             ("t1_6_130m", lambda: t1_6(lib, a.src, "130m")), ("t1_6_1_2b", lambda: t1_6(lib, a.src, "1_2b"))]
    for name, fn in cases:
        if only is None or name in only:
            r.case(name, fn)
    OUT["device"] = torch.cuda.get_device_name()
    if a.json:
        json.dump(OUT, open(a.json, "w"), indent=1)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
