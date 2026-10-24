"""S3 gate: microbatch metadata (doc_meta), fixed-order reductions (reduce_partials_rows, sum_fixed) and the f32 -> bf16
cast. Every reference is numpy on the host.

doc_meta cases: EOS at a row start, at the last position, consecutive EOS (length-1 segments), a row without EOS, a row
with more than max_segs segments (overflow flag, first max_segs starts kept), an invalid row (weights 0), S in
{4096, 4224, 4480, 19456}.
"""
import numpy as np
import torch

from nmk import init, ptr, rel

lib, R = init("s3_misc")
EOS = 128001


def doc_meta_ref(tok, valid, eos, max_segs):
    rows, S = tok.shape
    targets = np.concatenate([tok[:, 1:], tok[:, :1]], 1)
    weight = np.ones((rows, S), np.float32)
    weight[:, -1] = 0
    weight[~valid] = 0
    pos = np.tile(np.arange(S, dtype=np.int32), (rows, 1))
    rss = np.full((rows, max_segs + 1), S, np.int32)
    cu, overflow = [], False
    for r in range(rows):
        starts = [0] + [p for p in range(1, S) if tok[r, p - 1] == eos]
        if len(starts) > max_segs:
            overflow = True
            starts = starts[:max_segs]
        rss[r, :len(starts)] = starts
        cu += [r * S + s for s in starts]
    n = len(cu)
    cu_full = np.full(rows * max_segs + 1, rows * S, np.int32)
    cu_full[:n] = cu
    return targets, weight, pos, rss, cu_full, n, overflow


def run_doc_meta(tok, valid, max_segs):
    rows, S = tok.shape
    t = torch.from_numpy(tok).cuda()
    v = torch.from_numpy(valid.astype(np.uint8)).cuda()
    T = rows * S
    targets = torch.empty(T, dtype=torch.int32, device="cuda")
    weight = torch.empty(T, device="cuda")
    pos = torch.empty(T, dtype=torch.int32, device="cuda")
    cu = torch.full((rows * max_segs + 1,), -1, dtype=torch.int32, device="cuda")
    nseg = torch.zeros(1, dtype=torch.int32, device="cuda")
    rss = torch.full((rows, max_segs + 1), -1, dtype=torch.int32, device="cuda")
    ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
    lib.nm_doc_meta(ptr(t), ptr(v), rows, S, EOS, max_segs, ptr(targets), ptr(weight), ptr(pos), ptr(cu), ptr(nseg),
                    ptr(rss), ptr(ovf), None)
    torch.cuda.synchronize()
    return (targets.cpu().numpy().reshape(rows, S), weight.cpu().numpy().reshape(rows, S),
            pos.cpu().numpy().reshape(rows, S), rss.cpu().numpy(), cu.cpu().numpy(), int(nseg.item()),
            bool(ovf.item()))


@R.case
def doc_meta_many_rows():
    """The pack kernel holds [rows + 1] ints in dynamic shared memory: rows past 12287 need more than 48 KiB (opt-in).
    rows = 12287, 12288, 16384 (the limit) pass; 16385 is rejected."""
    rng = np.random.default_rng(1)
    S, max_segs = 64, 4
    for rows in (12287, 12288, 16384):
        tok = rng.integers(0, 128000, size=(rows, S)).astype(np.int32)
        tok[rng.random((rows, S)) < 0.03] = EOS
        valid = rng.random(rows) > 0.01
        ref = doc_meta_ref(tok, valid, EOS, max_segs)
        got = run_doc_meta(tok, valid, max_segs)
        for n, a, b in zip(("targets", "weight", "pos", "row_seg_start", "cu_seqlens", "n_segments", "overflow"), got,
                           ref):
            R.check(np.array_equal(np.asarray(a), np.asarray(b)), f"rows={rows}: {n} differs")
        print(f"    rows={rows} S={S}: {ref[5]} segments, overflow={ref[6]}: all outputs equal the reference")
    try:
        run_doc_meta(np.zeros((16385, 8), np.int32), np.ones(16385, bool), 2)
        R.check(False, "rows = 16385 accepted")
    except RuntimeError as e:
        R.check("16384" in str(e), f"unexpected message: {e}")


@R.case
def doc_meta_cases():
    rng = np.random.default_rng(0)
    for S in (4096, 4224, 4480, 19456):
        rows, max_segs = 6, 24
        tok = rng.integers(0, 128000, size=(rows, S)).astype(np.int32)
        tok[0, 0] = EOS                       # EOS at the row start: segment [0, 1) then [1, ...)
        tok[0, S - 1] = EOS                   # EOS at the last position: no segment starts at S
        tok[1, 100:103] = EOS                 # consecutive EOS: length-1 segments
        tok[1, 2047] = EOS                    # a segment boundary at 2048 (tile edge)
        for p in rng.choice(S - 1, 30, replace=False):   # row 3: 31 segments > max_segs: overflow
            tok[3, p] = EOS
        tok[4, rng.choice(S - 1, 10, replace=False)] = EOS
        valid = np.ones(rows, bool)
        valid[5] = False                      # an invalid (padded eval) row
        ref = doc_meta_ref(tok, valid, EOS, max_segs)
        got = run_doc_meta(tok, valid, max_segs)
        names = ("targets", "weight", "pos", "row_seg_start", "cu_seqlens", "n_segments", "overflow")
        for n, a, b in zip(names, got, ref):
            ok = np.array_equal(np.asarray(a), np.asarray(b))
            R.check(ok, f"S={S}: {n} differs")
        print(f"    S={S}: {ref[5]} segments, overflow={ref[6]}: all outputs equal the reference")
    # no overflow case
    tok = rng.integers(0, 1000, size=(2, 4096)).astype(np.int32)
    got = run_doc_meta(tok, np.ones(2, bool), 24)
    R.check(not got[6] and got[5] == 2, "two rows without EOS must give 2 segments and no overflow")


@R.case
def reduce_partials():
    rng = np.random.default_rng(1)
    for nb, w in ((1, 64), (7, 100), (1024, 2048), (257, 512)):
        p = rng.standard_normal((nb, w)).astype(np.float32)
        out0 = rng.standard_normal(w).astype(np.float32)
        ref = p.astype(np.float64).sum(0)
        pt = torch.from_numpy(p).cuda()
        for acc in (0, 1):
            for rb in (0, 1):
                o = torch.from_numpy(out0.copy()).cuda()
                lib.nm_reduce_partials_rows(ptr(pt), nb, w, ptr(o), acc, rb, None)
                torch.cuda.synchronize()
                got = o.cpu().numpy().astype(np.float64) - (out0 if acc else 0)
                tol = 4e-3 if rb else 1e-6
                r = np.linalg.norm(got - ref) / np.linalg.norm(ref)
                R.check(r <= tol, f"nb={nb} w={w} acc={acc} round={rb}: rel {r:.2e}")
                if rb and not acc:
                    bits = o.cpu().numpy().view(np.uint32)
                    R.check(bool(((bits & 0xFFFF) == 0).all()), "round_bf16 left low mantissa bits")
        print(f"    nb={nb} w={w}: ok")


@R.case
def sum_fixed():
    rng = np.random.default_rng(2)
    for n in (0, 1, 1000, 1 << 20):
        x = rng.standard_normal(n).astype(np.float32)
        xt = torch.from_numpy(x).cuda()
        outs = []
        for _ in range(2):
            o = torch.full((1,), float("nan"), device="cuda")
            lib.nm_sum_fixed(ptr(xt), n, ptr(o), None)
            torch.cuda.synchronize()
            outs.append(o.item())
        ref = float(x.astype(np.float64).sum())
        R.check(outs[0] == outs[1] or (np.isnan(outs[0]) and np.isnan(outs[1])), "sum_fixed not repeatable")
        R.check(abs(outs[0] - ref) <= 1e-6 * max(1.0, float(np.abs(x).sum())), f"n={n}: {outs[0]} vs {ref}")
        print(f"    n={n}: {outs[0]:.6f} vs f64 {ref:.6f}")


@R.case
def cast_rne():
    x = torch.randn(1 << 20, device="cuda") * 1e3
    special = torch.tensor([float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-42, 3.3895314e38, 1.0 + 2 ** -8,
                            1.0 + 3 * 2 ** -8], device="cuda")
    x[: special.numel()] = special
    out = torch.empty(x.numel(), dtype=torch.bfloat16, device="cuda")
    lib.nm_cast_f32_bf16(ptr(x), ptr(out), x.numel(), None)
    torch.cuda.synchronize()
    R.check(torch.equal(out.view(torch.int16), x.to(torch.bfloat16).view(torch.int16)), "cast differs from RNE")


R.run(__import__("nmk").ARGS.cases)
