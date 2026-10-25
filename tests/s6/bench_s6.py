"""S6 bench (report only): time of one optimizer step of the engine per size, on synthetic data in levanter's
stacked layout, per BUILD_LOG "Decisions for S4-S6" item 2 (1 GPU, 5 warm-up and 20 timed calls, median, CUDA events).

Rows per size and NS precision (TF32 = the decided default; f32 = the f32 policy; tf32s = symmetric NS blocks, an
option):
  full     every tensor on one GPU (replicated optimizer, optim.zero1 = off)
  ranks 4  each rank's share under ZeRO-1 (DESIGN.md §6.1) by the LPT rule of plan_ownership_specs (engine.cpp, same
           units, costs and order; s6_ownership gates the C++ one): whole MuonH layer matrices, vocab rows split
           evenly, every other tensor (one per layer, as the model stores the gains) whole; reported as the slowest
           rank (the step waits for it) and rank 0 (the all-reduces between phases are not timed)
and per phase (sumsq; update = clip, Adam, MuonH, AdamH pass 1; AdamH pass 2; AdamH pass 3), plus the NS GEMM
algorithm choice (cuBLASLt's first heuristic vs a timed search). --only takes sizes, or cases size:prec:ranks:timed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s6lib import PREC, Lib, OptGroup, OptTensor, args, ptr, stream, torch  # noqa: E402

import ctypes as C  # noqa: E402

SIZES = {  # D, L, H, KV, hd, F
    "130m": (512, 6, 8, 8, 64, 1792), "300m": (768, 12, 12, 12, 64, 2688),
    "520m": (1024, 24, 16, 8, 64, 3584), "1_2b": (2048, 16, 16, 8, 128, 7168)}
V = 128256


def tensors(size):
    D, L, H, KV, hd, F = SIZES[size]
    out = []   # (group, rows, cols, n_mats, mat_rows)
    for out_rows, cols in ((H * hd, D), (KV * hd, D), (KV * hd, D), (D, H * hd), (F, D), (F, D), (D, F)):
        out.append(("muonh", L * out_rows, cols, L, out_rows))
    out.append(("adamh", V, D, 0, 0))
    out.append(("adam", V, D, 0, 0))
    for _ in range(4):
        out.append(("adam", L, D, 0, 0))
    out.append(("adam", L, hd, 0, 0))
    out.append(("adam", L, hd, 0, 0))
    out.append(("adam", 1, D, 0, 0))
    return out


def ns_seconds(out_rows, in_cols):
    m, n = min(out_rows, in_cols), max(out_rows, in_cols)
    return 5.0 * (4.0 * m * m * n + 2.0 * m * m * m) / 4e14


def ew_seconds(numel, bytes_per_elem):
    return numel * bytes_per_elem / 3e12


def lpt_owned(size, nranks):
    """owned[rank][tensor] = row ranges, by plan_ownership_specs' rule: units in the model's parameter order (per layer
    q, k, v, o, gate, up, down, the four gains, q_norm, k_norm; then the final gain), largest estimated H100 time first
    onto the least-loaded rank (ties to the lower rank); vocab-sized tensors split evenly by rows first."""
    D, L = SIZES[size][0], SIZES[size][1]
    specs = tensors(size)
    owned = [[[] for _ in specs] for _ in range(nranks)]
    if nranks == 1:
        return [[[(0, s[1])] for s in specs]]
    load = [0.0] * nranks
    units = []   # (cost, tensor, rows)
    for l in range(L):
        for i in range(7):   # MuonH leaves of layer l
            _, _, cols, _, mr = specs[i]
            units.append((ns_seconds(mr, cols) + ew_seconds(mr * cols, 40), i, (l * mr, (l + 1) * mr)))
        for i in range(9, 15):   # per-layer gains: rows of the stacked [L, D] / [L, hd] tensors
            units.append((ew_seconds(specs[i][2], 28) + 2e-6, i, (l, l + 1)))
    units.append((ew_seconds(specs[15][2], 28) + 2e-6, 15, (0, 1)))
    for i in (8, 7):   # embedding (adam), lm_head (adamh): row split
        grp, rows, cols = specs[i][:3]
        per = (rows + nranks - 1) // nranks
        for r in range(nranks):
            b, e = min(rows, r * per), min(rows, (r + 1) * per)
            if b < e:
                owned[r][i].append((b, e))
            load[r] += ew_seconds((e - b) * cols, 44 if grp == "adamh" else 28)
    for cost, i, rows in sorted(units, key=lambda u: -u[0]):   # stable: equal costs keep their order
        r = min(range(nranks), key=lambda j: load[j])
        load[r] += cost
        owned[r][i].append(rows)
    return owned


def alloc(size):
    bufs = []
    for grp, rows, cols, n_mats, mat_rows in tensors(size):
        d = {k: torch.randn(rows, cols, device="cuda") * s for k, s in (("p", 0.02), ("g", 1e-3), ("s0", 1e-3))}
        d["s1"] = torch.rand(rows, cols, device="cuda") * 1e-6
        bufs.append(d)
    return bufs


def bench(lib, size, bufs, prec, owned, timed_gemm):
    """One engine over the row ranges `owned[tensor]` (tensors without rows are left out)."""
    specs = tensors(size)
    tens, keep = [], []
    for i, (grp, rows, cols, n_mats, mat_rows) in enumerate(specs):
        own = sorted(owned[i])
        if not own:
            continue
        d = bufs[i]
        t = OptTensor()
        t.group = ("muonh", "adamh", "adam").index(grp)
        t.rows, t.cols, t.base_row = rows, cols, 0
        t.p32, t.p16, t.g, t.s0, t.s1 = ptr(d["p"]), None, ptr(d["g"]), ptr(d["s0"]), ptr(d["s1"])
        t.grad_scale = t.update_scale = 1.0
        t.n_mats, t.mat_rows, t.row_map = n_mats, mat_rows, None
        arr = (C.c_int64 * (2 * len(own)))(*[x for r in own for x in r])
        keep.append(arr)
        t.n_owned, t.owned = len(own), C.cast(arr, C.c_void_p)
        t.u_out = None
        tens.append(t)
    ta = (OptTensor * len(tens))(*tens)
    gs = (OptGroup * 3)()
    for i in range(3):
        gs[i].rule, gs[i].lr_table, gs[i].clip_group = i, 1 if i == 2 else 0, i
        gs[i].lr_scale, gs[i].b1, gs[i].b2, gs[i].eps, gs[i].momentum = 1.0, 0.9, 0.98, 1e-15, 0.98
        gs[i].max_grad_norm, gs[i].weight_decay = 1.0, 0.0
    n = 64
    lr = torch.full((n,), 0.01, device="cuda")
    alr = torch.full((n,), 0.002, device="cuda")
    bc1 = torch.full((n,), 0.5, device="cuda")
    bc2 = torch.full((n,), 0.5, device="cuda")
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    eng = C.c_void_p()
    lib.nm_opt_create(gs, 3, ta, len(tens), PREC[prec], 5, 1e-5, 0, timed_gemm, 0, C.byref(eng))
    try:
        lib.nm_opt_set_tables(eng, ptr(lr), ptr(alr), ptr(bc1), ptr(bc2), ptr(step), n)
        lib.nm_opt_prepare(eng, stream())
        nmat, nch = C.c_int(), C.c_int()
        lib.nm_opt_buffers(eng, None, None, None, None, None, None, C.byref(nmat), C.byref(nch))
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        times = []
        for it in range(25):
            step.zero_()
            ev[0].record()
            lib.nm_opt_phase(eng, 0, stream())
            ev[1].record()
            lib.nm_opt_phase(eng, 1, stream())
            ev[2].record()
            lib.nm_opt_phase(eng, 2, stream())
            ev[3].record()
            lib.nm_opt_phase(eng, 3, stream())
            ev[4].record()
            torch.cuda.synchronize()
            if it >= 5:
                times.append([ev[j].elapsed_time(ev[j + 1]) for j in range(4)])
        t = torch.tensor(times)
        med = t.median(dim=0).values.tolist()
        total = t.sum(dim=1).median().item()
    finally:
        lib.nm_opt_destroy(eng)
    return dict(n_mats=nmat.value, n_chunks=nch.value, total_ms=total, sumsq_ms=med[0], update_ms=med[1],
                adamh_w_ms=med[2], adamh_apply_ms=med[3])


def main():
    a = args()
    lib = Lib(a.lib)
    rows = []
    if a.only and ":" in a.only:   # explicit cases size:prec:ranks:timed[,...]
        cases = [(s, p, int(n), int(t)) for s, p, n, t in (c.split(":") for c in a.only.split(","))]
    else:
        cases = [(size, prec, nranks, timed) for size in (a.only.split(",") if a.only else list(SIZES))
                 for prec in ("tf32", "tf32s", "f32") for nranks in (1, 4)
                 for timed in ((0, 1) if prec.startswith("tf32") else (0,))
                 if not (prec.startswith("f32") and size in ("520m", "1_2b") and nranks == 1)]
    bufs, bufs_size = None, None
    for size, prec, nranks, timed in cases:
        if bufs_size != size:
            bufs = None
            torch.cuda.empty_cache()
            bufs, bufs_size = alloc(size), size
        per_rank = [bench(lib, size, bufs, prec, own, timed) for own in lpt_owned(size, nranks)]
        slow = max(range(nranks), key=lambda j: per_rank[j]["total_ms"])
        r = dict(per_rank[slow], size=size, prec=prec, ranks=nranks, timed_gemm=timed, slowest_rank=slow,
                 rank0_ms=per_rank[0]["total_ms"], per_rank_ms=[x["total_ms"] for x in per_rank])
        rows.append(r)
        print(f"{size:5s} {prec:5s} ranks {nranks} timed {timed}: {r['total_ms']:8.2f} ms (slowest rank {slow}; rank 0 "
              f"{r['rank0_ms']:.2f}; sumsq {r['sumsq_ms']:.2f}, update {r['update_ms']:.2f}, adamh w "
              f"{r['adamh_w_ms']:.2f}, apply {r['adamh_apply_ms']:.2f}; {r['n_mats']} matrices, {r['n_chunks']} NS "
              f"chunks)", flush=True)
        torch.cuda.empty_cache()
    out = dict(device=torch.cuda.get_device_name(), rows=rows)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
