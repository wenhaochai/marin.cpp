"""S4 attention gates (DESIGN.md §3.4, §10.2): the portable reference attention against f64 (sm_80 and sm_90), and every
FA3 launch mode against the reference at the owner shapes on the real segment rows (sm_90 only).

    python tests/s4/test_attention.py --lib build/libmarincpp_kernels.so --src . [case ...]

Bounds
  reference, f32 storage vs f64 (same inputs): rel <= 1e-6 per tensor (f32 math, DESIGN.md §10.2), rows <= 10x.
  reference, bf16 storage: O within 1.05x the single-rounding floor rel(bf16(f64), f64) (one rounding at the store);
  dq, dk, dv within 1.25x (their D = rowsum(dO O) reads the stored bf16 O, as FA3's does); L (f32) rel <= 1e-6.
  FA3 vs the reference run in f32 storage on the same bf16 inputs (the oracle at full size): rel <= 2.5x the floor
  rel(bf16(oracle), oracle) for O, dq, dk, dv (E1 measured 1.2-1.9x for FA3), L rel <= 1e-5; rows <= 10x the tensor
  rel. Deterministic mode: two runs bitwise equal; persistent and sm_grid share the summation order (bitwise equal).
  Production entry points (capi.h nm_attention_*: make_attention, fa3::fwd / bwd, the scheduler mapping): with
  deterministic on bitwise equal to the mode they must run (persistent -> the work list on one CTA per slot,
  doc_causal -> doc_causal); with it off within the FA3 bounds; ref_f32 bitwise equal to nm_ref_attention_*.
  GQA group size 4 (H 16, KV 4, hd 64 and 128; check_shape accepts any H % KV == 0) runs with the same bounds.
"""
import numpy as np
import torch

import nmk4
from nmk4 import ATTN_SHAPES, Attn, Segs, rel

lib, R = nmk4.init("s4_attention")
ARGS = nmk4.nmk.ARGS
PLAN_MAX_SEGS = 24      # the planner's bound (DESIGN.md §2.5) sizes the segment slots
GQA4_SHAPES = {"g4.hd64": (16, 4, 64), "g4.hd128": (16, 4, 128)}   # G = 4: not an owner shape, but accepted


def arch():
    return lib.nm_device_arch()


def bounded(name, k, orc, factor, floor_of=None, tol=None):
    """rel(k, orc) <= factor x floor (floor = rel(bf16(orc), orc)) or <= tol; per-row report (rows of the last dim)."""
    r = rel(k, orc)
    if tol is not None:
        bound = tol
        msg = f"    {name}: rel {r:.3e} (bound {tol:.1e})"
    else:
        fl = rel((floor_of if floor_of is not None else orc).float().to(torch.bfloat16), orc)
        bound = factor * fl
        msg = f"    {name}: rel {r:.3e}  floor {fl:.3e}  ratio {r / fl:.3f} (bound {factor})"
    print(msg)
    R.check(r <= bound, f"{name}: rel {r:.3e} > {bound:.3e}")
    width = k.shape[-1] if k.dim() > 1 else 64
    ok, worst = nmk4.row_report(name, k, orc, width, r)
    R.check(ok, f"{name}: a row exceeds 10x the tensor rel")
    return r


def small_cases():
    gen_np = np.random.default_rng(7)
    out = []
    for name, (H, KV, hd) in {**ATTN_SHAPES, **GQA4_SHAPES}.items():
        seg = nmk4.synthetic_segments(2, 1024, gen_np)
        out.append((name, H, KV, hd, seg))
    return out


@R.case
def ref_f32_storage_vs_f64():
    gen = torch.Generator(device="cuda").manual_seed(1)
    for name, H, KV, hd, seg in small_cases():
        segs = Segs(seg)
        a = Attn(lib, segs, H, KV, hd, gen, dtype=torch.float32)
        o = a.new_out()
        a.ref(o)
        torch.cuda.synchronize()
        orc = nmk4.oracle_f64(a.q, a.k, a.v, a.dO, segs, H, KV, hd, a.scale)
        print(f"  {name}: H {H} KV {KV} hd {hd}, {segs.n} segments (min length {segs.lengths.min()})")
        for t in ("O", "L", "dq", "dk", "dv"):
            bounded(f"{name}.{t}", o[t], orc[t], None, tol=1e-6)


@R.case
def ref_bf16_storage_vs_f64():
    gen = torch.Generator(device="cuda").manual_seed(2)
    for name, H, KV, hd, seg in small_cases():
        segs = Segs(seg)
        a = Attn(lib, segs, H, KV, hd, gen, dtype=torch.bfloat16)
        o = a.new_out()
        a.ref(o)
        torch.cuda.synchronize()
        orc = nmk4.oracle_f64(a.q, a.k, a.v, a.dO, segs, H, KV, hd, a.scale)
        print(f"  {name}")
        bounded(f"{name}.O", o["O"], orc["O"], 1.05)
        bounded(f"{name}.L", o["L"], orc["L"], None, tol=1e-6)
        for t in ("dq", "dk", "dv"):
            bounded(f"{name}.{t}", o[t], orc[t], 1.25)


@R.case
def ref_deterministic_and_tables():
    """Two runs bitwise equal; segments from cu_seqlens and from row_seg_start give the same bits."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    for name, H, KV, hd, seg in small_cases():
        segs = Segs(seg, max_segs=PLAN_MAX_SEGS)
        a = Attn(lib, segs, H, KV, hd, gen)
        o1, o2, o3 = a.new_out(), a.new_out(), a.new_out()
        a.ref(o1)
        a.ref(o2)
        a.ref(o3, use_cu=False)
        torch.cuda.synchronize()
        for t in o1:
            R.check(torch.equal(o1[t], o2[t]), f"{name}.{t}: two runs differ")
            R.check(torch.equal(o1[t], o3[t]), f"{name}.{t}: cu_seqlens vs row_seg_start differ")
        print(f"  {name}: bitwise repeatable, tables agree")


def owner_segments(src):
    seg = nmk4.real_segments(src)
    if seg is None:
        print("  (bench/e1_attention/segments absent: synthetic segments)")
        seg = nmk4.synthetic_segments(32, 4096, np.random.default_rng(11), edge_cases=False)
    return seg


def fa3_modes_at(name, H, KV, hd, seg, gen, modes, dets, with_f64_rows=True, api=False):
    segs = Segs(seg, max_segs=PLAN_MAX_SEGS)
    a = Attn(lib, segs, H, KV, hd, gen)
    # oracle at full size: the reference in f32 storage on the same bf16 values
    a32 = Attn(lib, segs, H, KV, hd, gen, dtype=torch.float32, inputs=(a.q.float(), a.k.float(), a.v.float(),
                                                                       a.dO.float()))
    orc = a32.new_out()
    a32.ref(orc)
    torch.cuda.synchronize()
    print(f"  {name}: H {H} KV {KV} hd {hd}, rows {segs.rows} x S {segs.S}, {segs.n} segments (max {segs.max_in_data} "
          f"per row, longest {segs.max_len}), slots {segs.rows * segs.max_segs}")
    if with_f64_rows:   # the f32 reference against f64 on two rows of the real shape
        rows = [0, 1]
        f64 = nmk4.oracle_f64(a.q, a.k, a.v, a.dO, segs, H, KV, hd, a.scale, rows=rows)
        idx = nmk4.token_rows(segs, rows)
        for t in ("O", "dq", "dk", "dv"):
            bounded(f"{name}.ref32.{t} rows 0-1 vs f64", orc[t][idx].contiguous(), f64[t][idx], None, tol=1e-6)
        bounded(f"{name}.ref32.L rows 0-1 vs f64", orc["L"][:, idx].contiguous(), f64["L"][:, idx], None, tol=1e-6)
    results = {}
    for mode in modes:
        for det in dets:
            ws = a.fa3_ws(mode)
            o = a.new_out()
            a.fa3_fwd(mode, o, ws)
            a.fa3_bwd(mode, o, ws, det)
            torch.cuda.synchronize()
            tag = f"{name}.{mode}.det{det}"
            bounded(f"{tag}.O", o["O"], orc["O"], 2.5)
            bounded(f"{tag}.L", o["L"], orc["L"], None, tol=1e-5)
            for t in ("dq", "dk", "dv"):
                bounded(f"{tag}.{t}", o[t], orc[t], 2.5)
            if det:
                o2 = a.new_out()
                a.fa3_fwd(mode, o2, ws)
                a.fa3_bwd(mode, o2, ws, det)
                torch.cuda.synchronize()
                same = all(torch.equal(o[t], o2[t]) for t in o)
                print(f"    {tag}: second run bitwise equal: {same}")
                R.check(same, f"{tag}: deterministic mode, two runs differ")
            results[(mode, det)] = o
    if ("persistent", 1) in results and ("sm_grid", 1) in results:
        same = all(torch.equal(results[("persistent", 1)][t], results[("sm_grid", 1)][t]) for t in ("dq", "dk", "dv"))
        print(f"    {name}: persistent and sm_grid (deterministic) bitwise equal: {same}")
        R.check(same, f"{name}: persistent vs sm_grid deterministic differ")
    if api:   # the production entry points run the mode of their scheduler
        for sched in nmk4.SCHEDS:
            for det in dets:
                ws = a.api_ws("fa3", sched)
                o = a.new_out()
                a.api_fwd("fa3", sched, o, ws)
                a.api_bwd("fa3", sched, o, ws, det)
                torch.cuda.synchronize()
                tag = f"{name}.api.{sched}.det{det}"
                if det and (sched, 1) in results:
                    same = all(torch.equal(o[t], results[(sched, 1)][t]) for t in o)
                    print(f"    {tag}: bitwise equal to FA3 mode {sched}: {same}")
                    R.check(same, f"{tag}: production entry point differs from mode {sched}")
                else:
                    bounded(f"{tag}.O", o["O"], orc["O"], 2.5)
                    for t in ("dq", "dk", "dv"):
                        bounded(f"{tag}.{t}", o[t], orc[t], 2.5)


@R.case
def fa3_modes_small():
    """Every FA3 mode on two synthetic rows with edge-case segments (length 1, tile-edge ends), all shapes."""
    if arch() != 90:
        raise nmk4.Skip("FA3 needs sm_90")
    gen = torch.Generator(device="cuda").manual_seed(5)
    for name, H, KV, hd, seg in small_cases():
        fa3_modes_at(name, H, KV, hd, seg, gen, list(nmk4.FA3_MODES), [0, 1], with_f64_rows=False)


@R.case
def fa3_owner_shapes():
    """Every FA3 mode, deterministic on and off, at the 130m / 300m / 520m / 1.2B shapes on the real segment rows."""
    if arch() != 90:
        raise nmk4.Skip("FA3 needs sm_90")
    seg = owner_segments(ARGS.src)
    gen = torch.Generator(device="cuda").manual_seed(6)
    for name, (H, KV, hd) in ATTN_SHAPES.items():
        fa3_modes_at(name, H, KV, hd, seg, gen, list(nmk4.FA3_MODES), [0, 1], api=True)


@R.case
def fa3_gqa_group4():
    """G = H / KV = 4 (hd 64 and 128) on the real segment rows: the training modes and the SM-count grid,
    deterministic on and off. With G >= 3 a dK / dV tile is added to by several non-first heads in order."""
    if arch() != 90:
        raise nmk4.Skip("FA3 needs sm_90")
    seg = owner_segments(ARGS.src)
    gen = torch.Generator(device="cuda").manual_seed(10)
    for name, (H, KV, hd) in GQA4_SHAPES.items():
        fa3_modes_at(name, H, KV, hd, seg, gen, ["persistent", "doc_causal", "sm_grid"], [0, 1], with_f64_rows=False,
                     api=True)


@R.case
def production_ref_f32():
    """The ref_f32 backend through the production entry points (make_attention) equals nm_ref_attention_* bitwise."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    for name, H, KV, hd, seg in small_cases():
        segs = Segs(seg, max_segs=PLAN_MAX_SEGS)
        a = Attn(lib, segs, H, KV, hd, gen)
        o1, o2 = a.new_out(), a.new_out()
        a.ref(o1)
        ws = a.api_ws("ref_f32", "doc_causal")
        a.api_fwd("ref_f32", "doc_causal", o2, ws)
        a.api_bwd("ref_f32", "doc_causal", o2, ws, 1)
        torch.cuda.synchronize()
        same = all(torch.equal(o1[t], o2[t]) for t in o1)
        print(f"  {name}: nm_attention_* (ref_f32) == nm_ref_attention_*: {same}")
        R.check(same, f"{name}: ref_f32 through make_attention differs")


@R.case
def fa3_full_slots():
    """Rows whose real segment count fills every slot (24 segments per row, max_segs 24) run in the training modes.
    Contract (k_doc_meta, BUILD_LOG S4 request): a row never has more than max_segs segments (k_doc_meta keeps the
    first max_segs starts and flags the overflow), so n_segs <= rows * max_segs and cu_seqlens[rows * max_segs] = T."""
    if arch() != 90:
        raise nmk4.Skip("FA3 needs sm_90")
    gen_np = np.random.default_rng(9)
    seg = np.zeros((2, 2048), np.int32)
    for r in range(2):   # 24 segments per row, lengths 1 .. 200
        cuts = np.sort(gen_np.choice(np.arange(1, 2048), 23, replace=False))
        prev = 0
        for i, c in enumerate(list(cuts) + [2048]):
            seg[r, prev:c] = i
            prev = c
    gen = torch.Generator(device="cuda").manual_seed(8)
    fa3_modes_at("520m.24segs", 16, 8, 64, seg, gen, ["persistent", "doc_causal"], [0, 1], with_f64_rows=False)


R.run(ARGS.cases)
