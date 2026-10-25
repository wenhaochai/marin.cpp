"""E1b (DESIGN.md §3.4, O1; BUILD_LOG decisions 6-8): the vendored FA3 in every launch mode, deterministic off and on,
on the real segment rows (bench/e1_attention/segments), at the owner shapes. Per (shape, mode, det): forward and
forward + backward median ms (5 warm-up, 20 timed, CUDA events), document TFLOP/s (E1's convention), and the error of
O, dq, dk, dv against the reference attention in f32 storage (the full-size oracle). Writes JSON lines and a markdown
summary with the O1 rule applied.

    python tests/s4/bench_attention.py --lib <libmarincpp_kernels.so> --src <repo> --out <dir> [--shapes 130m,...]

Modes: stock_exact = FA3 with the real segment count and longest segment (E1's setup, the bar), stock_padded = FA3's
own backward over rows * 24 segment slots (version 1), sm_grid = the work list on a grid of SM count (E1b arm),
persistent = the work list with one CTA per slot (training `persistent`), doc_causal (training `doc_causal`; its
forward is the varlen forward at hd 64).
"""
import argparse
import os

import numpy as np
import torch

import nmk4  # noqa: E402
from nmk4 import Attn, Segs, rel, time_ms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="130m,300m,520m,1_2b,300m_S4224,300m_S4480,300m_S19456")
ap.add_argument("--modes", default="stock_exact,stock_padded,sm_grid,persistent,doc_causal")
ap.add_argument("--no-check", action="store_true")
args = ap.parse_args()

lib = nmk4.nmk.Lib(args.lib)
os.makedirs(args.out, exist_ok=True)
JS = os.path.join(args.out, "e1b.jsonl")

# E1 FA3 varlen (torch, deterministic off): the S4 bar per shape (fwd+bwd, fwd), docs/bench/RESULTS.md §1
BAR = {"130m": (2.51, 0.66), "300m": (3.75, 0.95), "520m": (5.68, 1.27), "1_2b": (9.21, 1.73),
       "300m_S4224": (3.72, 0.93), "300m_S4480": (4.71, 1.23), "300m_S19456": (6.16, 1.69)}
SHAPE = {"130m": ("130m", 4096, 32), "300m": ("300m", 4096, 32), "520m": ("520m", 4096, 32), "1_2b": ("1_2b", 4096, 32),
         "300m_S4224": ("300m", 4224, 32), "300m_S4480": ("300m", 4480, 32), "300m_S19456": ("300m", 19456, 4)}

dev = torch.cuda.get_device_name()
print(f"device {dev}, lib {args.lib}", flush=True)
results = []
for sname in args.shapes.split(","):
    size, S, B = SHAPE[sname]
    H, KV, hd = nmk4.ATTN_SHAPES[size]
    seg = nmk4.real_segments(args.src, S=S, rows=B)
    if seg is None:
        print(f"{sname}: segment file absent, skipped")
        continue
    segs = Segs(seg, max_segs=24 if S <= 4480 else max(24, int(np.ceil(S / 4096 * 24))))
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = Attn(lib, segs, H, KV, hd, gen)
    flops = a.fwd_flops()
    orc = None
    if not args.no_check:
        a32 = Attn(lib, segs, H, KV, hd, gen, dtype=torch.float32,
                   inputs=(a.q.float(), a.k.float(), a.v.float(), a.dO.float()))
        orc = a32.new_out()
        a32.ref(orc)
        torch.cuda.synchronize()
        del a32
    print(f"{sname}: rows {B} S {S} H {H} KV {KV} hd {hd}: {segs.n} segments, longest {segs.max_len}, "
          f"max {segs.max_in_data} per row, slots {B * segs.max_segs}; fwd {flops / 1e9:.0f} GFLOP", flush=True)
    for mode in args.modes.split(","):
        for det in (0, 1):
            try:
                ws = a.fa3_ws(mode)
                o = a.new_out()
                t_f = time_ms(lambda: a.fa3_fwd(mode, o, ws))
                t_fb = time_ms(lambda: (a.fa3_fwd(mode, o, ws), a.fa3_bwd(mode, o, ws, det)))
                err = {}
                if orc is not None:
                    a.fa3_fwd(mode, o, ws)
                    a.fa3_bwd(mode, o, ws, det)
                    torch.cuda.synchronize()
                    for t in ("O", "dq", "dk", "dv"):
                        err[t] = rel(o[t], orc[t]) / rel(orc[t].to(torch.bfloat16), orc[t])
                    err["L"] = rel(o["L"], orc["L"])
                rec = dict(shape=sname, mode=mode, det=det, fwd_ms=t_f, fwdbwd_ms=t_fb, bwd_ms=t_fb - t_f,
                           fwd_tflops=flops / t_f / 1e9, fwdbwd_tflops=3.5 * flops / t_fb / 1e9,
                           ws_bytes=int(ws.numel()), err_over_floor=err, gpu=dev, segments=segs.n,
                           bar_fwdbwd=BAR[sname][0], bar_fwd=BAR[sname][1])
                del ws
            except Exception as e:  # noqa: BLE001 (record and continue with the other modes)
                rec = dict(shape=sname, mode=mode, det=det, error=str(e), gpu=dev)
            results.append(rec)
            nmk4.emit(JS, rec)
            if "error" in rec:
                print(f"  {mode:13s} det {det}: ERROR {rec['error']}", flush=True)
            else:
                e = " ".join(f"{k} {v:.2f}" for k, v in rec["err_over_floor"].items() if k != "L")
                print(f"  {mode:13s} det {det}: fwd {t_f:.3f} ms  fwd+bwd {t_fb:.3f} ms ({rec['fwdbwd_tflops']:.0f} "
                      f"TFLOP/s)  bar {BAR[sname][0]:.2f}  err/floor {e}", flush=True)
            torch.cuda.empty_cache()

# summary + O1 rule
by = {(r["shape"], r["mode"], r["det"]): r for r in results if "error" not in r}
lines = ["| shape | mode | det | fwd ms | fwd+bwd ms | TFLOP/s | E1 bar fwd+bwd (fwd) | vs bar | max err / floor |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in results:
    if "error" in r:
        lines.append(f"| {r['shape']} | {r['mode']} | {r['det']} | error: {r['error'][:80]} | | | | | |")
        continue
    me = max((v for k, v in r["err_over_floor"].items() if k != "L"), default=float("nan"))
    lines.append(f"| {r['shape']} | {r['mode']} | {r['det']} | {r['fwd_ms']:.3f} | {r['fwdbwd_ms']:.3f} | "
                 f"{r['fwdbwd_tflops']:.0f} | {r['bar_fwdbwd']:.2f} ({r['bar_fwd']:.2f}) | "
                 f"{r['fwdbwd_ms'] / r['bar_fwdbwd']:.3f} | {me:.2f} |")
o1 = []
for det in (0, 1):
    sums = {}
    for mode in ("persistent", "doc_causal", "sm_grid", "stock_exact", "stock_padded"):
        ks = [(s, mode, det) for s in ("130m", "520m", "1_2b")]
        if all(k in by for k in ks):
            sums[mode] = sum(by[k]["fwdbwd_ms"] for k in ks)
    o1.append((det, sums))
txt = "\n".join(lines) + "\n\nO1 sums of fwd+bwd over 130m, 520m, 1.2B (ms):\n"
for det, sums in o1:
    txt += f"- deterministic {det}: " + ", ".join(f"{m} {v:.3f}" for m, v in sums.items()) + "\n"
    if "persistent" in sums and "doc_causal" in sums:
        p, d = sums["persistent"], sums["doc_causal"]
        win = "doc_causal" if d <= p * 1.03 else "persistent"
        txt += f"  rule (ties within 3% -> doc_causal): {win} (persistent / doc_causal = {p / d:.3f})\n"
with open(os.path.join(args.out, "e1b_summary.md"), "w") as f:
    f.write(txt)
print(txt)
