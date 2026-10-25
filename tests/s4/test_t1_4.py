"""T1.4 (kaiyue_baseline.md): attention against the levanter fixtures of tools/levref (cache/levref/<size>/t1_units/attn,
batch-0 rows 0-1 with their EOS-derived segment ids) and the f64 oracle (cache/levref/<size>/t1_oracle/attn).

Oracle rule per output tensor (O, L, dq, dk, dv): rel(nano, f64 oracle) <= 1.25 x rel(levanter, f64 oracle), with
levanter's rel from the oracle's rel.json. Backends: every built one (FA3 persistent and doc_causal, deterministic on
and off, on sm_90, through the production entry points nm_attention_*; the reference attention in bf16 storage
everywhere). Two roundings:
  levanter  q_scaled = bf16(q * s) (the fixture's q_scaled) into the kernel with softmax_scale = 1, then
            dq = bf16(bf16(dq_scaled) * s) (numerics.md §3.4); s = bf16(1/sqrt(hd)) (0x3E00 hd 64, 0x3DB5 hd 128)
  accurate  q into the kernel with softmax_scale = s (DESIGN.md §3.4)
Cases skip when a fixture is absent (cache/ is not in the repository).

    python tests/s4/test_t1_4.py --lib build/libmarincpp_kernels.so --src . [case ...]
"""
import json
import os

import numpy as np
import torch

import nmk4
from nmk4 import Attn, Segs, rel

lib, R = nmk4.init("s4_t1_4")
ARGS = nmk4.nmk.ARGS
RULE = 1.25


def load(d, name):
    with open(os.path.join(d, "index.json")) as f:
        idx = json.load(f)
    t = idx["tensors"][name]
    p = os.path.join(d, t["file"])
    if t["dtype"] == "bfloat16":
        raw = np.load(p, allow_pickle=False)
        u16 = raw.view(np.uint16) if raw.dtype.itemsize == 2 else raw.astype(np.uint16)
        return torch.from_numpy(u16.astype(np.int32)).to(torch.int16).view(torch.bfloat16).reshape(t["shape"])
    return torch.from_numpy(np.load(p)).reshape(t["shape"])


def to_ours_q(x):    # [B, KV, G, S, hd] -> [B*S, (KV*G)*hd]
    B, KV, G, S, hd = x.shape
    return x.permute(0, 3, 1, 2, 4).reshape(B * S, KV * G * hd).contiguous()


def to_ours_k(x):    # [B, KV, S, hd] -> [B*S, KV*hd]
    B, KV, S, hd = x.shape
    return x.permute(0, 2, 1, 3).reshape(B * S, KV * hd).contiguous()


def to_ours_L(x):    # [B, KV, G, S] -> [H, B*S]
    B, KV, G, S = x.shape
    return x.permute(1, 2, 0, 3).reshape(KV * G, B * S).contiguous()


def run_size(size):
    units = nmk4.nmk.src_path("cache", "levref", size, "t1_units", "attn")
    oracle = nmk4.nmk.src_path("cache", "levref", size, "t1_oracle", "attn")
    if not (os.path.exists(os.path.join(units, "index.json")) and os.path.exists(os.path.join(oracle, "rel.json"))):
        raise nmk4.Skip(f"fixture cache/levref/{size}/t1_units/attn or t1_oracle/attn absent")
    with open(os.path.join(units, "results.json")) as f:
        res = json.load(f)
    with open(os.path.join(oracle, "rel.json")) as f:
        lev_rel = json.load(f)
    s = float(res["scale_bf16_value"])
    q = load(units, "q").cuda()
    B, KV, G, S, hd = q.shape
    H = KV * G
    seg = load(units, "segment_ids_q").numpy()
    segs = Segs(seg, max_segs=24)
    q_ours = to_ours_q(q)
    qs_ours = to_ours_q(load(units, "q_scaled").cuda())
    k_ours = to_ours_k(load(units, "k").cuda())
    v_ours = to_ours_k(load(units, "v").cuda())
    do_ours = to_ours_q(load(units, "dO").cuda())
    orc = {
        "O": to_ours_q(load(oracle, "O").cuda()), "dq": to_ours_q(load(oracle, "dq").cuda()),
        "dk": to_ours_k(load(oracle, "dk").cuda()), "dv": to_ours_k(load(oracle, "dv").cuda()),
        "L": to_ours_L(load(oracle, "L").cuda()),
    }
    print(f"  {size}: B {B} S {S} H {H} KV {KV} hd {hd}, scale {s}, segments per row {[len(x) for x in segs.starts]}")
    backends = [("ref_bf16", None, None)]
    if lib.nm_device_arch() == 90:
        backends += [("fa3.persistent", "persistent", d) for d in (0, 1)]   # production entry points (capi.h)
        backends += [("fa3.doc_causal", "doc_causal", d) for d in (0, 1)]
    for rounding in ("levanter", "accurate"):
        qin, scale = (qs_ours, 1.0) if rounding == "levanter" else (q_ours, s)
        a = Attn(lib, segs, H, KV, hd, None, inputs=(qin, k_ours, v_ours, do_ours), scale=scale)
        for name, mode, det in backends:
            o = a.new_out()
            if mode is None:
                a.ref(o)
            else:
                ws = a.api_ws("fa3", mode)
                a.api_fwd("fa3", mode, o, ws)
                a.api_bwd("fa3", mode, o, ws, det)
            torch.cuda.synchronize()
            if rounding == "levanter":   # dq = bf16(bf16(dq_scaled) * s): the product of two bf16 values is exact in f32
                o["dq"] = (o["dq"].float() * s).to(torch.bfloat16)
            tag = f"{size}.{rounding}.{name}" + ("" if det is None else f".det{det}")
            for t in ("O", "L", "dq", "dk", "dv"):
                rn = rel(o[t], orc[t])
                rl = lev_rel[t]["rel_lev_oracle"]
                ok = rn <= RULE * rl
                print(f"    {tag}.{t}: rel(nano, f64) {rn:.3e}  rel(levanter, f64) {rl:.3e}  ratio {rn / rl:.3f}"
                      f"  {'ok' if ok else 'FAIL'}")
                R.check(ok, f"{tag}.{t}: {rn:.3e} > {RULE} x {rl:.3e}")


@R.case
def t1_4_130m():
    run_size("130m")


@R.case
def t1_4_520m():
    run_size("520m")


@R.case
def t1_4_1_2b():
    run_size("1_2b")


R.run(ARGS.cases)
