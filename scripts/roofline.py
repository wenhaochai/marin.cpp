"""Roofline ceiling for the owner's Della workloads on H100 SXM (marincpp survey, 2026-10-24).

Pure arithmetic, no JAX. Run on a vis node:
    ssh della-vis1 'nice -n 19 $ROOT/project/marin/.venv/bin/python \
        $ROOT/project/marin.cpp/scripts/roofline.py'
Prints Markdown tables (pasted into docs/survey/roofline.md) and writes roofline.json next to this script.

Sources of the configs (read 2026-10-24, read-only):
  project/marin/experiments/references/della_muonh_qwen3_scaling.py:426-435 (SIZES, PER_DEVICE_PARALLELISM)
  project/marin-pls/experiments/references/della_muonh_qwen3_scaling.py:155,166-187 (LAYERS, PARAM_MAP, PDP)
  levanter models/llama.py:60-81 (head_dim = hidden/heads, gradient_checkpointing=True, scan_layers=True)
  levanter utils/flop_utils.py:5-61 (MFU flop formula), main/train_lm.py:388-389 (flops_per_example = 3 * fpt * S)
  fray device_flops.py:20-28 (H100 bf16 = 1.979e15/2)
Every efficiency constant marked ESTIMATE below is an assumption, not a measurement.
"""

import json
import os

# ---------------------------------------------------------------- hardware (H100 SXM 80GB)
PEAK_BF16 = 989.5e12      # dense bf16, fray device_flops.py:25 (levanter MFU denominator)
PEAK_FP8 = 1979e12
PEAK_TF32 = 494.7e12      # dense tf32 (f32 matmuls in JAX on GPU default to TF32)
HBM_BW = 3.35e12          # B/s
NVLINK_BUSBW = 350e9      # ESTIMATE: achievable NCCL bus bandwidth per GPU on one 8-GPU NVSwitch node (spec 450 GB/s/dir)
HBM_GIB = 80

# ---------------------------------------------------------------- efficiency assumptions (ESTIMATE)
ETA_BW = 0.80             # fraction of HBM bandwidth a well-fused elementwise kernel reaches


def eta_gemm(k, n):
    """Tensor-core efficiency of a bf16 GEMM with contraction k and output width n, M (tokens) large.
    ESTIMATE: cuBLAS/CUTLASS on H100 reach ~0.75-0.8 of peak for k,n >= 2048; small k (512) leaves few k-iterations
    per output tile, so prologue/epilogue and the output write dominate."""
    m = min(k, n)
    if m >= 2048:
        return 0.78
    if m >= 1024:
        return 0.72
    if m >= 768:
        return 0.66
    return 0.60


def eta_attn(hd):
    """(fwd, bwd) efficiency of an FA3-class causal kernel at seq 4096 (ESTIMATE; FA3 paper peaks ~0.75 fwd at hd 128
    non-causal; hd 64 and causal masking cost more)."""
    return (0.60, 0.45) if hd >= 128 else (0.45, 0.35)


# ---------------------------------------------------------------- workloads
V = 128256
S = 4096
SIZES = {   # della_muonh_qwen3_scaling.py:427-430; pdp from :435 (-1 = whole per-device batch)
    "130m": dict(h=512, I=1792, L=6, nh=8, kv=8, batch=128, steps=4959, pdp=None),
    "300m": dict(h=768, I=2688, L=12, nh=12, kv=12, batch=128, steps=11444, pdp=None),
    "520m": dict(h=1024, I=3584, L=24, nh=16, kv=8, batch=256, steps=9918, pdp=32),
    "1_2b": dict(h=2048, I=7168, L=16, nh=16, kv=8, batch=256, steps=22888, pdp=32),
}
# measured, levanter, 4xH100 (logs/kaiyue_muonh_qwen3_h100x4.yaml:50,69,85,97 in project/marin)
MEASURED = {"130m": dict(tok_s=831e3, mfu=14.36), "300m": dict(tok_s=362.8e3, mfu=15.29),
            "520m": dict(tok_s=155.7e3, mfu=15.92), "1_2b": dict(tok_s=128.272e3, mfu=27.97)}

# ss schedule, della_muonh_qwen3_scaling.py:189 default "0.57:24576,0.81:36864,0.93:65536", full softmax after 0.93
SS_SCHEDULE = [(0.57, 24576), (0.81, 36864), (0.93, 65536), (1.0, V)]
OV_M = 12_800_000


def cfg(name, **over):
    c = dict(SIZES[name])
    c.update(over)
    c["hd"] = c["h"] // c["nh"]
    return c


# ---------------------------------------------------------------- FLOPs per token
def gemms(c):
    """(name, k, n) of every per-layer GEMM in the forward pass (per token: 2*k*n FLOPs)."""
    h, I, nh, kv, hd = c["h"], c["I"], c["nh"], c["kv"], c["hd"]
    return [("qkv", h, (nh + 2 * kv) * hd), ("o", nh * hd, h), ("gate_up", h, 2 * I), ("down", I, h)]


def fwd_flops(c, S=S):
    layer_gemm = sum(2 * k * n for _, k, n in gemms(c))
    attn_full = 4 * c["nh"] * c["hd"] * S          # QK^T + PV over all S keys (levanter/Megatron count)
    attn_soft = 3 * S * c["nh"]                     # levanter adds the softmax term
    attn_causal = attn_full / 2                     # average S/2 keys per query
    head = 2 * c["h"] * V
    return dict(layer_gemm=layer_gemm, attn_full=attn_full, attn_soft=attn_soft, attn_causal=attn_causal, head=head)


def levanter_fpt(c, S=S):
    f = fwd_flops(c, S)
    return c["L"] * (f["layer_gemm"] + f["attn_full"] + f["attn_soft"]) + f["head"]


def params(c):
    per_layer = sum(k * n for _, k, n in gemms(c)) + 6 * c["h"] + 2 * c["hd"]  # 4 hybrid norms + 2 qk-norms
    return dict(layers=c["L"] * per_layer, embed=V * c["h"], head=V * c["h"], linear=c["L"] * sum(k * n for _, k, n in gemms(c)))


# ---------------------------------------------------------------- bytes per token of the bandwidth-bound parts
def elementwise_bytes_fwd(c):
    """Per layer per token, bf16, assuming norm/residual/act fusions but no fusion into GEMM epilogues."""
    h, I, nh, kv, hd = c["h"], c["I"], c["nh"], c["kv"], c["hd"]
    norms = 4 * (2 * h * 2)                 # 4 RMSNorms (pre/post attn, pre/post mlp; hybrid_norm): read h, write h
    resid = 2 * (3 * h * 2)                 # 2 residual adds: read 2h, write h
    qk = 2 * ((nh + kv) * hd * 2)           # qk-norm + rotary fused: read q,k, write q,k
    swiglu = (3 * I) * 2                    # read gate, up; write act
    return dict(norms=norms, resid=resid, qk_rope=qk, swiglu=swiglu)


def ce_bytes(h, Vh, mode):
    """HBM bytes per token of the LM-head softmax-CE, excluding GEMM operand reads.
    current: levanter H100 b-tiled path (batched_xla.py:317-366 fwd, :552-617 bwd): fwd writes f32 logits (4V) and
      reads them twice (max, sum-exp: 8V); bwd GEMM writes bf16 logits (2V), multiply fusion reads them and writes f32
      dlogits (2V+4V), scatter in place, convert reads f32 writes bf16 (4V+2V), two GEMMs read bf16 dlogits (4V).
      (tmp/pls_speed/plan.md:55-61 lists the same five bwd kernels.)
    a1: same with the one-hot select fusing multiply/scatter/convert (plan.md A1): bwd 2V+2V+2V+4V.
    bf16: logits and dlogits kept in bf16 everywhere: fwd 2V+4V, bwd 2V+2V+2V+4V... (half of current's f32 terms).
    fused: chunked kernel keeping logits in SMEM/registers (Liger/CCE/FA-style): ~0 beyond the GEMM operands."""
    if mode == "current":
        return 12 * Vh + 18 * Vh
    if mode == "a1":
        return 12 * Vh + 10 * Vh
    if mode == "bf16":
        return 6 * Vh + 10 * Vh
    if mode == "fused":
        return 0
    raise ValueError(mode)


# ---------------------------------------------------------------- time model per token on one GPU
FP8 = {"on": False}       # layer GEMMs in FP8 at ETA_FP8 of 1979 TFLOP/s (ESTIMATE 0.6); head/attention stay bf16
ETA_FP8 = 0.60


def gemm_time(k, n, flops_mult=1.0):
    """Time per token of one GEMM pass (2kn FLOPs) at its shape efficiency, bounded by activation bytes."""
    t_c = 2 * k * n * flops_mult / (eta_gemm(k, n) * PEAK_BF16)
    if FP8["on"]:
        t_c = min(t_c, 2 * k * n * flops_mult / (ETA_FP8 * PEAK_FP8))
    t_m = 2 * (k + n) / (ETA_BW * HBM_BW)
    return max(t_c, t_m)


def layer_time(c, remat, S=S):
    """Seconds per token for all L layers: GEMMs, attention, elementwise."""
    nf = 2 if remat else 1                  # forward passes over the layers (remat recomputes once in bwd)
    t_g = 0.0
    for _, k, n in gemms(c):
        t_g += nf * gemm_time(k, n) + gemm_time(n, k) + gemm_time(k, n)   # fwd(s) + dX + dW (dW ~ same shape class)
    f = fwd_flops(c, S)
    ef, eb = eta_attn(c["hd"])
    t_a = nf * f["attn_causal"] / (ef * PEAK_BF16) + 2.5 * f["attn_causal"] / (eb * PEAK_BF16)
    eb_fwd = sum(elementwise_bytes_fwd(c).values())
    t_e = (nf + 2) * eb_fwd / (ETA_BW * HBM_BW)   # bwd elementwise ~2x fwd bytes
    return dict(gemm=c["L"] * t_g, attn=c["L"] * t_a, elem=c["L"] * t_e)


def head_time(h, Vh, mode, n_heads=1):
    """Seconds per token for n_heads LM heads of width Vh. FLOPs: fused/a1/current recompute logits in bwd -> 8hV."""
    # tensor-core time only; every logits/dlogits byte is in ce_bytes (fused: none leave the SM)
    t_logit = 2 * (2 * h * Vh) / (eta_gemm(h, Vh) * PEAK_BF16)   # fwd logits + bwd recompute: contraction h
    t_dx = 2 * h * Vh / (eta_gemm(Vh, h) * PEAK_BF16)            # dlogits @ W^T: output width h
    t_dw = 2 * h * Vh / (eta_gemm(2048, 2048) * PEAK_BF16)       # x^T @ dlogits: contraction over tokens
    t_mem = ce_bytes(h, Vh, mode) / (ETA_BW * HBM_BW)
    return n_heads * (t_logit + t_dx + t_dw + t_mem), n_heads * t_mem


# ---------------------------------------------------------------- per-step: optimizer, Newton-Schulz, comm, memory
def ns_flops(c):
    """Muon Newton-Schulz FLOPs per step (5 iterations, levanter optim/util.py:244-297): per matrix m<=n,
    A = X X^T (2 m^2 n), A @ A (2 m^3), B @ X (2 m^2 n) per iteration. Run in f32 (TF32 tensor cores)."""
    tot, rep = 0, 0
    for _, k, n in gemms(c):
        mats = [(k, n // 2), (k, n // 2)] if _ == "gate_up" else [(k, n)]
        for a, b in mats:
            m, nn = min(a, b), max(a, b)
            tot += 5 * (4 * m * m * nn + 2 * m ** 3)
            rep += 5 * 2 * m ** 3
    return c["L"] * tot, c["L"] * rep


def step_terms(c, ngpu, mode="fused", n_heads=1, Vh=V, remat=True, S=S, extra_head_tokens=0.0):
    tok_gpu = c["batch"] * S / ngpu
    lt = layer_time(c, remat, S)
    ht, hmem = head_time(c["h"], Vh, mode, n_heads)
    emb = (2 * c["h"] + 2 * 4 * c["h"]) / (ETA_BW * HBM_BW)   # gather fwd + f32 scatter-add bwd
    per_tok = lt["gemm"] + lt["attn"] + lt["elem"] + ht + emb
    t_compute = per_tok * tok_gpu
    p = params(c)
    P = p["layers"] + p["embed"] + p["head"] * n_heads
    # optimizer elementwise: f32 p,g,mu (+m,v for adam/adamh tables); read 5 write 3 words avg (ESTIMATE), sharded /ngpu
    t_opt = P * 4 * 8 / ngpu / (ETA_BW * HBM_BW)
    ns_tot, ns_rep = ns_flops(c)
    t_ns = (ns_tot - ns_rep) / ngpu / (0.5 * PEAK_TF32) + ns_rep / (0.5 * PEAK_TF32)  # ESTIMATE eta 0.5 on TF32
    # communication per GPU per step
    nmb = 1 if not c.get("pdp") else int(c["batch"] / ngpu / c["pdp"])
    f = (ngpu - 1) / ngpu
    ddp_bf16 = 2 * f * P * 2                          # ring all-reduce of bf16 grads once per step
    ddp_f32 = 2 * f * P * 4
    fsdp = nmb * ((2 if remat else 1) * f * P * 2 + f * P * 4)   # bf16 all-gathers (fwd + remat) + f32 reduce-scatter per microbatch
    fsdp_f32gather = nmb * ((2 if remat else 1) * f * P * 4 + f * P * 4)
    return dict(tok_gpu=tok_gpu, per_tok=dict(**lt, head=ht, head_mem=hmem, emb=emb), t_compute=t_compute,
                t_opt=t_opt, t_ns=t_ns, ns_flops=ns_tot, nmb=nmb, P=P,
                comm=dict(ddp_bf16=ddp_bf16, ddp_f32=ddp_f32, fsdp_bf16gather=fsdp, fsdp_f32gather=fsdp_f32gather))


def realistic_step(c, ngpu, exposed_comm=0.1, **kw):
    st = step_terms(c, ngpu, **kw)
    t_comm = st["comm"]["ddp_bf16"] / NVLINK_BUSBW
    t = st["t_compute"] + st["t_opt"] + st["t_ns"] + exposed_comm * t_comm
    return t, st


def activation_mem(c, tokens_mb, remat):
    """Bytes per GPU of saved activations for one microbatch of tokens_mb tokens."""
    h, I, nh, kv, hd, L = c["h"], c["I"], c["nh"], c["kv"], c["hd"], c["L"]
    full = 2 * (h + h + (nh + 2 * kv) * hd + (nh + kv) * hd + nh * hd + h + h + h + 2 * I + I + h) + 4 * nh
    select = 2 * (h + (nh + 2 * kv) * hd + nh * hd + h + 2 * I + h) + 4 * nh   # GEMM inputs + flash out/lse; norms recomputed
    if remat:
        per = 2 * h * L + full            # residual per layer + one layer's internals in flight
        return per * tokens_mb
    return dict(full=full * L * tokens_mb, select=select * L * tokens_mb)


def state_mem(c, ngpu, fsdp=True, n_heads=1, extra=0):
    p = params(c)
    lin, emb, head = p["linear"], p["embed"], p["head"] * n_heads
    other = p["layers"] - lin
    b = lin * (4 + 4 + 4) + (emb + head) * (4 + 4 + 8) + other * (4 + 4 + 8) + extra   # f32 p, g, muon mu | adam m,v
    return b / ngpu if fsdp else b


# ---------------------------------------------------------------- report
def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def main():
    out = {}
    rows = []
    print("## 1. FLOPs per token (forward; training = 3x) and parameter counts\n")
    print("| size | h | L | hd | layer GEMM MFLOP | attn causal MFLOP | attn levanter-count MFLOP | LM head MFLOP | head share (lev.) | levanter fpt (GFLOP) | params non-embed (M) | params total (M) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in SIZES:
        c = cfg(name)
        f = fwd_flops(c)
        lev = levanter_fpt(c)
        p = params(c)
        print(f"| {name} | {c['h']} | {c['L']} | {c['hd']} | {fmt(c['L']*f['layer_gemm']/1e6)} | {fmt(c['L']*f['attn_causal']/1e6)} | {fmt(c['L']*(f['attn_full']+f['attn_soft'])/1e6)} | {fmt(f['head']/1e6)} | {f['head']/lev*100:.0f}% | {lev/1e9:.3f} | {p['layers']/1e6:.0f} | {(p['layers']+p['embed']+p['head'])/1e6:.0f} |")
        out.setdefault("sizes", {})[name] = dict(cfg=c, fwd=f, levanter_fpt=lev, params=p)
    for nm, L, nheads in (("pls d48 1 head", 48, 1), ("pls d48 48 heads", 48, 48)):
        c = cfg("130m", L=L)
        f = fwd_flops(c)
        lev = levanter_fpt(c) + (nheads - 1) * f["head"]
        p = params(c)
        print(f"| {nm} | 512 | 48 | 64 | {fmt(48*f['layer_gemm']/1e6)} | {fmt(48*f['attn_causal']/1e6)} | {fmt(48*(f['attn_full']+f['attn_soft'])/1e6)} | {fmt(nheads*f['head']/1e6)} | {nheads*f['head']/lev*100:.0f}% | {lev/1e9:.3f} | {p['layers']/1e6:.0f} | {(p['layers']+p['embed']+p['head']*nheads)/1e6:.0f} |")

    print("\n## 2. Check: levanter MFU formula reproduces the measured MFU\n")
    print("| size | measured tok/s (4 GPU) | levanter fpt x3 (GFLOP/token) | implied MFU | logged MFU | s/step | true-causal model TFLOP/s/GPU | hardware TFLOP/s/GPU (remat + CE recompute) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in MEASURED.items():
        c = cfg(name)
        f = fwd_flops(c)
        lev3 = 3 * levanter_fpt(c)
        mfu = m["tok_s"] * lev3 / (4 * PEAK_BF16) * 100
        true3 = 3 * (c["L"] * (f["layer_gemm"] + f["attn_causal"]) + f["head"])
        hw = c["L"] * (4 * f["layer_gemm"] + 4.5 * f["attn_causal"] * 1.25) + 4 * f["head"]   # flash 1024-blocks compute 10/16 of the square = 1.25x causal
        print(f"| {name} | {fmt(m['tok_s']/1e3)}k | {lev3/1e9:.3f} | {mfu:.2f}% | {m['mfu']}% | {c['batch']*S/m['tok_s']:.2f} | {m['tok_s']/4*true3/1e12:.0f} | {m['tok_s']/4*hw/1e12:.0f} |")

    print("\n## 3. Per-token time budget on one H100 (realistic ceiling model, fused CE, remat on), microseconds per 1k tokens\n")
    print("| size | layer GEMMs | attention | elementwise | LM head (fused) | head if current CE | total realistic (fused CE) | total with current CE |")
    print("|---|---|---|---|---|---|---|---|")
    for name in SIZES:
        c = cfg(name)
        st = step_terms(c, 4)
        pt = st["per_tok"]
        cur, _ = head_time(c["h"], V, "current")
        tot = pt["gemm"] + pt["attn"] + pt["elem"] + pt["head"] + pt["emb"]
        print(f"| {name} | {pt['gemm']*1e9:.1f} | {pt['attn']*1e9:.1f} | {pt['elem']*1e9:.1f} | {pt['head']*1e9:.1f} | {cur*1e9:.1f} | {tot*1e9:.1f} | {(tot - pt['head'] + cur)*1e9:.1f} |")

    print("\n## 4. Ceiling per config on 4xH100 (per-step compute; evals/checkpoints/compile excluded)\n")
    print("| config | GPUs | measured s/step | speed-of-light s/step (100% peak, no remat, causal) | realistic s/step remat on | realistic remat off | realistic remat off + FP8 GEMMs | realistic MFU levanter conv. (on / off) | realistic MFU true-causal (on / off) | ceiling speedup (on / off / off+FP8) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    configs = []
    for name in SIZES:
        configs.append((name, 4, cfg(name), MEASURED[name]["tok_s"], ""))
    # 8-GPU levanter runs without evals (tmp/pls_speed/steptimes.txt: mtpl-520m-ntp_15591473 4.2 s, mtpl-1_2b-ntp_15591483 5.0 s;
    # PLS_W=0 adds stride-16 stop-gradient readouts, so they are slightly slower than a pure baseline)
    configs.append(("520m", 8, cfg("520m", pdp=16), 256 * S / 4.2, " (pls0-noev)"))
    configs.append(("1_2b", 8, cfg("1_2b", pdp=16), 256 * S / 5.0, " (pls0-noev)"))
    for name, ng, c, meas_toks, tag in configs:
        meas = c["batch"] * S / meas_toks
        t_on, st = realistic_step(c, ng, remat=True)
        t_off, _ = realistic_step(c, ng, remat=False)
        FP8["on"] = True
        t_f8, _ = realistic_step(c, ng, remat=False)
        FP8["on"] = False
        f = fwd_flops(c)
        true = 3 * (c["L"] * (f["layer_gemm"] + f["attn_causal"]) + f["head"]) * st["tok_gpu"]
        sol = true / PEAK_BF16
        lev = 3 * levanter_fpt(c) * st["tok_gpu"]
        print(f"| {name}{tag} | {ng} | {meas:.2f} | {sol:.2f} | {t_on:.2f} | {t_off:.2f} | {t_f8:.2f} | {lev/t_on/PEAK_BF16*100:.0f}% / {lev/t_off/PEAK_BF16*100:.0f}% | {true/t_on/PEAK_BF16*100:.0f}% / {true/t_off/PEAK_BF16*100:.0f}% | {meas/t_on:.1f}x / {meas/t_off:.1f}x / {meas/t_f8:.1f}x |")
        rows.append(dict(config=name + tag, gpus=ng, measured=meas, sol=sol, real_remat=t_on, real_noremat=t_off, real_noremat_fp8=t_f8))

    # pls d48 (4 GPUs, PDP=4 -> 8 microbatches of 4 sequences; pls_depth_controls.yaml:151,184; pls.md:640)
    print("\n## 5. pls d48 (width 512, 48 layers, 4xH100, batch 128)\n")
    print("| config | measured s/step | head part measured | realistic s/step, current CE | realistic, CE a1 | realistic, CE bf16 | realistic, fused CE | fused CE, remat off | speedup (fused) |")
    print("|---|---|---|---|---|---|---|---|---|")
    c = cfg("130m", L=48, pdp=4)
    for nh_, meas, head_meas in ((1, 15.2 - 12.3, None), (48, 15.2, 12.3)):
        res = []
        for mode in ("current", "a1", "bf16", "fused"):
            t, st = realistic_step(c, 4, mode=mode, n_heads=nh_)
            res.append(t)
        t_nr, _ = realistic_step(c, 4, mode="fused", n_heads=nh_, remat=False)
        hm = f"{head_meas}" if head_meas else "-"
        print(f"| d48, {nh_} head(s) | {meas:.1f}{' (15.2-12.3)' if nh_ == 1 else ''} | {hm} | {res[0]:.2f} | {res[1]:.2f} | {res[2]:.2f} | {res[3]:.2f} | {t_nr:.2f} | {meas/res[3]:.1f}x |")
        rows.append(dict(config=f"pls d48 {nh_} heads", measured=meas, real=res, real_noremat=t_nr))
    # per-head time check
    tok_gpu = 128 * S / 4
    for mode in ("current", "a1", "bf16", "fused"):
        ht, hm = head_time(512, V, mode)
        print(f"\n- one 128K head at h=512, mode {mode}: {ht*tok_gpu:.3f} s/step per GPU ({hm*tok_gpu:.3f} s of it HBM traffic); x47 = {47*ht*tok_gpu:.2f} s", end="")
    print()
    print(f"\n- 47 extra heads, measured 12.3 s/step (pls.md:640): {47*8*512*V*tok_gpu/12.3/1e12:.0f} TFLOP/s/GPU counting 4 GEMMs (8hV), "
          f"{47*6*512*V*tok_gpu/12.3/1e12:.0f} counting the model's 6hV")

    # sampled softmax and OV
    print("\n## 6. Sampled softmax and over-vocab (4xH100 unless noted)\n")
    vbar = 0
    prev = 0
    for end, p in SS_SCHEDULE:
        vbar += (end - prev) * p
        prev = end
    print(f"- ss schedule average candidate width = {vbar:,.0f} classes ({vbar/V*100:.0f}% of V); time-weighted head cost scales with it (plus a row gather of P x h)")
    print("\n| size | realistic s/step full softmax (fused) | realistic s/step ss (fused, avg P) | ss gain ceiling | measured ss gain (sampled-softmax.md:3-4) |")
    print("|---|---|---|---|---|")
    meas_ss = {"130m": 1.30, "300m": 1.13, "520m": 1.07, "1_2b": 1.08}
    for name in SIZES:
        c = cfg(name)
        t_full, _ = realistic_step(c, 4)
        t_ss, _ = realistic_step(c, 4, Vh=int(vbar))
        print(f"| {name} | {t_full:.2f} | {t_ss:.2f} | {t_full/t_ss:.2f}x | {meas_ss[name]:.2f}x |")
    print("\n| size | OE tables x dim | OE table state GiB total (f32 p,g,m,v) | per GPU (4 / 8) | Adam traffic per step per GPU (8 GPU) ms | OD 2nd head + dxd extra s/step (fused, 8 GPU) |")
    print("|---|---|---|---|---|---|")
    for name in SIZES:
        c = cfg(name)
        k = max(1, round(c["h"] / (3 * 256)))
        nt = 2 * k
        dim = c["h"] // (3 * k)
        elems = nt * OV_M * dim
        st_b = elems * 16
        traffic = elems * 28 / 8 / (ETA_BW * HBM_BW)
        ht, _ = head_time(c["h"], V, "fused")
        extra = (ht + 3 * gemm_time(c["h"], c["h"])) * c["batch"] * S / 8
        print(f"| {name} | {nt} x {dim} | {st_b/2**30:.0f} | {st_b/4/2**30:.0f} / {st_b/8/2**30:.0f} | {traffic*1e3:.0f} | {extra:.2f} |")

    # memory
    print("\n## 7. Memory per GPU (GiB): activations per microbatch and persistent state\n")
    print("| config | tokens/microbatch/GPU | act. remat on | act. no remat (save all) | act. no remat (selective) | state FSDP/4 | state replicated | CE logits temp current (8192-row f32 block, x3 live) |")
    print("|---|---|---|---|---|---|---|---|")
    mem_cfgs = [(n, cfg(n), 1) for n in SIZES] + [("pls d48 48 heads", cfg("130m", L=48, pdp=4), 48)]
    for name, c, nh_ in mem_cfgs:
        per_gpu_seq = c["batch"] // 4
        mb = (c["pdp"] or per_gpu_seq) * S
        a_on = activation_mem(c, mb, True)
        a_off = activation_mem(c, mb, False)
        print(f"| {name} | {mb:,} | {a_on/2**30:.1f} | {a_off['full']/2**30:.1f} | {a_off['select']/2**30:.1f} | {state_mem(c, 4, True, nh_)/2**30:.2f} | {state_mem(c, 4, False, nh_)/2**30:.2f} | {3*8192*V*4/2**30:.1f} |")

    # comm and optimizer
    print("\n## 8. Communication and optimizer per step\n")
    print("| config | params (M) | microbatches | DDP bf16 all-reduce GB/GPU | t @350 GB/s (s) | FSDP (bf16 gathers x2 + f32 RS) per mb GB | FSDP total GB (4 GPU / 8 GPU) | t FSDP 4 GPU (s) | NS TFLOP/step | NS s/step (4 GPU, FSDP-sharded) | compute s/step (realistic) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, c, nh_ in mem_cfgs:
        st4 = step_terms(c, 4, n_heads=nh_)
        st8 = step_terms(c, 8, n_heads=nh_)
        cm = st4["comm"]
        print(f"| {name} | {st4['P']/1e6:.0f} | {st4['nmb']} | {cm['ddp_bf16']/1e9:.2f} | {cm['ddp_bf16']/NVLINK_BUSBW:.3f} | {cm['fsdp_bf16gather']/st4['nmb']/1e9:.2f} | {cm['fsdp_bf16gather']/1e9:.1f} / {st8['comm']['fsdp_bf16gather']/1e9:.1f} | {cm['fsdp_bf16gather']/NVLINK_BUSBW:.2f} | {st4['ns_flops']/1e12:.2f} | {st4['t_ns']:.3f} | {st4['t_compute']:.2f} |")

    # arithmetic intensity
    print("\n## 9. Arithmetic intensity of the K=h GEMMs (ridge = 989.5/3.35 = 295 FLOP/B)\n")
    print("| GEMM | k | n | AI bf16 out (FLOP/B) | AI f32 out | bound |")
    print("|---|---|---|---|---|---|")
    for h in (512, 768, 1024, 2048):
        for nm, k, n in (("logits", h, V), ("qkv/o", h, h), ("gate_up", h, int(3.5 * h) * 2)):
            ai16 = 2 * k * n / (2 * k + 2 * n)
            ai32 = 2 * k * n / (2 * k + 4 * n)
            print(f"| {nm} h={h} | {k} | {n} | {ai16:.0f} | {ai32:.0f} | {'BW' if ai32 < 295 else 'compute'} (f32 out) |")

    # off-1024 sequence lengths (tokenizer session)
    print("\n## 10. SEQ_LEN 4224/4480 (300m, tokenizer session)\n")
    c = cfg("300m")
    for s in (4096, 4224, 4480):
        f = fwd_flops(c, s)
        print(f"- S={s}: levanter fpt {levanter_fpt(c, s)/1e9:.3f} GFLOP/token; attention share {c['L']*(f['attn_full']+f['attn_soft'])/levanter_fpt(c, s)*100:.0f}% of the count")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "roofline.json"), "w") as fh:
        json.dump(dict(out=out, rows=rows), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
