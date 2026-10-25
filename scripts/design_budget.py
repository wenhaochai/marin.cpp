"""Memory plans and step budgets of docs/DESIGN.md §2.5 and §12 (planner formulas; CPU only, no dependencies).

Usage: python3 scripts/design_budget.py   (prints every table of DESIGN.md §2.5 and §12.2)

Version 2 (after the perf/parity/research review): ZeRO-1 by ownership for every optimizer group, no varlen padding
rows in the attention workspaces, CE chunk = largest that fits, B1 / B1.5 / B2 CE traffic, stage-1 epilogue fusions,
FA3 backward side traffic, embedding reduce-scatter exposure, eval time per run, and the levanter_strict plan
(remat=layer, 2 x 32 sequences per GPU at 520m / 1_2b).

Version 2.1 (S0 review): the strict plans are what `--preset levanter_strict` runs (CE chunk fixed at 8,192 rows plus
the per-microbatch head-gradient accumulator mb.ce_dw_accum); ZeRO-1 owner shards are ceil(n / NGPU) elements, as the
C++ planner sizes them; CE head-gradient read-modify-writes count ceil(micro tokens / chunk) chunks per microbatch.

Version 3 (2026-10-25, H100 benches, docs/bench/RESULTS.md): `--measured` prints the §12.2 v3 table. It replaces the
GEMM and attention efficiencies with the E1 measurements and counts only the CE traffic that no GEMM hides (the
logits store write is inside the measured GEMM time). Without `--measured` the output is version 2.1 unchanged: the
C++ `step_budget` and the CE chunk rule reproduce it.
"""
import sys

GiB = 1024**3
V = 128256
S = 4096
NGPU = 4
PEAK = 989.5e12           # bf16 dense, levanter MFU denominator
TF32 = 494.7e12
FP32_FMA = 67e12
HBM = 3.35e12 * 0.80      # fused elementwise efficiency
NVL = 350e9               # NCCL bus bandwidth (ESTIMATE)
EVAL_TOKENS = 14529 * 4096
CAUSAL_COMPUTED = 0.504   # computed / full-square ratio on the real rows with 192x128 forward tiles (perf review)

SIZES = {
    # evals = periodic evals (s % 1000 == 0, 0 < s < N) + the final eval; the init eval is added in show()
    "130m":    dict(D=512,  L=6,  H=8,  KV=8, hd=64,  F=1792, B=128, N=4959,  heads=1, evals=5),
    "300m":    dict(D=768,  L=12, H=12, KV=12, hd=64, F=2688, B=128, N=11444, heads=1, evals=12),
    "520m":    dict(D=1024, L=24, H=16, KV=8, hd=64,  F=3584, B=256, N=9918,  heads=1, evals=10),
    "1_2b":    dict(D=2048, L=16, H=16, KV=8, hd=128, F=7168, B=256, N=22888, heads=1, evals=23),
    "pls_d48": dict(D=512,  L=48, H=8,  KV=8, hd=64,  F=1792, B=128, N=4959,  heads=48, evals=5),
}


def gemm_eff(D):
    return 0.60 if D <= 512 else 0.66 if D <= 768 else 0.72 if D <= 1024 else 0.78


# E1 GEMM sweep (1 H100, bf16, TFLOP/s, best of cuBLAS and XLA per orientation): (M, K, N) -> (fwd, dgrad, wgrad)
E1_GEMM = {
    (131072, 512, 1536): (606, 707, 728), (131072, 512, 1792): (645, 715, 712),
    (131072, 768, 2304): (692, 754, 671), (131072, 768, 2688): (690, 760, 712),
    (131072, 1024, 3584): (760, 756, 718), (131072, 2048, 7168): (733, 733, 700),
    (131072, 512, V): (671, 685, 696), (131072, 768, V): (668, 723, 709),
    (16384, 512, V): (692, 724, 704),
}
# E1 FA3 varlen (real cu_seqlens, deterministic off), ms per layer per 32 x 4096 tokens: size -> (fwd, fwd+bwd)
E1_ATTN = {"130m": (0.66, 2.51), "300m": (0.95, 3.75), "520m": (1.27, 5.68), "1_2b": (1.73, 9.21),
           "pls_d48": (0.66, 2.51)}
E8_FA3_DET = 1.14          # E8c: FA3 varlen fwd+bwd, deterministic / default
CE_EXPOSED_V = {"B1": 4, "B1.5": 2, "B2": 1}   # bytes per token per vocab entry that no GEMM hides (v3)


def meas_eff(shapes):
    t = [x for s in shapes for x in E1_GEMM[s]]
    return sum(t) / len(t) * 1e12 / PEAK


def gemm_eff_meas(D):
    return meas_eff([s for s in E1_GEMM if s[1] == D and s[2] != V and s[0] == 131072])


def ce_eff_meas(D, M):
    s = (M, D, V)
    if s in E1_GEMM:
        return meas_eff([s])
    if (131072, D, V) in E1_GEMM:
        return meas_eff([(131072, D, V)])
    return gemm_eff_meas(D)


def params(c):
    D, L, H, KV, hd, F = c["D"], c["L"], c["H"], c["KV"], c["hd"], c["F"]
    muon = L * (D * H * hd + 2 * D * KV * hd + H * hd * D + 3 * D * F)
    norms = L * (4 * D + 2 * hd) + D + (c["heads"] - 1) * D
    emb = V * D
    heads = c["heads"] * V * D
    return dict(muon=muon, norms=norms, emb=emb, heads=heads, total=muon + norms + emb + heads)


def act_per_token_layer(c, policy):
    D, H, KV, hd, F = c["D"], c["H"], c["KV"], c["hd"], c["F"]
    b = dict(
        x_in=2 * D, a=2 * D, qkv=2 * (H + 2 * KV) * hd, qk_rot=2 * (H + KV) * hd, attn_o=2 * H * hd,
        oproj=2 * D, x_mid=2 * D, m=2 * D, gu=4 * F, h=2 * F, y=2 * D,
        f32_stats=4 * (4 + H + KV + H),  # 4 norm rstd, q/k-norm rstd, attention lse
    )
    if policy == "min":
        for k in ("a", "qk_rot", "m", "h"):
            b[k] = 0
    if policy == "layer":
        b = {k: (v if k == "x_in" else 0) for k, v in b.items()}
    return b


def ce_rows_for(T, heads, budget_left):
    """Largest CE chunk (8192 * 2^i, <= T) whose extra bf16 logits buffer fits in budget_left bytes."""
    rows = 8192
    while rows * 2 <= T and (rows * 2 - 8192) * V * 2 <= budget_left:
        rows *= 2
    return rows


SEGS = 24      # training b_max per row; sizes only cu_seqlens / segment tables now (no padding rows)


def detailed(name, mb, pol, zero1=True, ce_rows=8192, logits_f32=False, strict=False):
    c = SIZES[name]; D, L, H, KV, hd, F = c["D"], c["L"], c["H"], c["KV"], c["hd"], c["F"]
    p = params(c); T = mb * S; spg = c["B"] // NGPU
    def shard(n):  # elements of the largest ZeRO-1 owner shard (even split until S6 plan_ownership)
        return -(-n // NGPU) if zero1 else n
    rows = []
    def add(n, life, count, each): rows.append((n, life, count, int(each), int(count * each)))
    tag = ", ZeRO-1 owner shard" if zero1 else ""
    add("param.f32 muonh (q,k,v,o,gate,up,down)" + tag, "P", 1, 4 * shard(p["muon"]))
    add("param.f32 adam (embedding + norm gains)" + tag, "P", 1, 4 * shard(p["emb"] + p["norms"]))
    add("param.f32 adamh (lm_head%s)%s" % (" + 47 aux heads" if c["heads"] > 1 else "", tag), "P", 1, 4 * shard(p["heads"]))
    add("param.bf16 (all, replicated compute copy)", "P", 1, 2 * p["total"])
    add("grad.f32 (all; bucket order)", "P", 1, 4 * p["total"])
    add("muonh.momentum.f32" + tag, "P", 1, 4 * shard(p["muon"]))
    add("adam.mu,nu.f32" + tag, "P", 1, 8 * shard(p["emb"] + p["norms"]))
    add("adamh.mu,nu.f32" + tag, "P", 1, 8 * shard(p["heads"]))
    mats = [(D, H * hd), (KV * hd, D), (D, F)]
    mxn = max(a * b for a, b in mats); mm = max(min(a, b) ** 2 for a, b in mats)
    add("ns.workspace.f32 (X, X', A, B; 4-matrix batch)", "P", 1, 4 * (2 * mxn + 2 * mm) * 4)
    add("sched.tables.f32 (lr, adam_lr, bc1, bc2) x N", "P", 1, 16 * c["N"])
    add("rope.cos,sin.bf16 [4096, hd]", "P", 1, 2 * 2 * 4096 * hd)
    add("data.tokens.i32 [rows x 4096] x2", "P", 2, spg * S * 4)
    add("metrics.ring + stat slots", "P", 1, 64 * 2**20)
    add("mb.targets.i32 / mb.pos.i32", "A", 2, T * 4)
    add("mb.seg_table.i32 [rows*%d+1]" % SEGS, "A", 1, (mb * SEGS + 1) * 4)
    if strict:
        add("mb.ce_dw_accum.f32 [V, D] (strict per-microbatch head-gradient accumulator)", "A", 1, 4 * V * D)
    per = act_per_token_layer(c, pol)
    for k, v in per.items():
        if v: add(f"act.L*.{k}", "A", L, v * T)
    add("act.x_final.bf16 + xf.bf16 + rstd_f", "A", 1, (4 * D + 4) * T)
    if pol == "layer":
        full = sum(act_per_token_layer(c, "all").values())
        add("tr.layer_recompute (one layer's full activations)", "T", 1, full * T)
    if c["heads"] > 1:
        add("tr.head_norm_out.bf16 (recomputed per head)", "T", 1, 2 * D * T)
    add("tr.dx ping-pong bf16 [T,D] x2", "T", 2, 2 * D * T)
    add("tr.da / dm / do bf16 [T,D]", "T", 1, 2 * D * T)
    add("tr.dqkv bf16 [T,(H+2KV)hd]", "T", 1, 2 * (H + 2 * KV) * hd * T)
    add("tr.dO bf16 [T,H*hd]", "T", 1, 2 * H * hd * T)
    add("tr.dgu bf16 [T,2F]", "T", 1, 4 * F * T)
    add("tr.dh bf16 [T,F] (unfused path only)", "T", 1, 2 * F * T)
    add("tr.recompute (a|m|h|qk_rot) bf16", "T", 1, (max(2 * D, 2 * F, 2 * (H + KV) * hd) * T if pol == "min" else 0))
    add("tr.norm_gain_partials.f32 [blocks, D] x4", "T", 4, 4 * D * 1024)
    add("ws.fa3.dq_accum.f32 [H, T, hd]", "W", 1, 4 * H * hd * T)
    if H != KV: add("ws.fa3.dk,dv_accum.f32 [KV, T, hd]", "W", 2, 4 * KV * hd * T)
    add("ws.fa3.softmax_d, lse_log2 f32 [H, T]", "W", 2, 4 * H * T)
    add("ws.ce.logits %s [%d, V]" % ("f32" if logits_f32 else "bf16", ce_rows), "W", 1, (4 if logits_f32 else 2) * ce_rows * V)
    add("ws.ce.row_stats.f32 [rows, V/128] x2 + lse, gscale", "W", 1, ce_rows * (V // 128 + 1) * 8 + ce_rows * 8)
    add("ws.embed_bwd.sort (keys, idx, cub temp)", "W", 1, T * 16 + 32 * 2**20)
    add("ws.cublaslt x2 streams (split-K sized)", "W", 2, 64 * 2**20)
    tot = sum(r[4] for r in rows)
    return rows, tot


def perf(c, mb_seqs, pol, zero1=True, ce_stage="B1", fused=True, ns_prec="tf32", ce_rows=8192, name=None, meas=False):
    D, L, H, KV, hd, F = c["D"], c["L"], c["H"], c["KV"], c["hd"], c["F"]
    p = params(c)
    T = c["B"] * S // NGPU
    e = gemm_eff_meas(D) if meas else gemm_eff(D)
    n_micro = c["B"] // NGPU // mb_seqs
    out = {}
    fwd_gemm = 2 * T * (D * (H + 2 * KV) * hd + H * hd * D + 3 * D * F)
    out["gemm_trunk"] = 3 * fwd_gemm * L / (e * PEAK) * (4 / 3 if pol == "layer" else 1)
    keys = CAUSAL_COMPUTED * S / 2
    fa_fwd = 4 * T * keys * H * hd
    ef, eb = (0.40, 0.30) if hd == 64 else (0.55, 0.42)
    if meas:                                                        # E1 FA3 ms per 131,072 tokens per layer
        f_ms, fb_ms = E1_ATTN[name]
        ef = fa_fwd * 131072 / T / (f_ms * 1e-3 * PEAK)
        eb = 2.5 * fa_fwd * 131072 / T / ((fb_ms - f_ms) * 1e-3 * PEAK)
    out["attn_fwd"] = L * fa_fwd / (ef * PEAK) * (2 if pol == "layer" else 1)
    out["attn_bwd"] = L * 2.5 * fa_fwd / (eb * PEAK)
    if meas:
        out["attn_det_extra"] = (E8_FA3_DET - 1) * (out["attn_fwd"] + out["attn_bwd"])
    ce_flops = 6 * T * D * V * c["heads"]
    out["ce_gemm"] = ce_flops / ((ce_eff_meas(D, ce_rows) if meas else e) * PEAK)
    extra = (CE_EXPOSED_V if meas else {"B1": 6, "B1.5": 4, "B2": 2})[ce_stage]
    out["ce_mem"] = extra * V * T * c["heads"] / HBM
    chunks_per_head = n_micro * -(-(mb_seqs * S) // ce_rows)        # per step: ceil(micro tokens / chunk) per microbatch
    out["ce_head_grad_rmw"] = c["heads"] * chunks_per_head * 2 * V * D * 4 / HBM
    ew = (40 * D + 16 * F + 10 * (H + KV) * hd)
    if fused:
        ew -= 4 * F + 4 * F + 2 * (H + KV) * hd + 4 * (H + KV) * hd + 2 * H * hd + (4 * D if D == 512 else 0)
    ew_side = 14 * H * hd                                           # FA3 bwd accum clear, dq accum, post/pre-process
    if fused:
        ew_side -= 2 * H * hd                                       # preprocess folded into the o_proj dgrad epilogue
    ew_tot = (ew + ew_side) * T * L
    if pol == "min":
        ew_tot += (4 * D + 4 * D + 6 * F + 4 * (H + KV) * hd) * T * L
    out["elementwise"] = ew_tot / HBM
    out["embed_fb"] = (8 * D + 24) * T / HBM
    zf = 1 / NGPU if zero1 else 1
    opt_bytes = (40 * p["muon"] + 30 * (p["norms"] + p["emb"]) + 54 * p["heads"]) * zf + 2 * p["total"] * zf
    out["optimizer_elementwise"] = opt_bytes / HBM                  # clip sum-of-squares runs on comm during bwd
    mats = [(D, H * hd)] + [(KV * hd, D)] * 2 + [(D, H * hd), (D, F), (D, F), (D, F)]
    ns = sum(5 * (4 * min(a, b) ** 2 * max(a, b) + 2 * min(a, b) ** 3) for a, b in mats) * L
    rate = 0.5 * TF32 if ns_prec == "tf32" else (2 * 0.75 * FP32_FMA / 0.7 if ns_prec == "3xtf32" else 0.75 * FP32_FMA)
    out["newton_schulz"] = ns / rate * zf
    wire = (4 + 2) if zero1 else 8
    comm = wire * p["total"] * (NGPU - 1) / NGPU / NVL
    emb_rs = 4 * p["emb"] * (NGPU - 1) / NGPU / NVL
    out["comm_exposed"] = 0.10 * comm + max(0.0, emb_rs - out["newton_schulz"])
    step = sum(out.values())
    # eval: forward only over EVAL_TOKENS / NGPU tokens, every readout
    tok = EVAL_TOKENS / NGPU
    ev = (fwd_gemm / T * L / (e * PEAK) + 4 * keys * H * hd * L / (ef * PEAK) + 2 * D * V * c["heads"] / (e * PEAK)
          + ew * L / 3 / HBM) * tok
    return out, step, ev


def lev_fpt(c):
    D, L, H, KV, F = c["D"], c["L"], c["H"], c["KV"], c["F"]
    hd = D / H
    return L * (2 * 3 * D * F + 2 * D * D) + L * (2 * D * (H * hd + 2 * KV * hd) + 2 * S * H * hd + 3 * S * H
                                                  + 2 * S * hd * H) + 2 * D * V * c["heads"]


CHOSEN = [("130m", 32, "all", True), ("300m", 32, "all", True), ("520m", 8, "all", True),
          ("1_2b", 8, "all", True), ("pls_d48", 4, "min", False), ("pls_d48", 8, "all", True)]
STRICT = [("520m", 32, "layer", True), ("1_2b", 32, "layer", True)]   # --preset levanter_strict (§3.0)
STRICT_CE_ROWS = 8192   # the preset fixes the CE chunk (§3.8)
BUDGET = 72 * GiB


def chunk_rule(name, mb, pol, z):
    """CE chunk rule (DESIGN.md §3.8): the smallest chunk (8192 * 2^i) whose head-gradient read-modify-write costs
    <= 0.5% of the step, capped by what fits with 2 GiB headroom."""
    c = SIZES[name]
    _, tot0 = detailed(name, mb, pol, z, 8192)
    cap = ce_rows_for(mb * S, c["heads"], BUDGET - tot0 - 2 * GiB)
    ce_rows = 8192
    while ce_rows < cap:
        out, step, _ = perf(c, mb, pol, z, ce_rows=ce_rows)
        if out["ce_head_grad_rmw"] <= 0.005 * step:
            break
        ce_rows *= 2
    return ce_rows


def show(name, mb, pol, z, perf_rows=True, strict=False, meas=False):
    c = SIZES[name]
    ce_rows = STRICT_CE_ROWS if strict else chunk_rule(name, mb, pol, z)
    rows, tot = detailed(name, mb, pol, z, ce_rows, strict=strict)
    print(f"\n### {name} micro={mb} seqs/GPU ({mb*S} tokens) remat={pol} zero1={z} ce_rows={ce_rows}: "
          f"total {tot:,} B = {tot/GiB:.2f} GiB")
    print("| buffer | life | count | bytes each | bytes total |")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:,} | {r[4]:,} |")
    sub = {}
    for r in rows:
        sub[r[1]] = sub.get(r[1], 0) + r[4]
    print("subtotals GiB:", {k: round(v / GiB, 2) for k, v in sub.items()})
    _, totf = detailed(name, mb, pol, z, 8192, logits_f32=True, strict=strict)
    print(f"with f32 CE logits store at 8192 rows: {totf/GiB:.2f} GiB")
    if not perf_rows:
        return
    for stage in ("B1", "B1.5", "B2"):
        out, step, ev = perf(c, mb, pol, z, ce_stage=stage, ce_rows=ce_rows, name=name, meas=meas)
        mfu = 3 * lev_fpt(c) * S * c["B"] / step / (NGPU * PEAK) * 100
        print(f"budget CE {stage} (ms): " + ", ".join(f"{k}={v*1e3:.1f}" for k, v in out.items()) +
              f" | step {step*1e3:.1f} ms, tok/s {c['B']*S/step:,.0f}, levanter-MFU {mfu:.1f}%,"
              f" eval {ev:.1f} s x {c['evals']+1} = {ev*(c['evals']+1):.0f} s,"
              f" run {(step*c['N'] + ev*(c['evals']+1))/3600:.2f} h")
    out_u, step_u, _ = perf(c, mb, pol, z, ce_stage="B1", fused=False, ce_rows=ce_rows, name=name, meas=meas)
    print(f"unfused elementwise (B1): {out_u['elementwise']*1e3:.1f} ms, step {step_u*1e3:.1f} ms")
    for prec in ("f32", "3xtf32"):
        out_p, _, _ = perf(c, mb, pol, z, ce_stage="B1", ns_prec=prec, ce_rows=ce_rows, name=name, meas=meas)
        print(f"newton_schulz {prec}: {out_p['newton_schulz']*1e3:.1f} ms")


def main():
    meas = "--measured" in sys.argv[1:]
    if meas:
        print("# version 3: measured H100 efficiencies (E1 GEMM and FA3, E8 FA3 determinism, exposed CE traffic)")
        print("GEMM eff (trunk, M=131072):", {D: round(gemm_eff_meas(D), 3) for D in (512, 768, 1024, 2048)})
        print("CE GEMM eff:", {(D, M): round(ce_eff_meas(D, M), 3) for D, M in ((512, 16384), (512, 32768),
                                                                               (768, 32768), (1024, 16384), (2048, 16384))})
    for name, mb, pol, z in CHOSEN:
        show(name, mb, pol, z, meas=meas)
    print("\n## levanter_strict plans (2 x 32 sequences per GPU, remat=layer, CE chunk 8192, head-gradient accumulator)")
    for name, mb, pol, z in STRICT:
        show(name, mb, pol, z, strict=True, meas=meas)


if __name__ == "__main__":
    main()
