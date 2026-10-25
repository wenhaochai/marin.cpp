"""Collect the T1.0-T1.6 levanter fixtures into one JSONL table, a markdown table and a README.

    python -m tools.levref.t1_report [--out cache/levref/t1]

Reads cache/levref/<size>/{t1_precision,t1_units,t1_oracle}/ (tools.levref.precision, .units, .oracle) and the A100 debug
facts cache/levref/130m/precision_a100_vis/ when present, and writes <out>/results.jsonl (one row per check x size x
tensor), <out>/table.md and <out>/README.md (what ran, table, T1.0 readings, conclusion, caveats, review fixes).
Missing fixtures are reported, not skipped.
"""
import argparse
import json
import os
import time

from tools.levref import common

SIZES_PER_CHECK = {"T1.2": ("130m", "1_2b"), "T1.3": ("130m", "1_2b"), "T1.4": ("130m", "520m", "1_2b"),
                   "T1.5": ("130m", "1_2b"), "T1.6": ("130m", "1_2b")}
FIXTURES = (("130m", "t1_precision", "T1.0 facts + T1.6 inputs, 4 GPUs (t1_gpu.sh precision130)"),
            ("1_2b", "t1_precision", "T1.6 inputs, 1 GPU (t1_gpu.sh precision1_2b)"),
            ("130m", "t1_units", "T1.2-T1.5, 1 GPU (t1_gpu.sh units)"), ("520m", "t1_units", "T1.4, 1 GPU (t1_gpu.sh units)"),
            ("1_2b", "t1_units", "T1.2-T1.5, 1 GPU (t1_gpu.sh units)"),
            ("130m", "t1_oracle", "f64 oracles, CPU (t1_finish.sh)"), ("520m", "t1_oracle", "f64 oracles, CPU (t1_finish.sh)"),
            ("1_2b", "t1_oracle", "f64 oracles, CPU (t1_finish.sh)"))
# A100 x1, batch 1 debug runs (they check that the scripts run; not H100 references): the review rerun with the fixed
# precision.py first, else the first run (its facts.json regenerated with the fixed hlofacts)
DEBUG_DIRS = (("130m", "precision_a100_review_vis"), ("130m", "precision_a100_vis"))

REVIEW_FIXES = """\
Review of 2026-10-24 (checker findings on the first T1 submission; every item below is fixed in the code or stated here):

1. precision.py asserted momentum == 0.95, so the 1_2b T1.6 task (momentum 0.98; 300m and 520m also 0.98) stopped with
   an AssertionError right after the config build. The assertion now covers only the NS call (5 steps, eps 1e-5,
   "simple", Nesterov), and momentum and max_grad_norm must equal the logged W&B config of the size. step.json reports
   |u| / ((1 + momentum) |g|) (`u_over_1pm_g_layer0`, was a fixed 1.95), and ns/index.json names the size's
   clip_by_global_norm(max_grad_norm) (2.0 at 1_2b, was a fixed 1.0).
2. The DEFAULT == TF32 reading for the Newton-Schulz and flash dV dots was an inference: the runtime probe compiled to
   cuBLASLt, while all 105 in-step NS dots and the flash dV dot are Triton nested GEMM fusions. precision.py now compiles
   NS per GEMM emitter (XLA's autotuned choice; Triton pinned with xla_gpu_cublas_fallback=false; cuBLAS pinned with
   xla_gpu_enable_triton_gemm=false) at DEFAULT, under default_matmul_precision('tensorfloat32') and at HIGHEST, records
   bitwise DEFAULT == tensorfloat32 and the GEMMs of each compiled module. It also runs f32 GEMM probes whose operands
   come out of a fusion, as in NS (512^3 compiles to a Triton nested GEMM fusion, 2048^3 stays on cuBLASLt), with an
   input-format fingerprint: rel(output, f64 product of the operands rounded to tf32 / bf16 / kept f32). On the A100
   debug run both emitters give DEFAULT == tensorfloat32 bitwise and pick tf32 round-to-nearest (rel 7e-6 to 1e-5,
   against 2.9e-4 for f32 inputs, 8.7e-4 for truncation, 2.4e-3 for bf16); H100 must repeat it (t1_gpu.sh precision130).
   With --dump, precision.py now also turns off the JAX persistent compilation cache (set by the login profile), because
   a cache hit leaves the dump without config.pbtxt, so the T1.0 flags block came out empty.
3. The A100 debug CE facts came from a silent fallback: 'Requested weight tile exceeds GPU shared-memory budget', then
   'selected implementation: xla'. On H100 levanter takes batched_xla (B-tiled, `_backward_b_tiled_from_lse`), whose
   backward logits dot has no preferred_element_type (bf16 output, then astype(f32)) and whose dlogits are cast to bf16,
   so grad_x and grad_w are bf16 x bf16 GEMMs (grad_w summed in f32 over the row tiles). The XLA fallback instead has f32
   backward logits, f32 x f32 dlogits GEMMs and grad_x summed in bf16 over 16 vocab blocks (xla.py gx_init =
   zeros_like(x)); that bf16 sum explains rel(grad_x) = 1.3e-2 of the CPU debug oracle. The debug CE readings (T1.0
   items 2 and 5) therefore must not feed exec.ce_logits or any other decision. precision.py now stores the CE
   implementation log, and hlofacts names the CE source file of every CE GEMM (batched_xla.py vs xla.py / reference.py).
4. The f32 x f32 dots of the debug step are 105 Newton-Schulz + 2 CE (fallback only) + 1 flash-attention dV
   (f32[8,1024,1024] x f32[8,1024,64], DEFAULT precision, Triton); the earlier reading left out the dV GEMM.
5. 'Norm-gain reductions accumulate in f32 and store bf16' was too broad. In every case the summands are products
   already rounded to bf16 and the accumulator is f32. The final-norm gain gradient stays f32 (it goes straight into the
   Adam fusion); only the 4 in-scan block-norm gains per layer (hybrid norm) and the q/k-norm gains are stored as bf16.
6. hlofacts.py bugs: (a) `stored` took the dtype of the first tuple element of a multi-output fusion
   (input_convert_reduce_fusion.3 returns (bf16[...], f32[128,64]); the reduce output is f32); it now follows the reduce
   to its own fusion output element; (b) `_consumers` did not follow a GEMM that is the root of its (Triton) fusion out to
   the users of that fusion, so bf16_convert_after could be a false negative; it does now; (c) gain_reductions listed 35
   optimizer [6,512] per-layer norms; only backward reductions are listed now (the others are counted as excluded).
7. cache/levref/130m/precision_a100_vis/facts.json was the stale output of the buggy hlofacts ('2_ce_logit_dots' 0/0,
   scatter reducer 'f32'); it is regenerated with the fixed hlofacts from the same dump.
8. T1.2 dg reduces over rows 0-1 only (see Caveats for the lengths), not over the 524,288 positions of the spec's T1.2
   row; the spec contradicts itself here (its T1 preamble says rows 0-1).
9. No task wrote <size>/t1_oracle or this directory, and t1_report wrote no README.md. tools/levref/t1_finish.sh now runs
   tools.levref.oracle (130m, 520m, 1_2b) and this report on della-vis1 (CPU) after the GPU tasks; this README is part
   of the report.
"""


def _j(*p):
    path = os.path.join(*p)
    return json.load(open(path)) if os.path.exists(path) else None


def _e(x, f="{:.2e}"):
    return f.format(x) if isinstance(x, (int, float)) else str(x)


def t10_lines(facts: dict, step: dict | None) -> list[str]:
    """Markdown bullets of the six T1.0 items from a facts.json (hlofacts) and its step.json."""
    s = facts.get("summary", {})
    L = []
    gr = s.get("1_gradient_reduction_dtypes") or {}
    L.append(f"1. Cross-GPU gradient reductions: {gr or 'none (one device)'}; collectives {s.get('1_collective_totals') or {}}.")
    cp = s.get("2_ce_path") or {}
    lg, bk = (s.get("2_ce_logit_gemms") or {}).get("forward", {}), (s.get("2_ce_logit_gemms") or {}).get("backward", {})
    gg = s.get("2_ce_grad_gemms") or {}
    L.append(f"2. CE path: {cp.get('reading', '?')}. Logits GEMMs: forward {lg.get('operands')} -> {lg.get('results')} "
             f"(convert to bf16 after: {lg.get('bf16_convert_after')}), backward {bk.get('operands')} -> {bk.get('results')} "
             f"(convert to bf16 after: {bk.get('bf16_convert_after')}); dlogits GEMMs {gg.get('operands')} -> {gg.get('results')} "
             f"via {gg.get('implementations')}.")
    for x in s.get("3_scatter") or []:
        L.append(f"3. Vocab-sized scatter ({x.get('role', 'embedding gradient')}) {x.get('result')}: {x.get('reducer')}.")
    if not s.get("3_scatter"):
        L.append("3. Embedding-gradient scatter: none found.")
    gains = s.get("4_gain_reductions") or {}
    L.append("4. Norm-gain gradient reductions (backward only; summands = dtype before the upcast; leaves_fusion_as = dtype "
             "the value is stored in): " + "; ".join(f"{k} (x{v})" for k, v in gains.items())
             + f". Excluded forward/optimizer reductions of the same shapes: {s.get('4_excluded_reductions')}.")
    fd = s.get("5_f32_dots") or {}
    cats = "; ".join(f"{k} {v.get('n')} via {v.get('implementations')} at {v.get('operand_precision')}, algorithm "
                     f"{v.get('algorithms')}, e.g. {(v.get('shapes') or ['?'])[0]}" for k, v in (fd.get("by_category") or {}).items())
    L.append(f"5. f32 x f32 dots: {fd.get('n')} = {cats}.")
    rt = fd.get("runtime") or {}
    for k, v in rt.items():
        if k.startswith("f32_gemm_probe"):
            L.append(f"   - runtime, {k}: {v}")
    for em, v in (rt.get("newton_schulz") or {}).items():
        L.append(f"   - runtime, Newton-Schulz standalone, emitter {em}: {v}")
    L.append(f"6. Flags: {s.get('6_flags')}.")
    if s.get("libraries"):
        L.append(f"   Libraries mapped: {', '.join(x for x in s['libraries'] if any(t in x for t in ('cublas', 'cudnn', 'nccl')))}.")
    if step:
        L.append(f"   Step: devices {step.get('devices')} ({step.get('device_kind')}), debug_batch {step.get('debug_batch')}, "
                 f"train_step loss {(step.get('train_step') or {}).get('loss')}, CE log {step.get('ce_implementation_log')}.")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = os.path.join(common.nano_dir(), "cache", "levref")
    out = a.out or os.path.join(base, "t1")
    os.makedirs(out, exist_ok=True)
    rows, md = [], []

    def row(check, size, tensor, **kv):
        rows.append(dict(check=check, size=size, tensor=tensor, **kv))

    # T1.0
    f = _j(base, "130m", "t1_precision", "facts.json")
    st = _j(base, "130m", "t1_precision", "step.json")
    if f is None:
        row("T1.0", "130m", None, missing="130m/t1_precision/facts.json")
    else:
        row("T1.0", "130m", "summary", **f["summary"])
        row("T1.0", "130m", "step", devices=st.get("devices"), device_kind=st.get("device_kind"), train_step=st.get("train_step"),
            f32_gemm_probe_by_emitter=st.get("f32_gemm_probe_by_emitter") or st.get("f32_gemm_probe"), batch0=st.get("batch0"),
            ce_implementation_log=st.get("ce_implementation_log"))
    # T1.2 / T1.4 / T1.5 oracle rels, T1.3 bitwise facts, T1.6 NS
    norm_positions = {}
    for check, part in (("T1.2", "norm"), ("T1.4", "attn"), ("T1.5", "ce")):
        for size in SIZES_PER_CHECK[check]:
            r = _j(base, size, "t1_oracle", part, "rel.json")
            res = _j(base, size, "t1_units", part, "results.json")
            if part == "norm" and res:
                norm_positions[size] = {k: v.get("n_positions") for k, v in res.items()}
            if r is None:
                row(check, size, None, missing=f"{size}/t1_oracle/{part}/rel.json", units_present=res is not None)
                continue
            for t, v in r.items():
                row(check, size, t, **v)
            if res is not None:
                if part == "norm":
                    row(check, size, "results", **res)
                else:
                    row(check, size, "results", **{k: v for k, v in res.items() if k not in ("implementation_log",)},
                        implementation_log=(res.get("implementation_log") or [])[-3:])
    for size in SIZES_PER_CHECK["T1.3"]:
        res = _j(base, size, "t1_units", "rope", "results.json")
        row("T1.3", size, "results", **(res or dict(missing=f"{size}/t1_units/rope/results.json")))
    for size in SIZES_PER_CHECK["T1.6"]:
        r = _j(base, size, "t1_oracle", "ns", "rel.json")
        sj = _j(base, size, "t1_precision", "step.json")
        if r is None:
            row("T1.6", size, None, missing=f"{size}/t1_oracle/ns/rel.json", precision_present=sj is not None)
        else:
            for t, v in r.items():
                row("T1.6", size, t, **v)
        if sj is not None:
            for t, v in (sj.get("newton_schulz") or {}).items():
                row("T1.6", size, f"{t} emitters", **{em: {k: x for k, x in e.items() if k != "gemm"} for em, e in (v.get("emitters") or {}).items()})

    with open(os.path.join(out, "results.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    md.append("| check | size | tensor | rel(lev, f64 oracle) | max abs(lev - oracle) |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        if "rel_lev_oracle" in r:
            md.append(f"| {r['check']} | {r['size']} | {r['tensor']} | {r['rel_lev_oracle']:.3e} | {r['max_abs_lev_minus_oracle']:.3e} |")
        elif "rel_default_oracle" in r:
            md.append(f"| {r['check']} | {r['size']} | NS {r['tensor']} | default {r['rel_default_oracle']:.3e}, highest "
                      f"{r['rel_highest_oracle']:.3e} | rel(default, highest) {r['rel_default_highest']:.3e} |")
        elif str(r.get("tensor", "")).endswith(" emitters"):
            em = {k: v for k, v in r.items() if isinstance(v, dict)}
            txt = ", ".join(f"{k}: default==tf32 {v.get('default_bitwise_equal_tensorfloat32')}" for k, v in em.items())
            md.append(f"| {r['check']} | {r['size']} | NS {r['tensor']} | {txt} | |")
        elif r.get("tensor") == "weighted_mean":
            md.append(f"| {r['check']} | {r['size']} | weighted mean loss | abs rel diff {r['abs_rel_diff']:.3e} | "
                      f"lev {r['lev_f64_of_f32']:.7f} |")
        elif "missing" in r:
            md.append(f"| {r['check']} | {r['size']} | MISSING {r['missing']} | | |")
    open(os.path.join(out, "table.md"), "w").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- README
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    ran, missing = [], []
    for size, name, what in FIXTURES:
        man = _j(base, size, name, "manifest.json")
        if man is None:
            missing.append(f"{size}/{name}")
            ran.append(f"| {size}/{name} | {what} | not run | | | |")
            continue
        env = man.get("environment") or {}
        dev = env.get("devices") or [env.get("numpy") and "cpu (numpy " + env["numpy"] + ")"]
        vers = f"jax {env.get('jax')}, jaxlib {env.get('jaxlib')}" if env.get("jax") else f"numpy {env.get('numpy')}"
        ran.append(f"| {size}/{name} | {what} | `{man.get('command', '?')}` | {man.get('hostname')}, {len(dev)} x {dev[0] if dev else '?'} "
                   f"| {man.get('written_utc')} | {vers}, marin {str(env.get('marin_commit', ''))[:10]} |")
    have_h100 = f is not None and "H100" in str((st or {}).get("device_kind"))
    R = [f"# levref-t1: levanter unit fixtures T1.0-T1.6 (130m, 520m, 1_2b)", "",
         f"Status ({now}): {len(FIXTURES) - len(missing)} of {len(FIXTURES)} fixture directories written"
         + (f"; missing: {', '.join(missing)}." if missing else "."), "",
         "## What ran", "", "| fixture | purpose | command | host, devices | written | versions |", "|---|---|---|---|---|---|", *ran, "",
         "GPU fixtures run through tools/devq (1 or 4 H100 of the shared allocation) with `tools/levref/t1_gpu.sh <preset>`; "
         "the f64 oracles and this report run on della-vis1 with `tools/levref/t1_finish.sh`.", "",
         "## Results", "", *md, ""]
    if f is not None:
        R += ["## T1.0 readings (130m train step)", "", *t10_lines(f, st), ""]
    # conclusion
    R += ["## Conclusion", ""]
    if not have_h100:
        R += ["No H100 fixture exists yet: the GPU presets wait in the shared devq queue, so every T1 tolerance class is still "
              "open. The scripts ran end to end on della-vis1 (CPU and A100 x1, debug batches), and the fixed hlofacts gives "
              "the readings below from the A100 debug dump. Those readings are not H100 facts: the A100 CE ran levanter's XLA "
              "fallback, so T1.0 items 2 and 5 from it must not feed exec.ce_logits or any other decision.", ""]
    else:
        s = f["summary"]
        fd = s.get("5_f32_dots", {})
        nsr = (fd.get("runtime") or {}).get("newton_schulz") or {}
        tri = nsr.get("triton") or nsr.get("autotuned") or {}
        R += [f"On H100 the 130m step reduces gradients in {list((s.get('1_gradient_reduction_dtypes') or {}).keys())}, its CE "
              f"takes {(s.get('2_ce_path') or {}).get('reading')}, and its f32 dots are "
              + ", ".join(f"{v.get('n')} {k}" for k, v in (fd.get("by_category") or {}).items())
              + f" at DEFAULT precision. Newton-Schulz at DEFAULT equals 'tensorfloat32' bitwise on "
              f"{tri.get('default==tensorfloat32')} of {tri.get('matrices')} matrices on the Triton emitter, so T1.6 uses the "
              "oracle rule with levanter at HIGHEST as the oracle when DEFAULT != HIGHEST. The rel(lev, oracle) values in the "
              "table are levanter's own error, the reference the port must not exceed by more than 25% (oracle rule).", ""]
    # caveats
    R += ["## Caveats", ""]
    np_txt = "; ".join(f"{s_}: " + ", ".join(f"{k} {v}" for k, v in d.items()) for s_, d in norm_positions.items())
    R.append("- T1.2 dg is reduced over batch-0 rows 0-1 only (units.py). Summands per gain element: "
             + (np_txt + " (units results.json n_positions)" if np_txt else
                "130m: block norm 8,192, q norm 65,536, k norm 65,536; 1_2b: block 8,192, q 131,072 (16 heads), k 65,536 (8 KV heads)")
             + ". That is not the 524,288 positions of the spec's T1.2 row (the spec's T1 preamble says rows 0-1), so the dg "
               "tolerance is a rows-0-1 result, not a full-microbatch result.")
    R.append("- T1.0 item 5 readings are HLO attributes (DEFAULT precision, no algorithm attribute). What DEFAULT means at run "
             "time is measured on the f32 GEMM probe and on standalone Newton-Schulz compiles per emitter, not inside the "
             "step's own kernels; the triton rows measure the same fusion kind as the in-step NS and flash dV dots.")
    R.append("- Bitwise DEFAULT == tensorfloat32 needs both compiles to pick the same kernels, and separately autotuned compiles "
             "need not. A100 debug: on pinned cuBLAS, NS was bitwise equal for 5 of 7 matrices; gate_proj and up_proj differ by "
             "rel 5.9e-4 and 5.5e-4, yet both variants sit 6e-4 from the f64 oracle (HIGHEST: 1.2e-6), so both are TF32 class. "
             "The format reading is the probe fingerprint and rel(NS, oracle); bitwise equality only supports it. The autotuner's "
             "Triton/cuBLAS split of the NS GEMMs also changed between A100 runs (92/13 vs 43/62 of 105).")
    R.append("- The 1_2b T1.6 input u comes from 1 GPU (8 microbatches of 32 rows instead of the run's 2 x 128 on 4 GPUs); it "
             "differs from the run's update 0 only in f32 accumulation order, and T1.6 uses it as a fixed input.")
    R.append("- T1.5 and T1.0 item 2 describe the CE implementation named in each fixture's implementation log; a run that "
             "logs a fallback (as the A100 debug run did) does not describe the H100 batched_xla path.")
    R.append("- Debug fixtures (`*_vis`, `*_debug*`) only show that the scripts run; they use 1-4 rows and other devices.")
    R.append("")
    # debug readings
    DEBUG = next((d for d in DEBUG_DIRS if os.path.exists(os.path.join(base, *d, "facts.json"))), DEBUG_DIRS[-1])
    df, ds = _j(base, *DEBUG, "facts.json"), _j(base, *DEBUG, "step.json")
    if df is not None:
        R += [f"## Debug readings: A100 x1, batch 1 (`{DEBUG[0]}/{DEBUG[1]}/facts.json`, not H100)", "",
              "Regenerated with the fixed hlofacts from the same dump. The CE lines describe levanter's XLA fallback, not the "
              "H100 batched_xla path; do not use them for exec.ce_logits.", "", *t10_lines(df, ds), ""]
        dorc = {p: _j(base, "130m", "oracle_cpu_vis", p, "rel.json") for p in ("norm", "attn", "ce")}
        if all(dorc.values()):
            R += ["CPU debug oracle (`130m/oracle_cpu_vis`, 2 rows, CPU f32 dots, XLA-fallback CE): "
                  + ", ".join(f"{p}.{t} {v['rel_lev_oracle']:.1e}" for p, d in dorc.items() for t, v in d.items() if "rel_lev_oracle" in v)
                  + ". rel(grad_x) = 1.3e-2 comes from the fallback's bf16 grad_x sum over 16 vocab blocks.", ""]
    R += ["## Review fixes", "", REVIEW_FIXES]
    open(os.path.join(out, "README.md"), "w").write("\n".join(R) + "\n")
    print("\n".join(md))
    print("wrote", out, len(rows), "rows; README.md;", "missing:", missing)


if __name__ == "__main__":
    main()
