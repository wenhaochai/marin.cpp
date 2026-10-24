#!/usr/bin/env python3
"""s0_plan_vs_budget: `marincpp-train plan --json` reproduces scripts/design_budget.py for every plan of DESIGN.md §2.5
(the script's CHOSEN and STRICT lists): the planner's choices (micro_seqs, remat, CE chunk), every buffer group to the
byte, the P/A/T/W subtotals and totals, the f32-logits variant, and the step budget of §12.2 (all rows, three CE stages,
unfused, Newton-Schulz precisions, eval, MFU; every row bitwise in double, totals with CPython 3.12's sum()). The
CHOSEN plans run with every choice on auto; the STRICT plans run exactly `--size S --preset levanter_strict`, the
command a strict run uses. It also checks the numbers printed in DESIGN.md §2.5 and §12.2.

Usage: test_plan_vs_budget.py --bin <marincpp-train> --src <repo root>      (plain Python 3, no dependencies)
"""
import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys

GIB = 1024 ** 3
FAILS = []

# script row name prefix -> C++ buffer group (the order matters where one prefix extends another)
GROUP_OF = [
    ("param.f32 muonh", "param.f32.muonh"), ("param.f32 adamh", "param.f32.adamh"),
    ("param.f32 adam", "param.f32.adam"), ("param.bf16", "param.bf16"), ("grad.f32", "grad.f32"),
    ("muonh.momentum.f32", "muonh.momentum"), ("adam.mu,nu.f32", "adam.mu_nu"), ("adamh.mu,nu.f32", "adamh.mu_nu"),
    ("ns.workspace.f32", "ns.workspace"), ("sched.tables.f32", "sched.tables"), ("rope.cos,sin.bf16", "rope"),
    ("data.tokens.i32", "data.tokens"), ("metrics.ring", "metrics.ring"),
    ("mb.targets.i32 / mb.pos.i32", "mb.targets_pos"), ("mb.seg_table.i32", "mb.seg_table"),
    ("mb.ce_dw_accum.f32", "mb.ce_dw_accum"),
    ("act.x_final.bf16", "act.final"), ("tr.layer_recompute", "tr.layer_recompute"),
    ("tr.head_norm_out.bf16", "tr.head_norm_out"), ("tr.dx ping-pong", "tr.dx"), ("tr.da / dm / do", "tr.da_dm_do"),
    ("tr.dqkv", "tr.dqkv"), ("tr.dO", "tr.dO"), ("tr.dgu", "tr.dgu"), ("tr.dh", "tr.dh"),
    ("tr.recompute", "tr.recompute"), ("tr.norm_gain_partials", "tr.norm_gain_partials"),
    ("ws.fa3.dq_accum", "ws.fa3.dq_accum"), ("ws.fa3.dk,dv_accum", "ws.fa3.dkv_accum"),
    ("ws.fa3.softmax_d, lse_log2", "ws.fa3.stats"), ("ws.ce.logits", "ws.ce.logits"),
    ("ws.ce.row_stats", "ws.ce.row_stats"), ("ws.embed_bwd.sort", "ws.embed_bwd"), ("ws.cublaslt", "ws.cublaslt"),
]


def group_of(row_name):
    m = re.match(r"act\.L\*\.(\w+)$", row_name)
    if m:
        return row_name
    for prefix, g in GROUP_OF:
        if row_name.startswith(prefix):
            return g
    raise KeyError("script row without a C++ group: " + row_name)


def neumaier_sum(values):
    """CPython >= 3.12 sum() of floats, which the C++ budget mirrors. The script's own `step` depends on the
    interpreter (3.11 and older add plainly), so the expected totals are recomputed here from the script's rows."""
    s, c = 0.0, 0.0
    for x in values:
        t = s + x
        c += (s - t) + x if abs(s) >= abs(x) else (x - t) + s
        s = t
    return s + c if c and math.isfinite(c) else s


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("    FAIL " + msg)


def plan(binary, args):
    out = subprocess.run([binary, "plan", "--json"] + args, capture_output=True, text=True)
    if out.returncode not in (0, 1) or not out.stdout.strip():
        raise RuntimeError("marincpp-train plan %s failed (%d): %s" % (" ".join(args), out.returncode, out.stderr))
    return json.loads(out.stdout)


def compare_case(db, binary, name, mb, pol, z, strict):
    tag = "%s micro=%d remat=%s zero1=%s%s" % (name, mb, pol, z, " (--preset levanter_strict)" if strict else "")
    print("case " + tag)
    args = ["--size", name] + ([] if z else ["--set", "optim.zero1=off"])
    if strict:
        args += ["--preset", "levanter_strict"]
    j = plan(binary, args)
    ch = j["choice"]
    ce_rows = db.STRICT_CE_ROWS if strict else db.chunk_rule(name, mb, pol, z)
    check(ch["micro_seqs"] == mb, "%s: micro_seqs %d, script %d" % (tag, ch["micro_seqs"], mb))
    check(ch["remat"] == pol, "%s: remat %s, script %s" % (tag, ch["remat"], pol))
    check(ch["zero1"] == z, "%s: zero1 %s" % (tag, ch["zero1"]))
    check(ch["ce_chunk_rows"] == ce_rows, "%s: CE chunk %d, script %d" % (tag, ch["ce_chunk_rows"], ce_rows))
    check(j["fits"], tag + ": does not fit")

    rows, tot = db.detailed(name, mb, pol, z, ce_rows, strict=strict)
    want, want_cls, sub = {}, {}, {}
    for r in rows:
        if r[4] == 0:
            continue
        g = group_of(r[0])
        want[g] = want.get(g, 0) + r[4]
        want_cls[g] = r[1]
        sub[r[1]] = sub.get(r[1], 0) + r[4]
    got = {g["name"]: g["bytes"] for g in j["groups"]}
    got_cls = {g["name"]: g["class"] for g in j["groups"]}
    for g in sorted(set(want) | set(got)):
        check(want.get(g) == got.get(g), "%s: group %s: C++ %s B, script %s B" % (tag, g, got.get(g), want.get(g)))
        if g in want and g in got:
            check(want_cls[g] == got_cls[g], "%s: group %s class %s, script %s" % (tag, g, got_cls[g], want_cls[g]))
    check(j["payload_bytes"] == tot, "%s: total %d B, script %d B" % (tag, j["payload_bytes"], tot))
    for k in "PATW":
        check(j["subtotals"][k] == sub.get(k, 0), "%s: subtotal %s %d, script %d" % (tag, k, j["subtotals"][k], sub.get(k, 0)))
    check(j["device_bytes"] - j["payload_bytes"] < 256 * len(j["groups"]) * 8,
          "%s: arena alignment overhead %d B" % (tag, j["device_bytes"] - j["payload_bytes"]))
    print("    total %d B = %.2f GiB, CE chunk %d: groups equal" % (tot, tot / GIB, ce_rows))

    # f32 logits store at 8192 rows (the script's "with f32 CE logits store" line; a byte count only, so the budget is
    # raised: pls option A needs 72.30 GiB with it)
    jf = plan(binary, ["--size", name, "--set", "exec.micro_seqs=%d" % mb, "--set", "exec.remat=" + pol,
                       "--set", "exec.ce_chunk_rows=8192", "--set", "exec.ce_logits=f32",
                       "--set", "exec.mem_budget_gib=79"]
              + ([] if z else ["--set", "optim.zero1=off"]) + (["--preset", "levanter_strict"] if strict else []))
    _, totf = db.detailed(name, mb, pol, z, 8192, logits_f32=True, strict=strict)
    check(jf["payload_bytes"] == totf, "%s: f32-logits total %d, script %d" % (tag, jf["payload_bytes"], totf))

    # step budget, every row, bitwise in double
    c = db.SIZES[name]
    for stage in ("B1", "B1.5", "B2"):
        out, _, ev = db.perf(c, mb, pol, z, ce_stage=stage, ce_rows=ce_rows)
        step = neumaier_sum(out.values())
        b = j["budget"][stage]
        for k, v in out.items():
            check(b["rows"][k] == v, "%s %s: budget row %s %r, script %r" % (tag, stage, k, b["rows"][k], v))
        check(b["step_s"] == step, "%s %s: step %r, script %r" % (tag, stage, b["step_s"], step))
        check(b["eval_s"] == ev, "%s %s: eval %r, script %r" % (tag, stage, b["eval_s"], ev))
        mfu = 3 * db.lev_fpt(c) * db.S * c["B"] / step / (db.NGPU * db.PEAK) * 100
        check(b["mfu_pct"] == mfu, "%s %s: MFU %r, script %r" % (tag, stage, b["mfu_pct"], mfu))
        check(b["tokens_per_s"] == c["B"] * db.S / step, "%s %s: tokens/s" % (tag, stage))
    out_u, _, _ = db.perf(c, mb, pol, z, ce_stage="B1", fused=False, ce_rows=ce_rows)
    step_u = neumaier_sum(out_u.values())
    check(j["budget"]["unfused_B1_step_s"] == step_u, "%s: unfused step %r, script %r" % (tag, j["budget"]["unfused_B1_step_s"], step_u))
    for prec, key in (("f32", "ns_f32_s"), ("3xtf32", "ns_3xtf32_s")):
        o, _, _ = db.perf(c, mb, pol, z, ce_stage="B1", ns_prec=prec, ce_rows=ce_rows)
        check(j["budget"][key] == o["newton_schulz"], "%s: NS %s %r, script %r" % (tag, prec, j["budget"][key], o["newton_schulz"]))
    check(j["budget"]["n_evals"] == c["evals"] + 1, "%s: n_evals %d, script %d" % (tag, j["budget"]["n_evals"], c["evals"] + 1))
    print("    step B1 %.1f ms, B1.5 %.1f ms, B2 %.1f ms: budget rows equal"
          % tuple(j["budget"][s]["step_s"] * 1e3 for s in ("B1", "B1.5", "B2")))
    return j


def design_tables(src, results):
    """The numbers DESIGN.md prints in §2.5 and §12.2 (they came from the script; this guards the document)."""
    text = open(src + "/docs/DESIGN.md", encoding="utf-8").read()
    j130 = results[("130m", False)]
    m = re.search(r"\*\*total\*\* \| \| \*\*([0-9,]+) \(([0-9.]+) GiB\)\*\*", text)
    check(m and int(m.group(1).replace(",", "")) == j130["payload_bytes"], "DESIGN §2.5 130m total bytes")
    for key, label in ((("pls_d48", False), "Option B (default)"), (("pls_d48", True), "Option A, pure DDP")):
        m = re.search(re.escape(label) + r"[^*]*\*\*([0-9.]+) GiB\*\*", text)
        check(m and m.group(1) == "%.2f" % (results[key]["payload_bytes"] / GIB), "DESIGN §2.5 pls %s total" % label)
    names = {"300m": ("300m", False), "520m": ("520m", False), "1.2B": ("1_2b", False),
             "520m strict": ("520m", "strict"), "1.2B strict": ("1_2b", "strict")}
    for label, key in names.items():
        m = re.search(r"^\| " + re.escape(label) + r" \|(.*)\|\s*$", text, re.M)
        if not m:
            check(False, "DESIGN §2.5 row %s not found" % label)
            continue
        cells = [x.strip() for x in m.group(1).split("|")]
        j = results[key]
        sub = [round(j["subtotals"][k] / GIB, 2) for k in "PATW"]
        check([float(x) for x in cells[4:8]] == sub, "DESIGN §2.5 %s subtotals %s vs %s" % (label, cells[4:8], sub))
        check(cells[8] == "%.2f GiB" % (j["payload_bytes"] / GIB), "DESIGN §2.5 %s total %s" % (label, cells[8]))
        check(int(cells[3].replace(",", "")) == j["choice"]["ce_chunk_rows"], "DESIGN §2.5 %s CE chunk" % label)
    m = re.search(r"\| levanter_strict step \(B1\) \|(.*)\|", text)
    cells = [x.strip() for x in m.group(1).split("|")] if m else []
    want = ["—", "—"] + ["%.1f" % (results[k]["budget"]["B1"]["step_s"] * 1e3) for k in
                         (("520m", "strict"), ("1_2b", "strict"))] + ["—"]
    check(cells == want, "DESIGN §12.2 levanter_strict step (B1): %s vs C++ %s" % (cells, want))
    order = [("130m", False), ("300m", False), ("520m", False), ("1_2b", False), ("pls_d48", False)]
    for stage, pat in (("B1", r"\| \*\*step \(B1\)\*\* \|(.*)\|"), ("B1.5", r"\| step \(B1\.5\) \|(.*)\|"),
                       ("B2", r"\| step \(B2\) \|(.*)\|")):
        m = re.search(pat, text)
        cells = [x.strip().strip("*") for x in m.group(1).split("|")] if m else []
        want = ["%.1f" % (results[k]["budget"][stage]["step_s"] * 1e3) for k in order]
        check(cells == want, "DESIGN §12.2 step (%s): %s vs C++ %s" % (stage, cells, want))
    print("DESIGN.md §2.5 and §12.2 numbers checked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("design_budget", a.src + "/scripts/design_budget.py")
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    results = {}
    for name, mb, pol, z in db.CHOSEN:
        results[(name, not z)] = compare_case(db, a.bin, name, mb, pol, z, strict=False)
    for name, mb, pol, z in db.STRICT:
        results[(name, "strict")] = compare_case(db, a.bin, name, mb, pol, z, strict=True)
    design_tables(a.src, results)
    print("s0_plan_vs_budget: %d cases, %d failures" % (len(db.CHOSEN) + len(db.STRICT), len(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
