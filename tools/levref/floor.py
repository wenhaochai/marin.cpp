"""e_floor and the bf16-vs-f32 floors per tensor, from the step fixtures (CPU; run after tools.levref.step).

    python -m tools.levref.floor 130m

Reads cache/levref/<size>/{step0_bf16, step0_f32, rep_bf16, rep_f32}/ (the rep_* directories are optional) and writes
cache/levref/<size>/floor/{e_floor.json, manifest.json}. All rel values are ||a - b||_2 / ||b||_2 in float64 over one
tensor (kaiyue_baseline.md, Parity test plan, Conventions).

e_floor.json
  floors.<kind>_<policy>.<tensor> = {inproc, xproc, e_floor}
      kind grad   = g0 (step-0 gradient), update0 = u0, state1 = optimizer state after update 0 (per state leaf)
      inproc  rel(repeat 1, repeat 0) of ref_step in the step0_<policy> process (rep_inproc.json)
      xproc   rel(rep_<policy>, step0_<policy>): two processes, each with its own compilation and autotuning
      e_floor max of the two (the larger levanter-vs-levanter difference; null parts are skipped)
  bf16_vs_f32.<tensor> = {grad_rel, grad_cos, update0_rel, update1_rel}: levanter bf16 vs levanter f32 (reference = f32)
  tol.<tensor> = the pass bounds, ready to use:
      a4         max(1e-5, 2 e_floor(grad_f32))                 T2.4: rel(g_nano_f32, g_lev_f32) <= a4
      a5_rel     2 rel(g_lev_bf16, g_lev_f32)                    T2.5: rel(g_nano_bf16, g_lev_f32) <= a5_rel, cos >= 0.9999
      a7_update0 2 rel(u0_lev_bf16, u0_lev_f32)                  T2.7 130m: rel(u0_nano, u0_lev) <= a7_update0
      a7_update1 2 rel(u1_lev_bf16, u1_lev_f32)                  T2.7 300m+ (iii)
  tol_state.<policy>.<state leaf> = max(1e-5, 2 e_floor(state1_<policy>))   T2.7 300m+ (i)
  losses.<policy> = L0 of every repeat (f32 hex) and the in-/cross-process |dL|
  group_norms.<policy> = the pre-clip norms of every repeat and their largest relative spread
  consistency = per fixture: init digest, batch-0 hash, debug rows, devices, matmul precision of the main chain, CE
      implementation. A floor compares two processes only if they started from the same inputs and ran the same
      program, so floor.py stops (exit 1) when the init digests or batch hashes differ between any two fixtures, or when
      a rep_<policy> process differs from its step0_<policy> process in devices, precision or CE implementation.
No floor is measured for update 1: the T2.7 update-1 gate (300m and larger) is 2 x the bf16-vs-f32 update-1 difference
(a7_update1), and the only levanter-vs-levanter sample at update 1 is checks.json train_step_vs_ref_step of each fixture.
"""
import argparse
import json
import math
import os

import numpy as np

from tools.levref import common

KINDS = (("grad", "g0"), ("update0", "u0"), ("state1", "s1"))


def rel(a, b) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    num = float(np.sqrt(((a64 - b64) ** 2).sum()))
    den = float(np.sqrt((b64 ** 2).sum()))
    if num == 0.0:
        return 0.0
    return num / den if den > 0 else math.inf


def cos(a, b) -> float:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    return float(a64 @ b64 / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)


def load_index(d, sub):
    p = os.path.join(d, sub, "index.json")
    return json.load(open(p)) if os.path.exists(p) else None


def arr(d, sub, name):
    return np.load(os.path.join(d, sub, f"{name}.npy"))


def consistency(A: dict, R: dict, have_rep: dict) -> tuple[dict, list]:
    """(facts per fixture, problems): see `consistency` in the module docstring."""
    facts = {}
    for key, d in [*((f"step0_{p}", A[p]) for p in A), *((f"rep_{p}", R[p]) for p in R if have_rep[p])]:
        C = json.load(open(os.path.join(d, "checks.json")))
        M = json.load(open(os.path.join(d, "manifest.json")))
        facts[key] = dict(init=C["init"]["combined_sha256"], batch0=C["batches"]["0"]["sha256"], debug_rows=C.get("debug_rows"),
                          devices=M["environment"]["devices"], precision=(M.get("matmul_precision") or {}).get("main_chain"),
                          ce=(M.get("ce_implementation") or {}).get("selected"))
    problems = []
    for k in ("init", "batch0", "debug_rows"):
        vals = {key: f[k] for key, f in facts.items()}
        if len({json.dumps(v) for v in vals.values()}) > 1:
            problems.append(f"{k} differs between fixtures: {vals}")
    for p in R:
        if have_rep[p]:
            a, r = facts[f"step0_{p}"], facts[f"rep_{p}"]
            for k in ("devices", "precision", "ce"):
                if a[k] != r[k]:
                    problems.append(f"rep_{p} vs step0_{p}: {k} {r[k]} != {a[k]}")
    return facts, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size")
    ap.add_argument("--suffix", default="", help="fixture-name suffix, e.g. _debug4 for the CPU debug fixtures")
    args = ap.parse_args()
    base = os.path.join(common.nano_dir(), "cache", "levref", args.size)
    A = {p: os.path.join(base, f"step0_{p}{args.suffix}") for p in ("bf16", "f32")}
    R = {p: os.path.join(base, f"rep_{p}{args.suffix}") for p in ("bf16", "f32")}
    for p, d in A.items():
        if not os.path.exists(os.path.join(d, "manifest.json")):
            raise SystemExit(f"missing fixture {d}")
    have_rep = {p: os.path.exists(os.path.join(R[p], "manifest.json")) for p in R}
    facts, problems = consistency(A, R, have_rep)
    if problems:
        raise SystemExit("floor.py: the fixtures are not comparable:\n  " + "\n  ".join(problems))

    floors, losses, gn = {}, {}, {}
    for p in ("bf16", "f32"):
        inproc = json.load(open(os.path.join(A[p], "rep_inproc.json")))
        for kind, sub in KINDS:
            idx = load_index(A[p], sub)
            ridx = load_index(R[p], sub) if have_rep[p] else None
            fl = {}
            for name, e in idx.items():
                ip = inproc[sub][name]["rel"]
                xp = None
                if ridx is not None and e["npy"]:
                    xp = 0.0 if ridx[name]["sha256"] == e["sha256"] else rel(arr(R[p], sub, name), arr(A[p], sub, name))
                elif ridx is not None and "value" in e:
                    a, b = float(ridx[name]["value"]), float(e["value"])
                    xp = 0.0 if a == b else (abs(a - b) / abs(b) if b else math.inf)
                parts = [v for v in (ip, xp) if v is not None]
                fl[name] = dict(inproc=ip, xproc=xp, e_floor=max(parts))
            floors[f"{kind}_{p}"] = fl
        L = json.load(open(os.path.join(A[p], "losses.json")))
        l0 = [x["value"] for x in L["L0_ref"]]
        rec = dict(L0_ref=L["L0_ref"], inproc_abs=abs(l0[1] - l0[0]) if len(l0) > 1 else None)
        G = json.load(open(os.path.join(A[p], "group_norms.json")))
        norms = {g: [r[g]["norm"] for r in G["g0"]] for g in ("muonh", "adamh", "adam", "total")}
        if have_rep[p]:
            Lr = json.load(open(os.path.join(R[p], "losses.json")))
            rec["rep_L0"] = Lr["L0_ref"]
            rec["xproc_abs"] = abs(Lr["L0_ref"][0]["value"] - l0[0])
            Gr = json.load(open(os.path.join(R[p], "group_norms.json")))
            for g in norms:
                norms[g] += [r[g]["norm"] for r in Gr["g0"]]
        losses[p] = rec
        gn[p] = {g: dict(values=v, max_rel_spread=(max(v) - min(v)) / min(v) if min(v) > 0 else None) for g, v in norms.items()}

    bvf, tol = {}, {}
    for name, e in load_index(A["f32"], "g0").items():
        gb, gf = arr(A["bf16"], "g0", name), arr(A["f32"], "g0", name)
        ub, uf = arr(A["bf16"], "u0", name), arr(A["f32"], "u0", name)
        u1b, u1f = arr(A["bf16"], "u1", name), arr(A["f32"], "u1", name)
        r = dict(grad_rel=rel(gb, gf), grad_cos=cos(gb, gf), update0_rel=rel(ub, uf), update1_rel=rel(u1b, u1f))
        bvf[name] = r
        tol[name] = dict(a4=max(1e-5, 2 * floors["grad_f32"][name]["e_floor"]), a5_rel=2 * r["grad_rel"], a5_cos=0.9999,
                         a7_update0=2 * r["update0_rel"], a7_update1=2 * r["update1_rel"])
    tol_state = {p: {n: max(1e-5, 2 * v["e_floor"]) for n, v in floors[f"state1_{p}"].items()} for p in ("bf16", "f32")}

    out = common.fixture_dir(args.size, "floor" + args.suffix)
    res = dict(size=args.size, sources={k: os.path.relpath(v, base) for k, v in {**{f"step0_{p}": A[p] for p in A},
                                                                                 **{f"rep_{p}": R[p] for p in R if have_rep[p]}}.items()},
               definitions=__doc__.split("e_floor.json", 1)[1].strip(), tol=tol, tol_state=tol_state, bf16_vs_f32=bvf,
               floors=floors, losses=losses, group_norms=gn, consistency=facts)
    common.write_json(os.path.join(out, "e_floor.json"), res)
    src_manifests = {}
    for d in [*A.values(), *[R[p] for p in R if have_rep[p]]]:
        src_manifests[os.path.basename(d)] = common.sha256_file(os.path.join(d, "manifest.json"))
    common.write_manifest(out, checks=["T2.4 tolerance (a4)", "T2.5 tolerance (a5)", "T2.7 tolerances (a7 update 0, update 1, state)",
                                       "T2.1/T2.2 levanter loss floor"],
                          env=dict(note="CPU post-processing of the step fixtures; numpy only"),
                          extra=dict(size=args.size, source_manifest_sha256=src_manifests, have_rep=have_rep))
    worst = max(floors["grad_f32"].items(), key=lambda kv: kv[1]["e_floor"])
    print(f"{args.size}: grad_f32 e_floor max {worst[1]['e_floor']:.3e} ({worst[0]}); "
          f"L0 floors {json.dumps({p: (losses[p]['inproc_abs'], losses[p].get('xproc_abs')) for p in losses})}")
    print("wrote", out)


if __name__ == "__main__":
    main()
