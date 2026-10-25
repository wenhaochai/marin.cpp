"""T1.7 (kaiyue_baseline.md): one optimizer update of the marincpp engine on a fixed gradient, against levanter's
MuonHConfig.build(N) optax transform (tools/levref/step.py, f32 policy, H100). Cases:
  A  init params (p0), zero state, G0, k = 0 (130m: full lr; 300m: lr 0)
  B  case A's params and state (p1, s1), G0, k = 1
  C  130m only: params and state of the plain step-4958 checkpoint, G0, k = 4958 (inputs exported from the
     checkpoint by tests/s6/export_case_c.py into cache/s6/caseC_in; their sha256 equal caseC/pin)
Bounds (T1.7 row, with BUILD_LOG "Decisions for S4-S6" item 15):
  - the three pre-clip group norms and clip factors: rel <= 1e-6, and the same clip decision
  - MuonH buffer, Adam and AdamH mu, nu: rel <= 1e-6 per tensor; the count (k + 1 after the step) exact
  - Adam updates: rel <= 1e-6 (a stray weight decay breaks it)
  - MuonH and AdamH updates (target - p), TF32 NS (T1.0: levanter ran TF32), item 15: oracle rule with levanter at
    HIGHEST as the oracle, rel(nano, HIGHEST) <= 1.25 x rel(levanter DEFAULT, HIGHEST); where levanter's DEFAULT
    update equals its HIGHEST one (AdamH: no dot product): rel <= 1e-5, the f32-dot bound of the row
  - per-layer Frobenius norm of every MuonH weight and the whole-matrix norm of lm_head preserved to <= 1e-6
  - lr 0 (300m case A): parameters after the update within 1 f32 ulp of the parameters before, per element (a7 (ii));
    the update itself is rounding noise there and is not compared
  - the f32 policy (true-f32 NS): MuonH and AdamH updates rel <= 1e-5 against HIGHEST; two TF32 runs bitwise equal
  - the symmetric NS blocks (EngOptions::ns_symmetric, TF32): MuonH updates to the same rule as the full products
f64 oracle (kaiyue_baseline.md "Conventions": the whole update chain in f64 from the same f32 inputs and constants):
at a small lr the update target - p is ~lr x |p| while the f32 renormalization (w / ||w||) * n leaves ~1 f32 ulp of p
per element, so the update carries ~1e-2 relative rounding noise in any f32 implementation; levanter's DEFAULT and
HIGHEST programs share that noise (the same reduction order), marincpp has its own. A MuonH / AdamH update that misses
item 15 but holds the spec's oracle rule against the f64 oracle, rel(nano, f64) <= 1.25 x rel(levanter, f64) (TF32:
levanter DEFAULT; f32 policy: levanter HIGHEST), is a SPECFAIL (BUILD_LOG S6 request 7 proposes that rule; until the
owner decides, item 15 binds and the exit code is 1), reported with both numbers; missing both is FAIL. Every case
whose fixture exists is mandatory (cases B and C carry the power against momentum-form bugs that case A lacks).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s6lib import (PREC, Lib, d2h, OptGroup, OptTensor, Runner, args, check, index, ptr, rel, retry_oom,  # noqa
                   schedule_tables, stream, torch, ulp_diff_max)

import ctypes as C  # noqa: E402
import numpy as np  # noqa: E402

SIZES = {"130m": dict(N=4959, momentum=0.95, eps=1e-20, mgn=1.0),
         "300m": dict(N=11444, momentum=0.98, eps=1e-15, mgn=1.0)}
GROUPS = ("muonh", "adamh", "adam")
OUT = {}


def group_of(name):
    if name == "lm_head.weight":
        return "adamh"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
        return "muonh"
    return "adam"


def layout(name, shape):
    """(rows, cols, n_mats, mat_rows): the 2-D storage view and the MuonH layer matrices [OUT, IN] (optim.md §2.2)."""
    if group_of(name) == "muonh":
        L = shape[0]
        cols = shape[-2] * shape[-1] if "o_proj" in name else shape[-1]   # o_proj: In = (heads, head_size)
        mat_rows = int(np.prod(shape[1:])) // cols
        return L * mat_rows, cols, L, mat_rows
    cols = shape[-1]
    return int(np.prod(shape[:-1])) if len(shape) > 1 else 1, cols, 0, 0


STATE_KEYS = {"muonh": [("momentum_buffer", "s0")], "adamh": [("mu", "s0"), ("nu", "s1")],
              "adam": [("mu", "s0"), ("nu", "s1")]}


def state_name(grp, key, name):
    return f"{grp}.{key}.{name}"


def load_npy_dir(d, names):
    return {n: np.load(os.path.join(d, n + ".npy")) for n in names}


def params_after(p_dir, p_in, u):
    """Levanter's parameters after an update: p_dir (p1/, caseB/p, caseC/p) stores only their sha256, and the fixture
    checks p = f32(p_in + u) bitwise (checks.json); rebuilt here and verified against every stored sha256."""
    idx = index(p_dir)
    out = {}
    for n in p_in:
        x = (p_in[n].astype(np.float32) + u[n].astype(np.float32)).astype(np.float32)
        check(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() == idx[n]["sha256"],
              f"{p_dir}: {n} is not f32(p_in + u)")
        out[n] = x
    return out


class Run:
    """One engine step over every tensor of a size."""

    def __init__(self, lib, size, params, state, grads, k, prec, tables):
        cfg = SIZES[size]
        self.names = list(params)
        self.dev = {}
        tens = []
        for i, n in enumerate(self.names):
            grp = group_of(n)
            rows, cols, n_mats, mat_rows = layout(n, params[n].shape)
            d = dict(p=torch.from_numpy(params[n]).cuda().reshape(rows, cols).contiguous(),
                     g=torch.from_numpy(grads[n]).cuda().reshape(rows, cols).contiguous(),
                     u=torch.zeros(rows, cols, device="cuda"))
            for key, slot in STATE_KEYS[grp]:
                sn = state_name(grp, key, n)
                d[slot] = (torch.from_numpy(state[sn]).cuda().reshape(rows, cols).contiguous() if state is not None
                           else torch.zeros(rows, cols, device="cuda"))
            self.dev[n] = d
            t = OptTensor()
            t.group = GROUPS.index(grp)
            t.rows, t.cols, t.base_row = rows, cols, 0
            t.p32, t.p16, t.g = ptr(d["p"]), None, ptr(d["g"])
            t.s0, t.s1 = ptr(d["s0"]), ptr(d.get("s1"))
            t.grad_scale = t.update_scale = 1.0
            t.n_mats, t.mat_rows, t.row_map = n_mats, mat_rows, None
            t.n_owned, t.owned = 0, None
            t.u_out = ptr(d["u"])
            tens.append(t)
        gs = (OptGroup * 3)()
        for i, grp in enumerate(GROUPS):
            gs[i].rule, gs[i].lr_table, gs[i].clip_group = i, 1 if grp == "adam" else 0, i
            gs[i].lr_scale, gs[i].b1, gs[i].b2, gs[i].eps = 1.0, 0.9, 0.98, cfg["eps"]
            gs[i].momentum, gs[i].max_grad_norm, gs[i].weight_decay = cfg["momentum"], cfg["mgn"], 0.0
        ta = (OptTensor * len(tens))(*tens)
        self.tab = [torch.from_numpy(x).cuda() for x in tables]
        self.step = torch.tensor([k], dtype=torch.int32, device="cuda")
        eng = C.c_void_p()
        lib.nm_opt_create(gs, 3, ta, len(tens), PREC[prec], 5, 1e-5, 0, 0, 0, C.byref(eng))
        try:
            lib.nm_opt_set_tables(eng, *[ptr(x) for x in self.tab], ptr(self.step), len(tables[0]))
            lib.nm_opt_prepare(eng, stream())
            lib.nm_opt_step(eng, 1, stream())
            cs, ncl, cst = C.c_void_p(), C.c_int(), C.c_void_p()
            nmat, nch = C.c_int(), C.c_int()
            lib.nm_opt_buffers(eng, C.byref(cs), C.byref(ncl), None, None, None, C.byref(cst), C.byref(nmat),
                               C.byref(nch))
            torch.cuda.synchronize()
            sums = d2h(cs.value, 8 * ncl.value).view(np.float64)
            states = d2h(cst.value, 16 * ncl.value).view(np.float32)
            self.n_mats, self.n_chunks = nmat.value, nch.value
        finally:
            lib.nm_opt_destroy(eng)
        st = states.view(np.int32).reshape(-1, 4)
        stf = states.reshape(-1, 4)
        self.norms = {grp: dict(sumsq=float(sums[i]), gn=float(stf[i, 0]), clip=bool(st[i, 1]), factor=float(stf[i, 3]))
                      for i, grp in enumerate(GROUPS)}
        self.k_after = int(self.step.item())
        # results to the host, device memory released (three runs per case on a shared 40 GB GPU)
        self.host = {n: {w: t.cpu().numpy() for w, t in d.items() if w != "g"} for n, d in self.dev.items()}
        del self.dev, self.tab
        torch.cuda.empty_cache()

    def get(self, n, what, shape):
        return self.host[n][what].reshape(shape)


def f32c(x):
    return float(np.float32(x))


def oracle_updates(size, names, shapes, p_in, s_in, grads, k, tables):
    """MuonH and AdamH updates of the f64 oracle (torch f64 on the GPU): the clip, momentum, Newton-Schulz (5 simple
    steps), moments and norm-preserving projections of optim.md §3-§5 in f64, from the f32 inputs and levanter's f32
    constants (optim.md §5.3) upcast."""
    cfg = SIZES[size]
    dd = torch.float64
    lr, bc1, bc2 = float(tables[0][k]), float(tables[2][k]), float(tables[3][k])
    G = {n: torch.from_numpy(grads[n]).cuda().to(dd) for n in names}
    scale = {}
    for grp in GROUPS:
        gn = float(sum((G[n] * G[n]).sum().item() for n in names if group_of(n) == grp)) ** 0.5
        scale[grp] = cfg["mgn"] / gn if not gn < cfg["mgn"] else 1.0
    mu_c, b1, b2 = f32c(cfg["momentum"]), f32c(0.9), f32c(0.98)
    omb1, omb2, eps = f32c(1 - 0.9), f32c(1 - 0.98), f32c(cfg["eps"])
    a_, b_, c_ = f32c(3.4445), f32c(-4.7750), f32c(2.0315)
    floor, meps = f32c(1e-10), f32c(1e-5)
    out = {}
    for n in names:
        grp = group_of(n)
        if grp == "adam":
            continue
        g = G[n] * scale[grp]
        p = torch.from_numpy(p_in[n]).cuda().to(dd)
        if grp == "muonh":
            rows, cols, L, mr = layout(n, shapes[n])
            buf = (torch.from_numpy(s_in[state_name(grp, "momentum_buffer", n)]).cuda().to(dd) if s_in is not None
                   else torch.zeros_like(g))
            buf = mu_c * buf + g
            u = (mu_c * buf + g).reshape(L, mr, cols)
            X = u / (torch.linalg.vector_norm(u, dim=(1, 2), keepdim=True) + meps)
            tr = mr > cols
            if tr:
                X = X.transpose(1, 2)
            for _ in range(5):
                A = X @ X.transpose(1, 2)
                B = b_ * A + (c_ * A) @ A
                X = a_ * X + B @ X
            O = X.transpose(1, 2) if tr else X
            P = p.reshape(L, mr, cols)
            nrm = torch.linalg.vector_norm(P, dim=(1, 2), keepdim=True)
            m = torch.clamp(torch.linalg.vector_norm(O, dim=(1, 2), keepdim=True), min=floor)
            W = P - lr * O * nrm / m
            nw = torch.clamp(torch.linalg.vector_norm(W, dim=(1, 2), keepdim=True), min=floor)
            out[n] = (W / nw * nrm - P).reshape(shapes[n]).cpu().numpy()
        else:
            mu0 = torch.zeros_like(g) if s_in is None else torch.from_numpy(s_in[state_name(grp, "mu", n)]).cuda().to(dd)
            nu0 = torch.zeros_like(g) if s_in is None else torch.from_numpy(s_in[state_name(grp, "nu", n)]).cuda().to(dd)
            mu = omb1 * g + b1 * mu0
            nu = omb2 * g * g + b2 * nu0
            del mu0, nu0
            a = (mu / bc1) / (torch.sqrt(nu / bc2) + eps)
            del mu, nu
            nrm = torch.linalg.vector_norm(p)
            m = max(torch.linalg.vector_norm(a).item(), floor)
            W = p - lr * a * nrm / m
            out[n] = (W / torch.linalg.vector_norm(W) * nrm - p).cpu().numpy()
        del g, p
        torch.cuda.empty_cache()
    del G
    torch.cuda.empty_cache()
    return out


def default_updates(case_dir_default, base_dir):
    """{name: u} of levanter at DEFAULT precision (case*_default/u holds npy only where it differs from HIGHEST)."""
    idx = index(os.path.join(case_dir_default, "u"))
    out = {}
    for n, e in idx.items():
        if e.get("same_as_base", False) or not e.get("npy"):
            out[n] = np.load(os.path.join(base_dir, n + ".npy"))
        else:
            out[n] = np.load(os.path.join(case_dir_default, "u", n + ".npy"))
    return out


def run_case(lib, src, size, case, tables):
    fx = os.path.join(src, "cache", "levref", size, "step0_f32")
    if not os.path.isdir(fx):
        return "skip"
    names = [n for n, e in index(os.path.join(fx, "p0")).items() if isinstance(e, dict) and "shape" in e]
    grads = load_npy_dir(os.path.join(fx, "g0"), names)
    gnorms = json.load(open(os.path.join(fx, "group_norms.json")))
    if case == "A":
        k, p_in, s_in = 0, load_npy_dir(os.path.join(fx, "p0"), names), None
        ref_u, ref_s, ref_p = os.path.join(fx, "u0"), os.path.join(fx, "s1"), os.path.join(fx, "p1")
        ref_norms, dflt = gnorms["g0"][0], os.path.join(fx, "caseA_default")
    elif case == "B":
        k = 1
        p_in = params_after(os.path.join(fx, "p1"), load_npy_dir(os.path.join(fx, "p0"), names),
                            load_npy_dir(os.path.join(fx, "u0"), names))
        s_in = {n: np.load(os.path.join(fx, "s1", n + ".npy")) for n in index(os.path.join(fx, "s1"))
                if os.path.exists(os.path.join(fx, "s1", n + ".npy"))}
        ref_u, ref_s, ref_p = (os.path.join(fx, "caseB", x) for x in ("u", "s", "p"))
        ref_norms, dflt = gnorms["caseB"], os.path.join(fx, "caseB_default")
    else:
        cin = os.path.join(src, "cache", "s6", "caseC_in")
        check(os.path.isdir(os.path.join(fx, "caseC")), f"{fx}/caseC missing (case C is mandatory)")
        check(os.path.isdir(cin), "case C inputs missing: run tests/s6/export_case_c.py (marin venv) first")
        cidx = json.load(open(os.path.join(cin, "index.json")))
        check(all(v["matches_fixture"] for v in cidx["pin"].values()), "case C inputs do not match caseC/pin")
        k = 4958
        p_in = load_npy_dir(os.path.join(cin, "pin"), names)
        s_in = {n: np.load(os.path.join(cin, "sin", n + ".npy")) for n in cidx["sin"]}
        ref_u, ref_s, ref_p = (os.path.join(fx, "caseC", x) for x in ("u", "s", "p"))
        ref_norms, dflt = gnorms["caseC"], os.path.join(fx, "caseC_default")
    shapes = {n: p_in[n].shape for n in names}
    lev_u = load_npy_dir(ref_u, names)
    lev_d = default_updates(dflt, ref_u)
    lev_p = params_after(ref_p, p_in, lev_u)
    sidx = index(ref_s)
    lev_s = {n: np.load(os.path.join(ref_s, n + ".npy")) for n, e in sidx.items() if isinstance(e, dict) and e.get("npy")}
    lev_count = int(sidx["count"]["value"]) if "count" in sidx else None

    res = dict(k=k, lr=float(tables[0][k]), adam_lr=float(tables[1][k]))
    bad = []
    run = retry_oom(Run, lib, size, p_in, s_in, grads, k, "tf32", tables)
    res["n_mats"], res["n_chunks"] = run.n_mats, run.n_chunks
    # 1. group norms and clip
    for grp in GROUPS:
        ln = ref_norms[grp]
        r_n = abs(run.norms[grp]["gn"] - ln["norm"]) / ln["norm"]
        r_f = abs(run.norms[grp]["factor"] - ln["clip_factor"]) / ln["clip_factor"]
        res[f"norm_{grp}"] = dict(nano=run.norms[grp]["gn"], lev=ln["norm"], rel=r_n, factor_rel=r_f,
                                  clip=run.norms[grp]["clip"])
        if not (r_n <= 1e-6 and r_f <= 1e-6 and run.norms[grp]["clip"] == ln["clip_triggered"]):
            bad.append(f"{grp} norm rel {r_n:.2e} factor rel {r_f:.2e} clip {run.norms[grp]['clip']}")
    # 2. state and count
    worst_state = 0.0
    for n in names:
        grp = group_of(n)
        for key, slot in STATE_KEYS[grp]:
            sn = state_name(grp, key, n)
            e = rel(run.get(n, slot, shapes[n]), lev_s[sn])
            worst_state = max(worst_state, e)
            if not e <= 1e-6:
                bad.append(f"state {sn} rel {e:.2e}")
    res["state_rel_max"] = worst_state
    if lev_count is not None and run.k_after != lev_count:
        bad.append(f"count {run.k_after} != {lev_count}")
    res["count"] = run.k_after
    # 3-5. updates, norm preservation, params; then the f32 policy (true-f32 NS) on the same inputs
    run32 = retry_oom(Run, lib, size, p_in, s_in, grads, k, "f32", tables)
    orac = retry_oom(oracle_updates, size, names, shapes, p_in, s_in, grads, k, tables) if res["lr"] > 0 else {}
    rows = {}
    specfail = []
    for n in names:
        grp = group_of(n)
        u = run.get(n, "u", shapes[n])
        u32 = run32.get(n, "u", shapes[n])
        p_new = run.get(n, "p", shapes[n])
        r_hi = rel(u, lev_u[n])
        r_d = rel(lev_d[n], lev_u[n])
        row = dict(group=grp, rel_hi=r_hi, lev_default_rel=r_d, rel_vs_default=rel(u, lev_d[n]),
                   p_rel=rel(p_new, lev_p[n]), f32_rel_hi=rel(u32, lev_u[n]))
        if grp == "adam":
            if not r_hi <= 1e-6:
                bad.append(f"Adam update {n} rel {r_hi:.2e}")
            if not row["f32_rel_hi"] <= 1e-6:
                bad.append(f"f32 policy Adam update {n} rel {row['f32_rel_hi']:.2e}")
        elif res["lr"] > 0:
            ok15 = r_hi <= 1.25 * r_d if r_d > 0 else r_hi <= 1e-5
            ok15_32 = row["f32_rel_hi"] <= 1e-5
            if r_d > 0:
                row["ratio"] = r_hi / r_d
            o = orac[n]
            row.update(f64_nano=rel(u, o), f64_lev_default=rel(lev_d[n], o), f64_lev_highest=rel(lev_u[n], o),
                       f64_nano_f32=rel(u32, o))
            row["f64_ratio"] = row["f64_nano"] / row["f64_lev_default"]
            row["f64_ratio_f32"] = row["f64_nano_f32"] / row["f64_lev_highest"]
            ok64 = row["f64_ratio"] <= 1.25
            ok64_32 = row["f64_ratio_f32"] <= 1.25
            for tag, a15, a64, msg in (("TF32", ok15, ok64, f"rel to HIGHEST {r_hi:.2e} vs levanter DEFAULT "
                                        f"{r_d:.2e}; f64 oracle: nano {row['f64_nano']:.2e}, levanter DEFAULT "
                                        f"{row['f64_lev_default']:.2e} (x{row['f64_ratio']:.3f})"),
                                       ("f32 policy", ok15_32, ok64_32, f"rel to HIGHEST {row['f32_rel_hi']:.2e} > "
                                        f"1e-5; f64 oracle: nano {row['f64_nano_f32']:.2e}, levanter HIGHEST "
                                        f"{row['f64_lev_highest']:.2e} (x{row['f64_ratio_f32']:.3f})")):
                if a15:
                    continue
                (specfail if a64 else bad).append(f"{tag} update {n}: {msg}")
        if grp in ("muonh", "adamh"):
            a = p_in[n].astype(np.float64)
            b = p_new.astype(np.float64)
            if grp == "muonh":
                a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
                na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            else:
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
            npres = float(np.max(np.abs(nb - na) / na))
            row["norm_change"] = npres
            if not npres <= 1e-6:
                bad.append(f"norm preservation {n} {npres:.2e}")
        if res["lr"] == 0.0:
            ulp = ulp_diff_max(p_new, p_in[n])
            row["ulp_vs_p_in"] = ulp
            if ulp > 1:
                bad.append(f"lr 0: {n} moved {ulp} ulp")
        rows[n] = row
    res["tensors"] = rows
    f32_worst = max(r["f32_rel_hi"] for r in rows.values() if r["group"] == "muonh")
    res["f32_policy_muonh_rel_max"] = f32_worst
    # symmetric NS blocks (EngOptions::ns_symmetric): the same item 15 / f64 oracle verdicts as the full products
    runs = retry_oom(Run, lib, size, p_in, s_in, grads, k, "tf32s", tables)
    for n in names:
        if group_of(n) != "muonh" or res["lr"] == 0:
            continue
        us_ = runs.get(n, "u", shapes[n])
        r_hi, r_d = rel(us_, lev_u[n]), rows[n]["lev_default_rel"]
        rows[n]["sym_rel_hi"] = r_hi
        rows[n]["sym_f64_ratio"] = rel(us_, orac[n]) / rows[n]["f64_lev_default"]
        if not r_hi <= 1.25 * r_d:
            (specfail if rows[n]["sym_f64_ratio"] <= 1.25 else bad).append(
                f"TF32 symmetric update {n}: rel to HIGHEST {r_hi:.2e} vs levanter DEFAULT {r_d:.2e}; f64 ratio "
                f"{rows[n]['sym_f64_ratio']:.3f}")
    del runs
    run2 = retry_oom(Run, lib, size, p_in, s_in, grads, k, "tf32", tables)
    same = all(np.array_equal(run.host[n][w].view(np.uint32), run2.host[n][w].view(np.uint32))
               for n in names for w in ("p", "u", "s0"))
    res["repeat_bitwise"] = same
    if not same:
        bad.append("two TF32 runs differ")
    OUT[f"{size}/{case}"] = res
    mu = [r for r in rows.values() if r["group"] == "muonh"]
    print(f"  {size} case {case} (k = {k}, lr {res['lr']:.6g}, adam_lr {res['adam_lr']:.6g}): norms rel "
          + ", ".join(f"{g} {res['norm_' + g]['rel']:.1e}" for g in GROUPS)
          + f"; state rel max {worst_state:.1e}; MuonH update ratio max "
          + (f"{max(r.get('ratio', 0) for r in mu):.3f}" if any('ratio' in r for r in mu) else "n/a")
          + f" (rel {min(r['rel_hi'] for r in mu):.2e}..{max(r['rel_hi'] for r in mu):.2e}); AdamH rel "
          + f"{rows['lm_head.weight']['rel_hi']:.1e}; Adam rel max "
          + f"{max(r['rel_hi'] for r in rows.values() if r['group'] == 'adam'):.1e}; norm change max "
          + f"{max(r.get('norm_change', 0) for r in rows.values()):.1e}; f32 policy MuonH {f32_worst:.1e}"
          + (f"; f64 oracle ratio max TF32 {max(r['f64_ratio'] for r in rows.values() if 'f64_ratio' in r):.3f}, f32 "
             f"{max(r['f64_ratio_f32'] for r in rows.values() if 'f64_ratio_f32' in r):.3f}, TF32 symmetric "
             f"{max(r['sym_f64_ratio'] for r in rows.values() if 'sym_f64_ratio' in r):.3f} (item 15 ratio max "
             f"{max(r['sym_rel_hi'] / r['lev_default_rel'] for r in rows.values() if 'sym_rel_hi' in r):.3f})"
             if orac else ""))
    res["specfail"] = specfail
    check(not bad, "; ".join(bad[:12]) + (f" (+{len(bad) - 12})" if len(bad) > 12 else ""))
    if specfail:
        return ("specfail", f"{len(specfail)} updates miss item 15 (binding) but hold the f64 oracle rule of "
                f"request 7: " + "; ".join(specfail[:4]) + (f" (+{len(specfail) - 4})" if len(specfail) > 4 else ""))


def main():
    a = args()
    lib = Lib(a.lib)
    r = Runner("s6_t1_7")
    only = set(a.only.split(",")) if a.only else None
    for size, cases in (("130m", "ABC"), ("300m", "AB")):
        tables = None
        for case in cases:
            name = f"{size}_{case}"
            if only is not None and name not in only:
                continue
            if tables is None:
                tables = schedule_tables(lib, a.src, size, SIZES[size]["N"])
            if tables is None:
                r.case(name, lambda: "skip")
                continue
            r.case(name, run_case, lib, a.src, size, case, tables)
            torch.cuda.empty_cache()
    OUT["device"] = torch.cuda.get_device_name()
    if a.json:
        json.dump(OUT, open(a.json, "w"), indent=1)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
