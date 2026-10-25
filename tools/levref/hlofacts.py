"""T1.0: precision facts read from the optimized HLO of the train step (an XLA --xla_dump_to dump).

    python -m tools.levref.hlofacts <dump_dir> --hidden 512 --head-dim 64 [--step-json step.json] [--libs-json libs.json]
                                    [--out facts.json]

Picks the train-step module (the largest *_after_optimizations.txt whose module name contains "train_step") and
reports, as data plus a one-line reading each (kaiyue_baseline.md T1.0, numerics.md §10 O1/O2):
  1 collectives      every all-reduce / reduce-scatter / all-gather / all-to-all / collective-permute (also async
                     starts): opcode, element dtypes, reducer; totals by (opcode, dtype)
  2 ce_gemms         every GEMM (dot, also inside Triton GEMM fusions, or a cuBLAS custom call) linked to the lm_head:
                     an operand whose producer chain (<= 6 hops, through fusion parameters and tuple elements) carries a
                     vocab-sized dimension (vocab .. 1.05 x vocab, i.e. padded vocab too), and whose op_name or source
                     frame is the fused-CE code (category cross_entropy). Per GEMM: operand and result
                     dtypes and shapes, contracted size, role (logits: contracts over hidden; grad: contracts over a
                     row or vocab block), forward or backward (transpose(...) in op_name, or a *_bwd / backward source
                     frame), the levanter source frame, the CE implementation file (batched_xla.py = the H100 B-tiled path;
                     xla.py / reference.py = the XLA streaming fallback), and what the first consumers do with a logits
                     result (a convert to bf16 = the logits keep bf16 rounding). A GEMM that is the root of its fusion
                     (Triton) is followed out to the users of that fusion.
  3 scatters         every scatter: result dtype and shape, reducer (param / arithmetic / root dtypes)
  4 gain_reductions  every BACKWARD reduce (op_name has transpose(...)) whose result is [D] or [hd] or a partial stage of
                     one: role (final norm: outside the layer scan; block norm: inside it; q/k norm: [hd]), summand dtype
                     (the dtype before a pure upcast: bf16 = products already rounded to bf16), accumulator, and the dtype
                     in which the value leaves its fusion (the fusion output element that carries the reduce, through
                     convert / bitcast / reshape / copy / transpose only). Forward and optimizer reductions of the same
                     shapes (e.g. per-layer norms [L, D] in the optimizer) are counted under `excluded`, not listed.
  5 f32_dots         every GEMM whose two operands are f32: shapes, algorithm and operand precision (dot attributes or
                     cuBLAS backend_config), implementation (cuBLAS call, Triton fusion kind), source category
                     (newton_schulz / flash_attention / cross_entropy / other), plus the runtime readings passed in
                     (cuBLASLt f32 GEMM probe; Newton-Schulz at DEFAULT vs 'tensorfloat32' vs HIGHEST per GEMM emitter)
  6 flags            DebugOptions fields of the module (config.pbtxt) about ftz, fast math, excess precision, TF32,
                     determinism, autotuning, GEMM backends; the non-default options file; the XLA_FLAGS of the process
gemm_table(path_or_text) lists the GEMMs of any optimized HLO module (used by precision.py on compiled probe modules).
HLO text notes: an operand `%x#k` is element k of the tuple-valued `%x`; the module header tables FileNames /
FunctionNames / FileLocations / StackFrames resolve metadata stack_frame_id to source frames.
"""
import argparse
import collections
import glob
import json
import os
import re

COLLECTIVES = {"all-reduce", "all-reduce-start", "reduce-scatter", "all-gather", "all-gather-start", "all-to-all",
               "collective-permute", "collective-permute-start", "ragged-all-to-all"}
_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s*=\s*(?P<type>\(.*?\)|[\w]+\[[^\]]*\](?:\{[^}]*\})?|[\w]+\[\])\s+"
                    r"(?P<op>[\w\-]+)\((?P<args>.*)$")
_COMP = re.compile(r"^(?:ENTRY\s+)?%(?P<name>[\w.\-]+)\s*\(.*\)\s*->\s*.*\{\s*$")
_ARR = re.compile(r"(\w+)\[([\d,]*)\]")
_REF = re.compile(r"%([\w.\-]+)(?:#(\d+))?")
FLAG_PAT = re.compile(r"ftz|fast|tf32|precision|determin|excess|autotune|triton_gemm|cublas|cudnn|denormal|deterministic|"
                      r"enable_triton|gemm|dot", re.I)
# ops that move or retype elements without mixing them: the lm_head keeps its identity through them (a fusion counts
# when every non-scalar instruction inside it is one of these, so an embedding gather or an elementwise product does not)
PASS_THROUGH = {"bitcast", "copy", "convert", "transpose", "reshape", "dynamic-slice", "slice", "pad", "concatenate",
                "get-tuple-element"}
# ops that carry one value unchanged in content (up to its dtype) from a reduce to the fusion output
PASS_VALUE = {"convert", "bitcast", "reshape", "copy", "transpose"}


def _types(t: str):
    """[(dtype, [dims])] of a result type string (tuple or array), in tuple order."""
    return [(d, [int(x) for x in s.split(",") if x]) for d, s in _ARR.findall(t)]


def _fmt(t) -> str:
    return f"{t[0]}{t[1]}" if t else "?"


def parse(path: str | None = None, text: str | None = None):
    """({computation: [instr dict]}, {instr name: (computation, instr)}, header tables) of a file or an HLO text."""
    comps, cur = collections.OrderedDict(), None
    tables, table = collections.defaultdict(dict), None
    lines = text.splitlines() if text is not None else open(path, errors="replace")
    for line in lines:
        if cur is None:
            m = _COMP.match(line)
            if m:
                cur, table = m.group("name"), None
                comps[cur] = []
                continue
            s = line.strip()
            if s in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
                table = s
                continue
            if table and s:
                k, _, v = s.partition(" ")
                if k.isdigit():
                    tables[table][int(k)] = v
                continue
            continue
        if line.startswith("}"):
            cur = None
            continue
        m = _INSTR.match(line)
        if m:
            d = m.groupdict()
            d["line"] = line.strip()
            d["comp"] = cur
            comps[cur].append(d)
    by_name = {}
    for c, ins in comps.items():
        for i in ins:
            by_name[i["name"]] = (c, i)
    return comps, by_name, tables


def _attr(line: str, key: str):
    m = re.search(key + r"=\"?([%\w.\-$]+)", line)
    return m.group(1).lstrip("%") if m else None


def _refs(args: str):
    """Operand references [(name, tuple index or None)] of an instruction's argument list."""
    head = args.split("),")[0]
    return [(n, int(k) if k else None) for n, k in _REF.findall(head)]


def _short(line: str, n: int = 400) -> str:
    line = re.sub(r", metadata=\{[^}]*\}", "", line)
    line = re.sub(r", backend_config=.*$", "", line)
    return line[:n]


class Module:
    def __init__(self, path: str | None = None, text: str | None = None):
        self.path = path
        self.comps, self.by_name, self.tables = parse(path, text)
        self.caller = {}  # called computation -> (caller computation, caller instr)
        self.root = {}  # computation -> its ROOT instruction
        for c, ins in self.comps.items():
            for i in ins:
                if i["line"].startswith("ROOT"):
                    self.root[c] = i
                for key in ("calls", "body", "condition", "to_apply"):
                    t = _attr(i["line"], key)
                    if t:
                        self.caller[t] = (c, i)
        self._simple = {}
        self.users = collections.defaultdict(list)  # name -> [(instr, tuple index)]
        for c, ins in self.comps.items():
            for i in ins:
                for n, k in _refs(i["args"]):
                    self.users[n].append((i, k))

    def rtype(self, name, k=None):
        """(dtype, dims) of instruction `name`, or of its tuple element k."""
        if name not in self.by_name:
            return None
        ts = _types(self.by_name[name][1]["type"])
        if not ts:
            return None
        if k is None:
            return ts[0]
        return ts[k] if k < len(ts) else None

    def operand_types(self, ins):
        return [self.rtype(n, k) for n, k in _refs(ins["args"])]

    def gemm_result(self, ins):
        t = _types(ins["type"])
        return t[0] if t else None

    def frames(self, line: str, depth: int = 12):
        """Source frames (innermost first) of an instruction's stack_frame_id: 'path:line function'."""
        m = re.search(r"stack_frame_id=(\d+)", line)
        if not m:
            return []
        out, fid, seen = [], int(m.group(1)), set()
        T = self.tables
        while fid and fid not in seen and len(out) < depth:
            seen.add(fid)
            fr = T["StackFrames"].get(fid)
            if not fr:
                break
            loc = int(re.search(r"file_location_id=(\d+)", fr).group(1))
            par = int(re.search(r"parent_frame_id=(\d+)", fr).group(1))
            L = T["FileLocations"].get(loc, "")
            fn = re.search(r"file_name_id=(\d+)", L)
            fu = re.search(r"function_name_id=(\d+)", L)
            ln = re.search(r"\bline=(\d+)", L)
            fname = T["FileNames"].get(int(fn.group(1)), "?").strip('"') if fn else "?"
            func = T["FunctionNames"].get(int(fu.group(1)), "?").strip('"') if fu else "?"
            for cut in ("/src/", "site-packages/"):
                if cut in fname:
                    fname = fname.split(cut, 1)[1]
                    break
            out.append(f"{fname}:{ln.group(1) if ln else '?'} {func}")
            if par == fid:
                break
            fid = par
        return out

    def op_name(self, line: str):
        m = re.search(r'op_name="([^"]*)"', line)
        return m.group(1) if m else None

    def simple_fusion(self, called) -> bool:
        if called not in self._simple:
            ok = called in self.comps
            for i in self.comps.get(called, []):
                t = _types(i["type"])
                scalar = bool(t) and all(not dims for _, dims in t)
                if not scalar and i["op"] not in PASS_THROUGH | {"parameter", "constant"}:
                    ok = False
                    break
            self._simple[called] = ok
        return self._simple[called]

    def ancestors(self, ins, hops: int) -> set:
        """Names of every instruction within `hops` producer steps of `ins` (any op; fusion parameters map to the
        fusion's operands, and a fusion is entered through its root's operands)."""
        seen, todo = set(), [(n, 1) for n, _ in _refs(ins["args"])]
        while todo:
            n, h = todo.pop()
            if n in seen or n not in self.by_name:
                continue
            seen.add(n)
            if h >= hops:
                continue
            c, i = self.by_name[n]
            if i["op"] == "parameter":
                cal = self.caller.get(c)
                if cal and cal[1]["op"] == "fusion":
                    idx = int(re.search(r"parameter\((\d+)\)", i["line"]).group(1))
                    refs = _refs(cal[1]["args"])
                    if idx < len(refs):
                        todo.append((refs[idx][0], h + 1))
                continue
            todo += [(rn, h + 1) for rn, _ in _refs(i["args"])]
        return seen

    def producers(self, name, k, hops: int):
        """Types seen walking back from operand (name, k) through pass-through ops and fusion parameters."""
        seen, out, todo = set(), [], [(name, k, 0)]
        while todo:
            n, kk, h = todo.pop()
            if (n, kk) in seen or n not in self.by_name:
                continue
            seen.add((n, kk))
            t = self.rtype(n, kk)
            if t:
                out.append(t)
            if h >= hops:
                continue
            c, ins = self.by_name[n]
            if ins["op"] == "parameter":
                if kk is None and ins["type"].startswith("("):
                    continue  # a whole tuple parameter: only its referenced element counts
                idx = int(re.search(r"parameter\((\d+)\)", ins["line"]).group(1))
                cal = self.caller.get(c)
                if cal and cal[1]["op"] == "fusion":
                    refs = _refs(cal[1]["args"])
                    if idx < len(refs):
                        todo.append((refs[idx][0], refs[idx][1], h + 1))
                continue
            if (ins["op"] in PASS_THROUGH or (ins["op"] == "fusion" and self.simple_fusion(_attr(ins["line"], "calls")))) and kk is None:
                for rn, rk in _refs(ins["args"]):
                    todo.append((rn, rk, h + 1))
        return out

    def value_chain(self, name, k=None, hops: int = 8):
        """dtypes of one value walking back from (name, k) through value-carrying ops only (convert, bitcast, reshape,
        copy, transpose, fusion parameters, tuple elements), nearest first; stops at the first op that computes."""
        out, n, kk = [], name, k
        for _ in range(hops):
            if n not in self.by_name:
                break
            c, ins = self.by_name[n]
            t = self.rtype(n, kk)
            out.append(t[0] if t else "?")
            if ins["op"] == "parameter":
                cal = self.caller.get(c)
                if not (cal and cal[1]["op"] == "fusion"):
                    break
                idx = int(re.search(r"parameter\((\d+)\)", ins["line"]).group(1))
                refs = _refs(cal[1]["args"])
                if idx >= len(refs):
                    break
                n, kk = refs[idx]
                continue
            if ins["op"] == "fusion" and kk is not None:  # a tuple element of a multi-output fusion: its root operand
                root = self.root.get(_attr(ins["line"], "calls"))
                if root is None or root["op"] != "tuple":
                    break
                refs = _refs(root["args"])
                if kk >= len(refs):
                    break
                n, kk = refs[kk]
                continue
            if ins["op"] not in PASS_VALUE or kk is not None:
                break
            refs = _refs(ins["args"])
            if not refs:
                break
            n, kk = refs[0]
        return out

    def leaves_fusion_as(self, ins):
        """(dtype, where) of the value of `ins` where it leaves its computation: the output element of the calling fusion
        that carries it through PASS_VALUE ops only, or (None, 'consumed inside ...') when arithmetic uses it first."""
        c = ins["comp"]
        cal = self.caller.get(c)
        own = _types(ins["type"])
        if not cal or cal[1]["op"] != "fusion":
            return (own[0][0] if own else None), "own buffer (not in a fusion)"
        reach, frontier = {ins["name"]}, [ins["name"]]
        for _ in range(8):
            nxt = []
            for n in frontier:
                for u, _k in self.users.get(n, []):
                    if u["comp"] == c and u["op"] in PASS_VALUE and u["name"] not in reach:
                        reach.add(u["name"])
                        nxt.append(u["name"])
            if not nxt:
                break
            frontier = nxt
        root = self.root.get(c)
        if root is None:
            return None, "no root"
        if root["name"] in reach:
            rt = _types(root["type"])
            return (rt[0][0] if rt else None), f"output of {cal[1]['name']}"
        if root["op"] == "tuple":
            ft = _types(cal[1]["type"])
            for j, (n, _k) in enumerate(_refs(root["args"])):
                if n in reach:
                    return (ft[j][0] if j < len(ft) else None), f"output {j} of {cal[1]['name']}"
        return None, f"consumed inside {cal[1]['name']}"


def _reducer(mod: Module, line: str):
    r = _attr(line, "to_apply")
    if not r or r not in mod.comps:
        return None
    ins = mod.comps[r]
    params = sorted({_types(i["type"])[0][0] for i in ins if i["op"] == "parameter" and _types(i["type"])})
    arith = [(i["op"], _types(i["type"])[0][0]) for i in ins if i["op"] in ("add", "maximum", "minimum", "multiply", "or", "and")
             and _types(i["type"])]
    root = [i for i in ins if i["line"].startswith("ROOT")]
    rt = _types(root[0]["type"])[0][0] if root and _types(root[0]["type"]) else None
    return dict(params=params, arithmetic=sorted({f"{o}:{d}" for o, d in arith}), root=rt,
                text=f"params {'/'.join(params)}, {'/'.join(sorted({f'{o} in {d}' for o, d in arith}))}, root {rt}")


def _is_gemm(ins) -> bool:
    if ins["op"] == "dot":
        return True
    if ins["op"] == "custom-call":
        tgt = _attr(ins["line"], "custom_call_target") or ""
        return bool(re.search(r"gemm|matmul|cublas", tgt, re.I))
    return False


def _is_bwd_frame(frame: str) -> bool:
    """A custom-VJP backward source frame (innermost only: e.g. batched_xla.py _backward_b_tiled_from_lse.<locals>.body);
    the forward of a custom VJP (..._with_custom_backward_fwd) is not one."""
    func = frame.split(" ", 1)[-1].split(".<locals>")[0]
    return func.startswith(("_backward", "backward")) or func.endswith("_bwd")


def _gemm_info(mod: Module, ins) -> dict:
    line = ins["line"]
    ops = mod.operand_types(ins)[:2]
    res = mod.gemm_result(ins)
    if ins["op"] == "dot":
        lc = [int(x) for x in (re.search(r"lhs_contracting_dims=\{([\d,]*)\}", line).group(1).split(",")) if x] \
            if re.search(r"lhs_contracting_dims=\{([\d,]*)\}", line) else []
        alg = _attr(line, "algorithm")
        prec = re.findall(r"operand_precision=\{([^}]*)\}", line)
        prec = prec[0] if prec else "DEFAULT (attribute absent)"
        target = None
    else:
        m = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', line)
        lc = [int(x.strip('"')) for x in m.group(1).split(",") if x.strip('"')] if m else []
        a = re.search(r'"algorithm":"(\w+)"', line)
        alg = a.group(1) if a else None
        p = re.search(r'"operand_precision":\[([^\]]*)\]', line)
        prec = p.group(1).replace('"', "") if p else None
        target = _attr(line, "custom_call_target")
    contracted = None
    if ops and ops[0] and lc:
        contracted = 1
        for d in lc:
            contracted *= ops[0][1][d] if d < len(ops[0][1]) else 1
    cal = mod.caller.get(ins["comp"])
    impl = target
    if cal is not None and cal[1]["op"] == "fusion":
        kinds = re.findall(r'"kind":"([\w$]+)"', cal[1]["line"])
        impl = (kinds[0] if kinds else _attr(cal[1]["line"], "kind")) or "fusion"
    on = mod.op_name(line) or (mod.op_name(cal[1]["line"]) if cal else None)
    fr = mod.frames(line) or (mod.frames(cal[1]["line"]) if cal else [])
    bwd = bool(on and "transpose(" in on) or bool(fr and _is_bwd_frame(fr[0]))
    return dict(name=ins["name"], computation=ins["comp"], op=ins["op"], implementation=impl or "dot (unfused)",
                fusion=cal[1]["name"] if cal and cal[1]["op"] == "fusion" else None,
                operands=[_fmt(t) for t in ops], result=_fmt(res), contracted=contracted, algorithm=alg, operand_precision=prec,
                backward=bwd, op_name=(on or "")[-200:], frames=fr[:6], line=_short(line, 300))


def gemm_table(path_or_text: str) -> list[dict]:
    """Every GEMM of an optimized HLO module (a file path or the text itself, e.g. Compiled.as_text())."""
    mod = Module(text=path_or_text) if "\n" in path_or_text else Module(path=path_or_text)
    out = []
    for c, ins in mod.comps.items():
        for i in ins:
            if _is_gemm(i):
                g = _gemm_info(mod, i)
                out.append({k: g[k] for k in ("name", "implementation", "operands", "result", "algorithm", "operand_precision")})
    return out


def gemm_summary(table: list[dict]) -> dict:
    return dict(n=len(table), implementations=dict(collections.Counter(str(g["implementation"]) for g in table)),
                operand_precision=dict(collections.Counter(str(g["operand_precision"]) for g in table)),
                algorithms=dict(collections.Counter(str(g["algorithm"]) for g in table)),
                operands=dict(collections.Counter("/".join(o.split("[")[0] for o in g["operands"]) for g in table)))


def _category(info) -> str:
    s = " ".join(info.get("frames") or []) + " " + (info.get("op_name") or "")
    if re.search(r"newtonschulz|zeropower|optim/util", s, re.I):
        return "newton_schulz"
    if re.search(r"flash_attention", s):
        return "flash_attention"
    if re.search(r"cross_entropy|loss\.py|fused_cross", s):
        return "cross_entropy"
    if re.search(r"optim/", s):
        return "optimizer_other"
    return "other"


def _ce_file(info):
    """The fused-CE module a GEMM comes from: batched_xla.py (H100 B-tiled path), xla.py / reference.py (XLA fallback).
    The op_name decides first: the B-tiled forward calls reference.py's linear_softmax_cross_entropy_loss_reference per
    tile, so its innermost frame is reference.py although the GEMM runs inside the batched_xla dispatch."""
    if "batched_xla" in (info.get("op_name") or ""):
        return "batched_xla.py"
    for f in info.get("frames") or []:
        m = re.search(r"fused_cross_entropy_loss/(\w+\.py)", f)
        if m:
            return m.group(1)
    return None


def _consumers(mod: Module, name: str, k=None, depth: int = 2):
    """What the first consumers do with result (name, k): [(op, result dtype, via)]. Looks inside consumer fusions, and
    follows a value that is the root (or a root tuple element) of its fusion out to the users of that fusion."""
    out = []
    ent = mod.by_name.get(name)
    if ent is not None and depth > 0:
        c, ins = ent
        cal = mod.caller.get(c)
        if ins["line"].startswith("ROOT") and cal and cal[1]["op"] == "fusion":
            for o, d, v in _consumers(mod, cal[1]["name"], None, depth - 1):
                out.append((o, d, f"after {cal[1]['name']} ({v})"))
    for u, uk in mod.users.get(name, []):
        if k is not None and uk is not None and uk != k:
            continue
        if u["op"] == "get-tuple-element":
            idx = int(re.search(r"index=(\d+)", u["line"]).group(1))
            if k is None or idx == k:
                out += _consumers(mod, u["name"], None, depth)
            continue
        if u["op"] == "tuple" and u["line"].startswith("ROOT"):
            cu = mod.caller.get(u["comp"])
            if cu and cu[1]["op"] == "fusion" and depth > 0:
                for j, (n, _kk) in enumerate(_refs(u["args"])):
                    if n == name:
                        for o, d, v in _consumers(mod, cu[1]["name"], j, depth - 1):
                            out.append((o, d, f"after {cu[1]['name']}#{j} ({v})"))
            continue
        if u["op"] == "fusion":
            called = _attr(u["line"], "calls")
            refs = _refs(u["args"])
            pidx = [i for i, (n, kk) in enumerate(refs) if n == name and (k is None or kk == k or kk is None)]
            for pi in pidx:
                for p in mod.comps.get(called, []):
                    if p["op"] == "parameter" and re.search(rf"parameter\({pi}\)", p["line"]):
                        for v, _ in mod.users.get(p["name"], []):
                            t = _types(v["type"])
                            out.append((v["op"], t[0][0] if t else "?", f"inside {u['name']}"))
                            if v["op"] in ("bitcast", "copy", "reshape", "transpose") and depth > 0:
                                for w, _ in mod.users.get(v["name"], []):
                                    tw = _types(w["type"])
                                    out.append((w["op"], tw[0][0] if tw else "?", f"inside {u['name']} after {v['op']}"))
            continue
        t = _types(u["type"])
        out.append((u["op"], t[0][0] if t else "?", "direct"))
    return out


def pick_module(dump_dir: str, needle: str = "train_step") -> str | None:
    files = [f for f in glob.glob(os.path.join(dump_dir, "module_*after_optimizations.txt")) if needle in os.path.basename(f)]
    return max(files, key=os.path.getsize) if files else None


def _gain_role(result_dims, op_name, hidden, head_dim):
    inscan = "scan(" in op_name or "while/body" in op_name
    if result_dims[-1] == head_dim and head_dim != hidden:
        return "qk_norm"
    return "block_norm (in layer scan)" if inscan else "final_norm"


def _runtime_readings(probe, ns):
    """One-line readings of the runtime measurements, each with its scope (which GEMM emitter it measured)."""
    r = {}
    probes = ({"autotuned": probe} if "shape" in probe else probe) if probe else {}
    for em, p in probes.items():
        impl = (p.get("gemm") or {}).get("implementations")
        fp = p.get("rounding_fingerprint_default")
        best = min(fp, key=fp.get) if fp else None
        r[f"f32_gemm_probe_{em}"] = (f"{p.get('shape')} f32 GEMM via {impl or 'unrecorded emitter'}: default == 'tensorfloat32' bitwise "
                                     f"{p.get('default_bitwise_equal_tensorfloat32')}, default == highest bitwise "
                                     f"{p.get('default_bitwise_equal_highest')}, rel(default, f64) {p.get('rel_default_vs_f64'):.2e}"
                                     + (f", closest input format {best} (rel {fp[best]:.1e}; "
                                        + ", ".join(f"{k} {v:.1e}" for k, v in fp.items()) + ")" if fp else ""))
    if ns:
        per = collections.defaultdict(lambda: collections.Counter())
        impls = collections.defaultdict(collections.Counter)
        for name, v in ns.items():
            for em, e in (v.get("emitters") or {}).items():
                per[em]["matrices"] += 1
                per[em]["default==tensorfloat32"] += bool(e.get("default_bitwise_equal_tensorfloat32"))
                per[em]["default==highest"] += bool(e.get("default_bitwise_equal_highest"))
                for k2, n2 in ((e.get("gemm") or {}).get("default") or {}).get("implementations", {}).items():
                    impls[em][k2] += n2
            if not v.get("emitters"):  # older layout: default vs highest only
                per["autotuned"]["matrices"] += 1
                per["autotuned"]["default==highest"] += bool(v.get("bitwise_equal"))
        r["newton_schulz"] = {em: dict(c, default_gemm_implementations=dict(impls.get(em, {}))) for em, c in per.items()}
    return r


def analyze(dump_dir: str, *, hidden: int, head_dim: int, vocab: int = 128256, libs=None, probe=None, ns=None,
            ce_log=None) -> dict:
    path = pick_module(dump_dir)
    if path is None:
        return dict(error=f"no train_step module in {dump_dir}", modules=sorted(os.listdir(dump_dir))[:50])
    mod = Module(path)
    comps = mod.comps
    base = path[: path.index(".sm_")] if ".sm_" in path else re.sub(r"\.(cpu_)?after_optimizations\.txt$", "", path)
    facts: dict = dict(module_file=os.path.basename(path), bytes=os.path.getsize(path), computations=len(comps),
                       instructions=sum(len(v) for v in comps.values()))

    # 1 collectives
    coll = []
    for c, ins in comps.items():
        for i in ins:
            if i["op"] in COLLECTIVES:
                ts = _types(i["type"])
                coll.append(dict(computation=c, name=i["name"], op=i["op"], n_elements=len(ts),
                                 dtypes=sorted({d for d, _ in ts if d not in ("u32", "s32", "token")}),
                                 shapes=[_fmt(t) for t in ts][:8], reducer=(_reducer(mod, i["line"]) or {}).get("text"),
                                 replica_groups=_attr(i["line"], "replica_groups"), line=_short(i["line"], 240)))
                cal = mod.caller.get(c)
                on = mod.op_name(i["line"]) or (mod.op_name(cal[1]["line"]) if cal else None) or ""
                coll[-1].update(op_name=on[-200:], backward="transpose(" in on)
    tot = collections.Counter()
    for x in coll:
        for d in x["dtypes"] or ["?"]:
            tot[f"{x['op']}:{d}"] += 1
    facts["collectives"] = dict(totals=dict(sorted(tot.items())), ops=coll)

    # 2 + 5: GEMMs
    vmax = int(vocab * 1.05)
    gemms, ce, f32d, gi = [], [], [], {}
    for c, ins in comps.items():
        for i in ins:
            if not _is_gemm(i):
                continue
            info = _gemm_info(mod, i)
            info["category"] = _category(info)
            gemms.append(info)
            gi[i["name"]] = (i, info)
            ops = mod.operand_types(i)[:2]
            if len(ops) == 2 and all(t and t[0] == "f32" for t in ops):
                f32d.append(info)
    # pass 1: GEMMs with an operand that is (a slice / retype of) the vocab-sized lm_head, or a vocab-sized result
    w_linked = {}
    for name, (i, info) in gi.items():
        refs = _refs(i["args"])[:2]
        res = mod.gemm_result(i)
        lk = [j for j, (n, k) in enumerate(refs) if any(any(vocab <= d <= vmax for d in t[1]) for t in mod.producers(n, k, 6))]
        if lk or (res and any(vocab <= d <= vmax for d in res[1])):
            w_linked[name] = lk
    # only GEMMs of the CE source (op_name or frame): on a 4-GPU FSDP step every layer GEMM reaches the all-gathered
    # embedding within 6 hops, so the vocab link alone also picks the q/k/v/o and MLP GEMMs
    w_linked = {n: v for n, v in w_linked.items() if gi[n][1]["category"] == "cross_entropy"}
    logits = {n for n in w_linked if gi[n][1]["contracted"] == hidden}
    # pass 2: GEMMs fed (within 8 hops, any op) by a logits GEMM: the dlogits products of the backward
    fed = {n for n, (i, inf) in gi.items()
           if n not in logits and inf["category"] == "cross_entropy" and mod.ancestors(i, 8) & logits}
    for name, (i, info) in gi.items():
        if name not in w_linked and name not in fed:
            continue
        res = mod.gemm_result(i)
        e = dict(info, w_operand=w_linked.get(name), fed_by_logits=name in fed, ce_file=_ce_file(info))
        if name in logits:
            e["role"] = "logits"
            k0 = 0 if i["op"] == "custom-call" else None
            cons = _consumers(mod, i["name"], k0)
            e["consumers"] = sorted({f"{o}->{d} ({v.split(' ')[0]})" for o, d, v in cons})[:20]
            e["bf16_convert_after"] = any(o == "convert" and d == "bf16" for o, d, _ in cons)
        elif res and hidden in res[1] and (any(vocab <= d <= vmax for d in res[1]) or name in fed and not w_linked.get(name)):
            e["role"] = "grad_w (dlogits^T x)"
        elif name in fed and w_linked.get(name):
            e["role"] = "grad_x (dlogits W)"
        else:
            e["role"] = "other vocab-linked"
        ce.append(e)
    facts["ce_gemms"] = ce
    facts["gemm_totals"] = dict(collections.Counter(f"{g['category']} {'/'.join(g['operands'])}->{g['result'].split('[')[0]} "
                                                    f"{g['implementation']}" for g in gemms))

    # 3 scatters
    sc = []
    for c, ins in comps.items():
        for i in ins:
            if i["op"] == "scatter":
                ts = _types(i["type"])
                cal = mod.caller.get(c)
                sc.append(dict(computation=c, name=i["name"], result=[_fmt(t) for t in ts], reducer=_reducer(mod, i["line"]),
                               fusion=cal[1]["name"] if cal else None, frames=mod.frames(i["line"])[:4], line=_short(i["line"], 240)))
    facts["scatters"] = sc

    # 4 backward reductions to [D] or [hd] (norm-gain gradients and their partial stages)
    red, excluded = [], collections.Counter()
    for c, ins in comps.items():
        for i in ins:
            if i["op"] != "reduce":
                continue
            ts = _types(i["type"])
            if not ts or not ts[0][1] or ts[0][1][-1] not in (hidden, head_dim) or len(ts[0][1]) > 3:
                continue
            in_types = [t for t in mod.operand_types(i) if t][:1]
            n_in, n_out = 1, 1
            for x in (in_types[0][1] if in_types else []):
                n_in *= x
            for x in ts[0][1]:
                n_out *= x
            if ts[0][1] not in ([hidden], [head_dim]) and n_in < 64 * n_out:
                continue
            on = mod.op_name(i["line"]) or ""
            cal = mod.caller.get(c)
            if "transpose(" not in on:  # forward or optimizer (e.g. per-layer norms [L, D]): not a gain gradient
                where_on = re.sub(r"^jit\([^)]*\)/", "", on)[-80:] or "?"
                excluded[f"{_fmt(ts[0])} {where_on}"] += 1
                continue
            rd = _reducer(mod, i["line"]) or {}
            stored, where = mod.leaves_fusion_as(i)
            refs = _refs(i["args"])
            chain = mod.value_chain(*refs[0]) if refs else []
            summands = "bf16 (upcast to f32 before the reduce)" if "bf16" in chain[1:] and chain[:1] == ["f32"] else (chain[0] if chain else "?")
            stage = "final" if ts[0][1] in ([hidden], [head_dim]) else "partial"
            red.append(dict(computation=c, name=i["name"], role=_gain_role(ts[0][1], on, hidden, head_dim), stage=stage,
                            result=_fmt(ts[0]), accumulator=ts[0][0], summands=summands, summand_chain=chain[:6],
                            reducer=rd.get("text"), input=[_fmt(t) for t in in_types],
                            reduced_elements_per_output=n_in // max(n_out, 1), stored=stored, stored_where=where,
                            op_name=on[-160:], frames=mod.frames(i["line"])[:4],
                            fusion=cal[1]["name"] if cal and cal[1]["op"] == "fusion" else None, line=_short(i["line"], 240)))
    acc = collections.Counter(f"{r['role']} {r['stage']} {r['result']} summands={r['summands'].split(' ')[0]} acc={r['accumulator']} "
                              f"leaves_fusion_as={r['stored']}" for r in red)
    facts["gain_reductions"] = dict(totals=dict(acc), ops=red, excluded=dict(excluded),
                                    note="only backward reduces; stored = dtype of the fusion output element that carries the reduce")

    # 5 f32 dots
    def _detail(xs):
        return dict(n=len(xs), implementations=dict(collections.Counter(str(x["implementation"]) for x in xs)),
                    operand_precision=dict(collections.Counter(str(x["operand_precision"]) for x in xs)),
                    algorithms=dict(collections.Counter(str(x["algorithm"]) for x in xs)),
                    shapes=sorted({f"{'/'.join(x['operands'])}->{x['result']}" for x in xs})[:6],
                    fusions=sorted({str(x["fusion"]) for x in xs})[:6])
    by_cat = collections.defaultdict(list)
    for x in f32d:
        by_cat[x["category"]].append(x)
    runtime = _runtime_readings(probe, ns)
    facts["f32_dots"] = dict(n=len(f32d), by_category=dict(collections.Counter(x["category"] for x in f32d)),
                             detail={k: _detail(v) for k, v in sorted(by_cat.items())},
                             algorithms=dict(collections.Counter(str(x["algorithm"]) for x in f32d)),
                             operand_precision=dict(collections.Counter(str(x["operand_precision"]) for x in f32d)),
                             implementations=dict(collections.Counter(str(x["implementation"]) for x in f32d)), ops=f32d,
                             runtime=runtime, runtime_gemm_probe=probe, newton_schulz=ns)

    # 6 flags
    flags = {}
    cfgp = base + ".config.pbtxt"
    if os.path.exists(cfgp):
        for line in open(cfgp):
            line = line.strip()
            m = re.match(r"(xla_[\w]+):\s*(.*)$", line)
            if m and FLAG_PAT.search(m.group(1)):
                flags.setdefault(m.group(1), []).append(m.group(2))
    nondef = open(base + ".debug_options").read() if os.path.exists(base + ".debug_options") else None
    facts["flags"] = dict(config_pbtxt=os.path.basename(cfgp), selected={k: v if len(v) > 1 else v[0] for k, v in flags.items()},
                          non_default_debug_options=nondef, has_xla_gpu_ftz_field="xla_gpu_ftz" in flags,
                          XLA_FLAGS=os.environ.get("XLA_FLAGS"))
    if libs is not None:
        facts["libraries"] = libs

    # one-line readings
    grad_coll = [x for x in coll if x["op"] in ("all-reduce", "all-reduce-start", "reduce-scatter")]
    logit = [x for x in ce if x["role"] == "logits"]

    def _ce_rd(xs):
        return dict(n=len(xs), results=dict(collections.Counter(x["result"].split("[")[0] for x in xs)),
                    operands=dict(collections.Counter("/".join(o.split("[")[0] for o in x["operands"]) for x in xs)),
                    bf16_convert_after=dict(collections.Counter(str(x.get("bf16_convert_after")) for x in xs)),
                    implementations=dict(collections.Counter(str(x["implementation"]) for x in xs)),
                    shapes=sorted({f"{'/'.join(x['operands'])}->{x['result']}" for x in xs})[:6])
    files = collections.Counter(str(x.get("ce_file")) for x in ce)
    ce_path = ("batched_xla (H100 B-tiled path)" if files and set(files) <= {"batched_xla.py"} else
               "XLA streaming fallback (xla.py / reference.py), NOT the H100 batched_xla path: items 2 and 5 do not describe H100"
               if files and set(files) & {"xla.py", "reference.py"} else f"unclassified: {dict(files)}")
    summary = {
        # gradient reductions = reductions of the backward (op_name transpose(...)); the others are optimizer-internal
        # (Newton-Schulz partial Grams on sharded matrices, norms) or the loss
        "1_gradient_reduction_dtypes": dict(collections.Counter(d for x in grad_coll if x.get("backward") for d in x["dtypes"])),
        "1_other_reduction_dtypes": dict(collections.Counter(d for x in grad_coll if not x.get("backward") for d in x["dtypes"])),
        "1_collective_totals": facts["collectives"]["totals"],
        "2_ce_path": dict(reading=ce_path, gemm_source_files=dict(files), implementation_log=ce_log),
        "2_ce_logit_gemms": dict(forward=_ce_rd([x for x in logit if not x["backward"]]),
                                 backward=_ce_rd([x for x in logit if x["backward"]])),
        "2_ce_grad_gemms": _ce_rd([x for x in ce if x["role"] != "logits"]),
        "3_scatter": [dict(result=x["result"], reducer=(x["reducer"] or {}).get("text"),
                           role="CE one-hot (dlogits - onehot)" if "cross_entropy" in " ".join(x.get("frames") or [])
                           else "embedding gradient") for x in sc if any(str(vocab) in r for r in x["result"])],
        "4_gain_reductions": dict(acc),
        "4_excluded_reductions": sum(excluded.values()),
        "5_f32_dots": dict(n=len(f32d), by_category={k: dict(n=v["n"], implementations=v["implementations"],
                                                               operand_precision=v["operand_precision"], algorithms=v["algorithms"],
                                                               shapes=v["shapes"][:3])
                                                     for k, v in facts["f32_dots"]["detail"].items()},
                           runtime=runtime),
        "6_flags": {k: facts["flags"]["selected"].get(k) for k in ("xla_allow_excess_precision", "xla_gpu_enable_fast_min_max",
                                                                   "xla_enable_fast_math", "xla_cpu_ftz", "xla_gpu_ftz",
                                                                   "xla_gpu_deterministic_ops", "xla_gpu_autotune_level",
                                                                   "xla_gpu_enable_triton_gemm", "xla_gpu_enable_cublaslt",
                                                                   "xla_gpu_cublas_fallback")},
    }
    if libs is not None:
        names = sorted({os.path.basename(d.get("realpath", d.get("path", ""))) for d in libs.get("proc_self_maps", [])})
        summary["libraries"] = names
    facts["summary"] = summary
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--step-json", default=None, help="precision.py step.json: adds the runtime GEMM probe and NS readings")
    ap.add_argument("--libs-json", default=None, help="precision.py libs.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = json.load(open(a.step_json)) if a.step_json else {}
    libs = json.load(open(a.libs_json)) if a.libs_json else None
    facts = analyze(a.dump_dir, hidden=a.hidden, head_dim=a.head_dim, vocab=a.vocab, libs=libs,
                    probe=st.get("f32_gemm_probe_by_emitter") or st.get("f32_gemm_probe"),
                    ns=st.get("newton_schulz"), ce_log=st.get("ce_implementation_log"))
    print(json.dumps(facts.get("summary", facts), indent=1))
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(facts, f, indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
