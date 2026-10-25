"""Compare two init fixtures (e.g. 4 vs 1 GPU, H100 vs CPU): per-tensor sha256, and element-level ulp distance when both saved npy.

    python -m tools.levref.compare <size> <name_a> <name_b>

Names are fixture subdirectories of cache/levref/<size>/. Writes cache/levref/<size>/compare/<name_a>_vs_<name_b>.json
(outside both fixtures, so their manifests stay valid).
"""
import argparse
import json
import os

import numpy as np

from tools.levref import common


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|ordinal(a) - ordinal(b)| for float32, with the sign-magnitude bit patterns mapped to a monotone integer line."""
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**31)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**31)) - ib, ib)
    return np.abs(ia - ib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size")
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    base = os.path.join(common.nano_dir(), "cache", "levref", args.size)
    ia = json.load(open(os.path.join(base, args.a, "index.json")))
    ib = json.load(open(os.path.join(base, args.b, "index.json")))
    rows = []
    for name, ta in ia["tensors"].items():
        tb = ib["tensors"][name]
        r = dict(name=name, sha_equal=ta["sha256"] == tb["sha256"], sharding_a=ta.get("sharding"), sharding_b=tb.get("sharding"))
        pa, pb = (os.path.join(base, d, "npy", f"{name}.npy") for d in (args.a, args.b))
        if not r["sha_equal"] and os.path.exists(pa) and os.path.exists(pb):
            xa, xb = np.load(pa), np.load(pb)
            d = ulp_distance(xa, xb)
            r.update(n_diff=int((d > 0).sum()), numel=int(d.size), max_ulp=int(d.max()),
                     max_abs=float(np.abs(xa.astype(np.float64) - xb.astype(np.float64)).max()))
        rows.append(r)
        print(f"{name:55s} equal={r['sha_equal']}" + (f" n_diff={r['n_diff']}/{r['numel']} max_ulp={r['max_ulp']}" if "n_diff" in r else ""))
    res = dict(size=args.size, a=args.a, b=args.b, a_summary=ia["summary"], b_summary=ib["summary"],
               combined_equal=ia["summary"]["combined_sha256"] == ib["summary"]["combined_sha256"], tensors=rows)
    print("combined sha256 equal:", res["combined_equal"])
    out = os.path.join(common.fixture_dir(args.size, "compare"), f"{args.a}_vs_{args.b}.json")
    common.write_json(out, res)
    print("wrote", out)


if __name__ == "__main__":
    main()
