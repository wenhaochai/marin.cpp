"""Export the inputs of T1.7 case C (the plain 130m step-4958 checkpoint: parameters and optimizer state) as .npy.

The levanter fixture (cache/levref/130m/step0_f32/caseC/) stores the outputs of case C and only the sha256 of its input
parameters (caseC/pin/index.json); the inputs live in the levanter checkpoint (zarr3 arrays in an OCDBT kvstore). This
script reads them with tensorstore, read-only, checks every parameter against the sha256 of caseC/pin, and writes

  <out>/pin/<levanter leaf name>.npy               parameters before the update
  <out>/sin/<group>.<state>.<leaf name>.npy        optimizer state before the update (fixture s/ naming)
  <out>/index.json                                 shapes, sha256, the checkpoint's stored step and counts

Run with the marin venv (tensorstore) on a CPU node:
  $NANOMARIN_ROOT/project/marin/.venv/bin/python tests/s6/export_case_c.py --src . \
      [--ckpt $MARIN_PREFIX/speedrun/muonh-qwen3-130m-della4xh100-restore/2026.09.13/checkpoints/step-4958]
Default output: <src>/cache/s6/caseC_in (gitignored).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import tensorstore as ts

CKPT_REL = "speedrun/muonh-qwen3-130m-della4xh100-restore/2026.09.13/checkpoints/step-4958"


def leaf_name(path: str) -> str:
    """model/transformer/layers/stacked/mlp/up_proj/weight -> transformer.layers.mlp.up_proj.weight"""
    return path.replace("layers/stacked/", "layers/").replace("/", ".")


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="marin.cpp repo root")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.environ.get("NANOMARIN_ROOT")
    prefix = os.environ.get("MARIN_PREFIX") or (os.path.join(root, "marin_store_big") if root else None)
    ckpt = args.ckpt or (os.path.join(prefix, CKPT_REL) if prefix else None)
    if not ckpt or not os.path.isdir(ckpt):
        print("checkpoint not found (set --ckpt, MARIN_PREFIX or NANOMARIN_ROOT)")
        return 1
    out = args.out or os.path.join(args.src, "cache", "s6", "caseC_in")
    pin_index = json.load(open(os.path.join(args.src, "cache/levref/130m/step0_f32/caseC/pin/index.json")))
    manifest = json.load(open(os.path.join(ckpt, "manifest.json")))
    os.makedirs(os.path.join(out, "pin"), exist_ok=True)
    os.makedirs(os.path.join(out, "sin"), exist_ok=True)
    index = {"checkpoint": CKPT_REL, "pin": {}, "sin": {}, "scalars": {}}
    bad = 0
    for a in manifest["arrays"]:
        path = a["path"]
        spec = {"driver": a.get("driver", manifest["array_driver"]),
                "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(ckpt), "path": path}}
        x = np.asarray(ts.open(spec, read=True).result().read().result())
        if x.ndim == 0:
            index["scalars"][path] = x.item()
            continue
        if path.startswith("model/"):
            name = leaf_name(path[len("model/"):])
            ref = pin_index[name]["sha256"]
            ok = sha(x) == ref
            bad += not ok
            np.save(os.path.join(out, "pin", name + ".npy"), x)
            index["pin"][name] = dict(shape=list(x.shape), sha256=sha(x), matches_fixture=ok)
        elif path.startswith("opt_state/inner_state/inner_states/"):
            rest = path[len("opt_state/inner_state/inner_states/"):]
            group, _, _, state, leaf = rest.split("/", 4)   # <group>/inner_state/1/<state>/<leaf path>
            name = f"{group}.{state}.{leaf_name(leaf)}"
            np.save(os.path.join(out, "sin", name + ".npy"), x)
            index["sin"][name] = dict(shape=list(x.shape), sha256=sha(x))
        print(path, x.shape, x.dtype)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    print("parameters not matching caseC/pin sha256:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
