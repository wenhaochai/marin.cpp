#!/usr/bin/env python3
"""Rerun levanter's bytes-per-token rule (T0.5 recheck): byte_length_of_token for every id of the marin tokenizer.

  HF_HUB_OFFLINE=1 $NANOMARIN_ROOT/project/marin/.venv/bin/python -I tools/convert/bytes_levanter.py OUT.i32

Imports levanter read-only from $NANOMARIN_ROOT/project/marin (the code the baseline ran, data.md "Code version") and
loads the tokenizer from the HF hub cache snapshot, exactly as levanter's eval does (eval.py
_calculate_bytes_per_token_type: one byte_length_of_token call per id of get_vocab()). Writes int32 LE [128256]; the
converter (levanter_to_flat.py --levanter-bytes OUT.i32) requires it to equal cache/data/bytes_per_token_marin.i32.
"""
import hashlib
import os
import sys

import numpy as np

SNAPSHOT = "models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = os.environ.get("NANOMARIN_ROOT") or sys.exit("NANOMARIN_ROOT is not set")
    sys.path.insert(0, os.path.join(root, "project", "marin", "lib", "levanter", "src"))
    from levanter.tokenizers import load_tokenizer
    from levanter.utils.hf_utils import byte_length_of_token
    import levanter

    tok = load_tokenizer(os.path.join(root, "cache", "huggingface", "hub", SNAPSHOT))
    n = len(tok.get_vocab())
    t = np.empty(n, dtype="<i4")
    for i in range(n):
        t[i] = byte_length_of_token(tok, i)
    t.tofile(sys.argv[1])
    print("levanter", os.path.relpath(levanter.__file__, root), "vocab", n,
          "sha256", hashlib.sha256(t.tobytes()).hexdigest(), "zero ids", int((t == 0).sum()))


if __name__ == "__main__":
    main()
