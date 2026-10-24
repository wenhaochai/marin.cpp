#!/usr/bin/env python3
"""S1 gate: the converter's outputs and the data C ABI (stdlib only; Python 3.9+).

  python3 tests/s1/test_convert_outputs.py --lib build/libmarincpp_kernels.so --src . [--full]

1. Every file listed in each converted set's meta.json has its size and, for files under 1 GiB (all with --full,
   including the 40 GB tokens.u32), its sha256; meta format 2.
2. T0.1: order_seed42.u32 sha256; T0.5: bytes_levanter.i32 sha256 in both sets; T0.4: the 16 Paloma domain hashes of
   data.md section 8.1, re-hashed from tokens.u32 slices, 14,529 windows, 114 batches with 65 real rows in the last.
3. Exact segment counts (segs_per_seq.u16 = 1 + #EOS at positions 0..4094 of each window): Paloma recounted from
   tokens.u32 for every window, max 291 = meta.json max_segs_per_seq; fineweb max 17 = meta and histogram, and with
   --full every window recounted in the same pass that re-hashes tokens.u32. EOS is counted as the aligned byte
   pattern 01 f4 01 00: with ids < 2^17 the pattern cannot occur at an unaligned offset.
4. The C ABI through ctypes: nm_threefry2x32 (Random123 known answers), nm_levanter_order (s(0..7) of data.md 4.4,
   and an error for an unknown seed), nm_plan_seq_of on a config given as TOML text.
Exit 0 = pass, 1 = failure, 77 = skipped (converted sets absent; the ABI checks still run and must pass).
"""
import argparse
import array
import ctypes
import hashlib
import json
import os
import sys

PALOMA_SHA = {
    "4chan": (226, "418421ca41b128b5a6e15b6d0f620d9a4e4215c0b26a6712acf04b12802caee0"),
    "c4_100_domains": (2310, "167b9ea6e7dbb4b47d7691ba9931c76a3af1641d5a5d0130d2008326ef9767f4"),
    "c4_en": (236, "5ef2fc5286e6a1fac552f34e9e48b3b664ddec8e3550dacba889de051d433b69"),
    "dolma-v1_5": (703, "ef6c458e93f48c42e725ee9d3f3807ac77aa3c52b2de1c6979179ae8e0e04080"),
    "dolma_100_programing_languages": (2075, "8bd793b5b43e0a1319ef70ea2a0567ee73cae27053b9fd6e5e4508f644272a8f"),
    "dolma_100_subreddits": (2303, "f0548824ea12008b094e09c55aead5a826843982561cd7d179dbe0dd940a2abc"),
    "falcon-refinedweb": (236, "0f26d6b4aa56892d8194623578dae32260c1b3459e2dddf7867934c52a37b0c9"),
    "gab": (234, "7ebf2640b9eba8524cc34631b5b4f5e3b67f9dbf21d4eb9e8de73458af00f850"),
    "m2d2_s2orc_unsplit": (4112, "f9e4608406c0bb372a6d93c94b52ec14162f6f22b3b5f344fff2ecff05f55df0"),
    "m2d2_wikipedia_unsplit": (1181, "4d8d5225bc27e7859e0ee980579ff431314803937acb6bfc9a8ae66c38d4ac77"),
    "manosphere_meta_sep": (235, "09b6b8efed1a3847107a353540f51e501eb6b65922f3a130606983249de23f1e"),
    "mc4": (236, "081f5f67541a337845e3fd9f89b1589112f20eb90871ead87f5049e75030a6e3"),
    "ptb": (21, "ffb45ee0e9a0b5f9f6ed16c51175cf6f7a7ca89756128727a704233dd7b374e7"),
    "redpajama": (156, "8bc20349c6e0191412591b9f940c5b49d35f723bb70b8c0eac4e73ec99696e7e"),
    "twitterAAE_HELM_fixed": (205, "9d110d797b7a233ab5b82fdcdd3c7202f4f5bb93ffd14b5701b28774ccf32d93"),
    "wikitext_103": (60, "a3cadf50589690a770299069a1a7f9f81f0bd532c180713f276f20e6dc91002a"),
}
ORDER_SHA = "dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b"
SEQ = 4096
EOS_LE = (128001).to_bytes(4, "little")
BYTES_SHA = "d3be74509077b5ba84a2822f0348c7574b685404178786b7e0d473c64a334e4d"
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def sha_file(path, start=0, length=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = os.path.getsize(path) - start if length is None else length
        while left > 0:
            b = f.read(min(left, 64 << 20))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def data_root():
    if os.environ.get("NM_DATA"):
        return os.environ["NM_DATA"]
    if os.environ.get("NANOMARIN_ROOT"):
        return os.path.join(os.environ["NANOMARIN_ROOT"], "marin_store_big", "marincpp")
    return None


def read_u16(path):
    a = array.array("H")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def hash_and_count(path, n_windows):
    """sha256 of the whole token file and the segment count of each of its first n_windows windows, in one read."""
    h, counts, win_bytes = hashlib.sha256(), array.array("H"), SEQ * 4
    per = 4096  # windows per read: 64 MiB
    with open(path, "rb") as f:
        while True:
            buf = f.read(per * win_bytes)
            if not buf:
                break
            h.update(buf)
            for o in range(0, len(buf) - win_bytes + 1, win_bytes):
                if len(counts) < n_windows:
                    counts.append(1 + buf.count(EOS_LE, o, o + win_bytes - 4))
    return h.hexdigest(), counts


def check_set(d, full):
    meta = json.load(open(os.path.join(d, "meta.json")))
    check(meta["format"] == 2, f"{meta['name']} meta format {meta['format']} (2: exact segs_per_seq.u16)")
    check(not os.path.exists(os.path.join(d, "segs_per_seq.u8")), f"{meta['name']} has no format-1 segs_per_seq.u8")
    recount = None
    for f, info in sorted(meta["files"].items()):
        p = os.path.join(d, f)
        size = os.path.getsize(p)
        check(size == info["bytes"], f"{meta['name']}/{f} size {size}")
        if f == "tokens.u32" and (full or size < (1 << 30)):   # one pass: hash + segment recount of every window
            sha, recount = hash_and_count(p, meta["n_seqs"])
            check(sha == info["sha256"], f"{meta['name']}/{f} sha256")
        elif full or size < (1 << 30):
            check(sha_file(p) == info["sha256"], f"{meta['name']}/{f} sha256")
    check(sha_file(os.path.join(d, "bytes_levanter.i32")) == BYTES_SHA, f"T0.5 {meta['name']} bytes_levanter sha256")
    check(meta["checks"]["bytes_levanter"]["levanter_recheck"]["equal"], f"T0.5 {meta['name']} levanter recheck recorded")
    stored = read_u16(os.path.join(d, "segs_per_seq.u16"))
    check(len(stored) == meta["n_seqs"] and min(stored) >= 1 and max(stored) == meta["max_segs_per_seq"],
          f"{meta['name']} segs_per_seq.u16: {len(stored)} counts, max {max(stored)} = meta max_segs_per_seq "
          f"{meta['max_segs_per_seq']}")
    if recount is not None:
        bad = sum(1 for a, b in zip(stored, recount) if a != b)
        check(len(recount) == len(stored) and bad == 0,
              f"{meta['name']} segs_per_seq.u16 equals a recount from tokens.u32 for all {len(recount)} windows "
              f"({bad} differ)")
    return meta, stored, recount


def abi(lib_path, root):
    lib = ctypes.CDLL(lib_path)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.nm_threefry2x32.argtypes = [ctypes.c_uint32, ctypes.c_uint32, u32p, u32p, ctypes.c_int64]
    lib.nm_levanter_order.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, u32p]
    lib.nm_plan_seq_of.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    lib.nm_last_error.restype = ctypes.c_char_p
    x0 = (ctypes.c_uint32 * 1)(0x243F6A88)
    x1 = (ctypes.c_uint32 * 1)(0x85A308D3)
    check(lib.nm_threefry2x32(0x13198A2E, 0x03707344, x0, x1, 1) == 0 and (x0[0], x1[0]) == (0xC4923A9C, 0x483DF7A0),
          "C ABI nm_threefry2x32 known answer")
    out = (ctypes.c_uint32 * 8)()
    rc = lib.nm_levanter_order(42, 2441406, 2048, 0, 8, out)
    check(rc == 0 and list(out) == [1909879, 335277, 2165778, 1350661, 1748337, 1843177, 292285, 1995702],
          "C ABI nm_levanter_order s(0..7)")
    rc = lib.nm_levanter_order(41, 2441406, 2048, 0, 8, out)
    check(rc == 1 and b"no LCG constants" in lib.nm_last_error(), "C ABI nm_levanter_order unknown seed -> NM_ERR_INVALID")
    if root and os.path.exists(os.path.join(root, "fineweb-edu-10B-marin128k", "meta.json")):
        seq = ctypes.c_uint64()
        toml = b'[model]\nsize = "1_2b"\n[data]\nglobal_batch = 256\n[optim]\nnum_steps = 22888\n'
        rc = lib.nm_plan_seq_of(toml, 9536, 0, ctypes.byref(seq))
        check(rc == 0 and seq.value == 1316203, f"C ABI nm_plan_seq_of(1_2b, 9536, 0) = {seq.value} (data.md 7: 1316203)"
              + ("" if rc == 0 else " error: " + lib.nm_last_error().decode()))
        rc = lib.nm_plan_seq_of(b"", 4958, 0, ctypes.byref(seq))   # defaults: 130m, B 128
        check(rc == 0 and seq.value == 1568248, f"C ABI nm_plan_seq_of(defaults, 4958, 0) = {seq.value} (data.md 7: 1568248)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    root = data_root()
    abi(a.lib, root)
    fw = os.path.join(root, "fineweb-edu-10B-marin128k") if root else None
    pal = os.path.join(root, "paloma-marin128k") if root else None
    have = 0
    if fw and os.path.exists(os.path.join(fw, "meta.json")):
        have += 1
        m, stored, recount = check_set(fw, a.full)
        check(sha_file(os.path.join(fw, "order_seed42.u32")) == ORDER_SHA, "T0.1 order_seed42.u32 sha256")
        check(m["n_tokens"] == 10000000738 and m["n_seqs"] == 2441406 and m["n_docs"] == 9966814, "fineweb counts")
        check(all(m["checks"]["batch_hashes_equal"].values()) and len(m["checks"]["batch_hashes_equal"]) == 12,
              "T0.2 converter recorded 12 equal batch hashes")
        hist = {int(k): v for k, v in m["segs_per_seq_hist"].items()}
        stored_hist = {}
        for c in stored:
            stored_hist[c] = stored_hist.get(c, 0) + 1
        check(m["max_segs_per_seq"] == 17 and max(hist) == 17 and hist == stored_hist,
              "fineweb max_segs_per_seq 17 = histogram max; histogram = counts of segs_per_seq.u16")
        check(all("seq_len" in e for e in m["lcg"].values()), "fineweb meta.json lcg entries carry seq_len")
    if pal and os.path.exists(os.path.join(pal, "meta.json")):
        have += 1
        m, stored, recount = check_set(pal, a.full)   # tokens.u32 is 238 MB: always hashed and recounted
        tags = read_u16(os.path.join(pal, "tags.u16"))
        above = [t for c, t in zip(recount or [], tags) if c > 255]
        check(m["max_segs_per_seq"] == 291 and recount is not None and max(recount) == 291 and len(above) == 85 and
              set(above) == {14},
              "T0.4 paloma max_segs_per_seq 291 (85 windows above 255, all in twitterAAE_HELM_fixed, tag 14)")
        tok = os.path.join(pal, "tokens.u32")
        first = 0
        for dom in m["domains"]:
            n, sha = PALOMA_SHA[dom["name"]]
            check(dom["first_seq"] == first and dom["n_seqs"] == n, f"T0.4 {dom['name']} position {first} + {n}")
            check(sha_file(tok, first * 4096 * 4, n * 4096 * 4) == sha, f"T0.4 {dom['name']} sha256 from tokens.u32")
            first += n
        check([d["name"] for d in m["domains"]] == list(PALOMA_SHA), "T0.4 component order")
        check(first == 14529 and m["n_batches"] == 114 and m["last_batch_real_rows"] == 65, "T0.4 14529 windows, 114 batches, 65 rows")
    if FAILS:
        print(f"{len(FAILS)} failures")
        return 1
    return 0 if have == 2 else 77


if __name__ == "__main__":
    sys.exit(main())
