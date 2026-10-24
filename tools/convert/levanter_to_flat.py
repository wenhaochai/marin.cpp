#!/usr/bin/env python3
"""Convert levanter zarr3 token caches to the flat files of DESIGN.md §5.1 (run once, CPU, vis node).

  python -I tools/convert/levanter_to_flat.py fineweb [--threads 7] [--seeds 42] [--levanter-bytes FILE]
  python -I tools/convert/levanter_to_flat.py paloma  [--levanter-bytes FILE]
  python -I tools/convert/levanter_to_flat.py verify  <out dir>      # re-hash every file listed in its meta.json

Inputs (read only): $NANOMARIN_ROOT/marin_store_big/{fineweb-edu-10B,paloma}/... (the caches the baseline read, data.md
§2-§8), the tokenizer.json of marin-community/marin-tokenizer from the HF hub cache, and the verified references in
cache/data/ of this repo (order table sha, 12 batch hashes, Paloma domain hashes, levanter byte table).
Outputs: $NANOMARIN_ROOT/marin_store_big/marincpp/<name>/ (never inside the repo). Every file is written as <file>.tmp
and every check reads the .tmp files; only when all checks pass does commit() remove the old meta.json, rename the .tmp
files, delete the files the old meta.json listed and the new one does not, and write meta.json last. A failed check
leaves the previous set untouched (plus .tmp files); a crash during the commit leaves no meta.json, so no reader pairs
old metadata with new files.

Train set fineweb-edu-10B-marin128k/:
  tokens.u32          all 10,000,000,738 tokens in cache order, uint32 LE (ids < 2^17)
  docs.u64            document start offsets (token index), n_docs + 1 entries (the last = n_tokens)
  doc_bytes.u64       [n_docs, 2] (begin, end) UTF-8 byte offsets of each document's text in the concatenated text of
                      all documents (bytes_exact of its tokens; BOS/EOS and other special ids count 0)
  segs_per_seq.u16    segments of each 4096-token window: 1 + #EOS at positions 0..4094 (data.md §3.1), uint16 LE,
                      exact (counted in int64; a window holds at most 4096 segments). Paloma's twitterAAE windows hold
                      up to 291, so 8 bits do not suffice (data.hpp and DESIGN.md §5.1 still say .u8)
  order_seed<s>.u32   s(j) for j < 5,859,328 (the T0.1 table; one per seed in --seeds)
  bytes_levanter.i32  levanter's bytes_per_token (U+FFFD rule), copied from cache/data (sha256 d3be7450...)
  bytes_exact.i32     exact UTF-8 bytes of each id from the byte-level BPE vocabulary (special ids 0)
  meta.json           counts, BOS/EOS, vocab, max_segs_per_seq (+ histogram), sha256 + size of every file, LCG constants
                      per seed, source paths + shard ledger, and the checks the converter ran
Eval set paloma-marin128k/: the floor(tokens/4096) windows of the 16 domains concatenated in component order (data.md
§8.1), tags.u16 (domain index per window), docs.u64 / segs_per_seq.u16 as above, both byte tables, and per domain in
meta.json: first_seq, n_seqs, scored_tokens, the levanter target bytes, covered_bytes (exact UTF-8 bytes of the
domain token stream its windows cover, and covered_text_bytes without the space marin appends to every
document), sha256 of its windows; plus the estimator batch layout (128 rows, 114 batches, 65 real
rows in the last).

Dependencies: numpy, numcodecs (its bundled c-blosc via ctypes), google_crc32c; the zarr shard reader and the order
code come from bench/n1_loader/nmloader.py (verified against levanter: 103/103 checks).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "bench", "n1_loader"))
import nmloader as nm  # noqa: E402

FORMAT_VERSION = 2  # 2: exact segs_per_seq.u16 replaces segs_per_seq.u8 (counts wrapped mod 256 in format 1)
SEQ = 4096
VOCAB = 128256
BOS, EOS = 128000, 128001
ORDER_ENTRIES = 5_859_328  # all four baseline sizes (1_2b: 22888 x 256)
ORDER_SHA = "dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b"
BYTES_SHA = "d3be74509077b5ba84a2822f0348c7574b685404178786b7e0d473c64a334e4d"
TOKENIZER_SNAPSHOT = "models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
PALOMA = [("4chan", "4chan_meta_sep"), ("c4_100_domains", "c4_100_domains"), ("c4_en", "c4_en"),
          ("dolma-v1_5", "dolma-v1_5"), ("dolma_100_programing_languages", "dolma_100_programing_languages"),
          ("dolma_100_subreddits", "dolma_100_subreddits"), ("falcon-refinedweb", "falcon-refinedweb"),
          ("gab", "gab"), ("m2d2_s2orc_unsplit", "m2d2_s2orc_unsplit"),
          ("m2d2_wikipedia_unsplit", "m2d2_wikipedia_unsplit"), ("manosphere_meta_sep", "manosphere_meta_sep"),
          ("mc4", "mc4"), ("ptb", "ptb"), ("redpajama", "redpajama"),
          ("twitterAAE_HELM_fixed", "twitterAAE_HELM_fixed"), ("wikitext_103", "wikitext_103")]
EVAL_BATCH = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def root() -> str:
    r = os.environ.get("NANOMARIN_ROOT")
    if not r:
        raise SystemExit("NANOMARIN_ROOT is not set")
    return r


def store() -> str:
    return os.path.join(root(), "marin_store_big")


def out_root() -> str:
    return os.path.join(store(), "marincpp")


def sha256_file(path: str, bufsize: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        buf = bytearray(bufsize)
        mv = memoryview(buf)
        while True:
            n = f.readinto(buf)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


class Writer:
    """Sequential writer of one output file: <path>.tmp with a running sha256; commit() renames it."""

    def __init__(self, path: str):
        self.path, self.tmp = path, path + ".tmp"
        self.f = open(self.tmp, "wb", buffering=0)
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, arr: np.ndarray):
        b = memoryview(np.ascontiguousarray(arr)).cast("B")
        self.h.update(b)
        off = 0
        while off < len(b):
            off += self.f.write(b[off:])
        self.n += len(b)

    def close(self) -> dict:
        os.fsync(self.f.fileno())
        self.f.close()
        return {"sha256": self.h.hexdigest(), "bytes": self.n}


def write_array(path: str, arr: np.ndarray) -> dict:
    """Writes <path>.tmp (renamed later by commit())."""
    w = Writer(path)
    w.write(arr)
    return w.close()


def fsync_dir(d: str):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(out_dir: str, meta: dict):
    """Commit a converted set after every check passed on the .tmp files (module docstring): drop the old meta.json,
    rename every <file>.tmp of meta["files"], delete the files only the old meta.json listed, write meta.json."""
    p = os.path.join(out_dir, "meta.json")
    old_files = set(json.load(open(p))["files"]) if os.path.exists(p) else set()
    if os.path.exists(p):
        os.remove(p)
        fsync_dir(out_dir)
    for f in meta["files"]:
        os.replace(os.path.join(out_dir, f + ".tmp"), os.path.join(out_dir, f))
    for f in sorted(old_files - set(meta["files"])):   # e.g. segs_per_seq.u8 of format-1 converters before the u16 fix
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
            log("removed", f, "(listed by the old meta.json only)")
    fsync_dir(out_dir)
    with open(p + ".tmp", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(p + ".tmp", p)
    fsync_dir(out_dir)
    log("meta.json sha256", sha256_file(p))


# ---------------------------------------------------------------------------------------------- byte tables
def tokenizer_json() -> str:
    p = os.path.join(root(), "cache", "huggingface", "hub", TOKENIZER_SNAPSHOT, "tokenizer.json")
    if not os.path.exists(p):
        raise SystemExit(f"tokenizer.json not found: {p}")
    return p


def bytes_exact_table(path: str) -> np.ndarray:
    """Exact UTF-8 bytes per id: invert GPT-2's bytes_to_unicode over each byte-level BPE token string; special ids 0."""
    t = json.load(open(path, encoding="utf-8"))
    assert t["normalizer"] is None, t["normalizer"]  # lossless: decoded text bytes = concatenated token bytes
    assert t["decoder"]["type"] == "ByteLevel", t["decoder"]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    table = np.full(VOCAB, -1, dtype=np.int64)
    for tok, i in t["model"]["vocab"].items():
        table[i] = len(bytes(u2b[ch] for ch in tok))
    for a in t["added_tokens"]:
        assert a["special"], a
        table[a["id"]] = 0
    assert (table >= 0).all(), np.nonzero(table < 0)[0][:10]
    return table.astype("<i4")


def bytes_levanter_table(levanter_file: str | None) -> tuple[np.ndarray, dict]:
    ref = os.path.join(REPO, "cache", "data", "bytes_per_token_marin.i32")
    raw = open(ref, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != BYTES_SHA:
        raise SystemExit(f"{ref}: sha256 {got} != {BYTES_SHA}")
    t = np.frombuffer(raw, dtype="<i4").copy()
    assert t.shape == (VOCAB,)
    check = {"source": "cache/data/bytes_per_token_marin.i32", "sha256": got}
    if levanter_file:  # tools/convert/bytes_levanter.py output: levanter's byte_length_of_token rerun today
        lt = np.fromfile(levanter_file, dtype="<i4")
        check["levanter_recheck"] = {"file": os.path.basename(levanter_file), "equal": bool(np.array_equal(lt, t)),
                                     "mismatches": int((lt != t).sum()) if lt.shape == t.shape else -1}
        if not check["levanter_recheck"]["equal"]:
            raise SystemExit(f"levanter byte table recheck failed: {check['levanter_recheck']}")
    return t, check


# ---------------------------------------------------------------------------------------------- zarr bulk reader
class BulkZarr:
    """Sequential bulk decode of a 1-D int64 zarr3 array (sharding_indexed: bytes LE + blosc, index at shard end):
    whole inner chunks, decompressed in a thread pool through c-blosc (GIL released)."""

    def __init__(self, path: str):
        self.z = nm.ZarrInt64(path, mode="chunk", check_crc=True)
        self.lib = nm.ZarrInt64._lib
        self.inner = self.z.inner

    def chunk(self, cid: int, out: np.ndarray):
        s, c = divmod(cid, self.z.per_shard)
        fd, idx = self.z._shard(s)
        off, nbytes = idx[c]
        if off == nm._EMPTY:
            out[:] = self.z.fill
            return
        src = np.empty(int(nbytes), dtype=np.uint8)
        got = os.preadv(fd, [memoryview(src)], int(off))
        assert got == int(nbytes), (cid, got, nbytes)
        r = self.lib.blosc_decompress_ctx(src.ctypes.data, out.ctypes.data, out.nbytes, 1)
        assert r == self.inner * 8, (cid, r)

    def read(self, pool: ThreadPoolExecutor, start_chunk: int, n_chunks: int) -> np.ndarray:
        out = np.empty((n_chunks, self.inner), dtype=np.int64)
        list(pool.map(lambda i: self.chunk(start_chunk + i, out[i]), range(n_chunks)))
        return out.reshape(-1)


def read_offsets(field_dir: str) -> tuple[int, np.ndarray]:
    z = nm.ZarrInt64(os.path.join(field_dir, "offsets"), mode="chunk")
    n_rows = int(z.read(0, 1)[0])
    ends = z.read(1, n_rows).copy()
    assert (np.diff(ends) > 0).all() and ends[0] > 0, "document offsets not strictly increasing"
    return n_rows, ends


def segs_per_window(tok: np.ndarray, n_win: int) -> np.ndarray:
    """Segments of each window, 1 + #EOS at positions 0..SEQ-2, counted in int64 (never narrowed here)."""
    w = tok[: n_win * SEQ].reshape(n_win, SEQ)
    return 1 + (w[:, : SEQ - 1] == EOS).sum(axis=1, dtype=np.int64)


def segs_u16(c: np.ndarray) -> np.ndarray:
    """The stored form of exact segment counts: uint16 LE; refuses any count that does not fit (none can at SEQ 4096)."""
    assert c.dtype == np.int64 and (len(c) == 0 or (c.min() >= 1 and c.max() <= 0xFFFF)), (c.min(), c.max())
    return c.astype("<u2")


def lcg_table(n_tokens: int, seeds: list[int], seq_lens: list[int]) -> dict:
    """levanter's LCG constants per (seed, N = n_tokens // seq_len): key "<seed>" for seq_len 4096, "<seed>@<S>" else."""
    out = {}
    for S in seq_lens:
        n_seqs = n_tokens // S
        for s in seeds:
            o = nm.LevanterOrder(n_seqs, s)
            key = str(s) if S == SEQ else f"{s}@{S}"
            out[key] = {"N": n_seqs, "seq_len": S, "a": o.a, "b": o.b, "pcg64_seed": o.lcg_seed, "mix_key": list(o.mix_key)}
    return out


# ---------------------------------------------------------------------------------------------- fineweb
def convert_fineweb(args):
    src_split = os.path.join(store(), "fineweb-edu-10B", "2026.06.28", "train")
    field = os.path.join(src_split, "input_ids")
    out_dir = os.path.join(out_root(), "fineweb-edu-10B-marin128k")
    os.makedirs(out_dir, exist_ok=True)
    ledger = json.load(open(os.path.join(src_split, "shard_ledger.json")))
    ref = json.load(open(os.path.join(REPO, "cache", "data", "data_ref.json")))
    t0 = time.time()

    n_docs, ends = read_offsets(field)
    n_tokens = int(ends[-1])
    assert n_docs == ledger["total_num_rows"] == ref["fineweb"]["num_rows"], n_docs
    assert n_tokens == ref["fineweb"]["data_size"], n_tokens
    starts = np.concatenate([[0], ends]).astype(np.uint64)  # n_docs + 1
    n_seqs = n_tokens // SEQ
    log(f"fineweb: {n_docs} docs, {n_tokens} tokens, {n_seqs} windows")

    bex = bytes_exact_table(tokenizer_json()).astype(np.int64)
    blev, blev_check = bytes_levanter_table(args.levanter_bytes)

    files = {}
    files["docs.u64"] = write_array(os.path.join(out_dir, "docs.u64"), starts.astype("<u8"))
    zr = BulkZarr(os.path.join(field, "data"))
    inner = zr.inner
    assert inner % SEQ == 0
    n_chunks = -(-n_tokens // inner)
    per_batch = args.batch_chunks
    wt = Writer(os.path.join(out_dir, "tokens.u32"))
    ws = Writer(os.path.join(out_dir, "segs_per_seq.u16"))
    doc_byte_start = np.zeros(n_docs + 1, dtype=np.uint64)
    byte_pos = 0  # exact UTF-8 bytes before the current batch
    max_id, n_eos, n_bos, win_done = 0, 0, 0, 0
    hist = np.zeros(SEQ + 1, dtype=np.int64)  # over the exact int64 counts (1..SEQ)
    pool = ThreadPoolExecutor(max_workers=1)  # decodes batch i+1 while batch i is written
    pool_inner = ThreadPoolExecutor(max_workers=args.threads)
    nxt = pool.submit(zr.read, pool_inner, 0, min(per_batch, n_chunks))
    for c0 in range(0, n_chunks, per_batch):
        nb = min(per_batch, n_chunks - c0)
        raw = nxt.result()
        if c0 + per_batch < n_chunks:
            nxt = pool.submit(zr.read, pool_inner, c0 + per_batch, min(per_batch, n_chunks - c0 - per_batch))
        tok0 = c0 * inner
        n_here = min(nb * inner, n_tokens - tok0)
        tok = raw[:n_here]
        lo, hi = int(tok.min()), int(tok.max())
        assert lo >= 0 and hi < VOCAB, (c0, lo, hi)
        max_id = max(max_id, hi)
        if c0 + nb >= n_chunks:  # the last chunk: elements past the data end must be the fill value
            assert not raw[n_here:].any()
        u = tok.astype("<u4")
        wt.write(u)
        n_eos += int((u == EOS).sum())
        n_bos += int((u == BOS).sum())
        # windows wholly inside [0, n_seqs * SEQ): batches start on a window boundary (inner % SEQ == 0)
        nw = min(n_here // SEQ, n_seqs - win_done)
        sp = segs_per_window(u, nw)
        hist += np.bincount(sp, minlength=SEQ + 1)
        ws.write(segs_u16(sp))
        win_done += nw
        # document byte offsets: prefix sums of exact bytes, read at the document starts inside this batch
        cb = np.cumsum(bex[u], dtype=np.int64)
        i0 = np.searchsorted(starts, tok0, side="left")
        i1 = np.searchsorted(starts, tok0 + n_here, side="left")
        loc = starts[i0:i1].astype(np.int64) - tok0
        doc_byte_start[i0:i1] = byte_pos + np.where(loc > 0, cb[np.maximum(loc - 1, 0)], 0)
        byte_pos += int(cb[-1])
        if (c0 // per_batch) % 20 == 0:
            el = time.time() - t0
            log(f"  chunk {c0 + nb}/{n_chunks}  {(tok0 + n_here) / 1e9:.2f} Gtok  {(tok0 + n_here) / el / 1e6:.1f} Mtok/s")
    pool.shutdown()
    pool_inner.shutdown()
    doc_byte_start[n_docs] = byte_pos
    assert win_done == n_seqs and int(hist.sum()) == n_seqs
    files["tokens.u32"] = wt.close()
    files["segs_per_seq.u16"] = ws.close()
    assert files["tokens.u32"]["bytes"] == 4 * n_tokens and files["segs_per_seq.u16"]["bytes"] == 2 * n_seqs
    db = np.stack([doc_byte_start[:-1], doc_byte_start[1:]], axis=1).astype("<u8")
    files["doc_bytes.u64"] = write_array(os.path.join(out_dir, "doc_bytes.u64"), db)
    files["bytes_levanter.i32"] = write_array(os.path.join(out_dir, "bytes_levanter.i32"), blev.astype("<i4"))
    files["bytes_exact.i32"] = write_array(os.path.join(out_dir, "bytes_exact.i32"), bex.astype("<i4"))
    log(f"tokens written in {time.time() - t0:.0f} s")

    # order tables + LCG constants
    seeds = [int(s) for s in args.seeds.split(",")]
    lcg = lcg_table(n_tokens, seeds, [int(x) for x in args.seq_lens.split(",")])
    order_tables = {}
    for s in seeds:
        o = nm.LevanterOrder(n_seqs, s)
        tab = np.empty(ORDER_ENTRIES, dtype="<u4")
        for j0 in range(0, ORDER_ENTRIES, 1 << 20):
            j = np.arange(j0, min(j0 + (1 << 20), ORDER_ENTRIES))
            tab[j0 : j0 + len(j)] = o.seq_ids(j)
        name = f"order_seed{s}.u32"
        files[name] = write_array(os.path.join(out_dir, name), tab)
        order_tables[str(s)] = {"file": name, "count": ORDER_ENTRIES, "block": nm.MIX_BLOCK,
                                "sha256": files[name]["sha256"]}
        if s == 42 and files[name]["sha256"] != ORDER_SHA:
            raise SystemExit(f"T0.1 order table sha256 {files[name]['sha256']} != {ORDER_SHA}")

    # checks against the levanter-verified references (data.md §7) on the written, not yet committed, files
    checks = {"bytes_levanter": blev_check}
    tokmm = np.memmap(wt.tmp, dtype="<u4", mode="r")
    assert len(tokmm) == n_tokens
    order = nm.LevanterOrder(n_seqs, 42)
    bh = {}
    for key, v in ref["batch_hashes"].items():
        B, k = (int(x) for x in key[1:].replace("_step", " ").split())
        sids = order.seq_ids(np.arange(k * B, (k + 1) * B))
        h = hashlib.sha256()
        for s in sids:
            h.update(memoryview(np.ascontiguousarray(tokmm[s * SEQ : (s + 1) * SEQ])).cast("B"))
        bh[key] = h.hexdigest() == v["sha256"]
        if not bh[key]:
            raise SystemExit(f"T0.2 batch hash {key}: {h.hexdigest()} != {v['sha256']}")
    checks["batch_hashes_equal"] = bh
    # an independent decode path (nmloader block mode, blosc_getitem) on random windows: the tokens and the stored
    # segment counts of each sampled window, and the tail
    rng = np.random.default_rng(20261024)
    jt = nm.JaggedTokens(field, mode="block")
    segmm = np.memmap(ws.tmp, dtype="<u2", mode="r")
    samp = rng.integers(0, n_seqs, size=args.spot_windows)
    bad = bad_segs = 0
    for s in samp:
        a = jt.data.read(int(s) * SEQ, SEQ)
        bad += not np.array_equal(a.astype("<u4"), tokmm[s * SEQ : (s + 1) * SEQ])
        bad_segs += int(segmm[s]) != 1 + int((a[: SEQ - 1] == EOS).sum())
    tail = jt.data.read(n_seqs * SEQ, n_tokens - n_seqs * SEQ)
    bad += not np.array_equal(tail.astype("<u4"), tokmm[n_seqs * SEQ :])
    checks["spot_windows_block_decoder"] = {"windows": int(len(samp)) + 1, "mismatches": int(bad),
                                            "segs_per_seq_mismatches": int(bad_segs)}
    if bad or bad_segs:
        raise SystemExit(f"spot check against the block decoder: {bad} token / {bad_segs} segment-count mismatches")
    max_segs = int(np.nonzero(hist)[0].max())
    if int(segmm.max()) != max_segs:
        raise SystemExit(f"segs_per_seq.u16 max {int(segmm.max())} != histogram max {max_segs}")
    del tokmm, segmm

    meta = {
        "format": FORMAT_VERSION, "name": "fineweb-edu-10B-marin128k", "kind": "train",
        "n_tokens": n_tokens, "seq_len": SEQ, "n_seqs": n_seqs, "n_docs": n_docs, "vocab": VOCAB,
        "bos": BOS, "eos": EOS, "max_token_id": max_id,
        "max_segs_per_seq": max_segs, "segs_per_seq_hist": {str(i): int(c) for i, c in enumerate(hist) if c},
        "n_eos": n_eos, "n_bos": n_bos, "text_bytes_exact": int(byte_pos),
        "tokens_sha256": files["tokens.u32"]["sha256"],
        "files": files, "lcg": lcg, "order_tables": order_tables,
        "source": {"cache": "fineweb-edu-10B/2026.06.28/train (marin_store_big)", "ledger": ledger,
                   "tokenizer": TOKENIZER_SNAPSHOT, "dtype": "int64 zarr3 sharding_indexed blosc lz4"},
        "checks": checks, "converter": {"script": "tools/convert/levanter_to_flat.py", "seconds": round(time.time() - t0, 1)},
    }
    commit(out_dir, meta)
    log(f"fineweb done in {time.time() - t0:.0f} s: max_segs_per_seq {meta['max_segs_per_seq']}")


# ---------------------------------------------------------------------------------------------- paloma
def raw_doc_bytes(dirname: str) -> list[int]:
    d = os.path.join(store(), "raw", "paloma-fc6827", "65cd6fc", dirname, "val")
    out = []
    for f in sorted(x for x in os.listdir(d) if x.startswith("val") and x.endswith(".jsonl.gz")):
        with gzip.open(os.path.join(d, f), "rt", encoding="utf-8") as g:
            for line in g:
                out.append(len(json.loads(line)["text"].encode("utf-8")))
    return out


def convert_paloma(args):
    out_dir = os.path.join(out_root(), "paloma-marin128k")
    os.makedirs(out_dir, exist_ok=True)
    ref = json.load(open(os.path.join(REPO, "cache", "data", "data_ref.json")))["paloma"]
    t0 = time.time()
    bex = bytes_exact_table(tokenizer_json()).astype(np.int64)
    blev, blev_check = bytes_levanter_table(args.levanter_bytes)
    wt = Writer(os.path.join(out_dir, "tokens.u32"))
    tags, segs, docs, domains = [], [], [], []
    first = 0
    for di, (name, rawdir) in enumerate(PALOMA):
        split = os.path.join(store(), "paloma", f"{name}-marin-tokenizer", "2026.06.28", "validation")
        parts = sorted(p for p in os.listdir(split) if p.startswith("part-"))
        assert len(parts) == 1, parts
        field = os.path.join(split, parts[0], "input_ids")
        ledger = json.load(open(os.path.join(split, parts[0], "shard_ledger.json")))
        n_docs, ends = read_offsets(field)
        n_tok = int(ends[-1])
        z = nm.ZarrInt64(os.path.join(field, "data"), mode="chunk")
        tok = z.read(0, n_tok)
        assert tok.min() >= 0 and tok.max() < VOCAB
        n_seqs = n_tok // SEQ
        win = tok[: n_seqs * SEQ].astype("<u4")
        sha = hashlib.sha256(memoryview(win).cast("B")).hexdigest()
        r = ref[name]
        starts = np.concatenate([[0], ends[:-1]])
        assert starts.dtype == np.int64
        exp = {"docs": r["docs"], "tokens": r["tokens"], "seqs": r["seqs"], "sha256_seqs": r["sha256_seqs"]}
        got = {"docs": n_docs, "tokens": n_tok, "seqs": n_seqs, "sha256_seqs": sha}
        if got != exp:
            raise SystemExit(f"T0.4 paloma {name}: {got} != {exp}")
        w2 = win.reshape(n_seqs, SEQ)
        # levanter target bytes: bytes_per_token[next token] over scored positions 0..4094 of every window
        target_bytes = int(blev[w2[:, 1:]].astype(np.int64).sum())
        if target_bytes != r["target_bytes"]:
            raise SystemExit(f"paloma {name}: target bytes {target_bytes} != {r['target_bytes']}")
        # exact bytes: per document (text check against the raw jsonl) and the part the windows cover
        dbytes = np.add.reduceat(bex[tok], starts)
        raw = raw_doc_bytes(rawdir) if not args.no_raw_check else None
        text_check = None
        if raw is not None:
            # marin's tokenize step appends one space to every document before EOS (4chan doc 0 ends "... 220 128001"):
            # the token stream holds each raw text plus exactly 1 byte, in raw file order
            raw = np.asarray(raw, dtype=np.int64)
            delta = dbytes - raw if len(raw) == n_docs else None
            text_check = {"raw_docs": int(len(raw)), "stream_minus_text_bytes": None if delta is None else
                          sorted(set(int(x) for x in np.unique(delta)))}
            if delta is None or not (delta == 1).all():
                raise SystemExit(f"paloma {name}: exact byte table disagrees with the raw text: {text_check}")
        covered = int(bex[win].sum())
        # raw text bytes the windows cover: the stream bytes minus the appended space of each document whose end the
        # windows include (its last text token sits at ends - 2, before the EOS)
        covered_text = covered - int((ends - 2 < n_seqs * SEQ).sum())
        kept = starts[starts < n_seqs * SEQ]
        wt.write(win)
        tags.append(np.full(n_seqs, di, dtype="<u2"))
        segs.append(segs_per_window(win, n_seqs))
        docs.append((kept + first * SEQ).astype("<u8"))
        domains.append({
            "name": name, "tag": f"paloma/{name}-marin-tokenizer", "first_seq": first, "n_seqs": n_seqs,
            "scored_tokens": n_seqs * (SEQ - 1), "docs": n_docs, "docs_in_windows": int(len(kept)), "tokens": n_tok,
            "dropped_tail": n_tok - n_seqs * SEQ, "target_bytes_levanter": target_bytes,
            "covered_bytes_begin": 0, "covered_bytes_end": covered, "covered_text_bytes": covered_text,
            "stream_bytes_exact": int(dbytes.sum()), "text_bytes_raw": int(dbytes.sum()) - n_docs,
            "sha256": sha, "ledger_rows": ledger["total_num_rows"], "text_check": text_check,
        })
        log(f"  {name}: {n_seqs} windows, sha ok, covered {covered} of {int(dbytes.sum())} stream bytes, {text_check}")
        first += n_seqs
    n_seqs = first
    assert n_seqs == 14529
    files = {"tokens.u32": wt.close()}
    tag = np.concatenate(tags)
    files["tags.u16"] = write_array(os.path.join(out_dir, "tags.u16"), tag)
    sp = np.concatenate(segs)  # exact int64 counts
    files["segs_per_seq.u16"] = write_array(os.path.join(out_dir, "segs_per_seq.u16"), segs_u16(sp))
    files["docs.u64"] = write_array(os.path.join(out_dir, "docs.u64"),
                                    np.concatenate(docs + [np.array([n_seqs * SEQ], dtype="<u8")]))
    files["bytes_levanter.i32"] = write_array(os.path.join(out_dir, "bytes_levanter.i32"), blev.astype("<i4"))
    files["bytes_exact.i32"] = write_array(os.path.join(out_dir, "bytes_exact.i32"), bex.astype("<i4"))
    # checks on the written, not yet committed, files: the 16 domain hashes re-read from tokens.u32.tmp, the stored
    # segment counts recounted from it
    tokf = np.fromfile(wt.tmp, dtype="<u4")
    assert len(tokf) == n_seqs * SEQ
    for dom in domains:
        a, n = dom["first_seq"] * SEQ, dom["n_seqs"] * SEQ
        if hashlib.sha256(memoryview(tokf[a : a + n]).cast("B")).hexdigest() != ref[dom["name"]]["sha256_seqs"]:
            raise SystemExit(f"T0.4 paloma {dom['name']}: tokens.u32.tmp slice differs from the reference hash")
    recount = segs_per_window(tokf, n_seqs)
    stored = np.fromfile(os.path.join(out_dir, "segs_per_seq.u16.tmp"), dtype="<u2").astype(np.int64)
    if not np.array_equal(stored, recount):
        raise SystemExit(f"segs_per_seq.u16: {int((stored != recount).sum())} windows differ from a recount")
    del tokf
    n_batches = -(-n_seqs // EVAL_BATCH)
    meta = {
        "format": FORMAT_VERSION, "name": "paloma-marin128k", "kind": "eval",
        "n_tokens": n_seqs * SEQ, "seq_len": SEQ, "n_seqs": n_seqs, "n_docs": int(sum(len(d) for d in docs)),
        "vocab": VOCAB, "bos": BOS, "eos": EOS, "max_segs_per_seq": int(sp.max()),
        "tokens_sha256": files["tokens.u32"]["sha256"], "files": files, "domains": domains,
        "eval_batch": EVAL_BATCH, "n_batches": n_batches, "last_batch_real_rows": n_seqs - (n_batches - 1) * EVAL_BATCH,
        "source": {"cache": "paloma/<domain>-marin-tokenizer/2026.06.28/validation (marin_store_big)",
                   "raw": "raw/paloma-fc6827/65cd6fc/<dir>/val/val*.jsonl.gz", "tokenizer": TOKENIZER_SNAPSHOT},
        "checks": {"bytes_levanter": blev_check, "domain_sha256_equal": True, "target_bytes_equal": True,
                   "domain_sha256_from_written_file": True, "segs_per_seq_recount_equal": True},
        "converter": {"script": "tools/convert/levanter_to_flat.py", "seconds": round(time.time() - t0, 1)},
    }
    commit(out_dir, meta)
    log(f"paloma done: {n_seqs} windows, {n_batches} batches, last {meta['last_batch_real_rows']} rows, "
        f"max_segs_per_seq {meta['max_segs_per_seq']}")


def verify(args):
    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    bad = 0
    for f, info in sorted(meta["files"].items()):
        p = os.path.join(args.dir, f)
        size = os.path.getsize(p)
        h = sha256_file(p)
        ok = size == info["bytes"] and h == info["sha256"]
        bad += not ok
        log(("OK  " if ok else "FAIL"), f, size, h)
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["fineweb", "paloma", "verify"])
    ap.add_argument("dir", nargs="?", help="verify: the output directory")
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--batch-chunks", type=int, default=64, help="inner chunks (262,144 tokens each) per write batch")
    ap.add_argument("--seeds", default="42", help="data seeds whose LCG constants and order tables are written")
    ap.add_argument("--seq-lens", default="4096", help="window lengths whose LCG constants are written (data plans)")
    ap.add_argument("--levanter-bytes", default=None, help="levanter byte table rerun by tools/convert/bytes_levanter.py")
    ap.add_argument("--spot-windows", type=int, default=2000)
    ap.add_argument("--no-raw-check", action="store_true", help="paloma: skip the exact-bytes check against raw text")
    args = ap.parse_args()
    if args.what == "fineweb":
        convert_fineweb(args)
    elif args.what == "paloma":
        convert_paloma(args)
    else:
        if not args.dir:
            ap.error("verify needs the output directory")
        verify(args)


if __name__ == "__main__":
    main()
