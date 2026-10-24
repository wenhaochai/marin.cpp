"""S3 gate: embedding gather and the deterministic sorted segment-sum backward (DESIGN.md §3.2, §11).

Checks: forward bitwise equal to table[tokens]; backward (EmbedAccum::F32) rel(kernel, f64 torch autograd of
F.embedding) <= 1e-6 per touched row block, rows of absent tokens bitwise untouched, accumulation into dtable (+=),
touched-row flags exact, bitwise repeatable; EmbedAccum::Bf16Sequential equals a bf16 running sum in ascending position
order bitwise; edge cases at the 256-entry chunk boundaries (a run ending on a chunk edge, one run over every chunk,
T = 1, 255, 256, 257); f32 storage; V = 128,256 at D = 512 (T = 131,072: one 130m microbatch per GPU) and D = 2048.
"""
import numpy as np
import torch

from nmk import BF16S, F32S, bits16, init, ptr, rb, rel

lib, R = init("s3_embed")
V_FULL = 128256


def zipf_tokens(T, V, gen, a=1.1):
    """Token ids with a heavy head (runs of thousands for the top ids), as real text."""
    rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,), generator=gen, device="cuda").item()))
    t = rng.zipf(a, size=T) - 1
    t = np.where(t < V, t, rng.integers(0, V, size=T))
    perm = rng.permutation(V)   # spread the frequent ids over the table
    return torch.from_numpy(perm[t].astype(np.int32)).cuda()


def workspace(T, V):
    nb = lib.nm_embed_bwd_workspace_bytes(T, V)
    R.check(nb > 0, "workspace bytes = 0")
    return torch.empty(nb, dtype=torch.uint8, device="cuda"), nb


def run_bwd(dtype, dout, tok, dtable, T, D, V, accum, touched=None):
    ws, nb = workspace(T, V)
    lib.nm_embed_bwd_dt(dtype, ptr(dout), ptr(tok), ptr(dtable), T, D, V, accum, ptr(touched), ptr(ws), nb, None)
    torch.cuda.synchronize()


def compact_ref_f64(dout, tok, D):
    """torch autograd of F.embedding in f64, on the rows of the tokens present only: (present ids, [n, D] f64)."""
    ids, inv = torch.unique(tok.long(), return_inverse=True)
    tab = torch.zeros(ids.numel(), D, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.nn.functional.embedding(inv, tab).backward(dout.double())
    return ids, tab.grad


def bwd_case(D, T, V=V_FULL, storage=BF16S, seed=0, tok=None, init_random=None):
    """dtable starts random when V * D is small (checks the += accumulation), else at zero (saves a V x D copy on the
    shared GPU). Checks: touched rows vs f64 autograd, untouched rows bitwise unchanged, touched flags, repeatability."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tok = zipf_tokens(T, V, gen) if tok is None else tok
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    dout = torch.randn(T, D, generator=gen, device="cuda").to(dt)
    init_random = (V * D <= (1 << 24)) if init_random is None else init_random
    dtable0 = torch.randn(V, D, generator=gen, device="cuda") if init_random else None
    ids, ref = compact_ref_f64(dout, tok, D)
    present = torch.zeros(V, dtype=torch.bool, device="cuda")
    present[ids] = True
    dtable = dtable0.clone() if init_random else torch.zeros(V, D, device="cuda")
    touched = torch.zeros(V, dtype=torch.uint8, device="cuda")
    run_bwd(storage, dout, tok, dtable, T, D, V, 0, touched)
    got = dtable[ids] - (dtable0[ids] if init_random else 0)
    r = rel(got, ref)
    counts = torch.bincount(tok.long(), minlength=V)
    print(f"    D={D} T={T} V={V} storage={'bf16' if storage else 'f32'}: {ids.numel()} distinct tokens, "
          f"longest run {int(counts.max())}, rel(sum, f64) {r:.3e}")
    R.check(r <= 1e-6, f"rel {r:.3e} > 1e-6")
    if init_random:
        same = (dtable.view(torch.int32) == dtable0.view(torch.int32)).all(1)
    else:
        same = (dtable.view(torch.int32) == 0).all(1)
    R.check(bool(same[~present].all()), "rows of absent tokens changed")
    R.check(torch.equal(touched.bool(), present), "touched-row flags differ from the token set")
    first = dtable[ids].clone()
    dtable.copy_(dtable0) if init_random else dtable.zero_()
    run_bwd(storage, dout, tok, dtable, T, D, V, 0, None)
    R.check(torch.equal(dtable[ids].view(torch.int32), first.view(torch.int32)), "two runs differ")
    return tok


@R.case
def forward_bitwise():
    for D, T in ((512, 131072), (2048, 32768), (64, 5)):
        gen = torch.Generator(device="cuda").manual_seed(D)
        tok = zipf_tokens(T, V_FULL, gen)
        for storage, dt in ((BF16S, torch.bfloat16), (F32S, torch.float32)):
            if D == 2048 and storage == F32S:
                continue
            table = torch.randn(V_FULL, D, generator=gen, device="cuda").to(dt)
            out = torch.empty(T, D, dtype=dt, device="cuda")
            lib.nm_embed_fwd_dt(storage, ptr(table), ptr(tok), ptr(out), T, D, None)
            torch.cuda.synchronize()
            ref = table[tok.long()]
            R.check(torch.equal(out.view(torch.int16 if storage else torch.int32),
                                ref.view(torch.int16 if storage else torch.int32)), f"gather differs D={D}")
        print(f"    D={D} T={T}: bitwise")


@R.case
def grad_embed_130m_pls():
    bwd_case(512, 131072)


@R.case
def grad_embed_1_2b():
    bwd_case(2048, 32768)


@R.case
def grad_embed_f32_storage():
    bwd_case(512, 16384, storage=F32S)


@R.case
def chunk_edges():
    D, V = 256, 1000
    cases = {
        "T=1": torch.tensor([7], dtype=torch.int32),
        "T=255": torch.arange(255, dtype=torch.int32) % 3,
        "T=256 one token": torch.full((256,), 5, dtype=torch.int32),
        "T=257": torch.arange(257, dtype=torch.int32) % 2,
        "one run over 9 chunks": torch.full((2300,), 42, dtype=torch.int32),
        "run ends on chunk edge": torch.cat([torch.full((512,), 3), torch.full((300,), 4), torch.arange(100) + 10]).int(),
        "run starts on chunk edge": torch.cat([torch.arange(256) + 500, torch.full((700,), 1)]).int(),
        "through chunks then short": torch.cat([torch.full((256 * 3 + 1,), 9), torch.full((1,), 8),
                                                torch.full((511,), 9)]).int(),
    }
    for name, tok in cases.items():
        tok = tok[torch.randperm(len(tok))].cuda() if "edge" not in name else tok.cuda()
        print(f"  {name}")
        bwd_case(D, len(tok), V=V, tok=tok, seed=len(tok))


def seq_bf16_ref(dout, tok, V, D, rnd=rb):
    """bf16 running sum per token in ascending position order (levanter's precision): all runs advance in lockstep over
    a dense [runs, longest run] index table, with no host synchronization inside the loop. rnd = identity: the f32
    storage form (a sequential f32 sum)."""
    T = tok.numel()
    order = torch.argsort(tok.long() * T + torch.arange(T, device="cuda"))
    st = tok.long()[order]
    starts = torch.ones(T, dtype=torch.bool, device="cuda")
    starts[1:] = st[1:] != st[:-1]
    run_id = torch.cumsum(starts.long(), 0) - 1
    first = torch.nonzero(starts).squeeze(1)
    j = torch.arange(T, device="cuda") - first[run_id]
    nrun, L = int(first.numel()), int(j.max()) + 1
    idx = torch.full((nrun, L), -1, dtype=torch.long, device="cuda")
    idx[run_id, j] = order
    rows = dout.float()
    acc = torch.zeros(nrun, D, device="cuda")
    for k in range(L):
        col = idx[:, k]
        valid = (col >= 0).unsqueeze(1)
        acc = torch.where(valid, rnd(acc + rows[col.clamp_min(0)]), acc)
    out = torch.zeros(V, D, device="cuda")
    out[st[first]] = acc
    return out


def bf16_seq_case(D, T, V, tok=None, storage=BF16S, time_it=False, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(T)
    tok = zipf_tokens(T, V, gen) if tok is None else tok
    dt = torch.bfloat16 if storage == BF16S else torch.float32
    dout = (scale * torch.randn(T, D, generator=gen, device="cuda")).to(dt)
    dtable = torch.zeros(V, D, device="cuda")
    run_bwd(storage, dout, tok, dtable, T, D, V, 1)
    ref = seq_bf16_ref(dout, tok, V, D, rb if storage == BF16S else (lambda x: x))
    eq = torch.equal(dtable.view(torch.int32), ref.view(torch.int32))
    ids, ref = compact_ref_f64(dout, tok, D)
    r = rel(dtable[ids], ref)
    longest = int(torch.unique(tok, return_counts=True)[1].max())
    msg = f"    D={D} T={T} V={V} longest run {longest}: bitwise equal to the sequential sum: {eq}; rel(seq, f64) {r:.2e}"
    if time_it:   # report only (the vis GPU is shared)
        ws, nb = workspace(T, V)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ms = []
        for _ in range(10):
            ev[0].record()
            lib.nm_embed_bwd_dt(storage, ptr(dout), ptr(tok), ptr(dtable), T, D, V, 1, None, ptr(ws), nb, None)
            ev[1].record()
            torch.cuda.synchronize()
            ms.append(ev[0].elapsed_time(ev[1]))
        msg += f"; {sorted(ms)[len(ms) // 2]:.3f} ms per call (median of 10, sort included)"
    print(msg)
    R.check(eq, "Bf16Sequential differs from the ascending-order running sum")


@R.case
def bf16_sequential_small():
    bf16_seq_case(256, 3000, 50)


@R.case
def bf16_sequential_vocab():
    bf16_seq_case(512, 16384, V_FULL)


@R.case
def bf16_sequential_130m_microbatch():
    """T = 131,072, D = 512, Zipf tokens (runs up to ~12,000): the strict preset's embedding backward of one 130m
    microbatch per GPU; f32 storage at D = 520 (a partial 64-column slice)."""
    bf16_seq_case(512, 131072, V_FULL, time_it=True)
    bf16_seq_case(520, 8192, 3000, storage=F32S)


@R.case
def bf16_sequential_chunk_edges():
    """The sequential path's own chunk logic (binary searches for the first own entry and the last run's end)."""
    D, V = 192, 1000
    cases = {
        "T=1": torch.tensor([7], dtype=torch.int32),
        "T=257": torch.arange(257, dtype=torch.int32) % 2,
        "one run over 9 chunks": torch.full((2300,), 42, dtype=torch.int32),
        "run ends on chunk edge": torch.cat([torch.full((512,), 3), torch.full((300,), 4), torch.arange(100) + 10]),
        "run starts on chunk edge": torch.cat([torch.arange(256) + 500, torch.full((700,), 1)]),
        "through chunks then short": torch.cat([torch.full((256 * 3 + 1,), 9), torch.full((1,), 8),
                                                torch.full((511,), 9)]),
        "max token id": torch.cat([torch.full((300,), V - 1), torch.zeros(300)]),
    }
    for name, tok in cases.items():
        tok = tok.int()
        tok = tok[torch.randperm(len(tok))].cuda() if "edge" not in name else tok.cuda()
        print(f"  {name}")
        bf16_seq_case(D, len(tok), V, tok=tok)
    # the packed bf16 add must keep bf16 subnormals (|x| < 2^-126) exactly as the f32 add then RNE does
    print("  subnormal and tiny values (scale 1e-38)")
    bf16_seq_case(D, 3000, 50, scale=1e-38)


@R.case
def workspace_too_small():
    T, D, V = 1024, 512, 1000
    tok = torch.zeros(T, dtype=torch.int32, device="cuda")
    dout = torch.zeros(T, D, dtype=torch.bfloat16, device="cuda")
    dtable = torch.zeros(V, D, device="cuda")
    ws = torch.empty(4 * 4096 + 1024, dtype=torch.uint8, device="cuda")
    try:
        lib.nm_embed_bwd(ptr(dout), ptr(tok), ptr(dtable), T, D, V, 0, None, ptr(ws), ws.numel(), None)
        R.check(False, "a too-small workspace was accepted")
    except RuntimeError as e:
        R.check("workspace" in str(e), f"unexpected message: {e}")


R.run(__import__("nmk").ARGS.cases)
