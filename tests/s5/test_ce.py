"""S5 gates of the fused linear CE (DESIGN.md §3.8, BUILD_LOG "Decisions for S4-S6" items 10-13) on random inputs:

  grad_V<V>          implementations A, B1, B1.5, B2 (bf16 store; cuBLASLt and own plain-dz GEMMs) at h = 512 / 768 /
                     1024 / 2048 against f64 torch autograd: per-token loss and lse within 2e-5 nats (T1.5's loss bar),
                     dX and dW by the oracle rule against the levanter emulation (numerics.md §5.2) and within 1.05x of
                     the emulation of the same dX / f32 dW policy, weight-0 rows exactly 0, loss_partial (the per-chunk
                     weighted loss sum behind train/loss) bitwise equal to an emulation of its reduction
  f32_store          the exec.ce_logits=f32 fallback for B1, B1.5, B2 (oracle rule against the f32-store emulation)
  bitwise            determinism; loss, lse and dX independent of the chunk size (one numeric path for every microbatch
                     size); B1 / B1.5 / B2 give the same dX and dW bits with the own GEMMs; forward-only = the loss of
                     the gradient call
  inplace_dx         dx == x (readouts write d(norm out) over the norm output) equals a separate dX, every stage
  accumulate_dx      dx += : levanter bf16(old + f32(bf16(acc))) bitwise; accurate bf16(old + acc) within 1 bf16 ulp
                     of it, different from it, strictly closer to f64, and the same bits from B1, B1.5, B2 on both GEMM
                     backends
  strict_dw          CeDwRound::PerChunkBf16 + ce_finish_microbatch over two microbatches, bitwise against the same
                     rounding applied to per-chunk f32 dW
  f32_io             implementation A under the f32 policy (f32 x, W, dX, dW; true f32 GEMMs), with a logit bias, in a
                     workspace of exactly nm_ce_workspace_bytes followed by a guard band that must stay untouched
  odd_shapes         V not a multiple of 8 or 128, D not a multiple of 128, T below one chunk, a 1-row tail chunk,
                     chunks of 1022 rows
  readout            the multi-head entry point (norm + CE + norm backward, dW in place) equals the composed C ABI calls
                     bitwise: full and partial visible width, gain-only (no backbone gradient), forward-only
  grad_scale         mult / sum(w) in f32; the spec's 128 x 4096 weights (last position 0) give f32(1 / 524160)
  two_devices        the same call on GPU 0 and GPU 1 of one process gives the same bits (per-device cuBLASLt handles
                     and plans, per-device kernel attributes; the trainer runs one thread per GPU); needs 2 GPUs
  fast_stats_exp     the opt-in ex2.approx statistics exponential (off by default; numerics.md §1 A/B): loss, lse, dX,
                     dW within the bars of the grad cases, and its distance from the accurate path
"""
import math

import nmce
from nmce import torch, rel, ce_call, make_case, oracle_f64, levanter_emulation, IMPLS, LEV, ACC, BF16S, F32S
import nmk

lib, R = nmce.init("s5_ce")
LOSS_TOL = 2e-5          # kaiyue_baseline.md T1.5: per-token loss max abs diff (nats)
ORACLE = 1.25            # DESIGN.md §3.0 oracle rule: rel(nano, f64) <= 1.25 rel(levanter, f64)
TIGHT = 1.05             # regression bar: rel(nano, f64) <= 1.05 rel(emulation of the same rounding policy, f64)


def gscale_for(w):
    return torch.tensor([1.0 / float(w.sum().item())], dtype=torch.float32, device="cuda")


def set_backends(own_dgrad, own_wgrad, sm90=-1):
    """sm90: -1 keeps the library's choice of sm_90a kernels (bit 0 logits, bit 1 GEMMs)."""
    return lib.nm_ce_set_backends(int(own_dgrad), int(own_wgrad), int(sm90))


def run(impl, x, W, tg, w, gs, **kw):
    V, D = W.shape
    fwd = kw.get("forward_only", False)
    dx = kw.pop("dx", None)
    if dx is None and not fwd:
        dx = torch.empty_like(x)
    dw = kw.pop("dw", None)
    if dw is None and not fwd:
        dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
    chunk = kw.get("chunk", 1024)
    lp = torch.empty((x.shape[0] + chunk - 1) // chunk, dtype=torch.float32, device="cuda")
    loss, lse = ce_call(lib, impl, x, W, tg, w, gs, dx=dx, dw=dw, loss_partial=lp, **kw)
    torch.cuda.synchronize()
    return loss, lse, dx, dw, lp


def check_partial(name, got, w, chunk):
    """loss_partial[c] (train/loss) against the emulation of k_ce_chunk_loss (bitwise) and the f64 per-chunk sum of
    w x loss_tok (rel <= 1e-6)."""
    loss, lp = got[0], got[4].cpu()
    ref = nmce.chunk_loss_ref(loss, w, chunk)
    f64 = torch.tensor([float((w[r0:r0 + chunk].double() * loss[r0:r0 + chunk].double()).sum().item())
                        for r0 in range(0, loss.numel(), chunk)], dtype=torch.float64)
    rr = ((lp.double() - f64).abs() / f64.abs().clamp_min(1e-30)).max().item()
    R.check(torch.equal(lp, ref), f"{name}: loss_partial {lp.tolist()} != the emulated reduction {ref.tolist()}")
    R.check(rr <= 1e-6, f"{name}: loss_partial rel {rr:.2e} to the f64 per-chunk sum")
    return rr


def check_against(name, got, orc, emu, w, emu_label="levanter emulation"):
    loss, lse, dx, dw = got[:4]
    oloss, olse, odx, odw = orc
    eloss, else_, edx, edw = emu[:4]
    on = w != 0
    dl = (loss.double() - oloss)[on].abs().max().item()
    dlse = (lse.double() - olse).abs().max().item()
    el = (eloss.double() - oloss)[on].abs().max().item()
    else_ = (else_.double() - olse).abs().max().item()
    # T1.5's 2e-5 nats is the bar against levanter on the fixture; on these N(0, 9) logits a bf16 tensor-core GEMM
    # alone moves a logit by up to ~5e-5 at h = 2048, so the bar is the oracle rule against the emulation's error
    R.check(dl <= max(LOSS_TOL, ORACLE * el), f"{name}: loss max abs diff {dl:.3e} > max(2e-5, {ORACLE} x {el:.3e})")
    R.check(dlse <= max(LOSS_TOL, ORACLE * else_), f"{name}: lse max abs diff {dlse:.3e} > max(2e-5, {ORACLE} x "
                                                   f"{else_:.3e})")
    R.check(bool((loss[~on] == 0).all()), f"{name}: loss of a weight-0 position is not 0")
    msg = f"    {name}: loss {dl:.2e} (emu {el:.2e}) lse {dlse:.2e}"
    if dx is not None:
        r_k, r_e = rel(dx, odx), rel(edx, odx)
        msg += f"  dx rel {r_k:.3e} (emu {r_e:.3e}, x{r_k / r_e:.3f})"
        R.check(r_k <= ORACLE * r_e, f"{name}: dx rel {r_k:.3e} > {ORACLE} x {emu_label} {r_e:.3e}")
        R.check(r_k <= TIGHT * r_e, f"{name}: dx rel {r_k:.3e} > {TIGHT} x {emu_label} {r_e:.3e}")
        R.check(bool((dx[~on].float() == 0).all()), f"{name}: dx of a weight-0 position is not 0")
    if dw is not None:
        r_k, r_e = rel(dw, odw), rel(edw, odw)
        msg += f"  dW rel {r_k:.3e} (emu {r_e:.3e}, x{r_k / r_e:.3f})"
        R.check(r_k <= ORACLE * r_e, f"{name}: dW rel {r_k:.3e} > {ORACLE} x {emu_label} {r_e:.3e}")
        if len(emu) > 4:   # the f32 dW of the emulation's dz: the same rounding policy as the port's dW
            r_32 = rel(emu[4], odw)
            msg += f" (f32-dW emu x{r_k / r_32:.3f})"
            R.check(r_k <= TIGHT * r_32, f"{name}: dW rel {r_k:.3e} > {TIGHT} x f32-dW {emu_label} {r_32:.3e}")
    print(msg, flush=True)


def grad_case(V, Ds=(512, 768, 1024, 2048), T=2 * 1024 + 77, chunk=1024):
    for D in Ds:
        x, W, tg, w = make_case(T, D, V, seed=V + D)
        gs = gscale_for(w)
        orc = oracle_f64(x, W, tg, w, gs.item())
        emu = levanter_emulation(x, W, tg, w, gs.item(), f32_dw=True)
        for impl, own in (("A", 0), ("B1", 0), ("B1", 1), ("B1.5", 0), ("B1.5", 1), ("B2", 1)):
            set_backends(own, own)
            got = run(impl, x, W, tg, w, gs, chunk=chunk)
            check_against(f"V{V} D{D} {impl}{'/own' if own else '/lt'}", got, orc, emu, w)
            check_partial(f"V{V} D{D} {impl}{'/own' if own else '/lt'}", got, w, chunk)
            del got
        del orc, emu, x, W
        torch.cuda.empty_cache()
    set_backends(0, 0)


@R.case
def grad_V512():
    grad_case(512)


@R.case
def grad_V8256():
    grad_case(8256)


@R.case
def grad_V64256():
    grad_case(64256)


@R.case
def grad_V128256():
    grad_case(128256)


@R.case
def f32_store():
    for V, D in ((8256, 512), (128256, 768)):
        T = 1500
        x, W, tg, w = make_case(T, D, V, seed=11 + D)
        gs = gscale_for(w)
        orc = oracle_f64(x, W, tg, w, gs.item())
        emu = levanter_emulation(x, W, tg, w, gs.item(), store_bf16=False, f32_dw=True)
        for impl in ("A", "B1", "B1.5", "B2"):
            for own in (0, 1):
                set_backends(own, own)
                got = run(impl, x, W, tg, w, gs, chunk=512, store=F32S)
                check_against(f"V{V} D{D} {impl} f32 store{'/own' if own else '/lt'}", got, orc, emu, w,
                              "f32-store emulation")
    set_backends(0, 0)


def same_bits(name, a, b):
    eq = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    R.check(eq, f"{name}: not bitwise equal")
    return eq


@R.case
def bitwise():
    for V, D, T in ((8256, 512, 3000), (128256, 768, 2100)):
        x, W, tg, w = make_case(T, D, V, seed=5)
        gs = gscale_for(w)
        for own in (0, 1):
            set_backends(own, own)
            ref = {}
            for impl in ("A", "B1", "B1.5", "B2"):
                if impl in ("B2",) and not own:
                    continue
                a = run(impl, x, W, tg, w, gs, chunk=1024)
                b = run(impl, x, W, tg, w, gs, chunk=1024)
                tag = f"V{V} D{D} {impl}{'/own' if own else '/lt'}"
                for i, n in enumerate(("loss", "lse", "dx", "dw", "loss_partial")):
                    same_bits(f"{tag} repeat {n}", a[i], b[i])
                check_partial(tag, a, w, 1024)
                for ch in (128, 512, 4096):
                    c = run(impl, x, W, tg, w, gs, chunk=ch)
                    for i, n in enumerate(("loss", "lse", "dx")):
                        same_bits(f"{tag} chunk {ch} vs 1024 {n}", c[i], a[i])
                    check_partial(f"{tag} chunk {ch}", c, w, ch)
                    rd = rel(c[3], a[3])
                    R.check(rd < 1e-5, f"{tag} chunk {ch}: dW rel {rd:.2e} (only the summation order may differ)")
                f = run(impl, x, W, tg, w, gs, chunk=1024, forward_only=True)
                same_bits(f"{tag} forward-only loss", f[0], a[0])
                same_bits(f"{tag} forward-only lse", f[1], a[1])
                same_bits(f"{tag} forward-only loss_partial", f[4], a[4])
                ref[impl] = a
            if own:
                for impl in ("B1.5", "B2"):
                    for i, n in ((0, "loss"), (1, "lse"), (2, "dx"), (3, "dw"), (4, "loss_partial")):
                        same_bits(f"V{V} D{D} {impl} vs B1 (own GEMMs) {n}", ref[impl][i], ref["B1"][i])
            print(f"    V{V} D{D} {'own' if own else 'lt'}: repeat, chunk 128/512/1024/4096, forward-only, loss_partial"
                  f"{', B1 = B1.5 = B2' if own else ''}: bitwise", flush=True)
    set_backends(0, 0)


@R.case
def inplace_dx():
    V, D, T = 8256, 768, 2500
    x, W, tg, w = make_case(T, D, V, seed=6)
    gs = gscale_for(w)
    for own in (0, 1):
        set_backends(own, own)
        for impl in ("A", "B1", "B1.5", "B2"):
            a = run(impl, x, W, tg, w, gs, chunk=1024)
            xi = x.clone()
            dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
            ce_call(lib, impl, xi, W, tg, w, gs, chunk=1024, dx=xi, dw=dw)
            torch.cuda.synchronize()
            same_bits(f"{impl}{'/own' if own else '/lt'} in-place dx", xi, a[2])
            same_bits(f"{impl}{'/own' if own else '/lt'} in-place dW", dw, a[3])
    print("    dx == x equals a separate dX (A, B1, B1.5, B2; cuBLASLt and own GEMMs)")
    set_backends(0, 0)


@R.case
def accumulate_dx():
    V, D, T = 8256, 1024, 2300
    x, W, tg, w = make_case(T, D, V, seed=7)
    gs = gscale_for(w)
    orc = oracle_f64(x, W, tg, w, gs.item())
    g = torch.Generator(device="cuda").manual_seed(70)
    old = (torch.randn(T, D, generator=g, device="cuda") * 3e-4).to(torch.bfloat16)
    want = orc[2] + old.double()
    acc_bits = {}
    for own in (0, 1):
        set_backends(own, own)
        for impl in ("A", "B1", "B1.5", "B2"):
            plain = run(impl, x, W, tg, w, gs, chunk=1024)[2]
            lev_ref = (old.float() + plain.float()).to(torch.bfloat16)
            tag = f"{impl}{'/own' if own else '/lt'}"
            dl = old.clone()
            run(impl, x, W, tg, w, gs, chunk=1024, dx=dl, acc_dx=True, rounding=LEV)
            same_bits(f"{tag} levanter accumulate", dl, lev_ref)
            da = old.clone()
            run(impl, x, W, tg, w, gs, chunk=1024, dx=da, acc_dx=True, rounding=ACC)
            # bf16(old + acc) vs bf16(old + bf16(acc)): apart by at most one bf16 ulp of the largest of old, acc, sum
            big = torch.maximum(torch.maximum(old.float().abs(), plain.float().abs()), lev_ref.float().abs())
            ulp_big = torch.exp2(torch.floor(torch.log2(big.clamp_min(1e-30))) - 7)
            excess = ((da.float() - lev_ref.float()).abs() / ulp_big).max().item()
            R.check(excess <= 1.0, f"{tag} accurate accumulate: {excess:.2f} operand ulps from the levanter form")
            # the two forms must differ: a result equal to bf16(old + bf16(acc)) everywhere is the levanter form
            frac = (da.view(torch.int16) != lev_ref.view(torch.int16)).double().mean().item()
            R.check(frac > 1e-3, f"{tag} accurate accumulate differs from the levanter form in only {frac:.2e} of the "
                                 f"elements")
            ra, rl = rel(da, want), rel(dl, want)
            R.check(ra < rl, f"{tag} accurate accumulate rel {ra:.3e} not below levanter {rl:.3e}")
            if impl != "A":   # B1, B1.5, B2 share the dz bits and the own dgrad K order
                acc_bits[tag] = da
            print(f"    {tag}: levanter bitwise; accurate within {excess:.2f} operand ulp, differs in {100 * frac:.2f}% "
                  f"of the elements, rel {ra:.3e} vs levanter {rl:.3e}")
    ref_tag = "B1/own"
    for tag, da in acc_bits.items():
        if tag != ref_tag:
            same_bits(f"{tag} vs {ref_tag} accurate accumulate", da, acc_bits[ref_tag])
    set_backends(0, 0)


@R.case
def strict_dw():
    V, D, chunk = 8256, 512, 1024
    for own in (0, 1):
        set_backends(own, 1)   # the strict path always runs the own wgrad; compare with own per-chunk f32 dW
        for impl in ("A", "B1", "B1.5", "B2"):
            dw = (torch.randn(V, D, device="cuda") * 1e-3).float()
            dw0 = dw.clone()
            mb = torch.zeros(V, D, dtype=torch.float32, device="cuda")
            want = dw0.clone()
            for micro in range(2):
                T = 2 * chunk + 300
                x, W, tg, w = make_case(T, D, V, seed=80 + micro)
                gs = gscale_for(w)
                ce_call(lib, impl, x, W, tg, w, gs, chunk=chunk, dx=torch.empty_like(x), dw=dw, dw_mb=mb, dw_round=1)
                lib.nm_ce_finish_microbatch(nmce.ptr(dw), nmce.ptr(mb), V * D, None)
                acc = torch.zeros(V, D, dtype=torch.float32, device="cuda")
                for r0 in range(0, T, chunk):
                    r1 = min(T, r0 + chunk)
                    dwc = torch.zeros(V, D, dtype=torch.float32, device="cuda")
                    ce_call(lib, impl, x[r0:r1].contiguous(), W, tg[r0:r1].contiguous(), w[r0:r1].contiguous(), gs,
                            chunk=chunk, dw=dwc)
                    acc += dwc.to(torch.bfloat16).float()
                want = want + acc.to(torch.bfloat16).float()
            torch.cuda.synchronize()
            same_bits(f"{impl} strict dW over two microbatches", dw, want)
            R.check(bool((mb == 0).all()), f"{impl}: the microbatch accumulator is not cleared")
    print("    CeDwRound::PerChunkBf16 + finish_microbatch: bitwise against the per-chunk rounding of f32 chunk dW")
    set_backends(0, 0)


@R.case
def f32_io():
    V, D, T = 8256, 512, 1700
    x, W, tg, w = make_case(T, D, V, seed=9, dtype=torch.float32)
    gs = gscale_for(w)
    g = torch.Generator(device="cuda").manual_seed(90)
    bias = torch.randn(V, generator=g, device="cuda")
    chunk = 512
    nb = lib.nm_ce_workspace_bytes(IMPLS["A"], T, D, V, chunk, F32S)
    guard = 4 * chunk * D   # larger than the f32 dX scratch the f32-io layout once placed past the reported size
    for b in (None, bias):
        orc = oracle_f64(x, W, tg, w, gs.item(), bias=b)
        dx = torch.empty_like(x)
        dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
        wsb = torch.full((nb + guard,), 0xA5, dtype=torch.uint8, device="cuda")
        loss, lse = ce_call(lib, "A", x, W, tg, w, gs, chunk=chunk, dx=dx, dw=dw, bias=b, store=F32S, ws=wsb[:nb])
        torch.cuda.synchronize()
        R.check(bool((wsb[nb:] == 0xA5).all()), "f32 io: the call wrote past nm_ce_workspace_bytes")
        on = w != 0
        dl = (loss.double() - orc[0])[on].abs().max().item()
        rx, rw = rel(dx, orc[2]), rel(dw, orc[3])
        print(f"    A f32 io{' + bias' if b is not None else ''}: loss {dl:.2e}  dx rel {rx:.2e}  dW rel {rw:.2e}")
        R.check(dl <= 1e-5, f"f32 io loss {dl:.3e}")
        R.check(rx <= 1e-5 and rw <= 1e-5, f"f32 io: dx rel {rx:.3e}, dW rel {rw:.3e} > 1e-5")
    # bf16 io with a bias (A only)
    xb, Wb = x.to(torch.bfloat16), W.to(torch.bfloat16)
    orc = oracle_f64(xb, Wb, tg, w, gs.item(), bias=bias)
    loss, lse, dx, dw, _ = run("A", xb, Wb, tg, w, gs, chunk=512, bias=bias)
    dl = (loss.double() - orc[0])[w != 0].abs().max().item()
    R.check(dl <= LOSS_TOL, f"A bf16 + bias loss {dl:.3e}")
    print(f"    A bf16 + bias: loss {dl:.2e}  dx rel {rel(dx, orc[2]):.2e}  dW rel {rel(dw, orc[3]):.2e}")


@R.case
def odd_shapes():
    # chunk 1022: chunk offsets not a multiple of 4 rows (unaligned lse / targets: the sm_90a GEMMs fall back)
    for V, D, T, chunk in ((1000, 520, 700, 256), (8257, 136, 1025, 1024), (130, 64, 1, 128), (128256, 512, 513, 512),
                           (1000, 512, 2100, 1022)):
        x, W, tg, w = make_case(T, D, V, seed=V + T, zero_every=5 if T > 1 else 0)
        if T == 1:
            w[0] = 1.0
        gs = gscale_for(w)
        orc = oracle_f64(x, W, tg, w, gs.item())
        emu = levanter_emulation(x, W, tg, w, gs.item())
        worst = 0.0
        for own in (0, 1):
            set_backends(own, own)
            for impl in ("A", "B1", "B1.5", "B2"):
                for store in (BF16S, F32S):
                    got = run(impl, x, W, tg, w, gs, chunk=chunk, store=store)
                    oloss, olse = orc[0], orc[1]
                    dl = (got[0].double() - oloss)[w != 0].abs().max().item()
                    dlse = (got[1].double() - olse).abs().max().item()
                    rx, rw = rel(got[2], orc[2]), rel(got[3], orc[3])
                    ex, ew = rel(emu[2], orc[2]), rel(emu[3], orc[3])
                    tag = f"V{V} D{D} T{T} chunk {chunk} {impl} {'bf16' if store else 'f32'} store {'own' if own else 'lt'}"
                    R.check(dl <= LOSS_TOL and dlse <= LOSS_TOL, f"{tag}: loss {dl:.3e} lse {dlse:.3e}")
                    R.check(rx <= ORACLE * ex, f"{tag}: dx rel {rx:.3e} > {ORACLE} x emu {ex:.3e}")
                    R.check(rw <= ORACLE * ew, f"{tag}: dW rel {rw:.3e} > {ORACLE} x emu {ew:.3e}")
                    worst = max(worst, rx / max(ex, 1e-30), rw / max(ew, 1e-30))
                    check_partial(tag, got, w, chunk)
            print(f"    V{V} D{D} T{T} chunk {chunk} ({'own' if own else 'lt'}): every stage and store within bars "
                  f"(worst dX / dW error x{worst:.3f} the emulation's)")
    set_backends(0, 0)


def rmsnorm_ref(state, gain, rounding, eps):
    T, D = state.shape
    y = torch.empty_like(state)
    rstd = torch.empty(T, dtype=torch.float32, device="cuda")
    lib.nm_rmsnorm_fwd(1, rounding, nmce.ptr(state), nmce.ptr(gain), nmce.ptr(y), nmce.ptr(rstd), T, D, eps, None)
    return y, rstd


@R.case
def readout():
    V, D, T, chunk, eps = 8256, 512, 2100, 1024, float(nmk.EPS)
    g = torch.Generator(device="cuda").manual_seed(12)
    state = (torch.randn(T, D, generator=g, device="cuda") * 2).to(torch.bfloat16)
    gain = (1.0 + 0.2 * torch.randn(D, generator=g, device="cuda")).to(torch.bfloat16)
    _, W, tg, w = make_case(T, D, V, seed=13)
    gs = torch.tensor([0.37 / float(w.sum().item())], dtype=torch.float32, device="cuda")
    P = lib.nm_norm_partial_rows(T)
    for impl in ("A", "B1", "B1.5", "B2"):
        for rounding in (LEV, ACC):
            for vis, with_state, with_gain, fwd_only in ((D, 1, 1, 0), (D // 2 + 64, 1, 1, 0), (D, 0, 1, 0),
                                                          (D, 1, 0, 0), (D, 0, 0, 0), (D, 1, 1, 1)):
                tag = f"{impl} {'lev' if rounding == LEV else 'acc'} vis {vis} d_state {with_state} gain {with_gain}" \
                      f"{' fwd-only' if fwd_only else ''}"
                # composed reference
                y, rstd = rmsnorm_ref(state, gain, rounding, eps)
                if vis < D:
                    y[:, vis:] = 0
                dyr = torch.empty_like(y)
                dwr = torch.zeros(V, D, dtype=torch.float32, device="cuda")
                lr, lser = ce_call(lib, impl, y, W, tg, w, gs, chunk=chunk, dx=None if fwd_only else dyr,
                                   dw=None if fwd_only else dwr, forward_only=bool(fwd_only), rounding=rounding)
                d0 = (torch.randn(T, D, generator=g, device="cuda") * 1e-3).to(torch.bfloat16)
                dsr = d0.clone()
                pr = torch.zeros(P, D, dtype=torch.float32, device="cuda")
                if not fwd_only:
                    if vis < D:
                        dyr[:, vis:] = 0
                    scratch = torch.empty_like(state)
                    lib.nm_rmsnorm_bwd_add_partial(1, rounding, nmce.ptr(dyr), nmce.ptr(state), nmce.ptr(gain),
                                                   nmce.ptr(rstd), nmce.ptr(dsr), nmce.ptr(dsr if with_state else scratch),
                                                   nmce.ptr(pr), T, D, eps, None)
                # the entry point
                nout = torch.empty_like(state)
                nr = torch.empty(T, dtype=torch.float32, device="cuda")
                dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
                ds = d0.clone()
                pp = torch.zeros(P, D, dtype=torch.float32, device="cuda")
                loss = torch.empty(T, dtype=torch.float32, device="cuda")
                lse = torch.empty(T, dtype=torch.float32, device="cuda")
                ws = nmce.Ws.get(lib.nm_ce_workspace_bytes(IMPLS[impl], T, D, V, chunk, BF16S))
                lib.nm_ce_readout(IMPLS[impl], rounding, BF16S, nmce.ptr(state), nmce.ptr(gain), nmce.ptr(nr),
                                  nmce.ptr(nout), nmce.ptr(W), nmce.ptr(dw), nmce.ptr(pp) if with_gain else None,
                                  nmce.ptr(gs), nmce.ptr(ds) if with_state else None, vis, nmce.ptr(tg), nmce.ptr(w),
                                  nmce.ptr(loss), nmce.ptr(lse), None, fwd_only, T, D, V, chunk, eps, nmce.ptr(ws),
                                  None)
                torch.cuda.synchronize()
                same_bits(f"{tag} loss", loss, lr)
                same_bits(f"{tag} lse", lse, lser)
                if not fwd_only:
                    same_bits(f"{tag} dW", dw, dwr)
                    if with_state:
                        same_bits(f"{tag} d_state", ds, dsr)
                    else:
                        same_bits(f"{tag} d_state untouched", ds, d0)
                    if with_gain:
                        same_bits(f"{tag} gain partials", pp, pr)
                    if not with_state and not with_gain:   # no dY needed: the norm output was not overwritten
                        same_bits(f"{tag} norm output kept", nout, y)
        print(f"    {impl}: readout = composed calls bitwise (both roundings, partial width, gain-only, no backbone, "
              f"forward-only)", flush=True)
    # one f64 sanity check of the composed math (impl B1, levanter): d_state, dgain, dW
    st = state.double().requires_grad_(True)
    gn = gain.double().requires_grad_(True)
    W64 = W.double().requires_grad_(True)
    yy = st * torch.rsqrt((st * st).mean(dim=1, keepdim=True) + eps) * gn
    ce = torch.nn.functional.cross_entropy(yy @ W64.T, tg.long(), reduction="none")
    (ce * w.double() * gs.item()).sum().backward()
    nout = torch.empty_like(state)
    nr = torch.empty(T, dtype=torch.float32, device="cuda")
    dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
    ds = torch.zeros_like(state)
    pp = torch.zeros(P, D, dtype=torch.float32, device="cuda")
    loss = torch.empty(T, dtype=torch.float32, device="cuda")
    ws = nmce.Ws.get(lib.nm_ce_workspace_bytes(1, T, D, V, chunk, BF16S))
    lib.nm_ce_readout(1, LEV, BF16S, nmce.ptr(state), nmce.ptr(gain), nmce.ptr(nr), nmce.ptr(nout), nmce.ptr(W),
                      nmce.ptr(dw), nmce.ptr(pp), nmce.ptr(gs), nmce.ptr(ds), D, nmce.ptr(tg), nmce.ptr(w),
                      nmce.ptr(loss), None, None, 0, T, D, V, chunk, eps, nmce.ptr(ws), None)
    torch.cuda.synchronize()
    r_s, r_g, r_w = rel(ds, st.grad), rel(pp.sum(0), gn.grad), rel(dw, W64.grad)
    print(f"    f64 autograd: d_state rel {r_s:.2e}  dgain rel {r_g:.2e}  dW rel {r_w:.2e}")
    R.check(r_s < 2e-2 and r_g < 2e-2 and r_w < 2e-2, "readout against f64 autograd")


@R.case
def sm90_vs_portable():
    """H100 only: the sm_90a kernels against the portable ones on the same inputs (logits store bits, loss, lse, dX,
    dW). The two GEMMs sum in different orders, so a few store elements differ by one bf16 ulp; everything else must
    agree to the rounding of those flips."""
    if lib.nm_ce_device_arch() != 90:
        raise nmce.Skip("not an sm_90 device")
    default = (set_backends(0, 0) >> 2) & 3
    for V, D, T in ((128256, 512, 3000), (128256, 2048, 1500), (8256, 768, 4100), (1000, 512, 700)):
        x, W, tg, w = make_case(T, D, V, seed=21)
        gs = gscale_for(w)
        res = {}
        for sm90 in (default, 0):
            for impl, own in (("B1", 0), ("B2", 1)):
                set_backends(own, own, sm90)
                res[(sm90, impl)] = run(impl, x, W, tg, w, gs, chunk=1024)
            # the store survives a B2 call without dW (dgrad reads it; nothing rewrites it)
            set_backends(1, 1, sm90)
            ce_call(lib, "B2", x, W, tg, w, gs, chunk=4096, dx=torch.empty_like(x), dw=None)
            torch.cuda.synchronize()
            ld = (V + 7) // 8 * 8
            res[(sm90, "store")] = nmce.Ws.buf[: 4096 * ld * 2].view(torch.bfloat16).view(4096, ld)[:T, :V].clone()
        flips = (res[(default, "store")].view(torch.int16) != res[(0, "store")].view(torch.int16)).double().mean().item()
        ulp = nmk.ulp_diff_bf16(res[(default, "store")], res[(0, "store")]).max().item()
        R.check(ulp <= 1 and flips < 1e-3, f"V{V} D{D}: store differs by {ulp} ulp in {flips:.2e} of the elements")
        for impl in ("B1", "B2"):
            a, b = res[(default, impl)], res[(0, impl)]
            dl = (a[0] - b[0]).abs().max().item()
            dlse = (a[1] - b[1]).abs().max().item()
            rx, rw = rel(a[2], b[2]), rel(a[3], b[3])
            R.check(dl < 1e-4 and dlse < 1e-4, f"V{V} D{D} {impl}: loss {dl:.2e} lse {dlse:.2e} sm90 vs portable")
            R.check(rx < 2e-3 and rw < 2e-3, f"V{V} D{D} {impl}: dx rel {rx:.2e} dW rel {rw:.2e} sm90 vs portable")
            print(f"    V{V} D{D} {impl}: sm90 vs portable loss {dl:.1e} lse {dlse:.1e} dx rel {rx:.1e} dW rel {rw:.1e}"
                  f"  store: {flips:.1e} of elements differ (max {ulp} ulp)", flush=True)
    set_backends(0, 0, default)


@R.case
def sm90_gemm():
    """H100 only: the sm_90a dX / dW kernels (plain dz: B1, B1.5's wgrad; dz from the store: B2 and B1.5's dgrad,
    which also writes dz in place) against the f64 oracle and the portable kernels; repeat and chunk bitwise."""
    if lib.nm_ce_device_arch() != 90:
        raise nmce.Skip("not an sm_90 device")
    prev = (set_backends(0, 0) >> 2) & 3
    for V, D, T in ((8256, 512, 2100), (128256, 768, 1500), (128256, 1024, 1100), (64256, 2048, 1000)):
        x, W, tg, w = make_case(T, D, V, seed=31)
        gs = gscale_for(w)
        orc = oracle_f64(x, W, tg, w, gs.item())
        emu = levanter_emulation(x, W, tg, w, gs.item(), f32_dw=True)
        for impl in ("B1", "B1.5", "B2"):
            set_backends(1, 1, 3)
            got = run(impl, x, W, tg, w, gs, chunk=1024)
            check_against(f"V{V} D{D} {impl}/sm90 GEMMs", got, orc, emu, w)
            set_backends(1, 1, 1)
            ref = run(impl, x, W, tg, w, gs, chunk=1024)
            rx, rw = rel(got[2], ref[2]), rel(got[3], ref[3])
            R.check(rx < 1e-3 and rw < 1e-4, f"V{V} D{D} {impl}: sm90 GEMMs vs portable dx rel {rx:.2e} dW rel {rw:.2e}")
            print(f"    V{V} D{D} {impl}: sm90 GEMMs vs portable: dx rel {rx:.1e} dW rel {rw:.1e}", flush=True)
        # repeat and chunk independence with the sm90 GEMMs
        set_backends(1, 1, 3)
        a = run("B2", x, W, tg, w, gs, chunk=1024)
        b = run("B2", x, W, tg, w, gs, chunk=512)
        same_bits(f"V{V} D{D} B2 sm90 chunk 512 vs 1024 dx", b[2], a[2])
        c = run("B2", x, W, tg, w, gs, chunk=1024)
        same_bits(f"V{V} D{D} B2 sm90 repeat dW", c[3], a[3])
    set_backends(0, 0, prev)


@R.case
def fast_stats_exp():
    for V, D in ((128256, 512), (128256, 2048), (8256, 768)):
        T = 2100
        x, W, tg, w = make_case(T, D, V, seed=41)
        gs = gscale_for(w)
        orc = oracle_f64(x, W, tg, w, gs.item())
        emu = levanter_emulation(x, W, tg, w, gs.item())
        acc = run("B1", x, W, tg, w, gs, chunk=1024)
        prev = lib.nm_ce_set_fast_stats_exp(1)
        try:
            fast = run("B1", x, W, tg, w, gs, chunk=1024)
        finally:
            lib.nm_ce_set_fast_stats_exp(prev)
        check_against(f"V{V} D{D} B1 fast stats exp", fast, orc, emu, w)
        dlse = (fast[1].double() - acc[1].double()).abs().max().item()
        flips = (fast[2].view(torch.int16) != acc[2].view(torch.int16)).double().mean().item()
        print(f"    V{V} D{D}: fast vs accurate statistics exp: lse max |d| {dlse:.2e}, dx bits differ in "
              f"{100 * flips:.4f}% of the elements", flush=True)
        R.check(dlse < 2e-6, f"V{V} D{D}: fast statistics exp moves lse by {dlse:.2e}")


@R.case
def two_devices():
    if torch.cuda.device_count() < 2:
        raise nmce.Skip("needs 2 GPUs")
    V, D, T = 128256, 768, 2100
    out = []
    for dev in (0, 1):
        with torch.cuda.device(dev):
            x, W, tg, w = make_case(T, D, V, seed=51)
            gs = gscale_for(w)
            nmce.Ws.buf = None
            res = []
            for impl, own in (("B1", 0), ("B1", 1), ("B2", 1)):
                set_backends(own, own)
                res.append([t.cpu() for t in run(impl, x, W, tg, w, gs, chunk=1024)])
            out.append(res)
            nmce.Ws.buf = None
    for k, name in enumerate(("B1/lt", "B1/own", "B2/own")):
        for i, n in enumerate(("loss", "lse", "dx", "dw")):
            same_bits(f"{name} {n} GPU 0 vs GPU 1", out[0][k][i], out[1][k][i])
    print("    GPU 0 and GPU 1 give the same loss, lse, dX, dW bits (B1 cuBLASLt, B1 own, B2)")
    set_backends(0, 0)


@R.case
def grad_scale():
    g = torch.Generator(device="cuda").manual_seed(3)
    w = (torch.rand(524288, generator=g, device="cuda") < 0.9).float()
    out = torch.empty(1, dtype=torch.float32, device="cuda")
    lib.nm_ce_grad_scale(nmce.ptr(w), w.numel(), 1.0, nmce.ptr(out), None)
    torch.cuda.synchronize()
    want = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(w.sum().item()), dtype=torch.float32)
    R.check(out.item() == want.item(), f"grad_scale {out.item()!r} != f32(1 / sum) {want.item()!r}")
    print(f"    1 / sum(w) = {out.item():.9e} (sum {int(w.sum().item())})")
    # the spec's batch: 128 sequences x 4096 positions, the last position of each weight 0 -> 1 / (128 x 4095); two
    # microbatches of it -> exactly half
    for B in (128, 256):
        w = torch.ones(B, 4096, dtype=torch.float32, device="cuda")
        w[:, -1] = 0
        lib.nm_ce_grad_scale(nmce.ptr(w), w.numel(), 1.0, nmce.ptr(out), None)
        torch.cuda.synchronize()
        want = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(B * 4095), dtype=torch.float32)
        R.check(out.item() == want.item(), f"grad_scale {B} x 4096: {out.item()!r} != f32(1 / {B * 4095}) "
                                           f"{want.item()!r}")
        print(f"    {B} x 4096 (last position 0): 1 / {B * 4095} = {out.item():.9e}")
    R.check(torch.tensor(1 / 524160, dtype=torch.float32).item() == 2 * torch.tensor(1 / 1048320, dtype=torch.float32)
            .item(), "f32(1 / 524160) != 2 f32(1 / 1048320)")


if __name__ == "__main__":
    R.run(nmce.ARGS.cases)
