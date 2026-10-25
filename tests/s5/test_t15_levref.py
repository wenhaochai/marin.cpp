"""T1.5 (kaiyue_baseline.md): the fused CE against levanter's own fused CE on real tensors and the f64 oracle.

Fixtures: cache/levref/<size>/t1_units/ce (tools/levref/units.py: x = final-normed hidden states [32 x 4096, D] of one
GPU's rows of batch 0, W = init lm_head, labels, weights; levanter's per-token loss, grad_x, grad_w from the H100
batched_xla path under the cotangent 1/524160) and cache/levref/<size>/t1_oracle/ce (f64 oracle and levanter's own
error, rel.json). Bars of the T1.5 row:
  per-token loss    max |nano - levanter| <= 2e-5 nats over weight-1 positions; weight-0 positions exactly 0
  weighted mean     |nano - levanter| / levanter <= 1e-6
  grad_x, grad_w    oracle rule: rel(nano, f64) <= 1.25 x rel(levanter, f64)
Each implementation (A, B1, B1.5, B2), GEMM backend (cuBLASLt / own), chunk (8192 = levanter's tile, 16384, 32768),
logits store (bf16 = exec.ce_logits=auto, f32 = the fallback) and dW rounding (fast f32; strict per-chunk bf16 at 8192
rows, then bf16 once, as levanter's bf16 grad_w) is one row of the report; every row must pass.
"""
import json
import os

import numpy as np

import nmce
from nmce import torch, ce_call, Skip, IMPLS, LEV, BF16S, F32S

lib, R = nmce.init("s5_t15_levref")
DENOM = 524160


def fixture_dirs():
    root = os.path.join(nmce.ARGS.src, "cache", "levref")
    out = []
    if not os.path.isdir(root):
        return out
    for size in sorted(os.listdir(root)):
        u = os.path.join(root, size, "t1_units", "ce")
        o = os.path.join(root, size, "t1_oracle", "ce")
        if os.path.isfile(os.path.join(u, "index.json")) and os.path.isfile(os.path.join(o, "rel.json")):
            out.append((size, u, o))
    return out


def load(path, name, cuda=True):
    a = np.load(os.path.join(path, name), mmap_mode="r")
    if a.dtype == np.uint16:
        t = torch.from_numpy(np.array(a).view(np.int16)).view(torch.bfloat16)
    else:
        t = torch.from_numpy(np.array(a))
    return t.cuda() if cuda else t


def rel_chunked(a, orc_np, rows=8192):
    """||a - orc|| / ||orc|| in f64, a on the GPU [N, ...], orc a (memory-mapped) f64 numpy array of the same shape."""
    num = den = 0.0
    a2 = a.reshape(a.shape[0], -1)
    o2 = orc_np.reshape(orc_np.shape[0], -1)
    for r0 in range(0, a2.shape[0], rows):
        o = torch.from_numpy(np.array(o2[r0:r0 + rows])).cuda()
        d = a2[r0:r0 + rows].double() - o
        num += float((d * d).sum().item())
        den += float((o * o).sum().item())
    return (num / den) ** 0.5


def eq_frac(a, b, to_bf16=False, rows=8192):
    """Fraction of bitwise-equal elements, counted in row blocks (no full-size temporaries on a shared GPU)."""
    a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    n = 0
    for r0 in range(0, a2.shape[0], rows):
        x = a2[r0:r0 + rows]
        if to_bf16:
            x = x.to(torch.bfloat16).view(torch.int16)
            y = b2[r0:r0 + rows].view(torch.int16)
        else:
            y = b2[r0:r0 + rows]
        n += int(torch.count_nonzero(x == y).item())
    return n / a2.numel()


def one_size(size, upath, opath, configs):
    idx = json.load(open(os.path.join(upath, "index.json")))
    relj = json.load(open(os.path.join(opath, "rel.json")))
    res = json.load(open(os.path.join(upath, "results.json")))
    x = load(upath, "x.bf16.npy")
    D = x.shape[-1]
    x = x.reshape(-1, D).contiguous()
    W = load(upath, "W.bf16.npy").contiguous()
    V = W.shape[0]
    tg = load(upath, "labels.npy").reshape(-1).to(torch.int32).contiguous()
    w = load(upath, "loss_weight.npy").reshape(-1).float().contiguous()
    lev_loss32 = load(upath, "loss_tok.npy").reshape(-1)
    lev_loss = lev_loss32.double()
    lev_gx = load(upath, "grad_x.bf16.npy").reshape(-1, D)
    lev_gw = load(upath, "grad_w.bf16.npy")
    o_gx = np.load(os.path.join(opath, "grad_x.npy"), mmap_mode="r").reshape(-1, D)
    o_gw = np.load(os.path.join(opath, "grad_w.npy"), mmap_mode="r")
    o_loss = load(opath, "loss_tok.npy").reshape(-1)
    T = x.shape[0]
    on = w != 0
    lev_mean = float((lev_loss * w.double()).sum().item() / w.double().sum().item())
    r_lev_x, r_lev_w = relj["grad_x"]["rel_lev_oracle"], relj["grad_w"]["rel_lev_oracle"]
    r_lev_l = relj["loss_tok"]["rel_lev_oracle"]
    print(f"  {size}: T {T} D {D} V {V}; levanter rel to f64: loss {r_lev_l:.3e} grad_x {r_lev_x:.4e} grad_w "
          f"{r_lev_w:.4e}; bars x {1.25 * r_lev_x:.4e} w {1.25 * r_lev_w:.4e}  ({idx['part']}, "
          f"{res['device_kind']})", flush=True)
    gs = torch.tensor([np.float32(1.0) / np.float32(DENOM)], dtype=torch.float32, device="cuda")
    rows = []
    for cfg in configs:
        impl, own, chunk, store, strict = cfg[:5]
        fast = len(cfg) > 5 and cfg[5]
        lib.nm_ce_set_backends(own, own, -1)
        lib.nm_ce_set_fast_stats_exp(int(fast))
        dx = torch.empty_like(x)
        dw = torch.zeros(V, D, dtype=torch.float32, device="cuda")
        mb = torch.zeros(V, D, dtype=torch.float32, device="cuda") if strict else None
        loss, lse = ce_call(lib, impl, x, W, tg, w, gs, chunk=chunk, store=store, dx=dx, dw=dw, dw_mb=mb,
                            dw_round=1 if strict else 0, rounding=LEV)
        lib.nm_ce_set_fast_stats_exp(0)
        if strict:
            lib.nm_ce_finish_microbatch(nmce.ptr(dw), nmce.ptr(mb), V * D, None)
            del mb
            dw = dw.to(torch.bfloat16).float()   # levanter's grad_w is bf16
        torch.cuda.synchronize()
        dl = (loss.double() - lev_loss)[on].abs().max().item()
        zero_ok = bool((loss[~on] == 0).all())
        mean = float((loss.double() * w.double()).sum().item() / w.double().sum().item())
        dmean = abs(mean - lev_mean) / abs(lev_mean)
        r_l = (torch.linalg.vector_norm(loss.double() - o_loss) / torch.linalg.vector_norm(o_loss)).item()
        r_x = rel_chunked(dx, o_gx)
        r_w = rel_chunked(dw, o_gw)
        # report only: the fraction of elements bitwise equal to levanter's (loss f32, grad_x bf16, grad_w bf16)
        eq_l = eq_frac(loss.view(torch.int32), lev_loss32.view(torch.int32))
        eq_x = eq_frac(dx.view(torch.int16), lev_gx.view(torch.int16))
        eq_w = eq_frac(dw, lev_gw, to_bf16=True)
        ok = dl <= 2e-5 and zero_ok and dmean <= 1e-6 and r_x <= 1.25 * r_lev_x and r_w <= 1.25 * r_lev_w
        tag = f"{impl}/{'own' if own else 'lt'} chunk {chunk} {'bf16' if store else 'f32'} store" \
              f"{' strict dW' if strict else ''}{' fast stats exp' if fast else ''}"
        print(f"    {'ok  ' if ok else 'FAIL'} {tag:42s} loss max|d| {dl:.2e}  mean rel {dmean:.1e}  loss rel(f64) "
              f"{r_l:.2e}  grad_x rel {r_x:.4e} (x{r_x / r_lev_x:.3f})  grad_w rel {r_w:.4e} (x{r_w / r_lev_w:.3f})"
              f"  = levanter bits: loss {100 * eq_l:.2f}% grad_x {100 * eq_x:.2f}% grad_w(bf16) {100 * eq_w:.2f}%",
              flush=True)
        R.check(dl <= 2e-5, f"{size} {tag}: per-token loss max abs diff {dl:.3e} > 2e-5")
        R.check(zero_ok, f"{size} {tag}: weight-0 loss not 0")
        R.check(dmean <= 1e-6, f"{size} {tag}: weighted mean rel diff {dmean:.3e} > 1e-6")
        R.check(r_x <= 1.25 * r_lev_x, f"{size} {tag}: grad_x rel {r_x:.4e} > 1.25 x {r_lev_x:.4e}")
        R.check(r_w <= 1.25 * r_lev_w, f"{size} {tag}: grad_w rel {r_w:.4e} > 1.25 x {r_lev_w:.4e}")
        rows.append(dict(size=size, impl=impl, gemm="own" if own else "lt", chunk=chunk,
                         store="bf16" if store else "f32", strict=bool(strict), fast_stats_exp=bool(fast),
                         loss_maxabs=dl, mean_rel=dmean, loss_rel=r_l, grad_x_rel=r_x,
                         grad_w_rel=r_w, grad_x_ratio=r_x / r_lev_x, grad_w_ratio=r_w / r_lev_w, ok=ok,
                         eq_loss=eq_l, eq_grad_x=eq_x, eq_grad_w_bf16=eq_w,
                         arch=lib.nm_ce_device_arch()))
        del dx, dw, loss, lse
        torch.cuda.empty_cache()
    lib.nm_ce_set_backends(0, 0, -1)
    out = os.environ.get("NM_S5_T15_JSONL")
    if out:
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def configs(full):
    # chunk 32,768 (A: a 16.8 GB f32 logits workspace) only on GPUs with >= 60 GiB (the H100; the vis A100 is shared)
    big = torch.cuda.get_device_properties(0).total_memory >= 60 * 2**30
    c = []
    for impl in ("A", "B1", "B1.5", "B2"):
        for own in ((0, 1) if impl in ("B1", "B1.5") else ((1,) if impl == "B2" else (0,))):
            c.append((impl, own, 8192, BF16S, 0))
            if full:
                c.append((impl, own, 16384, BF16S, 0))
                if big:
                    c.append((impl, own, 32768, BF16S, 0))
    for impl in ("B1", "B1.5", "B2"):
        c.append((impl, 1, 8192, BF16S, 1))      # strict head dW: bf16 per 8192-row chunk, then bf16
        c.append((impl, 1, 8192, F32S, 0))       # the exec.ce_logits=f32 fallback
    c.append(("B1", 0, 8192, BF16S, 0, 1))       # the opt-in fast statistics exponential (data for its A/B)
    return c


def make_case(size_filter):
    @R.case
    def t15():
        found = [f for f in fixture_dirs() if size_filter is None or f[0] == size_filter]
        if not found:
            raise Skip(f"no T1.5 fixture (cache/levref/{size_filter or '<size>'}/t1_units/ce + t1_oracle/ce)")
        for size, u, o in found:
            one_size(size, u, o, configs(full=size == "130m"))
    t15.__name__ = f"t15_{size_filter}"
    return t15


make_case("130m")
make_case("1_2b")

if __name__ == "__main__":
    R.run(nmce.ARGS.cases)
