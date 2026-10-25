"""Shared helpers of the S5 gate tests (fused linear CE): the CE entry points of libmarincpp_kernels.so through ctypes,
the levanter emulation of the CE (numerics.md §5.2) and f64 autograd oracles in torch. The case runner, bf16 helpers
and error metrics come from tests/s3/nmk.py.

Every script takes --lib <libmarincpp_kernels.so> --src <repo root>. Exit 0 = all cases passed, 77 = all skipped
(no GPU), 1 = a failure.
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "s3"))
import nmk  # noqa: E402
from nmk import torch, rel, Runner, Skip, args_parse  # noqa: E402,F401

C = ctypes
VP, I32, I64, U64, F32 = C.c_void_p, C.c_int, C.c_int64, C.c_uint64, C.c_float

SIGS = {
    "nm_last_error": (C.c_char_p, []),
    "nm_ce_workspace_bytes": (U64, [I32, I64, I32, I32, I32, I32]),
    "nm_fused_linear_ce": (I32, [I32, I32, I32, I32, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, I32, VP, VP, I32, I32, I64,
                                 I32, I32, I32, VP, VP]),
    "nm_ce_set_backends": (I32, [I32, I32, I32]),
    "nm_ce_finish_microbatch": (I32, [VP, VP, I64, VP]),
    "nm_ce_grad_scale": (I32, [VP, I64, F32, VP, VP]),
    "nm_ce_readout": (I32, [I32, I32, I32, VP, VP, VP, VP, VP, VP, VP, VP, VP, I32, VP, VP, VP, VP, VP, I32, I64, I32,
                            I32, I32, F32, VP, VP]),
    "nm_ce_device_arch": (I32, []),
    "nm_ce_dlogits": (I32, [I32, VP, I64, VP, VP, VP, I32, I32, VP]),
    "nm_ce_set_fast_stats_exp": (I32, [I32]),
    "nm_rmsnorm_fwd": (I32, [I32, I32, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_rmsnorm_bwd_add_partial": (I32, [I32, I32, VP, VP, VP, VP, VP, VP, VP, I64, I32, F32, VP]),
    "nm_norm_partial_rows": (I64, [I64]),
}
NO_STATUS = {"nm_last_error", "nm_ce_workspace_bytes", "nm_ce_set_backends", "nm_ce_device_arch",
             "nm_ce_set_fast_stats_exp", "nm_norm_partial_rows"}

IMPLS = {"A": 0, "B1": 1, "B1.5": 2, "B2": 3}
LEV, ACC = 0, 1
F32S, BF16S = 0, 1


class Lib:
    def __init__(self, path):
        self.dll = C.CDLL(path)
        for name, (res, args) in SIGS.items():
            f = getattr(self.dll, name)
            f.restype, f.argtypes = res, args

    def __getattr__(self, name):
        f = getattr(self.dll, name)

        def call(*a):
            r = f(*a)
            if name not in NO_STATUS and r != 0:
                raise RuntimeError(f"{name} returned {r}: {self.dll.nm_last_error().decode()}")
            return r
        return call


def ptr(t):
    return None if t is None else VP(t.data_ptr())


ARGS = None


def init(suite):
    global ARGS
    ARGS = args_parse()
    nmk.ARGS = ARGS
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return Lib(ARGS.lib), Runner(suite)


# ------------------------------------------------------------------------------------------------ calls
class Ws:
    """One CE workspace, grown on demand (cudaMalloc through torch)."""
    buf = None

    @classmethod
    def get(cls, nbytes):
        if cls.buf is None or cls.buf.numel() < nbytes:
            cls.buf = None
            torch.cuda.empty_cache()
            cls.buf = torch.empty(max(nbytes, 256), dtype=torch.uint8, device="cuda")
        return cls.buf


def ce_call(lib, impl, x, W, targets, weight, gscale, *, rounding=LEV, store=BF16S, chunk=1024, dx=None, acc_dx=False,
            dw=None, dw_mb=None, dw_round=0, forward_only=False, bias=None, loss_partial=None, lse=None, io=None,
            ws=None):
    """Runs nm_fused_linear_ce. Returns (loss_tok, lse). dx / dw / dw_mb are written in place. ws: a caller-owned
    workspace (default: the shared one, sized by nm_ce_workspace_bytes, which also bounds the f32-io layout)."""
    T, D = x.shape
    V = W.shape[0]
    io = (0 if x.dtype == torch.float32 else 1) if io is None else io
    loss = torch.empty(T, dtype=torch.float32, device="cuda")
    lse = torch.empty(T, dtype=torch.float32, device="cuda") if lse is None else lse
    if ws is None:
        ws = Ws.get(lib.nm_ce_workspace_bytes(IMPLS[impl], T, D, V, chunk, store))
    lib.nm_fused_linear_ce(IMPLS[impl], rounding, io, store, ptr(x), ptr(W), ptr(targets), ptr(weight), ptr(gscale),
                           ptr(bias), ptr(loss), ptr(lse), ptr(loss_partial), ptr(dx), int(acc_dx), ptr(dw),
                           ptr(dw_mb), dw_round, int(forward_only), T, D, V, chunk, ptr(ws), None)
    return loss, lse


# ------------------------------------------------------------------------------------------------ references
def make_case(T, D, V, seed, logit_std=3.0, zero_every=7, dtype=torch.bfloat16):
    """x ~ N(0, 1), W ~ N(0, logit_std^2 / D) (logits ~ N(0, logit_std^2), as E2), uniform targets, weight 1 except
    every `zero_every`-th position and the last one (weight 0, as levanter's last position of a row)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, D, generator=g, device="cuda").to(dtype)
    W = (torch.randn(V, D, generator=g, device="cuda") * (logit_std / D ** 0.5)).to(dtype)
    tg = torch.randint(0, V, (T,), generator=g, device="cuda", dtype=torch.int32)
    w = torch.ones(T, dtype=torch.float32, device="cuda")
    if zero_every:
        w[::zero_every] = 0.0
    w[-1] = 0.0
    return x, W, tg, w


def oracle_f64(x, W, tg, w, gs, rows=512, bias=None):
    """f64 torch autograd of sum_t gs * w_t * CE_t on the given (rounded) inputs, in row blocks. Returns
    (loss_tok f64 [weight-0 rows 0], lse f64, dx f64, dW f64)."""
    T, D = x.shape
    V = W.shape[0]
    W64 = W.double().detach().requires_grad_(True)
    b64 = bias.double() if bias is not None else None
    loss = torch.empty(T, dtype=torch.float64, device="cuda")
    lse = torch.empty(T, dtype=torch.float64, device="cuda")
    dx = torch.empty(T, D, dtype=torch.float64, device="cuda")
    for r0 in range(0, T, rows):
        r1 = min(T, r0 + rows)
        xb = x[r0:r1].double().detach().requires_grad_(True)
        z = xb @ W64.T
        if b64 is not None:
            z = z + b64
        ce = torch.nn.functional.cross_entropy(z, tg[r0:r1].long(), reduction="none")
        lw = w[r0:r1].double()
        (ce * lw * float(gs)).sum().backward()
        with torch.no_grad():
            loss[r0:r1] = ce * (lw != 0)
            lse[r0:r1] = torch.logsumexp(z, dim=1)
            dx[r0:r1] = xb.grad
        del z, ce
    return loss, lse, dx, W64.grad.detach()


def mm32(a, b):
    """a @ b with bf16 operands on the tensor cores, f32 accumulator and f32 output (XLA's
    dot(preferred_element_type=f32) on the GPU); f32 operands: an f32 GEMM (TF32 off)."""
    if a.dtype == torch.bfloat16:
        return torch.mm(a, b, out_dtype=torch.float32)
    return a @ b


def chunk_loss_ref(loss, w, chunk):
    """k_ce_chunk_loss in torch, to the bit: per chunk, f32(w * loss) summed by 1024 thread-strided sequential f32
    sums, then a fixed binary tree. Returns f32 [n_chunks]."""
    T = loss.numel()
    out = []
    for r0 in range(0, T, chunk):
        prod = (w[r0:r0 + chunk].float() * loss[r0:r0 + chunk].float()).cpu()
        k = (prod.numel() + 1023) // 1024
        P = torch.zeros(k * 1024, dtype=torch.float32)
        P[:prod.numel()] = prod
        P = P.view(k, 1024)
        s = torch.zeros(1024, dtype=torch.float32)
        for i in range(k):
            s = s + P[i]
        o = 512
        while o > 0:
            s[:o] = s[:o] + s[o:2 * o]
            o >>= 1
        out.append(s[0].item())
    return torch.tensor(out, dtype=torch.float32)


def levanter_emulation(x, W, tg, w, gs, tile=8192, rows=1024, store_bf16=True, dw_mode="levanter", f32_dw=False):
    """numerics.md §5.2 in torch: f32 logits (bf16 tensor-core GEMM, f32 accumulate) -> f32 lse;
    zb = bf16(z) (the bf16 logit store);
    p = exp(zb - lse); dz = p g, dz[target] -= g; dzb = bf16(dz); dx = bf16(dzb W); dW per `tile` rows as bf16(dzb^T x)
    summed in f32, bf16 at the end (dw_mode "levanter"), or f32 throughout ("f32"). Returns (loss f32, lse f32, dx bf16,
    dW f32), plus the f32-throughout dW of the same dz (the port's f32 dW policy) when f32_dw."""
    T, D = x.shape
    V = W.shape[0]
    loss = torch.empty(T, dtype=torch.float32, device="cuda")
    lse = torch.empty(T, dtype=torch.float32, device="cuda")
    dx = torch.empty(T, D, dtype=torch.bfloat16, device="cuda")
    dW = torch.zeros(V, D, dtype=torch.float32, device="cuda")
    dW32 = torch.zeros(V, D, dtype=torch.float32, device="cuda") if f32_dw else None
    g = (w * torch.tensor(gs, dtype=torch.float32, device="cuda")).float()
    for t0 in range(0, T, tile):
        t1 = min(T, t0 + tile)
        dw_t = torch.zeros(V, D, dtype=torch.float32, device="cuda")
        for r0 in range(t0, t1, rows):
            r1 = min(t1, r0 + rows)
            z = mm32(x[r0:r1], W.T)
            m = z.max(dim=1).values
            l = torch.log(torch.exp(z - m[:, None]).sum(dim=1)) + m
            idx = tg[r0:r1].long()
            zt = z.gather(1, idx[:, None])[:, 0]
            loss[r0:r1] = torch.where(w[r0:r1] != 0, l - zt, torch.zeros_like(l))
            lse[r0:r1] = l
            zs = z.to(torch.bfloat16).float() if store_bf16 else z
            del z
            dz = torch.exp(zs - l[:, None]) * g[r0:r1, None]
            dz.scatter_add_(1, idx[:, None], -g[r0:r1, None])
            dzb = dz.to(torch.bfloat16)
            del dz, zs
            dx[r0:r1] = mm32(dzb, W).to(torch.bfloat16)
            dw_t += mm32(dzb.T.contiguous(), x[r0:r1])
            del dzb
        if dw_mode == "levanter":
            dW += dw_t.to(torch.bfloat16).float()
        else:
            dW += dw_t
        if f32_dw:
            dW32 += dw_t
    if dw_mode == "levanter":
        dW = dW.to(torch.bfloat16).float()
    return (loss, lse, dx, dW, dW32) if f32_dw else (loss, lse, dx, dW)
