"""torch references of the S3 kernels.

1. Levanter emulation: the rounding points of levanter's jaxpr (numerics.md §3.2-§3.5) written op by op in torch f32
   with an explicit bf16 rounding (rb) after every op that levanter computes in bf16. Each torch op is one kernel, so no
   contraction crosses an op; torch's exp / rsqrt / division are CUDA's expf / rsqrtf / IEEE division, the functions
   the S3 kernels and XLA:GPU use. Reductions run in torch's order (the kernels' order differs: class S), so a sum can
   differ in its last f32 bit and flip a rare bf16 rounding.
2. f64 oracles: the same math in float64 through torch autograd (DESIGN.md §10.2: the gradient check of each kernel
   against autograd of its reference, from the same bf16 inputs).
"""
import torch

from nmk import rb


def div_n(x, n):
    """x / n as an IEEE division per element (torch turns `tensor / python_scalar` into a multiply by the reciprocal,
    which is not the same rounding when n is not a power of two)."""
    return x / torch.full_like(x, float(n))


# ------------------------------------------------------------------------------------------------ RMSNorm
def norm_stats(x32, eps):
    n = x32.shape[-1]
    ms = div_n((x32 * x32).sum(-1, keepdim=True), n)
    r = torch.rsqrt(ms + eps)
    return ms, r


def lev_norm_fwd(x32, g32, r):
    y = rb(x32 * r)
    return rb(g32 * y), y


def lev_norm_bwd(x32, g32, dy32, r, ms, eps):
    """dx (bf16-rounded) and the per-element gain contributions f32(bf16(dy * y))."""
    n = x32.shape[-1]
    a = rb(g32 * dy32)
    y = rb(x32 * r)
    dgc = rb(dy32 * y)
    s = (x32 * a).sum(-1, keepdim=True)
    c = div_n(s * (-0.5 * (r / (ms + eps))), n)
    dx = rb(a * r + c * (2.0 * x32))
    return dx, dgc


def f64_norm(x64, g64, eps):
    ms = (x64 * x64).mean(-1, keepdim=True)
    return g64 * x64 * torch.rsqrt(ms + eps)


# ------------------------------------------------------------------------------------------------ RoPE
def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def rot_half_t(u):
    """Transpose of rot_half: rot^T(u) = [u2, -u1]."""
    h = u.shape[-1] // 2
    return torch.cat([u[..., h:], -u[..., :h]], -1)


def lev_rope_fwd(n32, c32, s32):
    return rb(rb(n32 * c32) + rb(rot_half(n32) * s32))


def lev_rope_bwd(d32, c32, s32):
    return rb(rb(d32 * c32) + rot_half_t(rb(d32 * s32)))


# ------------------------------------------------------------------------------------------------ SwiGLU
def lev_logistic(g32):
    e = rb(torch.exp(-g32))
    return rb(1.0 / rb(1.0 + e))


def lev_swiglu_fwd(g32, u32):
    s = lev_logistic(g32)
    silu = rb(g32 * s)
    return rb(silu * u32)


def lev_swiglu_bwd(g32, u32, dh32):
    s = lev_logistic(g32)
    silu = rb(g32 * s)
    sp = rb(s * rb(1.0 - s))
    dsilu = rb(dh32 * u32)
    du = rb(dh32 * silu)
    dg = rb(rb(dsilu * s) + rb(rb(g32 * dsilu) * sp))
    return dg, du


def interleave_gu(g, u):
    """[T, F] gate and up -> [T, 2F] in 64-column blocks gate_b | up_b."""
    T, F = g.shape
    return torch.stack([g.reshape(T, F // 64, 64), u.reshape(T, F // 64, 64)], 2).reshape(T, 2 * F)


def split_gu(gu):
    T, F2 = gu.shape
    v = gu.reshape(T, F2 // 128, 2, 64)
    return v[:, :, 0, :].reshape(T, F2 // 2), v[:, :, 1, :].reshape(T, F2 // 2)
