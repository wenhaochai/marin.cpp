# Numerics, precision and kernels

Stage-1 spec section. It covers the dtype of every tensor and every rounding point in one training step of the plain
`muonh_qwen3` baseline. It also covers the attention and cross-entropy kernels, sharding, microbatching,
rematerialization, XLA/JAX flags and the sources of nondeterminism. Each item is classified as **M** (math: must
match), **P** (precision: match it for the parity run) or **S** (speed only: free to change). See §1.

Model structure is in `model.md`. Optimizer math is in `optim.md`. Parity gates are in `targets.md`. This section
restates their content only where it concerns numerics.

## 0. Sources and notation

Paths are relative to `$ROOT/project/marin` at HEAD `1793624070`. `git diff 59ac0d6 HEAD --
lib/` changes only `lib/levanter/src/levanter/trainer.py`, and that change is a no-op context variable (+17 lines,
`_TRACED_TRAIN_STEP`). The other three baseline commits (`28fd599`, `ac6af86`, `cead09a`) lie between `59ac0d6` and
HEAD, so every library line cited here is the code that ran.

| Alias | Path |
|---|---|
| `L` | `experiments/references/della_muonh_qwen3_scaling.py` (HEAD; the same knobs are at `git show 59ac0d6:<path>` lines 61-160) |
| `LV` | `lib/levanter/src/levanter` |
| `HX` | `lib/haliax/src/haliax` |
| `CE` | `LV/kernels/pallas/fused_cross_entropy_loss` |
| `FA` | `LV/models/flash_attention.py` |
| `NC` | `$ROOT/project/nanomarin/cache` |
| `J130` | `NC/hlo/130m/train_step.jaxpr.squeezed.txt`: the 130m baseline's traced train-step jaxpr, written by the run itself on 2026-09-14 04:18 (`logs/muonh-qwen3-130m-della4xh100/artifacts/train_step.jaxpr.txt.gz`, runs of spaces squeezed, line numbers unchanged) |
| `H<size>` | `NC/hlo/<size>/train_step.hlo.txt`: the lowered StableHLO of each size's train step (pre-optimization, pre-SPMD), copied from `logs/muonh-qwen3-<size>-della4xh100/artifacts/` |

**Which step program these files show.** `LV/trainer.py:534-546` saves the program of a step with jit hooks
(`hooks_this_time`: `step % 10 == 0`, watch `interval: 10` in `WB`) under the name `train_step`, and the program of the
other 9 of 10 steps under the name `train_step_hooks`. The names are swapped relative to their meaning. So `J130` and
`H<size>` are the jit-hook-step program, and `NC/hlo/<size>/train_step_hooks.{jaxpr.txt.gz,hlo.txt}` are the
program of the other steps. Check (130m, 2026-10-24): the two jaxprs have the same 146 `dot_general` equations (same
md5 of the sorted output-dtype/shape and `preferred_element_type` list), the same 476 equations with a bf16 output
(same md5 of the sorted `dtype[shape] = primitive` list), and the two StableHLO files have the same md5 of the sorted
`stablehlo.dot_general` type signatures. The extra 826 jaxpr lines of the hook program are f32 watch statistics. So
every dtype and rounding claim below holds for both programs.
| `WB` | `NC/wandb/api/reself__marin-della__muonh-qwen3-<size>-della4xh100.json`: the logged config, fetched from W&B on della-vis1 |
| `SL` | `NC/hlo/130m/slurm_13846675.out`: Slurm log of the 130m baseline job |

Software of the runs. Package versions come from the dist-info directories in
`.venv/lib/python*/site-packages/` (installed 2026-09-13 01:09, before the runs): jax 0.11.1 (also `SL:7`), jaxlib
0.11.1, jax_cuda13_plugin and jax_cuda13_pjrt 0.11.1, equinox 0.13.2, optax 0.2.6, jmp 0.0.4. Device kind:
`NVIDIA H100 80GB HBM3` (`J130` mesh records). The venv holds two CUDA stacks, and the CUDA-12 and CUDA-13 wheels of
NCCL and cuDNN install to the same file paths:

| Library | Packages in the venv | File on disk at the path the XLA plugin loads from |
|---|---|---|
| NCCL | `nvidia_nccl_cu12` 2.28.9, `nvidia_nccl_cu13` 2.31.2 | `nvidia/nccl/lib/libnccl.so.2`: 400796408 B, the size in the cu12 RECORD; `strings` gives `NCCL version 2.28.9+cuda12.9` |
| cuDNN | `nvidia_cudnn_cu12` and `nvidia_cudnn_cu13`, both 9.19.0.56 | `nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9`: 515924288 B, the size in the cu12 RECORD |
| cuBLAS | `nvidia_cublas` 13.4.1.1 (`nvidia/cu13/lib/libcublas.so.13`), `nvidia_cublas_cu12` 12.8.4.1 (`nvidia/cublas/lib/libcublas.so.12`) | the RPATH of `jax_plugins/xla_cuda13/xla_cuda_plugin.so` lists `nvidia/cu13/lib`, `nvidia/cudnn/lib` and `nvidia/nccl/lib`, not `nvidia/cublas/lib` |

`SL:414` (`NCCL version 2.28.9+cuda12.9`) confirms that the process loaded the CUDA-12 build of NCCL. No line of `SL`
names cuBLAS or cuDNN. The GEMMs most likely used cuBLAS 13.4.1.1, because it is the only `libcublas.so.13` on the
plugin's RPATH, but this is not verified (an `LD_LIBRARY_PATH` entry could supply another one). torch 2.11.0+cu128 is
also installed and is not used by the train step. Open item O2 checks the loaded libraries.

`B` = sequences per microbatch (global), `S` = 4096, `D` = hidden, `hd` = head dim, `V` = 128256, `PDP` =
`per_device_parallelism` (sequences per GPU per microbatch), `B_dev = PDP * S` = tokens per GPU per microbatch.
`bf16(x)` is round-to-nearest-even to bfloat16.

**Limit of this evidence.** `J130` and `H<size>` show the dtypes that JAX requests. XLA:GPU then optimizes the
program. Its default `xla_allow_excess_precision=true` lets it keep some bf16 intermediates in f32 inside fusions, and
it may also remove some bf16 round trips. So the rounding points listed below are an upper bound on the real ones. Only
an optimized-HLO dump from a GPU node can pin them (§10, open item O1).

**Transcendentals and denormals (assumption, not verified in code).** The jaxpr fixes the dtype of `exp`, `log`,
`logistic`, `rsqrt`, `cos` and `sin`, but not their implementation. This spec assumes XLA:GPU's defaults: a bf16
transcendental is evaluated in f32 and rounded to bf16, the f32 functions are accurate to a few ulp over the full
argument range, and denormals are not flushed to zero. Denormals do occur: in the CE backward, `p * g_loss` with
`g_loss = 1/524160` is below 2^-126 for every `p` below about 6e-33. Each such term is below 1.2e-38, so a flush to zero
changes no gradient that matters, but a bit-level comparison sees it. A rewrite must not use fast-math intrinsics
(`__expf`, `__cosf`, `--use_fast_math`) without an A/B test (§1). O1 records the flags.

## 1. Classification and summary

* **M (math):** the item changes the function that is computed. nanomarin must reproduce it exactly.
* **P (precision):** the item is a rounding point or a dtype. The math is the same, and only the rounding changes. Use
  the reference choice in the parity run. A change that raises precision is allowed after an A/B test shows no effect:
  n >= 4 runs per arm, gates `targets.md` §4(c). A change that lowers precision is not allowed in stage 1.
* **S (speed):** the item changes the schedule, the data layout or the reduction order, but not the dtypes. It is free
  to change.

A reference rerun with the same config and seed already differs by about 1 pool sd (§9.2). So no item in this file
can be matched bit-for-bit across implementations, and parity is judged on distributions (`targets.md` §4).

| # | Item | Reference behavior (plain baseline, all four sizes) | Class |
|---|---|---|---|
| 1 | Master params, grads after cast, optimizer states, update | f32 | M |
| 2 | Compute dtype of matmul inputs and activations | bf16; GEMMs accumulate in f32 and write bf16 | P |
| 3 | Residual stream (scan carry) | bf16 `[B,S,D]`; residual adds in bf16 | P |
| 4 | RMSNorm (7 kinds) | statistics and normalization in f32; eps 1e-5 inside the rsqrt; cast to bf16; gain multiply in bf16; backward: bf16 `g*dy`, then f32, one bf16 cast of `dx`; gain grads are bf16 reductions | M (formula) / P (casts) |
| 5 | RoPE | angles, cos and sin in f32, tables cast to bf16; rotation in bf16; positions 0..4095, not reset at documents | M / P |
| 6 | Attention scale | `q * bf16(1/sqrt(hd))` in bf16, after RoPE, outside the flash VJP (so `dq` is scaled after the kernel); 0.125 (hd 64), 0.08837890625 (hd 128) | M |
| 7 | Attention mask | causal AND same segment; masked logit := -1e10 | M |
| 8 | Attention kernel | pure-JAX FlashAttention-2 (`FA`), block 1024x1024, causal block skip | S |
| 9 | Attention numerics | scores, running max, running sum and output accumulator in bf16; row sum accumulated in f32 then rounded to bf16 | P |
| 10 | CE math | f32 logits and logsumexp; label = next token; mean over positions 0..S-2 of the microbatch; no z-loss; no soft cap | M |
| 11 | CE kernel and tiles | `batched_xla`, H100 full-vocab B-tiled path, 8192-row tiles (any `PDP >= 2`) | S |
| 12 | CE backward numerics | logits recomputed in bf16; dlogits cast to bf16 before both GEMMs; grad_w summed over tiles in f32, then bf16 | P |
| 13 | Gradient dtypes before the f32 cast | all per-layer param grads bf16; embedding grad = bf16 scatter-add; lm_head grad psum in bf16 | P |
| 14 | Sharding | FSDP (ZeRO-3) over 4 GPUs along the `embed` axis; batch split 4 ways | S |
| 15 | Microbatching | 1 microbatch (130m, 300m) or 2 (520m, 1_2b); f32 sum then divide | S |
| 16 | Remat | `jax.checkpoint` on each scanned layer, policy None: save the layer input only | S |
| 17 | Matmul precision for f32 x f32 | `DEFAULT` everywhere, which permits TF32 on H100 (not verified) | P |
| 18 | XLA flags and environment | only `--xla_gpu_nccl_termination_timeout_seconds=600`; memory fraction 0.9; no determinism flag | S |
| 19 | Nondeterminism | atomic bf16 scatter-add in the embedding backward; XLA autotuning across compilations | S |

## 2. Mixed-precision policy

* Policy: `jmp.get_policy("p=f32,c=bfloat16")` (`L:782`). Logged: `param_dtype float32`, `compute_dtype bfloat16`,
  `output_dtype bfloat16` (`WB .config.trainer.mp`, identical for all four sizes). `output_dtype` is never used: no
  call to `cast_to_output` exists in `LV` or `HX` (grep).
* The model is built in f32 (`LV/main/train_lm.py:142`). Params stay f32 in the train state (`H130` arguments 1-16 are
  `f32`).
* The cast happens inside the loss function. `_mp.cast_to_compute(model)` (`LV/trainer.py:253-254`) casts every
  floating leaf to bf16: Linear weights, norm gains, the embedding table and `lm_head` (`J130:15-58`). The cast lies
  inside `eqx.filter_value_and_grad` (`LV/trainer.py:799`). So the gradient of each f32 param is
  `f32(bf16 cotangent)`: every param gradient carries bf16 precision before it reaches the optimizer.
* Train-step inputs: tokens `i32 [B,S]`, loss weight `f32 [B,S]`, query and key segment ids `i32 [B,S]` (`H130`
  arguments 48-51).
* f32 everywhere after the cast back: microbatch accumulation (`LV/grad_accum.py:96-162`), clipping norms, Muon
  momentum, Newton-Schulz, Adam/AdamH moments, the norm-preserving projection, `p + u` (`optim.md` §3; `J130:540-700`).

### 2.1 Ops in f32

* RMSNorm statistics and the normalized value, before its cast (§3.2). In the backward, every step after the bf16
  product `g * dy`: the rsqrt-derivative terms, the division by `n` and the sum, until `dx` is rounded to bf16 (§3.2).
* RoPE angles, cos and sin, before their cast (§3.3).
* Attention: the row sum of `P` (accumulated in f32, then rounded to bf16); the storage of `L`. `L` itself is computed
  in bf16 (bf16 `log`, bf16 add) and then upcast to f32, so its precision is bf16 (§4.3). In the backward:
  `P = exp(f32(S) - L)`, `dS = P * f32(dP - D)`, the `dV` GEMM, and the f32 sums of `D` and of the GQA group (§4.4).
* Cross-entropy: logits, logsumexp and loss in the forward. Probabilities and dlogits in the backward, before their
  cast (§5).
* Loss reduction, microbatch accumulation, the optimizer, and the casts of gradients back to f32.

### 2.2 Ops in bf16

* Every GEMM except the CE forward logits and the attention `dV`.
* The residual stream; SiLU and the gate product, forward and backward (§3.5); the gain multiplies; the RoPE rotation;
  the attention-scale multiply, forward and backward (§3.4).
* Attention scores, exp, the running max, the running sum and the output accumulator.
* Every activation cotangent that passes from one op to the next, and all per-layer parameter cotangents. Inside some
  backward ops the math is f32: the RMSNorm backward (§3.2), the flash `P`, `dS` and `dV` (§4.4) and the CE backward
  (§5.2). Each of them rounds its output to bf16.
* The embedding scatter-add and the `lm_head` gradient psum.
* All 7 norm-gain gradients: each is a bf16 `reduce_sum` of the bf16 product `dy * y` over all B*S = 524288 positions
  (for `q_norm`/`k_norm` also over the heads), with a bf16 result. Per GPU the local sum covers 131072 positions
  before the cross-GPU reduction. `J130:468` (`kc`, final norm), `J130:4043, 4130, 4159, 4368` (`edx`, `efe`, `eft`,
  `ejq`: `bf16[512]`, block norms), `J130:4264, 4292` (`ehz`, `ein`: `bf16[64]`, QK-norm). XLA:GPU may accumulate
  these reductions in f32 internally (O1).

## 3. Forward pass: dtype of each op

`model.md` §2 gives the order of operations. This section adds the dtypes, read from the forward layer body
`J130:2530-2794` (`jaxpr1`) and the main step `J130:30-470`.

### 3.1 Embedding and linear layers

* Embedding: `gather` on the bf16 table, then `* 1.0:bf16` (`J130:62-70`; `HX/nn/embedding.py:74-81`).
  Out-of-range ids are mapped with `id < 0 -> id + V` and `mode=PROMISE_IN_BOUNDS`.
* Linear: `dot_general(x_bf16, (W * 1.0)_bf16, preferred_element_type=bf16)` (`J130:2569-2599`;
  `HX/nn/linear.py:83-100`; `HX/quantization.py:150-170`). `precision` is unset: no dot in `J130` has a `precision=`
  attribute. cuBLAS accumulates in f32 and rounds the output to bf16. No biases.

### 3.2 RMSNorm

The same code serves all 7 norms (`HX/nn/normalization.py:127-139`; `J130:2550-2567` for the input norm and
`J130:2601-2638` for QK-norm):

```
x32 = f32(x_bf16)
ms  = reduce_sum(x32*x32, axis) / n        # n = D, or hd for q_norm and k_norm; f32 divide by the literal n
r   = rsqrt(ms + 9.999999747378752e-06)    # f32(1e-5)
y   = bf16(x32 * r)
out = g_bf16 * y                           # bf16 multiply; g = bf16(f32 gain)
```

* M: eps lies inside the rsqrt. The gain multiplies after the normalization.
* P: there are two roundings. The first is `y` to bf16. The second is the bf16 product with `g`, which was itself
  rounded from f32.
* Backward (autodiff of the code above; final norm at `J130:271-285` and `J130:466-490`, the block norms follow the
  same pattern inside the layer backward). With `dy` the bf16 cotangent of `out` and `y`, `x32`, `ms`, `r` from the
  forward:

```
a   = g_bf16 * dy                                # bf16 product (ka)
dg  = reduce_sum_positions(dy * y)               # bf16 product, bf16 reduce_sum, bf16 result (kb, kc)
a32 = f32(a)                                     # kd
t1  = a32 * r                                    # kf
s   = reduce_sum_axis(x32 * a32)                 # f32 (ke, kh)
c   = (s * (-0.5 * (r / (ms + eps)))) / n        # f32; gr = rsqrt(ms+eps)/(ms+eps), gs = -0.5*gr (J130:276-277); ki, kj
t2  = c * (2.0 * x32)                            # f32; gm = 2*x32 (J130:271); kl
dx  = bf16(t1 + t2)                              # km, kn: the only rounding of dx
```

  * P: `dx` has two roundings: `a` (bf16 product) and the final cast. `dg` is rounded at every bf16 add of the
    reduction, as far as the jaxpr shows (§2.2). A norm backward that runs fully in bf16, or fully in f32 including
    `g * dy`, changes these rounding points.
* QK-norm is on. `Qwen3Config.attention_config` forces it (`LV/models/qwen.py:374-376`), although the logged config
  says `use_qk_norm: false` (`model.md` §2.1). It runs before RoPE (`LV/layers/attention.py:1436-1444`).

### 3.3 RoPE

* `inv_freq` is an f32 constant computed at trace time (`LV/layers/rotary.py:35-38`; the closed-over constants
  `a, b: f32[32]` in `J130:1`). Under `jax.ensure_compile_time_eval` JAX runs it eagerly in f32, not in float64:
  `k = int32 [0, 2, ..., hd-2]`; `e = k / hd` (true division, f32); `p = theta ** e` (`theta = 10000.0`, weak-typed,
  so f32); `inv_freq = 1.0 / p` (f32); then `inv_freq / factor` with `factor = 1.0`. The jaxpr does not print the
  values of `a` and `b`.
* The angles are `f32(pos) * inv_freq`, one f32 multiply (`J130:197`, `fb`), duplicated as `concat(freqs, freqs)`
  (`fc`). cos and sin are computed in f32 and cast to bf16 (`J130:199-202`, `fd`..`fg`; `rotary.py:40-43`).
* P: the angle reaches `4095 * inv_freq[0] = 4095` rad. Half a bf16 ulp near 1 is 2^-9 (about 2e-3). A cos/sin
  without full-range argument reduction (CUDA `__cosf`/`__sinf`, `--use_fast_math`) can err by a large fraction of
  that at 4095 rad and flip table entries after the bf16 cast. A 1-ulp difference in an `inv_freq` entry moves the
  angle at pos 4095 by up to about 2.4e-4 rad and can also flip entries. So nanomarin either uses accurate f32
  cos/sin on the f32 angle `f32(f32(pos) * inv_freq)`, with `inv_freq` built in f32 in the order above, or copies the
  exact f32 `inv_freq` and the bf16 cos/sin tables `[4096, hd]` from the reference (O4).
* The rotation runs in bf16: `q*cos + concat(-q[hd/2:], q[:hd/2])*sin` (`J130:2639-2668`; `rotary.py:17-22, 45`).
  Each multiply and each add rounds to bf16.
* `pos = arange(S)` for every row (`attention.py:1441-1442`).

### 3.4 Attention scale

* `scaling_factor = 1 / math.sqrt(hd)` is a Python float (`attention.py:176-177`). It is applied as `q = q *
  scaling_factor` (`FA:92-96`). As a weak-typed scalar it becomes a bf16 constant.
  * 130m, 300m, 520m (hd 64): `0.125:bf16` (`J130:2672`; `H520` constant `1.250000e-01 : tensor<bf16>`).
  * 1_2b (hd 128): `8.837890e-02 : tensor<bf16>` (`H1_2b`), that is 0.08837890625, not 0.0883883476. `model.md` §2.3
    quotes the unrounded value.
* The product `q * scale` is rounded to bf16, and it is applied after RoPE.
* The multiply lies outside the custom VJP: `FA:96` runs before `_flash_attention` (`FA:98`, `filter_custom_vjp` at
  `FA:114`). So the flash forward and backward see the scaled, bf16-rounded `q`: `S = dot(q_scaled, k)`, and the
  backward forms `dK = dot(dS, q_scaled)` (`FA:401`). The flash backward returns the cotangent of `q_scaled`, and
  autodiff then multiplies it by the same bf16 constant in bf16: `dq = bf16(dq_scaled * scale)` (`J130:4228`, `ehb =
  mul egt 0.125:bf16`; 1_2b jaxpr line 5309, `ftl = mul ftd 0.08837890625:bf16`). At hd 64 this multiply is exact
  (a power of two). At hd 128 it is one more rounding.
* For parity, use the bf16 constant itself, scale `q` before the kernel, and scale `dq` after it. A kernel that folds
  the scale into the softmax (FlashAttention-2/3 `softmax_scale`) changes the forward and the backward roundings
  (class P).

### 3.5 MLP, post-norms, residual

* `gate` and `up` are bf16 GEMMs. `silu(x) = x * logistic(x)`, with both ops in bf16 (`J130:2749-2750`). The product
  with `up` is bf16, and `down` is a bf16 GEMM.
* SiLU backward (autodiff; residuals `J130:3992-3995`, backward `J130:4105-4109`). With `g` the bf16 cotangent of
  `silu(x)`, every op is bf16 and rounds:

```
s   = logistic(x)                  # ecx
sp  = s * (1 - s)                  # ecz = 1 - s, ecy = s * ecz
dx  = (g * s) + ((x * g) * sp)     # eev = g*s, eeu = x*g, eew = eeu*sp, eet = eev + eew
```

  A fused SwiGLU backward (f32 math, or the form `g * s * (1 + x * (1 - s))`) changes these rounding points (class P).
* Post-attention and post-MLP RMSNorm follow §3.2.
* Residual: `x = x + branch` in bf16 (`J130:2715, 2794`; `LV/models/llama.py:334-347`). The scan carry is
  `bf16[B,S,D]` (`J130:258`).

### 3.6 Loss reduction

* `loss_tok` is the f32 output of the CE kernel (§5). The weights are `w[s] = 1` for `s < S-1` and `w[S-1] = 0`
  (`LV/data/text/examples.py:141-149, 200-210`). There is no `ignore_id`, and no other position is masked.
* `loss = sum(loss_tok * w) / sum(w)` over the whole microbatch `[B,S]`, in f32 (`HX/nn/loss.py:132-138`;
  `J130:376-381`). The denominator is `B * 4095`. It is 524160 at every step of every size, because each microbatch
  holds 128 sequences, with one exception: updates 0-85 of the 520m baseline ran one microbatch of 256 sequences
  (`PDP = 64`, §6.3; `NC/wandb/config/run-20260914_063908-muonh-qwen3-520m-della4xh100.config.yaml:498`), so the
  denominator there was 1048320 and `g_loss = 1/1048320`. The mean is the same.
* With two microbatches, `train/loss = (loss_1 + loss_2)/2` and `grad = (g_1 + g_2)/2`, in f32
  (`LV/grad_accum.py:155-162`). Each sequence carries the same weight sum (4095), so this equals the global mean.

## 4. Attention kernel

### 4.1 Selection

* `attn_backend=AttentionBackend.JAX_FLASH` (`L:661`; logged `JAX_FLASH`). The GPU default is NVTE
  (`attention.py:84-89`), and Della has no Transformer Engine.
* JAX_FLASH has no `case` in the dispatch, so the call falls through to `flash_attention` (`attention.py:276-300`):
  pure JAX, `lax.while_loop`s and an `equinox.filter_custom_vjp` (`FA:114-145, 448-449`). No cuDNN, Triton or Pallas
  kernel is involved.
* `upcast_attn=False`, so `attention_dtype = x.dtype = bf16` (`attention.py:1357`).
* Dropout, bias, soft cap and sinks are all off.

### 4.2 Blocks and SEQ_LEN constraint

* `block_size = BLOCK_SIZE = 1024` for both Q and K (`FA:23, 63-64`; `flash_attention_block_size: null` logged).
* S = 4096 gives 4 x 4 blocks. Causal skipping visits only `j <= i` (`FA:278-279`) in the forward and `i >= j`
  (`FA:417`) in the backward: 10 of 16 block pairs.
* Constraints:
  * `S % block_size == 0`, or a `ValueError` is raised (`FA:166-169`).
  * If `S < block_size`, the call falls back to the VANILLA O(S^2) path (`FA:69-90`).
  * At HEAD the launcher picks block 512, 256 or 128 when SEQ_LEN is not a multiple of 1024 (`L:112-114`). That
    applies to non-baseline runs only.
* Block sizes change only tiling, so they are class S. One caveat: with bf16 running statistics (§4.3) the block size
  also moves the rounding points. With f32 accumulators, as in FlashAttention-2/3, it does not.

### 4.3 Forward numerics (`FA:147-299`; `J130:4396-4538`, `jaxpr6`)

For each query block `i` and each key block `j <= i`, with all tensors bf16 unless marked:

```
S   = dot(q_i, k_j)                      # bf16 out (preferred bf16)
S   = where(mask_ij, S, bf16(-1e10))     # -1e10 -> -9999220736 in bf16
m'  = max(m, rowmax(S))                  # bf16
P   = exp(S - m')                        # bf16
c   = exp(m - m')                        # bf16
l   = c*l + bf16(sum_f32(P))             # jnp.sum upcasts bf16 to f32, then the result is rounded
O   = c*O + dot(P, v_j)                  # bf16 dot out, bf16 accumulate
```

* Init: `m = -inf`, `l = 0`, `O = 0`, all bf16 (`FA:180, 205-206`).
* End of row block: `O = O / l` in bf16, and `L = f32(m + log(l))`, with `m + log(l)` computed in bf16 (`FA:288-293`).
* The mask is `(q_pos >= k_pos) & (seg_q == seg_k)` (`J130:4475-4497`). The segment ids come from the data:
  `seg = cumsum(roll(tokens,1) == eos)`, with the first token in segment 0 (`LV/data/text/examples.py:165-171`). The
  EOS token belongs to the document it ends.
* A key block can be fully masked for some rows (a different document). Those rows then get `m = -1e10` and
  `P = 1` (junk) until the diagonal block. There `c = exp(-1e10 - m') = 0` erases the junk exactly, because the
  diagonal block is always last in the loop. So a varlen kernel that skips fully masked tiles is equivalent in exact
  arithmetic.

### 4.4 Backward numerics (`FA:302-445`; `J130:4964-5195`, `jaxpr11`)

* `D = bf16(sum_f32(f32(bf16(dO * O))))` over hd (`FA:335`, `hax.sum(dO * O, axis=Key)` on bf16 `dO` and `O`;
  `J130:4206-4212`: `egl:bf16 = mul`, `egm = f32(egl)`, f32 `reduce_sum`, `ego = bf16`). Two roundings: the
  elementwise product is rounded to bf16 before the f32 sum, and the sum is rounded to bf16.
* The loop runs over key blocks `j` (outer) and query blocks `i >= j` (inner). S is recomputed in bf16 and masked.
* `P = exp(f32(S) - L)` in f32. `dP = dot(dO, v)` in bf16. `dS = bf16(P * f32(dP - D))`, where `dP - D` is a bf16
  subtraction.
* Per inner block, for each q-head `g` of the group (`FA:400-401`):
  * `dV_g = bf16(dot_f32(P, f32(dO)))`: a mixed-dtype dot (`P` f32, `dO` bf16) with `preferred_element_type=float32`,
    so `dO` is upcast and the GEMM runs in f32 (TF32-eligible, §8); `.astype(dV_j.dtype)` rounds it to bf16
    (`J130:5150-5157`, `eyn` -> `eyo`).
  * `dK_g = dot_bf16(dS, q_scaled)`: a bf16 GEMM with bf16 output (`eyp`); `q_scaled` is the scaled `q` (§3.4).
* Group sum (`FA:403-405`), then accumulate (`FA:407-408`):
  `dV_blk = bf16(sum_f32 over g of f32(dV_g))`, the same for `dK_blk`, then `dV_j += dV_blk` and `dK_j += dK_blk` as
  bf16 adds (`J130:5162-5170`: `eyq = f32(eyo)`, `eyr = reduce_sum[axes=(2,)]` in f32, `eys = bf16(eyr)`).
  * So `dV` has three roundings per block (GEMM output, group sum, accumulate) and `dK` has three (bf16 GEMM output,
    group sum, accumulate). The f32 GEMM outputs are not summed over `g` before the first rounding.
  * This path runs at every size. At G = 1 (130m, 300m) the group sum is over a size-1 axis, and the bf16 -> f32 ->
    bf16 round trip is exact. At G = 2 (520m, 1_2b) it is a real sum: 520m jaxpr
    (`NC/hlo/520m/train_step.jaxpr.txt.gz`) lines 6702-6733, `hiw:f32` dot -> `hix:bf16` -> `hjc:f32` -> `hjd`
    reduce_sum -> `hje:bf16` -> `hjg` add.
* `dQ_i += bf16(dot(dS, k))` is a bf16 GEMM with bf16 output and a bf16 add (`FA:410`).
* The accumulators `dQ`, `dK` and `dV` are bf16 (`FA:337-339, 398-412`).

### 4.5 Class

* M: the mask, the scale constant, and the absolute RoPE positions.
* S: the kernel itself, the block sizes, and causal or varlen tile skipping.
* P: the bf16 softmax statistics and the bf16 accumulators.
* A FlashAttention-3 kernel (f32 softmax, f32 accumulators) raises precision. It is allowed after the A/B test of §1;
  it is the natural first deviation.

## 5. Fused cross-entropy

### 5.1 Call chain and inputs

1. `compute_next_token_loss` (`LV/models/lm_model.py:319-354`) calls `maybe_fused_next_token_loss`
   (`LV/models/loss.py:35-98`). It passes `dtype = loss_weight.dtype = f32` (`loss.py:78-80`),
   `block_size=None`, `precision=None` and `logsumexp_weight = z_loss_weight = 0.0` (`L:819`;
   `LV/main/train_lm.py:205`).
2. Labels are `roll(tokens, -1)` (`loss.py:76`). The last position's label is token 0 of the same row, and its weight
   is 0.
3. `fused_cross_entropy_loss_and_logsumexp_penalty` (`loss.py:207-299`) wraps the kernel in `shard_map` over the
   compute mapping (`loss.py:286-291`).
   * Each GPU sees `x: bf16[B_dev, D]` (its own sequences, flattened), `labels: i32[B_dev]` and
     `w: bf16[D, V]`, which is replicated with `in_spec P(None, None)` (`J130:324-336`).
   * The kernel returns per-token loss with `reduction=None`, and the weighted mean happens outside (§3.6).
4. `implementation=None` selects the defaults `("batched_xla", "xla")` on a non-GB10 GPU (`CE/api.py:159-173,
   672-676`). The first one that does not raise wins (`CE/api.py:685-825`).

### 5.2 Path taken by the plain baseline: `batched_xla`, H100 full-vocab B-tiled

* Dispatch (`CE/batched_xla.py:369-389, 432-442`): take the B-tiled path if
  1. the device kind contains `h100`,
  2. `x` and `w` are both bf16,
  3. `B_dev >= 8192`,
  4. `V >= 65536`, and
  5. `return_argmax` is false.
* The tile is 8192 rows (`_H100_FULL_VOCAB_B_TILED_BLOCK_SIZE`, `CE/batched_xla.py:33`).
* Baseline numbers: `PDP = 32` for all four sizes (`WB .config.trainer.per_device_parallelism`; 520m updates 0-85:
  64, §6.3), so `B_dev = 131072` and there are 16 tiles. `scan length=16` appears in `J130:352-356`. `SL:111` logs
  `Fused cross-entropy selected implementation: batched_xla`. `H300`, `H520` and `H1_2b` contain the same two
  functions, `_linear_softmax_cross_entropy_loss_full_vocab_b_tiled` and `_backward_b_tiled_from_lse`.
* The general rule is: B-tiled if and only if `PDP * SEQ_LEN >= 8192`.
* Block sizes:
  * The tuned table gives `(b, h, v) = (256, 256, 2048)` at 130m, 300m and 520m (bucket `large-batch-small-h`) and the
    default `(1024, 512, 1024)` at 1_2b (`CE/tuned_block_sizes.py:68-79, 381-398`).
  * The B-tiled path never reads them.
  * `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` (`L:97`) prevents the 7-candidate sweep at 1_2b.
* Forward, per tile (`CE/batched_xla.py:316-366`, `CE/reference.py:66-110`; `J130:5196-5315`, `jaxpr12/13`):

```
z   = dot(x_t, w, preferred_element_type=f32)              # f32 [8192, V], not rounded to bf16
m   = rowmax(z); m = 0 if not finite(m); m = stop_gradient(m)
lse = log(abs(sum(exp(z - m)))) + m                        # jax.nn.logsumexp, all f32
loss_tok = lse - z[r, label_r]                             # f32
```

* Backward, per tile (`CE/batched_xla.py:551-629`; `J130:4722-4824`, `jaxpr9`). `g_loss = w/sum(w)` (f32: 1/524160 for
  valid tokens) and `g_lse = 0`.

```
zb  = f32(dot(x_t, w))                     # no preferred_element_type: bf16 out, then f32
p   = exp(zb - lse_fwd)                    # f32
dz  = p * (g_loss + g_lse)
dz[r, label_r] += -g_loss_r                # scatter-add; one hit per row, so no collisions
dzb = bf16(dz)
gx_t = f32(dot(dzb, w^T))                  # bf16 GEMM, bf16 out
gw  += f32(dot(x_t^T, dzb))                # bf16 GEMM, bf16 out; f32 accumulator over the 16 tiles
grad_x = bf16(gx);  grad_w = bf16(gw)
```

* After the kernel, `grad_w` is all-reduced across the 4 GPUs **in bf16**: `psum[axes=(data, replica, model,
  replica_dcn)]` on `bf16[512,128256]` (`J130:456-459`). This is the transpose of the replicated `in_spec`.
* P, quirk: the forward `lse` comes from f32 logits, but the backward `p` comes from bf16-rounded logits. So
  `sum(p) != 1`, and the relative error of each `p` is up to about `|z| * 2^-8` (for `|z|` in [8, 16): up to 3 %).
  This holds only if XLA keeps the bf16 output of that dot (open item O1).
* Memory (S): each f32 `[8192, V]` buffer is 4.2 GB.

### 5.3 The other path: `PDP = 1` (`B_dev = 4096 < 8192`)

1. `batched_xla` tries its streaming path with the inferred blocks `(1024, 512, 1024)`, because no bucket matches
   `B_dev = 4096` at any baseline width. The weight tile is `512*1024*2 = 1,048,576 B`, above the H100 limit of
   101,376 B (`CE/batched_xla.py:22, 175-200`). The kernel raises `BatchedXlaUnsupportedError`. `api.py:747-758`
   catches it and falls back to `xla`. The run logs contain this warning 195 times (`grep -rh "falling back to XLA"
   logs/`).
2. `CE/xla.py:681-781` runs with `block_sizes=None`.
   * `v_block = 8192` (`CE/tuned_block_sizes.py:773-794`), `b_block = 4096` (`:797-815`).
   * Forward: V is padded to 131072 with zero columns, which are masked to `-inf`. The kernel streams over 16 vocab
     blocks: f32 logits (`preferred_element_type=f32`), a block logsumexp, and a running `jnp.logaddexp`, all in f32
     (`CE/reference.py:113-194`).
   * Backward: `fast_backward` is off (`LEVANTER_CE_XLA_FAST_BWD` unset; `CE/xla.py:35-69`), so the code is
     `CE/xla.py:280-395`. For each vocab block: f32 logits, `p` in f32, and `delta` in f32 with a scatter-add.
     `gx_blk = bf16(dot(delta_f32, w_bf16))` is an f32 GEMM on upcast `w` (TF32-eligible). `gx` is **accumulated in
     bf16 over the 16 vocab blocks**. `gw_blk = bf16(dot(x, delta))`.
3. Differences from §5.2:
   * Here dlogits stay f32 (more precise).
   * Here `grad_x` is rounded 16 times (less precise).
   * The backward logits are not rounded here.

   This is the "different CE backward" that the pls session reported (`docs/survey/needs_from_sessions.md`, pls item 3).
   **The plain baseline never takes this path.** For training, eval (`per_device_eval_parallelism=32`) and both
   microbatch sizes, `B_dev >= 131072`. Updates 0-85 of the 520m run used `PDP=64`, so `B_dev = 262144` (32 tiles)
   there (§6.3).

### 5.4 Class

* M: f32 logits and lse in the forward, the label shift and mask, the global-mean reduction, and the absence of a
  z-loss and a soft cap.
* S: the implementation and the 8192-row tile, and whether the logits are materialized or streamed.
* P: the bf16 logits in the backward, bf16 dlogits, the bf16 `grad_w` and its bf16 psum, and the bf16 `grad_x`.
* For the parity run, round the recomputed backward logits and dlogits to bf16. The cost is negligible.

## 6. Sharding, data parallelism, microbatching

### 6.1 Mesh

* There is one process with 4 local devices. JAX distributed is not initialized (`SL:37`). The client is the Fray
  LocalClient, which runs the trainer in a thread of the same process (`lib/fray/src/fray/local_backend.py:110-114`).
* Axes: `data=4`, `replica=1`, `model=1`, `replica_dcn=1` (`L:802-811`; `LV/utils/mesh.py:16-19, 73-129`; `H130:2`).
  The partitioner is Shardy (`sdy.mesh` in `H130`).
* Compute mapping: `batch`, `token` and `token_repeat` map to `(replica_dcn, replica, data)`, which is 4-way. `mlp` and
  `heads` map to `model` (size 1) (`mesh.py:42-55`; `WB .config.trainer.mesh`).
* Param mapping: `embed -> data` (the default, `mesh.py:40`). `mlp` and `heads` map to `model`.

### 6.2 Parameter and optimizer-state placement (FSDP / ZeRO-3)

* Every tensor with an `embed` axis is sharded 4 ways along it (`H130` argument shardings).
  * `q_proj [L, KV, G, hd, D]`, `k_proj` and `v_proj [L, KV, hd, D]`, `gate` and `up [L, F, D]`: sharded on the last
    axis.
  * `o_proj [L, D, H, hd]` and `down [L, D, F]`: sharded on axis 1.
  * The 4 block-norm gains `[L, D]` and the final gain `[D]`: sharded on D.
  * The embedding and `lm_head [V, D]`: sharded on D. At 130m each GPU holds 128 of the 512 columns.
  * `q_norm` and `k_norm [L, hd]` have no `embed` axis, so they are replicated.
* The optimizer states have the same sharding as their params (`H130` arguments 21-46).
* In compute, `embed` is not mapped, so params are gathered to full replicas after the bf16 cast. Each GPU then runs
  the whole model on its own `B/4` sequences. This is data parallelism with sharded storage.
* Newton-Schulz adds `with_sharding_constraint(X, P(None, ("data","model")))` (`LV/optim/util.py:281-282`).
* Gradient reductions:
  * The `lm_head` psum is explicit and bf16 (§5.2).
  * The partitioner inserts all other reductions. At the HLO level each partial gradient is bf16 before its f32 cast.
    So the reduce-scatters are most likely bf16, but this is not verified (O1).
* Class S, except that the dtype of the gradient reduction is P. A plain data-parallel layout (replicated params plus
  an all-reduce) is equivalent in exact arithmetic. At these sizes it fits in 80 GB.

### 6.3 Microbatching

* `per_device_parallelism = 32` for all four sizes in the final configs (`WB`).
  * 130m and 300m: the `-1` default resolves to `128 / 4` (`LV/trainer.py:1136-1149`).
  * 520m and 1_2b: set by `L:435`.
* `microbatch_size = PDP * data_axis_size = 128` sequences (`LV/trainer.py:986-990`).
  * Batch 128 (130m, 300m): `microbatched` returns `fn` unchanged (`LV/grad_accum.py:75-76`), so there is one pass.
  * Batch 256 (520m, 1_2b): 2 accumulation steps (`H520`, `H1_2b`: `tensor<2x128x4096xi32>`).
  * The accumulator is f32 and is inferred from the f32 grads (`grad_accum.py:96-98`). It is summed in a `hax.fold`
    and then divided by 2 (`:152-162`).
* 520m exception:
  * The first segment ran at commit `59ac0d6` with `PDP = 64`: one microbatch of 256, `B_dev = 262144`
    (`NC/wandb/run-20260914_063908-...520m...jsonl.gz`, steps 2-119; config `NC/wandb/config/run-20260914_063908-*`).
  * That run saved the temporary checkpoint `step-85` and then failed a 35.5 GiB allocation (`L:432-434`). The next
    segment resumed from `state.step` 86 with `PDP = 32` (`project/marin/logs/slurm/muonh-qwen3-520m_13852140.out:110-111,
    207`). So the surviving trajectory used `PDP = 64` for updates 0-85 and `PDP = 32` from update 86 (`optim.md` §8).
  * The math is the same; only the summation order changed.
* Class S: the mean of microbatch means equals the global mean, because the weights are equal (§3.6).

## 7. Rematerialization and layer scan

* `scan_layers=True`: the layers run under `Stacked` and `lax.scan` (`LV/models/llama.py:392-404`).
* `gradient_checkpointing=True` maps to `ScanCheckpointPolicy(simple=True)`, which is
  `eqx.filter_checkpoint(layer, prevent_cse=False)` with no policy (`HX/_src/scan.py:145-185`). `J130:499-510` shows
  `remat2[policy=None, prevent_cse=False]`.
* Each layer saves only its input carry, `bf16 [B,S,D]`. In the backward it recomputes the whole layer, including the
  flash forward.
* Class S.

## 8. JAX and XLA flags, environment

* JAX config: `jax_threefry_partitionable=True` and `jax_softmax_custom_jvp=True` (`LV/trainer.py:98-101,
  1099-1104`; `WB .config.trainer.jax_config`).
  * The first flag fixes the PRNG streams for init. It matters for seed-level init parity (`model.md` §4.2), and it
    does not change the init distribution.
  * The second has no effect: no `jax.nn.softmax` runs in the training path.
* `XLA_FLAGS`:
  * The sbatch file and the launcher set none.
  * Marin adds `--xla_gpu_nccl_termination_timeout_seconds=600` and `NCCL_DEBUG=WARN`
    (`lib/marin/src/marin/training/training.py:412-433`). They reach `os.environ` through `setdefault` before levanter
    starts (`training.py:523-559`). The XLA backend starts after that (`SL:39` at 04:18:22, after the "Model config"
    line at 04:18:20).
  * No other flag is set: no `--xla_gpu_deterministic_ops`, no autotune level, no TF32 switch.
* Other environment: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (sbatch line 28) with the default BFC allocator, and
  `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` (`L:97`). A persistent JAX compilation cache is on
  (`training.py:472-476`, or the ambient `JAX_COMPILATION_CACHE_DIR`).
* Matmul precision:
  * `jax_default_matmul_precision` is unset, and no dot carries a `precision` attribute (`J130`). So every dot is
    `DEFAULT`.
  * XLA:GPU allows TF32 for f32 x f32 GEMMs at `DEFAULT`. This affects Newton-Schulz (`optim.md` §4.3), the attention
    `dV` and the `PDP=1` CE backward. bf16 GEMMs are unaffected.
  * This is not verified (O2). Class P: keep TF32 for those GEMMs in the parity run.
* Class S for every flag above, except matmul precision.

## 9. Determinism

### 9.1 Sources

1. **Embedding backward.**
   * `weight.take` (`HX/nn/embedding.py:80`) transposes to `scatter-add[unique_indices=False]` of
     `bf16[B,S,D]` updates into a zero `bf16[V,D]` buffer (`J130:516-530`).
   * XLA:GPU implements this with atomics, so the order of the additions changes from run to run. Each GPU adds 131072
     rows, and frequent tokens collide thousands of times.
   * The accumulation is also bf16 (class P). Of all the P items, this one is the most likely to matter: it sums
     thousands of rounded additions for frequent tokens.
   * Earlier diagnosis: `docs/della/objective-hillclimb.md:70` found that this op is the only run-to-run leak, and
     that it differs from step 1 on.
2. **XLA autotuning.** GEMM algorithm choice and fusion tiling can differ between two compilations. The persistent
   cache reduces this (`docs/della/objective-hillclimb.md:70`).
3. **Two compiled step programs.** A step with jit hooks (watch, `interval: 10`, `LV/trainer.py:534-546, 656-658`) is
   compiled separately from a step without hooks. Both are deterministic, but XLA may fuse them differently. Their
   pre-optimization dtypes are identical (§0, "Which step program these files show").
4. Deterministic despite appearances:
   * The CE label scatter: one hit per row.
   * The data order: a pure function of `data_seed=42` (`data.md` §4).
   * The trainer RNG: the baseline has no dropout.
   * The NCCL reductions, for a fixed topology.

### 9.2 Measured floor

The rerun `muonh-qwen3-130m-della4xh100-restore` (2026-09-29) used the same config and seed 0, after the data restore
(identical token counts, `docs/della/sampled-softmax.md:94-104`). Compared with the original (local W&B summaries
`wandb-runs/wandb/run-*/files/wandb-summary.json`):

| metric | original | rerun | difference | in pool sd |
|---|---|---|---|---|
| `eval/paloma/c4_en-marin-tokenizer/bpb` | 1.163581 | 1.163220 | -0.00036 | 1.1 (pool sd 0.00033) |
| `eval/paloma/macro_loss` | 4.17983 | 4.18197 | +0.0021 | 0.4 (pool sd 0.00577) |
| final `train/loss` | 3.23515 | 3.23342 | -0.0017 | – |

* Same-seed H100 pairs differ in `train/loss` by 1e-4 to 1.2e-3 at step 2 (`targets.md` §3.3).
* A run with `--xla_gpu_deterministic_ops=true` was queued but held, and it never ran
  (`docs/della/objective-hillclimb.md:516, 729`). The throughput cost of determinism is therefore unmeasured.
* Consequence: the reference alone shows about 1 sd of run-to-run spread. This floor is what makes every P item
  testable only by distribution.
* For nanomarin, a deterministic embedding backward (sorted segment-sum, f32 accumulator) is S and recommended. It
  makes A/B tests of single P items possible at n = 1 per seed for debugging. Gates still follow `targets.md`.

## 10. Open items (need a GPU node; nothing was submitted)

* **O1. Effective rounding points.** Dump the optimized HLO for one 130m step: `SMOKE_STEPS=2`,
  `XLA_FLAGS=--xla_dump_to=<dir> --xla_dump_hlo_as_text`. Read four things from it:
  1. the dtype of each reduce-scatter and all-reduce (§6.2);
  2. whether the bf16 output of the CE backward logit dot survives (§5.2);
  3. the accumulator dtype of the embedding scatter (§9.1);
  4. the accumulator dtype of the 7 bf16 norm-gain `reduce_sum`s over 524288 positions (§2.2). This is a larger
     rounding effect than item 5;
  5. which bf16 elementwise chains fuse with f32 intermediates;
  6. the flags that fix transcendentals and denormals (§0): `xla_gpu_ftz` and the fast-math options in the dumped
     `DebugOptions`, and whether bf16 `exp`/`logistic`/`rsqrt` are evaluated in f32.
* **O2. TF32 and loaded libraries.** In the same dump, check the cuBLAS calls of f32 dots (`__cublas$gemm`
  `precision_config`/algorithm) for Newton-Schulz and the flash `dV`. In the same job, record the loaded libraries
  (`/proc/<pid>/maps` or `LD_DEBUG=libs`) to confirm which `libcublas`, `libcudnn` and `libnccl` builds ran (§0).
* **O3. Scale of P items.** One A/B at 130m against the 8-run pool: f32 attention softmax and accumulators versus the
  reference bf16 (§4.5). This is the first deviation nanomarin will want.
* **O4. RoPE tables.** On a GPU node, run the `DefaultRotaryEmbeddings` code path once at hd 64 and hd 128 and save the
  f32 `inv_freq` and the bf16 cos/sin tables `[4096, hd]` (§3.3). nanomarin either loads them or tests its own tables
  against them bit for bit.

## 11. Findings that belong to other sections (verified while reading)

* QK-norm is on although `use_qk_norm: false` is logged (`LV/models/qwen.py:374-376`); see `model.md` §2.1.
* `lm_head` and the embedding are drawn with the same key `k_emb` and the same shape `[V, D]`
  (`LV/models/qwen.py:389-395`). The scales are `1/sqrt(D)` for the head (`HX/nn/mup.py:63-64`) and `1/D` for the
  embedding (`HX/nn/mup.py:157-158`; `HX/nn/embedding.py:65-67`). So at init, `lm_head = sqrt(D) * embedding`, to f32
  rounding.
* `weight_decay=0.1` is logged but never read (`LV/optim/muonh.py:54-104`); see `optim.md` §7.
* Each optimizer group is clipped by its own global norm (`J130:598-640`); see `optim.md` §3.1.

## Review log (2026-10-24)

Two skeptic reviews, each issue checked against the code, the venv and the jaxprs. Duplicate issues are merged.

| # | Issue | Verdict | Reason |
|---|---|---|---|
| 1 | §0: software versions cited to `SL:414`; NCCL/cuBLAS builds | applied, refined | `SL:414` names only `NCCL version 2.28.9+cuda12.9`; jax is at `SL:7`. Versions now cite the dist-info directories. Added: the on-disk `libnccl.so.2` and cuDNN files are the cu12 builds (sizes match the cu12 RECORDs); cuBLAS 13.4.1.1 is the only `libcublas.so.13` on the plugin RPATH, so it is likely but not verified. O2 now records the loaded libraries. |
| 2 | §4.4: `D` "accumulated in f32, stored bf16" (both reviews) | applied | `J130:4206-4212`: bf16 product, f32 sum, bf16 store. Now `D = bf16(sum_f32(f32(bf16(dO*O))))`. |
| 3 | §4.4: GQA `dK`/`dV` rounding order (both reviews) | applied | `FA:400-401` cast each per-head partial to bf16 before the group sum at `FA:403-405`; confirmed in `J130:5150-5170` (G=1) and the 520m jaxpr lines 6702-6733 (G=2). Three roundings per block, at every size. |
| 4 | §3.6: denominator 524160 "for every size" | applied | 520m config line 498: `per_device_parallelism: 64` with batch 256, so steps 1-119 used 1048320. |
| 5 | §2.1/§2.2/§3.2: RMSNorm backward rounding points missing (both reviews) | applied | `J130:466-490`: bf16 `g*dy`, f32 internals, one bf16 cast of `dx`; written out in §3.2 with the exact f32 order. |
| 6 | §2.2: all 7 norm-gain gradients are bf16 reductions, not only the final norm's | applied, extended | Confirmed at `J130:468, 4043, 4130, 4159, 4264, 4292`; the review missed the fourth block norm at `J130:4368` (`ejq`). Added as O1 item 4. |
| 7 | §0: provenance of `J130` (hook vs no-hook program) | applied | `LV/trainer.py:540-545` saves the hook-step program as `train_step`. Rechecked: identical `dot_general` signatures (jaxpr and StableHLO) and identical bf16-output equation multisets between the two programs. |
| 8 | §2.1: `L` listed as computed in f32 | applied | `FA:290-293`: `max_i + log(sumexp_i)` on bf16, upcast only at the store (`FA:182`, `FA:293`). |
| 9 | §3.5: SiLU backward rounding order unspecified | applied | `J130:3992-3995, 4105-4109`: all bf16, order written out. |
| 10 | §3.4: scale outside the custom VJP; backward `dq` multiply | applied | `FA:96` before `FA:98`; `J130:4228` (0.125) and 1_2b jaxpr line 5309 (0.08837890625) multiply `dq` in bf16. |
| 11 | §3.3: RoPE table accuracy and `inv_freq` dtype | applied, softened | `rotary.py:35-38` runs eagerly in f32 under `ensure_compile_time_eval`. The reviewer's "about 1e-3 error" for fast-math cos at 4095 rad is not checked here; the spec states the requirement without that number and adds O4 to dump the reference tables. |
| 12 | §0/O1: transcendental implementations and denormals | applied | Added as a stated assumption in §0 and O1 item 6. The denormal case is real (`p * g_loss` below 2^-126 for `p` below about 6e-33) but each such term is below 1.2e-38. XLA's defaults are not verified in code. |

No issue was rejected.

Merge cross-check (2026-10-24): values stated in two or more sections were compared; the corrections are listed in `kaiyue_baseline.md` §C.
