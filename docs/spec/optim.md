# Optimizer: MuonH / AdamH, groups and schedule

Stage-1 spec section. It covers everything between "the f32 gradient of the mean loss exists" and "the f32 parameters of
the next step exist": gradient accumulation, clipping, the three optimizer groups, their update math, the learning-rate
schedules, batch size and step counts.

Every training step does this, in order: (1) average the microbatch gradients, (2) clip each group by its own norm,
(3) transform the gradients: MuonH for the transformer Linear weights, AdamH for `lm_head`, plain Adam for the rest,
(4) apply `p <- p + u`. No weight decay is applied anywhere.

## 0. Sources and notation

Paths are relative to the marin checkout `$ROOT/project/marin` (HEAD `1793624070`). All line
numbers are HEAD line numbers. The runs did not use HEAD. `WB` `config.git_commit` (the commit of the last segment) is
`59ac0d64` (130m), `28fd5996` (300m), `ac6af868` (520m; its first segment ran `59ac0d64`, §8) and `cead09a2` (1_2b).
`git diff <commit> HEAD` over `LV/optim/`, `LV/grad_accum.py`, `LV/trainer_state.py`, `LV/trainer.py`, `LV/callbacks/`,
`LV/models/loss.py`, `LV/models/lm_model.py`, `HX/nn/loss.py` and `HX/quantization.py` shows one change, the same for
all four commits: `209e8d1a9d` (2026-09-28, VARIANT=ss) adds 17 lines to `trainer.py`. It adds the `_TRACED_TRAIN_STEP`
contextvar (HEAD `:108-116`) and splits `_train_step` into a wrapper and `_train_step_body` (HEAD `:742-749`). The math
does not change. In the run commits, every `trainer.py` line from HEAD `:749` on is 17 lines earlier (HEAD `:768` = run
`:751`); `git show 209e8d1a9d^:lib/levanter/src/levanter/trainer.py` is the code the runs used. The optax 0.2.6 and jax
0.11.1 dist-info directories are dated 2026-09-13, before the runs.

| Alias | Path |
|---|---|
| `L` | `experiments/references/della_muonh_qwen3_scaling.py` (launcher, plain baseline: no VARIANT, no knobs) |
| `LV` | `lib/levanter/src/levanter` |
| `HX` | `lib/haliax/src/haliax` |
| `OX` | `.venv/lib/python3.12/site-packages/optax` (optax 0.2.6; jax 0.11.1, equinox 0.13.2) |
| `WB` | `$ROOT/project/nanomarin/cache/wandb/api/reself__marin-della__muonh-qwen3-<size>-della4xh100.json` (logged config and summary of the finished runs `reself/marin-della/muonh-qwen3-<size>-della4xh100`) |

The logged W&B configs (`WB` `.config.optimizer`, `.config.trainer`) agree with `L:426-431` and `L:736-755` for all four
sizes, field by field.

Notation. `k` is the 0-based update index, `k = 0 .. N-1`, with `N = num_train_steps`. All parameters, gradients and
optimizer states are float32 (`mp = p=f32,c=bfloat16`, `L:782`; `WB` `trainer.mp.param_dtype = float32`). `||X||` is the
Frobenius norm, `sqrt(sum(X*X))`.

Upstream origin. The launcher imports `MuonHConfig` from `LV/optim/muonh.py` (`L:52`). Kaiyue Wen's archived recipe
`experiments/speedrun/muonh_qwen3_scaling/{muonh_sweep.py,selected_runs.py}` (git `7cd7b7ea85`, 2026-04-19, since
deleted) uses the same optimizer class. Its values are LR-sweep selections that differ from this baseline: for example,
130m uses cosine, warmup 1000, `adam_lr` 0.004615, `epsilon` 1e-15 and quintic NS coefficients there. This spec follows
the della W&B config, which the task names as authoritative. Do not mix the two sets of values.

## 1. Values per size

From `L:426-431` and `L:736-755`, confirmed against `WB` `.config`:

| | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|
| `learning_rate` (MuonH and AdamH peak) | 0.02 | 0.01 | 0.01 | 0.01 |
| `adam_lr` (Adam peak) | 0.008 | 0.002 | 0.002 | 0.0015 |
| `momentum` (MuonH) | 0.95 | 0.98 | 0.98 | 0.98 |
| `epsilon` (AdamH and Adam) | 1e-20 | 1e-15 | 1e-15 | 1e-15 |
| `max_grad_norm` (per group, §3) | 1.0 | 1.0 | 1.0 | 2.0 |
| `lr_schedule` | linear | cosine | cosine | cosine |
| `warmup` | 0 | 1000 | 1000 | 1000 |
| `decay` | 0.8 | None | None | None |
| `min_lr_ratio` | 0 | 0 | 0 | 0 |
| sequences per step | 128 | 128 | 256 | 256 |
| `num_train_steps` N | 4959 | 11444 | 9918 | 22888 |

These values are the same at every size: `beta1 = 0.9`, `beta2 = 0.98`, `nesterov = True`, `backend_steps = 5`,
`muon_epsilon = 1e-5`, `coefficient_type = "simple"`, `weight_decay = 0.1` (logged, **never read**, §7), `cooldown`,
`cycles`, `cycle_length`, `haps` = None, `rewarmup = 0` (`L:736-755`; `WB`).

## 2. Parameter groups (routing)

### 2.1 Rule

`MuonHConfig.create_mask` (`LV/optim/muonh.py:106-127`) labels every parameter. It builds
`paths = leaf_key_paths(params)` (`LV/utils/jax_utils.py:238-286`), a tree shaped like `params` whose array slots hold
dotted path strings. It maps `mask_fn(param, path)` over `(params, paths)` with `haliax.tree_util.tree_map` and
`is_leaf = isinstance(x, haliax.nn.Linear)`; that map also treats every `NamedArray` as a leaf (`HX/tree_util.py:23-31`).
So `path` is not a string. At a `Linear` node it is the matching `Linear` of the paths tree, whose `weight` is a
`NamedArray` holding the string. At a norm or embedding weight it is a `NamedArray` holding the string. `path_str =
str(path)` (`muonh.py:116`) is the repr of that object, e.g.
`NamedArray(???('vocab', 'embed'), embeddings.token_embeddings.weight)` (`HX/core.py:416-429`). The tests, in order:

1. `path_str` contains `"Embedding"` → `"adam"` (`:117-118`).
2. `path_str` contains `"lm_head"` → `"adamh"` (`:119-120`). The repr of the `lm_head` `Linear` contains
   `lm_head.weight`.
3. The node is a `haliax.nn.Linear` → its `weight` gets `"muonh"` and its `bias` gets `"adam"`
   (`:121-123`; `LV/optim/util.py:33-43`). There are no biases (`use_bias = False`, `WB` `model.use_bias`).
4. Everything else → `"adam"` (`:124-125`).

`optax.multi_transform` (an alias of `optax.partition`, `OX/transforms/_combining.py:176`) then runs one sub-optimizer
per label (`muonh.py:96-102`). Each sub-optimizer receives the full tree with every other group's leaves replaced by an
empty `MaskedNode` (`OX/transforms/_masking.py:31,90-92,131-135`). Labels are recomputed from the gradient tree at each
update (`OX/transforms/_combining.py:274-285`: `labels = param_labels(updates)` at `:275`, one `masked` sub-optimizer per
group at `:278-285`). The gradient tree has the same module structure, so the labels are the same.

Quirk, with no effect here: test 1 is case-sensitive, and no repr contains `"Embedding"` (the paths and axis names are
lower case). The token embedding therefore never matches test 1. It is an `hnn.Embedding`, not a `Linear`, so it falls
through to test 4 and gets `"adam"` anyway. A reimplementation can route by module role: embedding → adam, `lm_head` →
adamh, other Linear weights → muonh, everything else → adam.

### 2.2 Inventory

`D` = hidden, `I` = intermediate, `H` = heads, `KV` = kv heads, `hd = D/H` (64, 64, 64, 128), `L` = layers,
`V = 128256`. The count `2*V*D + L*(2*D*H*hd + 2*D*KV*hd + 3DI + 4D + 2hd) + D` matches the logged
`parameter_count` at all four sizes (130m: 154,154,240; `model.md` §5). All four sizes have `KV*hd` and `H*hd` as listed in §4.3. Leaf names are from the `grad/norm/*` keys
in `WB` `.summary`.

| Group | Leaves (per layer `l` unless noted) | Shape (stored) |
|---|---|---|
| muonh | `self_attn.q_proj.weight` | `[L, KV, H/KV, hd, D]` (Out = (KVHeads, QHeadsPerGroup, HeadSize), In = Embed, out_first; `LV/layers/attention.py:1281-1287`) |
| muonh | `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | `[L, KV, hd, D]` (`attention.py:1288-1301`) |
| muonh | `self_attn.o_proj.weight` | `[L, D, H, hd]` (Out = Embed, In = (Heads, HeadSize); `attention.py:1302-1308`) |
| muonh | `mlp.gate_proj.weight`, `mlp.up_proj.weight` | `[L, I, D]` (`LV/models/llama.py:278-279`) |
| muonh | `mlp.down_proj.weight` | `[L, D, I]` (`llama.py:280`) |
| adamh | `lm_head.weight` (one, not stacked) | `[V, D]` (`LV/models/qwen.py:395`) |
| adam | `embeddings.token_embeddings.weight` (one) | `[V, D]` (`llama.py:482`) |
| adam | `input_layernorm.weight`, `post_attention_layernorm.weight` | `[L, D]` (`llama.py:319-320`) |
| adam | `post_attn_layernorm.weight`, `post_mlp_layernorm.weight` (hybrid_norm = True) | `[L, D]` (`llama.py:323-325`) |
| adam | `self_attn.q_norm.weight`, `self_attn.k_norm.weight` (Qwen3 forces qk_norm) | `[L, hd]` (`qwen.py:374-376`; `attention.py:1312-1314`) |
| adam | `transformer.norm.weight` (one) | `[D]` (`llama.py:390`) |

The layers are stacked (`scan_layers = True`, `llama.py:81,394-398`). `haliax.vmap` prepends the layer axis, so it is
axis 0 of every stacked array (`HX/hof.py:156,168`). The rotary module holds no arrays (`LV/layers/rotary.py:30-32`).
Every floating-point leaf is trainable.

## 3. Step pipeline and order of operations

One training step (`LV/trainer.py:749-768`; `LV/trainer_state.py:225-255`):

1. **Gradient.** `grads = d mean_loss / d params` in f32, through the bf16 compute cast (`trainer.py:754,789-813`). The
   loss section defines the loss and its token mean. `z_loss_weight = 0` (`L:819`).
2. **Accumulation** (520m and 1_2b only, and 520m only from update 86 on, §8). Microbatch gradients are summed in f32 (`LV/grad_accum.py:144`), then
   each leaf is divided by the number of microbatches (`:157-162`). The loss is averaged the same way (`:135,157`).
3. **Hyperparameters.** `optax.inject_hyperparams` evaluates `learning_rate = sched_main(k)` and
   `adam_lr = sched_adam(k)`, casts them to the dtype of the first update leaf (f32), and builds the inner optimizer
   with these two scalars (`muonh.py:104`; `OX/schedules/_inject.py:176-203`). Its schedule counter starts at 0 and
   increments after each update (`_inject.py:261-283`), so **update k uses sched(k)**. The logged `_step` is k: the
   hook logs `log_optimizer_hyperparams(step.opt_state, step=step.step)` (`LV/callbacks/_metrics.py:80`) with
   `StepInfo.step = state.step − 1` (`LV/callbacks/_core.py:76`), and `opt_state.hyperparams` holds the values used in
   that same update (`_inject.py:182-203`). W&B confirms this: 130m logs `optim/learning_rate` = 0.019994957 at
   `_step` 993 = 0.02·(1 − 1/3967) (§6), and 300m logs 2.0000152e-05 = η(2) at `_step` 2. The first logged
   `optim/*` values are at `_step` 2; `_step` 0 and 1 carry only throughput keys.
4. **Per-group transform** (`muonh.py:61-102`). Each group is an `optax.chain`:
   - muonh: `clip_by_global_norm(max_grad_norm)` → `scale_with_muonh(...)` (`:62-77`)
   - adamh: `clip_by_global_norm(max_grad_norm)` → `scale_by_adamh(b1, b2, eps, learning_rate)` (`:79-85`)
   - adam: `clip_by_global_norm(max_grad_norm)` → `scale_by_adam(b1, b2, eps)` → `scale(-adam_lr)` (`:87-94`)
5. **Apply.** `p_new = p + u` for every leaf, in f32 (`trainer_state.py:255`; `HX/quantization.py:103-125` →
   `equinox.apply_updates`, `model + update`). MuonH and AdamH return `u = target − p` (§4.4), so their new parameter
   is the f32 sum `p + (target − p)`, not `target` itself. Reproduce this subtraction-then-add for bit equality.

So the order is **accumulate → clip (per group) → momentum / moments → orthogonalize (MuonH only) → norm-preserving
projection (MuonH, AdamH) → p + u**.

### 3.1 Clipping is per group, not global

Each group's chain begins with its own `optax.clip_by_global_norm`. That clip sees only its own group's leaves, because
the masked leaves are empty `MaskedNode`s with no array leaves. So there are **three** clip norms per step:

```
for group in (muonh, adamh, adam):
    gn = sqrt( sum over leaves g in group of sum(g*g) )        # OX/_src/linear_algebra.py:35-39
    if not (gn < max_grad_norm):                               # OX/transforms/_clipping.py:98; equality also clips
        g <- (g / gn) * max_grad_norm   for every g in group   # :101-104, divide first, then multiply
```

- The muonh norm covers all `7L` transformer Linear weights together.
- The adamh norm is `||grad lm_head||` alone.
- The adam norm covers the embedding and every norm weight together.

The logged `grad/norm/total` is the norm over all parameters. It is **not** any of the three clip quantities. Logged
final values (130m, `WB` `.summary`): total 0.166, lm_head 0.0745, token embedding 0.0087. A reimplementation that clips
by one global norm over all parameters is a different algorithm. The task phrase "global norm" means "global within
each group" in this code.

## 4. MuonH (transformer Linear weights)

`scale_with_muonh(momentum, nesterov=True, steps=5, muon_eps=1e-5, learning_rate, coefficient_type="simple")`
(`LV/optim/muonh.py:136-188`). Its state is one f32 momentum buffer per weight, initialized to zero (`:142-144`). It
has no step counter and no bias correction.

### 4.1 Momentum (Nesterov, without the (1−μ) factor)

With `g` the clipped gradient and `μ = momentum` (`:147-162`):

```
buf <- μ * buf + g            # stored state
u   <- μ * buf + g            # nesterov=True: uses the NEW buf
```

### 4.2 Shaping into matrices

`map_flattened_linear_layers` (`LV/optim/util.py:158-209`) does the following:

1. It flattens each Linear weight to 2-D: all Out axes merge into `__OUT__` and all In axes into `__IN__`, row-major in
   declaration order. The array is then rearranged to `(..., __OUT__, __IN__)`, because `out_first = True` for every
   Linear here (`util.py:96-107`).
2. It vmaps over the stacked layer axis (`HX/tree_util.py:63-66`), so the Newton-Schulz iteration runs **once per
   layer per matrix**.

For `q_proj`, row index = `(kv·(H/KV) + g)·hd + d` and column = embed. For `o_proj`, column index = `h·hd + d`. Newton-Schulz
is equivariant under row and column permutations, so another layout gives the same math. Only the summation order, and
so the last bits, can differ.

### 4.3 Newton-Schulz ("zeropower", `util.py:244-298`)

```
X = U_l (2-D, f32)                          # one layer's [OUT, IN] update matrix
X = X / (||X|| + 1e-5)                      # :266   eps = muon_epsilon = 1e-5, added to the norm
if X.rows > X.cols: X = X.T; T = True       # :270-272   iterate on the wide orientation
repeat 5 times:                             # :285-293   backend_steps = 5
    (a, b, c) = (3.4445, -4.7750, 2.0315)   # "simple": the same triple every iteration (:217, :287 cycles a 1-list)
    A = X @ X.T
    B = b*A + (c*A) @ A                     # source "b * A + c * A @ A"; Python * and @ are left-assoc, same precedence
    X = (a*X) + B @ X
if T: X = X.T                               # :295-296
O_l = X
```

The arithmetic is f32 (no bf16 cast, unlike modded-nanogpt). The sharding constraint at `:281-282` changes placement
only. **Open item: matmul precision.** No code sets `jax_default_matmul_precision` or XLA TF32 flags (grep over
`LV`, `lib/marin`, the launcher and the sbatch). XLA:GPU `DEFAULT` precision lets f32 dots on H100 run in TF32. Most
likely every f32 matmul here, NS included, ran in TF32. Verify on a GPU node (one NS call with `precision=DEFAULT` vs
`HIGHEST` against a levanter dump) before you claim bit-exactness.

Matrix shapes `[OUT, IN]` per layer, and the orientation NS iterates on (wide = rows ≤ cols):

| | q_proj | k_proj, v_proj | o_proj | gate_proj, up_proj | down_proj |
|---|---|---|---|---|---|
| 130m | 512×512 | 512×512 | 512×512 | 1792×512 → transposed to 512×1792 | 512×1792 |
| 300m | 768×768 | 768×768 | 768×768 | 2688×768 → 768×2688 | 768×2688 |
| 520m | 1024×1024 | 512×1024 | 1024×1024 | 3584×1024 → 1024×3584 | 1024×3584 |
| 1_2b | 2048×2048 | 1024×2048 | 2048×2048 | 7168×2048 → 2048×7168 | 2048×7168 |

### 4.4 Norm-preserving projection (the "H")

`norm_preserving_update(p, O, η)` with `η = learning_rate` at step k (`muonh.py:179-184`; `util.py:301-322`). It is
applied to the **unflattened, stacked** arrays, so `p` has ndim > 2 and the axis-0 (layer) branch runs (`util.py:317-322`):

```
for each layer l (reduce over all axes except axis 0):
    n   = sqrt(sum(p_l^2))                              # recomputed from the current p every step; never cached
    m   = sqrt(sum(O_l^2))
    w   = p_l - ((η * O_l) * n) / max(m, 1e-10)         # source "p - lr * u * pn / max(un, 1e-10)", left-assoc
    nw  = sqrt(sum(w^2))
    u_l = (w / max(nw, 1e-10)) * n - p_l                # returned update; the caller then adds p_l + u_l
```

The 2-D branch (`:312-315`) is the same with `jnp.linalg.norm` and without the floor in the final division. It runs for
unstacked 2-D weights, which here means only `lm_head` in AdamH.

Properties:

- **The norm is per layer per matrix.** The ND branch takes one Frobenius norm over all non-leading axes. For a stacked
  `[L, KV, H/KV, hd, D]` array, that is the norm of layer l's whole `q_proj` matrix. It is not a per-head norm.
- **The relative step is exactly η.** Before re-projection, `||w − p|| = η·||p||`, so η = 0.02 moves each matrix by 2%
  of its norm, then rescales it back onto the sphere of radius `n`.
- **`n` is recomputed from the current f32 `p` at every step; never cache it** (`util.py:313`, `:318`). In real numbers
  `n` equals the init norm forever. In f32 each renormalization rounds, so `n` drifts slowly, and a port that caches
  the init norm runs a different, non-bit-exact algorithm. The init is `truncated_normal(-3, 3) / sqrt(fan_in)` (`HX/nn/linear.py:74`;
  `HX/nn/mup.py:63-68`). Its expected norm is `0.98658·sqrt(fan_out_total)`. The logged final norms match the init
  draws: layer-0 `down_proj` 22.316 (130m), 27.347 (300m), 31.572 (520m), 44.672 (1_2b) against 22.32, 27.34, 31.57,
  44.65 expected; `lm_head` 353.32 to 353.40 against 0.98658·sqrt(128256) = 353.32 (`WB` `.summary` `params/norm/*`).
- **First update of the cosine sizes.** At k = 0, η = 0 (warmup starts at 0, §6). Then `w = p` exactly and
  `u = (p/||p||)·||p|| − p`, which is rounding noise, not 0. Reproduce it or accept a 1-ulp difference at step 0.
  The last 1_2b update, k = N−1, is also an η = 0 step of this kind (§6.2).

## 5. AdamH (`lm_head`) and Adam (embedding, norms)

### 5.1 AdamH

`scale_by_adamh(b1=0.9, b2=0.98, eps, learning_rate)` (`LV/optim/adamh.py:105-161`). Its state is
`count` (int32, starts at 0), `mu` and `nu` (f32 zeros; `mu_dtype=None` canonicalizes to f32, `:129-134`).

```
mu    = (1 - b1) * g    + b1 * mu          # OX/tree_utils/_tree_math.py:353-362, term order as written
nu    = (1 - b2) * g**2 + b2 * nu          # :377-397 (g**2 = g*g for real g)
t     = count + 1                          # safe_increment; first update t = 1
mu_h  = mu / (1 - b1**t)                   # :400-411, b**t = XLA pow (§5.3), then 1 - . in f32, then divide
nu_h  = nu / (1 - b2**t)
a     = mu_h / (sqrt(nu_h) + eps)          # adamh.py:143-148; eps OUTSIDE the sqrt, no eps_root
u     = norm_preserving_update(W, a, η)    # adamh.py:152-157; W = lm_head [V, D] is 2-D: util.py:312-315
count = t
```

- AdamH uses the **main** `learning_rate` η (0.02 or 0.01), not `adam_lr` (`muonh.py:83`).
- `lm_head` is not stacked, so its norm is **one Frobenius norm over the whole [V, D] matrix**. It is not per row.
- The projection divides out `||a||`. So bias correction cancels except through `eps`, and `eps` (1e-20 or 1e-15) is
  negligible. Implement it as written anyway, for bit equality.

### 5.2 Adam

`optax.scale_by_adam(b1=0.9, b2=0.98, eps)` then `optax.scale(-adam_lr)` (`muonh.py:91-92`;
`OX/_src/transform.py:246-308,455-467`; `eps_root = 0`, `nesterov = False`). It uses the same `mu`, `nu`, bias
correction and `a = mu_h / (sqrt(nu_h + 0) + eps)` as AdamH, then `u = (-adam_lr) * a`. It has no projection and no
weight decay. Embedding rows absent from the batch still move through their momentum.

### 5.3 Scalar constants and the bias-correction power

Rule 1: a scalar expression made only of Python constants is evaluated in Python float64 and rounded to f32 once, as a
weak-typed scalar, when it meets an f32 array (x64 is off). This covers `(1 - decay)` in `(1 - decay) * (g**order) +
decay * t` (`OX/tree_utils/_tree_math.py:353-362,377-397`; `decay` is the Python float `b1`/`b2` from the closure,
`LV/optim/adamh.py:137-138`, `OX/_src/transform.py:282-283`), the warmup `(init_value - end_value)`
(`OX/schedules/_schedule.py:145`) and the cosine `(1 - alpha)` (`_schedule.py:387`). A port that computes `1.0f - b` in
f32 is not bit-exact. The f32 values:

| Constant | f32 hex | f32 value |
|---|---|---|
| `b1` = 0.9 | `0x3F666666` | 0.89999997615814209 |
| `b2` = 0.98, MuonH `μ` = 0.98 | `0x3F7AE148` | 0.98000001907348633 |
| `1 - b1` (f64, then f32) | `0x3DCCCCCD` | 0.10000000149011612 (not `f32(1) - f32(0.9)` = `0x3DCCCCD0`) |
| `1 - b2` (f64, then f32) | `0x3CA3D70A` | 0.019999999552965164 (not `f32(1) - f32(0.98)` = `0x3CA3D700`, 10 ulp away) |
| MuonH `μ` = 0.95 | `0x3F733333` | 0.94999998807907104 |
| `eps` = 1e-20 | `0x1E3CE508` | 9.9999996826552254e-21 |
| `eps` = 1e-15 | `0x26901D7D` | 1.0000000036274937e-15 |
| `muon_eps` = 1e-5 | `0x3727C5AC` | 9.9999997473787516e-06 |
| `max_grad_norm` 1.0 / 2.0 | `0x3F800000` / `0x40000000` | 1 / 2 |
| NS `a` = 3.4445 | `0x405C72B0` | 3.4444999694824219 |
| NS `b` = −4.7750 | `0xC098CCCD` | −4.7750000953674316 |
| NS `c` = 2.0315 | `0x40020419` | 2.0315001010894775 |
| norm floor 1e-10 | `0x2EDBE6FF` | 1.000000013351432e-10 |
| peak 0.02 / 0.01 | `0x3CA3D70A` / `0x3C23D70A` | 0.019999999552965164 / 0.0099999997764825821 |
| `adam_lr` 0.008 / 0.002 / 0.0015 | `0x3C03126F` / `0x3B03126F` / `0x3AC49BA6` | 0.0080000003799796104 / 0.0020000000949949026 / 0.001500000013038516 |
| warmup `init − end` = −peak (0.01) | `0xBC23D70A` | −0.0099999997764825821 |
| `jnp.pi`, 0.5 (cosine) | `0x40490FDB`, `0x3F000000` | 3.1415927410125732, 0.5 |

Computed with numpy `float32` on della-vis1. The `b1 * mu` and `b2 * nu` terms use `f32(b)` from the table.

Rule 2, bias correction. `tree_bias_correction` is `jax.jit(inline=True)` (`_tree_math.py:400-411`), so the Python
float `decay` enters as a weak f32 argument, `f32(b)`, and `count` is int32. `decay**count` is `jnp.power` with a traced
integer exponent, which takes case 3, `lax.pow(x1, x2)` (`jax/_src/numpy/ufuncs.py:2703-2716`; `lax.integer_pow`, i.e.
repeated multiplication, is used only for a concrete exponent, `:2682-2690`). The lowering converts the int32 exponent
to f32 and emits `stablehlo.power(f32, f32)` (`jax/_src/lax/lax.py:4904-4911`). So the denominator is
`1 - pow_f32(f32(b), f32(t))`, evaluated in f32 by XLA:GPU's f32 pow. It is not repeated multiplication and not a float64
pow. Which device routine XLA:GPU uses for f32 pow (and so its ulp error) is open (§10). AdamH cancels this factor
through the projection, except through `eps`. Plain Adam (embedding, norms) does not.

## 6. Learning-rate schedules

`OptimizerConfig.lr_scheduler(N, override_lr)` (`LV/optim/config.py:286-362`) runs twice. It builds `sched_main` with
peak `learning_rate` (used by MuonH and AdamH) and `sched_adam` with peak `adam_lr` (used by Adam) (`muonh.py:58-59`).
The two schedules have the same shape. `min_lr = peak * min_lr_ratio = 0`. No cooldown, one cycle (`config.py:383-422`).
Warmup and decay lengths convert by `_convert_frac_or_steps`: if `x ≤ 1`, the length is `int(x·N)`, else `int(x)`
(`config.py:441-448`). `T_d` is the number of decay steps (`D` in §2.2 is the hidden size, a different thing).

### 6.1 130m: constant, then linear decay to 0 (WSD-like, no warmup)

- warmup = `int(0·4959)` = 0. decay = `int(0.8·4959)` = `int(3967.2000000000003)` = **3967**. stable = 4959 − 3967 =
  **992** (`config.py:312-332`).
- `optax.join_schedules([constant(peak), linear_schedule(peak, 0, 3967)], boundaries=[992, 4959])`. The second boundary
  is unused, because `zip` truncates (`OX/schedules/_join.py:40-44`).

```
k <  992:  η(k) = peak
k >= 992:  c = clip(k - 992, 0, 3967);  frac = 1 - c/3967;  η(k) = (peak - 0) * frac + 0   # OX/schedules/_schedule.py:142-145
```

`c/3967` is an f32 division of the int32 count (converted exactly) by 3967, and `1 - .` is exact in f32. The last
update, k = 4958, uses `f32(0.02) · (1 − f32(3966/3967))` = **5.041361e-06** (main) and `f32(0.008) · (1 − f32(3966/3967))`
= **2.016544e-06** (adam), both as logged. These are not the real numbers peak/3967 = 5.04159e-6 and 2.01664e-6: the f32
rounding of `3966/3967` before the subtraction makes them 4.5e-5 (relative) lower. Computing `peak*(T_d - c)/T_d` or
`peak/3967` gives different bits at every late step. The rate never reaches 0 during training.

### 6.2 300m, 520m, 1_2b: linear warmup, then cosine to 0

- warmup = 1000 (an integer > 1, so steps). decay = None, so decay = N − 1000 = **T_d**: 10444 (300m), 8918 (520m),
  21888 (1_2b). stable = 0 (`config.py:317-332`).
- `join_schedules([linear_schedule(0.0, peak, 1000), cosine_decay_schedule(peak, T_d, alpha=0.0, exponent=1.0)],
  [1000, N])`.

```
k <  1000: c = clip(k, 0, 1000);  η(k) = (0 - peak) * (1 - c/1000) + peak     # polynomial form; η(0) = 0
k >= 1000: c = minimum(int32(k - 1000), float(T_d))     # promotes to f32; exact, since c < 2^24
           η(k) = peak * ((1 - 0) * (0.5 * (1 + cos((f32(pi) * c) / T_d)))**1.0 + 0)   # OX/schedules/_schedule.py:378-388
```

`(f32(pi) * c) / T_d` is left-associative, two rounded f32 operations (`_schedule.py:386`), and `cos` is XLA:GPU's f32
cos. `**1.0` is `stablehlo.power(x, 1.0)`; XLA's algebraic simplifier should fold it to `x` (unverified).

The warmup is computed in the form `(init − end)·frac + end`, not `peak·k/1000`. This matters for bits: 300m logs
2.0000152e-05 at k = 2, which is this form's f32 rounding. At k = 1000, η = peak.

Near the end, `1 + cos(.)` is quantized to multiples of 2^-24 (the f32 spacing just above −1), so η takes values that
are multiples of `peak·2^-25`. Real-number math does not give these values. The last update, k = N−1, has c = T_d − 1:

| | true `1 + cos(pi·(T_d−1)/T_d)` | η(N−1) main | η(N−1) adam | logged at `_step` N−1 |
|---|---|---|---|---|
| 300m (T_d = 10444) | 4.5e-8 → f32 2^-24 | 2.98023e-10 | 5.96046e-11 | yes, both |
| 520m (T_d = 8918) | 6.2e-8 → f32 2^-24 | 2.98023e-10 | 5.96046e-11 | yes, both |
| 1_2b (T_d = 21888) | 1.03e-8 → f32 **0** | **0** | **0** | yes, both (`_step` 22887); `_step` 22886 logs 2.98023e-10 |

At 1_2b the true value is below half an f32 ulp at −1, so `1 + cos` rounds to 0. The last 1_2b update is therefore an
η = 0 step, like k = 0: `w = p` and `u = (p/||p||)·||p|| − p`, rounding noise only (§4.4). numpy `float32` on della-vis1
reproduces every row (`run-20260914_123815-...-300m`, `run-20260915_170105-...-520m`,
`run-20260917_142202-...-1_2b` in `cache/wandb/`).

**Lookup table.** A bit-exact port should not trust its own cos, pow and division here. W&B logs the exact f32
`optim/learning_rate` and `optim/adam_lr` at `_step` k ≥ 2 (§3 step 3). Load them as a per-k table from the cached
histories `$ROOT/project/nanomarin/cache/wandb/run-2026091[4-7]_*-muonh-qwen3-<size>-della4xh100.jsonl.gz`
(one file per resumed segment; a step re-run after a resume appears once). The table has holes: of k = 2..N−1, 130m
has all 4957 steps, 300m 11435 of 11442, 520m 9900 of 9916, 1_2b 22822 of 22886 (the missing steps are presumably the
first steps of resumed segments). Take the holes and k = 0, 1 from the formulas, and check each against its logged
neighbours. η(0) = 0 for the cosine sizes and peak for 130m; η(1) = `f32(-peak)·(1 − f32(1/1000)) + f32(peak)` for
the cosine sizes.

All schedule arithmetic is f32 with an int32 counter. The peaks are f32 roundings of the decimal values (0.02 →
0.019999999552965164, as logged).

## 7. What is not in the optimizer

- **Weight decay.** `weight_decay = 0.1` is logged, but `MuonHConfig.build` never reads it (`muonh.py:54-104`; compare
  `AdamConfig`, `config.py:515-521`, which does). The fixed-norm projection plays the role of decay for the matrices.
  The docstring says MuonH's lr ≈ sqrt(lr·wd) of a decayed Muon (`muonh.py:39`). The embedding and the norm weights get
  no decay.
- There is no update-RMS clipping, no skip-bad-steps, no z-loss (`L:819`), no EMA (`EMA_BETA = 0`, `L:813`), no
  per-parameter LR multipliers, and no μP scaling (`LinearStandardParam.lr_scale = 1`, `HX/nn/mup.py:71-72`).
- There is no shape-based update scaling (such as `sqrt(max(1, rows/cols))`) in MuonH. The projection replaces it:
  every matrix moves by exactly η of its own norm.

## 8. Batch, steps, accumulation

| | seqs/step | seq len | tokens/step | N | total tokens | GPUs | seqs/GPU/step | microbatches |
|---|---|---|---|---|---|---|---|---|
| 130m | 128 | 4096 | 524,288 | 4959 | 2,599,944,192 | 4 | 32 | 1 |
| 300m | 128 | 4096 | 524,288 | 11444 | 5,999,951,872 | 4 | 32 | 1 |
| 520m | 256 | 4096 | 1,048,576 | 9918 | 10,399,776,768 | 4 | 64 | 1 × 64 for updates 0..85, then 2 × 32 |
| 1_2b | 256 | 4096 | 1,048,576 | 22888 | 23,999,807,488 | 4 | 64 | 2 × 32 |

- 520m and 1_2b set `per_device_parallelism = 32` (`L:435,787`). The microbatch size is 32 × 4 = 128 sequences
  (`trainer.py:987-990`), so there are 2 microbatches. 130m and 300m use `-1`, which resolves to batch/4 = 32
  (`trainer.py:1139-1141`; logged 32). A microbatch of 128 ≥ batch 128 means no accumulation (`grad_accum.py:75-76`).
- **520m exception: its first segment did not accumulate.** Segment `run-20260914_063908` ran commit `59ac0d64`, where
  the launcher has `PER_DEVICE_PARALLELISM = {"1_2b": 32}` only (`git show 59ac0d64:` launcher `:79`). So 520m got
  `-1` → 256/4 = 64 (logged `per_device_parallelism: 64`, `cache/wandb/config/run-20260914_063908-...-520m-...config.yaml:498`),
  a microbatch of 256 ≥ batch 256, and no accumulation. Commit `28fd5996` ("Run 520m with two microbatches per step")
  came later; every later 520m segment ran 2 × 32. That segment logged `_step` 0..119 and saved one temporary
  checkpoint, `step-85` (Slurm log `project/marin/logs/slurm/muonh-qwen3-520m_13852139.out:129`), before it failed. The
  next segment (job 13852140, `run-20260914_083457`) loaded `checkpoints/step-85` and logged "Resuming training from
  step 86" (`muonh-qwen3-520m_13852140.out:110-111, 207`; `LV/main/train_lm.py:456-457` prints `state.step`), then
  "Step 86 is less than the current step 120. Cowardly refusing to log metrics" (`:215`). So updates 0..85 (batches
  0..85) used one microbatch of 64 sequences per GPU, and updates 86..9917 used 2 × 32. The W&B rows at `_step`
  86..119 come from the discarded PDP-64 trajectory. This changes only the summation order (last bits), but a
  bit-exact 520m replay must copy it.
  The 1_2b launcher entry was 32 from `59ac0d64` on, so 1_2b always ran 2 × 32.
- Microbatch j holds sequences `[128j, 128(j+1))` of the step's batch (`grad_accum.py:170-179`, row-major
  `unflatten_axis`). This split changes only the floating-point summation order.
- Accumulation is a layout choice. The math is the mean gradient over all `batch × 4095` loss positions (seq_len − 1
  per sequence) as the loss section defines it. The last position of each sequence has weight 0
  (`LV/data/text/examples.py:200-209`, `GrugLmExample.causal_loss_mask`, used by the text dataset at
  `LV/data/text/datasets.py:159-164`; same rule in `LV/models/loss.py:21-31` and `LV/models/lm_model.py:141-150`). The
  text-LM path sets no ignore id; it multiplies in `example_dict.get("loss_weights")`, which a plain token cache does
  not carry (the loss section owns this check). The reduction is `sum(w·l) / sum(w)` (`HX/nn/loss.py:132-138`). Every microbatch
  therefore has the same weight sum, 128·4095, and only because of that does `(mb0 + mb1)/2` (`grad_accum.py:157-162`)
  equal the global token mean. A change that makes the sums unequal (padding, ignore ids, document masks) breaks the
  equivalence. nanomarin can use any microbatch count, at the cost of last-bit differences.
- Data order (`data_seed = 42`, linear permutation) and init `seed = 0` belong to the data and model sections.

## 9. Reference pseudo-code (one step, f32)

```python
def step(P, G, S, k):                         # P params, G mean grads, S state, k = 0-based update index
    eta, eta_a = sched_main(k), sched_adam(k) # §6, f32 (bit-exact: logged lookup table, §6.2)
    # constants: (1-b1), (1-b2) etc. are f64 then rounded once to f32; b**t is XLA f32 pow (§5.3)
    for grp in ("muonh", "adamh", "adam"):    # §3.1: three separate clip norms
        gn = sqrt(sum((g*g).sum() for g in G[grp]))
        if not gn < max_grad_norm:
            G[grp] = [(g / gn) * max_grad_norm for g in G[grp]]
    for W, g, buf in muonh_leaves:            # per stacked weight, then per layer l
        buf[:] = mu*buf + g; u = mu*buf + g
        for l in layers:
            O = ns5(u[l].reshape(OUT, IN))    # §4.3
            W[l] = W[l] + project(W[l], O.reshape(W[l].shape), eta)   # §4.4, returns target - W[l]
    W = lm_head; m, v, t = adamh_state        # §5.1
    m[:] = (1-b1)*g + b1*m; v[:] = (1-b2)*g*g + b2*v; t += 1
    a = (m/(1-b1**t)) / (sqrt(v/(1-b2**t)) + eps)
    W += project2d(W, a, eta)
    for W, g, (m, v, t) in adam_leaves:       # §5.2
        m[:] = (1-b1)*g + b1*m; v[:] = (1-b2)*g*g + b2*v; t += 1
        W += (-eta_a) * ((m/(1-b1**t)) / (sqrt(v/(1-b2**t)) + eps))
```

## 10. Ambiguities and open items

1. **Matmul precision (TF32 or f32)** of the three Newton-Schulz matmuls per iteration, `X @ X.T`, `(c*A) @ A` and
   `B @ X` (`util.py:289,292,293`; §4.3). The code leaves it to the XLA:GPU default. Verify on a GPU before claiming
   bit equality. Within seed noise, either choice is very likely enough, but this is unverified. The norms are not dot
   ops, so TF32 cannot touch them: `jnp.linalg.norm` (ord None) is `sqrt(sum(real(x*conj(x))))`
   (`jax/_src/numpy/linalg.py:1276-1277`), the ND projection uses `sqrt(sum(square(.)))` (`util.py:318-321`), and the
   clip norm is `sqrt(sum(jnp.sum(abs_sq(x))))` (`OX/_src/linear_algebra.py:35-39`). They are plain f32 reductions;
   only their reduction order is open (item 4). The model's forward and backward matmuls run in bf16 compute.
7. **XLA:GPU f32 transcendentals.** The f32 `pow` in the bias correction (§5.3) and the f32 `cos` in the cosine
   schedule (§6.2) come from XLA:GPU's device routines, whose ulp errors a port will not match by default. For the
   schedule, use the logged lookup table (§6.2). For `pow`, compare a port's `1 - b**t` for t = 1..N against a jitted
   JAX run on a GPU node.
2. **"Global norm" clipping is per group** (§3.1). The code is unambiguous. The reading "one norm over all parameters"
   is wrong for this baseline.
3. **The `"Embedding"` path test never matches** (§2.1). The routing result is the same either way: the embedding goes
   to Adam.
4. **Reduction order.** FSDP sharding (`param_mapping embed→data`, `WB` `trainer.mesh`), the NS sharding constraint
   (`util.py:281-282`) and the Python `sum` over leaves in the clip norm fix a summation order that a reimplementation
   will not match bit for bit. These are last-bit effects only.
5. **Upstream recipe values differ** from the della baseline (§0). Use §1.
6. **Resumed segments.** The 520m and 1_2b runs resumed many times (one W&B id, many local run dirs). A resume
   restores the step and the full optimizer state, so the optimizer math does not change. The 520m microbatch count
   did change at its first resume (§8).

## Review log

Two skeptic reviews, 2026-10-24. Each issue was checked against the code and the cached W&B histories.

| # | Issue | Verdict | Reason |
|---|---|---|---|
| 1 | §6.2 last update "tiny positive" | applied | 1_2b `_step` 22887 logs 0 for both rates; numpy f32 reproduces 0 at T_d = 21888 and 2.98023e-10 at 300m/520m. §6.2 table, §4.4 note. |
| 2 | §8/§3: 520m first segment had no accumulation | applied | Segment `063908` config has `per_device_parallelism: 64` at `59ac0d64`, whose launcher lists only 1_2b. Resume step is an estimate (~85, from timing), stated as such. |
| 3 | §0: trainer.py changed after the runs | applied | `209e8d1a9d` (2026-09-28) adds 17 lines, no math change; it is the only diff from each run commit to HEAD over the listed files. HEAD line numbers stated; per-size commits from `WB` `config.git_commit`. |
| 4 | §2.1 cite `_combining.py:100` | applied | `:100` is in `chain`; the relabeling is `:275`, masking `:278-285`. |
| 5 | §10.1 TF32 "in the norms" | applied | All norms are elementwise products plus reductions, not dot ops. Item limited to the three NS matmuls. |
| 6 | §6.1 "peak/3967" | applied | Logged 5.041361e-06 = `f32(0.02)·(1 − f32(3966/3967))`, reproduced in numpy f32; real peak/3967 = 5.04159e-6. |
| 7 | Gap: rounding of scalar constants | applied | `(1 - decay)` is Python f64, rounded once to f32 (`_tree_math.py:353-362`). New §5.3 with f32 hex of every constant, computed on della-vis1. |
| 8 | Gap: bias-correction power primitive | applied | `tree_bias_correction` is jitted, so `f32(b)**int32 t` → `lax.pow` → `stablehlo.power` with the exponent converted to f32. §5.3; device routine left open (§10.7). |
| 9 | Gap: schedule transcendentals, lookup table | applied | §6.2 quantization note and lookup table. Added that the logged table has holes (300m 7, 520m 16, 1_2b 64 steps missing). |
| 10 | Gap: `n` recomputed, not cached | applied | `util.py:313,318` recompute it from the current `p`. §4.4 bullet and pseudo-code comment fixed. |
| 11 | Gap: two meanings of D | applied | Decay steps renamed `T_d` in §6. |
| 12 | §0 trainer.py (second reviewer) | applied | Same as 3; merged. |
| 13 | §8: `batch × 4096` loss positions | applied | The last position has weight 0. The text path uses `GrugLmExample.causal_loss_mask` (`data/text/examples.py:200-209`), not only the cited `LmExample` code; both mask it. Equal weight sums per microbatch stated as the condition. |
| 14 | §6.1 peak/3967 (second reviewer) | applied | Same as 6; merged, with the warning about `peak*(T_d-c)/T_d`. |
| 15 | §10.1 TF32 (second reviewer) | applied | Same as 5; merged. |
| 16 | §2.1: `path` is not a string | applied | `haliax.tree_util.tree_map` stops at `NamedArray`, so `path` is a `Linear` or `NamedArray` from the paths tree and `path_str` is its repr (`HX/core.py:416-429`). Routing result unchanged. |
| 17 | Gap: where scalar constants fold | applied | Same as 7; merged. Warmup `(init − end)` is at `_schedule.py:145`, not `:144`. |
| 18 | Gap: bias-correction pow (second reviewer) | applied | Same as 8; merged. |
| 19 | Gap: `n` cached (second reviewer) | applied | Same as 10; merged. |
| 20 | Gap: code evidence for `_step` = k | applied | `_metrics.py:80`, `_core.py:76` and `_inject.py:182-203` now cited in §3 step 3. |

No issue was rejected.

Merge cross-check (2026-10-24): values stated in two or more sections were compared; the corrections are listed in `kaiyue_baseline.md` §C.
