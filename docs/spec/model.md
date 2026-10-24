# Stage-1 spec: model architecture and initialization

Scope: the plain `muonh_qwen3` baseline (no `VARIANT`, no PLS knobs, `TIE=0`, `SEQ_LEN=4096`, `SEED=0`) as launched by
`experiments/references/della_muonh_qwen3_scaling.py` through `scripts/della/muonh_qwen3_h100x4.sbatch`.

Citations are `path:line` in `$ROOT/project/marin` at HEAD `1793624070`, except
`levanter/trainer.py`, which is cited at the run commits (§0). Paths under
`lib/levanter/src/levanter/` are shortened to `levanter/`, and paths under `lib/haliax/src/haliax/` to `haliax/`. JAX
is `.venv/lib/python3.12/site-packages/jax` (jax 0.11.1).

## 0. Provenance

| size | W&B run (reself/marin-della) | logged `git_commit` |
|---|---|---|
| 130m | `muonh-qwen3-130m-della4xh100` | `59ac0d647dda` |
| 300m | `muonh-qwen3-300m-della4xh100` | `28fd599610ac` |
| 520m | `muonh-qwen3-520m-della4xh100` | `ac6af86874a6` |
| 1_2b | `muonh-qwen3-1_2b-della4xh100` | `cead09a2590b` |

- Cached configs: `cache/wandb/muonh-qwen3-<size>-della4xh100.config.json`, written by `cache/wandb/_tools/fetch_config.py`
  (paths relative to `project/nanomarin`).
- `git diff --stat <commit> HEAD -- lib/levanter/src/levanter lib/haliax/src/haliax` lists exactly one file for each
  of the four commits: `levanter/trainer.py` (+17 lines, 0 deleted). Every other levanter and haliax file is unchanged,
  including all files this spec cites (qwen.py, llama.py, attention.py, attention_mask.py, flash_attention.py,
  rotary.py, normalization.py, linear.py, embedding.py, mup.py, random.py, loss.py, the fused_cross_entropy_loss
  kernels, flop_utils.py, activation.py, examples.py, datasets.py, train_lm.py, lm_model.py, muonh.py, adamh.py).
  The launcher changed after `59ac0d6`, but only in variant code and `PER_DEVICE_PARALLELISM`. So HEAD line numbers
  describe the code that ran, except in trainer.py.
- The trainer.py change adds `import contextvars`, the `_TRACED_TRAIN_STEP` contextvar with `current_train_step()`,
  and a `_train_step` wrapper that sets it around `_train_step_body`. It tracks the step for step-scheduled losses and
  does not change the math of the baseline. It shifts line numbers: at the run commits `DEFAULT_JAX_CONFIG` is
  trainer.py:97-100 (HEAD 98-101) and the loss-function `cast_to_compute` is trainer.py:244 (HEAD 254). This spec
  cites the run-commit numbers (`git show 59ac0d647dda:lib/levanter/src/levanter/trainer.py`).
- Upstream origin: `experiments/speedrun/muonh_qwen3_scaling/muonh_sweep.py`, first added in `2d6290a99c` ("Hyperball
  speedrun (#1774)", 2025-10-18), deleted on the HEAD line in `bcaf54f338` (2026-04-08), and re-added by Kaiyue Wen in `9f87569c3d`
  (2026-04-19, fixed in `7cd7b7ea85` and `63ac661820` the same day) on branch `upstream/qwen3-speedrun-muonh`, which
  is not an ancestor of HEAD. Read it with `git show 9f87569c3d:<path>`. Its
  `_to_qwen3_from_llama(llama_cfg, seq_len_override=4096, hybrid_norm=True)` converts `experiments/llama.py`
  `llama_150m / llama_300m / llama_600m / llama_1_4b` (`experiments/llama.py:193-227`) into a `Qwen3Config`. The Della
  launcher writes the same shapes by hand (`della_muonh_qwen3_scaling.py:426-431`).
- Model construction: `della_muonh_qwen3_scaling.py:649-664` builds `Qwen3Config(max_seq_len=4096, hidden_dim, intermediate_dim,
  num_layers, num_heads, num_kv_heads, hybrid_norm=True, attn_backend=JAX_FLASH)`. Every other field keeps its
  `LlamaConfig` default (`levanter/models/llama.py:56-88`). The logged config confirms these values.

## 1. Shapes per size

| | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|
| hidden D | 512 | 768 | 1024 | 2048 |
| layers L | 6 | 12 | 24 | 16 |
| query heads H | 8 | 12 | 16 | 16 |
| kv heads KV | 8 (MHA) | 12 (MHA) | 8 (GQA, 2 q per kv) | 8 (GQA, 2 q per kv) |
| head_dim hd = D/H | 64 | 64 | 64 | 128 |
| MLP width F | 1792 | 2688 | 3584 | 7168 |
| F/D | 3.5 | 3.5 | 3.5 | 3.5 |
| seq len S | 4096 | 4096 | 4096 | 4096 |
| vocab V | 128256 | 128256 | 128256 | 128256 |
| batch (seqs/step) | 128 | 128 | 256 | 256 |
| tokens/step | 524,288 | 524,288 | 1,048,576 | 1,048,576 |

Sources: shapes from `della_muonh_qwen3_scaling.py:427-430` and the W&B `model` config; `head_dim=None`, so
hd = `hidden_dim // num_heads` (`levanter/layers/attention.py:1203-1207`; `levanter/models/llama.py:249-254`).

The following settings are the same for every size:
- Tokenizer: `marin-community/marin-tokenizer`, the Llama-3 BPE with a custom chat template. 128000 BPE tokens plus
  256 added specials (`tokenizer.json`), so `V = len(tokenizer) = 128256` (`levanter/main/train_lm.py:250`). BOS
  `<|begin_of_text|>` = 128000; EOS `<|end_of_text|>` = 128001.
- Vocab padding: `round_axis_for_partitioning` (`train_lm.py:251`) does not change V. 128256 is divisible by the
  4-way mesh, and no "Rounding vocab size" event occurs.
- Embeddings: untied (`tie_word_embeddings=False`, logged). The LM head is a separate `hnn.Linear(In=Embed, Out=Vocab,
  use_bias=False)` (`levanter/models/qwen.py:392-395`).
- Biases: none anywhere. `use_bias=False` (`llama.py:83`) controls the q/k/v/o projections (`llama.py:241`; `use_output_bias=None`
  falls back to `use_bias`, `attention.py:1278`), the MLP (`llama.py:317`) and the norms (`llama.py:193`).
- Activation: SiLU-gated GLU (SwiGLU). `activation_function=silu` (`llama.py:63`) maps to `jax.nn.silu`
  (`levanter/utils/activation.py:54`; `haliax/nn/activations.py:38-39`).
- Sliding window: off. `use_sliding_window=False`. The logged `sliding_window=4096` is never used:
  `Qwen3Config.attention_config` does not pass it (`qwen.py:374-376`; `llama.py:234-247`).
- Other off switches: no input-embedding norm (`input_embedding_norm=False`, `llama.py:73`), no dropout, no MoE.
- Execution switches with no effect on the math: `scan_layers=True` (`jax.lax.scan` over a stacked layer axis) and
  `gradient_checkpointing=True` (one remat per layer) (`llama.py:80-81`).

## 2. Forward pass: exact order of operations

The model class is `Qwen3LMHeadModel` (`qwen.py:384-396`). It uses the Llama path:
- `Qwen3LMHeadModel.init` builds `LlamaTransformer`, `LlamaDecoderLayer` and `LlamaEmbedding` (`qwen.py:390-391`).
- `QwenTransformer` and `QwenDecoderLayer` (`qwen.py:143-224`) belong to Qwen2. They are not used here, and they
  have no hybrid norm.

Mixed precision is `jmp` policy `p=f32,c=bfloat16`, output bf16 (`della_muonh_qwen3_scaling.py:782`, logged `mp`):
- Master parameters are f32.
- Every loss evaluation first runs `mp.cast_to_compute(model)`, which casts all floating parameters to bf16
  (`levanter/trainer.py:244` at the run commits; HEAD :254).
- So every matmul and elementwise operation below takes bf16 inputs unless the text says otherwise.

```
# tokens: int32 [B, S]; pos = 0..S-1 for every row (not reset at document starts, see §3)
x = E_bf16[tokens]                                      # Embedding.embed: weight.take(...) * 1   haliax/nn/embedding.py:75-81
for l in range(L):                                      # LlamaDecoderLayer.__call__              llama.py:328-348
    # --- attention branch
    a = RMSNorm_in(x)                                   # input_layernorm                         llama.py:335
    q = a @ Wq^T  -> [B,S,KV,G,hd]   (G = H/KV)         # attention.py:1431-1433
    k = a @ Wk^T  -> [B,S,KV,hd]
    v = a @ Wv^T  -> [B,S,KV,hd]
    q = RMSNorm_q(q) over hd   (one gain vector [hd] per layer, shared by all heads)   attention.py:1435-1437
    k = RMSNorm_k(k) over hd
    q = rope(q, pos); k = rope(k, pos)                  # attention.py:1439-1444; QK-norm runs before RoPE
    o = attention(q, k, v, causal & same-segment)       # §2.3
    o = flatten (kv_head, q_per_group) -> heads; o = o.astype(bf16)   # attention.py:1367-1368
    o = o @ Wo^T                                        # [B,S,D]
    o = RMSNorm_post_attn(o)                            # hybrid_norm: post_attn_layernorm       llama.py:337-338
    x = x + o
    # --- MLP branch
    m = RMSNorm_pre_mlp(x)                              # named post_attention_layernorm (Llama name; it is the pre-MLP norm) llama.py:343
    h = silu(m @ Wgate^T) * (m @ Wup^T)                 # LlamaMlp.__call__                       llama.py:287-294
    y = h @ Wdown^T
    y = RMSNorm_post_mlp(y)                             # hybrid_norm: post_mlp_layernorm        llama.py:345-346
    x = x + y
x = RMSNorm_final(x)                                    # LlamaTransformer.norm                   llama.py:412
logits = x @ W_head^T   (f32 accumulation, f32 output; inside the fused CE)   §2.5
```

The residual stream is never normalized in place. Each block has a pre-norm and a post-branch norm ("hybrid" or
sandwich norm), and a final norm sits before the head. The embedding output has no scale factor (no `sqrt(D)`).

### 2.1 RMSNorm (all 7 kinds: in, q, k, post_attn, pre_mlp, post_mlp, final)

Implementation: `haliax/nn/normalization.py:122-139`, built by `levanter/layers/normalization.py:35-39` from
`LlamaConfig.norm_config` (`llama.py:189-195`). The steps are:

```
x32 = x.astype(float32)          # self.dtype is None -> jnp astype(None) = default float = f32 (jax/_src/numpy/lax_numpy.py:5317-5318)
var = mean(x32**2, axis=normed_axis)
out = (x32 * rsqrt(var + eps)).astype(x.dtype)   # back to bf16
out = weight.astype(bf16) * out                   # gain multiply in bf16, after the cast
```

- eps = `layer_norm_epsilon` = **1e-5** (`llama.py:65`, logged). `LayerNormConfigBase`'s default of 1e-6
  (`levanter/layers/normalization.py:19`) is overridden.
- The gain is learned, `use_layer_norm_weight=True`. It is initialized to **1** (`haliax/nn/normalization.py:46-47`).
  There is no bias.
- Normed axes: D for the block and final norms; hd for q_norm and k_norm, applied per (token, head).
- Read the QK-norm as on, although the logged config says `use_qk_norm: false`. `Qwen3Config.attention_config`
  always sets `qk_norm=self.norm_config` (`qwen.py:374-376`). The logged `parameter_count` matches only with q/k norm
  weights included (§5).

### 2.2 Rotary embedding

`DefaultRotaryEmbeddingsConfig(theta=10000, factor=1.0)` (`levanter/layers/rotary.py:75-90`; logged `rope: {theta: 10000, factor: 1}`).
The implementation is `rotary.py:30-46`:

```
inv_freq[i] = 1 / theta**((2*i)/hd) / factor,  i = 0 .. hd/2-1      # hax.arange(hd/2, step=2)/hd, f32, compile-time
freqs[p, i] = pos[p] * inv_freq[i]                                    # f32
emb = concat(freqs, freqs)  -> [S, hd]
cos, sin = cos(emb).astype(bf16), sin(emb).astype(bf16)
rotate_half(x) = concat(-x[hd/2:], x[:hd/2])                          # rotary.py:17-22 (NeoX / HF-Llama half split, not interleaved)
q' = q*cos + rotate_half(q)*sin                                       # in bf16
```

- RoPE covers all hd dims of q and k. v is not rotated.
- No scaling (factor 1) and no partial rotary.
- `pos_ids` is `None` in training (`levanter/models/lm_model.py:337` calls `activations` without it). So
  `pos = hax.arange(position)` = 0..4095 (`attention.py:1441-1442`).

### 2.3 Attention

- Scale: `1/sqrt(hd)`, computed as a Python float (`attention.py:176-177`) and passed through
  (`attention.py:285-305`) to `q = q * scaling_factor` (`levanter/models/flash_attention.py:96`). q is already bf16
  there (`attention_dtype = x.dtype`, `attention.py:1357`; cast at `flash_attention.py:59-61`). A Python float is
  weakly typed, so JAX first converts the constant to bf16 and then multiplies in bf16:
  `q_scaled = bf16(q_bf16 * bf16(1/sqrt(hd)))`.
  - hd=64: bf16(0.125) = 0.125, exact.
  - hd=128 (1_2b): bf16(0.08838834764831845) = **0.08837890625** (= 181/2048 = 1.4140625 * 2^-4; relative error
    -1.07e-4 against 1/sqrt(128)). A port that multiplies by the f32 or f64 constant differs from levanter.
  - The scale sits outside the custom VJP, so the q gradient passes back through the same bf16 multiply:
    `dq = bf16(dQ * 0.08837890625)` for 1_2b (exact power of two, 0.125, for the other sizes).
  - No logit soft cap (`logits_soft_cap=None`). No attention sink, no gating.
- GQA layout:
  - The `q_proj` output axes are `(kv_head, q_heads_per_group, head_size)` (`attention.py:1281-1287`).
  - The flattened head index is `h = kv*G + g`, and query head `h` reads kv head `h // G`. This is the HF `repeat_kv` order.
  - `o_proj` input axes are `(heads, head_size)` in the same order (`attention.py:1302-1308`).
- Backend: `AttentionBackend.JAX_FLASH` (`della_muonh_qwen3_scaling.py:661`). It falls through `match` to the pure-JAX
  FlashAttention-2 in `levanter/models/flash_attention.py` (`attention.py:285-305`). Its numerics are:
  - Block size 1024 (`flash_attention.py:23`; `flash_attention_block_size=None`), so S=4096 gives 4x4 blocks.
    Causal block skipping keeps `j <= i` (`flash_attention.py:279`).
  - `upcast_attn=False`, so `attention_dtype = x.dtype = bf16` (`attention.py:1357`). The running max, running sum
    and output accumulator are **bf16** (`flash_attention.py:180, 205-206`).
  - Forward, per query block i and key block j = 0..i in ascending order (`flash_attention.py:219-285`):
    ```
    S    = bf16 dot(q_i, k_j)                      # XLA accumulates in f32, rounds to bf16          :228
    S    = where(mask, S, bf16(-1e10))             # -1e10 is a weak Python float -> bf16 -9999220736 :253
    m    = max(m_old, rowmax(S))                   # bf16                                             :265
    P    = exp(S - m)                              # bf16                                             :266
    a    = exp(m_old - m)                          # bf16                                             :267
    r    = rowsum(P)                               # jnp.sum: accumulates in f32, result rounded to bf16 :269
    l    = a*l + r                                 # bf16                                             :270
    o    = a*o + bf16 dot(P, v_j)                  # bf16                                             :273-274
    ```
    `hax.sum` is `jnp.sum` with `dtype=None`, which upcasts bf16 to f32 for the reduction and casts the result back
    (`jax/_src/numpy/reductions.py:140-141, 236`). `max`, `exp` and the rescale stay in bf16.
  - Output: `o_i / l_i` in bf16 (`flash_attention.py:288`).
  - Log-normalizer: `ell_i = m_i + log(l_i)` is computed in **bf16** (`flash_attention.py:290`) and only then cast
    into the f32 buffer `ell` (`flash_attention.py:182, 293`). So `ell = f32(bf16(m + bf16(log(l))))`: f32 storage,
    bf16 precision. Its only consumer is the backward (residual `L`, `flash_attention.py:322`). A bit-exact port must
    round L to bf16 before storing it as f32, not use an f32 log-sum-exp.
  - Masked logits are replaced by bf16(-1e10), not by -inf. This is safe only because of the block order. If every
    key in an early block j < i is masked for a query row (another segment), that row gets `m = -1e10` and `P = 1` for
    every masked entry, so garbage (1024 per block in `l`, a sum of v rows in `o`) accumulates. A later block always
    holds a valid key, at least key = query in the diagonal block j = i (the last one, `:279`). There `m` becomes a
    real logit, `a = exp(-1e10 - m) = 0` exactly, and the garbage is erased. Segments are contiguous, so fully masked
    blocks always come before valid ones. A port that uses -inf, or another block order, must handle fully masked
    blocks itself (`exp(-inf - (-inf)) = NaN`).
  - The backward pass is a custom VJP in the same file (`flash_attention.py:303-450`). Its numerics, for reading 7.4(a):
    ```
    D    = rowsum(dO * O)                          # bf16 product, f32-accumulated jnp.sum, bf16 result  :335
    for j in key blocks, for i = j .. Tr-1 (causal: starts at i = j)                                   :417
      S  = bf16 dot(q_i, k_j); S = where(mask, S, bf16(-1e10))                                         :362, 384
      p  = exp(S - L_i)                            # bf16 - f32 promotes: p is f32, L has bf16 precision :386
      dP = bf16 dot(dO_i, v_j)                                                                          :396
      dS = bf16( p * bf16(dP - D_i) )              # subtraction in bf16, product in f32, cast to bf16    :397-398
      dV_ji = bf16( f32 dot(p, dO_i) )             # f32 p promotes the dot to f32                       :400
      dK_ji = bf16 dot(dS, q_i)                                                                         :401
      dV_ji, dK_ji = sum over q_heads_per_group    # GQA (G=2): f32-accumulated jnp.sum, bf16 result;   :403-405
                                                   #   size-1 axis (exact) for MHA
      dV_j += dV_ji; dK_j += dK_ji                 # bf16 accumulation across query blocks               :407-408
      dQ_i += bf16 dot(dS, k_j)                    # bf16 accumulation across key blocks                 :410
    ```
    Masked entries give `p = exp(-1e10 - L) = 0`, so fully masked blocks contribute exactly 0 here.

### 2.4 MLP

`gate_proj`, `up_proj`: D->F. `down_proj`: F->D. Output = `down(silu(gate(m)) * up(m))` (`llama.py:268-294`). The
elementwise math is bf16 in the JAX program. XLA:GPU may compute fused elementwise ops in f32 internally; this is a
compiler detail and does not show in the source.

### 2.5 Head, logits, loss knobs

- Loss path: `compute_next_token_loss` (`lm_model.py:319-354`) calls `maybe_fused_next_token_loss`
  (`levanter/models/loss.py:35-98`) with the final-normed hidden states (bf16) and `get_lm_head()` = `lm_head.weight`
  (bf16 [V, D]).
- The kernel runs inside `shard_map(..., check_rep=False)` over the compute mapping (`loss.py:291`). Each GPU sees
  its own rows, flattened to `x: [32*4096 = 131072, D]` per microbatch (logged `per_device_parallelism=32` for every
  size), and the full replicated head `w: [D, V]` (`embed` is mapped only in `param_mapping`).
- On GPU the kernel is `batched_xla` (`levanter/kernels/pallas/fused_cross_entropy_loss/api.py:160-169`), called with
  `precision=None` (`loss.py:49` → `api.py:742`). The loss dtype is f32: `loss_weight` is f32 (`examples.py:148-149`),
  and `loss.py:79-80` takes the dtype from it.
- Dispatch on H100: x and w are bf16, 131072 ≥ 8192 rows and V = 128256 ≥ 65536, so
  `_h100_full_vocab_b_tiled_block_size` returns 8192 (`batched_xla.py:29, 33-34, 369-389`). Both directions take the
  b-tiled path with 8192-row tiles, 16 tiles per GPU per microbatch: the forward at `batched_xla.py:432-441` →
  `:317-366`, the backward through the custom VJP (`:757, 780-791` → `_backward_b_tiled_from_lse`, `:552-629`).
- Forward, per tile (`reference.py:93-108`, called at `batched_xla.py:348`): `logits = dot_general(x, w,
  preferred_element_type=float32)` (f32 output), then f32 logsumexp and `loss = lse - logits[label]`.
- Backward, per tile. It recomputes the logits and does **not** differentiate the f32 forward:
  ```
  logits = f32( dot_general(x_blk, w) )            # no preferred_element_type: bf16 output, then upcast   :588-593
  probs  = exp(logits - lse_f32)                   # f32, but logits carry bf16 rounding                    :596
  dlog   = probs * (g_loss + g_lse)                # f32; g_lse = 0 (no z-loss)                             :597
  dlog[r, label_r] += -g_loss_r                    # f32                                                    :603
  dlog_b = bf16(dlog)                                                                                      # :610
  grad_x_blk = f32( bf16 dot(dlog_b, w^T) )                                                                # :611-616
  grad_w    += f32( bf16 dot(x_blk^T, dlog_b) )    # f32 sum over the 16 tiles                              :617-625
  return bf16(grad_x), bf16(grad_w)                                                                        # :629
  ```
  A port that autodiffs an f32-logit CE gets different softmax gradients. A bf16 logit is off by up to half an ulp,
  about |logit| * 2^-8 (0.0625 for |logit| in [16, 32)), and that is the relative error of each prob (up to ~6% there).
  The `grad_w` of each GPU is then summed over the 4 GPUs by the shard_map transpose (a psum of the bf16 cotangent,
  `jax/_src/shard_map.py:1955-1956`, since `check_vma=False`; HLO not inspected).
- Reduction: the kernel returns per-token loss (`reduction=None`, `loss.py:268`; `api.py:818-819`). Outside the
  shard_map, `maybe_reduce_loss` with `reduction=hax.mean` and the weight computes `sum(w*loss) / sum(w)` over
  (batch, position) (`loss.py:299` → `haliax/nn/loss.py:132-138`). w = 1 for positions 0..4094 and 0 for 4095
  (`examples.py:204`; `loss.py:22-32`). The sums run over the global microbatch, `B_micro` = 128 sequences
  (4 GPUs x 32) for every size (one exception: updates 0..85 of the 520m run used `B_micro` = 256, `optim.md` §8). So the loss of one microbatch is divided by `B_micro * 4095`, not by
  `B_micro * 4096`, and each row's `g_loss` is `w / (B_micro * 4095)` in f32. With 2 microbatches (520m/1_2b) both
  microbatch means have the same denominator; the accumulation belongs to the training-loop section.
- No logit scaling, no soft cap (`logit_soft_cap=None`), no muP output multiplier (`LinearStandardParam.active_scale = 1`,
  `haliax/nn/mup.py:66-68`).
- **z-loss: none.** `z_loss_weight=0.0` (`della_muonh_qwen3_scaling.py:819`, logged `z_loss_weight: 0`) is passed as
  `logsumexp_weight` (`train_lm.py:205`); `api.py:817` skips the term when the weight is 0.
- Targets are `roll(tokens, -1)`. The last position has weight 0 (`loss.py:22-32, 76`).

### 2.6 Parameter-gradient dtypes

- The loss function runs on `mp.cast_to_compute(model)` (`trainer.py:244` at the run commits). The transpose of an
  f32→bf16 cast is a bf16→f32 cast of the cotangent. So every f32 master gradient is a bf16 cotangent upcast to f32:
  f32 storage, bf16 precision per microbatch.
- `lm_head`: the f32 sum over 8192-row tiles is cast to bf16 inside the kernel (`batched_xla.py:629`), summed over GPUs
  in bf16 (§2.5), then upcast.
- Token embedding: the gradient is the transpose of `weight.take` on the bf16 table (`haliax/nn/embedding.py:80`),
  a bf16 scatter-add into [V, D] over 131072 tokens per GPU per microbatch.
- Unverified: whether XLA:GPU accumulates that scatter-add in bf16 or f32, and in which dtype the SPMD partitioner
  places the cross-GPU reduction of the other gradients (before or after the bf16→f32 convert). Both are compiler
  decisions that the JAX source does not fix. Pin them down from the optimized HLO of one train step
  (`XLA_FLAGS=--xla_dump_to=...`), or treat them as out of scope for bit-exactness.

## 3. Masking: causal plus cross-document

- Data: `LmDataConfig.block_cross_document_attention = True` (default `levanter/data/text/datasets.py:658`; logged
  `true`). Text components are unpacked (`_effective_pack` returns `False`, `datasets.py:362-367`). So each training
  example is `CausalLmDataset(TokenSeqDataset(cache, 4096))` (`datasets.py:529-533`) with
  `eos_id = tokenizer.eos_token_id = 128001` (`datasets.py:775`).
- `TokenSeqDataset` cuts the flat token stream into non-overlapping 4096-token windows (`datasets.py:78-110`). A window
  can start or end in the middle of a document.
- Segment ids (`levanter/data/text/examples.py:166-171`):
  ```
  eos_mask[p] = (tokens[p-1] == 128001)  for p >= 1;  eos_mask[0] = False
  seg[p] = cumsum(eos_mask)[p]
  ```
  The EOS token belongs to the document it ends. The token after it, normally the next document's BOS, opens a new segment.
- The attention mask allows query p to attend to key k iff `k <= p` and `seg[k] == seg[p]` (causal mask AND segment
  mask, `levanter/layers/attention_mask.py:105-166`; segment equality at `attention_mask.py:17-37`).
- Rotary positions are **not** reset at segment starts (§2.2). A document that starts at window offset 1000 sees
  positions 1000, 1001, ... Only relative distances within a segment enter attention, so this matters only through
  RoPE's relative form, which is shift-invariant. A reimplementation must still keep absolute window positions to
  match bit-for-bit, because bf16 cos/sin rounding depends on the absolute position.
- Loss weights: 1 for positions 0..4094, 0 for position 4095 (`examples.py:204`). No `ignore_id`, and no masking at
  document boundaries. So the EOS position is trained to predict the next document's first token.
- BOS/EOS framing inside the cache (`fineweb-edu-10B/2026.06.28`, 10,000,000,738 tokens) belongs to the data section.
  The tokenizer's post-processor prepends BOS, and the data config logs `enforce_eos: true`.

## 4. Initialization

### 4.1 Distribution and scale of every parameter

Every random tensor is `jax.random.truncated_normal(key, -3, 3, shape, float32) * scale`
(`haliax/random.py:125-131`; JAX algorithm at `jax/_src/random/core.py:1127-1147`):

```
a = erf(-3/sqrt2), b = erf(3/sqrt2); u = uniform(key, shape, f32, minval=a, maxval=b)
t = sqrt2 * erf_inv(u); t = clip(t, nextafter(-3, +inf), nextafter(3, -inf))
```

The truncated normal is **not** re-standardized. Its std is 0.98658, so the effective std is `0.98658 * scale`.
`initializer_range=0.02` is logged but **no parameter uses it**.

Linear layers use `scale = 1/sqrt(fan_in)` (`LinearStandardParam.init_scale`, `haliax/nn/mup.py:63-64`; `haliax/nn/linear.py:73-74`).
The embedding uses `scale = 1/D` (`EmbeddingStandardParam.init_scale(In=Vocab, Out=Embed) = 1/axis_size(Embed)`,
`mup.py:157-158`; `haliax/nn/embedding.py:64-67`).

| parameter (per layer unless noted) | stored shape (axes order) | init | scale | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|---|---|---|
| `embeddings.token_embeddings.weight` (once) | [V, D] | tn | 1/D | 1.953e-3 | 1.302e-3 | 9.766e-4 | 4.883e-4 |
| `lm_head.weight` (once) | [V, D] (out_first) | tn, **same key as embedding** | 1/sqrt(D) | 0.04419 | 0.03608 | 0.03125 | 0.02210 |
| `self_attn.q_proj.weight` | [L, KV, G, hd, D] | tn | 1/sqrt(D) | 0.04419 | 0.03608 | 0.03125 | 0.02210 |
| `self_attn.k_proj.weight` | [L, KV, hd, D] | tn | 1/sqrt(D) | same | same | same | same |
| `self_attn.v_proj.weight` | [L, KV, hd, D] | tn | 1/sqrt(D) | same | same | same | same |
| `self_attn.o_proj.weight` | [L, D, H, hd] | tn | 1/sqrt(H*hd) = 1/sqrt(D) | same | same | same | same |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | [L, F, D] | tn | 1/sqrt(D) | same | same | same | same |
| `mlp.down_proj.weight` | [L, D, F] | tn | 1/sqrt(F) | 0.02362 | 0.01929 | 0.01670 | 0.01181 |
| `input_layernorm`, `post_attention_layernorm`, `post_attn_layernorm`, `post_mlp_layernorm` `.weight` | [L, D] | ones | | | | | |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | [L, hd] | ones | | | | | |
| `transformer.norm.weight` (once) | [D] | ones | | | | | |

Sources: `qwen.py:387-396`, `llama.py:268-285, 306-326, 392-404, 480-486`, `attention.py:1272-1319`.

- No depth-dependent or residual-branch scaling: there is no `1/sqrt(2L)` and no zero-init output projection.
- No biases exist (§1).

### 4.2 PRNG key tree (needed for bit-exact init)

- PRNG settings: legacy `jax.random.PRNGKey` (threefry2x32), with `jax_threefry_partitionable=True`
  (`levanter/trainer.py:97-100` at the run commits; HEAD 98-101). This makes values independent of sharding.
- Seed: `trainer.seed = 0` (`della_muonh_qwen3_scaling.py:812`, logged).

```
data_key, loader_key, model_key, training_key = split(PRNGKey(0), 4)     # train_lm.py:218 (data_key is then replaced by PRNGKey(42))
k_t, k_emb = split(model_key, 2)                                         # qwen.py:389
layer_keys = split(k_t, L)                                               # shaped_rng_split, haliax/jax_utils.py:114-126; Stacked init = vmap over layer_keys
  per layer: k_attn, k_mlp = split(layer_key, 2)                         # llama.py:308
             k_q, k_k, k_v, k_o = split(k_attn, 4)                        # attention.py:1279
             k_gate, k_up, k_down = split(k_mlp, 3)                       # llama.py:277
embedding.weight = tn(k_emb, [V, D]) * (1/D)                             # qwen.py:391
lm_head.weight   = tn(k_emb, [V, D]) * (1/sqrt(D))                       # qwen.py:395  <- same key, same shape
```

**Consequence: the LM head starts exactly proportional to the input embedding.** Both tensors come from the
identical f32 array `t = tn(k_emb, [V, D])`:
- `embedding = t * f32(1/D)` and `lm_head = t * f32(1/sqrt(D))`.
- So row v of the head is `sqrt(D)` times row v of the embedding, up to one f32 rounding each.
- The tables are untied and diverge from step 1 on.
- A reimplementation that draws the two tables independently changes the initial head–embedding correlation from 1 to 0.
  This is a distributional change, not seed noise. Reproduce it: draw one table and scale it twice.

Bit-exact init outside JAX has two options:
- Port threefry2x32 (partitionable mode) and the uniform→erf_inv truncated normal above. All steps are f32
  (`jax/_src/random/core.py:511-553, 1127-1147`):
  ```
  lower, upper = f32(-3), f32(3); sqrt2 = f32(np.sqrt(2))
  a = lax.erf(lower / sqrt2); b = lax.erf(upper / sqrt2)          # f32                        core.py:1133-1137
  bits = _random_bits(key, 32, shape)                              # uint32, threefry2x32 (partitionable)  :532
  f = bitcast_f32((bits >> 9) | 0x3F800000) - 1.0                  # 23 random mantissa bits, f in [0, 1)  :541-550
  u = max(a, f * (b - a) + a)                                      # core.py:551-553
  t = clip(sqrt2 * lax.erf_inv(u), nextafter(lower, +inf), nextafter(upper, -inf))   # core.py:1141-1147
  ```
  `lax.erf` and `lax.erf_inv` lower to XLA's own f32 polynomial/rational approximations, which need not match
  libm bit for bit, so a bit-exact port must copy those too.
- Simpler: export the step-0 parameters once from levanter with seed 0 and load them. I did not check whether a
  step-0 checkpoint exists; the launcher keeps only `every=10000` and the final checkpoint.
  For the stage-1 criterion (match within seed noise) a correct distribution with the shared head/embedding draw is
  enough. Exact init is only needed for step-level A/B tests.

### 4.3 Init norms are frozen by the optimizer

The optimizer groups come from `MuonHConfig.create_mask` (`levanter/optim/muonh.py:106-127`). Leaf paths are dataclass
field names (`levanter/utils/jax_utils.py:238-280`).

| group | parameters | norm behaviour |
|---|---|---|
| `muonh` | every `hnn.Linear` weight in the layers: q, k, v, o, gate, up, down | per-layer Frobenius norm held at its init value (`levanter/optim/util.py:304-322`: stacked [L, ...] tensors keep each layer's matrix norm) |
| `adamh` | `lm_head` (path contains `lm_head`) | whole-table Frobenius norm held at its init value (`levanter/optim/adamh.py:153`) |
| `adam` | token embedding, all RMSNorm gains (including q/k norms) | free (plain Adam, `muonh.py:86-93`) |

The `"Embedding" in path_str` test (`muonh.py:117`) never matches, because the path is `embeddings.token_embeddings.weight`,
in lowercase. The embedding reaches `adam` through the final `else` instead (`muonh.py:124-125`). The result is the same.

So the init scales in §4.1 fix the weight norms for the whole run. Expected norms are `0.98658 * sqrt(numel) * scale`.
The realized f32 value at step 0 is what stays constant.

| matrix (per layer) | expected norm formula | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|---|
| q_proj, o_proj, down_proj | 0.98658*sqrt(D) | 22.32 | 27.34 | 31.57 | 44.65 |
| k_proj, v_proj | 0.98658*sqrt(KV*hd) | 22.32 | 27.34 | 22.32 | 31.57 |
| gate_proj, up_proj | 0.98658*sqrt(F) | 41.76 | 51.15 | 59.06 | 83.53 |
| lm_head (whole table) | 0.98658*sqrt(V) | 353.3 | 353.3 | 353.3 | 353.3 |

## 5. Parameter count

Count per layer = `D*H*hd + 2*D*KV*hd + H*hd*D` (attention) + `3*D*F` (MLP) + `4*D` (2 pre-norms + 2 hybrid post-norms)
+ `2*hd` (q/k norm). Total = `L*layer + D` (final norm) `+ 2*V*D` (embedding + untied head).

| | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|
| per layer | 3,803,264 | 8,555,648 | 14,160,000 | 56,631,552 |
| non-embedding (L*layer + D) | 22,820,096 | 102,668,544 | 339,841,024 | 906,106,880 |
| embedding V*D (= head V*D) | 65,667,072 | 98,500,608 | 131,334,144 | 262,668,288 |
| **total** | **154,154,240** | **299,669,760** | **602,509,312** | **1,431,443,456** |
| W&B `parameter_count` (`train_lm.py:348`) | 154,154,240 | 299,669,760 | 602,509,312 | 1,431,443,456 |

All four match exactly. `LlamaConfig.total_trainable_params` (`llama.py:212-232`) omits the q/k-norm term. It gives
`L*2*hd` fewer parameters (130m: 154,153,472), and that is the value speedrun reports quote.

## 6. FLOPs per token and MFU (as levanter logs them)

`flops_per_token` is `LlamaConfig.flops_per_token` (`llama.py:200-210`; `Qwen3Config` inherits it, while
`QwenConfig.flops_per_token` at `qwen.py:115` is not used). It calls `lm_flops_per_token(glu=True)`
(`levanter/utils/flop_utils.py:5-61`). These are forward FLOPs per token with `hd' = D/H` (a float) and `S = 4096`:

```
mlp      = 2*3*D*F
o_proj   = 2*D*D                                   # "dense_proj"
qkv      = 2*D*(H*hd' + 2*KV*hd')
attn     = 2*S*H*hd' + 3*S*H + 2*S*hd'*H           # QK^T + softmax/mask + PV over the FULL S x S map (no causal halving, no segment discount)
lm_head  = 2*D*V
fpt      = L*(mlp + o_proj) + L*(qkv + attn) + lm_head
```

The trainer then computes:

```
flops_per_example = 3 * fpt * S                                   # train_lm.py:388-389 (fwd + 2x bwd; V = len(tokenizer) = 128256)
MFU[%] = flops_per_example * batch / step_seconds / (n_gpu * 989.5e12) * 100   # levanter/callbacks/_metrics.py:50-67; fray device_flops.py:21-27 ("h100" bf16 = 1.979e15/2)
```

| | 130m | 300m | 520m | 1_2b |
|---|---|---|---|---|
| fpt (forward) | 227,868,672 | 555,024,384 | 1,349,517,312 | 2,877,292,544 |
| of which attention S-terms | 50,921,472 | 152,764,416 | 407,371,776 | 540,016,640 |
| 3*fpt (train FLOPs/token) | 683,606,016 | 1,665,073,152 | 4,048,551,936 | 8,631,877,632 |
| flops_per_example | 2.800050241536e12 | 6.820139630592e12 | 1.6582868729856e13 | 3.5356170780672e13 |
| W&B `throughput/flops_per_example` | identical | identical | identical | identical |

- Peak: W&B `throughput/theoretical_flops` = 3.958e15 for 4x "NVIDIA H100 80GB HBM3".
- Check: the 130m step at `_step` 2 took 0.61818 s, so 2.80005e12 * 128 / 0.61818 / 3.958e15 = 14.648 %. Logged:
  14.648 %.
- For nanomarin, keep this exact formula when reporting MFU, so numbers compare with levanter's. It omits norms,
  RoPE, the embedding gather and the causal saving.

## 7. Traps and ambiguities

1. `initializer_range=0.02` is logged and unused. Embedding std is ≈ 0.98658/D, and linear std is ≈ 0.98658/sqrt(fan_in).
2. `use_qk_norm: false` is logged, yet QK-norm is on (§2.1). The parameter count confirms it.
3. `lm_head` and the embedding share PRNG key `k_emb`, so the head starts as `sqrt(D)` × embedding (§4.2).
4. The flash-attention softmax statistics and accumulator are bf16 with 1024 blocks (§2.3). There are two readings
   for nanomarin:
   - (a) Bit-exact: replicate levanter's bf16 online softmax (f32-accumulated row sums), block 1024, ascending key
     blocks, bf16(-1e10) fill, bf16-precision L, and the bf16/f32 backward of §2.3.
   - (b) Practical: FA2/FA3/cuDNN with f32 statistics. This is numerically *better* and expected inside seed noise,
     but unverified. Measure it against the n=8 pool before claiming parity.
5. RoPE positions are absolute in the 4096 window, while attention is segment-blocked (§3).
6. No loss masking at document boundaries. The EOS position predicts the next document's BOS (§3).
7. The `"Embedding"` path test in `create_mask` is dead code. The embedding still goes to plain Adam (§4.3).
8. The logged `per_device_parallelism=32` differs from the HEAD launcher's -1 for 130m/300m. Microbatching does not
   change the model math (gradient accumulation over a fixed global batch); the training-loop section covers it.
9. `Qwen3Config.reference_checkpoint="Qwen/Qwen3-0.6B"` and `max_seq_len=4096` affect only HF export, which is off
   (`hf_save_steps=None`). They do not affect the model.
10. The 1_2b attention scale is bf16(1/sqrt(128)) = 0.08837890625, not 0.0883883... (§2.3).
11. The fused CE backward recomputes bf16-rounded logits and casts dlogits to bf16; it is not the gradient of the
    f32 forward (§2.5). Gradient accumulation dtypes inside XLA are unverified (§2.6).

## Review log

Two skeptic reviews, checked against the code at the run commits and HEAD `1793624070`.

| # | issue | verdict | reason |
|---|---|---|---|
| 1 | §2.3 hd=128 scale is bf16(1/sqrt(128)) = 0.08837890625 | applied | `attention.py:176-177` gives a Python float; `flash_attention.py:96` multiplies bf16 q; weak typing converts the constant to bf16 first |
| 2 | §2.3 `ell` has bf16 precision despite f32 storage | applied | `flash_attention.py:290` adds bf16 `max_i` and bf16 `log(sumexp_i)`; `:293` only casts into the f32 buffer of `:182` |
| 3 | §0 trainer.py differs from all four run commits | applied | `git diff --stat` shows +17 lines in trainer.py (contextvar step tracking, no math change); run-commit lines are 97-100 and 244; now cited that way |
| 4 | §2.5 CE backward numerics differ from an f32 CE backward | applied | `batched_xla.py:588-593` has no `preferred_element_type`, `:610` casts dlogits to bf16, `:625-629` sums f32 then casts; H100 routing at `:369-389` holds for 131072 rows and V=128256 |
| 5 | §2.5 loss reduction is sum(w*l)/sum(w), denominator B*4095 | applied | `loss.py:299` → `haliax/nn/loss.py:132-138`; weights from `examples.py:204` |
| 6 | §2.3 row sum accumulates in f32 | applied | `hax.sum` → `jnp.sum(dtype=None)`, which upcasts bf16 (`reductions.py:140-141, 236`) |
| 7 | §2.3 `-1e10` fill relies on the diagonal block coming last | applied | a fully masked early block sets max to -1e10 and P to 1; a later valid block makes `exp(-1e10 - m) = 0` (`flash_attention.py:253, 265-274, 279`) |
| 8 | duplicate of 2 | applied (merged with 2) | same evidence; adds that L feeds `p = exp(S - L)` at `flash_attention.py:386` |
| 9 | duplicate of 1 | applied (merged with 1) | same evidence; the bf16 conversion of `-1e10` (to -9999220736) is now stated too |
| 10 | duplicate of 4 | applied with a fix | same evidence; its bound "abs(logit) * 2^-9" is too small: half a bf16 ulp is up to abs(logit) * 2^-8 (0.0625 at abs(logit) = 16), so the spec states 2^-8 |
| 11 | §2.3 flash-attention backward numerics missing | applied | `flash_attention.py:335-417` checked line by line; also added that dV is bf16-accumulated and that GQA sums over `q_heads_per_group` (`:403-405`) |
| 12 | duplicate of 5 | applied (merged with 5) | per-token loss from `api.py:818-819` with `reduction=None` (`loss.py:268`) |
| 13 | parameter-gradient dtypes missing | applied as new §2.6 | bf16 cotangents upcast by the `cast_to_compute` transpose; lm_head bf16 at `batched_xla.py:629` and psum in bf16 by the shard_map transpose (`shard_map.py:1955-1956`); embedding scatter and cross-GPU reduction dtypes marked unverified |
| 14 | §4.2 uniform sampler details missing | applied | `jax/_src/random/core.py:511-553` (`bits >> 9 \| 0x3F800000`, minus 1, `max(a, f*(b-a)+a)`) and `:1127-1147`; added that XLA's erf/erf_inv are not libm |

No issue was rejected.

Merge cross-check (2026-10-24): values stated in two or more sections were compared; the corrections are listed in `kaiyue_baseline.md` §C.
