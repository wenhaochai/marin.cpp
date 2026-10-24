# Kaiyue muonh_qwen3 baseline: stage-1 spec

Merged 2026-10-24 from the six verified section files in this directory (`model.md`, `optim.md`, `data.md`,
`eval.md`, `numerics.md`, `targets.md`). The section files stay; their review logs live there only. Parts 1-6 below
are those files with the cross-check corrections of §C applied (the same corrections were applied to the section
files). A reference such as "`optim.md` §6.2" points to Part 2 §6.2. Each part keeps its own path aliases (in
`optim.md` and `numerics.md`, `L` is the launcher; in `data.md` and `eval.md`, `L/` is levanter), defined at the top of
that part.

## Summary

The baseline is a Qwen3-style decoder (untied embeddings, QK-norm, sandwich "hybrid" RMSNorm with eps 1e-5, SwiGLU at
F = 3.5 D, RoPE theta 10000 on absolute window positions, no biases, no z-loss) trained by levanter on 4x H100 with
jmp policy p=f32/c=bf16 and a pure-JAX FlashAttention-2 whose softmax statistics are bf16. Every parameter is drawn
from an un-restandardized truncated normal (std 0.98658 x scale) with scale 1/D for the embedding and 1/sqrt(fan_in)
elsewhere; the LM head reuses the embedding's PRNG key, so it starts as sqrt(D) x the embedding. The optimizer has
three groups, each clipped by its own global norm: MuonH (Nesterov momentum, 5 "simple" Newton-Schulz steps in f32,
then a per-layer norm-preserving step of relative size eta) for the 7 transformer Linear weights, AdamH (Adam
direction, then a whole-matrix norm-preserving step at the main lr) for `lm_head`, and plain Adam at `adam_lr` for the
embedding and all norm gains; no weight decay is applied, although 0.1 is logged. The data is fineweb-edu-10B
(pre-tokenized, marin tokenizer, V = 128256) cut into contiguous 4096-token windows, with document blocking derived from
the EOS token, ordered by a closed form (Threefry block shuffle of 2048, then an LCG with a = 830293, b = 746237, data
seed 42) that is pinned by 12 batch hashes. Eval is the full 16-domain Paloma val set in 114 batches of 128, logged
at step s with the parameters after s + 1 updates, with levanter's token-weighted per-batch bpb estimator. The 130m
pool (n = 8, W&B) gives c4_en bpb 1.16326 (sd 0.00033) and macro_loss 4.18646 (sd 0.00577); the task text's 1.16327
(sd 0.00035) and 4.1848 (sd 0.0043, an n = 4 value) are superseded. marincpp passes stage 1 when it clears the "Parity test
plan" at the end: static checks, unit kernels against levanter fixtures, step-0 loss and gradients, the first 200 steps, and
final metrics inside the pool at no lower throughput.

## Hyperparameters per size

All values from the launcher `experiments/references/della_muonh_qwen3_scaling.py:426-431, 736-820` and the logged
W&B configs of `reself/marin-della/muonh-qwen3-<size>-della4xh100`; cited per row to the part that derives them.

| | 130m | 300m | 520m | 1_2b | source |
|---|---|---|---|---|---|
| hidden D | 512 | 768 | 1024 | 2048 | P1 §1 |
| layers L | 6 | 12 | 24 | 16 | P1 §1 |
| query heads H / kv heads KV | 8 / 8 | 12 / 12 | 16 / 8 | 16 / 8 | P1 §1 |
| head dim hd = D/H | 64 | 64 | 64 | 128 | P1 §1 |
| MLP width F (= 3.5 D) | 1792 | 2688 | 3584 | 7168 | P1 §1 |
| vocab V (untied head) | 128256 | 128256 | 128256 | 128256 | P1 §1, P3 §2.2 |
| seq len S | 4096 | 4096 | 4096 | 4096 | P1 §1 |
| RoPE theta / factor | 10000 / 1 | same | same | same | P1 §2.2 |
| RMSNorm eps (f32) | 1e-5 (0x3727C5AC) | same | same | same | P1 §2.1, P5 §3.2 |
| QK-norm, hybrid post-norms, biases, z-loss | on, on, none, 0 | same | same | same | P1 §1-2 |
| attention scale (bf16 constant) | 0.125 | 0.125 | 0.125 | 0.08837890625 (bf16 0x3DB5) | P1 §2.3, P5 §3.4 |
| parameters (total = W&B `parameter_count`) | 154,154,240 | 299,669,760 | 602,509,312 | 1,431,443,456 | P1 §5 |
| non-embedding parameters | 22,820,096 | 102,668,544 | 339,841,024 | 906,106,880 | P1 §5 |
| init: truncated normal (-3, 3), std 0.98658 x scale; seed | 0 | 0 | 0 | 0 | P1 §4.1-4.2 |
| init scale, embedding (1/D) | 1.953e-3 | 1.302e-3 | 9.766e-4 | 4.883e-4 | P1 §4.1 |
| init scale, lm_head and D-input Linears (1/sqrt(D)) | 0.04419 | 0.03608 | 0.03125 | 0.02210 | P1 §4.1 |
| init scale, down_proj (1/sqrt(F)) | 0.02362 | 0.01929 | 0.01670 | 0.01181 | P1 §4.1 |
| `learning_rate` eta (MuonH, AdamH) | 0.02 | 0.01 | 0.01 | 0.01 | P2 §1 |
| `adam_lr` (Adam group) | 0.008 | 0.002 | 0.002 | 0.0015 | P2 §1 |
| MuonH momentum mu (Nesterov) | 0.95 | 0.98 | 0.98 | 0.98 | P2 §1, §4.1 |
| Adam / AdamH eps (outside sqrt) | 1e-20 | 1e-15 | 1e-15 | 1e-15 | P2 §1, §5 |
| beta1 / beta2 | 0.9 / 0.98 | same | same | same | P2 §1 |
| Newton-Schulz: steps, (a, b, c), eps | 5, (3.4445, -4.7750, 2.0315), 1e-5 | same | same | same | P2 §4.3 |
| max_grad_norm (each of 3 groups separately) | 1.0 | 1.0 | 1.0 | 2.0 | P2 §3.1 |
| weight decay applied | none (0.1 logged, never read) | none | none | none | P2 §7 |
| lr schedule | constant, then linear to 0 | warmup, cosine to 0 | warmup, cosine to 0 | warmup, cosine to 0 | P2 §6 |
| warmup / stable / decay steps T_d | 0 / 992 / 3967 | 1000 / 0 / 10444 | 1000 / 0 / 8918 | 1000 / 0 / 21888 | P2 §6 |
| eta at the last update k = N-1 (main / adam) | 5.041361e-06 / 2.016544e-06 | 2.98023e-10 / 5.96046e-11 | 2.98023e-10 / 5.96046e-11 | 0 / 0 | P2 §6 |
| sequences per step B | 128 | 128 | 256 | 256 | P2 §8 |
| tokens per step | 524,288 | 524,288 | 1,048,576 | 1,048,576 | P2 §8 |
| updates N (`num_train_steps`) | 4959 | 11444 | 9918 | 22888 | P2 §1 |
| total tokens | 2,599,944,192 | 5,999,951,872 | 10,399,776,768 | 23,999,807,488 | P2 §8, P3 §4.5 |
| passes over the 2,441,406 sequences | 0.260 | 0.600 | 1.040 | 2.400 | P3 §4.5 |
| seqs per GPU per microbatch (PDP) x microbatches | 32 x 1 | 32 x 1 | 32 x 2 (updates 0-85: 64 x 1) | 32 x 2 | P2 §8, §C1 |
| loss denominator per microbatch (sum of weights) | 524,160 | 524,160 | 524,160 (updates 0-85: 1,048,320) | 524,160 | P5 §3.6 |
| data seed / mixture block / permutation | 42 / 2048 / linear (LCG) | same | same | same | P3 §4 |
| eval: logged steps (params after s+1 updates; final after N) | 1000-4000, 4958 | 1000-11000, 11443 | 1000-9000, 9917 | 1000-22000, 22887 | P4 §1 |
| eval set / batch | 16 Paloma domains, 14,529 seqs, 114 batches of 128 | same | same | same | P3 §8, P4 §2 |
| `flops_per_example` (= 3 x 4096 x fpt) | 2.800050241536e12 | 6.820139630592e12 | 1.6582868729856e13 | 3.5356170780672e13 | P1 §6 |
| reference step time / tok/s / MFU % (p50, 4x H100) | 0.6316 / 830,158 / 14.34 | 1.4437 / 363,155 / 15.28 | 6.7415 / 155,541 / 15.91 | 8.2209 / 127,550 / 27.82 | P6 §4(d) |
| reference pool n (finished plain H100 runs) | 8 | 5 | 1 | 1 | P6 §3 |
| final c4_en bpb, pool mean (sd) | 1.16326 (0.00033) | 1.05641 (0.00029) | 0.98830 | 0.92763 | P6 §3 |
| final `eval/paloma/macro_loss`, pool mean (sd) | 4.18646 (0.00577) | 3.81386 (0.00574) | 3.57255 | 3.36087 | P6 §3 |
| final `eval/macro_bpb`, pool mean (sd) | 1.51379 (0.00215) | 1.38401 (0.00219) | 1.29952 | 1.22578 | P6 §3 |
| c1 99% prediction interval, c4_en bpb | [1.16204, 1.16449] | [1.05495, 1.05787] | 0.98830 +- 0.0011 | 0.92763 +- 0.0011 | P6 §4(c) |
| c1 99% prediction interval, macro_loss | [4.16504, 4.20788] | [3.78491, 3.84281] | 3.57255 +- 0.021 | 3.36087 +- 0.021 | P6 §4(c) |

P1-P6 = Parts 1-6 below (model, optimizer, data, eval, numerics, targets).

## C. Cross-section consistency check

Every value that two or more parts state was compared. These agree: shapes, V, S, B, N, token totals, all optimizer
constants, the attention scale, the CE path (H100 B-tiled, 8192-row tiles, 16 per GPU), the loss denominator, the
group routing (P1 §4.3, P2 §2, P6 §4 conventions), the init and frozen-norm values (P1 §4.3, P2 §4.4), the step and
eval semantics (P3 §8.2, P4 §1, P6 §1), the 16-domain Paloma counts (P3 §8.1 = P4 §2.2), the final metrics (P4 §4.2
= P6 §2), the parameter counts (P1 §5, P4 §6) and the FLOPs (P1 §6, P4 §6, P6 §4(d)). These disagreed and are resolved:

| # | statement | parts that disagreed | resolution (evidence) |
|---|---|---|---|
| C1 | When the 520m run switched from one microbatch of 256 to two of 128 | P2 §8 "near step 85, estimate"; P3 §5 "updates 1..120, switch at update 121"; P5 §3.6, §6.3 "steps 1-119" | The first segment saved temporary checkpoint `step-85` (`project/marin/logs/slurm/muonh-qwen3-520m_13852139.out:129`). The second loaded it and logged "Resuming training from step 86" (`muonh-qwen3-520m_13852140.out:110-111, 207`; `levanter/main/train_lm.py:456-457` prints `state.step`) and refused to log steps 86..119 (`:215`). So updates 0..85 (batches 0..85) used PDP 64, updates 86..9917 used PDP 32, and the W&B rows 86..119 come from the discarded trajectory. All five parts now say this. |
| C2 | 130m logged final lr | P4 §6 "= 0.02 / 3967.2"; P2 §6.1 "f32(0.02) x (1 - f32(3966/3967))" | P2 is right: decay steps are `int(0.8 x 4959)` = 3967 (`levanter/optim/config.py:312-332`), and the linear schedule is `(init - end) x (1 - c/T_d) + end` in f32 (`optax/schedules/_schedule.py:142-145`). That gives 5.041361e-06, the logged value; 0.02 / 3967.2 = 5.04134e-06 does not. P4 fixed. |
| C3 | 130m pool statistics | task text and P4 §4.2 (ledger: c4_en 1.16327 sd 0.00035; macro 4.1848 sd 0.0043; "no n=8 macro value"); P6 §3 (W&B: 1.16326 sd 0.00033; 4.18646 sd 0.00577, n = 8) | W&B wins (P6 §6): 4.1848 is the n = 4 raw-seed mean; sd 0.00035 is an arithmetic slip in the ledger. P4 and P5 §9.2 (sd units) now use the W&B values. |
| C4 | `eval/paloma/micro_loss` vs `eval/loss` on H100 | P6 §6 "equal exactly"; P4 §4.2 "differ by a few float32 ulp" | Two formulas (`levanter/eval.py:609` vs `:637`). History gaps up to 1.9e-6 (130m step 2000, 300m step 1000); equal at the final step. P6 fixed. |
| C5 | Upstream recipe commit | P1 §0 `9f87569c3d`; P2 §0 `7cd7b7ea85`; P6 §1 `2d6290a99c` | Same file, three versions: added in `2d6290a99c` (2025-10-18, an ancestor of HEAD), deleted on the HEAD line in `bcaf54f338` (2026-04-08), re-added in `9f87569c3d` and fixed in `7cd7b7ea85` and `63ac661820` on `upstream/qwen3-speedrun-muonh` (not an ancestor of HEAD). No spec value comes from that file (P2 §0). P1 fixed. |
| C6 | Parameter-count formula | P2 §2.2 `L(4D^2 + ...)`; P1 §5 general form | `4D^2` holds only for MHA (130m, 300m). P2 now uses `L(2*D*H*hd + 2*D*KV*hd + 3DF + 4D + 2hd) + D + 2VD`, which matches all four logged counts. |
| C7 | Loss denominator "B_micro = 128 for every size" | P1 §2.5 vs P5 §3.6 | 520m updates 0..85 used 256 sequences per microbatch (C1). P1 fixed. |
| C8 | macro_bpb bias of the levanter estimator | P3 §8.3 (survey note: W&B reads about 0.004 high); P4 §4.3 (measured: logged is 0.0004-0.0006 low) | Open. P4's number is a direct computation from the logged per-domain losses and exact byte counts; the survey note gives no computation. P3 now cites both. No gate depends on it (gates compare levanter-estimator keys). |
| C9 | Which W&B history rows are duplicated | P4 §1 (300m step 10000; 1_2b 10000, 20000); P6 §1 (multiples of 5000) | Different fetches of the same API. Both parts trace them to API pagination; the local `.wandb` files hold each eval once. Values are identical. Rule for every script: dedupe by `_step`. |
| C10 | Eval tolerance | P4 §8 test 3 (2e-4 nats per domain, trained checkpoint) vs P6 a3 (1e-4, init model) | Different checks; both stand (Parity test plan, T3.2 and T2.3). |
| C11 | `trainer.py` line numbers | P1 cites run-commit lines (`:97-100`, `:244`); P2, P4 and P5 cite HEAD lines (`:98-101`, `:254`) | Both correct under their stated convention. The 17 added lines shift HEAD numbers by +1 after the import, +10 after HEAD `:116`, +17 after HEAD `:749`. |

---

## Part 1. Model (`model.md`: Stage-1 spec: model architecture and initialization)

Scope: the plain `muonh_qwen3` baseline (no `VARIANT`, no PLS knobs, `TIE=0`, `SEQ_LEN=4096`, `SEED=0`) as launched by
`experiments/references/della_muonh_qwen3_scaling.py` through `scripts/della/muonh_qwen3_h100x4.sbatch`.

Citations are `path:line` in `$ROOT/project/marin` at HEAD `1793624070`, except
`levanter/trainer.py`, which is cited at the run commits (§0). Paths under
`lib/levanter/src/levanter/` are shortened to `levanter/`, and paths under `lib/haliax/src/haliax/` to `haliax/`. JAX
is `.venv/lib/python3.12/site-packages/jax` (jax 0.11.1).

### 0. Provenance

| size | W&B run (reself/marin-della) | logged `git_commit` |
|---|---|---|
| 130m | `muonh-qwen3-130m-della4xh100` | `59ac0d647dda` |
| 300m | `muonh-qwen3-300m-della4xh100` | `28fd599610ac` |
| 520m | `muonh-qwen3-520m-della4xh100` | `ac6af86874a6` |
| 1_2b | `muonh-qwen3-1_2b-della4xh100` | `cead09a2590b` |

- Cached configs: `cache/wandb/muonh-qwen3-<size>-della4xh100.config.json`, written by `cache/wandb/_tools/fetch_config.py`
  (paths relative to `project/marin.cpp`).
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

### 1. Shapes per size

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

### 2. Forward pass: exact order of operations

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

#### 2.1 RMSNorm (all 7 kinds: in, q, k, post_attn, pre_mlp, post_mlp, final)

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

#### 2.2 Rotary embedding

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

#### 2.3 Attention

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

#### 2.4 MLP

`gate_proj`, `up_proj`: D->F. `down_proj`: F->D. Output = `down(silu(gate(m)) * up(m))` (`llama.py:268-294`). The
elementwise math is bf16 in the JAX program. XLA:GPU may compute fused elementwise ops in f32 internally; this is a
compiler detail and does not show in the source.

#### 2.5 Head, logits, loss knobs

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

#### 2.6 Parameter-gradient dtypes

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

### 3. Masking: causal plus cross-document

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

### 4. Initialization

#### 4.1 Distribution and scale of every parameter

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

#### 4.2 PRNG key tree (needed for bit-exact init)

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

#### 4.3 Init norms are frozen by the optimizer

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

### 5. Parameter count

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

### 6. FLOPs per token and MFU (as levanter logs them)

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
- For marincpp, keep this exact formula when reporting MFU, so numbers compare with levanter's. It omits norms,
  RoPE, the embedding gather and the causal saving.

### 7. Traps and ambiguities

1. `initializer_range=0.02` is logged and unused. Embedding std is ≈ 0.98658/D, and linear std is ≈ 0.98658/sqrt(fan_in).
2. `use_qk_norm: false` is logged, yet QK-norm is on (§2.1). The parameter count confirms it.
3. `lm_head` and the embedding share PRNG key `k_emb`, so the head starts as `sqrt(D)` × embedding (§4.2).
4. The flash-attention softmax statistics and accumulator are bf16 with 1024 blocks (§2.3). There are two readings
   for marincpp:
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


---

## Part 2. Optimizer (`optim.md`: Optimizer: MuonH / AdamH, groups and schedule)

Stage-1 spec section. It covers everything between "the f32 gradient of the mean loss exists" and "the f32 parameters of
the next step exist": gradient accumulation, clipping, the three optimizer groups, their update math, the learning-rate
schedules, batch size and step counts.

Every training step does this, in order: (1) average the microbatch gradients, (2) clip each group by its own norm,
(3) transform the gradients: MuonH for the transformer Linear weights, AdamH for `lm_head`, plain Adam for the rest,
(4) apply `p <- p + u`. No weight decay is applied anywhere.

### 0. Sources and notation

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
| `WB` | `$ROOT/project/marin.cpp/cache/wandb/api/reself__marin-della__muonh-qwen3-<size>-della4xh100.json` (logged config and summary of the finished runs `reself/marin-della/muonh-qwen3-<size>-della4xh100`) |

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

### 1. Values per size

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

### 2. Parameter groups (routing)

#### 2.1 Rule

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

#### 2.2 Inventory

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

### 3. Step pipeline and order of operations

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

#### 3.1 Clipping is per group, not global

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

### 4. MuonH (transformer Linear weights)

`scale_with_muonh(momentum, nesterov=True, steps=5, muon_eps=1e-5, learning_rate, coefficient_type="simple")`
(`LV/optim/muonh.py:136-188`). Its state is one f32 momentum buffer per weight, initialized to zero (`:142-144`). It
has no step counter and no bias correction.

#### 4.1 Momentum (Nesterov, without the (1−μ) factor)

With `g` the clipped gradient and `μ = momentum` (`:147-162`):

```
buf <- μ * buf + g            # stored state
u   <- μ * buf + g            # nesterov=True: uses the NEW buf
```

#### 4.2 Shaping into matrices

`map_flattened_linear_layers` (`LV/optim/util.py:158-209`) does the following:

1. It flattens each Linear weight to 2-D: all Out axes merge into `__OUT__` and all In axes into `__IN__`, row-major in
   declaration order. The array is then rearranged to `(..., __OUT__, __IN__)`, because `out_first = True` for every
   Linear here (`util.py:96-107`).
2. It vmaps over the stacked layer axis (`HX/tree_util.py:63-66`), so the Newton-Schulz iteration runs **once per
   layer per matrix**.

For `q_proj`, row index = `(kv·(H/KV) + g)·hd + d` and column = embed. For `o_proj`, column index = `h·hd + d`. Newton-Schulz
is equivariant under row and column permutations, so another layout gives the same math. Only the summation order, and
so the last bits, can differ.

#### 4.3 Newton-Schulz ("zeropower", `util.py:244-298`)

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

#### 4.4 Norm-preserving projection (the "H")

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

### 5. AdamH (`lm_head`) and Adam (embedding, norms)

#### 5.1 AdamH

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

#### 5.2 Adam

`optax.scale_by_adam(b1=0.9, b2=0.98, eps)` then `optax.scale(-adam_lr)` (`muonh.py:91-92`;
`OX/_src/transform.py:246-308,455-467`; `eps_root = 0`, `nesterov = False`). It uses the same `mu`, `nu`, bias
correction and `a = mu_h / (sqrt(nu_h + 0) + eps)` as AdamH, then `u = (-adam_lr) * a`. It has no projection and no
weight decay. Embedding rows absent from the batch still move through their momentum.

#### 5.3 Scalar constants and the bias-correction power

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

### 6. Learning-rate schedules

`OptimizerConfig.lr_scheduler(N, override_lr)` (`LV/optim/config.py:286-362`) runs twice. It builds `sched_main` with
peak `learning_rate` (used by MuonH and AdamH) and `sched_adam` with peak `adam_lr` (used by Adam) (`muonh.py:58-59`).
The two schedules have the same shape. `min_lr = peak * min_lr_ratio = 0`. No cooldown, one cycle (`config.py:383-422`).
Warmup and decay lengths convert by `_convert_frac_or_steps`: if `x ≤ 1`, the length is `int(x·N)`, else `int(x)`
(`config.py:441-448`). `T_d` is the number of decay steps (`D` in §2.2 is the hidden size, a different thing).

#### 6.1 130m: constant, then linear decay to 0 (WSD-like, no warmup)

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

#### 6.2 300m, 520m, 1_2b: linear warmup, then cosine to 0

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
histories `$ROOT/project/marin.cpp/cache/wandb/run-2026091[4-7]_*-muonh-qwen3-<size>-della4xh100.jsonl.gz`
(one file per resumed segment; a step re-run after a resume appears once). The table has holes: of k = 2..N−1, 130m
has all 4957 steps, 300m 11435 of 11442, 520m 9900 of 9916, 1_2b 22822 of 22886 (the missing steps are presumably the
first steps of resumed segments). Take the holes and k = 0, 1 from the formulas, and check each against its logged
neighbours. η(0) = 0 for the cosine sizes and peak for 130m; η(1) = `f32(-peak)·(1 − f32(1/1000)) + f32(peak)` for
the cosine sizes.

All schedule arithmetic is f32 with an int32 counter. The peaks are f32 roundings of the decimal values (0.02 →
0.019999999552965164, as logged).

### 7. What is not in the optimizer

- **Weight decay.** `weight_decay = 0.1` is logged, but `MuonHConfig.build` never reads it (`muonh.py:54-104`; compare
  `AdamConfig`, `config.py:515-521`, which does). The fixed-norm projection plays the role of decay for the matrices.
  The docstring says MuonH's lr ≈ sqrt(lr·wd) of a decayed Muon (`muonh.py:39`). The embedding and the norm weights get
  no decay.
- There is no update-RMS clipping, no skip-bad-steps, no z-loss (`L:819`), no EMA (`EMA_BETA = 0`, `L:813`), no
  per-parameter LR multipliers, and no μP scaling (`LinearStandardParam.lr_scale = 1`, `HX/nn/mup.py:71-72`).
- There is no shape-based update scaling (such as `sqrt(max(1, rows/cols))`) in MuonH. The projection replaces it:
  every matrix moves by exactly η of its own norm.

### 8. Batch, steps, accumulation

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
  equivalence. marincpp can use any microbatch count, at the cost of last-bit differences.
- Data order (`data_seed = 42`, linear permutation) and init `seed = 0` belong to the data and model sections.

### 9. Reference pseudo-code (one step, f32)

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

### 10. Ambiguities and open items

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


---

## Part 3. Data (`data.md`: Data pipeline and exact data order)

Stage-1 spec, section "data". Target: a new loader that produces, for every step k, the same token batch as the
levanter baseline runs `muonh-qwen3-{130m,300m,520m,1_2b}-della4xh100` (W&B `reself/marin-della`), and the same
Paloma eval set and eval arithmetic.

**Status of every claim.** Facts marked **[verified]** were checked on 2026-10-24 by
`cache/data/_tools/data_order_ref.py` (run on della-vis1, CPU, marin `.venv`, log `cache/data/_tools/run.log`). That
script builds the training set through levanter's own `LmDataConfig.train_set` with the logged config, rebuilds the same
order with an independent closed form (numpy Threefry-2x32, LCG, raw zarr3 reads through tensorstore), and asserts that
the two agree. Its outputs are in `cache/data/data_ref.json`. All other facts cite code.

**Abbreviations.** `M/` = `$ROOT/project/marin` (read-only reference). `L/` =
`M/lib/levanter/src/levanter`. `J/` = `M/.venv/lib/python3.12/site-packages/jax/_src`. `S/` =
`$ROOT/marin_store_big`. `N/` = `$ROOT/project/marin.cpp`.
Line numbers are for marin HEAD `1793624070`.

**Code version.** Each run is a chain of resume segments, and each segment logs its own `git_commit`; the run-level
W&B config keeps the last segment's values. The cached segment configs (`N/cache/wandb/config/*.yaml`) show 59ac0d647d
(130m, its only segment), 28fd599610 (300m, segment `run-20260914_123815`), cead09a259 (1_2b, last segment) and, for
520m, **two** commits: 59ac0d647d in the first segment (`run-20260914_063908`, steps 0..119) and ac6af86874 in the last
(`run-20260915_170105`). The first 520m segment ran with `per_device_parallelism` 64 (section 5). `git diff <commit> HEAD` over `L/data`, `L/store`,
`L/main/train_lm.py`, `L/eval.py`, `L/models/lm_model.py`, `L/tokenizers.py`, `M/lib/marin/src/marin/experiment/data.py`,
`M/experiments/datasets` and `M/uv.lock` is empty for all four. Only `L/trainer.py` differs (a contextvar for the
traced step, no data effect), plus the launcher (new env knobs whose defaults keep the baseline). So the HEAD code below
is the code that ran. Library versions: jax/jaxlib 0.11.1, numpy 2.3.5, tensorstore 0.1.84, tokenizers 0.22.2,
transformers 5.12.1.

---

### 1. Summary in one block

```
tokens  = int64 zarr3 array S/fineweb-edu-10B/2026.06.28/train/input_ids/data, length 10,000,000,738
N       = 10,000,000,738 // 4096 = 2,441,406 sequences; sequence s = tokens[4096*s : 4096*s + 4096]
mixture index j (global example counter) -> sequence id:
    blk  = j // 2048 ;  p = pi_blk[j % 2048]          # pi_blk: Threefry-keyed permutation of 0..2047
    r    = (blk*2048 + p) mod N                        # "restart": wrap, no reshuffle
    s(j) = (830293 * r + 746237) mod N                 # LCG permutation, data_seed 42
step k uses j = k*B .. k*B + B - 1 (B = 128 at 130m/300m, 256 at 520m/1_2b), rows in that order
```

The precomputed table `N/cache/data/fineweb_order_seed42.u32` holds s(j) for j = 0 .. 5,859,327 (all four sizes),
little-endian uint32, sha256 `dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b` **[verified]**.

---

### 2. Training corpus

#### 2.1 Location and provenance

| item | value | source |
|---|---|---|
| dataset handle | `fineweb_edu_10B_dataset()` = `pretokenized("fineweb-edu-10B", repo_id="marin-community/fineweb-edu-pretokenized-10B", tokenizer=marin_tokenizer, version="2026.06.28")` | `M/experiments/datasets/prebuilt_caches.py:30-37` |
| launcher use | `train = {fineweb_edu_10B_dataset(): 1.0}` | `M/experiments/references/della_muonh_qwen3_scaling.py:544` |
| cache dir (logged) | `S/fineweb-edu-10B/2026.06.28` | config yaml `data.components.fineweb-edu-10B.cache_dir` |
| split dir | `S/fineweb-edu-10B/2026.06.28/train` | `L/data/text/datasets.py:569-574` (`cache_dir/<split>`) |
| upstream | HF dataset repo `marin-community/fineweb-edu-pretokenized-10B`, revision `null` (main); its README: subsample of fineweb-edu (`HuggingFaceFW/fineweb-edu@3c452cb`), "Seed 42" (the upstream subsampling seed, unrelated to `data_seed`), 10,000,000,738 tokens | `S/fineweb-edu-10B/2026.06.28/{.artifact.json,README.md}` |
| rows / tokens | 9,966,814 documents, 10,000,000,738 tokens **[verified]** | ledger `train/shard_ledger.json`, offsets array |

**Caveat (restore).** The files on disk were re-downloaded on 2026-09-28 (`.provenance.json` access_time), after the
09-24 store cleanup deleted the copy the 09-14 baseline runs read (`M/docs/della/sampled-softmax.md:94-101`). The path
is the same, the document and token counts equal the counts recorded before the cleanup, and the 09-28 restore
reproduced the 130m pool (restore baseline macro 4.1820, inside the pool; `sampled-softmax.md:122-123`). A token-level
identity with the 09-14 copy cannot be proven now; the batch hashes in section 7 pin the current copy.

#### 2.2 Tokenizer

* Name `marin-community/marin-tokenizer` (`M/experiments/marin_tokenizer.py:23`): the Llama 3 tokenizer with a custom
  chat template; ids 128002/128003 renamed `<|start_think|>`/`<|end_think|>` (`marin_tokenizer.py:29-32`). Chat
  template does not affect pretraining data.
* Loaded by levanter as `load_marin_tokenizer(self.tokenizer)` (`L/data/text/datasets.py:705-710`,
  `L/tokenizers.py:733-747`); offline from the HF hub cache snapshot
  `$ROOT/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`.
* `len(tokenizer)` = 128256, BOS = `<|begin_of_text|>` = 128000, EOS = `<|end_of_text|>` = 128001 **[verified]**
  (`tokenizer_config.json`: `bos_token`, `eos_token`). The model vocabulary axis is 128256.
* The training data is pre-tokenized; the new loader never runs the tokenizer for training. The tokenizer is needed
  only for the bytes-per-token table of the bpb metric (section 8.4).

#### 2.3 On-disk format

Levanter `TreeCache`, **consolidated** layout (the ledger has no `layout` key; default `consolidated`,
`L/store/cache.py:57,663`), one field `input_ids` stored as a `JaggedArrayStore` (`L/store/jagged_array.py:203-249`):

```
train/shard_ledger.json        {"total_num_rows": 9966814, "is_finished": true, ...}
train/input_ids/data/          zarr v3 array, int64, the concatenated tokens of all documents
train/input_ids/offsets/       zarr v3 array, int64
```

* `offsets[0]` = number of rows (9,966,814); `offsets[i+1]` = exclusive end of row i in `data`; row i spans
  `data[offsets[i] : offsets[i+1]]` with `offsets[0]` read as 0 for i = 0 (`jagged_array.py:217,276-305,542-554`).
  `offsets[1]` = 588 (first document has 588 tokens); `offsets[9966814]` = 10,000,000,738 = data size **[verified]**.
* zarr metadata (both arrays, `zarr.json`): `data_type` int64, logical shape 2^54 (resizable; the real length comes
  from the offsets), `chunk_grid` regular with `chunk_shape` [134217728], codec `sharding_indexed` with inner
  `chunk_shape` [262144], inner codecs `bytes` (little-endian) then `blosc` (`cname` lz4, `clevel` 5, `shuffle`
  byte-shuffle, `typesize` 8, `blocksize` 0), `index_codecs` `bytes` (little-endian) + `crc32c`, index location not
  given (zarr v3 default: end of shard). `fill_value` 0.
* Files: `data/c/0` … `data/c/74` (75 shard files, 134,217,728 elements each, the last one partial),
  `offsets/c/0` (one file). Each shard file holds 512 inner chunks and ends with the shard index: 512 entries of
  (offset uint64 LE, nbytes uint64 LE) followed by a 4-byte crc32c; an entry of (2^64-1, 2^64-1) is an empty chunk
  (fill value 0).
* Paloma caches use the **sharded** layout (section 8.1) and blosc `zstd` level 1 instead of lz4; the element layout
  is the same.

Reading from C++: tensorstore's C++ API opens these arrays directly with the spec
`{"driver": "zarr3", "kvstore": {"driver": "file", "path": ".../input_ids/data"}}` (the reference script reads them
this way, independent of levanter). Recommendation, not done: convert once to a flat uint32 file (40.0 GB) with a
sha256, because every id is < 2^17, and mmap it.

#### 2.4 Document format

Each document is stored as `[BOS, text tokens …, EOS]`, with no padding between documents. On two 20M-token windows
(offset 0 and offset 5,000,000,369; 19,649 and 20,302 whole documents) **[verified]**:

* every document starts with 128000 and ends with 128001;
* 128000 occurs once elsewhere per window and 128001 occurs once (second window) not at a document end, i.e. these
  ids appear inside document text, rarely (about 1 in 20,000 documents);
* min id 0, max id 128001; shortest document 58 tokens; mean 1003.3 tokens over the whole corpus
  (10,000,000,738 / 9,966,814).

The cache ledger carries no preprocessor metadata (levanter logs "Metadata mismatch … preprocessor_metadata"; benign),
so `enforce_eos=True` in the config (`L/data/text/datasets.py:636`) does not alter the stored tokens; BOS/EOS come
from the upstream tokenization.

---

### 3. Documents to sequences (packing)

Component format `TextLmDatasetFormat(text_key="text")`, `pack: null` → `_effective_pack` returns False
(`L/data/text/datasets.py:362-370`) → `CausalLmDataset(TokenSeqDataset(cache, 4096))`
(`datasets.py:529-534`). `TokenSeqDataset` (`datasets.py:78-114`):

* length `N = data_size // seq_len` (`datasets.py:91-92`) = 10,000,000,738 // 4096 = **2,441,406**; the last 1,762
  tokens are never used **[verified]**;
* sequence s = `data[4096*s : 4096*s + 4096]` (`datasets.py:105-106`, `L/store/cache.py:573-580`).

So training is a plain concatenation of all documents in cache order, cut into fixed windows. No BOS is inserted at a
window start, no EOS is added, a document freely crosses window boundaries, and nothing is padded. Example: step 0,
row 0 starts `[4, 8834, 23246, 605, 4, 50585, 595, 1912]` (mid-document) **[verified]**.
`train_seq_len` 4096 (`della_muonh_qwen3_scaling.py:816`); `Pos = model.max_Pos.resize(4096)` (`L/main/train_lm.py:245`).

#### 3.1 The training example built from one sequence

`GrugLmExample.causal(tokens, eos_id=128001, block_cross_document_attention=True)` (`datasets.py:157-168`,
`L/data/text/examples.py:123-175`):

| field | value | code |
|---|---|---|
| tokens | the 4096 ids, int32 inside jit (stored int64; JAX x64 is off) | **[verified]** dtype int32 |
| loss_weight | float32, 1 at positions 0..4094, 0 at position 4095 (no next token); sum 4095 per row | `examples.py:142-149,200-209` **[verified]** |
| segment ids | `seg[0] = 0`, `seg[i] = #{t < i : tokens[t] == 128001}` for i ≥ 1, i.e. `cumsum(roll(tokens,1)==EOS)` with position 0 forced to 0; int32; used as (q, kv) segment ids | `examples.py:166-171` **[verified]** for step 0 row 0 (EOS at 1380, 2370, 2578, 3243, 3888 → 6 segments) |
| attention | causal and same-segment only | `examples.py:165-173`; config `block_cross_document_attention: true` |
| positions | `arange(4096)` per row, **not** reset at document boundaries | `compute_next_token_loss` calls `self.activations(example.tokens, example.attn_mask, key=key)` with no `pos_ids` (`L/models/lm_model.py:337`); Qwen3 uses the standard `Attention` (`L/models/qwen.py:146,155`), whose default is `pos_ids = hax.arange(x.resolve_axis("position"))` (`L/layers/attention.py:1441-1442`) |

Consequences a reimplementation must copy:

1. Segments are derived from the **EOS token value**, not from the document offsets. An EOS inside text (section 2.4)
   starts a new segment; a BOS inside text does not.
2. The EOS token belongs to the segment it ends; the next token (BOS of the next document) opens the new segment.
3. The loss at the EOS position (predicting the next document's BOS) is **kept** (weight 1). Only position 4095 has
   weight 0.
4. RoPE positions run 0..4095 across documents inside a window, even though attention is blocked between them.

---

### 4. Exact data order

#### 4.1 Keys

All randomness of the data order comes from `data_seed = 42` (`della_muonh_qwen3_scaling.py:820`). The trainer seed
(`seed: 0`, `:812`) does not touch the data order.

```
data_key            = PRNGKey(42)                       # L/main/train_lm.py:218-222  -> raw (0, 42)
mix_key, shuffle_key = split(data_key)                   # L/data/text/datasets.py:788
perm_key            = split(shuffle_key)[1]              # key_iterator, L/utils/jax_utils.py:316-321, used at datasets.py:839-841
```

Values (uint32 pairs) **[verified]**: `mix_key = (1832780943, 270669613)`, `shuffle_key = (64467757, 2916123636)`,
`perm_key = (2350016172, 1168365246)`.

Threefry, as JAX 0.11.1 runs it with `jax_threefry_partitionable = True` (logged `trainer.jax_config`; also the JAX
default, `J/config.py:1411-1413`; `L/trainer.py:98-99`):

* `T(k, (x0, x1))` = Threefry-2x32 with 20 rounds: `ks = (k0, k1, k0 ^ k1 ^ 0x1BD11BDA)`; `x0 += ks0; x1 += ks1`;
  for i = 0..4: four rounds with rotations `[13,15,26,6]` (i even) or `[17,29,16,24]` (i odd), each round
  `x0 += x1; x1 = rotl32(x1, r); x1 ^= x0`; then `x0 += ks[(i+1)%3]; x1 += ks[(i+2)%3] + (i+1)`. All uint32 wrapping
  (`J/random/threefry2x32.py:109-114,129-177`).
* `split(k)[i] = T(k, (0, i))` for i = 0, 1 (`threefry2x32.py:305-310`, `J/random/prng.py:798`).
* `fold_in(k, d) = T(k, (0, d))` (`threefry2x32.py:312-318`, seed `(d >> 32, d & 0xFFFFFFFF)` = `(0, d)`).
* `bits32(k, n)[i] = T(k, (0, i)).x0 ^ T(k, (0, i)).x1` for i = 0..n-1 (`threefry2x32.py:333-350`).

The reference script implements exactly these four lines in numpy and matches JAX on every key and permutation.

#### 4.2 Stage A: LCG permutation of the 2,441,406 sequences

`shuffle: true`, `permutation_type: linear` (logged; set by `dataclasses.replace(mixture(..., shuffle=True),
permutation_type="linear")`, `della_muonh_qwen3_scaling.py:761`, overriding marin's Feistel default at
`M/lib/marin/src/marin/experiment/data.py:321-328`). `train_sets` shuffles the one nonzero-weight component
(`datasets.py:824-841`) with `PermutationDataset(ds, perm_key, "linear")` (`L/data/dataset.py:102-103,375-411`),
whose permutation is `LcgPermutation(N, perm_key)` (`L/data/_prp.py:61-78,88-146`):

```
seed = randint(perm_key, (), 0, 2**30)        # _prp.py:81-85
     = bits32(split(perm_key)[1], 1)[0] mod 2**30 = 937906096
       (randint: k1, k2 = split(key); multiplier = (2^16 mod 2^30)^2 mod 2^30 = 0, so only k2's bits count;
        J/random/core.py:709-740)
rng  = numpy.random.Generator(PCG64(937906096))
a    = rng.integers(1, N) repeated until gcd(a, N) == 1      -> 830293
b    = rng.integers(0, N)                                     -> 746237
sigma(r) = (a*r + b) mod N                                     # _prp.py:141
```

`N = 2,441,406`, `a = 830293`, `b = 746237` **[verified]**. Copy the two constants; do not reimplement numpy's bounded
integer sampler. Note: in levanter `a` is `np.int64` and the index array `uint64`, so numpy promotes `a*r + b` to
float64 (`_prp.py:141`); the product stays below 2^53 (max ≈ 2.0e12), so the result is exact and equal to integer
arithmetic.

#### 4.3 Stage B: the mixture's within-block shuffle

The data then goes through `MixtureDataset(datasets={"fineweb-edu-10B": …}, weights={"fineweb-edu-10B": 1.0},
block_size=2048, key=mix_key, stop_strategy="restart")` (`datasets.py:781-801`; `mixture_block_size: 2048`,
`stop_strategy: restart` logged). The 16 Paloma components have weight 0 and are dropped
(`datasets.py:729`, `L/data/mixture.py:84-88`). **Even with one dataset, the mixture permutes indices inside every
block of 2048** (`randomize_blocks=True`, `mixture.py:58,210-216`). This step is easy to miss; the launcher docstring
mentions only the linear permutation.

With one dataset of weight 1: counts per block = [2048] (`mixture.py:143-160`), base ids = `0..2047`
(`mixture.py:162-169`; dataset id 0 in the high 16 bits), and for mixture index j (`mixture.py:231-269,218-229`):

```
blk = j // 2048
pi_blk = jax.random.permutation(fold_in(mix_key, blk), arange(2048))     # mixture.py:497-500
r = blk*2048 + pi_blk[j % 2048]
r = r mod N                                                               # restart, mixture.py:283-297
```

`jax.random.permutation` on a 1-D array is `_shuffle` (`J/random/core.py:744-820`): `rounds = ceil(3*ln(2048) /
ln(2^32-1)) = 2`; each round `key, sub = split(key)` (key ← `split(key)[0]`, sub ← `split(key)[1]`), draw
`bits32(sub, 2048)`, and **stable**-sort the current array by those keys ascending (`lax.sort_key_val`, `is_stable`
defaults True, `J/lax/lax.py:3555-3556`). Key i belongs to the element currently at position i. Block 0 starts
`[416, 814, 925, 86, 1936, 560, 614, 1951, …]` **[verified]**.

#### 4.4 Closed form

```
s(j) = ( 830293 * ((blk*2048 + pi_blk[j mod 2048]) mod 2441406) + 746237 ) mod 2441406,   blk = j div 2048
```

and the token window of mixture index j is `tokens[4096*s(j) : 4096*s(j)+4096]`. First values:
`s(0..7) = 1909879, 335277, 2165778, 1350661, 1748337, 1843177, 292285, 1995702` **[verified]**.

#### 4.5 Epochs and restart

* There is no reshuffle per epoch (`dataset.py:378`: "TODO: add epoch reshuffling"). Wrapping happens on r, after the
  block shuffle and before the LCG: pass 2 reads the same LCG order, but shifted by N mod 2048 = 190 positions relative
  to the block grid and with fresh block keys (blocks ≥ 1192), so its order inside each block differs from pass 1.
* Block 1192 (j = 2,441,216 … 2,443,263) is the first block with wrapped entries (those with `pi < 190` stay in pass 1,
  `pi ≥ 190` wrap) **[verified first wrapped j = 2,441,216]**. Pass 3 starts in block 2384 (j = 4,882,432).
* `max_train_batches`, `num_validation_sequences`, `experiment_budget`, `target_budget`: all null (logged), so no
  slicing or train/val split happens.

| size | B | steps | sequences consumed | passes over N | tokens | first step with repeated data |
|---|---|---|---|---|---|---|
| 130m | 128 | 4959 | 634,752 | 0.260 | 2,599,944,192 | none |
| 300m | 128 | 11444 | 1,464,832 | 0.600 | 5,999,951,872 | none |
| 520m | 256 | 9918 | 2,539,008 | 1.040 | 10,399,776,768 | 9536 (mixed batch), 9537–9917 more |
| 1_2b | 256 | 22888 | 5,859,328 | 2.400 | 23,999,807,488 | 9536 (pass 2), 19072 (pass 3) |

(B, steps: `della_muonh_qwen3_scaling.py:426-431`, logged `trainer.train_batch_size`, `num_train_steps`.)

Because s(j) depends only on j, 130m and 300m read the same stream (300m continues past 130m's end), and the B = 256
batch k is the concatenation of B = 128 batches 2k and 2k+1.

---

### 5. Global batch assembly

* Batch schedule constant: `global_data_offset_by_step(k) = k*B`, `batch_size_at_step(k) = B`
  (`L/schedule.py:103-115`). The loader asks for mixture indices `k*B … k*B+B-1` and keeps that order as batch rows
  (`L/data/loader.py:271-315,421-472`). `iter_from_step(state.step)` starts at batch index = number of completed steps
  (`L/main/train_lm.py:455-460`), so the update from step k to k+1 uses batch k.
* Mesh: one process, 4 GPUs, axes `data: -1 (=4), replica: 1, model: 1` (`della_muonh_qwen3_scaling.py:802-811`);
  the batch axis maps to `data`. Device d holds rows `[d*B/4, (d+1)*B/4)` (`loader.py:160-175,204-216`). B is a
  multiple of 4, so no padding rows (`allow_nondivisible_batch_size=True` is inert, `:814`).
* Microbatching: `microbatch_size = per_device_parallelism * 4` (`L/trainer.py:987-990`). Logged
  `per_device_parallelism`:
  * 130m, 300m: 32, from the default `B // data_axis = 128 // 4` (`L/trainer.py:1139-1141`). One microbatch of 128.
  * 1_2b: 32 in every segment (`PER_DEVICE_PARALLELISM = {"1_2b": 32}` already at 59ac0d647d, launcher line 79).
    Two microbatches of 128.
  * 520m: **64 for steps 0..119, then 32.** The first segment (`run-20260914_063908`, commit 59ac0d647d, config lines
    497-498: `per_device_parallelism: 64`, `per_device_eval_parallelism: 64`) ran at a commit whose launcher has
    `PER_DEVICE_PARALLELISM = {"1_2b": 32}` (`git show 59ac0d647d:experiments/references/della_muonh_qwen3_scaling.py:79`),
    so 520m took the default `256 // 4 = 64`: `microbatch_size = 256`, one microbatch, no gradient accumulation. That
    segment logged `global_step` 2..119; the launcher comment (`della_muonh_qwen3_scaling.py:432-434`) records the OOM
    at step ~115. Commit 28fd599610 ("Run 520m with two microbatches per step on 4x80GB", 2026-09-14 06:54) added
    `"520m": 32`. The next segment (`run-20260914_083457`, job 13852140, started 08:34) loaded the temporary checkpoint
    `step-85` and resumed from `state.step` 86 (`M/logs/slurm/muonh-qwen3-520m_13852140.out:110-111, 207`); its
    replayed steps 86..119 were not logged (W&B refuses steps below 120, `:215`), so its first logged `global_step` is
    120. So updates 0..85 (batches 0..85) used 64 and all later updates used 32; the W&B rows 86..119 come from the
    discarded first-segment trajectory. The 083457 config is not cached; 32 is read from the later segment config `run-20260915_170105` (lines 499-500) and
    from the commit order. The run-level W&B config shows 32 only because later segments overwrote it. No eval ran in
    the 64 window (the first segment has no `eval/` rows), so every logged 520m eval used EvalBatch 128.
  At two microbatches: `unflatten_axis(Batch, (AccumStep=2, Microbatch=128))` (`L/grad_accum.py:170-176`), so
  microbatch 0 = rows 0..127 and microbatch 1 = rows 128..255; loss and gradients are the mean of the two microbatch
  values (`grad_accum.py:153-162`). The token stream of a step does not depend on device count or microbatching; only
  the floating-point reduction order does. The training-loop section must carry the 520m switch at update 86
  (0-based).
* The original TPU runs (64 devices) and these 4-GPU runs map step k to the same mixture indices; the order is
  device-count independent.

---

### 6. Resume

* The data path holds no state. The loader is rebuilt from the config on every start, the LCG constants and block
  keys are recomputed from `data_seed`, and training resumes with `iter_from_step(state.step)`
  (`L/main/train_lm.py:455-460`). The checkpoint's step is the only data position; batch k after a resume is
  bit-identical to batch k in an uninterrupted run.
* Temporary checkpoints every 10 min, keep the last one; permanent every 10,000 steps (logged checkpointer). The 1_2b
  run went through 20+ resume segments (`N/cache/wandb/_tools/baseline_sacct.txt`) with this guarantee.
* A new implementation reproduces this with `next_j = step * B` stored (or recomputed) in its checkpoint, and no
  shuffle buffer or iterator state.

---

### 7. Check: batch hashes

**Definition.** `H(B, k) = sha256( tokens of mixture indices k*B .. k*B+B-1 as int32 little-endian, row-major
[B, 4096] )`. Both levanter (`mix.get_batch`) and the closed form give these values **[verified]**:

| B | step k | sha256 | s(kB) .. s(kB+3) |
|---|---|---|---|
| 128 | 0 | `4626638f4644adcaf66a9084c34ea3d4d423262a6473d03f6b95123f6bc9e449` | 1909879 335277 2165778 1350661 |
| 128 | 1 | `691e9e1ad524af5b1e3a772db5aceadc58bcf9c6e93506eaef508f37e7aaf50c` | 1339913 1849497 1944176 1892489 |
| 128 | 2 | `ffd0b0e8f29bece94a5888c79fba44a12e7bb6dbff34a6a7c4e45e227e2dc64f` | 1055876 1152769 1997916 2146496 |
| 128 | 1000 | `d1d0427ec3b89db162f09b57dd8ecea5c7053117c50c115fc1aea2a9c48104bf` | 1579833 633365 2410126 1061554 |
| 128 | 4958 (130m last) | `f8ba38d570bd33a16887fa5927f6ffe0550d9589b3ed44b9240228da2437bab1` | 1568248 699391 893016 1658660 |
| 128 | 11443 (300m last) | `6bc7e8d141df40d3f5c210d123e9bd81c2f3b45291fec053874c00d058378e2c` | 1329954 370524 1757982 2239911 |
| 256 | 0 | `813ff307c4b074a6ec62e51efa1cc95085a5b7c45dd31841d0fbd46895f1624f` | 1909879 335277 2165778 1350661 |
| 256 | 1 | `0778706834532ed1b912f18c990a7e72c323b26fe7734dc32c37050eb5282c0e` | 1055876 1152769 1997916 2146496 |
| 256 | 4767 | `154a347fbd2ab2800744e25b3ca92d383cb5bb34c30831bb9873484d6e2f8025` | 106570 2399396 2369366 775482 |
| 256 | 9536 (first wrap) | `3fc488330822e8a49e96ba54aa722f50f467a7fd3d4ac622661585c0751188ec` | 1316203 1236539 1292493 1128898 |
| 256 | 9917 (520m last) | `38211b426b26c5c7c1ce1092fe24eb329cc96b53d2a698180dcf969ea57730c4` | 736658 885238 618430 1068115 |
| 256 | 22887 (1_2b last) | `53942bac0515978ee19cdb0c606636d7e722900d4fadbfb74409cf773309a0e7` | 218635 2363203 1661886 1018737 |

Proposed acceptance test for the new loader: (1) the order table sha256 of section 1; (2) all twelve hashes above;
(3) for the same batches, segment ids and loss weights equal the formulas of section 3.1. A loader that passes these
produces the levanter token stream for every step of all four sizes. For a run-time guard, log `H(B, k)` at k = 0 and
at every eval step.

---

### 8. Paloma eval data

#### 8.1 Caches and domains

`validation = list(paloma_datasets(tokenizer=marin_tokenizer).values())` (`della_muonh_qwen3_scaling.py:545`): 16
handles `tokenized("paloma/<subset>-marin-tokenizer", tokenizer="marin-community/marin-tokenizer",
version="2026.06.28", paths=["raw/paloma-fc6827/65cd6fc/<dir>/val/val*.jsonl.gz"], validation=True)`
(`M/experiments/datasets/paloma.py:17-57`), i.e. the Paloma **val** split of `allenai/paloma@65cd6fc`, tokenized with
the training tokenizer. Caches at `S/paloma/<subset>-marin-tokenizer/2026.06.28/validation/part-00000-of-00001/input_ids/{data,offsets}`
(sharded layout, one shard; int64; same element layout as section 2.3). Weight 0, so eval only.

Restored on 2026-09-28 like the training cache. All 16 are byte-identical (tokens and offsets, sha256) to the
Llama-3.1-tokenized caches that survived in `S/tokenized/paloma/` (`M/docs/della/sampled-softmax.md:102-105`).

Every document is `[BOS, …, EOS]` in all 16 domains (first/last token counts equal the document counts) **[verified]**.

| # | domain (tag `paloma/<d>-marin-tokenizer`) | docs | tokens | seqs = tokens//4096 | dropped tail | scored positions = seqs·4095 | target bytes | eval index range | eval batches | sha256 of the seqs·4096 tokens (int32 LE) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4chan | 523 | 929,479 | 226 | 3,783 | 925,470 | 3,108,620 | [0, 226) | 0–1 | `418421ca41b128b5a6e15b6d0f620d9a4e4215c0b26a6712acf04b12802caee0` |
| 2 | c4_100_domains | 14,059 | 9,464,109 | 2,310 | 2,349 | 9,459,450 | 43,843,015 | [226, 2536) | 1–19 | `167b9ea6e7dbb4b47d7691ba9931c76a3af1641d5a5d0130d2008326ef9767f4` |
| 3 | c4_en | 2,091 | 967,618 | 236 | 962 | 966,420 | 4,530,682 | [2536, 2772) | 19–21 | `5ef2fc5286e6a1fac552f34e9e48b3b664ddec8e3550dacba889de051d433b69` |
| 4 | dolma-v1_5 | 4,600 | 2,883,425 | 703 | 3,937 | 2,878,785 | 12,406,417 | [2772, 3475) | 21–27 | `ef6c458e93f48c42e725ee9d3f3807ac77aa3c52b2de1c6979179ae8e0e04080` |
| 5 | dolma_100_programing_languages | 8,123 | 8,502,810 | 2,075 | 3,610 | 8,497,125 | 28,826,435 | [3475, 5550) | 27–43 | `8bd793b5b43e0a1319ef70ea2a0567ee73cae27053b9fd6e5e4508f644272a8f` |
| 6 | dolma_100_subreddits | 46,347 | 9,435,238 | 2,303 | 2,150 | 9,430,785 | 41,467,459 | [5550, 7853) | 43–61 | `f0548824ea12008b094e09c55aead5a826843982561cd7d179dbe0dd940a2abc` |
| 7 | falcon-refinedweb | 1,553 | 970,564 | 236 | 3,908 | 966,420 | 4,371,173 | [7853, 8089) | 61–63 | `0f26d6b4aa56892d8194623578dae32260c1b3459e2dddf7867934c52a37b0c9` |
| 8 | gab | 23,249 | 958,643 | 234 | 179 | 958,230 | 3,354,865 | [8089, 8323) | 63–65 | `7ebf2640b9eba8524cc34631b5b4f5e3b67f9dbf21d4eb9e8de73458af00f850` |
| 9 | m2d2_s2orc_unsplit | 167 | 16,844,264 | 4,112 | 1,512 | 16,838,640 | 73,450,318 | [8323, 12435) | 65–97 | `f9e4608406c0bb372a6d93c94b52ec14162f6f22b3b5f344fff2ecff05f55df0` |
| 10 | m2d2_wikipedia_unsplit | 49 | 4,839,217 | 1,181 | 1,841 | 4,836,195 | 21,586,958 | [12435, 13616) | 97–106 | `4d8d5225bc27e7859e0ee980579ff431314803937acb6bfc9a8ae66c38d4ac77` |
| 11 | manosphere_meta_sep | 2,382 | 965,236 | 235 | 2,676 | 962,325 | 4,065,983 | [13616, 13851) | 106–108 | `09b6b8efed1a3847107a353540f51e501eb6b65922f3a130606983249de23f1e` |
| 12 | mc4 | 1,108 | 970,427 | 236 | 3,771 | 966,420 | 4,044,953 | [13851, 14087) | 108–110 | `081f5f67541a337845e3fd9f89b1589112f20eb90871ead87f5049e75030a6e3` |
| 13 | ptb | 1 | 87,135 | 21 | 1,119 | 85,995 | 394,630 | [14087, 14108) | 110 | `ffb45ee0e9a0b5f9f6ed16c51175cf6f7a7ca89756128727a704233dd7b374e7` |
| 14 | redpajama | 632 | 641,116 | 156 | 2,140 | 638,820 | 2,685,571 | [14108, 14264) | 110–111 | `8bc20349c6e0191412591b9f940c5b49d35f723bb70b8c0eac4e73ec99696e7e` |
| 15 | twitterAAE_HELM_fixed | 50,000 | 841,005 | 205 | 1,325 | 839,475 | 2,473,231 | [14264, 14469) | 111–113 | `9d110d797b7a233ab5b82fdcdd3c7202f4f5bb93ffd14b5701b28774ccf32d93` |
| 16 | wikitext_103 | 59 | 249,808 | 60 | 4,048 | 245,700 | 1,114,668 | [14469, 14529) | 113 | `a3cadf50589690a770299069a1a7f9f81f0bd532c180713f276f20e6dc91002a` |

Total 14,529 sequences, 114 eval batches of 128 **[verified]**. "Target bytes" = sum of `bytes_per_token[tokens[i+1]]`
over the scored positions i = 0..4094 of every sequence (section 8.4).

#### 8.2 How eval examples are formed

* Same path as training, without any shuffle: `validation_sets` → `CausalLmDataset(TokenSeqDataset(cache, 4096))`
  per domain (`L/data/text/datasets.py:873-891,509-534`). Sequence i of a domain = its tokens `[4096 i, 4096 i + 4096)`;
  documents are concatenated in cache order, windows cross documents, the tail is dropped (column "dropped tail").
* Loss weights, segment ids from EOS, positions 0..4095: identical to section 3.1. Each sequence scores 4095
  positions.
* `tagged_eval_sets` gives each domain the single tag = its component name (`datasets.py:1064-1070`; component tags
  are `[]`). The order is the component order of section 8.1.
* `DomainTaggedDataset` concatenates the 16 domains in that order into one index space of 14,529 sequences
  (`L/eval.py:82-174`); the evaluator does not shuffle (`shuffle=False`, `eval.py:520-529`). The `DataLoader` reads
  batches of `EvalBatch = per_device_eval_parallelism · data_axis = 32 · 4 = 128` (logged 32 for all sizes, except 64 in the first 520m segment, which ran no eval;
  `L/trainer.py:1048-1049,1168-1177`) in index order, so one batch can hold two domains (column "eval batches").
  `per_device_eval_parallelism` defaults to `per_device_parallelism` (`L/trainer.py:1168-1177`), so EvalBatch depends
  on the GPU count and the microbatch setting: on 8 GPUs, 130m/300m keep 128 (16 · 8), but 520m/1_2b get 256 (32 · 8)
  unless the launcher's `PDP` knob changes it (`della_muonh_qwen3_scaling.py:103-104,787`); the first 520m segment
  had 64 · 4 = 256 but ran no eval. Because the per-domain bpb depends on the batching (trap 2 of section 8.3), a
  reimplementation must form eval batches of exactly 128 rows in this order whatever its own device count, and must
  not derive the eval batch size from its GPU count.
* The last batch (113) holds 65 real rows and 63 padding rows. Padding rows are all zeros (tokens 0, loss weight 0,
  tag vector 0) (`L/data/loader.py:349-372,646-648`), so they contribute nothing.
* `max_eval_batches: null` → the full set every time (`L/main/train_lm.py:350-352`).
* Cadence (off by one; read carefully). The hook test is `info.step > 1 and info.step % hook.every == 0`
  (`L/trainer.py:153`), `every = steps_per_eval = 1000` (`L/main/train_lm.py:372`; `della_muonh_qwen3_scaling.py:445`).
  `StepInfo` is built from `result.new_state` after the update (`L/trainer.py:557`), and `StepInfo.step =
  int(state.step) - 1` (`L/callbacks/_core.py:76`; "0-indexed"). `TrainerState` sets `step = self.step + 1` on each
  update (`L/trainer_state.py:152`). So the hook fires at `state.step` = 1001, 2001, 3001, …, and the eval callback logs
  at `step=step.step` = 1000, 2000, … (`L/eval.py:340,349`). **The W&B eval point at logged step S (S = 1000, 2000, …)
  evaluates the weights after S + 1 updates (batches 0..S consumed).** The forced final eval (`L/trainer.py:626`)
  evaluates the weights after all `num_train_steps` updates and logs at `num_train_steps - 1` = 4958 (130m), 11443
  (300m), 9917 (520m), 22887 (1_2b). The cached 130m history
  (`N/cache/wandb/targets/della__muonh-qwen3-130m-della4xh100.hist.json.gz`) has eval rows at `_step` 1000, 2000,
  3000, 4000 and 4958. A reimplementation must evaluate after updates 1001, 2001, … and tag those evals 1000, 2000, …;
  evaluating after S updates is one update early at every intermediate point. The final targets are unaffected.
* Weights and dtype: the current (non-EMA) weights, cast to compute dtype bf16 (`eval.py:225-239`,
  `mp p=f32,c=bfloat16`). The baseline logs `model_averaging: null`, and the `eval/ema/*` branch runs only when
  `step.state.model_averaging is not None` (`L/eval.py:357`). The `-ema<beta>` variant runs set
  `model_averaging=EmaModelAveragingConfig(beta=EMA_BETA)` (`della_muonh_qwen3_scaling.py:813`) and log extra
  `eval/ema/...` keys; those are not baseline targets.

#### 8.3 Metric arithmetic (as logged)

Per batch b (`L/eval.py:546-594`), with per-position loss `l` (natural log), weight `w`, one-hot tag `t`:

```
L_b   = sum(l*w)                 W_b = sum(w)
L_b,d = sum over rows of tag d of l*w      W_b,d = same of w
Y_b,d = sum over rows of tag d of bytes_per_token[next token]*w
token_avg_loss  <- RunningMean.add(L_b / max(W_b,1), W_b)
loss_per_tag[d] <- RunningMean.add(L_b,d / W_b,d (0 if W_b,d = 0), W_b,d)
bpb_per_tag[d]  <- RunningMean.add(L_b,d / max(Y_b,d, 1) * log2(e), W_b,d)     # weighted by TOKENS
bpb             <- RunningMean.add(L_b / max(Y_b,1) * log2(e), W_b)
RunningMean.add(x, n): new_total = total + n; mean += (x - mean) * (n / new_total if new_total else 0)   # L/utils/stat_utils.py:24-30
```

Logged keys (`eval.py:444-485,596-660`), prefix `eval`:

All running means and totals are float32 (`RunningMean.zeros_like` of float32 zeros, `eval.py:597-600,837-841`), and
`W_d = loss_per_tag[d].total = bpb_per_tag[d].total` is a TOKEN count: both are accumulated with
`this_weights_per_tag` (`eval.py:564,573,589`). The name `total_bytes_per_tag_cpu` (`eval.py:629`) is misleading. NumPy
keeps float32 for float32 input: `np.mean` promotes only int/bool to float64 (`numpy/_core/_methods.py:127-129`), and
`np.average` uses `result_type(a, w)` = float32 (`numpy/lib/_function_base_impl.py:577-588`).

| key | code | value |
|---|---|---|
| `eval/paloma/<d>-marin-tokenizer/loss` | `float(loss_per_tag.mean[d])` (`eval.py:644`) | Σ_b W_b,d·(L_b,d/W_b,d) / Σ_b W_b,d = Σ_b L_b,d / Σ_b W_b,d up to float32 rounding |
| `eval/paloma/<d>-marin-tokenizer/bpb` | `float(bpb_per_tag.mean[d])` (`eval.py:646`) | Σ_b W_b,d·(L_b,d/max(Y_b,d,1))·log2 e / Σ_b W_b,d |
| `eval/loss` | `token_avg_loss.mean` (`eval.py:609`, logged `:448`), float32 on device | Σ_b W_b·(L_b/W_b) / Σ_b W_b |
| `eval/paloma/micro_loss` | `np.average(loss_per_tag.mean, weights=W·mask)` (`eval.py:637`), float32 | Σ_d W_d·loss_d / Σ_d W_d; equals `eval/loss` in exact arithmetic (130m step 1000: 4.7428202629089355 vs 4.742820739746094) |
| `eval/macro_loss` | `jnp.mean(loss_per_tag.mean)` (`eval.py:611`), float32, XLA reduction | mean of the 16 per-domain losses |
| `eval/paloma/macro_loss` | `np.mean(loss_per_tag.mean, where=mask)` (`eval.py:636`), mask = all 16 tags | float32 accumulator (NumPy float32 `add.reduce`, pairwise on contiguous runs; the exact masked order was not checked), then ÷ 16 (exact). Equal to `eval/macro_loss` up to ~1 float32 ulp (130m step 1000: both 4.8956499099731445, a float32 value) |
| `eval/bpb` | `bpb.mean` (`eval.py:614`, logged `:475`), float32 | Σ_b W_b·(L_b/max(Y_b,1))·log2 e / Σ_b W_b: per-batch bpb over all rows of the batch |
| `eval/paloma/bpb` | `np.average(bpb_per_tag.mean, weights=W·mask)` (`eval.py:641`, logged `:478-479`), float32 | Σ_d W_d·bpb_d / Σ_d W_d = Σ_b Σ_d W_b,d·(L_b,d/max(Y_b,d,1))·log2 e / Σ W. Differs from `eval/bpb` in exact arithmetic, because the ratio is per (batch, domain), not per batch (130m step 1000: 1.6721341609954834 vs 1.6692712306976318) |
| `eval/macro_bpb` | `jnp.mean(bpb_per_tag.mean)` (`eval.py:616`), float32 | mean of the 16 per-domain bpb |
| `eval/paloma/macro_bpb` | `np.mean(bpb_per_tag.mean, where=mask)` (`eval.py:640`), float32 | as `eval/paloma/macro_loss`; differs from `eval/macro_bpb` by one float32 ulp at 130m step 3000 (1.6098203659057617 vs 1.6098204851150513) |

Two traps:

1. **Key names.** On these della runs the per-domain keys carry the `-marin-tokenizer` suffix: the stage-1 target
   "eval/paloma/c4_en/bpb" is the W&B key `eval/paloma/c4_en-marin-tokenizer/bpb`. The 16-domain macro key has no
   suffix: `eval/paloma/macro_loss`.
2. **bpb is not total bits / total bytes.** It is a token-weighted mean of per-batch bpb values. It equals
   `Σ L / Σ Y · log2 e` only when a domain sits in one batch or its bytes per token are equal across its batches.
   c4_en spans batches 19–21 (24, 128 and 84 rows), so the logged value depends on the batch size 128 and on the
   concatenation order of section 8.1. To match logged numbers, use this exact batching and formula; also report the
   exact `Σ L / Σ Y · log2 e` beside it. (The vocabulary session notes that the W&B macro bpb reads about 0.004 high
   against an offline exact computation, `N/docs/survey/needs_from_sessions.md`, item 2; `eval.md` §4.3 measures the
   opposite sign and a smaller size: logged macro_bpb is 0.0004-0.0006 below the exact value. Open point.)

#### 8.4 Bytes per token

`bytes_per_token[i] = byte_length_of_token(tokenizer, i)` for i in 0..127,255 (`L/eval.py:182-192`,
`L/utils/hf_utils.py:24-47`): 0 for the 256 special ids (128000–128255, including BOS and EOS); for byte-fallback
tokens `<0xNN>` the hex length; otherwise `len(decode([id('.'), i]).utf8) - 1`. Table saved as
`N/cache/data/bytes_per_token_marin.i32` (128256 int32 LE, sha256
`d3be74509077b5ba84a2822f0348c7574b685404178786b7e0d473c64a334e4d`) **[verified]**. Bytes are counted for the
**target** token (`per_pos_token_id = roll(tokens, -1)`, `eval.py:237`), with the same weights as the loss, so the BOS
target after an EOS adds 0 bytes but its loss counts.

**Do not recompute this table from the vocabulary.** The rule decodes through `tokenizer.decode`
(`L/utils/hf_utils.py:46-47`, `L/tokenizers.py:638-639`), i.e. the HF tokenizers ByteLevel decoder, which turns the
decoded bytes into a string with lossy UTF-8 conversion. A byte-level BPE token that holds an incomplete UTF-8 sequence
therefore counts the bytes of the U+FFFD replacement character(s) (3 bytes each), not its true 1–3 raw bytes. A loader
that sums raw byte lengths from the BPE byte map gets a different table for those ids. Load
`bytes_per_token_marin.i32` as is and check its sha256. (Mechanism from the decoder; the number of affected ids was
not counted here.)

---

### 9. Ambiguities and open points

1. **Cache identity across the 09-24 deletion** (section 2.1). Counts match and the 130m pool reproduced after the
   restore; no token hash of the 09-14 copy exists. All hashes here pin the current copy.
2. **Segments from EOS vs from offsets** (section 3.1). Levanter uses the EOS token value. A loader that builds
   segments from document offsets differs on the rare documents with an EOS id inside text. Reading 1 (match
   levanter): segment boundary after every 128001. Reading 2 (cleaner): boundaries at document offsets. Stage 1 must
   use reading 1.
3. **bpb aggregation** (section 8.3). Reading 1 (match W&B): token-weighted mean of per-batch bpb with EvalBatch 128 and
   the order of section 8.1. Reading 2 (exact): Σ bits / Σ bytes. Compare to the logged pool with reading 1.
4. **Original TPU runs.** The launcher reproduces Kaiyue Wen's `qwen3_<size>_muonh_4096_*` runs (2025-10-12, 64 TPU
   devices; docstring `della_muonh_qwen3_scaling.py:3-11`). Whether the 2025 levanter mixture also shuffled inside
   blocks was not checked; it does not matter for stage 1, whose target is the della runs.
5. **Microbatch reduction order** (section 5) is a numerics question for the training-loop section; the token stream
   is fixed by this section. That section must carry the 520m schedule: one microbatch of 256 for updates 0..85
   (batches 0..85), two microbatches of 128 from update 86 on (`optim.md` §8).

---


---

## Part 4. Eval (`eval.md`: Evaluation and logged metrics)

Stage-1 spec section. Goal: the new stack evaluates the same text in the same way and logs the same W&B keys as the
levanter muonh_qwen3 baseline, so `scripts/della/objective_compare.py` runs on its runs unchanged.

### 0. Sources

- Reference repo: `project/marin` (read-only). Line numbers below are at HEAD `1793624070`.
- Commits. The 130m run is one segment at `59ac0d64` (2026-09-13). The other sizes resumed over several segments, each
  at the launcher commit of its own job. The `.wandb` file of each segment
  (`$ROOT/wandb-runs/wandb/run-*/run-*.wandb`) records its commit:

  | size | segments | commits, in segment order |
  |---|---|---|
  | 130m | 1 | 59ac0d64 |
  | 300m | 3 | 59ac0d64, 28fd5996 x2 |
  | 520m | 11 | 59ac0d64, 28fd5996 x3, 33415cba x6, ac6af868 |
  | 1_2b | 27 | 59ac0d64, 28fd5996 x3, 33415cba x7, ac6af868, 0672e5ce x3, afdc3498 x2, cead09a2 x10 |

  W&B `config.git_commit` (cache/wandb/api/<run>.meta.json) holds only the last segment's commit: 59ac0d64, 28fd5996,
  ac6af868, cead09a2. For each of the seven commits, `git diff --stat <c> HEAD` over `eval.py`, `callbacks/`, `data/`,
  `models/`, `layers/`, `kernels/`, `tracker/`, `analysis/`, `utils/{hf,stat}_utils.py`, `tokenizers.py`,
  `main/train_lm.py`, `grad_accum.py`, `trainer.py`, `H/nn/loss.py`, `lib/marin/.../experiment/data.py` and
  `experiments/datasets/paloma.py` shows one file only: `trainer.py` (+17 lines, the context variable
  `_TRACED_TRAIN_STEP`, trainer.py:108-115, 742-747), which does not change the math. So the eval and logging library
  code is the same at every segment. The launcher differs between commits; its one eval-relevant difference is the
  520m microbatch size (§2.3.3).
- Paths: `L/` = `lib/levanter/src/levanter/`, `H/` = `lib/haliax/src/haliax/`, `launcher` =
  `experiments/references/della_muonh_qwen3_scaling.py`.
- Authoritative values: the W&B runs `reself/marin-della/muonh-qwen3-{130m,300m,520m,1_2b}-della4xh100` (state
  `finished`). The config, the summary and the logging steps are cached in
  `marincpp/cache/wandb/api/<run>.meta.json`, written by `cache/wandb/_tools/api_meta.py` on della-vis1. The history
  of each local run segment is cached in `marincpp/cache/wandb/run-*.jsonl.gz`.
- Default knobs: no VARIANT, EMA_BETA=0, TPP=20, VOCAB_K=0, SEQ_LEN=4096, BATCH=0, SMOKE_STEPS=0, NUM_GPUS=4.

### 1. Eval schedule

1. The trainer adds the eval hook with `every=config.trainer.steps_per_eval` (L/main/train_lm.py:362-372).
   `steps_per_eval = eval_every = 1000` (launcher:445, 789). Logged config: `trainer.steps_per_eval = 1000` for all
   four sizes.
2. A non-jit hook fires when `info.step > 1 and info.step % every == 0` (L/trainer.py:151-154). After the last step,
   `train()` runs every hook once more with `force=True` (L/trainer.py:625-626).
3. `info.step = state.step - 1` (L/callbacks/_core.py:76). `info.state` is the state after the update. So the eval
   that is logged at step `s` uses the parameters after `s + 1` updates. The final eval (logged at `N - 1`) uses the
   final parameters after all `N` updates.
4. No eval runs at step 0. `max_eval_batches = None` (launcher:790), so every eval reads the whole eval set.

| size | N = `num_train_steps` | logged eval steps | count |
|---|---|---|---|
| 130m | 4959 | 1000, 2000, 3000, 4000, 4958 | 5 |
| 300m | 11444 | 1000, 2000, ..., 11000, 11443 | 12 |
| 520m | 9918 | 1000, 2000, ..., 9000, 9917 | 10 |
| 1_2b | 22888 | 1000, 2000, ..., 22000, 22887 | 23 |

The W&B API (`scan_history`, page size 10000) returns step 10000 twice at 300m and steps 10000 and 20000 twice at
1_2b. The local `.wandb` files hold each of these evals once (cache/wandb/run-20260914_123815-...-300m...,
run-20260915_185803-...-1_2b..., run-20260917_080526-...-1_2b...). The duplicates are a pagination artifact.
Each eval step appears once.

Resumed segments. Each segment restarts from the latest temporary checkpoint (`save_interval: "0:10:00"`,
`keep_last_temporary_checkpoints: 1`, logged `trainer.checkpointer`), so up to about 10 min of steps before the kill
are trained again. The tracker drops every step below `run.step` (L/tracker/wandb.py:136-158). So the W&B rows for a
replayed step come from the killed segment, whose trajectory was discarded; the surviving trajectory restarted from the
checkpoint. Replay is deterministic in data but not in GPU arithmetic, so those rows can differ from the surviving
trajectory by run-to-run noise. Segment boundaries in the local history (last logged step -> first logged step):
520m 119->120, 1149->1154, 2211->2212, ...; 1_2b 8629->8629 (one step logged by both). The checkpoint step of each
resume is not in the caches. Only values logged by the last segment, among them the final eval at `N - 1`, come from
the surviving trajectory for sure. A step-by-step parity check against intermediate 300m/520m/1_2b history must allow
for this; 130m (one segment) has no replay.

Edge case (not reached by the four sizes): if `N - 1` were a multiple of 1000, the eval would run twice at the same
step. `cb_tagged_lm_evaluate` has no de-duplication (L/eval.py:339-384). The new stack must run it once.

### 2. Eval data

#### 2.1 Domains

The eval set is the 16 Paloma validation caches, tokenized with the Marin tokenizer
(`data.tokenizer = marin-community/marin-tokenizer`, logged), version `2026.06.28`:
`marin_store_big/paloma/<subset>-marin-tokenizer/2026.06.28/validation/` (experiments/datasets/paloma.py:23-57;
launcher:545). fineweb-edu-10B also has a component, but its cache has only `train/`. For a missing validation cache
with no validation source, `build_caches` skips the component (L/data/text/datasets.py:999-1002). So the evaluator
has exactly 16 tags and no others.

Tag of each domain: `component.tags + [name]` = `["paloma/<subset>-marin-tokenizer"]`
(L/data/text/datasets.py:1064-1070; logged `components.*.tags = []`). The hierarchy adds one parent tag, `paloma`
(L/eval.py:660-670).

The 16 subset names (the names in the keys). Each name equals its Paloma directory name, except `4chan`, which reads
the directory `4chan_meta_sep` (paloma.py:23-40). The long names `m2d2_s2orc_unsplit`, `m2d2_wikipedia_unsplit` and
`manosphere_meta_sep` are the directory names; a hand-written list with short names (e.g. `m2d2_s2orc`) does not match
the keys.

`4chan, c4_100_domains, c4_en, dolma-v1_5, dolma_100_programing_languages, dolma_100_subreddits, falcon-refinedweb,
gab, m2d2_s2orc_unsplit, m2d2_wikipedia_unsplit, manosphere_meta_sep, mc4, ptb, redpajama, twitterAAE_HELM_fixed,
wikitext_103`

#### 2.2 Sequences

1. Format: plain text, no packing (`pack = null`, so `_effective_pack` is False for `TextLmDatasetFormat`,
   L/data/text/datasets.py:362-370). Each domain is a `CausalLmDataset(TokenSeqDataset(cache, 4096))`
   (L/data/text/datasets.py:529-534).
2. The cache holds one flat token stream per domain (documents back to back, as tokenized with `enforce_eos = true`).
   Sequence `i` is the tokens `[4096 i, 4096 i + 4096)` of that stream (L/data/text/datasets.py:105-106). The count is
   `floor(tokens / 4096)` (L/data/text/datasets.py:91-92). The tail is dropped. Documents cross sequence boundaries.
3. Loss weight: `w[t] = 1` for `t = 0..4094`, `w[4095] = 0` (`causal_loss_mask`, L/data/text/examples.py:141-149,
   200-209). Position `t` predicts token `t + 1`. So each sequence scores 4095 targets.
4. Attention: causal, with document blocking. `segment_id[t] = cumsum(tokens[t-1] == eos_id)`, with
   `segment_id[0] = 0` (L/data/text/examples.py:166-171). `eos_id = tokenizer.eos_token_id`
   (L/data/text/datasets.py:775). The config logs `block_cross_document_attention = true`. The EOS token stays in the
   old document. The prediction across a boundary (EOS to the next token) is scored (`w = 1`).

Per-domain counts, read from each cache's `shard_ledger.json` (`total_num_rows`, `field_counts.input_ids`). `T` is the
number of scored targets, `seqs x 4095`.

| subset | docs | tokens | seqs | dropped tail | T (scored) |
|---|---|---|---|---|---|
| 4chan | 523 | 929,479 | 226 | 3,783 | 925,470 |
| c4_100_domains | 14,059 | 9,464,109 | 2,310 | 2,349 | 9,459,450 |
| c4_en | 2,091 | 967,618 | 236 | 962 | 966,420 |
| dolma-v1_5 | 4,600 | 2,883,425 | 703 | 3,937 | 2,878,785 |
| dolma_100_programing_languages | 8,123 | 8,502,810 | 2,075 | 3,610 | 8,497,125 |
| dolma_100_subreddits | 46,347 | 9,435,238 | 2,303 | 2,150 | 9,430,785 |
| falcon-refinedweb | 1,553 | 970,564 | 236 | 3,908 | 966,420 |
| gab | 23,249 | 958,643 | 234 | 179 | 958,230 |
| m2d2_s2orc_unsplit | 167 | 16,844,264 | 4,112 | 1,512 | 16,838,640 |
| m2d2_wikipedia_unsplit | 49 | 4,839,217 | 1,181 | 1,841 | 4,836,195 |
| manosphere_meta_sep | 2,382 | 965,236 | 235 | 2,676 | 962,325 |
| mc4 | 1,108 | 970,427 | 236 | 3,771 | 966,420 |
| ptb | 1 | 87,135 | 21 | 1,119 | 85,995 |
| redpajama | 632 | 641,116 | 156 | 2,140 | 638,820 |
| twitterAAE_HELM_fixed | 50,000 | 841,005 | 205 | 1,325 | 839,475 |
| wikitext_103 | 59 | 249,808 | 60 | 4,048 | 245,700 |
| total | | 59,550,094 | 14,529 | | 59,496,255 |

#### 2.3 Order and batches

1. `DomainTaggedDataset` concatenates the 16 domains in component order (L/eval.py:89-141). The evaluator does not
   shuffle (`shuffle=False`, L/eval.py:520, and `cb_tagged_lm_evaluate` does not pass it, L/eval.py:326-334).
2. Component order: `mixture()` inserts the validation handles in list order (lib/marin/src/marin/experiment/data.py:312-315),
   and the list is `paloma_datasets(...).values()` in `_PALOMA_SUBSETS` order (paloma.py:23-40, 55-57;
   launcher:545). `build_token_datasets` keeps that order (L/data/text/datasets.py:728). So the order is the list in
   §2.1.
   - The config object reaches the trainer in process: `importlib.import_module("levanter.main.train_lm").main(train_config)`
     (lib/marin/src/marin/training/training.py:560), with no reserialization of the data config (only `deepcopy`,
     which keeps dict order). `_PALOMA_SUBSETS` insertion order is byte-sorted, and the local W&B `config.yaml`
     (cache/wandb/config/run-20260914_041824-...-130m...config.yaml:60-285) lists the components in that same order.
     So the run order is the §2.1 order (reading A).
   - The W&B API config lists the components in another order (gab, mc4, ptb, 4chan, c4_en, ...: by key length, then
     bytes). That is the key order of W&B's server-side JSON, not the run order (reading B, rejected by the code). Only
     the bpb keys depend on the order (§4.3). §7, test 2 stays as a confirmation.
3. Eval batch: `EvalBatch = per_device_eval_parallelism x data_axis_size` (L/trainer.py:983-984, 1047-1049).
   `per_device_eval_parallelism = -1` defaults to `per_device_parallelism` (L/trainer.py:1168-1175). The W&B API
   config (last segment) has `per_device_eval_parallelism = 32` for all four sizes (130m/300m: 128 / 4 GPUs;
   520m/1_2b: PDP 32, launcher:435, 787). The config inside each segment's `.wandb` file gives
   `per_device_parallelism = per_device_eval_parallelism = 32` for every segment of every size, with one exception:
   520m segment 0 (run-20260914_063908, commit 59ac0d64, logged steps 0-119) has 64 for both
   (cache/wandb/config/run-20260914_063908-...-520m...config.yaml:497-498). At 59ac0d64 the launcher has
   `PER_DEVICE_PARALLELISM = {"1_2b": 32}` (`git show 59ac0d64:<launcher>`, line 79); 28fd5996 ("Run 520m with two
   microbatches per step on 4x80GB") adds `"520m": 32`. That segment ran no eval. So every logged eval, at every size,
   used an eval batch of 128 sequences on 4 GPUs.
   - On 8 GPUs the same code gives 130m/300m `16 x 8 = 128` but 520m/1_2b `32 x 8 = 256`. Batch 256 changes the bpb
     estimators (§4.3), not the loss keys. Parity runs keep 128.
4. Batch `k` holds global sequences `[128 k, 128 k + 128)`. There are 114 batches. The last batch holds 65 real
   sequences and 63 padding rows. Padding rows are all zeros: tokens 0, loss weight 0, tag vector 0
   (`pad_final_batch = True`, L/data/loader.py:81, 101, 367-368, 646-648). They add nothing to any sum. Batches
   straddle domain boundaries.

Batch layout under reading A (sequence index ranges are inclusive):

| subset | global seqs | batches |
|---|---|---|
| 4chan | 0-225 | 0-1 |
| c4_100_domains | 226-2535 | 1-19 |
| c4_en | 2536-2771 | 19-21 |
| dolma-v1_5 | 2772-3474 | 21-27 |
| dolma_100_programing_languages | 3475-5549 | 27-43 |
| dolma_100_subreddits | 5550-7852 | 43-61 |
| falcon-refinedweb | 7853-8088 | 61-63 |
| gab | 8089-8322 | 63-65 |
| m2d2_s2orc_unsplit | 8323-12434 | 65-97 |
| m2d2_wikipedia_unsplit | 12435-13615 | 97-106 |
| manosphere_meta_sep | 13616-13850 | 106-108 |
| mc4 | 13851-14086 | 108-110 |
| ptb | 14087-14107 | 110 |
| redpajama | 14108-14263 | 110-111 |
| twitterAAE_HELM_fixed | 14264-14468 | 111-113 |
| wikitext_103 | 14469-14528 | 113 |

### 3. Per-position loss

1. Weights: the raw parameters (`step.model = state.model`, L/callbacks/_core.py:72; L/eval.py:347-349). There is no
   EMA, see §5.
2. `inference_mode(model, True)`, then `mp.cast_to_compute(model)`: a bf16 copy of the f32 parameters (policy
   `p=f32,c=bfloat16`, launcher:782; L/eval.py:232-235). The forward pass is the training forward pass (model
   section).
3. `per_pos_loss = model.compute_next_token_loss(batch, reduction=None)` (L/eval.py:236; L/models/lm_model.py:319-354).
   This is `logsumexp(logits) - logit[target]` in natural log (nats). Its dtype is float32: the loss dtype follows
   the float32 loss weight (L/models/loss.py:78-80). There is no z-loss (`logsumexp_weight = None` at eval;
   `z_loss_weight = 0` in training, launcher:819). The target is `roll(tokens, -1)` (L/models/loss.py:76); the
   wrapped target at `t = 4095` has weight 0.
   - The returned value is already masked. The call passes `reduction=None, reduction_axis=()` (L/eval.py:236), so
     `maybe_reduce_loss` returns `arr * effective_weight` (H/nn/loss.py:143-144, called at L/models/loss.py:299), with
     `effective_weight = loss_weight * not_last_mask` (L/models/loss.py:22-32, 78-81). Then `compute_next_token_loss`
     adds `aux_loss` (L/models/lm_model.py:354), which is the integer 0 for plain Qwen3 (`split_activations`,
     L/models/lm_model.py:207-209). So `per_pos_loss[i,t] = CE[i,t] * w[i,t] + 0`, and the evaluator multiplies by `w`
     a second time (L/eval.py:555). With 0/1 weights the double product equals the single one. A NaN or inf CE on a
     weight-0 position still gives NaN (`inf * 0 = NaN`), so multiplying by 0 does not mask it. The new stack either
     reproduces the double multiply or masks with a select (`where(w > 0, CE, 0)`); both match the baseline whenever
     every CE is finite, which holds for the baseline (finite `eval/loss` at every eval).
4. `per_pos_token_id = roll(tokens, -1)`: the target id, used for the byte table (L/eval.py:238).
5. CE kernel path on H100 (code reading at HEAD; kernel code is unchanged since 59ac0d64,
   `git diff --stat 59ac0d64 HEAD -- L/kernels` is empty):
   1. `maybe_fused_next_token_loss` calls the fused API with `block_size=None`, `implementation=None`,
      `precision=None`, `dtype=float32` (L/models/loss.py:78-98, 264-277). With `implementation=None` the GPU list
      is `("batched_xla", "xla")` (L/kernels/pallas/fused_cross_entropy_loss/api.py:72, 160-169).
   2. The loss is computed per device inside `shard_map` (L/models/loss.py:291). Local rows `B = 32 x 4096 = 131,072`
      (eval batch 128 over 4 devices), `V = 128,256`, `x` = the bf16 final hidden states, `w` = the bf16 `lm_head`
      (policy `c=bfloat16`).
   3. `batched_xla` first checks `_h100_full_vocab_b_tiled_block_size` (fused_cross_entropy_loss/batched_xla.py:369-389):
      device kind contains `h100`, `x` and `w` are bf16, `B >= 8192` and `V >= 65536`, so it returns 8192. The
      forward then runs `_linear_softmax_cross_entropy_loss_full_vocab_b_tiled` (batched_xla.py:317-366, chosen at
      428-440), before any block-size validation. The inferred block sizes and the tuned table are not used on this
      path, and `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` (launcher:94-97) only skips the sweep.
   4. Math per tile of 8192 rows (fused_cross_entropy_loss/reference.py:66-110):
      `logits = dot_general(x_tile, w, precision=None, preferred_element_type=float32)`: a bf16 x bf16 GEMM over the
      full vocab, float32 output, `[8192, 128256]`, no vocab blocking. `lse = jax.nn.logsumexp(logits, axis=-1)` in
      float32 (reference.py:27-32; the GPU branch of `platform_dependent`). `loss = lse - logits[i, label_i]`. 16 tiles
      per device per eval batch.
   5. If `x` were not bf16 (e.g. the final norm returned float32), step 3 would not apply. Then `batched_xla` would
      use the tuned H100 entry (b 256, h 256, v 2048 for `("float32", "large-batch-small-h")` at hidden <= 1024,
      tuned_block_sizes.py:68-79; `BlockSizes()` = b 1024, h 512, v 1024 at 1_2b), whose weight tile
      `h x v x 2 B` exceeds the 101,376-byte NVIDIA limit (batched_xla.py:22, 175-200), so it would fall back to
      `xla`: streaming float32 logsumexp over vocab blocks of 8192 (tuned_block_sizes.py:773-794; reference.py:113-200).
      The dtype of `x` at the head is a model-section fact; the job log shows the fallback warning
      ("Fused cross-entropy implementation unavailable", api.py:176-185) if it happened.
   The float32 rounding of `l[i,t]` thus comes from the bf16 GEMM (float32 accumulate) and one full-row float32
   logsumexp. The new stack should use the same: bf16 operands, float32 logits, a float32 full-vocab (or exactly
   equivalent max-shifted) logsumexp. Test 3's 2e-4 tolerance assumes the remaining kernel differences
   (GEMM tiling and reduction order) stay at that level.

### 4. Aggregation and eval keys

#### 4.1 Per-batch accumulation (L/eval.py:546-592)

For batch `b`: per-position loss `l[i,t]`, weight `w[i,t]`, target id `y[i,t]`, tag one-hot `G[i,k]` (16 tags;
L/eval.py:124-131), bytes `beta[i,t] = bytes_per_token[y[i,t]]`. All in float32 on device:

```
L   = sum_{i,t} l*w            W   = sum_{i,t} w            B   = sum_{i,t} beta*w
L_k = sum_{i,t} l*w*G[i,k]     W_k = sum_{i,t} w*G[i,k]     B_k = sum_{i,t} beta*w*G[i,k]
log2e = log2(e) in float32 (L/eval.py:548)

token_avg_loss.add(L / max(W, 1), W)
loss_per_tag.add(where(W_k > 0, L_k / W_k, 0), W_k)
bpb.add(L / max(B, 1) * log2e, W)              # weighted by TOKENS, not bytes
bpb_per_tag.add(L_k / max(B_k, 1) * log2e, W_k)  # weighted by TOKENS, not bytes
```

`RunningMean.add(x, w)`: `new_total = total + w; mean = mean + (x - mean) * where(new_total, w / new_total, 0)`
(L/utils/stat_utils.py:24-30). The state starts at zeros (L/eval.py:597-602; L/eval.py:838-841).

Reductions as written in the code (L/eval.py:553-581):

- `l*w` is formed first (eval.py:555). `L = jnp.sum(l*w)` and `W = jnp.sum(w)` reduce over both axes (eval.py:556-557).
- `W_k = einsum("bt,bk->k", w, G)` and `L_k = einsum("bt,bk->k", l*w, G)` (eval.py:564-565);
  `B = jnp.sum(beta*w)` and `B_k = einsum("bt,bt,bk->k", beta, w, G)` (eval.py:578-581). `G` is int32
  (eval.py:128) and is promoted to float32. The einsums use DEFAULT precision and `out_sharding = P(None)`
  (replicated, eval.py:549). `beta` is the int32 byte table gathered at the target ids (eval.py:577), promoted to
  float32 in the products.
- The batch axis is sharded over the 4 devices, so each sum is a per-device partial sum followed by a cross-device
  all-reduce. The order of the float32 additions inside each reduction (over `t`, over `b`, across devices) is chosen
  by XLA and is not fixed by the code. A bitwise match of these sums is therefore not a target; float32 accuracy is.
- Observed precision constraint. If the einsum reduced over `t` first and then ran the contraction over `b` with TF32
  inputs, every row sum of `w` (4095) would round to 4096 and every `loss_k` would be low by the factor 4095/4096
  (about 1e-3 nats at loss 4), while `eval/loss` (plain `jnp.sum`) would not. The logged gap between `eval/loss` and
  `eval/paloma/micro_loss` is at most 1.9e-6 (§4.2). So the per-tag sums ran at effective float32 precision. The new
  stack computes all six sums in float32 or wider, never in TF32.

#### 4.2 Final values (L/eval.py:596-658) and keys (L/eval.py:444-484)

Let `loss_k = loss_per_tag.mean[k]`, `bpb_k = bpb_per_tag.mean[k]`, `n_k = loss_per_tag.total[k]` (scored tokens).

| key | value | source |
|---|---|---|
| `eval/loss` | `token_avg_loss.mean` = `sum L / sum W` (all 16 domains, token-weighted) | eval.py:609, 448 |
| `eval/paloma/micro_loss` | `np.average(loss_k, weights=n_k)` over the float32 `loss_k` and `n_k`; equals `eval/loss` only up to float32 accumulation error (see below) | eval.py:637, 472 |
| `eval/paloma/<subset>-marin-tokenizer/loss` (x16) | `loss_k` = `sum_b L_k / sum_b W_k` | eval.py:644, 469 |
| `eval/macro_loss` | `jnp.mean(loss_k)` over all 16 tags | eval.py:611, 455 |
| `eval/paloma/macro_loss` | `np.mean(loss_k, where=mask)`, mask = children of `paloma` with `n_k > 0` (all 16) | eval.py:631-636, 463 |
| `eval/bpb` | `bpb.mean`: token-weighted mean over batches of the batch ratio | eval.py:614, 475 |
| `eval/paloma/bpb` | `np.average(bpb_k, weights=bpb_per_tag.total)`; `.total` holds TOKENS (variable named bytes) | eval.py:629, 641, 479 |
| `eval/paloma/<subset>-marin-tokenizer/bpb` (x16) | `bpb_k` | eval.py:646, 479 |
| `eval/macro_bpb` | `jnp.mean(bpb_k)` over all 16 tags | eval.py:616, 477 |
| `eval/paloma/macro_bpb` | `np.mean(bpb_k, where=mask)` | eval.py:640, 483 |
| `eval/loading_time` | seconds spent waiting for eval batches | eval.py:449, 604 |
| `eval/total_time` | seconds for the whole `evaluate()` | eval.py:430-434, 450 |

That is 42 keys per eval. The history of every baseline run holds exactly these 42 `eval/` keys.

- macro_loss and macro_bpb: the unweighted arithmetic mean over the 16 domains of §2.1 (each domain counts 1/16).
  `eval/macro_*` and `eval/paloma/macro_*` hold the same value, because all tags are Paloma tags. They can differ in
  the last float32 bit (`jnp.mean` on device vs `np.mean` on host): 130m history step 3000, `eval/macro_bpb`
  1.6098204851150513 vs `eval/paloma/macro_bpb` 1.6098203659057617; 520m step 9917, 1.2995150089263916 vs
  1.2995151281356812; 1_2b step 22887, `eval/macro_loss` 3.360872268676758 vs `eval/paloma/macro_loss`
  3.360872507095337.
- `eval/loss` vs `eval/paloma/micro_loss`: two formulas, not one value. `eval/loss` is the `RunningMean` of the
  per-batch means `L / max(W, 1)` (eval.py:567, 609). `micro_loss` is `np.average` of the 16 float32 `loss_k`,
  weighted by the float32 `n_k` (eval.py:637). The history differs by a few float32 ULPs: 130m step 2000
  4.475258827 vs 4.475260735 (1.9e-6); 300m step 1000 4.594097614 vs 4.594095707 (1.9e-6); 520m step 9917
  3.338779449 vs 3.338778734 (7.2e-7); at other steps they are equal. The new stack computes each key with its own
  formula.
- History vs summary. The W&B summary (which `objective_compare.py` reads) can differ from the history row in the last
  float64 digit: 130m step 4958 history `eval/macro_bpb` = `eval/paloma/macro_bpb` = 1.5112996101379395; the summary
  (cache/wandb/api/*130m*.json) renders both as 1.5112996101379397, which is not a float32 value. Bit-level tests
  compare history rows or the raw float32 values, never `run.summary`.
- c4_en bpb: the key is `eval/paloma/c4_en-marin-tokenizer/bpb`, not `eval/paloma/c4_en/bpb`. c4_en spans batches
  19-21, so its value is a token-weighted mean of three batch ratios.
- The losses are exact token means (up to float32 rounding) and do not depend on the order or the batch size. The bpb
  keys depend on both (§4.3).

Logged final values (W&B summary, step N-1, seed 0):

| size | eval/paloma/macro_loss | eval/macro_bpb | eval/paloma/c4_en-marin-tokenizer/bpb | eval/loss | eval/bpb | eval/paloma/bpb | eval/total_time (s) |
|---|---|---|---|---|---|---|---|
| 130m | 4.179829 | 1.511300 | 1.163581 | 3.968565 | 1.387914 | 1.390518 | 21.6 |
| 300m | 3.808742 | 1.382257 | 1.056264 | | 1.253406 | 1.256051 | 43.6 |
| 520m | 3.572554 | 1.299515 | 0.988297 | | 1.166927 | 1.169594 | 95.6 |
| 1_2b | 3.360873 | 1.225783 | 0.927625 | | 1.089647 | 1.092350 | 112.9 |

Seed references at 130m (docs/della/objective-hillclimb.md in `project/marin`):

- c4_en bpb, raw baseline pool n=8: mean 1.16327, sd 0.00035 (line 58).
- macro_loss, 4-seed baseline n=4: 4.1848 +- 0.0043 (line 216; the same line gives c4_en 1.16316 +- 0.00038 for
  those 4 seeds). The ledger has no n=8 macro value.
- Run-to-run sd of macro: 0.007-0.010 (line 58).
- These ledger values are superseded by the W&B recomputation in `targets.md` §3 (same 8 runs): c4_en bpb 1.16326,
  sd 0.00033; macro_loss 4.18646, sd 0.00577 (n = 8). The gates use the W&B values.

#### 4.3 Bits per byte

Bytes table (L/eval.py:182-192; L/utils/hf_utils.py:24-47): an int32 array of length `len(tokenizer.get_vocab())`.
For token `i`:

1. `i` in `all_special_ids`: 0 bytes. This is Levanter's `MarinTokenizer` set (L/tokenizers.py:680-682), built by
   `_collect_special_ids` (L/tokenizers.py:750-768): the bos, eos and pad ids plus every `added_tokens_decoder`
   entry with `special: true` in the tokenizer config. It is not HF transformers' `all_special_ids` (built from the
   special-tokens map), which can omit reserved special tokens; a table built with transformers can then give them
   nonzero bytes and fail test 1.
2. Token string matches `<0xNN>`: `len(bytes.fromhex(NN))`. The token string is `convert_ids_to_tokens(i)`, which
   returns `<unk:i>` for an id missing from the vocab (L/tokenizers.py:670-673).
3. Otherwise: `len(decode([id("."), i]).encode("utf-8")) - 1`, where `id(".") = encode(".", add_special_tokens=False)[0]`
   and `decode` uses `skip_special_tokens=False` (L/tokenizers.py:638-639, the Rust `tokenizers` decoder). The "."
   prefix keeps a leading space.

The table is indexed by the TARGET id (L/eval.py:577). Special targets (EOS, BOS) add their loss to the numerator
and 0 bytes to the denominator.

Base: `log2e * nats / bytes`, so bpb is in bits (log2). The losses are in nats (natural log).

Two deviations from exact bits per byte:

1. Byte overcount. A byte-level BPE token that is only part of a multi-byte UTF-8 character decodes to U+FFFD, which
   counts 3 bytes. The documented Levanter/exact byte ratio, mean over the 16 subsets, is 1.0019 for the 128K Marin
   tokenizer and up to 1.0090 for its 8K truncation (docs/della/over-vocab.md:201-203; data:
   `tmp_plots/ss130/bpb_bytes_check_128000_64000.json`). Per subset at 128K: twitterAAE_HELM_fixed 1.0170,
   m2d2_s2orc_unsplit 1.0037, gab 1.0034, dolma_100_programing_languages 1.0033, dolma_100_subreddits 1.0011,
   dolma-v1_5 1.0005, wikitext_103 1.0005, mc4 1.0004, falcon-refinedweb 1.0003, m2d2_wikipedia_unsplit 1.0003,
   redpajama 1.0001, c4_100_domains 1.0001, c4_en 1.0001, manosphere_meta_sep 1.0001, 4chan 1.00004, ptb 1.0000.
   More bytes give a lower bpb. So the overcount alone makes the logged bpb low.
2. Estimator. `bpb_k` is a token-weighted mean of per-batch ratios `L_k,b / B_k,b`, not `sum L_k / sum B_k`. A
   whole-subset byte factor therefore cannot correct it (tmp_plots/vocab/bpb_exact_tables.py:1-9). The error is
   large where bytes per token vary between batches.

Exact per-domain bpb: `bpb_exact_k = loss_k * T_k / (ln 2 * B_exact_k)`. `T_k` is from §2.2. `B_exact_k` is the
exact UTF-8 bytes of the scored targets (one byte per character of the byte-level token string, specials 0), from
`tmp_plots/vocab/bpb_exact_tables.json["128000"]["4096"]`. Applied to the logged 130m seed-0 final per-domain losses:

| subset | B_exact | logged bpb | exact bpb | exact - logged |
|---|---|---|---|---|
| twitterAAE_HELM_fixed | 2,431,877 | 2.99021 | 3.01207 | +0.02186 |
| gab | 3,343,550 | 2.20459 | 2.21204 | +0.00745 |
| dolma_100_programing_languages | 28,736,507 | 1.99424 | 1.99597 | +0.00173 |
| dolma_100_subreddits | 41,422,446 | 1.45258 | 1.45381 | +0.00123 |
| wikitext_103 | 1,114,114 | 1.15985 | 1.16043 | +0.00058 |
| c4_en | 4,530,247 | 1.16358 | 1.16372 | +0.00014 |
| falcon-refinedweb | 4,369,773 | 1.24536 | 1.24542 | +0.00006 |
| 4chan | 3,108,487 | 1.51888 | 1.51889 | +0.00001 |
| ptb | 394,630 | 1.37550 | 1.37550 | 0.00000 |
| manosphere_meta_sep | 4,065,779 | 1.50705 | 1.50660 | -0.00045 |
| m2d2_wikipedia_unsplit | 21,580,646 | 1.12943 | 1.12892 | -0.00050 |
| c4_100_domains | 43,838,319 | 1.16046 | 1.15991 | -0.00055 |
| mc4 | 4,043,326 | 1.24089 | 1.24020 | -0.00069 |
| m2d2_s2orc_unsplit | 73,182,941 | 1.15287 | 1.15022 | -0.00266 |
| dolma-v1_5 | 12,400,184 | 1.33113 | 1.32313 | -0.00799 |
| redpajama | 2,685,179 | 1.55418 | 1.54070 | -0.01349 |

ptb and wikitext_103 each lie in one batch: there the estimator is exact, and the difference is the byte factor
alone (ptb 1.0, wikitext_103 1.0005 x 1.16 = 0.00058). That agrees with the layout of §2.3.

Macro and c4_en, logged vs exact (exact - logged), all sizes:

| size | macro_bpb logged | macro_bpb exact | diff | c4_en diff | pooled exact `sum loss_k T_k / (ln2 sum B_k)` |
|---|---|---|---|---|---|
| 130m | 1.51130 | 1.51172 | +0.00042 | +0.00014 | 1.35580 |
| 300m | 1.38226 | 1.38275 | +0.00050 | +0.00014 | 1.22468 |
| 520m | 1.29952 | 1.30014 | +0.00063 | +0.00013 | 1.14064 |
| 1_2b | 1.22578 | 1.22639 | +0.00061 | +0.00013 | 1.06509 |

- The logged macro_bpb reads 0.0004-0.0006 LOW. The survey note "the W&B macro reads about 0.004 high"
  (docs/survey/needs_from_sessions.md) disagrees in sign and size with this measurement. Open point.
- `eval/bpb` and `eval/paloma/bpb` are token-weighted. They are about 0.032 below the pooled byte-weighted value.
  Neither is a bits-per-byte of the whole set.

### 5. Raw or EMA weights

Raw weights. `model_averaging = null` in the logged config (launcher:813: an EMA config only if `EMA_BETA > 0`;
default 0, launcher:187). The callback evaluates `step.model` when `eval_current` is True (default) and evaluates
`step.eval_model` under prefix `eval/ema/` only when `model_averaging` is not None (L/eval.py:288-289, 347-360). The
baseline therefore logs no `eval/ema/*` key. The pool runs with suffix `-ema0.999` log both. The comparison script
reads only their raw `eval/` keys (§6).

### 6. Train-time metrics

Step convention for all keys: W&B `_step` = the `step` argument = `info.step` (L/tracker/wandb.py:132-157). At logged
step `s`, `theta_s` = the parameters after `s` updates, and batch `s` is the batch of update `s`.

| key | definition | logged at |
|---|---|---|
| `train/loss` | `result.loss.item()` (L/trainer.py:548): loss of batch `s` at `theta_s`. Per microbatch, `sum(l*w) / sum(w)` over all positions (H/nn/loss.py:132-138), nats, float32, no z-loss; averaged over microbatches (L/grad_accum.py:155-157; 1 microbatch at 130m/300m, 2 at 1_2b from step 0, 2 at 520m except its first segment: 520m segment 0 ran PDP 64, one microbatch of 256, and logged steps 0-119, so those `train/loss` rows and the updates before the first resume checkpoint (updates 0-85; the next segment resumed at `state.step` 86, `optim.md` §8) used one microbatch; §2.3.3). All positions carry equal weight (4095 per sequence), so this is the token mean of the batch. Logged by `log_step_info` (L/callbacks/_metrics.py:70-83, hook added at L/trainer.py:638-641). | steps 2..N-1; logged again at N-1 by the forced final hooks (same value) |
| `global_step` | `s` | steps 2..N-1 |
| `run_progress` | `offset(s+1) / offset(N)` = `(s + 1) / N` at a constant batch (L/callbacks/_metrics.py:73-79) | steps 2..N-1 |
| `optim/learning_rate`, `optim/adam_lr` | `opt_state.hyperparams` of the state after update `s` (L/tracker/helpers.py:23-34): the rates that update `s` used. Check, 130m final step 4958: 5.041361e-06 = f32(0.02) x (1 - f32(3966/3967)) and 2.016544e-06 = f32(0.008) x (1 - f32(3966/3967)), with 3967 = int(0.8 x 4959) decay steps (`optim.md` §6.1). The real-number value 0.02 / 3967.2 = 5.04134e-06 does not match the logged value. | steps 2..N-1 |
| `throughput/duration` | wall time of `train_step`: the jitted step plus `loss.item()`, so it includes the device sync; it excludes hooks and eval (L/trainer.py:537-557) | steps 2..N-1 |
| `throughput/examples_per_second` | `B / duration` (L/callbacks/_metrics.py:60) | steps 2..N-1 |
| `throughput/tokens_per_second` | `4096 * B / duration` | steps 2..N-1 |
| `throughput/gflops_per_second` | `flops_per_example * B / duration / 1e9` | steps 2..N-1 |
| `throughput/mfu` | `100 * flops_per_example * B / duration / (989.5e12 * num_devices)` (L/callbacks/_metrics.py:63-66, 100-104) | steps 2..N-1 |
| `throughput/total_tokens` | `4096 * B * (s + 1)` | steps 2..N-1 |
| `throughput/total_gflops` | `flops_per_example * B * (s + 1) / 1e9` | steps 2..N-1 |
| `throughput/hook_time` | wall time of all non-jit hooks and the outside part of the jit hooks at this step (L/trainer.py:559-566); includes eval and checkpoint time at those steps. It includes only the evals at multiples of `steps_per_eval`: the final eval runs in the forced `run_hooks` after `train()`'s loop (L/trainer.py:625-626), outside `capture_time`, and its row at `N - 1` was already logged. 130m step 4958: `hook_time` 0.0079 s, `eval/total_time` 21.6 s; step 1000: 22.94 s vs 22.91 s. The first eval's `eval/total_time` also includes the JIT compile of `accum_for_batch` (L/eval.py:430-433, 552) | steps 0..N-1 |
| `throughput/loading_time` | wall time waiting for the next train batch (L/trainer.py:578-598) | steps 0..N-1 |
| `grad/norm/total` | `optax.global_norm` (sqrt of the sum of squares over all leaves) of the raw gradient of batch `s` at `theta_s`: after the microbatch average, before clipping and before the optimizer (L/trainer.py:754, 771-775; L/analysis/tree_stats.py:150) | steps 10, 20, ..., floor((N-1)/10)*10 |
| `grad/norm/<param>` | global norm of one gradient leaf (L/analysis/tree_stats.py:122-124, 146-148) | same |
| `params/norm/total`, `params/norm/<param>` | the same norms of `theta_s`: the trainable parameters in f32 of the state BEFORE update `s` (`run_jit_hooks(state, ...)` receives the old state, L/trainer.py:775; L/callbacks/watch.py:181-195) | same |

At steps 0 and 1 only `throughput/hook_time` and `throughput/loading_time` are logged, because the hooks need
`info.step > 1` (L/trainer.py:153). The 130m history confirms: `train/loss` at steps 2..4958 (4957 rows); grad and
param norms at 10..4950 (495 rows).

Watch config (default `WatchConfig`, L/callbacks/watch.py:114-132; logged): targets `grads, params`, interval 10,
norms on, per-parameter norms on, histograms off, `split_scan_layers = true`. Jit hook cadence: the hook runs inside
the step when `state.step > 1 and state.step % 10 == 0` (L/trainer.py:170-175) and logs at `info.step`
(L/callbacks/watch.py:197-198). The forced final run does not include jit hooks (L/trainer.py:626 calls only
`run_hooks`).

Parameter names for the norm keys (13 per layer; the model uses `hybrid_norm`, so a layer has
`post_attention_layernorm`, `post_attn_layernorm` and `post_mlp_layernorm`):

```
embeddings.token_embeddings.weight
lm_head.weight
transformer.norm.weight
transformer.layers.{i}.{X}          for i in 0..L-1
transformer.layers.stacked.{X}      global norm of X over all layers (L/analysis/tree_stats.py:99-101)
X in: input_layernorm.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight,
      post_attention_layernorm.weight, post_attn_layernorm.weight, post_mlp_layernorm.weight,
      self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight,
      self_attn.q_proj.weight, self_attn.v_proj.weight
```

Keys per prefix: `13 (L + 1) + 4`, including `total`: 130m (L=6) 95, 300m (L=12) 173, 520m (L=24) 329, 1_2b (L=16)
225. The summaries confirm: 268, 424, 736 and 528 keys = 2 x (keys per prefix) + 78 others.

FLOPs per example (L/main/train_lm.py:388-392; `LlamaConfig.flops_per_token`, L/models/llama.py:200-210, inherited
by `Qwen3Config(LlamaConfig)`, L/models/qwen.py:299, used at launcher:649; L/utils/flop_utils.py:5-61):
`3 * 4096 * fpt`, with `V = len(tokenizer) = 128256` (L/main/train_lm.py:250) and

```
head_dim = hidden / heads
mlp      = 2 * 3 * hidden * inter
dense    = 2 * hidden * hidden
qkv      = 2 * hidden * (heads * head_dim + 2 * kv * head_dim)
attn     = (2 * S * S * heads * head_dim + 3 * S * S * heads + 2 * S * S * head_dim * heads) / S     (S = 4096)
fpt      = layers * (mlp + dense) + layers * (qkv + attn) + 2 * hidden * V
```

Check, 130m: `fpt = 227,868,672`, `3 * 4096 * fpt = 2,800,050,241,536` = logged `throughput/flops_per_example`. The
peak is 989.5e12 FLOP/s per H100 (logged `throughput/theoretical_flops_per_device`; lib/fray/src/fray/device_flops.py).
The new stack must use the same analytic FLOPs and the same peak, so its MFU stays comparable when its kernels
differ.

Summary-only keys (no history): `parameter_count` (130m 154,154,240; 300m 299,669,760; 520m 602,509,312; 1_2b
1,431,443,456; L/main/train_lm.py:348), `throughput/device_kind`, `throughput/theoretical_flops_per_device`,
`throughput/theoretical_flops`, `throughput/flops_per_example`, `throughput/{p10,p50,p90,mean,stddev}_mfu` and
`throughput/mfu_sample_count` over a trailing window of 500 steps (L/callbacks/_metrics.py:105-176),
`hardware_topology.*`, `backend`, `num_devices`, `num_hosts`.

Hooks that log nothing in the baseline: `mixture_weight_logging_hook` (the train set is a `NamedLmDataset`, not a
`MixtureDataset`; L/main/train_lm.py:400-403; L/data/text/datasets.py:801), `iris_status_reporter`, eval harness,
labeled eval and entropy logging (`eval_harness = null`, `labeled_eval = null`, `log_entropy = false`, logged).

### 7. Contract with `scripts/della/objective_compare.py`

The script reads W&B only. It reads `run.summary` (the last logged value of each key) and `run.config`.

1. Run path: `reself/marin-della/muonh-qwen3-{size}-della4x{device}{suffix}` (objective_compare.py:135). `size`
   defaults to `130m`, `device` to `h100`. The `della4x` part is fixed. A new-stack run must use this pattern with a
   new suffix, e.g. `-nm`, `-nm-s1`, in entity `reself`, project `marin-della` (launcher:774-776; sbatch line 26). An
   8-GPU run is visible to the script only under a `della4x` name; record the real GPU count in the config and tags.
2. Completion filters (objective_compare.py:52-74):
   - `run.config["trainer"]["num_train_steps"]` exists and is an `int` (lines 62-64). The new stack logs a nested
     config `{"trainer": {"num_train_steps": N, ...}}`.
   - `run.summary["_step"] == N - 1` (lines 61, 65-66). So the final eval and the final train metrics are logged at
     step `N - 1`, and nothing is ever logged at a step >= N.
   - `run.state == "finished"` (lines 67-68): `wandb.finish()` with exit code 0 after the last segment only.
     Resumed segments reuse the run id with `resume="allow"` (L/tracker/wandb.py:347) and never log a step below the
     last logged step (W&B drops it, L/tracker/wandb.py:136-157).
   - All admitted runs of pool and cell share one `num_train_steps` (lines 144-146).
   - The pool needs at least 2 admitted runs and the cell at least 1 (line 147).
3. Keys read:
   - `eval/paloma/macro_loss` (line 34)
   - `eval/macro_bpb` (line 35; top level, not under `eval/paloma`)
   - `eval/paloma/c4_en-marin-tokenizer/bpb` (line 36)
   - every key that matches `^eval/paloma/(.+)-marin-tokenizer/loss$` (lines 37-49): exactly the 16 domain loss keys
     of §4.2. The domain list is the union over all runs, and a domain missing from any run is skipped as incomplete
     (lines 163-171). The new stack must emit all 16 and no extra key that matches this pattern.
4. The current 130m pool: `--pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3`
   (docs/della/sampled-softmax.md:90). Example readout of four new-stack runs:
   `.venv/bin/python scripts/della/objective_compare.py --pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3 --cell=-nm,-nm-s1,-nm-s2,-nm-s3 --name=marincpp`
   (on a vis node, `--flag=value` form).

Optional exact bpb (a need of the Tokenizer Size session, docs/survey/needs_from_sessions.md): log it under a prefix
that matches none of the patterns above, e.g. `eval_exact/paloma/<subset>/bpb` and `eval_exact/macro_bpb`, computed
as `loss_k * T_k / (ln 2 * B_exact_k)` with an exact byte table (§4.3). The levanter-estimator keys stay as they are.

### 8. Implementation recipe and tests

Recipe for one eval at logged step `s` (after `s + 1` updates):

1. Load the 16 Paloma caches of §2.1, in the order of §2.3 (reading A). Cut each into `floor(n / 4096)` windows of 4096
   tokens; drop the tail. Concatenate the windows of all domains; give each window its domain index.
2. Split the 14,529 windows into 114 batches of 128 in order; pad the last batch with 63 zero rows (weight 0, no tag).
3. Per batch: bf16 forward of the raw parameters with document-blocked causal attention (segments start after each
   EOS); float32 per-position CE in nats; weight 1 at positions 0..4094 and 0 at 4095; bytes of the target ids from
   the Levanter byte table (§4.3).
4. Accumulate the four running means of §4.1 in float32, in batch order, with the `RunningMean` update rule.
5. Emit the 42 keys of §4.2 at step `s`. Emit `eval/loading_time` and `eval/total_time` as seconds.

Tests:

1. Byte table. Build the Levanter byte table from the Marin tokenizer snapshot
   (`cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2...`) and compare all 128,256
   entries with `levanter.utils.hf_utils.byte_length_of_token`. Required: identical.
2. Estimator and order. On one fixed levanter checkpoint of the baseline config, dump the per-position losses of the
   whole eval set once. Feed them to the new aggregation under reading A and reading B. The reading whose 16
   `bpb_k`, `eval/bpb` and `eval/paloma/bpb` match levanter's own eval of that checkpoint to float32 rounding (about
   1e-6) is the run order. The loss keys must match under both readings.
3. End to end. On the same checkpoint, run the new stack's eval. Proposed tolerance: each `loss_k` within 2e-4 nats
   of levanter's value (bf16 kernels differ). The run-to-run sd of the final c4_en bpb is 0.0004-0.0005
   (docs/della/objective-hillclimb.md:58), so eval error must stay well below it.
4. Logging contract. Run a short new-stack job with `N` small (e.g. 25 steps, `steps_per_eval` 10) into a scratch W&B
   project. Required: eval at steps 10, 20 and `N - 1` = 24; `train/loss` from step 2; norms at 10 and 20;
   `summary._step = 24`; state `finished`; `config.trainer.num_train_steps = 25`; `objective_compare.collect()`
   admits it.

### 9. Open points

1. Domain order (§2.3): the code fixes reading A (in-process config, byte-sorted insertion order); test 2 is a confirmation only.
2. The sign of the macro_bpb bias: this section measures logged = exact - 0.0004..0.0006; the survey says +0.004.
3. The eval tolerance in test 3 is a proposal. The owner sets it.
4. A parity run on 8 GPUs at 520m/1_2b must force the eval batch to 128 (per device 16), or its bpb keys differ from
   the 4-GPU baseline by the estimator effect.


---

## Part 5. Numerics (`numerics.md`: Numerics, precision and kernels)

Stage-1 spec section. It covers the dtype of every tensor and every rounding point in one training step of the plain
`muonh_qwen3` baseline. It also covers the attention and cross-entropy kernels, sharding, microbatching,
rematerialization, XLA/JAX flags and the sources of nondeterminism. Each item is classified as **M** (math: must
match), **P** (precision: match it for the parity run) or **S** (speed only: free to change). See §1.

Model structure is in `model.md`. Optimizer math is in `optim.md`. Parity gates are in `targets.md`. This section
restates their content only where it concerns numerics.

### 0. Sources and notation

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
| `NC` | `$ROOT/project/marin.cpp/cache` |
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

### 1. Classification and summary

* **M (math):** the item changes the function that is computed. marincpp must reproduce it exactly.
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

### 2. Mixed-precision policy

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

#### 2.1 Ops in f32

* RMSNorm statistics and the normalized value, before its cast (§3.2). In the backward, every step after the bf16
  product `g * dy`: the rsqrt-derivative terms, the division by `n` and the sum, until `dx` is rounded to bf16 (§3.2).
* RoPE angles, cos and sin, before their cast (§3.3).
* Attention: the row sum of `P` (accumulated in f32, then rounded to bf16); the storage of `L`. `L` itself is computed
  in bf16 (bf16 `log`, bf16 add) and then upcast to f32, so its precision is bf16 (§4.3). In the backward:
  `P = exp(f32(S) - L)`, `dS = P * f32(dP - D)`, the `dV` GEMM, and the f32 sums of `D` and of the GQA group (§4.4).
* Cross-entropy: logits, logsumexp and loss in the forward. Probabilities and dlogits in the backward, before their
  cast (§5).
* Loss reduction, microbatch accumulation, the optimizer, and the casts of gradients back to f32.

#### 2.2 Ops in bf16

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

### 3. Forward pass: dtype of each op

`model.md` §2 gives the order of operations. This section adds the dtypes, read from the forward layer body
`J130:2530-2794` (`jaxpr1`) and the main step `J130:30-470`.

#### 3.1 Embedding and linear layers

* Embedding: `gather` on the bf16 table, then `* 1.0:bf16` (`J130:62-70`; `HX/nn/embedding.py:74-81`).
  Out-of-range ids are mapped with `id < 0 -> id + V` and `mode=PROMISE_IN_BOUNDS`.
* Linear: `dot_general(x_bf16, (W * 1.0)_bf16, preferred_element_type=bf16)` (`J130:2569-2599`;
  `HX/nn/linear.py:83-100`; `HX/quantization.py:150-170`). `precision` is unset: no dot in `J130` has a `precision=`
  attribute. cuBLAS accumulates in f32 and rounds the output to bf16. No biases.

#### 3.2 RMSNorm

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

#### 3.3 RoPE

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
  angle at pos 4095 by up to about 2.4e-4 rad and can also flip entries. So marincpp either uses accurate f32
  cos/sin on the f32 angle `f32(f32(pos) * inv_freq)`, with `inv_freq` built in f32 in the order above, or copies the
  exact f32 `inv_freq` and the bf16 cos/sin tables `[4096, hd]` from the reference (O4).
* The rotation runs in bf16: `q*cos + concat(-q[hd/2:], q[:hd/2])*sin` (`J130:2639-2668`; `rotary.py:17-22, 45`).
  Each multiply and each add rounds to bf16.
* `pos = arange(S)` for every row (`attention.py:1441-1442`).

#### 3.4 Attention scale

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

#### 3.5 MLP, post-norms, residual

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

#### 3.6 Loss reduction

* `loss_tok` is the f32 output of the CE kernel (§5). The weights are `w[s] = 1` for `s < S-1` and `w[S-1] = 0`
  (`LV/data/text/examples.py:141-149, 200-210`). There is no `ignore_id`, and no other position is masked.
* `loss = sum(loss_tok * w) / sum(w)` over the whole microbatch `[B,S]`, in f32 (`HX/nn/loss.py:132-138`;
  `J130:376-381`). The denominator is `B * 4095`. It is 524160 at every step of every size, because each microbatch
  holds 128 sequences, with one exception: updates 0-85 of the 520m baseline ran one microbatch of 256 sequences
  (`PDP = 64`, §6.3; `NC/wandb/config/run-20260914_063908-muonh-qwen3-520m-della4xh100.config.yaml:498`), so the
  denominator there was 1048320 and `g_loss = 1/1048320`. The mean is the same.
* With two microbatches, `train/loss = (loss_1 + loss_2)/2` and `grad = (g_1 + g_2)/2`, in f32
  (`LV/grad_accum.py:155-162`). Each sequence carries the same weight sum (4095), so this equals the global mean.

### 4. Attention kernel

#### 4.1 Selection

* `attn_backend=AttentionBackend.JAX_FLASH` (`L:661`; logged `JAX_FLASH`). The GPU default is NVTE
  (`attention.py:84-89`), and Della has no Transformer Engine.
* JAX_FLASH has no `case` in the dispatch, so the call falls through to `flash_attention` (`attention.py:276-300`):
  pure JAX, `lax.while_loop`s and an `equinox.filter_custom_vjp` (`FA:114-145, 448-449`). No cuDNN, Triton or Pallas
  kernel is involved.
* `upcast_attn=False`, so `attention_dtype = x.dtype = bf16` (`attention.py:1357`).
* Dropout, bias, soft cap and sinks are all off.

#### 4.2 Blocks and SEQ_LEN constraint

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

#### 4.3 Forward numerics (`FA:147-299`; `J130:4396-4538`, `jaxpr6`)

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

#### 4.4 Backward numerics (`FA:302-445`; `J130:4964-5195`, `jaxpr11`)

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

#### 4.5 Class

* M: the mask, the scale constant, and the absolute RoPE positions.
* S: the kernel itself, the block sizes, and causal or varlen tile skipping.
* P: the bf16 softmax statistics and the bf16 accumulators.
* A FlashAttention-3 kernel (f32 softmax, f32 accumulators) raises precision. It is allowed after the A/B test of §1;
  it is the natural first deviation.

### 5. Fused cross-entropy

#### 5.1 Call chain and inputs

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

#### 5.2 Path taken by the plain baseline: `batched_xla`, H100 full-vocab B-tiled

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

#### 5.3 The other path: `PDP = 1` (`B_dev = 4096 < 8192`)

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

#### 5.4 Class

* M: f32 logits and lse in the forward, the label shift and mask, the global-mean reduction, and the absence of a
  z-loss and a soft cap.
* S: the implementation and the 8192-row tile, and whether the logits are materialized or streamed.
* P: the bf16 logits in the backward, bf16 dlogits, the bf16 `grad_w` and its bf16 psum, and the bf16 `grad_x`.
* For the parity run, round the recomputed backward logits and dlogits to bf16. The cost is negligible.

### 6. Sharding, data parallelism, microbatching

#### 6.1 Mesh

* There is one process with 4 local devices. JAX distributed is not initialized (`SL:37`). The client is the Fray
  LocalClient, which runs the trainer in a thread of the same process (`lib/fray/src/fray/local_backend.py:110-114`).
* Axes: `data=4`, `replica=1`, `model=1`, `replica_dcn=1` (`L:802-811`; `LV/utils/mesh.py:16-19, 73-129`; `H130:2`).
  The partitioner is Shardy (`sdy.mesh` in `H130`).
* Compute mapping: `batch`, `token` and `token_repeat` map to `(replica_dcn, replica, data)`, which is 4-way. `mlp` and
  `heads` map to `model` (size 1) (`mesh.py:42-55`; `WB .config.trainer.mesh`).
* Param mapping: `embed -> data` (the default, `mesh.py:40`). `mlp` and `heads` map to `model`.

#### 6.2 Parameter and optimizer-state placement (FSDP / ZeRO-3)

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

#### 6.3 Microbatching

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

### 7. Rematerialization and layer scan

* `scan_layers=True`: the layers run under `Stacked` and `lax.scan` (`LV/models/llama.py:392-404`).
* `gradient_checkpointing=True` maps to `ScanCheckpointPolicy(simple=True)`, which is
  `eqx.filter_checkpoint(layer, prevent_cse=False)` with no policy (`HX/_src/scan.py:145-185`). `J130:499-510` shows
  `remat2[policy=None, prevent_cse=False]`.
* Each layer saves only its input carry, `bf16 [B,S,D]`. In the backward it recomputes the whole layer, including the
  flash forward.
* Class S.

### 8. JAX and XLA flags, environment

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

### 9. Determinism

#### 9.1 Sources

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

#### 9.2 Measured floor

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
* For marincpp, a deterministic embedding backward (sorted segment-sum, f32 accumulator) is S and recommended. It
  makes A/B tests of single P items possible at n = 1 per seed for debugging. Gates still follow `targets.md`.

### 10. Open items (need a GPU node; nothing was submitted)

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
  reference bf16 (§4.5). This is the first deviation marincpp will want.
* **O4. RoPE tables.** On a GPU node, run the `DefaultRotaryEmbeddings` code path once at hd 64 and hd 128 and save the
  f32 `inv_freq` and the bf16 cos/sin tables `[4096, hd]` (§3.3). marincpp either loads them or tests its own tables
  against them bit for bit.

### 11. Findings that belong to other sections (verified while reading)

* QK-norm is on although `use_qk_norm: false` is logged (`LV/models/qwen.py:374-376`); see `model.md` §2.1.
* `lm_head` and the embedding are drawn with the same key `k_emb` and the same shape `[V, D]`
  (`LV/models/qwen.py:389-395`). The scales are `1/sqrt(D)` for the head (`HX/nn/mup.py:63-64`) and `1/D` for the
  embedding (`HX/nn/mup.py:157-158`; `HX/nn/embedding.py:65-67`). So at init, `lm_head = sqrt(D) * embedding`, to f32
  rounding.
* `weight_decay=0.1` is logged but never read (`LV/optim/muonh.py:54-104`); see `optim.md` §7.
* Each optimizer group is clipped by its own global norm (`J130:598-640`); see `optim.md` §3.1.


---

## Part 6. Targets (`targets.md`: Reference results and parity tolerances)

Stage-1 spec, section "targets". Written 2026-10-24.

**Summary.** The 130m reference pool has 8 finished levanter H100 runs: c4_en bpb 1.16326 (sd 0.00033), Paloma macro_loss 4.18646 (sd 0.00577). The 300m pool has 5 runs: 1.05641 (sd 0.00029) and 3.81386 (sd 0.00574). 520m and 1_2b have one H100 run each. The four TPU originals ran on **v5litepod-64 (TPU v5e)**, not v5p. marincpp passes stage 1 when it clears four gates: (a) step-0 loss and gradients, (b) the first 200 steps, (c) final metrics, (d) throughput.

All numbers below come from W&B (cached under `cache/wandb/targets/`). Each table is generated by a script in `cache/wandb/_tools/`, and no value was transcribed by hand. Where the ledgers quote a different value, the W&B value wins, and section 6 lists the conflicts.

---

### 1. Sources and reproduction

| item | path |
|---|---|
| fetch (vis node, cached; a finished run is never re-fetched) | `cache/wandb/_tools/targets_fetch.py` → `cache/wandb/targets/{della,orig}__<run>.{config,summary,meta}.json`, `.hist.json.gz` |
| per-run and pool statistics | `cache/wandb/_tools/targets_analyze.py` → `targets/_analysis_runs.json`, `targets/_analysis_pools.json`, `_tools/targets_analyze.out` |
| same-seed / cross-seed / TPU-vs-H100 curve agreement | `cache/wandb/_tools/targets_pairs.py` → `_tools/targets_pairs.out` |
| tables in this file | `cache/wandb/_tools/targets_tables.py` → `_tools/targets_tables.md` |
| H100 runs | W&B `reself/marin-della/<run id>` |
| TPU originals | W&B `marin-community/marin/<run id>`. Hardware and token counts come from `experiments/speedrun/muonh_qwen3_scaling/<size>/speedrun_results.json` at marin commit `2d6290a99c` ("Hyperball speedrun (#1774)"). The recipe is `experiments/speedrun/muonh_qwen3_scaling/muonh_sweep.py` at the same commit. |
| ledgers | `project/marin/logs/kaiyue_muonh_qwen3_h100x4.yaml`, `logs/objective_hillclimb_qwen3_h100x4.yaml`, `docs/della/objective-hillclimb.md`, `~/docs/projects/marin-moe-dense-2026-09-18.md` |

**Plain baseline: definition used here.** A run is plain when its id matches `^muonh-qwen3-(130m|300m|520m|1_2b)-della4xh100(-ema0.999)?(-s<k>)?$`. The id is built in `experiments/references/della_muonh_qwen3_scaling.py:543`. No VARIANT, PLS or schedule knobs are set. Three knobs produce the suffixes:

* `SEED` (`:184`) feeds `trainer.seed` (`:812`). It changes only the init and training keys: `data_key, loader_key, model_key, training_key = split(PRNGKey(seed), 4)` (`levanter/main/train_lm.py:217-218`). The data order is fixed by `data_seed=42` (`della_muonh_qwen3_scaling.py:820`, which overrides `data_key` at `train_lm.py:220-222`).
* `EMA_BETA=0.999` (`:187`, `:813`) adds an EMA copy of the weights. That copy is evaluated under `eval/ema/...`. The pool uses only the raw-weight keys `eval/paloma/...`.
* `RUN_TAG` (`:191`) appends a suffix to the run id and changes nothing else. The four `-det1/-det2` determinism probes were queued and then never ran (no sacct record).

The listing of all 891 runs in `reself/marin-della` (`targets/_marin_della_muonh_runs.json`) contains exactly 15 plain baselines. All 15 are `finished` at step `num_train_steps − 1`.

**Metric keys.** On the H100 runs the c4 key is `eval/paloma/c4_en-marin-tokenizer/bpb`. On the TPU originals the same quantity is `eval/paloma/c4_en/bpb`. The suffix comes from the Paloma cache-naming fix (`kaiyue_muonh_qwen3_h100x4.yaml:6`). `eval/paloma/macro_loss` is the unweighted mean of the 16 per-domain losses. `eval/paloma/macro_bpb` equals `eval/macro_bpb` to within 2e-7 on every H100 run.

**Eval cadence.** Full Paloma, with no batch cap (`max_eval_batches=None`, `:790`), every 1000 steps (`:445`, `:789`) and at the final step `num_train_steps − 1`. The TPU originals also logged an eval at step 0. The H100 runs did not. The TPU step-0 eval is **not** an eval at init: see "Step semantics" below.

**Train-loss logging.** On H100, `train/loss` first appears at `_step` 2: `throughput/total_tokens` = 3 × 524288 there, so step 2 is the third optimizer step. Steps 0 and 1 are not logged. The TPU originals log from step 0. The cause is code, not platform: the current hook guard `info.step > 1 and info.step % every == 0` (`levanter/trainer.py:153`, `:162`, `:170`) came with commit `8305d2d72f` ("Skip periodic trainer hooks at step 0", 2026-05-26). It removes steps 0 and 1 from every hook, the `every=1` hooks included (`log_step_info`, checkpointer, progress; `trainer.py:636-654`), although the commit message says `every=1` hooks still fire. The TPU-era levanter checked only `info.step % every == 0` (`git show 14c1488198:lib/levanter/src/levanter/trainer.py`, line 126; the copy moved into marin on 2025-11-04, after the TPU runs).

**Step semantics** (`levanter/callbacks/_core.py:76`; `trainer.py:557-565`, `:625-626`). `TrainerState.step` counts completed updates. After update k (k = 0, 1, ...), `state.step = k + 1` and the hooks receive `info.step = state.step − 1 = k` with the post-update state.

* `train/loss` logged at step k is the loss of batch k under the parameters after k updates (`StepInfo.loss` is the loss computed inside update k, `trainer.py:548,557`; the `log_step_info` hook logs it as `train/loss` with `step=info.step`, `callbacks/_metrics.py:70-82`, registered at `trainer.py:640`).
* A Paloma eval logged at step s (1000, 2000, ...) uses the parameters after **s + 1** updates.
* The final eval, logged at N − 1, uses the parameters after all N updates (`force=True`, `trainer.py:626`).
* The TPU eval logged at step 0 ran after update 0, so it uses the parameters after 1 update. At 130m (full LR at update 0) its c4_en loss is 11.6067, between the init `train/loss` on batch 0 (12.2796) and the step-1 `train/loss` (11.5683). At 300m, 520m and 1_2b lr(0) = 0 (section 4, a7), so that eval matches the init up to rounding (300m: c4_en loss 12.2521, step-0 `train/loss` 12.2454).
* The cached W&B API history repeats some rows: eval and train rows at `_step` 5000 and 10000 (300m), 5000 (520m), and 5000, 10000, 15000, 20000 (1_2b) appear twice with identical values (`della__muonh-qwen3-<size>-della4xh100.hist.json.gz`). They all fall on multiples of 5000; `eval.md` §1 traces them to API pagination (the local `.wandb` files hold each eval once). Every comparison script must dedupe by `_step`.

---

### 2. Every finished plain-baseline run, and the TPU originals

Step time, tok/s and MFU are p50 over steps ≥ 20 (`throughput/duration`, `throughput/tokens_per_second`, `throughput/mfu`). All H100 runs used 4× H100 80GB SXM on one node, jax 0.11.1, with the launcher run by `scripts/della/muonh_qwen3_h100x4.sbatch`. `git_commit` is the logged config value.

| size | run id (reself/marin-della) | seed | EMA | git_commit | created (UTC) | steps | c4_en bpb | c4_en loss | macro_loss | macro_bpb | train/loss @100 | @500 | @1000 | @2000 | @end | step s p50 | tok/s p50 | MFU % p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130m | `muonh-qwen3-130m-della4xh100` | 0 | no | `59ac0d647d` | 2026-09-14T08:18 | 4959 | 1.16358 | 3.78121 | 4.17983 | 1.51130 | 5.2118 | 3.9922 | 3.8572 | 3.6323 | 3.2352 | 0.6305 | 831,495 | 14.36 |
| 130m | `muonh-qwen3-130m-della4xh100-s1` | 1 | no | `a3d7b7a0dd` | 2026-09-19T12:40 | 4959 | 1.16291 | 3.77902 | 4.18282 | 1.51264 | 5.1637 | 3.9943 | 3.8750 | 3.6269 | 3.2310 | 0.6311 | 830,718 | 14.35 |
| 130m | `muonh-qwen3-130m-della4xh100-s2` | 2 | no | `a3d7b7a0dd` | 2026-09-19T13:40 | 4959 | 1.16278 | 3.77862 | 4.18879 | 1.51447 | 5.1630 | 3.9783 | 3.8647 | 3.6183 | 3.2316 | 0.6297 | 832,589 | 14.38 |
| 130m | `muonh-qwen3-130m-della4xh100-s3` | 3 | no | `2dc443959f` | 2026-09-19T14:02 | 4959 | 1.16338 | 3.78055 | 4.18798 | 1.51405 | 5.1740 | 3.9854 | 3.8532 | 3.6275 | 3.2322 | 0.6315 | 830,272 | 14.34 |
| 130m | `muonh-qwen3-130m-della4xh100-ema0.999` | 0 | 0.999 | `9c0aa7ab2d` | 2026-09-19T21:31 | 4959 | 1.16352 | 3.78103 | 4.18998 | 1.51553 | 5.2150 | 3.9970 | 3.8485 | 3.6328 | 3.2348 | 0.6321 | 829,475 | 14.33 |
| 130m | `muonh-qwen3-130m-della4xh100-ema0.999-s1` | 1 | 0.999 | `9c0aa7ab2d` | 2026-09-19T21:40 | 4959 | 1.16312 | 3.77971 | 4.17901 | 1.51097 | 5.1523 | 3.9869 | 3.8649 | 3.6234 | 3.2318 | 0.6315 | 830,291 | 14.34 |
| 130m | `muonh-qwen3-130m-della4xh100-ema0.999-s2` | 2 | 0.999 | `9c0aa7ab2d` | 2026-09-19T21:41 | 4959 | 1.16370 | 3.78158 | 4.19647 | 1.51745 | 5.1567 | 3.9872 | 3.8599 | 3.6247 | 3.2337 | 0.6324 | 829,074 | 14.32 |
| 130m | `muonh-qwen3-130m-della4xh100-ema0.999-s3` | 3 | 0.999 | `7fcca6f4d7` | 2026-09-19T22:04 | 4959 | 1.16313 | 3.77973 | 4.18682 | 1.51390 | 5.1747 | 3.9886 | 3.8716 | 3.6315 | 3.2341 | 0.6337 | 827,349 | 14.29 |
| 130m | TPU orig `qwen3_130m_muonh_4096_lr_0.02_adam_lr_0.008-354a89` | 0 | no | `02a83f0c92` | 2025-10-12T03:32 | 4959 | 1.16354 | 3.78069 | 4.18259 | 1.51467 | 5.2141 | 3.9999 | 3.8549 | 3.6362 | 3.2350 | 0.1296 | 4,046,239 | 21.94 |
| 300m | `muonh-qwen3-300m-della4xh100` | 0 | no | `28fd599610` | 2026-09-14T09:02 | 11444 | 1.05626 | 3.43249 | 3.80874 | 1.38226 | 7.8321 | 4.3786 | 3.7189 | 3.4575 | 2.8635 | 1.4440 | 363,078 | 15.27 |
| 300m | `muonh-qwen3-300m-della4xh100-s1` | 1 | no | `aff68ec496` | 2026-09-19T23:44 | 11444 | 1.05634 | 3.43273 | 3.80915 | 1.38179 | 7.8045 | 4.3751 | 3.7156 | 3.4497 | 2.8637 | 1.4432 | 363,272 | 15.28 |
| 300m | `muonh-qwen3-300m-della4xh100-s2` | 2 | no | `5bb5cace30` | 2026-09-20T15:34 | 11444 | 1.05618 | 3.43222 | 3.82246 | 1.38704 | 7.8295 | 4.3728 | 3.7181 | 3.4467 | 2.8630 | 1.4453 | 362,765 | 15.26 |
| 300m | `muonh-qwen3-300m-della4xh100-s3` | 3 | no | `0cb7e60fca` | 2026-09-21T08:17 | 11444 | 1.05691 | 3.43460 | 3.81240 | 1.38361 | 7.8362 | 4.3740 | 3.7112 | 3.4449 | 2.8614 | 1.4428 | 363,370 | 15.29 |
| 300m | `muonh-qwen3-300m-della4xh100-ema0.999` | 0 | 0.999 | `c58428f866` | 2026-10-22T19:42 | 11444 | 1.05634 | 3.43273 | 3.81656 | 1.38537 | 7.8321 | 4.3744 | 3.7169 | 3.4442 | 2.8647 | 1.4432 | 363,292 | 15.28 |
| 300m | TPU orig `qwen3_300m_muonh_4096_lr_0.01-f5ae20` | 0 | no | `02a83f0c92` | 2025-10-12T18:26 | 11444 | 1.05602 | 3.43136 | 3.80531 | 1.38251 | 7.8281 | 4.3760 | 3.7181 | 3.4454 | 2.8629 | 0.3654 | 1,434,928 | 18.95 |
| 520m | `muonh-qwen3-520m-della4xh100` | 0 | no | `ac6af86874` | 2026-09-14T10:39 | 9918 | 0.98830 | 3.21164 | 3.57255 | 1.29952 | 7.3745 | 4.0485 | 3.3841 | 3.1957 | 2.6757 | 6.7415 | 155,541 | 15.91 |
| 520m | TPU orig `qwen3_520m_muonh_4096_lr_0.01-cb97ec` | 0 | no | `02a83f0c92` | 2025-10-12T18:27 | 9918 | 0.98824 | 3.21101 | 3.56398 | 1.29668 | 7.3716 | 4.0486 | 3.3830 | 3.1912 | 2.6755 | 1.4001 | 748,915 | 24.05 |
| 1_2b | `muonh-qwen3-1_2b-della4xh100` | 0 | no | `cead09a259` | 2026-09-14T10:47 | 22888 | 0.92763 | 3.01449 | 3.36087 | 1.22578 | 6.5599 | 3.7400 | 3.2824 | 3.1134 | 2.4086 | 8.2209 | 127,550 | 27.82 |
| 1_2b | TPU orig `qwen3_1_2b_muonh_4096_low_lr-f47219` | 0 | no | `02a83f0c92` | 2025-10-12T08:45 | 22888 | 0.92731 | 3.01302 | 3.36047 | 1.22588 | 6.5612 | 3.7394 | 3.2825 | 3.1103 | 2.4078 | 1.6318 | 642,584 | 43.99 |

**Provenance notes**

* **Git commits.** The commit differs from run to run because the launcher file kept gaining variant code. The plain-baseline path did not change: `kaiyue_muonh_qwen3_h100x4.yaml:5` has the drift audit, and every pool run sets no variant knob. The 300m, 520m and 1_2b runs resumed across segments, and the W&B config holds the commit of a later segment (for example, 520m logs `ac6af86874` from 2026-09-15, although the run started 2026-09-14).
* **Slurm jobs.**
  * 130m: s0 `13846675`; s1 `14135794`; s2 `14140936`; s3 `14140937`; ema s0–s3 `14148473–76`. Each ran in one 2 h segment of about 57–61 min.
  * 300m: s0 `13852135–37`; s1 `14158448–50`; s2 `14178311–14`; s3 `14178315–17`.
  * 300m-ema: wq job `w20`, plan `logs/plans/ema-128.txt`. It ran inside a pilot host on GPUs 0–3 of an 8-GPU node, with a `-rep8` run on GPUs 4–7. Its step time (1.4432 s) is the same as the other 300m runs, so sharing the node did not slow it.
  * 520m: `13852139–47`, `13855294–95` (11 segments). The first segment OOMed at 64 seqs/GPU, so the run continued with `PER_DEVICE_PARALLELISM=32` from its `step-85` temporary checkpoint (`state.step` 86; updates 0–85 used one microbatch of 64 seqs/GPU, `optim.md` §8).
  * 1_2b: array `13855503_1..27`.
* **Microbatching** (`della_muonh_qwen3_scaling.py:435`, `:787`). 130m and 300m run 32 seqs/GPU in one microbatch. 520m and 1_2b run 64 seqs/GPU as 2 microbatches of 32. The math is identical.
* **TPU hardware.** All four TPU originals ran on `tpu_type: "v5litepod-64"` with `slice_count: 1` (`speedrun_results.json` @ `2d6290a99c`). That is 64 TPU v5e chips at 197 TFLOP/s bf16 each (`lib/fray/src/fray/device_flops.py:193-194`). The logged MFU confirms it: implied per-chip peak = tok/s × FLOP/token ÷ MFU ÷ 64 = 1.97e14, which matches v5e and not v5p (4.59e14). The ledgers' "v5p" label is wrong. End-to-end `training_time` (s): 130m 1015.9, 300m 4972.1, 520m 14494.6, 1_2b 38739.9.

---

### 3. Pools per size

Per-run sd uses ddof = 1, and se = sd/√n. "TPU orig" is the single TPU run.

| size | n | metric | mean | sd | se | min | max | TPU orig | mean − orig |
|---|---|---|---|---|---|---|---|---|---|
| 130m | 8 | c4_en bpb | 1.16326 | 0.00033 | 0.00012 | 1.16278 | 1.16370 | 1.16354 | −0.00027 |
| 130m | 8 | c4_en loss | 3.78018 | 0.00107 | 0.00038 | 3.77862 | 3.78158 | 3.78069 | −0.00050 |
| 130m | 8 | macro_loss | 4.18646 | 0.00577 | 0.00204 | 4.17901 | 4.19647 | 4.18259 | +0.00387 |
| 130m | 8 | macro_bpb | 1.51379 | 0.00215 | 0.00076 | 1.51097 | 1.51745 | 1.51467 | −0.00088 |
| 130m | 8 | train/loss end | 3.23304 | 0.00158 | 0.00056 | 3.23101 | 3.23515 | 3.23497 | −0.00193 |
| 300m | 5 | c4_en bpb | 1.05641 | 0.00029 | 0.00013 | 1.05618 | 1.05691 | 1.05602 | +0.00038 |
| 300m | 5 | c4_en loss | 3.43296 | 0.00094 | 0.00042 | 3.43222 | 3.43460 | 3.43136 | +0.00160 |
| 300m | 5 | macro_loss | 3.81386 | 0.00574 | 0.00257 | 3.80874 | 3.82246 | 3.80531 | +0.00856 |
| 300m | 5 | macro_bpb | 1.38401 | 0.00219 | 0.00098 | 1.38179 | 1.38704 | 1.38251 | +0.00150 |
| 300m | 5 | train/loss end | 2.86326 | 0.00120 | 0.00053 | 2.86141 | 2.86465 | 2.86291 | +0.00034 |
| 520m | 1 | c4_en bpb | 0.98830 | – | – | – | – | 0.98824 | +0.00005 |
| 520m | 1 | macro_loss | 3.57255 | – | – | – | – | 3.56398 | +0.00857 |
| 520m | 1 | macro_bpb | 1.29952 | – | – | – | – | 1.29668 | +0.00283 |
| 520m | 1 | train/loss end | 2.67568 | – | – | – | – | 2.67550 | +0.00018 |
| 1_2b | 1 | c4_en bpb | 0.92763 | – | – | – | – | 0.92731 | +0.00032 |
| 1_2b | 1 | macro_loss | 3.36087 | – | – | – | – | 3.36047 | +0.00040 |
| 1_2b | 1 | macro_bpb | 1.22578 | – | – | – | – | 1.22588 | −0.00010 |
| 1_2b | 1 | train/loss end | 2.40860 | – | – | – | – | 2.40785 | +0.00076 |

The 300m pool without the 2026-10-22 EMA run (n = 4, the pool used by the objective-hillclimb ledger, `objective_hillclimb_qwen3_h100x4.yaml:1914-1931`) gives c4_en 1.05642 (sd 0.00033) and macro_loss 3.81319 (sd 0.00639).

**What the pools show**

1. **H100 vs TPU.** Every TPU c4_en value sits inside the H100 pool range (130m, 300m) or within +0.0004 bpb of the single H100 run (520m, 1_2b). On macro_loss the TPU run lies at or below the H100 pool minimum at 300m (3.80531 vs min 3.80874) and 0.0086 below the single 520m H100 run. That gap is about 1.5 pool sd. With TPU n = 1 it is not a resolved TPU–GPU offset.
2. **Seed vs trajectory noise.** On the H100 runs, a different init seed and a same-seed rerun give spreads of the same size. Same-seed pairs (raw vs EMA copy) differ on c4_en by −0.00006, +0.00021, +0.00092, −0.00025 (`objective_hillclimb_qwen3_h100x4.yaml:881-899`). The leading **hypothesis** for the nondeterminism is one op: the atomic bf16 scatter-add in the embedding backward (`haliax/nn/embedding.py:80`; argued from code in `docs/della/objective-hillclimb.md:70`). It is not established. The only same-seed evidence (raw vs EMA) "cannot distinguish whether EMA changed the compiled program or the atomic add itself" (`objective-hillclimb.md:74`). The probes that would settle it (two byte-identical baselines, an `--xla_gpu_deterministic_ops` pair, `:74-76`) never ran (section 1, `RUN_TAG`). `numerics.md` §9.1 also lists XLA autotuning and the two compiled step programs as candidate sources. Either way, every run is one independent draw.

#### 3.1 Per-domain final values

The table gives pool mean and sd per domain. The TPU loss column is directly comparable. **The TPU bpb column is not comparable for every domain.** wikitext_103 has equal loss (3.6459 vs 3.6470) but bpb 1.1778 vs 1.1598, and m2d2_s2orc shows the same pattern. So the bytes-per-token normalisation of the TPU Paloma copy differs from the marin-tokenizer caches for some domains. Per-domain parity against TPU must use loss, not bpb.

**130m (n = 8)**

| domain | loss mean | loss sd | bpb mean | bpb sd | TPU loss | TPU bpb |
|---|---|---|---|---|---|---|
| 4chan | 3.5556 | 0.0123 | 1.5272 | 0.0053 | 3.5257 | 1.5143 |
| c4_100_domains | 3.7246 | 0.0010 | 1.1601 | 0.0003 | 3.7250 | 1.1603 |
| c4_en | 3.7802 | 0.0011 | 1.1633 | 0.0003 | 3.7807 | 1.1635 |
| dolma-v1_5 | 3.9539 | 0.0074 | 1.3323 | 0.0026 | 3.9543 | 1.3329 |
| dolma_100_programing_languages | 4.7235 | 0.0342 | 2.0133 | 0.0148 | 4.6857 | 1.9981 |
| dolma_100_subreddits | 4.4314 | 0.0114 | 1.4543 | 0.0037 | 4.4249 | 1.4523 |
| falcon-refinedweb | 3.9020 | 0.0019 | 1.2449 | 0.0006 | 3.9041 | 1.2458 |
| gab | 5.3466 | 0.0145 | 2.2032 | 0.0060 | 5.3466 | 2.2038 |
| m2d2_s2orc_unsplit | 3.4678 | 0.0031 | 1.1538 | 0.0010 | 3.4685 | 1.1570 |
| m2d2_wikipedia_unsplit | 3.4967 | 0.0046 | 1.1310 | 0.0015 | 3.4969 | 1.1312 |
| manosphere_meta_sep | 4.4199 | 0.0108 | 1.5097 | 0.0037 | 4.4182 | 1.5094 |
| mc4 | 3.5926 | 0.0051 | 1.2396 | 0.0018 | 3.5908 | 1.2393 |
| ptb | 4.3907 | 0.0231 | 1.3803 | 0.0073 | 4.3493 | 1.3674 |
| redpajama | 4.5151 | 0.0141 | 1.5634 | 0.0050 | 4.5109 | 1.5624 |
| twitterAAE_HELM_fixed | 6.0357 | 0.0204 | 2.9844 | 0.0101 | 6.0940 | 3.0193 |
| wikitext_103 | 3.6470 | 0.0103 | 1.1598 | 0.0033 | 3.6459 | 1.1778 |

**300m (n = 5)**

| domain | loss mean | loss sd | bpb mean | bpb sd | TPU loss | TPU bpb |
|---|---|---|---|---|---|---|
| 4chan | 3.3084 | 0.0252 | 1.4210 | 0.0108 | 3.3150 | 1.4239 |
| c4_100_domains | 3.3541 | 0.0009 | 1.0447 | 0.0003 | 3.3525 | 1.0443 |
| c4_en | 3.4330 | 0.0009 | 1.0564 | 0.0003 | 3.4314 | 1.0560 |
| dolma-v1_5 | 3.5432 | 0.0047 | 1.1935 | 0.0016 | 3.5399 | 1.1928 |
| dolma_100_programing_languages | 4.1316 | 0.0188 | 1.7612 | 0.0081 | 4.1285 | 1.7618 |
| dolma_100_subreddits | 4.1019 | 0.0031 | 1.3462 | 0.0010 | 4.0882 | 1.3418 |
| falcon-refinedweb | 3.5596 | 0.0011 | 1.1357 | 0.0004 | 3.5585 | 1.1355 |
| gab | 5.0621 | 0.0178 | 2.0859 | 0.0074 | 5.0331 | 2.0745 |
| m2d2_s2orc_unsplit | 3.1096 | 0.0035 | 1.0342 | 0.0012 | 3.1090 | 1.0367 |
| m2d2_wikipedia_unsplit | 3.1011 | 0.0038 | 1.0031 | 0.0012 | 3.1016 | 1.0033 |
| manosphere_meta_sep | 4.0769 | 0.0060 | 1.3925 | 0.0020 | 4.0712 | 1.3908 |
| mc4 | 3.2255 | 0.0031 | 1.1129 | 0.0011 | 3.2243 | 1.1128 |
| ptb | 3.8643 | 0.0317 | 1.2149 | 0.0100 | 3.8385 | 1.2067 |
| redpajama | 4.0229 | 0.0113 | 1.3930 | 0.0040 | 4.0157 | 1.3907 |
| twitterAAE_HELM_fixed | 5.9263 | 0.0352 | 2.9310 | 0.0174 | 5.8866 | 2.9173 |
| wikitext_103 | 3.2014 | 0.0086 | 1.0181 | 0.0027 | 3.1910 | 1.0309 |

**520m and 1_2b (n = 1 each): H100 loss / TPU loss**

| domain | 520m H100 | 520m TPU | 1_2b H100 | 1_2b TPU |
|---|---|---|---|---|
| 4chan | 3.2012 | 3.1710 | 3.1508 | 3.0992 |
| c4_100_domains | 3.1146 | 3.1140 | 2.9017 | 2.9008 |
| c4_en | 3.2116 | 3.2110 | 3.0145 | 3.0130 |
| dolma-v1_5 | 3.2860 | 3.2808 | 3.0581 | 3.0620 |
| dolma_100_programing_languages | 3.7581 | 3.7205 | 3.4434 | 3.4620 |
| dolma_100_subreddits | 3.8935 | 3.8924 | 3.7040 | 3.7003 |
| falcon-refinedweb | 3.3359 | 3.3356 | 3.1421 | 3.1402 |
| gab | 4.8463 | 4.8273 | 4.6368 | 4.6521 |
| m2d2_s2orc_unsplit | 2.8791 | 2.8795 | 2.6647 | 2.6665 |
| m2d2_wikipedia_unsplit | 2.8532 | 2.8479 | 2.6393 | 2.6359 |
| manosphere_meta_sep | 3.8620 | 3.8567 | 3.6617 | 3.6558 |
| mc4 | 2.9925 | 2.9913 | 2.7919 | 2.7929 |
| ptb | 3.5182 | 3.5145 | 3.2280 | 3.2588 |
| redpajama | 3.7026 | 3.6859 | 3.4089 | 3.4248 |
| twitterAAE_HELM_fixed | 5.7985 | 5.7971 | 5.6874 | 5.6737 |
| wikitext_103 | 2.9076 | 2.8982 | 2.6405 | 2.6295 |

#### 3.2 Eval trajectories, seed 0: H100 vs TPU (c4_en bpb / macro_loss)

The TPU step-0 row is the eval **after 1 optimizer update** (`state.step = 1`; section 1, "Step semantics"). At 130m it is not the init eval. At 300m, 520m and 1_2b it equals the init eval up to rounding, only because lr(0) = 0. It is not the reference for gate (a3); fixture 4 of section 4(a) is. Every other row at step s uses the parameters after s + 1 updates, and "final" after N updates.

| step | 130m H100 | 130m TPU | 300m H100 | 300m TPU | 520m H100 | 520m TPU |
|---|---|---|---|---|---|---|
| 0 | – | 3.57218 / 11.7600 | – | 3.77085 / 12.2693 | – | 3.77617 / 12.2507 |
| 1000 | 1.33791 / 4.8956 | 1.33643 / 4.8819 | 1.29749 / 4.7630 | 1.29889 / 4.7693 | 1.22102 / 4.4700 | 1.21960 / 4.4560 |
| 2000 | 1.28259 / 4.6474 | 1.28360 / 4.6608 | 1.23563 / 4.4989 | 1.22882 / 4.4720 | 1.14894 / 4.2025 | 1.14891 / 4.1861 |
| 3000 | 1.23496 / 4.4567 | 1.23617 / 4.4583 | 1.20370 / 4.3798 | 1.20319 / 4.3751 | 1.12218 / 4.0920 | 1.12500 / 4.1038 |
| 4000 | 1.19746 / 4.3076 | 1.19626 / 4.3067 | 1.18388 / 4.2759 | 1.18519 / 4.2911 | 1.10152 / 4.0139 | 1.10053 / 4.0017 |
| 5000 | – | – | 1.16171 / 4.2147 | 1.16276 / 4.2194 | 1.07732 / 3.9037 | 1.07829 / 3.9114 |
| 6000 | – | – | 1.14112 / 4.1296 | 1.14090 / 4.1216 | 1.05369 / 3.8285 | 1.05442 / 3.8472 |
| 7000 | – | – | 1.12003 / 4.0478 | 1.11923 / 4.0393 | 1.02789 / 3.7194 | 1.02713 / 3.7091 |
| 8000 | – | – | 1.09707 / 3.9612 | 1.09692 / 3.9643 | 1.00558 / 3.6399 | 1.00556 / 3.6305 |
| 9000 | – | – | 1.07792 / 3.8874 | 1.07761 / 3.8862 | 0.99176 / 3.5866 | 0.99180 / 3.5764 |
| 10000 | – | – | 1.06318 / 3.8333 | 1.06290 / 3.8294 | – | – |
| 11000 | – | – | 1.05662 / 3.8106 | 1.05649 / 3.8065 | – | – |
| final | 1.16358 / 4.1798 (4958) | 1.16354 / 4.1826 | 1.05626 / 3.8087 (11443) | 1.05602 / 3.8053 | 0.98830 / 3.5726 (9917) | 0.98824 / 3.5640 |

1_2b, step: H100 c4 / macro | TPU c4 / macro
0: – | 3.76872/12.2678; 1000: 1.18609/4.3380 | 1.18779/4.3399; 2000: 1.12475/4.1138 | 1.12555/4.0964; 3000: 1.10660/4.0463 | 1.10875/4.0506; 4000: 1.09607/3.9943 | 1.09623/3.9938; 5000: 1.08821/3.9645 | 1.08623/3.9456; 6000: 1.08111/3.9406 | 1.08105/3.9291; 7000: 1.07116/3.8869 | 1.07066/3.8940; 8000: 1.06288/3.8689 | 1.06305/3.8675; 9000: 1.05296/3.8202 | 1.05347/3.8205; 10000: 1.04381/3.8031 | 1.04362/3.8017; 11000: 1.03348/3.7547 | 1.03371/3.7802; 12000: 1.02151/3.7094 | 1.02184/3.7087; 13000: 1.01002/3.6621 | 1.01097/3.6692; 14000: 0.99830/3.6147 | 0.99889/3.6165; 15000: 0.98752/3.6300 | 0.98717/3.6210; 16000: 0.97523/3.5406 | 0.97505/3.5388; 17000: 0.96235/3.4873 | 0.96243/3.4974; 18000: 0.95190/3.4443 | 0.95243/3.4466; 19000: 0.94264/3.4023 | 0.94270/3.4103; 20000: 0.93536/3.3860 | 0.93528/3.3908; 21000: 0.93020/3.3717 | 0.93013/3.3734; 22000: 0.92791/3.3623 | 0.92757/3.3610; 22887: 0.92763/3.3609 | 0.92731/3.3605.

#### 3.3 Early training: how close curves stay (steps 2–200)

`d(t)` is the difference in `train/loss` at step t between two runs, over the 199 steps t = 2..200 (deduped by `_step`). "win10" is max |mean of d| over **non-overlapping** 10-step blocks aligned at step 2: blocks [2, 11], [12, 21], ..., [182, 191] (19 blocks). Steps 192–200 are not in any block (`_tools/targets_pairs.py:18`: `d[i:i+10].mean() for i in range(0, len(d)-9, 10)`). "mean d 100–200" is the mean over t = 100..200 (`d[98:]`, `:21`). Source: `_tools/targets_pairs.out`. The definition matters: with sliding 10-step windows (all 190 start steps 2..191), win10 is 0.01092 for 1_2b TPU vs H100 (above the 1_2b gate of 0.010), 0.0124–0.0287 for the 130m same-seed pairs and 0.00274 for the 300m same-seed pair (recomputed on della-vis1, 2026-10-24). Gate (b) uses the block definition.

| comparison | step 2 \|d\| | max \|d\| 2–20 | rms d 2–200 | max win10 | mean d 100–200 | first step with \|d\| > 1e-3 |
|---|---|---|---|---|---|---|
| 130m same seed, H100 vs H100 (s0..s3 raw vs EMA copy) | 0.00010–0.00119 | 0.020–0.069 | 0.0093–0.0152 | 0.0097–0.0233 | −0.0033 … +0.0086 | 2–5 |
| 130m seed 0, TPU vs H100 | 0.01163 | 0.0831 | 0.0153 | 0.0231 | +0.0096 | 2 |
| 130m different seeds, H100 (6 pairs) | 0.026–0.216 | 0.121–0.315 | 0.037–0.062 | 0.056–0.105 | −0.034 … +0.044 | 2 |
| 300m same seed, H100 vs H100 (s0 vs EMA copy) | 0.00005 | 0.00018 | 0.0012 | 0.0024 | +0.0010 | 126 |
| 300m seed 0, TPU vs H100 | 0.00002 | 0.0024 | 0.0040 | 0.0066 | −0.0027 | 12 |
| 300m different seeds, H100 (6 pairs) | 0.005–0.081 | 0.039–0.143 | 0.020–0.059 | 0.048–0.127 | −0.012 … +0.008 | 2 |
| 520m seed 0, TPU vs H100 | 0.00002 | 0.0038 | 0.0050 | 0.0096 | −0.0035 | 10 |
| 1_2b seed 0, TPU vs H100 | 0.00008 | 0.0033 | 0.0036 | 0.0092 | +0.0009 | 9 |

Pool envelope of `train/loss` (mean, sd, min, max):

* **130m** — step 2: 10.6261, 0.0823, 10.5181, 10.7343; step 10: 7.8421, 0.0416, 7.7882, 7.8974; step 50: 6.0756, 0.0362, 6.0469, 6.1434; step 100: 5.1764, 0.0241, 5.1523, 5.2150; step 200: 4.3982, 0.0179, 4.3727, 4.4279.
* **300m** — step 2: 12.2555, 0.0299, 12.2245, 12.3054; step 10: 11.7869, 0.0410, 11.7523, 11.8510; step 50: 9.6466, 0.0395, 9.5904, 9.6890; step 100: 7.8269, 0.0127, 7.8045, 7.8362; step 200: 5.9753, 0.0023, 5.9712, 5.9770.
* Every step from 2 to 200 is in `targets/_analysis_pools.json` (`envelope`).

**What this shows**

1. **The init is a pure function of the seed on every platform.** JAX threefry is platform-independent. At 300m, 520m and 1_2b the TPU and H100 seed-0 runs agree to 2e-5–8e-5 at step 2.
2. **The 130m recipe amplifies small differences.** It uses lr 0.02 with no warmup, so differences grow from step 2. Even two H100 runs with the same seed reach 0.02–0.07 apart by step 20.
3. **The warmup sizes are stable.** At 300m (warmup 1000) the only same-seed H100 pair (s0 vs EMA copy) stays within 1e-3 until step 126. 520m and 1_2b have no same-seed H100 pair. Their only pair is TPU vs H100, which passes 1e-3 at steps 10 and 9.
4. **A wrong init is easy to detect.** A different seed moves win10 by ≥ 0.048 at both sizes. A numerics-only difference (TPU vs H100) moves it by ≤ 0.023 at 130m and ≤ 0.0096 at 300m–1_2b.

---

### 4. Parity gates for marincpp stage 1

Conventions:

* "levanter-H100" means the reference code run on 4× H100 by `scripts/della/muonh_qwen3_h100x4.sbatch` with default knobs.
* "bf16 policy" is levanter's `mp: p=f32, c=bf16, out=bf16` (logged config `trainer.mp`).
* "f32 policy" is the same code with `compute_dtype=f32`.
* marincpp logs `train/loss` from step 0 with levanter's meaning: at step k, the loss of batch k under the parameters after k updates. Its periodic evals copy levanter's off-by-one: the eval logged at step s ∈ {1000, 2000, ...} runs after update s (parameters after s + 1 updates), and the final eval, logged at `num_train_steps − 1`, runs after all N updates (section 1, "Step semantics"). In addition it runs one eval **at init, before update 0**, and logs it as the init eval (for a3). That eval is not comparable with the TPU step-0 row at 130m.
* Optimizer groups (`muonh.py:106-127`; full inventory in `optim.md` §2.2). **muonh**: every transformer `hnn.Linear` weight (q, k, v, o, gate, up, down), stacked over layers. **adamh**: `lm_head.weight`. **adam**: `embeddings.token_embeddings.weight`, every RMSNorm gain (input, post-attention, the two hybrid post-norms, q_norm, k_norm, final norm) and any Linear bias (none exist, `use_bias = False`). The per-group gates below use these three groups.

#### (a) Step-0 loss and gradient agreement on a fixed batch from the same init

**Fixtures.** These are produced by a harness that imports levanter and runs on one GPU node. The harness lives in marincpp; nothing is added to the marin repo.

1. **Init.** The plain baseline builds the model at `train_lm.py:268`: `state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))`, with `model_key = split(PRNGKey(0), 4)[2]` (`:217-218`) and `Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), parameter_axis_mapping)` (`:250-251`). `trainer.initial_state` (`trainer.py:443`) runs `load_checkpoint_or_initialize` with the parameter sharding (`trainer.py:507`), and `TrainerState.init` casts by trainability (`cast_params_by_trainability`, `trainer_state.py:97-98`). (`train_lm.py:142` is the same expression inside `_load_lm_model_from_configured_source`, which runs only for an HF or checkpoint init source; it is not the baseline path.) The harness calls `trainer.initial_state` and saves `state.model`, or it shows that an eager `config.model.build` gives the same bits (this needs `jax_threefry_partitionable = True`, `trainer.py:99`). Save the model as a levanter tensorstore checkpoint. levanter can reload that checkpoint with `initialize_model_from_checkpoint_path` (`train_lm.py:86-92`, `:133-138`), which is also the round-trip check.
2. **Batch.** Batch 0 is the first `train_batch_size` sequences (128 for 130m/300m, 256 for 520m/1_2b, seq 4096) of exactly `config.data.train_set(Pos, config.trainer.batch_schedule, key=PRNGKey(42))` (`train_lm.py:256-260`, `data_key` overridden at `:222`). Inside it: `mix_key, shuffle_key = split(data_key)` (`data/text/datasets.py:788`); the per-component linear-permutation key is the first key of `key_iterator(shuffle_key)` (`:839-841`); the order then passes through a `MixtureDataset` with `mix_key` and `mixture_block_size` 2048 (`:794-800`). The key values and the closed form are in `data.md` §4.1–4.4. Save the token ids and the attention segment ids (EOS-based document masking) as `.npy`.
3. **Reference outputs.** Run levanter's loss and gradient at step 0, once with the bf16 policy and once with the f32 policy. Run each policy twice, and measure the levanter-vs-levanter floor `e_floor(T)` for **every** parameter tensor T. The atomic scatter-add makes the embedding floor nonzero, but no run has shown that the other tensors (fused CE backward, attention backward, reductions) are bitwise repeatable (section 3, item 2), so the floor is measured, not assumed. At 520m and 1_2b one optimizer step is 2 microbatches of 32 seqs/GPU (`per_device_parallelism = 32`, launcher `:435`, `:787`). The reference gradient is levanter's accumulated gradient: microbatch j = 0, 1 holds sequences [128j, 128j + 128) of the batch (`grad_accum.py:170-179`), the two gradients are summed in f32 in that order (`grad_accum.py:144`), and each leaf is then divided by 2 (`:157-162`; `optim.md` §8). marincpp copies that order for a4, or else it uses a4's bound with `e_floor` measured between levanter at 2 × 32 and levanter at another microbatch count.
4. **Init eval.** Run levanter's full Paloma eval on the saved init model, before any update, with both policies. This is the a3 reference. The eval set, its order, the 128-sequence eval batch and the zero-weight padding of the last batch (65 real rows, 63 padding rows) are specified in `eval.md` §2.2–2.3 (`trainer.py:1047-1049`, `:1168-1175`; `data/loader.py` `pad_final_batch`).

| id | quantity | pass |
|---|---|---|
| a1 | step-0 loss on batch 0, bf16 policy | \|L_nano − L_lev\| ≤ 1e-4 nats. Information only: TPU step-0 `train/loss` is 12.27958 / 12.24542 / 12.26019 / 12.25425 (130m / 300m / 520m / 1_2b). levanter-H100 is expected within 1e-4 of these, because the step-2 losses agree to ≤ 8e-5 at 300m+ (section 3.3). |
| a2 | step-0 loss, f32 policy | \|ΔL\|/L ≤ 1e-6 |
| a3 | full Paloma eval at init | Each of the 16 domain losses and macro_loss within 1e-4 of the levanter init eval (fixture 4). The bpb of each domain must use the same byte counts as levanter, on the same sequences and padding (`eval.md` §2.2–2.3). The TPU step-0 values (130m: c4_en bpb 3.57218, macro 11.7600) are after one update and are not a reference. |
| a4 | per-tensor gradient, f32 policy | For each parameter tensor T, relative L2 error ‖g_n − g_l‖/‖g_l‖ ≤ max(1e-5, 2·e_floor(T)), with `e_floor(T)` measured per tensor in fixture 3. |
| a5 | per-tensor gradient, bf16 policy | Error against the f32-policy levanter gradient ≤ 2× levanter's own bf16-vs-f32 error, tensor by tensor. Cosine ≥ 0.9999 for every tensor. |
| a6 | per-group gradient norms before clipping | Each group has its own clip (`muonh.py:64-65`, `:81-82`, `:89-90`). `optax.multi_transform` (`:102`) gives each chain the tree with the other groups' leaves replaced by empty `MaskedNode`s (`optax/transforms/_combining.py:265,278`; `_masking.py:31,92`), so each `clip_by_global_norm` sums only its own group (`optim.md` §3.1). Compare the three pre-clip norms ‖g_muonh‖, ‖g_adamh‖ and ‖g_adam‖ (groups in the conventions above): relative difference ≤ 1e-4 for each. Each group G is scaled by min(1, max_grad_norm/‖g_G‖) with `max_grad_norm` 1.0 (2.0 for 1_2b). A single norm over all parameters (the logged `grad/norm/total`) fixes none of the three clip factors, and marincpp must clip each group separately. |
| a7 | optimizer steps | **130m** (lr 0.02 / adam_lr 0.008 at update 0, no warmup): run update 0. Per-tensor parameter-update relative error ≤ 2× levanter's bf16-vs-f32 floor. The step-1 loss must agree with levanter-H100 to ≤ 2e-3; 2e-3 is about 2× the largest same-seed H100 step-2 \|d\|, which is 1.19e-3. **300m, 520m, 1_2b** (warmup 1000): the warmup is `optax.linear_schedule(0.0, peak, 1000)` (`optim/config.py:307`, `:317-318`) for both `learning_rate` and `adam_lr` (`muonh.py:58-59`), and `inject_hyperparams` evaluates it at count 0 for update 0 (`muonh.py:104`; `optim.md` §3, step 3). So lr = adam_lr = 0 at update 0. The Adam group applies `scale(-0)`, an exact zero. MuonH and AdamH apply `norm_preserving_update(p, u, 0)`, which returns `new/‖new‖·‖p‖ − p` with `new = p` (`optim/util.py:312-315` for the unstacked 2-D `lm_head`; `:317-322` per layer for the stacked MuonH weights, with the `_NORM_FLOOR` in the final division). That is a rounding-level renormalization, not 0 (`optim.md` §4.4). The update-0 relative error is therefore noise over noise and is not gated. Instead: (i) after update 0, compare the optimizer state (MuonH momentum buffer, AdamH and Adam mu and nu, count) per tensor at ≤ max(1e-5, 2·e_floor); (ii) compare the parameters after update 0 bitwise or to 1 ulp, which needs marincpp to run the renormalization at lr = 0 and not skip it; (iii) run update 1 at lr(1) = (0 − peak)·(1 − 1/1000) + peak (peak 0.01 at all three sizes, so ≈ 1.0e-5 in f32; adam_lr(1) ≈ adam_lr/1000 by the same form, `optim.md` §6.2) and gate its per-tensor update error at 2× levanter's bf16-vs-f32 floor; (iv) the step-1 loss checks the init on batch 1 (≤ 1e-4 against levanter-H100), and the step-2 loss checks update 1 (≤ 1e-4). TPU step-0 / step-1 / step-2 losses: 12.24542 / 12.25557 / 12.24745 (300m), 12.26019 / 12.25711 / 12.23668 (520m), 12.25425 / 12.25591 / 12.21986 (1_2b); 130m step 1: 11.56827. **Trap (all sizes):** the logged config has `weight_decay: 0.1` (`cache/wandb/targets/della__muonh-qwen3-130m-della4xh100.config.json:437`; launcher `:747`), but `MuonHConfig.build` never applies weight decay (`muonh.py:54-104`; `optim.md` §7). marincpp must apply none. At 130m a7 catches a stray decay at update 0; at 300m+ only part (iii) can. |

Gate (a) runs for every size. At 130m update 0 is the decisive test of the optimizer: the full LR applies from update 0, so the step-1 loss exercises MuonH, Newton–Schulz, the Hyperball projection, the AdamH lm_head and the Adam embedding/norms all at once. At 300m and above the same test is update 1 (a7 part iii), at 1/1000 of the peak LR.

#### (b) First 200 steps of train/loss

marincpp starts from the levanter seed-0 init and uses the same data order. It is compared step by step with the H100 seed-0 run `muonh-qwen3-<size>-della4xh100`, on the statistics of section 3.3 over steps 2–200.

| size | pass (no worse than TPU vs H100) | strong pass (no worse than H100 same seed) | fail signature (a different seed) |
|---|---|---|---|
| 130m | rms ≤ 0.016, max win10 ≤ 0.025, \|mean d 100–200\| ≤ 0.010 | rms ≤ 0.010, win10 ≤ 0.015 | rms ≥ 0.037, win10 ≥ 0.056 |
| 300m | max \|d\| 2–20 ≤ 0.003, rms ≤ 0.005, win10 ≤ 0.008 | max \|d\| 2–20 ≤ 5e-4, rms ≤ 0.002 | win10 ≥ 0.048 |
| 520m | max \|d\| 2–20 ≤ 0.004, rms ≤ 0.006, win10 ≤ 0.010 | – (no H100 same-seed pair) | – |
| 1_2b | max \|d\| 2–20 ≤ 0.004, rms ≤ 0.005, win10 ≤ 0.010 | – | – |

* **300m is the most informative size for this gate.** Its warmup keeps a correct implementation within 3e-3 for 20 steps, whereas at 130m the curves separate at the bf16 noise level from step 2.
* **Envelope check.** This is a secondary check, and the only one available if marincpp uses its own init. At 130m, the fraction of steps 2–200 outside the n = 8 pool [min, max] must be ≤ 0.50, and \|mean(nano − pool mean) over 100–200\| ≤ 0.035. Both bounds are the worst leave-one-out values of the pool members: 0.48 and 0.032 (`_tools/targets_analyze.out`, `LOO 130m`). At 300m the n = 5 pool is too tight after step 150 (sd 0.0023, LOO outside-fraction up to 0.87), so only the seed-matched test above applies there.

#### (c) Final metrics within the pool

Metrics: `c4_en bpb`, `eval/paloma/macro_loss`, `eval/paloma/macro_bpb`, plus the 16 domain losses. Every value is read at step `num_train_steps − 1`. A run is admitted only when its last eval is at that step (this guard exists because of the summary-of-a-running-run error, `scripts/della/objective_compare.py`).

**c1, single-run screen (every marincpp run).** The value must lie inside the pool's 99 % prediction interval, mean ± t₀.₉₉₅,ₙ₋₁ · sd · √(1 + 1/n):

| size | pool n | half-width | c4_en bpb | macro_loss | macro_bpb |
|---|---|---|---|---|---|
| 130m | 8 | 3.712 sd | [1.16204, 1.16449] | [4.16504, 4.20788] | [1.50581, 1.52177] |
| 300m | 5 | 5.043 sd | [1.05495, 1.05787] | [3.78491, 3.84281] | [1.37297, 1.39505] |
| 520m | 1 | √2 · 2.576 · sd₃₀₀ₘ (borrowed) | 0.98830 ± 0.0011 | 3.57255 ± 0.021 | 1.29952 ± 0.008 |
| 1_2b | 1 | same | 0.92763 ± 0.0011 | 3.36087 ± 0.021 | 1.22578 ± 0.008 |

**c2, acceptance (130m and 300m).** Run n_nano = 4 marincpp runs with seeds 0–3. Two tests apply to each of the three headline metrics:

* Welch's two-sample t-test against the pool, two-sided at α = 0.05: must **not** reject.
* TOST equivalence at α = 0.05 with margin δ = 2 pool sd: must **pass**.

| size | δ c4_en bpb | δ macro_loss | δ macro_bpb |
|---|---|---|---|
| 130m | 0.00066 | 0.0115 | 0.0043 |
| 300m | 0.00058 | 0.0115 | 0.0044 |

* **Power** (simulated: Welch df, α = 0.05 per side, true difference 0, normal data, 40k draws, della-vis1 2026-10-24). The low end of each range uses δ = 2 × sample pool sd, as the gate defines it; the high end uses δ = 2σ. TOST pass probability per metric: 130m (pool n = 8) 0.68–0.76 at n_nano = 4 and 0.89–0.97 at n_nano = 8; 300m (pool n = 5) 0.57–0.66 at n_nano = 4 and 0.76–0.88 at n_nano = 8. TOST and Welch together per metric: 130m 0.66–0.74 (n_nano = 4), 0.88–0.93 (n_nano = 8); 300m 0.57–0.65 (4), 0.76–0.85 (8). Acceptance needs both tests on all three metrics. The joint probability is lower than any per-metric figure: if the three metrics were independent it would be 0.29–0.40 for 130m at n_nano = 4 and 0.67–0.81 at n_nano = 8. The metrics are positively correlated, so the true joint value lies between that product and the per-metric value.
* **Escalation.** If Welch does not reject but TOST fails, add 4 more runs (n_nano = 8). The per-metric pass probability then rises to the n_nano = 8 figures above.
* **Per domain.** No domain loss may differ from the pool at Bonferroni α = 0.05/16 = 0.0031 (Welch). Against the TPU originals, compare loss only (section 3.1).

**c3 (520m, 1_2b).** One marincpp run must pass the c1 screen against the single levanter-H100 run. The verdict stays provisional until the levanter pool reaches n ≥ 4 at that size. That costs about 18.5 h of 4× H100 per 520m run and about 50 h per 1_2b run. The TPU original is a second draw for c4_en and the domain losses only.

#### (d) Throughput

All three throughput numbers below use one definition: p50 of the per-step `tokens_per_second` over steps ≥ 20, with eval steps excluded, on 4× H100 80GB SXM at the reference global batch. MFU uses levanter's definition: training FLOP/token = 3 × `lm_flops_per_token(...)` (`levanter/utils/flop_utils.py:5-61`, `main/train_lm.py:388-389`). That formula counts full, non-causal attention and the lm_head. The H100 peak is 989.5 TFLOP/s bf16 per GPU (`fray/device_flops.py:21-27`), and MFU = model FLOP/s ÷ (4 × peak) (`callbacks/_metrics.py:66`). The FLOP/token values are 683.6M (130m), 1665.1M (300m), 4048.6M (520m) and 8631.9M (1_2b); they equal `model_flops/total_tokens` in `speedrun_results.json`.

| size | levanter-H100 step s | levanter-H100 tok/s | levanter-H100 MFU % | TPU v5e-64 step s | TPU tok/s (total) | TPU MFU % | **gate: tok/s ≥** | **goal: tok/s ≥ (1.5×)** |
|---|---|---|---|---|---|---|---|---|
| 130m | 0.6316 | 830,158 | 14.34 | 0.1296 | 4,046,239 | 21.94 | 830,000 | 1,245,000 (MFU 21.5) |
| 300m | 1.4437 | 363,155 | 15.28 | 0.3654 | 1,434,928 | 18.95 | 363,000 | 545,000 (MFU 22.9) |
| 520m | 6.7415 | 155,541 | 15.91 | 1.4001 | 748,915 | 24.05 | 155,500 | 233,000 (MFU 23.9) |
| 1_2b | 8.2209 | 127,550 | 27.82 | 1.6318 | 642,584 | 43.99 | 127,500 | 191,000 (MFU 41.7) |

The H100 values are pool means at 130m and 300m (run-to-run sd 1584 and 243 tok/s) and single runs at 520m and 1_2b.

* **Gate.** marincpp is no slower than levanter-H100 at any size. The 1.5× goal brings the H100 MFU up to the TPU-original level.
* **Wall clock.** Report end-to-end time as well. Reference: the 130m s0 job took 3441 s (`sacct`), against 4959 × 0.6305 = 3127 s of pure stepping. The remaining ≈ 314 s is compile, five full Paloma evals and checkpoints.
* **Peak memory is not recorded by the reference.** The sbatch preallocates 90 % of each GPU (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, `scripts/della/muonh_qwen3_h100x4.sbatch:28`). The W&B system metric `memoryAllocatedBytes` (max 7.94e10 B, 92.9 %) therefore shows the preallocation, not the working set. Known limit: 520m at 64 seqs/GPU in one microbatch failed with an extra 35.45 GiB allocation (`kaiyue_muonh_qwen3_h100x4.yaml:53`), which is why 520m and 1_2b use 2 × 32. marincpp must log the true peak (for example `torch.cuda.max_memory_allocated` or the XLA memory stats) for each size. There is no reference number to beat.

---

### 5. Order of the gates

1. **(a)** at 130m and 300m. Fixtures cost one GPU job of under 30 min.
2. **(b)** at 300m: 200 steps, about 5 min.
3. **(b)** at 130m.
4. **(c1)** with one 130m seed-0 run, about 1 h.
5. **(c2)** with four 130m runs, then four 300m runs.
6. **(d)** from the same runs.
7. **(a)**, **(b)** and **(c3)** at 520m and 1_2b.

---

### 6. Corrections to values quoted elsewhere

| quoted | where | correct (W&B) | reason |
|---|---|---|---|
| 130m pool macro_loss 4.1848, sd 0.0043 | stage-1 task text; `objective_hillclimb_qwen3_h100x4.yaml:699,738` | **4.18646, sd 0.00577 (n = 8)** | 4.1848 is the mean of the 4 raw seed runs s0–s3 at 4 decimals. The n = 8 pool (same 8 runs as the c4_en pool) is at `:954-955`. |
| 130m pool c4_en 1.16327, sd 0.00035 | task text; `objective_hillclimb_qwen3_h100x4.yaml:897-898` | 1.16326, sd 0.00033 | Mean: rounding only. The ledger's own 8 five-decimal inputs (`:897`) give 1.163265, and the full-precision W&B values give 1.1632648. sd: an arithmetic slip in the ledger. The same rounded inputs give sd (ddof = 1) 0.000331, not 0.00035. |
| originals ran on "64 v5p chips" | `kaiyue_muonh_qwen3_h100x4.yaml:45,50`; `marin-moe-dense-2026-09-18.md:174` | v5litepod-64 (TPU v5e) | `speedrun_results.json` `tpu_type`; MFU-implied peak 1.97e14 FLOP/s per chip. |
| 300m pool n = 4 | objective-hillclimb ledger | n = 5 | `muonh-qwen3-300m-della4xh100-ema0.999` (2026-10-22, vocab-overfitting plan `ema-128`) is a finished plain baseline. |
| "same-seed reruns reproduce to 1e-4" | `kaiyue_muonh_qwen3_h100x4.yaml` (+0.00004 vs TPU) | same-seed H100 sd ≈ 0.0005 on c4_en | `objective_hillclimb_qwen3_h100x4.yaml:884-893`. The 0.00004 agreement was a single pair. |
| `eval/paloma/micro_loss` comparable to TPU | – | not comparable; compare `eval/loss` | On H100, `eval/paloma/micro_loss` equals `eval/loss` up to float32 rounding: equal at the final step (130m s0: 3.968565; 300m s0: 3.584760), at most 1.9e-6 apart at other eval steps (`eval.md` §4.2; two formulas, not one value). On TPU they differ: 130m `eval/loss` 3.97142 vs `micro_loss` 4.30306 (300m: 3.58326 vs 3.94576). The TPU-era levanter weighted the per-domain mean losses by the losses themselves: `total_tokens_per_tag_cpu = np.array(state.loss_per_tag.mean.array)` (`git show 14c1488198:lib/levanter/src/levanter/eval.py`, line 414; fixed to `.total` in `fc04a02cd1`, 2026-02-19, now `eval.py:627`). So TPU micro_loss = Σ L_i² / Σ L_i over the 16 domains; the TPU 130m domain losses of section 3.1 give exactly 4.30306. Compare `eval/loss` (the token-weighted mean) across platforms. Do not gate on micro_loss. |

---


---

## Parity test plan

A new implementation passes these checks in this order. A check runs only after every check above it passes. Gates
T2 to T6 restate `targets.md` §4 (Part 6) with the fixture recipe added; T0 and T1 are new.

### Conventions

- **Metric.** `rel(a, b) = ||a - b||_2 / ||b||_2`, computed in float64 over one tensor. "ulp" is the unit in the last
  place of the reference value in its own dtype.
- **Oracle rule** (for ops whose rounding is class P in `numerics.md` §1). For each output tensor, compute an f64
  oracle from the same inputs upcast to f64. Pass if (i) `rel(nano, oracle) <= 1.25 x rel(lev, oracle)`: marincpp
  is not less accurate than levanter, with 25% slack for accumulation order; and, when marincpp claims the
  bit-faithful reading (a) of `model.md` §7 item 4, also (ii) `rel(nano, lev) <= rel(lev, oracle)`: the port differs
  from levanter by less than levanter's own rounding error.
- **Levanter fixtures.** A harness in marincpp, `tools/levref/` (not written yet), imports levanter read-only from
  `$ROOT/project/marin/.venv`, builds the plain-baseline `TrainLmConfig` the way the launcher
  does (`della_muonh_qwen3_scaling.py:736-820`) with the **full** `num_train_steps`, and writes `.npy` and JSON files to
  `$ROOT/project/marin.cpp/cache/levref/<size>/`. It never writes into the marin tree.
  - Never use the launcher's `SMOKE_STEPS` for a reference: it sets `num_train_steps`, so it changes the decay length
    (130m: `int(0.8 x N)`), the cosine `T_d`, `steps_per_eval` and `max_eval_batches=1`
    (`della_muonh_qwen3_scaling.py:788-790`).
  - GPU fixtures run as Slurm jobs on `--account=pli --partition=pli-c --qos=pli-cp` (smoke tests only), 1 GPU for
    T1 and 4 GPUs for T2 and T3 (4-way sharding and 32 sequences per GPU reproduce the CE tiles and reduction layout),
    `--time` at most 00:30:00, `TMPDIR` under `$ROOT/tmp/job/$SLURM_JOB_ID`, no W&B.
  - CPU work and every W&B read run on `ssh della-vis1 'nice -n 19 $ROOT/project/marin/.venv/bin/python ...'`.
- **marincpp runs.** Runs of 30 min or less (T1-T4) also go to pli-cp. Full training runs (T5, T6) go to pli-short
  through `tools/wq/wq submit`, with `--partial` for any run over 3 h.

### T0. Static checks (CPU, della-vis1, no GPU)

| id | check | pass | levanter-side reference |
|---|---|---|---|
| T0.1 | Data order table s(j), j = 0..5,859,327 | sha256 of the little-endian uint32 table = `dde0ca7b...655b` (`data.md` §1) | exists: `cache/data/fineweb_order_seed42.u32`, made by `cache/data/_tools/data_order_ref.py` (levanter `LmDataConfig.train_set` plus an independent closed form; log `cache/data/_tools/run.log`) |
| T0.2 | Batch k token hashes `H(B, k)` | all 12 sha256 of `data.md` §7 equal, bitwise | exists: `cache/data/data_ref.json`; rerun `data_order_ref.py` on della-vis1 for any new k |
| T0.3 | Segment ids and loss weights of the same 12 batches | bitwise equal to the formulas of `data.md` §3.1 (segment boundary after every 128001; weight 0 only at position 4095) | same script (it calls `GrugLmExample.causal`) |
| T0.4 | Paloma eval set | the 16 per-domain sha256 of `data.md` §8.1, 14,529 sequences, 114 batches of 128, 63 zero rows in batch 113 | `data.md` §8.1 table (made by the same script) |
| T0.5 | Bytes-per-token table | sha256 `d3be7450...4e4d`, all 128,256 entries (`data.md` §8.4, `eval.md` §8 test 1) | `cache/data/bytes_per_token_marin.i32`; recheck against `levanter.utils.hf_utils.byte_length_of_token` on della-vis1 |
| T0.6 | Optimizer scalar constants | f32 bit patterns equal to the hex column of `optim.md` §5.3 | that table (numpy float32 on della-vis1) |
| T0.7 | Schedules eta(k), eta_adam(k), k = 0..N-1, all four sizes | f32-bitwise equal to the logged `optim/learning_rate` and `optim/adam_lr` at every logged k; at the holes and k = 0, 1, equal to the formulas of `optim.md` §6. Includes 130m k = 4958: 5.041361e-06 / 2.016544e-06; 1_2b k = 22887: exactly 0 | cached histories `cache/wandb/run-2026091[4-7]_*-muonh-qwen3-<size>-della4xh100.jsonl.gz` (dedupe by `_step`) |
| T0.8 | Parameter count and `flops_per_example` | exact integers of the hyperparameter table | W&B `parameter_count`, `throughput/flops_per_example` (cached `cache/wandb/api/*.json`) |

### T1. Unit kernels against levanter on fixed inputs (pli-cp, 1 GPU)

Fixed inputs, produced once by the harness: the levanter seed-0 init model (T1.1) and batch 0 (T0.2); the input of each
op is the tensor that the levanter bf16 forward feeds that op in layer 0 for batch-0 rows 0-1; cotangents (`dy`,
`dO`) are bf16 draws from `jax.random.normal(PRNGKey(1234))`. Shapes: the 130m and the 1_2b configuration (hd 64 MHA
and hd 128 GQA); attention also at the 520m configuration (hd 64 GQA). All outputs and the inputs are saved.

| id | check | pass | levanter-side reference |
|---|---|---|---|
| T1.0 | Precision facts | Read from the optimized HLO of one 130m train step: (1) dtype of the cross-GPU gradient reductions, (2) whether the CE backward logit dot keeps its bf16 output, (3) embedding scatter accumulator dtype, (4) norm-gain `reduce_sum` accumulator dtype, (5) whether f32 dots (Newton-Schulz, flash `dV`) run as TF32, (6) `xla_gpu_ftz` and fast-math options; record loaded `libcublas`, `libcudnn`, `libnccl`. These facts fix the tolerance class of T1.4-T1.6 (`numerics.md` §10, O1, O2) | harness runs one full-N 130m step on 4 GPUs with `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"` and `LD_DEBUG=libs` |
| T1.1 | Init | Port of threefry2x32 (partitionable) plus the uniform-to-`erf_inv` sampler (`model.md` §4.2): every tensor bitwise equal. Otherwise, per tensor of `model.md` §4.1: `abs(std / (0.98658 x scale) - 1) <= 5 / sqrt(2 x numel)`; `abs(mean) <= 5 x 0.98658 x scale / sqrt(numel)`; `max abs(w) < 3 x scale`; norm gains exactly 1. In both cases: `lm_head / embedding = sqrt(D)` elementwise to `<= 2.4e-7` relative (one shared draw, two f32 scalings), so their correlation is 1, never 0 | harness: `trainer.initial_state(training_key, model_init=...)` exactly as `train_lm.py:268` (seed 0, `jax_threefry_partitionable=True`); saves `state.model` as a levanter tensorstore checkpoint plus per-tensor stats JSON (`targets.md` §4(a) fixture 1) |
| T1.2 | RMSNorm (block norm over D, QK-norm over hd) | forward `out` and backward `dx`: at least 99% of elements bitwise equal and every element within 1 bf16 ulp of levanter (elementwise math with fixed rounding points, `numerics.md` §3.2); gain gradient `dg` (bf16 reduction over 524,288 positions): oracle rule | harness calls `haliax.nn.RMSNorm` from `LlamaConfig.norm_config` (`levanter/layers/normalization.py:35-39`) under `jax.vjp` |
| T1.3 | RoPE | f32 `inv_freq` (built as `numerics.md` §3.3) bitwise; bf16 cos and sin tables `[4096, hd]` for hd 64 and 128 bitwise; rotated q and k bitwise | harness calls `DefaultRotaryEmbeddings` (`levanter/layers/rotary.py:30-46`) with positions 0..4095 (`numerics.md` O4) |
| T1.4 | Attention (QK-norm, RoPE, scale, flash) on q, k, v of batch-0 rows 0-1 with their EOS-derived segment ids | scale constant bf16 `0x3E00` (hd 64) or `0x3DB5` (hd 128), applied to q before the kernel and to dq after it; output O, log-normalizer L, dq, dk, dv: oracle rule, where the oracle is f64 causal and same-segment softmax attention. A row's output must not depend on keys of other segments (the oracle rule enforces it) | harness calls `levanter.models.flash_attention.flash_attention` through `Attention` with `AttentionBackend.JAX_FLASH`, block 1024, `upcast_attn=False` (`numerics.md` §4) |
| T1.5 | Fused cross-entropy on x = final-normed hidden states `[131072, D]` (one GPU's rows of batch 0), W = init `lm_head`, labels = next tokens, weights per `data.md` §3.1 | forward per-token loss: `max abs diff <= 2e-5` nats (both sides use f32 logits from a bf16 GEMM and an f32 full-vocab logsumexp; only accumulation order differs); weighted mean: `abs diff <= 1e-6` relative; `grad_x`, `grad_w`: oracle rule (oracle: f64 softmax minus one-hot, times `1/524160`) | harness calls `maybe_fused_next_token_loss` (`levanter/models/loss.py:35-98`), which takes the H100 `batched_xla` B-tiled path (8192-row tiles) under `jax.vjp` |
| T1.6 | Newton-Schulz, one matrix per shape of `optim.md` §4.3 (input: levanter's Nesterov update `u` of each MuonH leaf after update 0 of 130m and 1_2b) | if T1.0 shows f32 dots: `rel(nano, lev) <= 1e-5`. If T1.0 shows TF32: oracle rule with levanter at `precision=HIGHEST` as the oracle | harness calls `levanter.optim.util.zeropower_via_newtonschulz5(X, steps=5, eps=1e-5, coefficient_type="simple")` (`util.py:244`; its defaults, eps 1e-7 and "quintic", are not the baseline) twice: default precision and under `jax.default_matmul_precision("highest")` |
| T1.7 | One optimizer update on a fixed gradient. Cases: (A) init params, zero state, gradient G0 = levanter's f32-policy step-0 gradient, k = 0 (130m: full lr; 300m: lr 0); (B) case A followed by k = 1 (300m: lr(1) per `optim.md` §6.2); (C) params and optimizer state of the plain 130m checkpoint below, gradient G0, k = 4958 | three pre-clip group norms and clip factors: `rel <= 1e-6`; MuonH buffer, Adam and AdamH `mu`, `nu`: `rel <= 1e-6` per tensor, `count` exact; Adam updates: `rel <= 1e-6`; MuonH and AdamH updates (`target - p`): `rel <= 1e-5` with f32 dots, oracle rule with TF32 (T1.6); per-layer Frobenius norm of every MuonH weight and the whole-matrix norm of `lm_head` preserved to `<= 1e-6` relative; case A at lr 0: parameters after the update bitwise equal or within 1 f32 ulp per element (`targets.md` a7 (ii)); no weight-decay term (a stray 0.1 decay breaks the 1e-6 bound on the Adam updates) | harness runs levanter's `MuonHConfig.build(N)` optax transform (`levanter/optim/muonh.py:54-104`) on the saved params and G0. Case C loads `marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-restore/2026.09.13/checkpoints/step-4958` (plain baseline config, seed 0, the 2026-09-29 rerun of `numerics.md` §9.2) |

### T2. Step-0 loss and gradients (`targets.md` gate (a); pli-cp, 4 GPUs; 130m and 300m first)

| id | check | pass | levanter-side reference |
|---|---|---|---|
| T2.1 | a1: step-0 loss on batch 0, bf16 policy | `abs(L_nano - L_lev) <= 1e-4` nats | harness: loss and gradient of the T1.1 init on batch 0, bf16 policy, run twice (`targets.md` §4(a) fixtures 1-3) |
| T2.2 | a2: step-0 loss, f32 policy | `abs(dL) / L <= 1e-6` | same harness with `compute_dtype=f32`, run twice |
| T2.3 | a3: full Paloma eval of the init model | each of the 16 domain losses and macro_loss within 1e-4 nats; bpb with levanter's byte table and batching (`eval.md` §2.3, §4.3) | harness runs levanter's `cb_tagged_lm_evaluate` path on the init model, both policies (fixture 4); not the TPU step-0 row |
| T2.4 | a4: per-tensor gradient, f32 policy | `rel(g_nano, g_lev) <= max(1e-5, 2 x e_floor(T))`; `e_floor(T)` = rel between the two levanter repeats of T2.2 | T2.2 outputs |
| T2.5 | a5: per-tensor gradient, bf16 policy | `rel(g_nano_bf16, g_lev_f32) <= 2 x rel(g_lev_bf16, g_lev_f32)` and cosine >= 0.9999 for every tensor | T2.1 and T2.2 outputs |
| T2.6 | a6: three pre-clip group norms (muonh, adamh, adam) | each within 1e-4 relative; each group clipped by its own norm | T2.1 outputs |
| T2.7 | a7: optimizer step(s) | 130m: update 0 per-tensor error <= 2x levanter's bf16-vs-f32 floor; step-1 loss within 2e-3 of levanter-H100. 300m, 520m, 1_2b: state after update 0 within `max(1e-5, 2 x e_floor)`; parameters after update 0 bitwise or 1 ulp; update 1 error <= 2x floor; step-1 and step-2 losses within 1e-4 (`targets.md` a7) | harness runs updates 0 and 1 in levanter after T2.1; at 520m and 1_2b the reference gradient is the 2 x 128 microbatch mean in levanter's order (`grad_accum.py:144, 157-162, 170-179`) |

### T3. Eval on a fixed trained checkpoint (pli-cp, 4 GPUs)

Checkpoint: `marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-restore/2026.09.13/checkpoints/step-4958` (weights
after all 4959 updates). Levanter's own eval of it is in `checkpoints/eval_metrics.jsonl` (last row, step 4958):
`eval/paloma/c4_en-marin-tokenizer/bpb` 1.163219928741455, `eval/paloma/macro_loss` 4.181971549987793, `eval/loss`
3.9724690914154053.

| id | check | pass | levanter-side reference |
|---|---|---|---|
| T3.1 | Aggregation only (`eval.md` §8 test 2): levanter's per-position losses fed to marincpp's aggregator | all 42 keys of `eval.md` §4.2 within 2 float32 ulp of levanter's values; the 16 `bpb_k`, `eval/bpb` and `eval/paloma/bpb` match under domain order A (`eval.md` §2.3) | harness dumps `per_pos_loss`, weights, target ids and tags for all 114 batches with `compute_next_token_loss(reduction=None)` on that checkpoint, and runs levanter's evaluator once |
| T3.2 | End to end (`eval.md` §8 test 3): marincpp loads the checkpoint and evaluates | each domain loss within 2e-4 nats; c4_en bpb within 1e-4; macro_loss within 2e-4 (every bound is below 0.35 pool sd of its metric) | the same levanter eval (T3.1) |
| T3.3 | Logging contract (`eval.md` §8 test 4): 25-step marincpp run, `steps_per_eval` 10, scratch W&B project | eval rows at steps 10, 20, 24; `train/loss` from step 2 (marincpp may also log steps 0 and 1); norms at 10, 20; `summary._step` = 24; state `finished`; `config.trainer.num_train_steps` = 25; `objective_compare.collect()` admits it; exactly the 16 domain-loss keys match `^eval/paloma/(.+)-marin-tokenizer/loss$` | `scripts/della/objective_compare.py` on della-vis1 |

### T4. First 200 steps (`targets.md` gate (b); 300m, then 130m; marincpp on pli-cp)

marincpp starts from the T1.1 levanter init, reads the T0.1 data order, uses the full-N schedule and runs 200
updates. It logs `train/loss` with levanter's meaning (loss of batch k under the parameters after k updates).

| size | pass | strong pass | levanter-side reference |
|---|---|---|---|
| 300m | max abs d over steps 2-20 <= 0.003, rms d over 2-200 <= 0.005, max win10 <= 0.008 | max abs d 2-20 <= 5e-4, rms <= 0.002 | W&B seed-0 history of `muonh-qwen3-300m-della4xh100`, steps 2-200 (cached `cache/wandb/targets/della__muonh-qwen3-300m-della4xh100.hist.json.gz`); statistics as `cache/wandb/_tools/targets_pairs.py` (non-overlapping 10-step blocks from step 2), on della-vis1. No levanter rerun |
| 130m | rms <= 0.016, max win10 <= 0.025, abs(mean d over 100-200) <= 0.010 | rms <= 0.010, win10 <= 0.015 | same, `muonh-qwen3-130m-della4xh100`; plus the n = 8 envelope check of `targets.md` §4(b) |

A different seed shows as win10 >= 0.048 at both sizes (fail signature).

### T5. Full 130m runs against the pool (pli-short via wq)

| id | check | pass | levanter-side reference |
|---|---|---|---|
| T5.1 | c1: one seed-0 run, values at step 4958 | c4_en bpb in [1.16204, 1.16449], macro_loss in [4.16504, 4.20788], macro_bpb in [1.50581, 1.52177] | the n = 8 pool of `targets.md` §2 (8 finished W&B runs) |
| T5.2 | c2: seeds 0-3 | for c4_en bpb, macro_loss, macro_bpb: Welch two-sided alpha 0.05 does not reject, and TOST alpha 0.05 with margins 0.00066 / 0.0115 / 0.0043 passes; if Welch passes and TOST fails, add seeds 4-7. Per domain: no Welch rejection at alpha 0.05/16 | on della-vis1: `.venv/bin/python scripts/della/objective_compare.py --pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3 --cell=-nm,-nm-s1,-nm-s2,-nm-s3 --name=marincpp` (run ids `muonh-qwen3-130m-della4xh100-nm[-s<k>]` in `reself/marin-della`, `eval.md` §7) |
| T5.3 | d: throughput, same runs | p50 tokens/s over steps >= 20, eval steps excluded, >= 830,000 on 4x H100 (goal 1,245,000); MFU by levanter's formula; true peak memory logged | `targets.md` §4(d) table |

### T6. Larger sizes

1. 300m: T2 already done; T4 done; four runs, seeds 0-3, for c1 and c2 (margins 0.00058 / 0.0115 / 0.0044), and d
   (>= 363,000 tok/s). Each run is about 4.6 h, so it runs as `--partial`.
2. 520m and 1_2b: T2 and T4, then one run each for c3 (the c1 screen against the single levanter run: c4_en +- 0.0011,
   macro_loss +- 0.021, macro_bpb +- 0.008) and d (>= 155,500 and >= 127,500 tok/s). The verdict stays provisional
   until the levanter pool reaches n >= 4 at that size. A bit-exact 520m replay must use one microbatch of 256 for
   updates 0-85 (§C1); a parity run need not.
