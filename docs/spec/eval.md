# Evaluation and logged metrics

Stage-1 spec section. Goal: the new stack evaluates the same text in the same way and logs the same W&B keys as the
levanter muonh_qwen3 baseline, so `scripts/della/objective_compare.py` runs on its runs unchanged.

## 0. Sources

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
  `nanomarin/cache/wandb/api/<run>.meta.json`, written by `cache/wandb/_tools/api_meta.py` on della-vis1. The history
  of each local run segment is cached in `nanomarin/cache/wandb/run-*.jsonl.gz`.
- Default knobs: no VARIANT, EMA_BETA=0, TPP=20, VOCAB_K=0, SEQ_LEN=4096, BATCH=0, SMOKE_STEPS=0, NUM_GPUS=4.

## 1. Eval schedule

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

## 2. Eval data

### 2.1 Domains

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

### 2.2 Sequences

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

### 2.3 Order and batches

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

## 3. Per-position loss

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

## 4. Aggregation and eval keys

### 4.1 Per-batch accumulation (L/eval.py:546-592)

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

### 4.2 Final values (L/eval.py:596-658) and keys (L/eval.py:444-484)

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

### 4.3 Bits per byte

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

## 5. Raw or EMA weights

Raw weights. `model_averaging = null` in the logged config (launcher:813: an EMA config only if `EMA_BETA > 0`;
default 0, launcher:187). The callback evaluates `step.model` when `eval_current` is True (default) and evaluates
`step.eval_model` under prefix `eval/ema/` only when `model_averaging` is not None (L/eval.py:288-289, 347-360). The
baseline therefore logs no `eval/ema/*` key. The pool runs with suffix `-ema0.999` log both. The comparison script
reads only their raw `eval/` keys (§6).

## 6. Train-time metrics

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

## 7. Contract with `scripts/della/objective_compare.py`

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
   `.venv/bin/python scripts/della/objective_compare.py --pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3 --cell=-nm,-nm-s1,-nm-s2,-nm-s3 --name=nanomarin`
   (on a vis node, `--flag=value` form).

Optional exact bpb (a need of the Tokenizer Size session, docs/survey/needs_from_sessions.md): log it under a prefix
that matches none of the patterns above, e.g. `eval_exact/paloma/<subset>/bpb` and `eval_exact/macro_bpb`, computed
as `loss_k * T_k / (ln 2 * B_exact_k)` with an exact byte table (§4.3). The levanter-estimator keys stay as they are.

## 8. Implementation recipe and tests

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

## 9. Open points

1. Domain order (§2.3): the code fixes reading A (in-process config, byte-sorted insertion order); test 2 is a confirmation only.
2. The sign of the macro_bpb bias: this section measures logged = exact - 0.0004..0.0006; the survey says +0.004.
3. The eval tolerance in test 3 is a proposal. The owner sets it.
4. A parity run on 8 GPUs at 520m/1_2b must force the eval batch to 128 (per device 16), or its bpb keys differ from
   the 4-GPU baseline by the estimator effect.

## Review log

Two skeptic reviews, checked against the code and the caches on 2026-10-24.

| # | issue | verdict | reason |
|---|---|---|---|
| 1 | §0: one commit 59ac0d64 for all runs | applied | The `.wandb` file of each segment gives seven commits (59ac0d64, 28fd5996, 33415cba, ac6af868, 0672e5ce, afdc3498, cead09a2); W&B `config.git_commit` holds only the last one. `git diff --stat <c> HEAD` over the eval, data, model, kernel, tracker and trainer code shows only trainer.py (+17) for each. Segment counts: 3, 11 and 27 local segments, not 31 at 1_2b. |
| 2 | §4.2: micro_loss "same value as" eval/loss | applied | Different formulas (eval.py:567, 609 vs 637); history gaps 1.9e-6, 1.9e-6, 7.2e-7 confirmed. |
| 3 | §4.2: macro example 1.5112996101379395 vs ...397 | applied | Both keys are ...395 in history; ...397 is the summary rendering of both. Replaced with 130m step 3000 (1.6098204851150513 vs 1.6098203659057617), added 520m and 1_2b examples and the history-vs-summary rule. |
| 4 | §6 train/loss: 520m always 2 microbatches | applied | 520m segment 0 has PDP 64 (`.wandb` config and cached config.yaml:497-498); launcher at 59ac0d64 has `{"1_2b": 32}` only. Steps 0-119 used one microbatch; no eval ran there. |
| 5 | §2.1: three names differ from Paloma directories | applied | paloma.py:23-40 maps them to identical directory names; only 4chan -> 4chan_meta_sep differs. |
| 6 | §4.1: per-tag reduction order and TF32 | applied, reworded | The einsums, dtypes, DEFAULT precision and P(None) sharding are as stated. The addition order inside an XLA reduction is not fixed by the code, so the spec targets float32 accuracy, not bitwise sums. The TF32 inference holds: a 4095 -> 4096 rounding would bias every loss_k by 2.4e-4 relative, against an observed eval/loss vs micro_loss gap <= 1.9e-6. |
| 7 | §3: per_pos_loss already masked; double multiply | applied | H/nn/loss.py:143-144 multiplies by `loss_weight * not_last_mask`; eval.py:555 multiplies again. Added that `inf * 0 = NaN`, so a select, not a product, is what masks. |
| 8 | §3: CE kernel path not pinned | applied, with a correction | The kernel does not use the inferred vocab block size on H100: `batched_xla` takes the H100 full-vocab path (batched_xla.py:369-389, 428-440), 8192-row tiles, bf16 GEMM with float32 output over all 128,256 columns, float32 `jax.nn.logsumexp`. The launcher comment "block sizes change only the tiling" was replaced with this. |
| 9 | §0 and §2.3.3: per-size commits; PDP 32 from the last segment | applied, partly superseded | Per-size commits and the 520m segment-0 eval batch of 256 (no eval) are right. The reviewer's "middle 520m segments not cached, batch 128 inferred from commit times" is superseded: the config in each segment's `.wandb` file gives PDP 32 for all middle segments, so the eval batch of 128 is confirmed, not inferred. |
| 10 | §4.2: macro example (duplicate of 3) | applied | As 3, plus the note that bit-level tests read history, not `run.summary`. |
| 11 | §4.2: n=8 label on macro_loss | applied | objective-hillclimb.md:58 gives n=8 only for c4_en (1.16327, sd 0.00035); line 216 gives the n=4 baseline macro 4.1848 +- 0.0043; line 58 gives the run-to-run macro sd 0.007-0.010. |
| 12 | §6: FLOPs cite qwen.py:115-125 | applied | Qwen3Config subclasses LlamaConfig (qwen.py:299); flops_per_token is llama.py:200-210. Same formula, so 227,868,672 stands. |
| 13 | §2.1: names (duplicate of 5) | applied | As 5. |
| 14 | §4.3: which `all_special_ids` | applied | MarinTokenizer set from `_collect_special_ids` (tokenizers.py:750-768, property 680-682); decode with `skip_special_tokens=False` (638-639); `<unk:i>` for missing ids (670-673). |
| 15 | §3.3: masked per_pos_loss and aux_loss (duplicate of 7) | applied | As 7; `aux_loss` is the integer 0 from `split_activations` (lm_model.py:207-209). |
| 16 | §6: hook_time at N-1 excludes the final eval | applied | trainer.py:625-626 runs the forced hooks outside `capture_time` (559-566); 130m step 4958 hook_time 0.0079 s vs eval/total_time 21.6 s confirmed. |
| 17 | §1/§7: replayed steps in resumed segments | applied, segment count corrected | Temporary checkpoints every 10 min and the tracker's step drop (wandb.py:136-158) confirm the mechanism. Local segment counts are 3, 11, 27 (not 31). The checkpoint step of each resume is not cached, so the size of each replay window is unknown. |
| 18 | §2.3: order ambiguity resolvable from code | applied | training.py:560 calls train_lm in process; data.py:307-315 inserts in list order; `_PALOMA_SUBSETS` is byte-sorted. The local config.yaml order agrees but alone proves little (YAML writers may sort keys); the code argument carries it. |

Merge cross-check (2026-10-24): values stated in two or more sections were compared; the corrections are listed in `kaiyue_baseline.md` §C.
