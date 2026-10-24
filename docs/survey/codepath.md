# Code path of one training step and one run (today's stack)

Survey dossier, 2026-10-24. Read-only survey of `project/marin` (branch `vocab-overfitting`, HEAD 1793624070) and
`project/marin-pls` (branch `pls`, HEAD 684b48ead8). Paths below are relative to `project/marin/` unless they start
with `marin-pls/`. `lev/` = `lib/levanter/src/levanter/`, `hax/` = `lib/haliax/src/haliax/`, `mar/` = `lib/marin/src/marin/`.
Every number carries a source. ESTIMATE marks a number that is derived, not measured.

Raw artifacts of this survey (reusable):
- `marincpp/cache/survey_codepath/logscan.tsv`: 1,383 training logs (marin/logs/slurm + marin-pls/logs) scanned for CE
  implementation, fallback reason, first-step time, trace/lower time (script `scan_logs.py` beside it).
- `marincpp/cache/survey_codepath/import_loc_marin.json`: every module a run imports, with line counts (script `import_loc.py`).
- W&B history used here was already cached by another agent: `marincpp/cache/wandb/run-20260914_*-muonh-qwen3-*-della4xh100.jsonl.gz`.

## 0. Summary

| Fact | Value | Source |
|---|---|---|
| Steady step time, 130m / 300m / 520m / 1.2B, 4xH100, seq 4096 | 0.631 / 1.445 / 6.77 / 8.23 s | W&B `throughput/duration` median, cached history `run-20260914_041824/050240/083457/064730` |
| Levanter MFU at the same runs | 14.4 / 15.3 / 15.8 / 27.8 % | same files, `throughput/mfu` median |
| Eval (Paloma, 16 sets) per 1000 steps | 21-23 s (130m), 43-45 s (300m), 98 s (520m) | same files, `eval/total_time` |
| Checkpoint stall at 1.2B (10-min temp ckpt) | 6.7-6.9 s every ~73 steps (~1.1 %, ESTIMATE from 6.8/(73x8.23)) | same file, `throughput/hook_time` > 5 s |
| First train step (compile included), median | 11-13 s (130m/300m main), 19 s (1.2B), 85 s (pls 130m d48) | logscan.tsv, `First train step completed in` |
| Persistent JAX compilation cache | not used (dir never created) | section 13 |
| Python import of launcher | 39.4 s on della-vis1 (nice 19); 6,252 files / 3.0 M lines imported, torch included | `who_imports.py`, `import_loc_marin.json` |
| Framework LoC a run imports (levanter + haliax + marin) | 52,108 + 16,250 + 6,886 = 75,244 lines (plus 41,863 in iris/zephyr/rigging/finelog/finestore/fray) | import_loc_marin.json |
| Research-specific LoC (all variants, launchers excluded) | ~6.6 K lines | section 15 |
| Layers a research change crosses (launcher env to GPU op) | 6-9 per change type | section 14 |

The four largest code-path pitfalls (details in each section):
1. **Attention is not a kernel.** `JAX_FLASH` is a pure-JAX double `while_loop` ("Crappy pure-jax Flash Attention impl",
   `lev/models/flash_attention.py:48`), block 1024, sequence must be a multiple of the block. Smaller blocks multiply loop
   trips (Tokenizer session: 2.5-2.7 s/step at SEQ_LEN 4224/4480 vs 1.2-1.4 at 4096, needs file).
2. **Two CE backward paths chosen by a hidden shape rule.** H100, bf16, vocab >= 65,536 and >= 8,192 tokens per device
   take the b-tiled `batched_xla` path; everything else (PDP=1 at seq 4096, every vocab < 64K) is rejected by a
   shared-memory guard and silently falls back to the `xla` streaming path with a different backward
   (`batched_xla.py:369-389`, `api.py:159-173, 747-752`). 1,016 of 1,383 logs ran batched_xla alone, 361 logged the
   fallback warning (logscan.tsv).
3. **No compile cache, compile repeated every segment.** `JAX_COMPILATION_CACHE_DIR` is put into `os.environ` after
   JAX has read it (section 13); Triton/CUDA caches live in the per-job TMPDIR that is deleted at exit; CE autotune is
   disabled because its cache key cannot be built.
4. **Every step syncs the host.** `loss = result.loss.item()` (`lev/trainer.py:548`) blocks before the next step is
   dispatched; metrics go through `jax.device_get` (`lev/tracker/tracker.py:42`).

## 1. One run, end to end (call chain)

```
sbatch scripts/della/muonh_qwen3_h100x4.sbatch                     (33 lines)
  source job_tmpdir.sh (TMPDIR per job, deleted at exit), wandb_guard.sh (offline if api.wandb.ai fails in 20 s)
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}  (:28)
  .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling
    module import: ~99 env knobs parsed at import time (:99-427), allocator env for 1_2b OV (:38-41),
      monkeypatch temp-ckpt path (:81-84), wandb artifacts disabled (:86-92), LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0 (:97)
    MEMSTATS thread (:842-868)
    muonh_qwen3_run(size) -> ArtifactStep(name=speedrun/<run_id>, version 2026.09.13)       (:443-839)
    marin.execution.lazy.run(step)                                     mar/execution/lazy.py:410-427
      _lower(): fingerprint of build_config(ctx) for every dep (17 deps: fineweb + 16 Paloma)   lazy.py:302-407
      StepRunner.run -> run_step: status file + lease lock (30 s heartbeat, 90 s stale)  step_runner.py:430-475,
                                                                       rigging/.../distributed_lock.py:35-36
        thread: fn(output_path) -> build_config(ctx) -> handle.run(config)
          remote(_run_size, cpu) -> Fray LocalClient.submit -> ThreadPoolExecutor thread   mar/execution/remote.py:66-88
            _run_size: resolve_training_env(); remote(run_levanter_train_lm, gpu)        launcher :438-440
              Fray LocalClient thread #2
                run_levanter_train_lm                                  mar/training/training.py:533-560
                  apply_output_path (ckpt base, temp base), _resolve_run_id, env -> os.environ.setdefault
                  levanter.main.train_lm.main(config)                  lev/main/train_lm.py:150-466
                    trainer.initialize: jax config, distributed (none), W&B init, logging   lev/trainer.py:1000-1027
                    optimizer = MuonHConfig.build(num_steps)
                    Trainer(...) as ctx: mesh + param axis mapping + tracker
                    train_set (permuted cache), tagged_eval_sets (Paloma)
                    trainer.initial_state: load latest ckpt from [base, temp] or init     trainer.py:443-517
                    hooks: Paloma eval every steps_per_eval, perf stats every step, iris status every 10
                    train_loader.iter_from_step(state.step); trainer.train(state, loader)   train_lm.py:455-463
```

Threads between `python -m` and the training loop: step-runner worker, Fray thread (cpu step), Fray thread (gpu step).
The log shows two `current_client: using LocalClient (fallback)` lines (logs/slurm/charwin-v8k_w34-charwin-v8k-full.log:34-35).

Measured startup of one 300m run (same log): step runner 11:27:56 -> W&B up 11:27:58-:00 -> caches loaded 11:28:02 ->
"No checkpoint found" 11:28:04 -> first batch 11:28:07 -> first step done 11:28:25 (17.9 s) -> second program traced
11:28:25-:27. About 30 s after imports, plus imports (39.4 s on a vis node; compute-node import not measured).

## 2. Launcher: `experiments/references/della_muonh_qwen3_scaling.py` (875 lines; pls copy 548)

| Item | Where | Note |
|---|---|---|
| Env knobs | :99-427 | 99 distinct `os.environ` keys (grep), 82 in `marin-pls/.../della_muonh_qwen3_scaling.py` |
| Guards on combinations | 37 `raise ValueError`, 60 `if/elif VARIANT` lines | each new variant adds env parse + tag + guard + model/opt branch |
| Size table | :426-431 | transcribed from the TPU originals; `PER_DEVICE_PARALLELISM = {"520m": 32, "1_2b": 32}` :435 |
| Step rescaling | :446-456 | TPP, VOCAB_K (token ratio file), BATCH/SEQ_LEN keep tokens fixed |
| Run id | :457-543 | `muonh-qwen3-{size}-della{N}x{dev}` + ~40 optional tags + CPT_TAG + RUN_TAG + `-smoke{n}` |
| Model | :546-664 | one `model_cls` per VARIANT; always `hybrid_norm=True`, `attn_backend=JAX_FLASH` (:659-662) |
| Flash block | :112-114 | 1024 if SEQ_LEN % 1024 == 0, else largest of 512/256/128 dividing it |
| Optimizer | :697-755 | one `opt_cls` per knob, `MuonHConfig(... weight_decay=0.1, coefficient_type="simple")` |
| Trainer | :771-824 | mp `p=f32,c=bfloat16`; ckpt every 10 min (60 for OV), keep every 10000 steps; mesh `data:-1`; `data_seed=42` |
| Output | `ArtifactStep(name=f"speedrun/{run_id}", version=VERSION)` :832-839 | path `$MARIN_PREFIX/speedrun/<run_id>/2026.09.13` |

Hazards visible in the code:
- Run identity = run id string. Output path, W&B run id and resume all key on it; two runs whose tags collide share a
  checkpoint dir. The code already works around one collision (`-fp8` from CVD_FROM_PASS=8 vs PRECISION=fp8, :684).
- A finished step (`SUCCESS` status file) is skipped on relaunch (`step_runner.py:375`); a killed run leaves a lease that
  blocks a new worker for up to 90 s (`distributed_lock.py:36`).
- Import-time side effects: allocator env (:38-41) must precede JAX backend init; anything that touches a device earlier
  silently ignores it. Same for the pls `PARAM_MAP` mem fraction (`marin-pls/...scaling.py:178-179`).
- `weight_decay=0.1` (:747) is never read by `MuonHConfig.build` (`lev/optim/muonh.py:54-104`): no decay on any group.
- Submit-shell leaks: the sbatch overrides only `XLA_PYTHON_CLIENT_MEM_FRACTION` (:28); the pls sbatch must `unset` 48
  variables (`marin-pls/scripts/della/pls_h100x8_single.sbatch:31-32`) because every knob is an env var.
- `XLA_FLAGS` from the submit shell suppresses marin's NCCL watchdog flag: the flag is added to the env dict, then copied
  with `os.environ.setdefault` (`mar/training/training.py:430-436, 523-530`).

## 3. Marin executor layer (`mar/execution/*`, 2,971 lines in the directory)

- `ArtifactStep` holds `build_config`, `run`, `deps`; `lower()` hashes `canonical_json(build_config(ctx_for_fingerprint))`
  (`lazy.py:213-225`). The fingerprint is a drift check only; the address is the hash-free `{prefix}/{name}/{version}`
  (`lazy.py:227-237`). Changing a config without changing run id or VERSION reuses the old output dir.
- `run()` lowers every dep (17 here) and checks each dep's status on disk (log lines 13-29: "Skip ... already succeeded").
- `remote()` wraps a callable as a Fray `JobRequest`; with no cluster, Fray's `LocalClient` runs it in a thread
  (`lib/fray/src/fray/local_backend.py:107-117`). The launcher nests two of them (`:836` cpu, `:440` gpu).
- `run_levanter_train_lm` (`training.py:533-560`): sets checkpoint base `<out>/checkpoints` and temp base
  `marin_temp_bucket(...)` = `$MARIN_PREFIX/tmp/checkpoints-temp/<absolute out path>/checkpoints` (`training.py:154-198`,
  `rigging/filesystem/cluster_config.py` fallback `{marin_prefix}/tmp/{prefix}`), run id from `env_vars["RUN_ID"]`
  (`training.py:226-244`), JAX compile cache dir, tensorstore curl watchdog, NCCL watchdog 600 s (`training.py:412-479`).
- Value for the research loop: provenance record, dep skip. Cost: 3 extra threads, ~7 K lines imported, run identity in
  a string, and an env-mutation step that runs after JAX import (section 13).

## 4. Levanter Trainer (`lev/trainer.py`, 1,213 lines; `trainer_state.py` 273; `grad_accum.py` 184)

Per step (`train_step`, :527-568):
1. `hooks_this_time = any(step % h.every == 0 for h in jit_hooks)` (:534). The default `WatchConfig` is a jit hook with
   `interval=10` (`lev/callbacks/watch.py:116-132`), so two programs are compiled: one with parameter/gradient/update
   norms (every 10th step) and one without. The log names are swapped: the hook program is logged as `train_step`, the
   plain one as `train_step_hooks` (:541-546).
2. `named_jit(self._train_step, axis_resources=param mapping, donate_args=(True,))` (:724-740): state donated.
3. `_train_step_body` (:749-787): split key; `eqx.filter_value_and_grad(loss_fn)`; `microbatched(...)` if
   `per_device_parallelism` is set (:789-814); `state.take_step(grads)` (optax update + apply, EMA if set,
   `trainer_state.py:126-152`); `hax.shard(new_state, param mapping)`; jit hooks under `lax.cond` (:165-178).
4. Host: `loss = result.loss.item()` (:548) **blocks every step**; NaN/Inf check; Python hooks (pbar, step info,
   checkpointer, perf stats, eval); `levanter.tracker.log(...)` (:566).
- Measured host overhead between steps: `throughput/hook_time` median 7.5-8.8 ms, `throughput/loading_time` 0.5-1.3 ms
  (cached W&B, 4 baselines). ESTIMATE: the serialized gap costs ~1.3 % at 130m (8 ms of 631 ms) and less at larger sizes.

Microbatching / grad accumulation (`grad_accum.py:31-167`): `microbatch_size = PDP x data_axis_size`
(`trainer.py:986-990`). Inputs reshape to `(accum_step, microbatch)`; `hax.fold` (= `lax.scan`) over microbatches;
accumulator is re-sharded to the parameter mapping each microbatch (:148), so under FSDP every microbatch does an
all-gather of weights and a reduce-scatter of gradients (ESTIMATE from the sharding constraints, not profiled). Loss and
grads are averaged over microbatches (:157-162): equal to a token mean only when every microbatch carries the same weight
sum (true here: only the last position is masked).

Mesh and axis mapping (`lev/utils/mesh.py:16-186`): ICI axes `{data:-1, replica:1, model:1}`, batch ->
`(replica_dcn, replica, data)`, shared `{mlp: model, heads: model}` (size 1), **param mapping default `{embed: data}`**
(:40), i.e. ZeRO-3/FSDP even for 130m on 4 GPUs. The launcher adds `token`/`token_repeat` -> data (:803-806). pls
replaces it with `PARAM_MAP=rep|vocab` because FSDP-sharded 48-head programs gave program-dependent readout losses
(`marin-pls/...scaling.py:166-169`, job 15255032).

Defaults with cost: `log_jaxprs=True`, `log_xla_hlo=True` (:885-888): both programs are traced to jaxpr and lowered to
StableHLO a second time each segment and written to `logs/<run_id>/artifacts` in the repo cwd (:816-852). Measured
trace+lower median 2.7 s (300m main) and up to 346 s (pls) per log (logscan.tsv); pls reports 3.5-4 min per segment
(needs file).

## 5. Haliax layer (16,250 lines imported of 16,285)

- `NamedArray` = jax array + axis tuple (`hax/core.py:198`); ops resolve axes by name at trace time.
- `named_jit` (`hax/partitioning.py:373-498`) derives `in/out_shardings` from the `ResourceMapping` and caches pjit
  wrappers (`_named_pjit_cache`, :627). `hax.shard` = `with_sharding_constraint` from axis names (:206-268). An axis
  name absent from the mapping is replicated silently (e.g. `vocab` in compute: the lm_head is all-gathered).
- `Stacked` (`hax/nn/scan.py`, `_src/scan.py`) = `lax.scan` over layers with a remat policy. `gradient_checkpointing=True`
  is the Llama/Qwen3 default (`lev/models/llama.py:80`) and maps to `ScanCheckpointPolicy(simple=True)`: save layer
  outputs, recompute layer internals (`hax/_src/scan.py:145-171`). ESTIMATE: +1 forward per layer, ~+33 % FLOPs. At 300m
  the device holds 15.2 GiB of a 71.3 GiB limit (MEMSTATS, logs/slurm/charwin-v8k_w34-charwin-v8k-full.log:135), so the
  recompute buys memory that is not needed at small sizes.
- Cost to research: every op goes through axis-name resolution (tracing time grows with layer count: pls 130m d48 first
  step 85 s median vs 11 s for the 6-layer 130m, logscan.tsv), and sharding is implicit in names.

## 6. Model: Qwen3 in levanter (`lev/models/qwen.py` 396, `llama.py` 665, `lm_model.py` 354)

- `Qwen3Config(LlamaConfig)` (`qwen.py:297-376`): Llama geometry + QK-norm (`attention_config` adds `qk_norm`, :374-376).
  `Qwen3LMHeadModel` = `LlamaLMHeadModel` (`qwen.py:384-396`): `LlamaTransformer` of `Stacked` decoder layers,
  `LlamaEmbedding`, separate `hnn.Linear` lm_head unless tied. `hybrid_norm=True` from the launcher adds post-attn/MLP norms
  (`llama.py:224, 323`).
- Forward: `embeddings.embed` -> `layers.fold` (scan) -> final norm (`llama.py:393-411`). Loss:
  `compute_next_token_loss` -> `maybe_fused_next_token_loss` (`lm_model.py:316-354`, `loss.py:35-98`).
- **Init pitfall (correctness, effect not measured):** the embedding and the untied lm_head are initialised from the same
  key `k_emb` (`qwen.py:389-395`, also `llama.py:533-539`), with the same call `hax.random.truncated_normal(key,
  (Vocab, Embed), -3, 3)` (`hax/nn/embedding.py:65`, `hax/nn/linear.py:74`). So at step 0 `W_head = c * W_embed`
  exactly: the model starts with a copy-the-current-token bias. Worth one A/B in the rewrite.
- `Vocab` is rounded up for partitioning (`train_lm.py:250-253`).
- Research variants subclass the config (draccus `@LmConfig.register_subclass`) and override
  `compute_next_token_loss` or `activations` (e.g. `marin-pls/.../per_layer_qwen3.py:177-179, 572-635`).

## 7. Attention kernel used on GPU

- Dispatch: `dot_product_attention` (`lev/layers/attention.py:93-300`). Default backend on GPU is NVTE
  (`attention.py:82-89`); Della has no Transformer Engine, so the launcher forces `JAX_FLASH` (launcher :660-661), which
  falls through to `levanter.models.flash_attention.flash_attention` (`attention.py:280-300`).
- Implementation (`lev/models/flash_attention.py`, 471 lines): "Crappy pure-jax Flash Attention impl, vaguely following
  the v2 paper" (:48). `BLOCK_SIZE = 1024` (:23). Forward = `lax.while_loop` over query blocks, inner `while_loop` over key
  blocks up to the diagonal (`j_end = min(i+1, Tc)` if causal, :279), online softmax; backward is a second pure-JAX
  double loop (:303-450) under `equinox.filter_custom_vjp`. Each block pair is separate XLA GEMMs; nothing fuses QK, softmax
  and PV into one kernel.
- Constraints: Q and K length must be multiples of the block (:165-169, error "q axis size 4480 is not a multiple of
  1024"); a sequence shorter than one block drops to VANILLA O(S^2) (:63-84). If `attn_backend` is None and the block does
  not divide, the dispatcher silently picks VANILLA (`attention.py:161-167`).
- Block pairs per sequence: 4096/1024 -> 4 blocks -> 10 causal pairs; 4480/128 -> 35 blocks -> 630 pairs (ESTIMATE,
  arithmetic). Matches the 2x step-time loss at SEQ_LEN 4224/4480 reported by the Tokenizer session (needs file).
- Document masking is on: segment ids from EOS (`lev/data/text/datasets.py:755`, `examples.py:166-171`); the loop still
  visits every causal block pair and materializes the mask slice per pair (`flash_attention.py:246-253, 457-464`), so
  cross-document blocks are computed and masked, not skipped.
- With layer remat on, attention forward runs twice per step plus its backward (ESTIMATE from the remat policy).

## 8. Fused cross-entropy (`lev/kernels/pallas/fused_cross_entropy_loss/`, 4,280 lines)

Entry: `loss.py:207-299` flattens `(batch, position)` to B rows and calls the kernel API inside `shard_map` (:286-291),
so B = tokens per device per microbatch = PDP x SEQ_LEN.

Implementation selection (`api.py:629-825`): on GPU the order is `("batched_xla", "xla")` (`api.py:159-173`).
`LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` (launcher :97) means inferred block sizes, no sweep. Then:

| Condition (per device) | Path | Forward | Backward | Source |
|---|---|---|---|---|
| H100, x and w bf16, V >= 65,536, B >= 8,192, no argmax | `batched_xla` full-vocab b-tiled | `fori_loop` over 8,192-row tiles, reference CE on a full `[8192, V]` f32 logits tile | `_backward_b_tiled_from_lse`: recompute f32 logits, `probs*g`, scatter `-g` at label, cast dlogits to bf16, two bf16 GEMMs, f32 accumulators | `batched_xla.py:22-34, 317-366, 369-389, 552-631, 770-806` |
| otherwise | `batched_xla` streaming | rejected before launch: default tile 512 x 1024 x 2 B = 1,048,576 B > 101,376 B shared-memory limit, raises `BatchedXlaUnsupportedError` | — | `batched_xla.py:175-208` |
| after that rejection | `xla` (RuntimeWarning "falling back to XLA") | streaming over vocab blocks | `_linear_softmax_cross_entropy_loss_streaming_bwd`: f32 dlogits, `.at[].add` scatter, f32 GEMMs (fast bf16 backward is off by default) | `api.py:747-752`, `xla.py:25-35, 280-396, 626-677` |

Consequences:
- **PDP=1 vs PDP>=2 (128K vocab, seq 4096):** B = 4,096 < 8,192 takes the `xla` path; B >= 8,192 takes the b-tiled path.
  The two backwards differ in GEMM precision (f32 vs bf16 inputs) and op order, so gradients differ numerically. This is
  the undocumented difference the pls session found (needs file). Every vocabulary below 65,536 (8K-64K, bytes) always
  takes `xla`.
- Measured across 1,383 logs (logscan.tsv): 128K-vocab runs selected `batched_xla` alone in 1,016 logs; 361 logs show
  both implementations or the fallback warning (some runs mix paths between train and eval/readout programs); every
  v8k/v16k/v32k/v64k/bytes log used `xla` after the warning (e.g. 48 logs at 300m v8k).
- Memory: one b-tiled tile holds 8,192 x 128,256 x 4 B = 3.9 GiB of f32 logits; the backward also builds probs and
  dlogits of that shape (ESTIMATE ~12 GiB transient per tile). This is why the 47 extra readout heads of pls are
  bandwidth-bound (needs file: 12.3 of 15.2 s/step).
- Autotune cannot cache: the key needs the kernel jaxpr, which is unavailable, so every segment would re-sweep 7
  candidates (~10 min at 1.2B, launcher comment :94-96; `api.py:234-264, 469-480`).
- `precision` comes in as None from `loss.py`, while the batched_xla public default is `HIGHEST` (`batched_xla.py:857-866`).

## 9. MuonH / AdamH (`lev/optim/muonh.py` 188, `adamh.py` 161, `util.py` 322, `config.py` 584)

- `MuonHConfig.build` (`muonh.py:54-104`): `optax.inject_hyperparams` over `optax.multi_transform` with three groups
  chosen by **path string** (`create_mask`, :106-127): `"Embedding"` in path -> Adam (`adam_lr`), `"lm_head"` -> AdamH,
  other `hnn.Linear` weights -> MuonH, everything else (norm gains, biases) -> Adam.
- MuonH update (:136-188): Nesterov momentum; `zeropower_via_newtonschulz5` per flattened 2-D matrix, vmapped over the
  layer axis (`util.py:158-243, 244-302`), sharding constraint `(None, ("data","model"))` (:281-282); then
  `norm_preserving_update`: step `lr * u * |p| / |u|`, re-normalize to `|p|` (`util.py:304-322`).
- AdamH (`adamh.py:105-161`): Adam moments, then the same norm-preserving step.
- Pitfalls:
  - `clip_by_global_norm(max_grad_norm)` is inside each of the three groups (`muonh.py:64-92`): three separate norms, not
    one global norm.
  - For a 2-D parameter (the lm_head `[Vocab, Embed]`) the norm is one Frobenius norm over the whole table
    (`util.py:312-315`). Freezing or slowing some rows changes the step of the others (Tokenizer session, needs file).
  - `weight_decay` is ignored (section 2).
  - Group assignment by substring of the module path: a new module whose path contains "Embedding" or "lm_head" changes
    optimizer silently.
- Research optimizer variants subclass `MuonHConfig` and override `build`/`create_mask` (e.g.
  `experiments/references/freeze_rows_muonh.py` 178 lines, `marin-pls/.../depth_arch_qwen3.py:724-887`).

## 10. Data loader

- Cache: levanter `TreeCache` = `JaggedArrayStore` on TensorStore (offsets + flat `input_ids`)
  (`lev/store/jagged_array.py:20-269`); prebuilt at `$MARIN_PREFIX/fineweb-edu-10B/2026.06.28/train/{input_ids,shard_ledger.json}`.
- Sequences: `TokenSeqDataset` cuts the flat token stream into fixed `seq_len` windows, index i -> tokens
  `[i*L, (i+1)*L)` (`datasets.py:78-112`); `CausalLmDataset` builds `LmExample` with EOS segment ids through an
  `eqx.filter_jit` on CPU per example, then `NamedLmDataset` another CPU jit (`datasets.py:141-171, 119-135`).
- Shuffle: launcher sets `permutation_type="linear"` (:761): `p(i) = (a*i + b) mod N`, a coprime to N, from the data
  key (`lev/data/_prp.py:92-150`), data key = `PRNGKey(data_seed=42)` (`train_lm.py:220-222`). Deterministic in
  (data_seed, N, seq_len).
- Repetition: `MixtureDataset` restart strategy uses `idx % len` (`lev/data/mixture.py:287-295`), so every repeated pass
  replays the same order; `max_train_batches` slices the first N permuted sequences (`datasets.py:858-866`). A new order
  per pass needs pre-materialized pass caches (`vocab_materialize_passes.py`, launcher :762-767).
- Loader: `DataLoader(max_buffered_batches=128, fetch_batch_size=32)` (`trainer.py:692-722`), background thread with
  asyncio, `jax.make_array_from_callback` per leaf to the batch sharding (`lev/data/loader.py:237-420`). Measured
  `throughput/loading_time` median 0.5-1.3 ms (cached W&B), one 2.0 s stall at the first batch (w34 log:112).
- Resume: `train_loader.iter_from_step(state.step)` (`train_lm.py:455-460`): exact order on resume if seq_len, batch
  schedule and data seed are unchanged. Forking at another step or rebuilding the unseen ranges has no API (needs file:
  re-implemented twice).

## 11. Checkpointing (`lev/checkpoint.py` 1,314, `tensorstore_serialization.py` 1,352)

- Format: TensorStore `zarr3` arrays on an `ocdbt` kvstore, `file` driver on Della (`tensorstore_serialization.py:48-49, 148`).
- Policy (launcher :796-801): temporary checkpoint every 10 min (60 min for OV), permanent every 10,000 steps (none for
  OV or TPP != 20), final at end. Temporary ones go to the temp base and the previous one is deleted after the new one
  commits (`checkpoint.py:578-651, 690-712`).
- Temp path: `$MARIN_PREFIX/tmp/checkpoints-temp/<absolute output path>/checkpoints/step-N` (w34 log:137); the launcher
  strips `file://` because tensorstore otherwise wrote `<cwd>/file:/...` (launcher :81-84).
- Save: stage to host (budget 16 GiB), async commit thread (w34 log:139-145). Measured 300m: 0.91 GiB in ~3 s
  (11:38:02 -> 11:38:05, w34 log:136-145); 1.2B: 6.7-6.9 s step stall every ~73 steps (cached W&B hook_time); pls 21 s
  every 10 min (needs file); OV 300m ~70 s per save (launcher comment :793-794).
- Resume: `initial_state` searches `[<out>/checkpoints, temp base]` for the newest complete step
  (`trainer.py:472-517`, `checkpoint.py:948, 1114-1160`); `load_checkpoint_or_initialize` restores the full state or
  initializes under `filter_eval_shape`. `initialize_from` / `allow_partial_checkpoint` give the CPT path (launcher :780-781).
- `Trainer.__exit__` blocks on the async commit (`trainer.py:413-441`).

## 12. Eval and logging

Eval (`lev/eval.py` 851): `cb_tagged_lm_evaluate` hooked every `steps_per_eval` (1000; `train_lm.py:354-372`), over 16
Paloma sets tokenized with the training tokenizer, `per_device_eval_parallelism = PDP` (w34 log:44). One jitted
`accum_for_batch` per batch (`eval.py:546-594`), then `.item()` reads on host (`eval.py:609-616`).
- **bpb is not exact:** bytes per token come from a per-token-type table (`_calculate_bytes_per_token_type`,
  `eval.py:182-192`), and bpb is a token-weighted mean of per-batch `loss/bytes` ratios (`eval.py:583-590`), not
  `sum(loss)/sum(bytes)`. Macro = mean of per-tag means. The Tokenizer session measured the W&B macro ~0.004 high
  (needs file).
- Cost: 21-23 s per eval at 130m (3.5 % of 630 s of training per 1000 steps, ESTIMATE), 43-45 s at 300m (3.0 %).
  Per-layer readout eval in pls: 9-13.5 min per 1000 steps (needs file).

Logging (`lev/tracker/*`, `callbacks/*`):
- Per step: `train/loss` and loss metrics, `throughput/{duration, tokens_per_second, mfu, gflops_per_second,
  hook_time, loading_time}`, `optim/{learning_rate, adam_lr}`, `run_progress` (cached W&B rows). Every 10 steps: the
  watch hook's per-parameter norms. Every 1000 steps: eval keys.
- Tracker = `CompositeTracker([wandb, telemetry])` (`trainer.py:855-866`); values go through `jax.device_get`
  (`lev/tracker/tracker.py:36-50`), W&B behind a background wrapper (`tracker/wandb.py:24`).
- Della specifics: artifacts disabled (launcher :86-92); offline fallback when the proxy fails (`wandb_guard.sh`).
- MFU is levanter's: `flops_per_token` (`qwen.py:115-125`, `lm_flops_per_token`) x 3 x tokens, remat recompute not
  counted.
- Per-layer activation/gradient statistics: only the watch hook's parameter-level norms exist; per-layer residual reads
  need a separate probe program (needs file).

## 13. Compile cache and per-segment fixed cost

- `resolve_training_env` puts `JAX_COMPILATION_CACHE_DIR = $MARIN_PREFIX/tmp/compilation-cache` into the env dict
  (`training.py:472-476`), and `run_levanter_train_lm` copies it into `os.environ` (`training.py:559`). JAX reads the
  variable once, when `jax._src.config` is imported (`optional_string_state(... 'jax_compilation_cache_dir')`,
  `.venv/.../jax/_src/config.py:1626`, defaults read with `os.getenv` at definition, :507-906). The launcher imports JAX
  long before (`haliax`, `levanter` at :43-60). `TrainerConfig.jax_compilation_cache_dir` is None (`trainer.py:957`).
  Evidence: `marin_store_big/tmp/` holds `checkpoints-temp`, `levanter_kernel_autotune`, `zephyr` and no
  `compilation-cache` (ls, 2026-10-24). So **no persistent XLA cache is ever written**.
- Triton and CUDA caches point into the per-job TMPDIR (`scripts/della/job_tmpdir.sh`), deleted at exit.
- CuTeDSL cache "installed at .../tmp/cutlass-kernels" (w34 log:42); the directory does not exist (ls).
- Measured first step: 11-13 s (130m/300m main), 24 s (520m), 19 s (1.2B), 85 s (pls 130m d48), p90 up to 204 s
  (logscan.tsv). Plus the second program, jaxpr/HLO logging, and in pls a 30-step smoke per resume segment (~20 min,
  needs file).

## 14. Layers a research change crosses

| Change | Layers (outer to inner) | Count |
|---|---|---|
| New model variant (e.g. dropout, composed vocab) | env parse + tag + guard (launcher) -> `model_cls/model_extra` -> `LmConfig` subclass (draccus registry) -> `LmHeadModel.init` -> `Stacked`/`BlockSeq` (haliax scan + remat) -> layer `__call__` -> haliax ops | 7 |
| New loss / readout | launcher -> config subclass -> `compute_next_token_loss` override -> `maybe_fused_next_token_loss` -> named `fused_cross_entropy...` (shard_map) -> kernel API dispatcher -> `batched_xla` or `xla` custom_vjp | 7 |
| New optimizer rule (freeze rows, LR mult) | launcher -> `opt_cls/opt_extra` -> `OptimizerConfig` subclass -> `build` -> `optax.multi_transform` + path-string mask -> `scale_with_muonh`/`scale_by_adamh` -> `map_flattened_linear_layers` -> `norm_preserving_update` | 8 |
| Attention change | launcher (block) -> `Qwen3Config.attention_config` -> `Attention` -> `dot_product_attention` dispatcher -> `flash_attention` -> `_flash_attention` custom_vjp -> forward/backward while-loops | 7 |
| Data order change | launcher -> `mixture()` (marin) -> `LmDataConfig` -> `train_sets` -> `PermutationDataset`/`MixtureDataset` -> `TokenSeqDataset` -> `TreeCache` | 7 |
| In-run metric | launcher -> loss returns metrics dict -> `WrappedLossFunction` auto-wrap -> microbatch `fold_metric` -> `TrainStepResult` -> `tracker.log` -> `device_get` | 6 |
| Run plumbing (any) | sbatch -> launcher module -> `ArtifactStep` -> `lazy.run` -> `StepRunner` -> Fray thread -> `_run_size` -> Fray thread -> `run_levanter_train_lm` -> `train_lm.main` -> `Trainer` | 9 |

The pls session counts 5 edit sites per new switch (config field, per_layer_qwen3.py, launcher env/tag/model_extra,
sbatch unset list, fetchable sbatch design) plus a CPU test (needs file).

## 15. Lines of code

Imported by a typical run (`import_loc.py`: import the launcher, `levanter.main.train_lm` and the lazily used modules;
`import_loc_marin.json`):

| Group | Files | Lines | Code lines |
|---|---|---|---|
| levanter | 146 | 52,108 | 41,260 |
| haliax | 47 | 16,250 | 12,205 |
| marin | 33 | 6,886 | 5,397 |
| iris + zephyr + rigging + finelog + finestore + fray + dupekit | 151 | 41,909 | 33,089 |
| experiments (launcher + variant modules) | 21 | 4,694 | 3,730 |
| site-packages + stdlib (torch 537 K, sympy 292 K, jax 211 K, pandas 205 K, ...) | ~5,850 | ~2.88 M | — |
| Total | 6,252 | 3,001,202 | — |

On disk (all `.py`, tests excluded): levanter 84,484; haliax 16,285; marin 57,832; iris 90,365; rigging 13,672;
zephyr 10,259; finelog 10,826; finestore 3,043; fray 2,699 (vis-node `find | wc`).

Hot path of one step (files read in this survey): levanter trainer/model/attention/optim/data/ckpt/eval 19,161 +
CE kernels 3,532 + haliax core/partitioning/scan/nn 6,169 + marin training/execution 2,866 = ~31.7 K lines.

Heavy imports that the training step never uses: torch via `transformers/__init__.py` (989 files), sympy via torch,
datasets + polars via `levanter/data/sharded_datasource.py:15`, pandas via wandb, IPython via `tqdm_loggable`
(`who_imports.py` output).

Research-specific code:

| Module | Lines | Repo |
|---|---|---|
| depth_arch_qwen3.py | 887 | marin-pls |
| per_layer_qwen3.py | 806 | marin-pls |
| objective_qwen3.py | 674 | both |
| sampled_softmax_qwen3.py | 286 | both |
| over_vocab_qwen3.py | 375 (224 in pls) | marin |
| vocab modules: table_variants 493, vocab_materialize_passes 274, composed_vocab 168, vocab_refit 163, vocab_temp_losses 147, small_vocab_tokenizer 139, vocab_token_losses 135, vocab_transplant_losses 128, convert_small_vocab_cache 111, vocab_subset_losses 44, vocab_cut_expand_map 27 | 1,829 | marin |
| OV evals: ov_memorization_eval 216, ov_rowfreq 114 | 330 | marin |
| optimizer variants: freeze_rows 178, decay_to_pieces 77, lazy_embed 60, embed_wd 52, lr_mult 40 | 407 | marin |
| other variants: full_bandwidth 260, dropout 175, fbt_pass_eval 172, focal 117 | 724 | marin |
| **Sum (launchers excluded)** | **~6,320 + completed_adamh 311 = ~6.6 K** | |
| Launchers | 875 + 548 | marin, marin-pls |

Ratio: ~6.6 K research lines ride on ~117 K in-repo framework lines imported (75 K levanter+haliax+marin), about 1:18;
on the ~31.7 K hot path, about 1:5.

## 16. Implications for the rewrite (from the code path only)

1. A real fused attention kernel for any SEQ_LEN (multiple of 64/128), with document-aware block skipping.
2. One CE implementation with one backward for every (PDP, vocab), bf16 GEMMs, f32 softmax, chunked so that logits never
   materialize at [8192, V] f32; it must also serve per-layer readout heads.
3. A persistent compile cache that is configured before JAX import, and kernel caches outside the per-job TMPDIR.
4. No per-step host sync: read the loss of step N-1 after dispatching step N; log asynchronously.
5. Plain data parallelism (replicated params) as default at <= 1.2B on one node; FSDP only by choice.
6. Remat off by default when memory allows; memory budget printed before step 1.
7. Exact bpb in-run (sum of loss over sum of bytes, real byte lengths of the decoded text).
8. Optimizer groups by explicit parameter tags, true global clipping, per-row/per-block norm options.
9. Data order as a pure function (seed, pass, index) with a per-pass permutation and a fork-at-step API.
10. Run identity from a config hash plus a human tag, not a 40-tag string; config as a file, not 99 env vars.
11. Init the untied head and embedding from independent keys (or measure the current coupling first).
