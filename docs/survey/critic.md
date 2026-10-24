# Completeness critic of the survey dossiers

Date: 2026-10-24. I read all 7 dossiers in `marincpp/docs/survey/` and checked each disputed number at its cited source.
This was read-only. I submitted no job and ran no Python on the login node (only `jq`, `grep` and `sacct`).
New raw data from this pass: `marincpp/cache/critic/wq_pieces.tsv` (98 wq run pieces taken from `wq/jobs/*/state.json` `history[]`).
Short names: MB = measured_bottlenecks.md, CP = codepath.md, RF = roofline.md, EX = external_stacks.md,
RO = reliability_ops.md, RN = research_needs.md, NS = needs_from_sessions.md. "MB:129" means line 129 of that file.

## 0. Verdict in six lines

1. Attention is the largest lever for dense runs, and EX:153 is wrong when it says the gain at seq 4096 is "a few percent". It mixed up the loop overhead (about 20 ms) with the total attention time (about 0.70 s of a 1.44 s step at 300m).
2. The dense 128K runs use the `batched_xla` CE path, not the XLA fallback. MB:129 and MB:204 are wrong. CP:219 is right.
3. A persistent JAX compile cache does exist. CP:23 and CP:326 are wrong. MB:217 is right, but the cache is now 32 GiB, not 18 GB.
4. RF's model of the current CE cost is 50-57% too high at 130m and 300m when checked against measured dense step times. So RF:232 ("the LM head is the largest single lever") holds for pls but not for dense runs.
5. No dossier measured the one fact that decides JAX against PyTorch: a fast attention kernel at head_dim 64 with document masking on H100. In JAX, the FA4 native SM90 backward runs only for GQA with hd128 (`grug/attention/_fa4_cute_backend.py:165`). That covers 1.2B and none of 130m/300m/520m/pls.
6. Ten smoke-size experiments (§3, about 20 GPU-h in total, about 4% of the pli-cp 30-day quota) are enough to choose a design route.

## 1. Contradictions between dossiers, checked at the source

| # | Claim A | Claim B | Checked at | Verdict |
|---|---|---|---|---|
| X1 | EX:153: "Seq 4096: only a few percent; the loops cost ~1.5%, and attention is ~16% of FLOPs at 300m" | MB:128: attention is 0.67-0.73 s of the 1.444 s 300m step (46-50%) | Speed journal, agent `find:attention` (a28a7a5d): "Attention is about 0.70 s of the 1.25 s bytes step at 4096 (about 55%)", expected FA4 gain −40% at 4096. Agent `find:parallelism-xla` (af9ca2db): the "~1.5%" is only the per-iteration loop overhead, 360 iterations × ~55 µs ≈ 20 ms. MB's fit (bytes runs: 1.215 s at S=4096, 3.40 s at S=16384, same tokens per step) predicts 18.74 s at S=19456; the measured value is 18.51-18.55 s. | **MB is right.** EX quoted the loop-overhead line as the whole attention cost. Attention takes 16% of the FLOPs but 46-55% of the time, because JAX_FLASH reaches about 4-8% of peak (MB:136). |
| X2 | MB:129 and MB:204 (#4): the dense 128K LM head runs "via the XLA fallback (fused kernel never launches)" | CP:219-230: V ≥ 65,536 with B ≥ 8,192 per device takes `batched_xla` b-tiled; only V < 64K or PDP=1 falls back | `marin/logs/slurm/muonh-qwen3-300m_13852135.out:111` and `muonh-qwen3-130m_13846675.out:111`: "selected implementation: batched_xla". Speed journal `find:head-loss` (a3ca29d3) agrees: V ≤ 64256 → `xla`, V = 128256 → `batched_xla`. | **CP is right.** The fallback applies to the 8K-64K and bytes runs (all of family 10/11 in RN) and to PDP=1. Bottleneck #4 still holds as "head + CE not fused", but its label and mechanism are wrong for 128K. |
| X3 | CP:23 and CP:320-326: "no persistent XLA cache is ever written" | MB:217: "persistent cache on (18 GB in cache/jax)" | `~/.bashrc:58` exports `JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax"`, with `XDG_CACHE_HOME=$SCRATCH_USER/cache` (`:27`). JAX reads it at import. `$ROOT/cache/jax` holds 3,797 entries, 1,116 of them `jit__train_step-*`, and 32 GiB (`du`, 2026-10-24). Entries are dated up to 2026-10-24 13:16. | **MB is right; CP is right only about levanter's own setting**, which arrives too late. The cache works because of the login env, which a self-contained fetchable job inherits through `~/.bashrc`. Cold and warm compile times were never measured separately (see E10). The cache is shared with other projects and has no GC. |
| X4 | RF:85 and RF:144: the current CE costs 2,270 µs/1k tok at 130m (= 0.298 s/step/GPU at 131,072 tok) and 2,596 µs/1k at 300m (= 0.340 s) | MB:120, MB:129, MB:139: measured head + CE = 0.190 s (130m, 0.630 − 0.440) and 0.228 s (300m, 1.443 − 1.215), from W&B medians over vocab-sweep runs | MB's numbers are differences of measured medians. RF's are model output. RF's total current-CE model at 130m is 3,135 µs/1k = 0.411 s, while the measured step is 0.6305 s. | **The measured numbers are right.** RF overestimates the dense head by 57% (130m) and 49% (300m), and underestimates the rest of the step by about 0.33 s at 130m. RF calibrates well only on pls (0.297 predicted vs 0.262 s measured per head). Consequence: at 130m attention (0.258 s) costs more than the head (0.190 s), so RF:232 overstates the head lever for dense runs. |
| X5 | RF:181-197: the current CE temp is 11.7 GiB (8192×V f32 tile × 3 live); CP:231 gives "~12 GiB transient per tile" (ESTIMATE) | MB:102: the 130m peak is 8.50 GiB in total | `marin/logs/slurm/vo130_15168741-128k-rep8.log` MEMSTATS: `peak_bytes_in_use=8.50 largest_alloc_size=7.83 GiB`. The whole XLA temp of the step is 7.8 GiB. | **The measured number is right.** The CE temp is at most 7.8 GiB, so at most about 2 f32 tiles are live, not 3. The RF memory table (activations 3.6 + CE 11.7 + state 0.55 = 15.9 GiB predicted against 8.5 measured) is about 2× high at 130m. Treat its remat-off fit claims (RF:194) as unverified. |
| X6 | MB:206 (#6): 8 GPUs lose 13-20% per GPU because of FSDP collectives (69/72 exposed) | RF:215: communication is at most 0.6% of step compute with DDP. Speed journal af9ca2db: collectives are about 1% at 4 GPUs | The 8-GPU rows (MB:106-108) are `-pls0-noev` runs (`wandb-runs/wandb/run-20261015_200043-...-della8xh100-pls0-noev`). PLS_W=0 adds stop-gradient readouts of every layer on every 16th position (`marin-pls/.../della_muonh_qwen3_scaling.py:203-206`), and eval is off. The 4-GPU rows are plain baselines with eval on. At 8 GPUs the per-GPU batch halves, so fixed per-step costs (replicated NS A·A, `RF:212`; host hooks) weigh twice as much. | **Unresolved; the comparison is confounded.** Neither "FSDP collectives" nor "≤ 0.6%" has been measured. The d48 "perfect 4→8 scaling" (MB:112) also compares two different variants (preln-sep w89 against gilns w15). See E4. |
| X7 | EX:11-12 and EX:132: 300m runs at 1.2-1.4 s/step, i.e. 101k tok/s/GPU and ~12% MFU | MB:103: 300m 128K runs at 1.444 s/step, 90.8k tok/s/GPU, levanter MFU 15.27%, true-causal 13.15% | NS:9 says "1.2-1.4 at 4096" for the Tokenizer runs, which are v8k-v64k (journal: v64k 1.4 s/it, v8k 1.2 s/it). The 128K run is W&B run-20260914_050240 at 1.444 s. | **MB is right for 128K.** EX applied a small-vocab step time to the 128K model. |
| X8 | EX:132 ("levanter, current fork … pls d48 27%") | MB:111: the logged levanter MFU of pls d48 preln sep is 18.76% | NS:31: 263 TFLOP/s (27%) is the readout heads' rate alone, counted as 8hV. RF:144 gives 197 TFLOP/s at 6hV. | **MB is right.** 27% is a component rate, not stack MFU. |
| X9 | RF:141 and RF:150: the d48 backbone takes 2.9 s, "inferred" as 15.2 − 12.3 | MB:155: 2.9 s measured | `marin-pls/logs/muonh-qwen3-130m-della4xh100-d48-diag600-pdp4.log.log`: tqdm 2.8-3.1 s/it | **Measured (MB).** RF's 1.21 s model is 2.4× below a measured value, not an inferred one. The gap of about 1.7 s at width 512 × 48 layers is the largest unattributed term in the survey (E3). |
| X10 | RO:78: resume overhead is ≈ 14% of a 3 h piece (ESTIMATE) | MB:184, MB:209: pls smoke 1.6% of the run; dense startup 46-138 s per segment | wq history (`cache/critic/wq_pieces.tsv`, n=98 pieces since 2026-10-21): median 4.98 h, p90 13.5 h; 22 pieces < 0.5 h (quick fails); pieces stopped with exit 124/143: n=30, median 6.9 h. | **Both are right at different scopes.** 14% holds only for pls (20 min smoke) at exactly 3 h pieces. At the real median piece of about 5-7 h, pls pays 6-8% and dense runs about 1-3%. It matters, but less than the step-time terms. |
| X11 | RF:134: 40-50% levanter-convention MFU is "about the top of what published stacks reach" | EX:76, EX:70, EX:85: levanter upstream 56.2% (2.7B, FA4 hd128); MaxText ~54% (ESTIMATE); llm.c 47.5% at 1.6B | Sources as cited in EX | **RF claims too low a ceiling for d ≥ 2048.** At d512-d1024 with hd64 there is no published anchor (EX:61, "small-model throughput is not published"). Treat the 130m-520m ceiling (36-41% true-causal) as an ESTIMATE that has no external check. |
| X12 | MB:82: the pls profiler probes 16144708/16144709 are PENDING | — | `sacct -j 16144708,16144709`: both CANCELLED, never started | **No H100 profile exists or is queued.** E3 must supply it. |
| X13 | RO:11: "nothing cached under marincpp/cache/wandb/" | MB:72: 75 run dirs are cached there | Timing: RO was written before MB | Stale; later agents should use MB's cache. |
| X14 | RN:100: 1.2B step "~8.2 s (ESTIMATE from SS:110-112)" | MB:105: 8.221 s measured (27 segments) | W&B medians | Use MB's measured value. |

The small differences in step medians (1.2B: 8.17 RF from yaml:97, 8.221 MB, 8.23 CP; 520m: 6.73/6.741/6.77) are below 1% and come from different run sets. They do not matter.

## 2. Load-bearing claims that are ESTIMATE or have no source

| # | Claim the design leans on | Where | Why it is weak | What settles it |
|---|---|---|---|---|
| U1 | A fast attention kernel exists for hd64 with document masking on H100, in the chosen framework | EX:150-154, MB:201, RF:17 (FA3 hd64 at 0.45 fwd / 0.35 bwd) | All timings are from A40/A100 (journal a28a7a5d). In JAX, FA4's native SM90 backward needs `qhead_per_kvhead > 1 and head_dim == 128` (`lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py:165`). The 130m (8/8 heads), 300m (12/12) and 520m (16/8, hd64) runs all take the generic segmented backward, which upstream calls "~40× slower per call" (EX:79). FA3 at hd64 has never run on Della (nanoswe used hd128). | E1 |
| U2 | A fused CE reaches about 50% of peak at h = 512-768 and V = 128K, which would take pls 15.2 → 6.2-7.8 s | RF:140-145 (6.17 s), EX:147 (7.8 s) | The two dossiers assume different FLOP counts (8hV against 6hV) and different efficiencies. The 50% anchor (#6979) is at a larger width. A logits GEMM with K = 512 is near the H100 ridge point (RF:107). The owner's Pallas GPU CE ran 8-9× slower than XLA (NS:47). | E2 |
| U3 | The step-compute ceiling is 2.6-2.8× (130m-520m) and 1.8× (1.2B) | RF:122-125, RF:228 | Every efficiency factor is an ESTIMATE (RF:11-22). The model misses 130m by 35% overall (X4) and the d48 backbone by 2.4× (X9). | E3, E6, E7 |
| U4 | Turning remat off gains 10-20% and fits in memory | RF:194, EX:155-157, MB:203 | The gains come from non-H100 runs (`scratch-memory/raw2.log`). RF's memory model is 2× high at 130m (X5). | E5 |
| U5 | 8-GPU runs lose 13-20% per GPU, and the cause is FSDP collectives | MB:206 | The comparison is confounded (X6). | E4 |
| U6 | The 520m "anomaly" (82% of the 1.2B step time at 47% of its FLOPs) comes from attention, 3.2-3.8 s | MB:143, RF:79 | No 520m sequence sweep and no profile. Hypothesis (ESTIMATE): JAX_FLASH cost scales with layer × head count at hd64 (520m: 24×16 = 384 head-layers; 1.2B: 16×16 = 256). | E3 |
| U7 | Removing launch and host overhead is worth 5-8% at 4096 | EX:158-163 | Command buffers gave about 2× on a contended A40 at seq 4224, "direction only". No GPU-idle fraction measured on H100. | E3, E9 |
| U8 | Deterministic mode is cheap enough to be the default (RO R9) | RO:152, RO:189 | Never measured (`objective-hillclimb.md:76`). Probes 14183277-280 were held and never run. | E8 |
| U9 | Monthly GPU-h lost per bottleneck, which ranks the work | MB:199-210 | Every row is an ESTIMATE on an ESTIMATE fleet split. The ranking also depends on X1, X2 and X4. | Recompute after E1-E3 |
| U10 | The PyTorch route ports MuonH/AdamH in about 300 lines with exact parity, and reads the levanter caches in the exact order | EX:108, EX:126 | No parity test and no loader throughput measured. Both are CPU work. | Non-GPU checks N1, N2 |
| U11 | A torch.compile warm cache brings compile down to seconds | EX:117 | The source is a Diffusers blog post, not an LLM run on Della. The measured cold time is 113-130 s (nanoswe). | E10 |
| U12 | Async save-on-TERM fits in 60 s | RO:156 | Measured: host staging at 1.65-2.7 GiB/s (MB:184). GPFS write bandwidth per node was not measured. An OV checkpoint of 227 GB cannot fit 60 s at that rate. | E11 |
| U13 | AllToAll first-call abort at 8 GPUs is caused by a late compile on one thread | RO:65 | Marked ESTIMATE and not diagnosed | It is route-neutral: a design without AllToAll in the step avoids it. No experiment needed now. |

One cross-dossier observation (ESTIMATE, from measured inputs): at width 1536-2048 the current stack is close to the owner's PyTorch stack. 1.2B levanter runs at 251 TFLOP/s/GPU true-causal (RF:75). nanoswe d24 runs at 311 TFLOP/s 6N-only, with FP8 worth about 8.6% (EX:28, EX:31). So the gain of a rewrite sits at 130m-520m and in pls, where hd64 attention, K = 512 GEMMs and the CE dominate. These are the shapes that E1, E2, E6 and E7 must use.

## 3. Missing measurements that choose between design routes

Routes: **A** keep JAX and strip levanter; **B** fix levanter in place; **C** PyTorch rewrite (nanochat style); **D** PyTorch on torchtitan or Megatron-Core; **E** C++/CUDA core.
Owner scope (NS:56-58) rules out general frameworks, so D needs a strong measured advantage to stay in play.

| Exp | Decides | Routes split |
|---|---|---|
| E1 attention bake-off | whether JAX has an FA3-class kernel at hd64 with doc masking | A/B vs C/D |
| E2 fused-CE bake-off | whether JAX reaches Liger/Triton CE speed at h = 512 | A/B vs C/D (pls above all) |
| E3 H100 profile of today's step | where the unattributed 0.3-1.7 s goes; launch overhead | all; E vs C |
| E4 4/8 GPU × FSDP/DP | whether the default layout is DDP; whether 8 GPUs are worth it | all (layout default) |
| E5 remat policy | the memory and time of remat-off at each size | all (memory planner scope) |
| E6 minimal PyTorch step | what route C reaches; GPU idle without CUDA graphs | C vs A; C vs E |
| E7 minimal stripped-JAX step | what route A reaches with the best JAX kernels | A vs C; A vs B |
| E8 determinism cost | whether bitwise mode can be the default | all |
| E9 command buffers (+WHILE) | whether B gets a cheap 1-40% | B |
| E10 cold/warm compile | resume cost per piece | A vs C |
| E11 checkpoint I/O | the save-on-TERM design | all |

All experiments are smoke-size and fit pli-cp (8 h wall, ≤ 8 GPUs). No experiment edits an existing worktree. Experiments that need code changes run from a **new** worktree (e.g. `marin-critic-*` from `vocab-overfitting@1793624070` or `pls@684b48ead8`), never from a live checkout (RO E4).

### E1. Attention kernels at the owner's shapes (1 H100, ~25 min, ~0.4 GPU-h)
- Shapes, per-GPU microbatch: B=32, S=4096 with heads/kv/hd = 8/8/64 (130m), 12/12/64 (300m), 16/8/64 (520m), 16/8/128 (1.2B). Also 300m at S=4224 and 4480 (B=32), and bytes at S=19456 (B=4).
- Segment ids: from the first 32 sequences of the real permuted `fineweb-edu-10B` cache (EOS-derived), so that tile skipping matches real document lengths.
- Candidates:
  - JAX: `JAX_FLASH` (block 1024 / 128 / largest divisor); grug FA4 CuTe segmented (records which backward runs); cuDNN THD via `jax._src.cudnn.fused_attention_stablehlo` with `q_offsets`/`kv_offsets`; tokamax Mosaic SM90 (install on a vis node).
  - PyTorch: FA3 `flash_attn_varlen_func` (hopper build); FA4 `flash_attn.cute` (`flash_attn_4-4.0.0b16` is already in the marin venv); SDPA cuDNN.
- Record: fwd ms and fwd+bwd ms (median of 20 after 5 warm-up), TFLOP/s at causal-and-document FLOPs, peak memory, and max abs error of out/dq/dk/dv against an f32 reference.
- Also run a GEMM sweep in the same job (bf16, M = 131,072, K×N ∈ {512×1536, 512×1792, 768×2304, 768×2688, 1024×3584, 2048×7168, 512×128256, 768×128256}). It calibrates RF's 0.60-0.78 GEMM factors.
- Start from: `tmp/speed_wt/scratch-attention/bench_attn.py`.
- Decision rule: if no JAX kernel gets within 1.5× of FA3's fwd+bwd at hd64, routes A and B lose most of the attention lever (41-50% of the dense step, MB:128, MB:139).

### E2. Fused linear-CE (1 H100, ~25 min, ~0.4 GPU-h)
- Rows B ∈ {16,384 (pls PDP 4 microbatch per GPU), 131,072 (dense per GPU, chunked)}; h ∈ {512, 768, 1024, 2048}; V ∈ {128,256, 64,256, 8,256, 512}.
- Candidates:
  - levanter `batched_xla`, `xla` streaming, and `xla` with `LEVANTER_CE_XLA_FAST_BWD=1`;
  - a JAX chunked CE in Liger style (gradient in the forward, bf16 logits, 1 chunk per device);
  - Liger FLCE (Triton); `nanoswe/chunked_lce.py`; Apple CCE.
- pls proxy: 47 heads × (16,384 × 512 × 128,256) in sequence, plus the main head.
- Record: fwd+bwd ms, TFLOP/s at 6hV, peak memory, and loss/grad error against f32.
- Decision rule: this gives the pls step floor (RF:140 says 6.2 s; EX:147 says 7.8 s) and shows whether JAX matches Triton.

### E3. H100 profile of today's step (4 H100, ~45 min, ~3 GPU-h)
- Configs: 300m baseline (`SMOKE_STEPS=40`), 520m (`SMOKE_STEPS=20`), pls d48 preln sep PDP 4 (`SMOKE_STEPS=20`, `PARAM_MAP=vocab`).
- Code change: in a new worktree, add `profiler=ProfilerConfig(enabled=True, start_step=10, num_steps=5)` to the `TrainerConfig` (`lev/callbacks/profiler.py:78`; there is no launcher knob). Optionally also capture the window with `nsys` for kernel-level gaps.
- Record per step, by category: JAX_FLASH loops, CE, GEMMs (QKVO/MLP, with remat), norms/rope/elementwise, Newton-Schulz, exposed collectives, GPU idle (host gaps and loop predicates).
- What it settles: X6, U3, U6, U7, and the 1.7 s d48 backbone gap (X9).
- If GPU idle is above 10% of the step, a graph-captured design (CUDA graphs or command buffers) or route E has a real target. If it is below 5%, route E has none.

### E4. Layout and GPU count (8 H100, ~30 min, ~4 GPU-h)
- Runs: plain 300m baseline from a new worktree of `pls@684b48ead8` (`PARAM_MAP` exists there, `:166-187`, `:508`), no VARIANT, `PLS_EVAL` not applicable, eval off via `SMOKE_STEPS=60`.
- Arms, in sequence in one job: 4 GPU FSDP (default); 4 GPU `PARAM_MAP=rep`; 8 GPU FSDP; 8 GPU rep. Repeat the 8 GPU FSDP arm at 1.2B (`SMOKE_STEPS=30`).
- Record: median s/step over steps 20-60, tok/s/GPU, MEMSTATS peak and `largest_alloc_size`.
- What it settles: whether the 13-20% loss is layout (communication) or per-step fixed cost, which in turn sets the DDP default and the 4-GPU packing rule.

### E5. Remat policy (4 H100, ~40 min, ~2.7 GPU-h)
- Code change: in a new worktree, add a knob that sets `gradient_checkpointing` to False, or a haliax `ScanCheckpointPolicy` that saves the attention and MLP outputs.
- Arms: 300m (PDP 32) and 520m (PDP 16 and 32) × {full remat, save attn+MLP out, none}; `SMOKE_STEPS=40` each.
- Record: s/step, MEMSTATS peak, `largest_alloc_size`, and the compiled `memory_analysis()` temp size.
- What it settles: U4 and X5. It also gives real data points for a memory planner (RO R1).

### E6. Minimal PyTorch step (4 H100, ~40 min, ~2.7 GPU-h)
- Code: nanoswe's Qwen3 path re-shaped to 300m (d768, 12 layers, 12 heads hd64, inter 2688, V = 128,256, hybrid norm, qk-norm, untied).
- Setup: DDP, torch.compile, FA3 varlen with document masking, `chunked_lce`, plain Muon/AdamW (throughput only; exact MuonH comes later). Batch 128×4096 from the real cache, 50 steps.
- Arms: compile on/off; CUDA graphs on/off; remat off. Add the d48 width-512 shape with 1 head and with 48 heads (microbatch 4×4096, 8 accumulation steps).
- Record: s/step, tok/s/GPU, peak memory, cold compile and warm compile (Inductor/FX cache on scratch), and the GPU-idle fraction from `torch.profiler`.
- Optional extra (+2 GPU-h): torchtitan v0.3 Qwen3 at the same 300m shape. This is the only cheap evidence that would keep route D in play.

### E7. Minimal stripped-JAX step (4 H100, ~40 min, ~2.7 GPU-h)
- Code: `experiments/grug/base` (1,207 lines, plain JAX, explicit sharding) at the same 300m and d48 shapes as E6.
- Setup: the best JAX attention from E1, the best JAX CE from E2, remat off, replicated parameters. XLA flags: latency-hiding scheduler, pipelined collectives, `--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL`.
- Record: the same as E6, with cold/warm compile using the persistent cache.
- What it settles: E6 and E7 together are the head-to-head between routes A and C on equal kernels, measured against the RF realistic target (300m 0.54 s, RF:123).

### E8. Determinism cost (4 H100, ~20 min, ~1.3 GPU-h)
- Runs: 130m, `SMOKE_STEPS=30`.
- Arms: default and `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, each run twice. Then the E6 script with `torch.use_deterministic_algorithms(True)`.
- Record: s/step, and whether `train/loss` at steps 2 and 30 is bitwise equal between the two twin runs.
- What it settles: RO R9 (deterministic by default). It also sets the n per arm, because rerun noise is about equal to seed noise (RO B3).

### E9. Command buffers on H100 (4 H100, ~20 min): coordinate, do not duplicate
- The speed workflow already holds plans `fa4-ab-sl4224/sl4096` and branch `speed-command-buffer-while@fe5a177339` (journal aa17b1c4, a6f04078).
- Use its H100 A/B when it lands. If it does not run, the arms are charwin v64k at SEQ_LEN 4224 and the 4096 baseline, flag on and off, `SMOKE_STEPS=40`.

### E10. Cold and warm compile per route (folded into E3, E6, E7; +0.5 GPU-h)
- For each program, time to first step with an empty cache dir (`JAX_COMPILATION_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` pointed at a fresh scratch dir) and with a warm one.
- Programs: 300m and d48.
- What it settles: per-piece resume cost at the measured wq piece median of 5-7 h (X10), and U11.

### E11. Checkpoint I/O (4 H100, ~15 min, ~1 GPU-h)
- State: the pls d48 state (37 GiB) and a synthetic 1.2B OV state (65 GiB per GPU at 4 GPUs, RF:173).
- Arms: device→pinned-host copy time; tensorstore write (today's path); per-rank raw file writes; torch DCP async.
- Record: GiB/s per stage, step stall, and the time from TERM to a durable state.
- What it settles: U12.

### Non-GPU checks (CPU job or vis node; not pli-cp)
- **N1. Loader.** A PyTorch/NumPy reader of the zarr3 cache must reproduce levanter's linear permutation (`lev/data/_prp.py:92-150`, seed 42) bit-exactly for 10^4 indices, and sustain at least 3 M tok/s per node. 3 M tok/s is RF's 130m ceiling of 4 × 131,072 tok / 0.18 s, an ESTIMATE.
- **N2. Optimizer parity.** One-step MuonH/AdamH parity between levanter and the port on identical weights and gradients, on CPU.
- **N3.** Cache hit or miss per segment, from the existing logs (no new run).

Total GPU cost: E1-E11 ≈ 20-22 GPU-h. That is about 4% of the ~500 GPU-h per 30 days pli-cp quota (CLAUDE.md, Della). Order by value: E1, E2, E3 first. E6 and E7 depend on E1 and E2. E4, E5, E8, E10 and E11 are independent.

## 4. Smaller corrections for dossier owners
- MB:129 and MB:204: relabel 128K as `batched_xla` (X2). The `xla` fallback and the 8K→16384 padding bug apply to V < 64K.
- MB:217: the cache is 32 GiB and comes from `~/.bashrc`. It is not set by the stack, so a clean-env job without bashrc would lose it.
- MB:82: the profiler probes are CANCELLED.
- CP:23, CP:38-40 and CP:320-326: replace "no persistent cache" with "levanter's cache setting is dead; the bashrc env one works".
- RF:85-88, RF:181-197 and RF:232-233: recalibrate the dense CE and memory models against MB's measured 0.190/0.228 s and the 7.83 GiB temp. Label the d48 backbone 2.9 s as measured.
- EX:11-16, EX:132 and EX:153: use the 128K numbers (1.444 s, 90.8k tok/s/GPU). Drop the "few percent at 4096" claim (X1). The pls MFU is 18.76%, not 27%.
- RN:100: the 1.2B step time is measured (8.221 s, MB:105).
- RO:11: stale. RO:78: scope the 14% to pls with 3 h pieces (X10).
