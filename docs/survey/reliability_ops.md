# Reliability and operations on Della: failure catalogue and operating contract

Survey dossier for the marincpp rewrite. Written 2026-10-24. Read-only survey of `project/marin` (branch vocab-overfitting),
`project/marin-pls` (+ `marin-pls-*` worktrees), `tools/`, `wq/`, Slurm logs, and memory files.

Path abbreviations: `M` = `$ROOT/project/marin`, `P` = `.../project/marin-pls`,
`T` = `$ROOT/tools`, `MEM` = `/home/USER/.claude/projects/-home-USER/memory`,
`NEEDS` = `marincpp/docs/survey/needs_from_sessions.md`. Logs are `M/logs/slurm/*`, `P/logs/slurm/*`.
GPU-hours (GPUh) come from `sacct -X -o ElapsedRaw,AllocTRES` (queried 2026-10-24) unless marked; for jobs run inside a
pilot host (wq ids `wN`) they come from `$ROOT/wq/jobs/<n>/state.json` `history[]`.
No W&B history was fetched for this dossier (nothing cached under `marincpp/cache/wandb/`).

## 0. Headline

1. Most lost GPU time is **memory**: XLA temp buffers that are unknown before the first step, BFC fragmentation,
   and env-controlled pool fractions. 24 RESOURCE_EXHAUSTED logs in the pls tree alone (§1 row A1-A4).
2. The costliest single class is **silent wrong numerics**, not crashes: FSDP sharding of stacked layer weights gave
   wrong activations; the 520m post-hoc probes ran 83.2 GPUh before anyone saw it (row F1).
3. **Run identity is a string assembled from ~95 env knobs** (`M` launcher) and doubles as output path, W&B id and
   executor cache key. Collisions resume someone else's run; leaked knobs change the run; scripts carry `unset` lists
   of 40-50 names (rows E1-E3).
4. **W&B is not a reliable record**: proxy blocks uploads, proxy outages hang `wandb.init`, `finished` does not mean
   complete, history pagination drops/duplicates rows, summaries are read mid-run as finals (rows W1-W6).
5. **Resume works but is expensive and fragile**: temp checkpoints hide under a mirrored absolute path, killed saves
   leave half-written dirs, and each resume segment pays ~25 min of smoke + compile + HLO logging (rows C1-C5, §3).
6. The Della contract (wq/pilot hosts, `--partial` resumable work with 3 h minimum pieces, 90-min idle kill,
   self-contained scripts, offline nodes) is already enforced by tooling around the stack, not by the stack.
   A rewrite should own these contracts natively (§2, §4).

Sum of GPUh in the failed jobs listed in §1 with sacct/wq numbers: about **240 GPUh** (OOM ~65, diagnostics ~21,
FSDP-invalid 83, rendezvous 14, W&B hang 11, compile/NaN smokes ~30, node fault 6, chain-tail no-ops 6). For scale, the pls
project record is 2306.9 H100-h total (`P/logs/pls_depth_controls.yaml:203`). This is a lower bound: it excludes
forfeited host time, queue re-waits (~11-29 h each, §2) and human/agent diagnosis time.

## 1. Failure table

Columns: mode | evidence (job ids, file:line) | root cause | GPUh in failed jobs | current workaround.

### A. Memory

| # | Failure mode | Evidence | Root cause | GPUh | Workaround today |
|---|---|---|---|---|---|
| A1 | **BFC fragmentation OOM** mid-run (enough free memory, no contiguous block) | 1.2B ov `14794354` OOM 30.18 GiB at step ~1290; ovss `14794356` 25.08 GiB at step ~130 (`M/logs/slurm/ov-1_2b_14794354.out`, `ovss-1_2b_14794356.out`); diag `14824921` BFC OOM at step 119 every time (`M/docs/della/over-vocab.md:294-302`). pls d48: hc `15266153` 32.91 GiB at step 257 (`P/logs/pls_depth_controls.yaml:155`), preln `15266145` 30.61 GiB (`P/logs/slurm/pls-d48g4-preln_15266145_a.log:414`); NEEDS pls §3 | ~27 GiB persistent + one 25-30 GiB contiguous temp per step; a long-lived small block lands beside the temp region; "largest free block ~0.7 GiB short" (`over-vocab.md:297`) | 15.9 + 1.5 + 12.6 + 11.4 = 41.4; diag 13.2 | `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` + preallocated pool sized to temp (ov 0.8, ovss 0.75), set inside the launcher before JAX import (`M/experiments/references/della_muonh_qwen3_scaling.py:30-41`, commit 83270354c5). Cost: +1.4% stage-1 step, +4% over run; no-prealloc +30%, vmm +17% (`over-vocab.md:302-305`). pls: pool fraction 0.95 for PARAM_MAP (9642035a00) and microbatch 1 |
| A2 | **cuda_async aborts at exit** (rc 134 after success) | `over-vocab.md:308-309`; `15006934` FAILED rc 134 after both runs logged "succeeded", 38.7 GPUh (`over-vocab.md:230-232`; sacct); 7 logs contain "terminate called without an active exception" (grep of both log trees) | allocator teardown | 0 lost, but job state lies | humans read step status, not sacct state |
| A3 | **Temp buffer unknown until first step; microbatch non-monotone** | tempsize `15322826`: microbatch 2 is worst for every design (hc 110.27, mhc 110.30, attnres 117.34, attnres_block 114.09, moda 127.79 GiB) vs microbatch 4 hc 32.91, attnres 53.36; microbatch 1 mhc 17.69 GiB (`P/logs/pls_depth_controls.yaml:184`). MTP ladder 520m smoke asked 90.80 GiB (`:231`), 1_2b 65.84 GiB at step ~90 (`:232`); 1_2b probes 109.30 GiB at PDP 2 with "68.29 GiB XLA floor independent of PDP" (`:294`). 520m baseline 35.45 GiB at 64 seqs/GPU after a passing 40-step smoke (`M/logs/kaiyue_muonh_qwen3_h100x4.yaml:53`). 24 distinct pls logs with RESOURCE_EXHAUSTED, 6 in marin | XLA scheduling/rematerialisation decides temp size; no model of it; per-layer readouts keep all layer outputs; 31 probe heads all-gathered (~15 GiB) + 31 f32 dW live before reduce-scatter (`:294`) | 1.0 + 0.2 + 1.8 + 0.6 + 0.7 + 0.7 + 0.3 + 0.3 = 5.6; tempsize diag 7.6; lens-48b 9.1 | trial and error; `PARAM_MAP_PDP1` forcing microbatch 1 (25.4 vs 16.0 s/step for hc, NEEDS pls §2); `--xla_gpu_memory_limit_slop_factor=70` (`:295`); dedicated memtest/tempsize jobs on pli-cp |
| A4 | **Host OOM / compile blow-up** on unrolled graphs | `14962098` OUT_OF_MEMORY (oom_kill, 192G host) compiling attnres/mhc; moda compile timeout 25 min (`P/logs/pls_depth_controls.yaml:24`); `14969858` attnres compile > 60 min, moda 103 GiB temp (`:34`) | Python-loop unrolled O(L^2) graphs at 48 layers | 6.2 + 13.4 = 19.6 | rewrite as two-level layer scan (22486e9370) |
| A5 | **mem-fraction env leakage** | session shell exports `XLA_PYTHON_CLIENT_MEM_FRACTION=.50`; `sbatch --export=ALL` carries it (`MEM/reference_della_xla_mem_fraction_env.md:12-14`). `14983240` rep-300m-freeze: MEMSTATS `bytes_limit=39.59 GiB`, OOM 13.25 GiB (`M/logs/slurm/rep-300m-freeze_14983240.out`) | `${VAR:-0.9}` default lets the submit shell win | 2.7 | sbatch sets `XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}` (`M/scripts/della/muonh_qwen3_h100x4.sbatch:28`, commit 11075c775c); pair scripts clear `PARAM_MAP_MEM_FRACTION` |
| A6 | **GPU slot conflict** (two guests on one GPU set) | orphaned nested fetch loop in host 15625117 put pls jobs on GPUs another guest held; XLA preallocation CUDA OOM 71.26 GiB, exit 4 (`P/logs/pls_depth_controls.yaml:253-255`; `P/logs/slurm/miles-fcs_..._15625117_a-smoke.log`) | fetch loop survived TERM as orphan (ppid 1); guests saw host's job name, so logs collided | not in sacct (inside host) | fetch_pending: exit if `FETCHED_JOB_ID` set; TERM ends loop; `FETCHED_JOB_NAME` (`MEM/project_fetch_pending_pilot.md:170`) |

### B. Kernels and numerics

| # | Failure mode | Evidence | Root cause | GPUh | Workaround |
|---|---|---|---|---|---|
| B1 | **Flash kernel SEQ_LEN constraint** | w32-w34 smokes `ValueError: q axis size 4480/4992/5504 is not a multiple of 1024` (`M/logs/slurm/ov-smoke_15960540_*`; wq events 2026-10-23T11:22) | JAX flash tiles in 1024 blocks | 0.3 | 555a625346: largest power-of-2 block dividing SEQ_LEN (`launcher:109-114`); 4480 gets block 128 and runs 2.5-2.7 s/step vs 1.2-1.4 at 4096 (NEEDS TS §2), which then caused B2 |
| B2 | **Task limit shorter than run** (knock-on of B1) | w32 charwin-v32k ran 7.58 h on 8 GPUs, both tasks stopped by the plan's 450-min limit at step ~9787 of ~11444 (`M/logs/vocab_overfitting_h100x4.yaml:258`; wq/jobs/32/state.json) | limit sized from a sibling's speed, not this config's | 60.6 GPUh run, work kept in temp ckpts; loss ≈ requeue + re-smoke (ESTIMATE <1 h × 8 GPU) | `feedback_rightsize_slurm_time.md:214-217`; requeued as w125 |
| B3 | **Nondeterministic embedding scatter** | same-seed reruns differ at step 2 (train/loss 10.638720 vs 10.639905), never equal again over 4954 steps (`M/docs/della/objective-hillclimb.md:74`); rerun spread ~0.0005 bpb ≈ seed spread 0.0003-0.0004 (`:58`) | `haliax/nn/embedding.py:80` `weight.take` backward = `scatter-add[unique_indices=False]` → atomics on GPU (`objective-hillclimb.md:70`) | n/a (inflates n needed per arm) | none; `--xla_gpu_deterministic_ops` probes 14183277-280/686-687 were held and never run; throughput cost unmeasured (`objective-hillclimb.md:76, 516, 729`) |
| B4 | **FSDP layout changes results** (silent) | 48 readout losses vary run to run under FSDP sharding of stacked layer weights (`P/logs/pls_depth_controls.yaml:150`, job 15264999; 15255032 L47 8.07); 520m probes INVALID: train-path L23 9.8 nats vs eval 3.547 (`:297`) | FSDP sharding of stacked (scanned) weights in the 8-GPU train-step program | see F1 | `PARAM_MAP=vocab` (shard only vocab-sized tensors; 7b02c544f2); smoke gate "last readout = main eval" |
| B5 | **Two CE backward paths** | PDP=1 and PDP≥2 use different CE backward code, numerics differ, undocumented (NEEDS pls §3) | microbatch-dependent kernel selection | n/a | none |
| B6 | **NaN at step 1** from data transform | w4 inzero: loss NaN at step 8 (first step with zeroed rows); zero first token of a segment × hybrid_norm post-norm multiplies backward by 1/sqrt(eps) (`M/logs/vocab_overfitting_h100x4.yaml:248`); hc/mhc NaN from Adam eps 1e-20 on symmetric init (`P/logs/pls_depth_controls.yaml:29,32`) | numerics of rare inputs | 0.2 + 6.6 | per-case fixes 70c4d0edfa, 8e6240cb56; GPU debug smoke with JAX_DEBUG_NANS |

### C. Collectives, nodes, hardware

| # | Failure mode | Evidence | Root cause | GPUh | Workaround |
|---|---|---|---|---|---|
| C1 | **AllToAll first-call rendezvous abort at 8 GPU** | w16 hc-static, w17 attnres-static: `Termination timeout for [6] first call after collective operation: kind=kAllToAll ... of 40 seconds exceeded ... only 7 of them arrived`, rc 134 (`P/logs/slurm/arch2-hc-static_w16_smoke.log`; `P/logs/pls_depth_controls.yaml:302`); two different nodes | one device thread late to first AllToAll (likely long first-call compile/autotune on one thread; ESTIMATE, not diagnosed) | 0.9 + 0.8 | run those designs on 4 GPUs (54c9c685c8) |
| C2 | **Clique-acquire abort at 4 GPU** | `15266143` sandwich run a: `Acquire clique: devices=4 ... Termination timeout of 600 seconds`, rc 134 at ~step 260, partner run b killed by pair script (`P/logs/slurm/pls-d48g4-sandwich_15266143_a.log:274-303`, `.out` tail) | session attributes to memory fragmentation (NEEDS pls §3); NCCL communicator setup needs device memory (ESTIMATE) | 12.4 | resume in next chain job (15266144 resumed from 229) |
| C3 | **Slow / faulty node** | della-k16g1 GPU 2 thermally throttled to 345 MHz, 3.2x slower collectives (`T/pilot_host.sbatch:15`; pls yaml:296); della-k17g3 97 s/step vs 34 (`:226`, job 15520124); della-k14g3 GPU 0-3 7-13% slower (`M/docs/della/vocab-overfitting.md:166`); GPFS D-state stall 6 min (`M/logs/kaiyue_fbt_qwen3_h100x4.yaml:41`) | hardware | 6.1 (15520124) | hand-maintained `--exclude` lists |
| C4 | **90-min idle-GPU cancel** risk | CSES mail at 30 min of 0%, cancel at 90 min (`MEM/reference_della_idle_gpu_kill.md:123-125`); 3 causes: reserved idle GPUs, packed partner finished/crashed, crashed-but-not-exited (`:127-134`); 8-GPU jobs split because unequal halves idled ~45 min (`M/logs/vocab_overfitting_h100x4.yaml:186`) | job shape | not quantified | `wait -n`, lanes, `set -euo pipefail`, jobwatch.sh IDLE alerts |

### D. Checkpoint and resume

| # | Failure mode | Evidence | Root cause | GPUh | Workaround |
|---|---|---|---|---|---|
| D1 | **Temp checkpoint location** | resume ckpts at `marin_store_big/tmp/checkpoints-temp/<absolute output path>/checkpoints/step-N` (verified by `find`, 30 step dirs, 368 GiB, 2026-10-24); "hard to find" (NEEDS TS §3) | levanter `temporary_checkpoint_base_path` mirrors the full path | 0 | none |
| D2 | **file:// temp path bug** | "arrays land in <cwd>/file:/...; resume finds only metadata.json" (`launcher:81-84`) | tensorstore treats `file://` URL as relative path | (pre-fix runs) | monkeypatch strips `file://` |
| D3 | **Half-written checkpoints after kill** | 2 step dirs without `metadata.json` since 2026-09-30, still on disk: `.../muonh-qwen3-1_2b-della4xh100-ss.../step-11172` (4.4 GiB), `.../muonh-qwen3-300m-della4xh100-ov12.8m/step-7180` (43 KiB) (find, 2026-10-24). "Levanter never deletes it" (`MEM/project_ss_ov_kaiyue_baseline.md:421-422`) | save interrupted by time limit; no atomic commit, no GC | disk only | manual deletion |
| D4 | **Save stall** | 300m OV ~70 s per save, ~11% wall at 10-min interval (`over-vocab.md:313-314`); pls 21 s per 10 min, ~3.4% (NEEDS pls §2); OV ckpt 79/112/~227 GB at 300m/520m/1.2B (`over-vocab.md:312`); 1.2B OV final 210 GiB in 122 s (`over-vocab.md:289`); pls 130m (47 heads) temp ckpt 38 GiB (du, 2026-10-24) | synchronous staging to host | ~3-11% of run wall | OV saves hourly (50dfe21fc9); no step-interval keeps for OV/TPP≠20 (`launcher:791-801`) |
| D5 | **Resume segment overhead** | 30-step smoke ~20 min per resume segment; first-step compile 70 s; jaxpr/HLO logging 3.5-4 min per segment (NEEDS pls §2); fused-CE autotune ~10 min at 1_2b per segment before the fix (`launcher:94-97`) | no persistent compile/autotune cache across segments; smoke gate not resume-aware | per 3 h piece ≈ 25 min/180 min ≈ **14%** (ESTIMATE from those numbers) | `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0`; packed_job skips smokes passed in earlier runs of a wq id (`M/scripts/della/packed_job.sbatch:19-21`) |
| D6 | **Resume reads stale state** | resume segment's smoke gate read the first segment's `wandb-summary.json` (NEEDS pls §3); gate picks newest dir by `ls -td .../wandb/*-$smoke_id` (`P/scripts/della/pls_h100x8_single.sbatch:150-151`) | gate state keyed by a name reused across segments; read from W&B files, not the run | n/a (gates wrong value) | none yet ("just found") |
| D7 | **Stale fail markers** | markers `logs/pair_failed/*.fail` from 15832439/41 made 15907476 exit 6 at once (`P/logs/pls_depth_controls.yaml:296`); 2 markers still present 2026-10-22 (`ls P/logs/pair_failed`) | file-based gate state with no owner/expiry | ~0 | manual `mv` to `cleared/` |
| D8 | **Chain tails that only discover "done"** | deepnorm/keel/sandwich/preln tails ran ~12 min each (smoke, then executor "already succeeded"), ~1.6 H100-h each (`P/logs/pls_depth_controls.yaml:192-193`) | afterany chains sized before the speed was known; smoke runs before the done-check | ~6.4 | wq `--partial` replaces afterany chains (`T/wq/README.md`) |
| D9 | **Lost W&B row at a stop** | attnres row 4595 lost at the 15266160 stop; `LOGGING_GAPS` allow-list (`$ROOT/tmp/pls_blog/fetch_d48g4.py:32-34`) | metrics buffered, not flushed before kill | n/a | allow-list per run |
| D10 | **Fork from mid-run** | `INIT_FROM` full-state restore exists (`launcher:124-126`) but 272 GB of non-final step dirs deleted 2026-10-18, so "a rerun from a mid-training step now needs a fresh run" (`vocab-overfitting.md:150-153`); exact data-order fork reimplemented twice (`ov_memorization_eval.py`, `vocab_refit.py`; NEEDS TS §4) | no cheap mid-run snapshot (weights+data cursor only) | n/a | none |

### E. Run identity, environment, code provenance

| # | Failure mode | Evidence | Root cause | Workaround |
|---|---|---|---|---|
| E1 | **Run-id collision (RUN_TAG / CPT_TAG hazard)** | `CPT_TAG` is empty without `INIT_FROM`; a run that changes only `TOTAL_STEPS` gets a byte-identical run id and **resumes the baseline** (`objective-hillclimb.md:458, 689`); `RUN_TAG` exists only to rename (`launcher:188-191`); run id = f-string over size, GPUs, ~30 variant tags, CPT_TAG, RUN_TAG, smoke (`launcher:543`) | identity = name built from a subset of knobs; name = output path = W&B id = executor cache key | `run_checked.sh` dry-runs the launcher and refuses a mismatch (`M/scripts/della/run_checked.sh`) |
| E2 | **Env-knob explosion / leakage** | 95 distinct env knobs in `M` launcher (875 lines), 80 in `P` (548 lines) (grep, 2026-10-24); `packed_job.sbatch:31-32` unsets 47 names, `P/scripts/della/pls_h100x8_single.sbatch:31-32` unsets 44; pls: each new switch touches 5 places incl. the unset list (NEEDS pls §1); fetched job with `SINGLE` from submit env not fetchable (`P/logs/pls_depth_controls.yaml:238`) | config is process env; Slurm `--export=ALL` default | denylist `unset`, `env -i` submits, run-id dry-run check |
| E3 | **Run-id check bugs kill jobs in seconds** | w27/w28 exit 2 in 26 s: check accepted only `della8xh100` (`P/logs/pls_depth_controls.yaml:304`); PROBE_FROM name check refused `-noev` drop, exit 2 in 20 s (`:247`); lens run-id parse bug left 21 `-lensat` dirs, 11.6 GB (`:278`) | string parsing of identity | patch per case; 2 quick fails ended pilot 15910552 early (guard raised 2→3, `T/fetch_pending.sh` MAXFAIL comment). Host 15910552 ran 16.36 h of 24 (sacct): ≤ 7.6 h × 8 GPU forfeited (ESTIMATE upper bound) |
| E4 | **Live-import hazard** | queued chain segments import the launcher from the checkout at start: "queued jobs read its launcher Python live" (`MEM/project_pls_per_layer_supervision_closed.md`); hc fix "read live, so it was in effect" (`P/logs/pls_depth_controls.yaml:156`); main repo edit-frozen ~7 h to protect a queued segment; `_editable_impl_marin_root.pth` puts the main repo on `sys.path`, only `python -m` cwd ordering saved the worktree run (`objective-hillclimb.md:690`) | editable installs + no snapshot of code at submit; Slurm snapshots only the batch script (`MEM/reference_slurm_sbatch_snapshot_at_submit.md`) | freeze edits, branch+worktree per experiment |
| E5 | **Wrapper scripts edited while running** | bash reads `.sh` by offset; in-place edit killed flashreinforce 14952937 (`MEM/reference_slurm_sbatch_snapshot_at_submit.md:82`) | non-atomic edit | write new file + `mv` |
| E6 | **Shell glue fails silently** | w15 passed smoke then died: `ls` of two globs exits 1 under pipefail (`P/logs/pls_depth_controls.yaml:303`, fix dfd3dcd028) | bash as control plane | `|| true` |
| E7 | **Offline-node surprises** | tiny smoke 13812723 died in 37 s: tokenizer not in HF cache (`M/logs/tiny_h100x1_smoke.yaml:13`); vo-passes 15390501 "No cache available", HFValidationError (`M/logs/slurm/vo-passes_*`) | HF not on proxy allowlist | `HF_HUB_OFFLINE=1`, prefetch on vis node |

### W. W&B

| # | Failure mode | Evidence | Root cause | Workaround |
|---|---|---|---|---|
| W1 | **Uploads blocked, finish() hangs** | proxy passes `api.wandb.ai`, not `storage.googleapis.com`; `run.finish()` hangs, run shows "crashed", GPUs held to walltime (`MEM/reference_della_wandb_proxy_uploads.md:30-35`) | Della proxy allowlist | `WANDB_IGNORE_GLOBS='*'` etc. + no-op `Run.log_artifact` (`launcher:86-92`) |
| W2 | **Proxy outage hangs init** | 2026-10-16 ~22:00: 15690438, 15694961 smokes hung in `wandb.init` 40 min, rc 124 (`P/logs/pls_depth_controls.yaml:258`) | no timeout on tracker init | 10.8 GPUh; `wandb_guard.sh` probes `api.wandb.ai/healthz` 20 s, else offline + ledger (`M/scripts/della/wandb_guard.sh`); 19 offline runs listed for manual `wandb sync` (`P/logs/wandb_offline.txt`) |
| W3 | **"finished" but truncated** | hc run died at step 259 and shows `finished` (`tmp/pls_blog/fetch_d48g4.py:9`); `crashed` means "segment hit wall clock", not failure (`M/logs/kaiyue_fbt_qwen3_h100x4.yaml:79`) | W&B state reflects process exit, not schedule completion | readers check last step == n−1, gaps == 0 (`fetch_d48g4.py:75-78`) |
| W4 | **History pagination** | one `scan_history` over a run drops and repeats a row at each page boundary; read in 1000-step windows (`fetch_d48g4.py:10-11, 36-40`); intermediate Paloma evals logged twice in scan_history (`MEM/project_llm_objective_hillclimb.md`) | W&B API | windowed reads + dedupe |
| W5 | **Mid-run summary read as final** | seeds 2/3 summaries read while training → whole comparison column flipped sign (`objective-hillclimb.md:448`) | summary = latest value | `objective_compare.py` admits a run only at `num_train_steps−1` |
| W6 | **API enumeration hangs; offline logs unreadable while written** | `api.projects`, `api.viewer` > 120 s; `api.runs` over marin-community SIGKILLed at 300 s (`MEM/reference_marin_wandb_project.md:61`; `M/logs/kaiyue_muonh_qwen3_h100x4.yaml:102`); "record checksum is invalid" for live offline runs (`reference_marin_wandb_project.md:63`) | API scale; binary transaction log | explicit run paths; on-disk history cache (CLAUDE.md: ~15 min to re-read ~200 runs) |

### F. Silent correctness (highest cost per incident)

| # | Failure | Evidence | GPUh | Detection |
|---|---|---|---|---|
| F1 | 520m post-hoc probes wrong (FSDP stacked weights) | 15699612 35.4, 15716064 35.4, 15738657 6.1, 15738658 6.3 GPUh, all INVALID (`P/logs/pls_depth_controls.yaml:297`; sacct) | **83.2** | found days later by a frozen-head consistency check; now a smoke gate |
| F2 | W&B mid-run summary used as final (W5) | `objective-hillclimb.md:448` | 0 | manual |
| F3 | packed ledger lost lines on Slurm stop | `vocab-overfitting.md:140` | accounting only | TERM trap added in `T/packed_task.sh:28-30` |

### S. Scheduling / control-plane

| # | Failure mode | Evidence | Workaround |
|---|---|---|---|
| S1 | Array rollover stuck in JobArrayTaskLimit | 6 times over the 1.2B 2 h chain; ~70 min idle once (`M/logs/kaiyue_muonh_qwen3_h100x4.yaml:74, 81-93`) | re-apply `ArrayTaskThrottle`; superseded by wq |
| S2 | afterany chain segments restart at AGE 0 | `MEM/reference_della_queue_priority.md:268` | fewer, longer segments; wq `--partial` |
| S3 | Queue wait dominates | 4-GPU median queue 29 h, 8-GPU 11 h (2026-10-07, `P/logs/pls_depth_controls.yaml:74`); we got 44 GPUh per job start vs 156 for a 24 h-job user (`MEM/project_fetch_pending_pilot.md:179`) | pilot hosts + wq |
| S4 | wq event log duplicates lines | first 3 END lines appear twice (`$ROOT/wq/events.log:1-6`) | none (minor) |

## 2. Della operating contract (what any rewrite must satisfy)

Sources: `T/wq/README.md`, `T/fetch_pending.sh:1-40`, `T/pilot_host.sbatch`, `T/packed_task.sh`, `M/scripts/della/job_tmpdir.sh`,
`M/scripts/della/wandb_guard.sh`, CLAUDE.md (Della section), `MEM/reference_della_queue_priority.md`, `MEM/reference_della_idle_gpu_kill.md`.

| Area | Requirement |
|---|---|
| Scheduling unit | GPU work goes through `wq submit` (`T/wq/README.md`); pilot hosts (8 GPU, 24 h, `-c 96`, 900G, pli-short) run wq jobs on disjoint GPU subsets. 6 pilots stay queued, named `pack-w<a>-w<b>`. Multi-node, >8 GPU, pli-cp smoke, A100, CPU stay on `sbatch` |
| Job shape | one node, 1-8 GPUs; `--time` ≤ 23 h so it fits a fresh pilot; on <8 GPUs no `srun`, no literal `CUDA_VISIBLE_DEVICES=` |
| Resumability | every job > 3 h is `--partial` (owner 2026-10-23): may be run for ≥ 3 h pieces (`MINPART=10800`, `T/fetch_pending.sh:44`) until the host ends; exit 0 = done; 99/124/143 = requeue, up to 30 runs. Must checkpoint and resume from the last checkpoint on every start |
| Stop signal | host stop arrives as TERM (via `timeout`, `--kill-after=60`, `T/packed_task.sh:26`) → ≤ 60 s to persist state and exit 124/143 |
| Self-contained | runs in a fresh login env (`~/.bashrc`), not the submit shell; nothing from `--export`/`VAR=... sbatch`; `#SBATCH` lines other than gres/time/mem/J/o/e/D/x ignored; script snapshotted at submit |
| Identity in host | `SLURM_JOB_ID` is the host's; per-job paths key on `${FETCHED_JOB_ID:-$SLURM_JOB_ID}` (= `w<n>`); `WQ_JOB_ID`, `FETCHED_JOB_NAME` set |
| TMPDIR | per-job dir `$ROOT/tmp/job/<id>`, removed at exit; Triton/CUDA caches and `APPTAINERENV_TMPDIR` there (`job_tmpdir.sh`). Node `/tmp` is small and shared |
| Network | compute nodes offline; `module load proxy/default` allowlists W&B API (and LLM APIs) only, not huggingface.co, not storage.googleapis.com. All data, tokenizers, eval sets pre-staged on scratch from a vis node. Tracker must degrade to offline within seconds |
| GPU utilisation | any GPU at 0% for 30 min → mail; 90 min → host cancelled (whole host, all guests). Repeated underutilisation can bring a 7-day ban (`T/fetch_pending.sh:44` comment). Evals, compile, data prep must not idle GPUs that long; packed halves must end within ~60 min of each other (`BAL=3600`) |
| Walls | pli-short 24 h; pilots 24 h; hosts reserve 10 min margin (`MARGIN=600`). Maintenance windows block jobs that cross them (next: 2026-11-10 06:00-18:00, `MEM/project_fetch_pending_pilot.md:181`) |
| Packing | 8-GPU host = one 8-GPU job or two 4-GPU jobs of similar length; 8-GPU jobs start sooner (11 h vs 29 h median, 2026-10-07). Ledger: `packed_task.sh` per task line for GPUh accounting |
| Caps | pli-c 64 GPUs per user across QoS; MaxJobsAccruePU 5; backfill tries top 3 jobs per partition; usage limit 12k GPUh per user, 15-day half-life (~23 GPUs sustained, admins 2026-10-23); pli-cp ~500 GPUh / 30 days, smoke only; CPU mem soft cap 115G/GPU |
| Login nodes | edit, git, squeue/sacct/sbatch only; analysis, W&B reads, builds go to vis nodes or CPU jobs |
| Disk | GROUP fileset alert at 95% (55 TiB); hit 96% on 2026-09-30 (`launcher:792`) |

## 3. Checkpoint / resume requirements

| Requirement | Why (evidence) | Target |
|---|---|---|
| **Bitwise data-order resume** | Data order is already a closed-form function: `data_seed=42`, LCG permutation `(a·index+b) mod N`, batch filled by global index (`objective-hillclimb.md:68`). Resume is order-exact; arithmetic is not (B3) | Keep: data cursor = (seed, global step) only; no loader state in checkpoint. Expose the index→example map as a library call so evals can rebuild seen/unseen ranges (NEEDS TS §4: reimplemented twice) |
| **Deterministic option** | same-seed reruns diverge at step 2; rerun noise ≈ seed noise (B3) | deterministic embedding backward (sorted segment-sum, not atomic scatter) on by default; measure its cost once |
| **Fork from mid-run** | INIT_FROM exists but mid-run steps are deleted for disk (D10) | cheap "fork points": weights + optimizer at chosen steps, or weights-only + data cursor; child run id derived from parent id + diff, never typed by hand (E1) |
| **Atomic commit** | half-written dirs persist (D3) | write to `step-N.tmp/`, fsync, rename; GC incomplete dirs on start |
| **Discoverable location** | resume ckpts hidden under mirrored absolute path (D1) | one run directory holds config, code hash, checkpoints (temp + kept), metrics log, events |
| **Save on TERM within 60 s** | hosts stop guests with TERM and 60 s kill-after (§2); temp ckpt every 10 min loses ≤ 10 min (avg ~5 min, ESTIMATE) per stop; 300m 2 h segments redid ~140 steps / 3.5 min (`kaiyue_muonh_qwen3_h100x4.yaml:57`) | async snapshot to host RAM then background write; pls 38 GiB ckpt at 21 s save fits 60 s; OV 227 GB does not (save would need ≥ 122 s, D4) → shard writes per GPU, or periodic only |
| **Save stall** | 3.4% (pls) to 11% (300m OV) of wall (D4) | < 1%: device→host copy overlapped with next step; disk write off the critical path |
| **Size** | 38 GiB temp ckpt for a 130m model with 47 heads; OV 79-227 GB (D4) | optimizer state dominates; per-tensor policy (e.g. no Adam state for frozen tables) |
| **Resume overhead ≤ 5% of a 3 h piece** | today ≈ 14% (smoke 20 min + compile 70 s + HLO logs 4 min, D5) | persistent compile + autotune cache keyed by program hash; smoke only on first start of a run id; no jaxpr/HLO dumps by default |
| **Done-check before work** | chain tails spent ~12 min / 1.6 GPUh discovering "already succeeded" (D8) | check completion marker before any GPU init |
| **Metrics durable across stops** | W&B row lost at stop (D9); truncated runs look finished (W3) | local append-only metrics file flushed every step, fsync at checkpoint; W&B is a mirror |

Measured anchors: 1.2B restore 12.62 GiB across 50 arrays in 3.1 s, first step 14.1 s (`M/logs/kaiyue_muonh_qwen3_h100x4.yaml:96`).

## 4. Design rules that would have prevented each failure class

| Rule | Prevents |
|---|---|
| R1. **Memory plan before launch**: compile the train step on the target mesh (CPU AOT or one GPU dry compile) and report persistent + peak temp per device; refuse to launch if > budget; pick microbatch by search over this report, not by trial jobs | A1, A3, A4 (OOM smokes, tempsize/memtest jobs ~35 GPUh) |
| R2. **Allocator owned by the trainer**: one allocator policy (cuda_async + prealloc pool sized from R1, or a static arena), set in code before backend init, logged with `bytes_limit` at step 0; exit status independent of allocator teardown | A1, A2, A5 |
| R3. **No config from process env**: one typed config file per run, checked into the run dir; the job script passes only its path; unknown env vars that the stack reads are an error | A5, E2, E7 |
| R4. **Run identity = hash of resolved config + code hash**, human name is a label; collisions impossible; resume only if the stored hash matches; fork = new hash with explicit parent | E1, E3, D6, D7 |
| R5. **Code snapshot at submit**: wheel/zip of the source tree (or git commit + immutable worktree) copied into the run dir; jobs never import from a live checkout | E4, E5 |
| R6. **Control plane in a real language with tests**, not bash: pre-flight (paths, caches, run id, memory plan) runs on CPU at submit; job-level exit codes are an enum (done / stopped-resumable / failed-deterministic / failed-infra) | E3, E6, B2, D8, W3 |
| R7. **Kernel shape contracts checked at config time**: attention supports any SEQ_LEN that is a multiple of 128 at full speed (masked tail block), or config validation rejects it before queueing | B1, B2 |
| R8. **Step-time-based limits**: per-task limits derived from the measured s/step of the same config (first 50 steps) × remaining steps × 1.25, recomputed at each resume; stop cleanly before the limit with a checkpoint | B2, D5 |
| R9. **Deterministic by default** (sorted scatter for embedding grad; fixed reduction order); a determinism smoke (two short runs bit-equal) in CI | B3, B5 |
| R10. **Sharding-invariance test**: every model/layout change runs a 2-step loss comparison 1 GPU vs N GPU vs each param mapping on CPU devices; any diff > tolerance blocks the run | B4, F1 |
| R11. **Collective warm-up**: run every distinct collective once at init (barrier-protected) before the first step; raise XLA rendezvous termination timeout for first calls; or prefer layouts with no AllToAll inside the step | C1, C2 |
| R12. **Node health probe at start**: 30 s matmul + all-reduce benchmark per GPU; refuse (exit infra, requeue elsewhere) if any GPU < 80% of fleet median; record node in run dir | C3 |
| R13. **Atomic checkpoints in the run dir**: tmp-dir + rename, GC on start, async device→host, save-on-TERM within 60 s, keep-policy per tensor | D1, D2, D3, D4 |
| R14. **Resume cost budget**: persistent compile/autotune cache on scratch keyed by HLO hash; smoke once per run id; dumps off by default | D5 |
| R15. **Local metrics are the source of truth**: append-only per-step file (parquet/jsonl) in the run dir; W&B is an optional async mirror with a 10 s init timeout and offline fallback; analysis reads local files; a run is "complete" only by its own completion marker | W1-W6, D9, F2 |
| R16. **Native wq contract**: the trainer itself implements `--partial` semantics (exit codes, TERM handling, `FETCHED_JOB_ID` paths, TMPDIR), so no per-project sbatch glue, unset lists or fail-marker files | E2, E6, D6-D8, S1-S2 |
| R17. **GPU-busy guarantee**: evals and data prep run inside the training loop on the same GPUs at bounded cost; two packed runs are planned to end within 60 min of each other from measured speed | C4 |

## 5. Open items (not measured here)

- Throughput cost of `--xla_gpu_deterministic_ops` or a sorted embedding backward: never measured (`objective-hillclimb.md:76`).
- Root cause of the 8-GPU AllToAll first-call abort (C1) and the 4-GPU clique-acquire abort (C2): not diagnosed; mechanisms above are ESTIMATES.
- GPUh lost inside pilot hosts by guests that failed (A6, E3) and by host time forfeited after quick-fail circuit breaks: not in sacct; only an upper bound given.
- Slurm `KillWait` on pli-c (time between TERM and KILL at the wall) not checked; the 60 s figure is the wq/packed_task `--kill-after`.
