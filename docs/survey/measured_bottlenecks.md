# Measured bottlenecks: where wall-clock time goes in today's runs

Survey dossier, 2026-10-24. I used existing evidence only and submitted no jobs. Every number carries its source.
**ESTIMATE** marks numbers that were not measured, with the reasoning next to them.

## 0. Sources and method

| source | what it gives | where |
|---|---|---|
| Local W&B `.wandb` files (complete per-step history; no API paging bugs) | `throughput/{duration,tokens_per_second,mfu,hook_time,loading_time}`, `eval/total_time`, `_timestamp` | `$ROOT/wandb-runs/wandb/run-*/`; parsed copies cached as `marincpp/cache/wandb/<run-dir>.jsonl.gz` (75 run dirs) |
| Parse and accounting scripts (reusable) | segment accounting, startup phases, redone steps, fleet GPU-h, wq waits | `marincpp/cache/wandb/_tools/{dump_local,seg_account,startup,redo,summ,fleet,ledger2,wqstats,fails}.py` |
| sacct | Start/End/Elapsed of every chain segment; 30-day fleet GPU-h | `_tools/baseline_sacct*.txt`, `_tools/sacct_30d.txt` |
| Slurm logs | import, first batch, first step, restore, resume step, ckpt staging, eval tqdm | `project/marin/logs/slurm/muonh-qwen3-*_<jid>.out` |
| MEMSTATS lines (max over devices, every 600 s) | peak bytes in use vs `bytes_limit` 71.26 GiB | per-log maxima in `_tools/memstats_peaks.txt` (260 logs) |
| Speed workflow wf_cb88f9ae-72b (journal, 5 results so far) | head/CE, attention, XLA/collectives, pipeline overheads | `.../wf_cb88f9ae-72b/journal.jsonl`; scratch `tmp/speed_wt/scratch-*` |
| pls speed study | readout, memory, overhead, XLA facts and plan | `tmp/pls_speed/results.json`, `plan.md`, `steptimes*.txt` |
| Ledgers | per-task GPU-h, exit codes | `project/marin{,-pls}/logs/packed/*.tsv` |
| wq store | submit, start and end of each run | `$ROOT/wq/jobs/*/{spec,state}.json` |

No `xplane`/nsys profile exists for any of these runs: a search of `tmp/`, `logs/` and `marin_store_big/speedrun` (depth 6) found none. The pls profiler probes 16144708/16144709 are still PENDING (sacct). So per-op splits below come from W&B step-time differences between runs that differ in one factor, from HLO analysis, and from non-H100 microbenchmarks.

## 1. MFU formula

Levanter, `lib/levanter/src/levanter/utils/flop_utils.py:31-61`, forward FLOPs per token:
`L*(2*3*d*f + 2*d*d) + L*(2*d*(H*hd + 2*kv*hd) + 4*S*H*hd + 3*S*H) + 2*d*V`.
`main/train_lm.py:388-389` multiplies by 3 (fwd+bwd) and by S per example. Peak = `h100` bf16 1.979e15/2 = 989.5 TFLOP/s (`lib/fray/src/fray/device_flops.py:21-28`).

How it differs from 6N + causal attention:
- It counts the full S x S attention map (no causal halving), so attention is counted 2x.
- It ignores remat recompute, the embedding, norms and the optimizer.

My recompute of levanter's formula reproduces the logged MFU exactly (130m 14.36, 300m 15.27, 520m 15.91, 1.2B 27.82). With 6*(N_nonemb + d*V) + 6*L*S*d (causal), the MFU is 10-16% lower in relative terms (table 2).

## 2. Per-size step metrics (H100 80GB SXM, seq 4096, V=128256, MuonH)

Medians over all steps of the finished run (W&B `throughput/*`). "6N+attn" is my recompute.

| config | GPUs | step s | tok/s/GPU | MFU levanter | MFU 6N+attn | peak mem / limit (GiB) | source |
|---|---|---|---|---|---|---|---|
| 130m (d512 L6), PDP 32 | 4 | 0.6305 (p10 0.627, p90 0.633) | 207.9k | 14.36% | 12.74% | 8.50 / 71.26 | W&B run-20260914_041824; mem: vo130_15168741-128k-rep8.log (same config) |
| 300m (d768 L12) | 4 | 1.444 (p10 1.442, p90 1.448) | 90.8k | 15.27% | 13.15% | 13.47-13.75 / 71.26 | 3 segments 13852135-37; mem: rep-300m_14940537_b, ema-128_w20 |
| 520m (d1024 L24), PDP 32 = 2 microbatches | 4 | 6.741 | 38.9k | 15.91% | 13.48% | 22.15 / 71.26 | 11 segments 13852139-13855295; mem: vo520-128k-rep8_15162227 |
| 1.2B (d2048 L16), PDP 32 = 2 microbatches | 4 | 8.221 (p10 8.17, p90 8.26) | 31.9k | 27.82% | 25.19% | not logged (no MEMSTATS in Sept runs) | 27 segments of array 13855503 |
| 300m pls0 (no readouts) | 8 | 0.833 | 78.7k (-13% vs 4 GPU) | 14.32% | 11.4% | - | run-20261015_200043 |
| 520m pls0 | 8 | 4.150 | 31.6k (-19%) | 14.13% | 10.9% | - | run-20261016_012052 |
| 1.2B pls0 | 8 | 4.99-5.14 (one 98-step segment 4.34) | 25.5-26.3k (-18 to -20%) | 23.5-24.2% | 20.7% | - | run-20261016_060549, 20261017_123322, 20261015_200418 |
| pls1-sep 130m, 6 layers, 6 heads | 4 | 1.742 | 75.2k | 20.18% | - | 15.9 (CPU memory_analysis, pls.md:46) | run-20260930_213852 |
| pls1-sep 300m, 12 layers | 4 | 4.431 | 29.6k | 24.41% | - | 30.9 (CPU, pls.md:46) | run-20261004_030549 |
| pls d48 preln sep, PDP 4 | 4 | 14.99 (p90 15.13) | 8.7k | 18.76% | 18.2% (incl. 48 readouts) | temp 30.61 (lns mb4, 15322826) | run-20261023_085637 (w89) |
| pls d48 preln (gilns), PDP 4 | 8 | 7.548 | 8.7k (perfect 4->8 scaling) | 18.63% | - | - | run-20261022_051955 (w15) |
| pls d48 hc rstatic, PDP 1 | 4 | 24.10-24.20 | 5.4k | 11.6% | - | temp 17.69 (mhc mb1, 15322826) | run-20261022_181312 / 20261023_124928 (w27) |
| pls d48 hc, PDP 4 (died at step 257, BFC fragmentation) | 4 | 16.11 | 8.1k | 17.5% | - | temp 32.91 | run-20261009_202619 (15266153) |
| pls d24 hc, PDP 1 | 4 | 12.51-12.59 | 10.5k | 11.2% | - | - | w80 |
| pls d48 attnres qclip, PDP 1 | 4 | 34.10 | 3.8k | 8.25% | - | temp 53.36 at mb4 | run-20261023_023921 (w78) |

Headroom: dense runs use 8.5-22 GiB of a 71.26 GiB pool (12-31%). Yet 520m at full batch (64 seqs/GPU) died on a 35.45 GiB allocation (13852139, kaiyue_muonh_qwen3_h100x4.yaml `h100_520m_oom`), and hc d48 at PDP 4 died at step 257 from fragmentation (pls-d48g4-hc_15266153_a.log:285-300). Memory is not short; one contiguous temp buffer per step under BFC is the problem.

Vocabulary sweep (300m, 4 GPU, same tokens per step; W&B medians): 128K 1.443 | 64K 1.363 | 32K 1.272 | 16K 1.224 | 8K 1.222 | bytes (V=512) 1.215 s. 130m: 0.630 | 0.551 | 0.479 | 0.444 | 0.440 | 0.440. 520m: 128K 6.732, 8K 6.193, bytes 6.16. Charwin (flash block 128): v64k sl4224 2.443, v32k sl4480 2.627 s.

## 3. Per-op breakdown of one step

### 3a. Dense 300m, 4 GPU, 1.444 s (H100 decomposition from W&B step-time differences)

| component | s/step | share | evidence |
|---|---|---|---|
| Attention, JAX_FLASH block 1024 (fwd + flash recompute + remat fwd + bwd) | 0.67-0.73 | 46-50% | Linear fit over bytes runs with the same tokens per step: 1.215 s at S=4096, 3.40 s at S=16384 (run-20261010_073122), so 1.778e-4 s per unit S, 0.728 s at 4096. The fit predicts 18.74 s at S=19456 against 18.51-18.55 s measured (run-20261023_114332). The lower bound subtracts ~0.25 s of loop overhead at 16384 (5.5k tile iterations x ~50 us). The speed workflow's attention agent got 0.70 s of 1.25 s independently. |
| LM head + CE (128K, XLA fallback path) | 0.228 | 15.8% | 1.443 (128K) minus 1.215 (bytes). Runs at ~340 TFLOP/s (34% of peak). |
| Everything else: QKVO/MLP GEMMs incl. remat, norms, RoPE, MuonH NS, collectives, embedding | ~0.49 | ~34% | Residual of the fit. GEMM FLOPs 8*N_nonemb*tokens = 1.08e14/GPU take ~0.18 s at 600 TFLOP/s (ESTIMATE), so ~0.3 s is non-GEMM. A non-H100 profile of the 300m layer splits 232 ms GEMM / 229 ms other (scratch-memory/raw2.log, policy full). |
| of which collectives | ~0.010-0.015 | ~1% | ESTIMATE: HLO counts 2.4 GB wire/GPU/step (26 all-gather, 14 reduce-scatter, 54 all-reduce, 10 all-to-all), at ~300 GB/s NVLink. 69 of 72 collectives are `is_sync` (exposed). gpu2_fsdp_v64k.hlo, speed journal af9ca2db. |
| of which MuonH Newton-Schulz | few ms compute + 35 f32 all-reduces of [12,768,768] (991 MB) | <1% | HLO inventory (same). NS is column-sharded by util.py:282. |
| of which embedding gather + scatter-add grad | <0.003 | <0.2% | ESTIMATE from A40 3.7-7.5 ms at B=131072 (bench_embed.py). |
| of which per-step host hooks | 0.007 median, 0.0435 every 10th step | ~0.9% | W&B hook_time by step%10 (pipeline agent, v8k run). |

Attention efficiency: model FLOPs (6*S*d per token per layer) are 2.97e13 per GPU per step, so 0.70 s is 4.3% of peak. Counting the remat forward, the flash recompute and the block-causal waste, it is ~8%. The kernel is memory/loop-bound: the A40/H100 time ratio (279 ms/layer x 12 vs 0.70 s, 4.8x) matches the HBM bandwidth ratio (~4.8x), not the FLOP ratio (6.6x).

### 3b. Dense 130m, 4 GPU, 0.630 s
Attention 0.258 s (41%), from bytes 0.440 s at 4096 and 1.215 s at S=16384 (run-20261009_063605). Head+CE 0.190 s (30%; 272 TFLOP/s, 27% of peak). Rest ~0.18 s (29%).

### 3c. Dense 520m, 4 GPU, 6.741 s
- Head+CE 0.57 s (8.5%), from 6.732 minus 6.16 (bytes run-20261011_001057); 363 TFLOP/s.
- Attention **ESTIMATE** 3.2-3.8 s (48-56%). I scaled the 300m per-layer cost by layers (x2), tokens per GPU (x2) and d (x1.33). No 520m seq sweep exists.

### 3d. Dense 1.2B, 4 GPU, 8.221 s
**ESTIMATE only.** GEMM FLOPs 8*906M*262144 = 1.9e15 per GPU take >= 1.93 s at peak, ~3 s at 65%. The head takes ~1.2 s at the 35% efficiency measured on smaller sizes. That leaves <= ~4 s, so attention is ~3-4 s (36-50%). The higher MFU (28%) fits bigger GEMMs (d=2048) and head_dim 128.

### 3e. pls d48 preln sep, PDP 4, 4 GPU, 15.2 s (tmp/pls_speed/results.json)

| component | s/step | share | evidence |
|---|---|---|---|
| 47 extra readout CEs (128K, b-tiled batched_xla) | 12.3 | 81% | 15.1-15.4 s against 2.9 s for the d48 baseline at PDP 4 (logs/muonh-qwen3-130m-della4xh100-d48-diag600-pdp4.log.log); 263 TFLOP/s, 27% of peak |
| of which f32-logit HBM traffic | ~9 | ~60% | Roofline: 27 TB/GPU/step at 3 TB/s plus ~4.7 s of GEMMs predicts 15 s against 15.2 s measured |
| of which the scatter+convert in the backward | ~2.2 | ~14% | 6.45 TB/step removable by a one-hot select (plan.md A1; bitwise on CPU) |
| Body (48 layers x 512) | 2.9 | 19% | same baseline; attention ESTIMATE ~2 s of it (48 x 43 ms/layer, from the 130m fit) |
| Head-grad all-reduce/reduce-scatter (PARAM_MAP=vocab) | ~0.84 at PDP 1 if not overlapped | - | inferred from the CPU HLO, not measured (results.json readout) |

hc/mhc at PDP 1 add +52-58% (24.2-25.4 s against 16.0-16.3 s at PDP 4), because of 32 microbatches and the `xla` CE fallback with f32-operand backward GEMMs (TF32). attnres: 31-34.6 s, set by its own source-buffer mixing (mb4 32.5 s gives no gain).

## 4. Time breakdown of a full run

### 4a. Dense baselines, 4xH100 in 2-h chain segments (my accounting: `_tools/seg_account.py`, `startup.py`, `redo.py`)

| bucket | 130m (1 seg, 3441 s) | 300m (3 seg, 18091 s) | 520m (11 seg, 72474 s) | 1.2B (27 seg, 200375 s) |
|---|---|---|---|---|
| startup (job start to first new step: import ~20-35 s, data/restore 13-120 s, first step 13-24 s) | 69 s, 2.0% | 204 s, 1.1% | 2140 s, 3.0% (1100 s in one import stall, 13852146) | 4064 s, 2.0% (1080 s stall, task 12) |
| redone steps after resume (W&B hides them) | 0 | 328 steps, 474 s, 2.6% | 260 steps, 1753 s, 2.4% | 288 steps, 2368 s, 1.2% |
| compile excess inside step durations | 37 s, 1.1% | 74 s, 0.4% | 50 s, 0.1% | 132 s, 0.07% |
| **steady train steps** | 3126 s, **90.8%** | 16512 s, **91.3%** | 66774 s, **92.1%** | 187641 s, **93.6%** |
| eval (Paloma, 114 batches; 21 / 43 / 94-98 / ~113 s each) | 88 s, 2.5% | 482 s, 2.7% | 868 s, 1.2% | 2522 s, 1.3% |
| checkpoint, blocking part (1.64 / 2.97 / 5.47 / 12.62 GiB staged; 1.2B 7 s per save every 10 min) | 13 s, 0.4% | 69 s, 0.4% | 355 s, 0.5% | 2307 s, 1.15% |
| other hooks (logging, watch, host sync) | 58 s, 1.7% | 141 s, 0.8% | 150 s, 0.2% | 294 s, 0.15% |
| unaccounted gap + loading + tail | 51 s, 1.5% | 139 s, 0.8% | 458 s, 0.6% | 1285 s, 0.6% |
| queue gaps between segments (not GPU time) | 0 | 3.42 h | 15.33 h | 25.28 h |
| first queue wait (submit to start) | 10.1 h | 8.4 h | 10.1 h | 8.75 h |
| submit to final result | 11.0 h | 16.9 h | 45.5 h | 89.7 h |

Notes:
- Every 2-h segment ran 02:05:02-02:05:30 (sacct): the final save and eval finish after the wall.
- Two of 42 resumed segments spent ~18 min between `jax devices` and the launcher's first log line. This was import or GPFS time (13852146: start 04:17:51, first log 04:36:11).
- The 520m first segment died of an OOM after 14.5 min (871 s of wall; 33 steps were redone).

### 4b. pls d48 (results.json "overhead"; 4 GPU, 24.1 s/step, 2 segments, 37.0 h)
Steps 90.1% | checkpoint 3.1% (19.5-22.5 s per save of 37.32 GiB state, host staging at 1.65-2.7 GiB/s) | per-layer eval 3.9% (5 x ~1045 s) | resume smoke 1.6% (11-23 min per segment) | small hooks 0.3% | startup, jaxpr logging (35-213 s per process, median 88 s), redone steps and teardown ~1.1%. hc d24: steps 91.4%, eval 4.0%, ckpt 1.6%, smoke 1.6%.

### 4c. Vocabulary runs (speed journal a49a0cf6)
Non-step overhead 3.5-5.5% per run. Paloma eval 2.2-3.5%, per-step hooks 0.45-1.05%, between-step gap 0.13-0.28%, loader GIL bursts 0.13-0.39%, checkpoint 0.08-0.13%, startup 45 s fresh / 92 s resumed. Charwin plans had a 450-min task limit against 482-730 min of need, so each run was killed at least once, losing 5.35 min of work per kill.

### 4d. Fleet overheads that are not per-step (last 30 days)
- Time-limit ends: vocab ledger 16 main tasks with rc 124; pls ledger 39 (rc 124) + 15 (rc 143) (`_tools/ledger2.py`).
- Lost work per kill: mean ~6 min, from 12 ends over 164-556 s (results.json). No SIGTERM checkpoint handler exists anywhere.
- wq (started 2026-10-21): first-start wait median 4.9 h (marin, n=49) and 5.8 h (pls, n=16); p90 30.9 h and 17.1 h; max 53 h. Resume gaps median 13.9 h (marin, n=4) and 1.2 h (pls, n=7) (`_tools/wqstats.py`).

## 5. Top 10 bottlenecks by GPU-hours lost per month

Fleet basis (**ESTIMATE**): sacct 2026-09-24..10-24 gives 29.8k GPU-h for the user. Of that, 13.2k is RL (miles/molt), 12.8k named pretraining jobs and 3.8k pilot hosts (`_tools/fleet.py`). So pretraining is about **16k GPU-h/month**: ~6k pls (pls ledger 5.9k) and ~10k dense (vocabulary, ss/ov, bytes, mtpl).
"Lost" is time an achievable fix would remove. The ranking uses midpoints.

| # | bottleneck | share of a run (measured) | GPU-h lost/month | evidence | fix class |
|---|---|---|---|---|---|
| 1 | **Attention kernel**: pure-JAX JAX_FLASH, 4-8% of peak | 41% (130m), 46-50% (300m), ~48-56% ESTIMATE (520m), ~36-50% ESTIMATE (1.2B); ~13% of the pls d48 step | **~3-4.5k** (ESTIMATE: 25-40% saving on 10k dense, 7-10% on 6k pls) | §3a fit; segmented FA4 fwd 6-12x faster than flash1024 on A40, fwd+bwd 3.4-7x on A100; cuDNN with a dense segment mask 1.7-2.0x (scratch-attention/*.jsonl) | segment-aware FA3/FA4/cuDNN-THD kernel (FA4 already in `levanter/grug/attention/_fa4_cute.py`) |
| 2 | **pls readout CE**, memory-bound on f32 logits | 81% of the d48 PDP 4 step; 64% of the 6-layer step (1.11 of 1.74 s) | **~2-2.5k** (ESTIMATE: a fused CE at ~2x throughput saves ~40% of pls) | §3e roofline; scatter+convert alone ~2.2 s/step | fused CE that never writes f32 logits; A1 one-hot first (~10%) |
| 3 | **Full per-layer remat while 70-88% of HBM is idle** | recompute of the whole layer forward; saving the attention output cuts the step 14-19% (non-H100, 300m layer, B=2/4) | **~0.8-1.2k** (ESTIMATE ~10% of dense; overlaps with #1, since most of the recompute is attention) | scratch-memory/raw2.log, rawp_4096.log; MEMSTATS 8.5-22 GiB of 71 | per-size remat policy chosen from a memory model |
| 4 | **Dense LM head + CE via the XLA fallback** (fused kernel never launches: "weight tile exceeds GPU shared-memory budget") | 30% (130m), 15.8% (300m), 8.5% (520m) of the step at 128K; 27-37% of peak | **~0.5-0.8k** (ESTIMATE: halve it on the ~half of dense GPU-h at 64K-128K) | §2 vocab sweep; journal a28a7a5d note 4; 8K pads to 16384 columns (2x head work) | fused Triton/CUTLASS CE; balanced vocab blocks |
| 5 | **Forced microbatch 1 for hc/mhc/attnres/moda** (BFC fragmentation; mb2 needs 107-128 GiB from a 1-trip CE tile) | +52-58% step for hc/mhc d48 (24.2-25.4 vs 16.0-16.3 s) | **~0.4-0.45k** (hc/mhc families ~1.2k GPU-h/month x 0.36) | 15322826 temp sizes; 15266153 OOM at step 257; ce_cse_probe.py (64.45 -> 8.08 GiB with a 2-trip tile) | non-fragmenting allocator plus planned temp buffers; CE tile <= half of rows |
| 6 | **8-GPU FSDP per-GPU loss** (69/72 collectives exposed) | -13% (300m), -19% (520m), -18 to -20% (1.2B) tok/s/GPU against 4 GPU | **~0.3k** (ESTIMATE ~2k GPU-h/month on 8-GPU dense runs) | §2 table; pls d48 with PARAM_MAP=vocab scales perfectly (14.99 -> 7.55 s) | overlapped collectives or DP+ZeRO-1 at these sizes |
| 7 | **In-training eval** (serial, on the training GPUs; pls runs it twice) | 1.2-2.7% (baselines), 2.2-3.5% (vocab), 3.3-3.9% (pls; +0.9% duplicate main eval, +1.3% from eval at PDP 1) | **~0.45k total, ~0.2k avoidable** | §4a-c; results.json evals | one fused eval pass; right-size the eval batch |
| 8 | **Blocking checkpoint saves** (host staging, not GPFS) | 3.1-3.4% (pls d48, 37 GiB), 1.15% (1.2B), 0.4-0.5% (130m-520m), ~11% (OV before the hourly interval) | **~0.25k** | Slurm "Checkpoint staged" lines; restore is fast (37 GiB in 7.4 s) | async device-to-host into pinned buffers, overlapped with steps |
| 9 | **Segment/resume overhead** | per resume: startup 46-138 s (2/42 had ~18-min import stalls), ~6 min of redone work (no SIGTERM save), pls smoke 11-23 min, jaxpr/HLO logging 3.5-4 min | **~0.3k** (54 pls + 16 vocab time-limit ends; smoke tasks 135 + 55 GPU-h in the ledgers) | §4a, §4d, ledgers | save on SIGTERM; cached, resume-safe smoke; jaxpr logging off by default |
| 10 | **Flash block 128 at SEQ_LEN not a multiple of 1024** (charwin) | +1.0-1.1 s of a 2.4-3.7 s step (~40%); 21.4k vs ~500 loop iterations per step | **~0.1-0.2k** while charwin runs | journal a28a7a5d; per-iteration cost ~32 us (A40) to ~55 us (H100 log fit) | block = largest divisor <= 1024, or #1 |

Below the line, all measured:
- Per-step host sync: 7 ms median, 43.5 ms every 10th step (watch hook: 388 keys, one transfer each), `loss.item()` every step; 0.2-1.7% of wall, ~0.1k GPU-h/month.
- Loader GIL bursts: 0.13-0.39%.
- W&B logging: 0.05%.
- MEMSTATS: 0.
- Compile: 13-24 s per first step with the persistent cache on (18 GB in `cache/jax`); two train-step variants compile per process.

Not GPU-hours but the largest latency term: queue.
- 1.2B baseline: 25.3 h of gaps between segments plus 8.75 h before the first start, for 55.7 h of compute.
- wq today: first-start wait median ~5 h, p90 17-31 h.

## 6. Reliability incidents seen in this evidence (not costed monthly)
- **OOM from fragmentation, not capacity**:
  - 520m full batch died on a 35.45 GiB allocation with ~22 GiB peak at PDP 32 (13852139).
  - hc d48 PDP 4 died at step 257 (15266153).
  - Two d48 runs died at step ~260 (15266143, 15266145; source: needs_from_sessions.md, pls section 3).
  - 1.2B OV died at steps ~130 and ~1290 with 44 GiB free but no contiguous 30 GiB (over-vocab.md:294-300).
- **Slurm "FAILED" is not a failure signal**: 3.9k non-RL GPU-h in 30 days ended FAILED. Most are 23-h chain segments that exited 1 or 124 at the wall (e.g. bytes-520m 15283034-36, vo520-v8k 15168742).
- **8-GPU AllToAll rendezvous abort after 40 s** in hc/attnres smokes (w16/w17).
- **JobArrayTaskLimit stall**: in the 1.2B array it struck 6 of 26 rollovers; one gap was 8.4 h (task 12: eligible 10:13, start 18:39).

## 7. What this means for the rewrite (measured facts only)
1. At 130m-1.2B dense, attention plus LM head take 55-70% of the step (130m 71%, 300m 62-66%, measured). A good attention kernel and a fused CE matter far more than any overhead item.
2. Non-step overheads (eval, checkpoint, hooks, startup, redo) total 6-9% of wall for dense runs and ~10% for pls d48. They are worth fixing, but they are second order.
3. Memory is underused (12-31% of HBM), yet allocation fails. Predictable, planned buffers would also remove remat, raise microbatch sizes and end PDP-dependent CE numerics.
4. Going from 4 to 8 GPUs costs 13-20% per GPU on dense runs today, so 4-GPU runs are the efficient default until collectives overlap.
