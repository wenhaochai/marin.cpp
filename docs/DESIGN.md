# marincpp design, version 2

Date 2026-10-24. Status: design and headers only; no implementation exists yet. Version 2 applies the perf, parity
and research reviews of version 1; the Review log (Appendix A) lists every point and what happened to it.

This document fixes the architecture of the marincpp training binary: one C++/CUDA process that trains the Kaiyue
muonh_qwen3 models (130m to 1.2B) and the pls d48 readout models on one node with at most 8 H100 SXM GPUs. The module
interfaces are the headers in `include/marincpp/`. The spec that every number below must reproduce is
`docs/spec/kaiyue_baseline.md` (parts: `model.md`, `optim.md`, `data.md`, `eval.md`, `numerics.md`, `targets.md`;
"§" references point into those parts; "T0.1"-"T6" are the parity-plan gates).

Binding owner decisions:
1. The training process is pure C++/CUDA. Forward and backward kernels are hand-written. Python appears only in tests
   (reference implementations, fixtures), in the offline converters, in the W&B mirror and in analysis.
2. Development effort never decides a design choice. Speed and correctness decide.
3. Scope: one node, at most 8 H100, only what the pls and Tokenizer Size groups need.
4. Stage 1: reproduce the Kaiyue baseline within seed noise of levanter, at higher tokens/s.

## 0. Summary of decisions

| # | Area | Decision | Rule that can change it |
|---|---|---|---|
| 1 | Process | One process, one host thread per GPU, NCCL communicators from `ncclCommInitAll` | none foreseen |
| 2 | Memory | One static arena per GPU, planned on the CPU from the config; no allocation after start | none |
| 3 | Graphs | The whole update (forward, backward, reductions, optimizer, metrics) is ONE CUDA graph `g_step`; one instantiation per step regime (stats step, readout phase, ...); the host picks the graph from k | none |
| 4 | Activations | remat=all when it fits; remat=min (elementwise recompute) next; remat=layer (levanter's per-layer recompute) only for the strict parity plans and variants that need it | planner rule §2.3 |
| 5 | Attention | Vendored FlashAttention-3 (hopper/, commit `767386d`); the backward never runs FA3's padded non-persistent varlen grid: either a varlen persistent backward scheduler or a doc-causal patch (batch = rows) | E1b on H100 (§3.4, O1) |
| 6 | Linear CE | Custom fused CE, persistent warp-specialized kernels; f32 logits never exist; stages B1 / B1.5 / B2 chosen per D; logits store dtype bf16 or f32 = what T1.0(2) shows | E2 + micro-bench (O2), T1.0(2) |
| 7 | GEMMs | cuBLASLt plus custom CUTLASS kernels; timed search per shape at plan time over deterministic algorithms only (split-K with non-atomic reduction and stream-K with deterministic fixup allowed); fused epilogues in stage 1 | per-shape timing |
| 8 | Numerics | `exec.policy` bf16 or f32 (f32 for parity tests only); `exec.rounding` levanter or accurate; a `levanter_strict` preset reverts every class-P deviation; FMA contraction decided per formula | numerics.md §1 rule |
| 9 | Optimizer | MuonH, AdamH, Adam fully on device; optimizer groups are data (`GroupSpec`); clip groups split so each starts when its buckets are reduced; schedules from a levanter fixture | T1.0 decides NS precision |
| 10 | Parallelism | DDP compute + ZeRO-1 state for every group: whole-matrix owners for MuonH, row shards for vocab-sized tensors; reduce to owners, bf16 all-gather at the start of the next step | §6 |
| 11 | Data | One-time Python conversion to a flat uint32 token file; C++ mmap reader with a 256-step pageable ring and 16-32 reader threads; order = pure function of (plan, step) | none |
| 12 | Checkpoint | Raw binary + JSON manifest, owner slices, async D2H, atomic rename, save on SIGTERM if it fits the 60 s budget; import from levanter checkpoints | measured write bandwidth |
| 13 | Metrics | Device ring, cross-rank sums inside the graph, host copies every 32 steps, local `metrics.jsonl` is the record, W&B mirror is a Python sidecar | none |
| 14 | Config | One `Config` from `knobs.def`; valid names come from the registries; run id = hash of the non-default hashed knobs + the knob-table version | none |
| 15 | Determinism | On by default | flips off if it costs > 3% of a step (O5) |
| 16 | Tests | f64/PyTorch references for baseline kernels, the existing JAX code for variants, whole-model gates V1-V4, sanitizer and poison modes; portable kernels also build for sm_80/86 so tests run on vis nodes | none |
| 17 | Build | CMake, sm_90a (plus sm_80/86 for the test library), CUDA 12.8 | O8 |

Expected step times against the levanter H100 reference (4 GPUs, §12): 130m 0.188 s vs 0.632 s (3.4x), 300m 0.378 s vs
1.444 s (3.8x), 520m 1.63 s vs 6.74 s (4.1x), 1.2B 3.07 s vs 8.22 s (2.7x), pls d48 6.57 s vs 15.2 s (2.3x; 5.97 s
with CE B1.5, 5.36 s with B2). These are budgets from stated efficiency assumptions; the H100 benches replace the
assumptions as they land.

## 1. Process and device model

### 1.1 One process, one thread per GPU

The process starts one worker thread per visible GPU. Rank r uses CUDA device r of `CUDA_VISIBLE_DEVICES`.

| | one process, N threads (chosen) | one process per GPU |
|---|---|---|
| NCCL setup | `ncclCommInitAll` in one call; no bootstrap, no rendezvous | unique-id exchange through a file or TCP |
| Launch | the wq script runs one command; no `srun`, no literal `CUDA_VISIBLE_DEVICES` (wq rule 4) | needs a launcher inside a wq job |
| SIGTERM | one signal thread decides once for all ranks | N processes must agree on the stop step |
| Shared host memory | one mmap of the token file, one prefetch ring, one JSONL writer, one pinned checkpoint stager | N copies |
| Host overhead | negligible: host work per step is one barrier, one memcpy and one graph launch | same |
| Fault isolation | a crash kills all ranks | same in practice (NCCL aborts the others) |

### 1.2 Threads

| thread | job |
|---|---|
| main | parse and resolve the config, open the run directory, plan memory, create NCCL communicators, join the workers, return the exit code |
| signal | `sigwait` on SIGTERM, SIGINT, SIGUSR1 (all other threads block them with `pthread_sigmask`); sets `StopControl::requested` |
| worker r | owns `DeviceContext` r: streams, cuBLASLt handle, arena, graphs; runs the step loop |
| reader pool (16-32) | fill the pageable prefetch ring for future steps (§5.4) |
| stager | copies ring entries into the 3 pinned slots |
| metrics writer | waits on copy events, formats JSON lines, appends to `metrics.jsonl` |
| checkpoint writer (per rank) | writes the pinned slice of a checkpoint to its file |

Per-step rendezvous: every worker calls `Process::step_barrier(k)` before it launches update k. The barrier's completion
function runs once and reads `StopControl`, so all ranks launch exactly the same steps. Cost: about 1 µs per step.

### 1.3 NCCL

`ncclCommInitAll(comms, N, nullptr)` from the main thread, before the workers start. Each worker uses `comms[r]` only
from its own thread. Collectives are captured into each rank's graph; every rank captures with
`cudaStreamCaptureModeThreadLocal`. Before capture, each worker runs every distinct collective once outside the graph
(warm-up, reliability rule R11). The binary records the NCCL version and every `NCCL_*` variable in `events.jsonl`.
With `exec.deterministic` it sets `NCCL_ALGO=Ring` and `NCCL_PROTO=Simple` unless the user set them.

### 1.4 How Slurm and wq launch it

A wq job script (`scripts/wq/train.sh`, self-contained, fetchable-partial):

```bash
#!/bin/bash
#SBATCH --gres=gpu:4 --time=23:00:00 --mem=400G -J nm-130m
#SBATCH --comment=fetchable-partial
#FETCHABLE
set -euo pipefail
module load proxy/default                       # W&B sidecar only; training needs no network
export TMPDIR=$ROOT/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
RUN=$ROOT/nm_runs/muonh-qwen3-130m-della4xh100-nm-<id>       # created by `marincpp plan --create`
exec "$RUN/bin/marincpp" train --config "$RUN/config.resolved.toml"
```

- `marincpp plan --config ... --create` (CPU only, vis node) resolves the config, writes `config.resolved.toml`,
  copies the binary and its shared libraries into `bin/`, and prints the memory plan and the step budget. A config
  that does not fit is refused before queueing.
- The job runs the snapshot `bin/marincpp`, never a live build (reliability R5). `exec` makes the trainer the
  process that receives TERM. The trainer checks `DONE` before touching a GPU and exits 0 at once if the run is
  finished (reliability D8).

### 1.5 SIGTERM path

wq hosts stop guests with TERM and kill them 60 s later. Timeline:

1. t = 0: TERM. The signal thread sets `requested = 15`.
2. At the next step barrier the completion function sets `decided_stop_step = k`. No rank launches update k.
3. Each worker synchronizes its streams. Launch lookahead is 2 graphs (independent of the data ring depth, §5.4), so
   the in-flight graphs finish within 2 step times: 0.4 s at 130m, 13 s at pls d48.
4. Save decision: the trainer knows the checkpoint size and the write bandwidth it measured at its last periodic
   checkpoint. It saves if `elapsed + bytes / bandwidth < 50 s`; otherwise it keeps the last periodic checkpoint.
5. Save (§7): D2H of each rank's owner slice, parallel file writes, fsync, manifest, rename.
6. Flush and fsync `metrics.jsonl` and `events.jsonl`, append a `stop` event.
7. Exit 143. wq requeues the job (99, 124 and 143 requeue).

Sizes: 130m checkpoint 1.76 GB, pls d48 40.1 GB (§7.2). At 1.65-2.7 GiB/s host staging the pls save takes 14-23 s if
GPFS sustains it with 4 writers; E11 measures it. Step 4 makes the path safe for any measured rate.

## 2. Memory

### 2.1 Static arena

- Each GPU gets one `cudaMalloc` of exactly the planned size at startup, plus one `cudaHostAlloc` for pinned buffers
  (token slots, checkpoint staging, metrics copies). The pageable prefetch ring (§5.4) is ordinary host memory.
- Every device pointer recorded in a graph points into the arena, so graphs stay valid for the whole run.
- Nothing allocates after start: cuBLASLt gets a workspace sized from the split-K counts of the chosen algorithms; FA3's
  glue buffers (semaphores, accumulators, scheduler metadata) are arena buffers whose zeroing is a memset node in the
  graph; CUB sort temp is planned; NCCL buffers are allocated by `ncclCommInitAll` before the arena. The trainer
  records `cudaMemGetInfo` after warm-up and asserts it is unchanged at every eval and checkpoint.
- Budget: `exec.mem_budget_gib` = 72 GiB of the 79.6 GiB an H100 80GB exposes.

### 2.2 Buffer classes and binding (`memory.hpp`)

| class | contents | placement |
|---|---|---|
| Persistent | owner shards of f32 masters, MuonH momentum, Adam/AdamH mu and nu; replicated bf16 compute copies; f32 gradients (bucket order); NS workspace; schedule tables; RoPE tables; token buffers; metric rings | one region |
| Microbatch | tensors that live for one microbatch's whole forward and backward but belong to no single layer (depth sources of attnres, readout states, per-microbatch head-gradient accumulator of the strict mode) | one region |
| Activation | tensors kept from a layer's forward to its backward within one microbatch | one region per layer |
| Transient | scratch inside one layer's forward or backward | one slot per role, reused by all layers |
| Workspace | FA3 accumulators, CE chunk buffer, embedding sort temp, cuBLASLt workspace | one slot per kernel family |

- Binding is by declared name with dtype and shape checks (`BindCtx::get("act.L3.gu", bf16, {T, 2F})`); a positional
  index never selects a buffer. A mismatch is fatal at bind time.
- Debug modes (`exec.debug_arena`, local knob): `canary` puts a 4 KiB guard band before and after every buffer and
  checks them after every step; `poison` also fills transients and workspaces (the recompute slots are transients)
  with NaN before each use, never saved activations. Both are run by
  the S7 gates and by every new variant's tests; `compute-sanitizer` memcheck, racecheck and synccheck run on a 2-layer
  model per variant (eager mode).

### 2.3 Activation policy (recompute)

Saved per layer per token by the baseline sandwich block (bf16 unless noted):

| tensor | bytes/token | needed by | remat=min |
|---|---|---|---|
| `x_in` residual entering the layer | 2D | input-norm backward | saved |
| `a` input-norm output | 2D | QKV weight gradient | recomputed |
| `qkv` raw projections | 2(H+2KV)hd | QK-norm backward, attention (v) | saved |
| `qk_rot` normed + rotated (+ scaled) q, k | 2(H+KV)hd | attention backward | recomputed |
| `attn_o` attention output | 2·H·hd | attention backward, o_proj weight gradient | saved |
| `oproj` o_proj output (before post-attn norm) | 2D | post-attn norm backward | saved |
| `x_mid` residual after attention | 2D | pre-MLP norm backward | saved |
| `m` pre-MLP norm output | 2D | gate/up weight gradient | recomputed |
| `gu` gate and up outputs | 4F | SwiGLU backward | saved |
| `h` = silu(gate)·up | 2F | down weight gradient | recomputed |
| `y` down output (before post-MLP norm) | 2D | post-MLP norm backward | saved |
| f32: 4 norm rstd, q/k-norm rstd (H+KV), attention lse (H) | 4(4+2H+KV) | backward of norms, attention | saved |

remat=layer keeps only `x_in` per layer and recomputes the whole layer forward (GEMMs and attention included) into one
transient layer slot before that layer's backward: levanter's policy. Cost: +1/3 trunk GEMMs, +1x attention forward.

Rule (`choose_plan`): take remat=all at the largest `micro_seqs` that divides the per-rank batch and fits. If that
microbatch holds fewer than 16,384 tokens, try remat=min and take it when it allows at least twice the tokens.
remat=layer is chosen only when it is requested (strict preset) or when nothing else fits. Result: 130m, 300m, 520m,
1.2B and pls d48 (option B) run remat=all.

### 2.4 The step as one graph per regime

```
host:   barrier(k) -> cudaMemcpyAsync(tokens of step k: pinned slot -> d_tokens) -> launch g_step[regime(k)]
g_step (streams: compute, comm, side; all forked from compute inside the capture):
  comm:    bf16 all-gather of the parameters updated by their owners in step k-1, per layer bucket, in forward order
  compute: for each microbatch j:  doc_meta -> input layer -> L x block forward (layer l waits on its all-gather event)
                                   -> final norm -> output head (+ readouts interleaved with) L x block backward
                                   -> input-layer backward
           last microbatch: GradReducer reduces each bucket to its owners on `comm` as the bucket completes, then
           computes that bucket's sum-of-squares partials on `comm`
  opt:     [wait: external event ckpt_copy_done]
           AdamH group (lm_head, heads): starts when the head buckets are reduced
           MuonH group: starts when the trunk buckets are reduced (momentum -> NS -> hyperball), overlapping the
                        input-layer backward and its reduction
           Adam group (embedding, gains): starts when the embedding bucket is reduced
           metrics all-reduce (loss sums, readout losses, non-finite flags) -> StepRecord -> k++
```

- One graph instead of a forward/backward graph plus an optimizer graph: the clip groups and the embedding reduction
  overlap Newton-Schulz only inside one graph, and the bf16 all-gather of step k-1's update overlaps step k's forward.
  The checkpoint copy needs no second graph: the optimizer section waits on the external event `ckpt_copy_done`,
  which the host records on the copy stream every step (right after the D2H copy on a checkpoint step, at once
  otherwise).
- Regimes (`StepRegime`, model.hpp): a list of step ranges with static flags known on the host: statistics step
  (`k % 10 == 0 && k > 1`), readout phases (`readout.off`, late start), dropout from pass p, sampled-softmax stages,
  freeze-at steps. Each distinct regime is one instantiated graph in the same arena (the plan is the maximum over
  regimes); the baseline has two (plain, stats). Data-dependent switches (sampled-softmax overflow fallback) are CUDA
  conditional nodes.
- The update index k lives in device memory; schedule lookups read `table[k]`. A capture lint rejects any
  step-dependent host scalar recorded as a kernel argument (every such value must be a device table or a regime flag).
- Graph capture failures are fatal at start. `exec.cuda_graph=false` exists only for `--profile-ops` and the f32
  policy (§3.0).
- Warm-up check (deterministic mode): before the first real step the trainer runs update k eagerly on a staged copy of
  the state and as a graph, and requires bitwise-equal checksums. The S7 test extends this to k0, k0+1 and every
  regime boundary.

### 2.5 Memory plans

All plans: 4 GPUs, the reference global batch, 72 GiB budget, ZeRO-1 for every group (option B), no varlen padding rows
(§3.4), CE chunk from the rule of §3.8. Numbers from the planner formulas (`scripts/design_budget.py` version 2, run on
della-vis1); the C++ planner must reproduce them (stage S0 test).

#### 130m: micro 32 sequences per GPU (131,072 tokens, one microbatch), remat=all, CE chunk 32,768, 29.26 GiB

| buffer | count | bytes total |
|---|---|---|
| param.f32 owner shards (muonh 22.8 MB, adam 65.7 MB, adamh 65.7 MB) | 3 | 154,154,240 |
| param.bf16 (all 154,154,240 parameters, replicated) | 1 | 308,308,480 |
| grad.f32 (all) | 1 | 616,616,960 |
| muonh.momentum.f32 owner shard | 1 | 22,806,528 |
| adam.mu,nu + adamh.mu,nu f32 owner shards | 2 | 262,695,424 |
| ns.workspace.f32 | 1 | 37,748,736 |
| sched tables, rope, tokens, metrics ring | 4 | 69,285,360 |
| mb.targets, mb.pos i32, segment table | 3 | 1,051,652 |
| act.L*.x_in, a, attn_o, oproj, x_mid, m, y (each [T, 512] bf16) | 6 x 7 | 5,637,144,576 |
| act.L*.qkv [T, 1536], qk_rot [T, 1024] | 6 x 2 | 4,026,531,840 |
| act.L*.gu [T, 3584], h [T, 1792] | 6 x 2 | 8,455,716,864 |
| act.L*.f32 stats; final residual, final-norm output, rstd | 7 | 357,040,128 |
| transients (dx x2, da/dm/do, dqkv, dO, dgu, dh, partials) | 8 | 2,357,198,848 |
| ws.fa3 dq_accum f32 [8, T, 64], softmax_d, lse_log2 | 3 | 276,824,064 |
| ws.ce logits bf16 [32768, 128256] + row stats | 2 | 8,668,577,792 |
| ws.embed_bwd sort, cublaslt x2 | 3 | 169,869,312 |
| **total** | | **31,421,570,804 (29.26 GiB)** |

Subtotals: P 1.37 GiB, A 17.21 GiB, T 2.20 GiB, W 8.49 GiB. With the f32 logits store of §3.8 at 8,192 rows: 25.17 GiB.

#### pls d48 (width 512, 48 layers, 47 separate readout heads + lm_head, 3,400,267,776 parameters)

Option B (default), ZeRO-1 for every group: micro 8 sequences (32,768 tokens, 4 microbatches), remat=all, CE chunk
16,384, **67.25 GiB**: P 28.43 GiB (head masters 3,152,019,456 B and head moments 6,304,038,912 B per rank; bf16 copies
6.80 GB and f32 gradients 13.60 GB replicated), A 33.98 GiB, T 0.59 GiB, W 4.26 GiB (CE logits [16384, V] bf16 4.20
GB). With the f32 logits store at 8,192 rows: 67.19 GiB.

Option A, pure DDP: micro 4 sequences (16,384 tokens, 8 microbatches), remat=min, CE chunk 8,192, **70.34 GiB**: P 56.42
GiB, A 11.36 GiB, T 0.35 GiB, W 2.21 GiB.

Both fit. Option B is faster by budget (6.57 s vs 6.75 s per step: remat=all, half the microbatches, half the head
gradient read-modify-writes, optimizer work divided by 4); option A has no reduce-to-owner path. Rule: run both for 20
steps on H100; keep the faster (O6).

If 47 more probe heads share the trunk (research need "several variants share one trunk"), their AdamH state adds
47 x 128,256 x 512 x 18 B = 55.5 GB replicated, 13.9 GB per rank sharded, plus 12.4 GB of replicated f32 gradients;
the planner then also reduce-scatters vocab-sized gradients after every microbatch (ZeRO-2 for those tensors only,
`optim.zero2_vocab=auto`), which removes the replicated gradient at the cost of one reduce-scatter per microbatch.

#### Other sizes (totals; full tables from `marincpp plan`)

| size | micro seqs/GPU (tokens) | microbatches | remat | CE chunk | P | A | T | W | total |
|---|---|---|---|---|---|---|---|---|---|
| 300m | 32 (131,072) | 1 | all | 32,768 | 2.56 | 51.24 | 3.29 | 8.62 | 65.71 GiB |
| 520m | 8 (32,768) | 8 | all | 16,384 | 4.94 | 31.75 | 1.05 | 4.45 | 42.19 GiB |
| 1.2B | 8 (32,768) | 8 | all | 16,384 | 11.78 | 42.34 | 2.09 | 4.70 | 60.91 GiB |
| 520m strict | 32 (131,072) | 2 | layer | 8,192 | 4.94 | 6.99 | 9.41 | 3.19 | 24.53 GiB |
| 1.2B strict | 32 (131,072) | 2 | layer | 8,192 | 11.78 | 9.98 | 18.80 | 4.19 | 44.76 GiB |

Every size fits. The strict rows are the plans `--size S --preset levanter_strict` runs: levanter's 2 x 32
microbatching (§3.0), the CE chunk fixed at 8,192 rows (§3.8), and the per-microbatch head-gradient accumulator (V x D
f32: 0.53 GB at 520m, 1.05 GB at 1.2B, Microbatch region, included in A).

Fit rule: a plan fits when its arena (`device_bytes`: payload, 256-byte alignment, and the 4 KiB guard bands before and
after every buffer when `exec.debug_arena` is on) is at most the budget; `choose_plan`, `marincpp plan` and the
trainer use this one measure. The tables above list payloads (alignment adds under 1 MB).

## 3. Kernel plan

### 3.0 Numerics policy

Three independent axes (core.hpp `Numerics`):

1. `exec.policy` = `bf16` (training) or `f32` (parity tests T2.2 and T2.4, gradient gate V1). The f32 policy stores
   every activation and the residual stream in f32, runs every GEMM in true f32 (`CUBLAS_COMPUTE_32F_PEDANTIC`), uses
   the `RefF32` attention backend (f32 math, 1024-key blocks in ascending order, f32 running statistics) and CE
   implementation A with f32 x and W, and links the f32 instantiations of the elementwise kernels into the main
   library. It runs eagerly (no graph). The levanter f32 fixture must be produced under
   `jax.default_matmul_precision("highest")`; `tools/levref/precision.py` records the precision of
   `cache/levref/130m/step0_f32_debug4`, and the fixture is regenerated if it was made at default precision.
2. `exec.rounding` = `levanter` (copy every class-P rounding point local to one kernel: bf16 casts between the
   elementwise ops of RMSNorm, RoPE, SwiGLU, the bf16 multiply of q and dq by the scale, bf16 dlogits, bf16 parameter
   gradients per microbatch for trunk GEMMs and norm gains) or `accurate` (f32 inside every kernel, one rounding at its
   output; f32 gradient accumulation directly in the GEMM).
3. The class-P items where the fast configuration departs from levanter, each with its own knob, all reverted by the
   preset `levanter_strict`:

| item | fast default | strict (levanter) | knob |
|---|---|---|---|
| attention statistics | FA3, f32 softmax statistics | LevRef (bf16 running statistics, block 1024) | `exec.attention` |
| gradient reduction dtype | f32 | bf16 | `exec.grad_reduce` |
| embedding backward accumulator | f32 sorted segment sum | bf16 running accumulator in ascending position order (sorted, deterministic) | `exec.embed_accum` |
| CE head dW | f32 accumulation across chunks and microbatches | bf16 per 8,192-row chunk, f32 sum, bf16 per microbatch | `exec.ce_dw_round`, `exec.ce_chunk_rows=8192` |
| CE logits store | what T1.0(2) shows (bf16 or f32) | same | `exec.ce_logits` |
| microbatching at 520m / 1.2B | 8 x 8 sequences per GPU, remat=all | 2 x 32, remat=layer | `exec.micro_seqs`, `exec.remat` |

   With bf16 per-microbatch parameter gradients, 8 microbatches round 8 times where levanter rounds twice; the strict
   plan removes that difference too.
- Same in all modes (class M, model.md §7): eps inside the rsqrt; the gain multiplies after the normalization;
  `bf16(1/sqrt(hd))` as the attention scale; absolute RoPE positions; EOS-derived segments; f32 logits and lse in the
  CE forward; f32 masters and optimizer.
- Acceptance (parity plan, oracle rule): for each class-P output, `rel(nano, f64 oracle) <= 1.25 x rel(levanter, f64
  oracle)`. `accurate` mode and every fast-default deviation become the training default only after the n >= 4 A/B of
  numerics.md §1; T5.2 run with the fast configuration is that A/B for the fast items.
- FMA contraction: host code builds with `-ffp-contract=off`. The device translation units of the levanter-mode
  elementwise kernels, the optimizer, the eval aggregation and the init build with `--fmad=false`; each formula writes
  its contractions explicitly (`__fmaf_rn`, or `__fmul_rn`/`__fadd_rn` to forbid one) as the T1.0 XLA PTX dump shows
  for that formula. Means are `sum / n`, never `sum * (1/n)`. `accurate`-mode kernels, GEMM and FA3 units keep nvcc's
  default contraction.

### 3.1 Schedule of one baseline layer (stage 1, fused)

T = tokens in the microbatch. Every fused kernel has an unfused reference path (the version-1 kernels) kept in the
library; in `levanter` mode the fused output must equal the unfused output bitwise, because every fusion below stays
inside one output tile and copies the same rounding points.

Forward:

| # | op | kernel | out (saved*) |
|---|---|---|---|
| F1 | input RMSNorm (layer 0; other layers get `a` from F10 below) | `k_rmsnorm_fwd` | a*, rstd_in* |
| F2+F3 | QKV projection with QK-norm, RoPE and q scale in the epilogue (N tile holds whole heads; raw qkv also written) | CUTLASS SM90 GEMM, epilogue visitor | qkv*, q_rot*, k_rot*, rstd_q/k* |
| F4 | attention | FA3 varlen forward (persistent) | attn_o*, lse* |
| F5+F6 | o_proj; at D = 512 the sandwich tail (post-attn norm + residual + pre-MLP norm) in the epilogue with a full-row 128 x 512 tile; at D > 512 a separate `k_sandwich_tail_fwd` | CUTLASS / cuBLASLt | oproj*, x_mid*, m*, rstds* |
| F7+F8 | gate+up projection with SwiGLU in the epilogue (Wg, Wu rows interleaved in 64-row blocks so a tile holds matching gate and up columns) | CUTLASS | gu*, h* |
| F9+F10 | down projection; sandwich tail as F5+F6 (fused at D = 512) | CUTLASS / cuBLASLt | y*, x_out, a_next*, rstds* |

Backward:

| # | op | kernel | out |
|---|---|---|---|
| B10 | through F10 (residual + next input-norm backward, then post-MLP norm backward) | `k_sandwich_tail_bwd` | dX_mid, dy, gain partials |
| B9+B8 | down dgrad with SwiGLU backward in the epilogue (loads gu, writes dgu; dh never exists); down wgrad | CUTLASS | dgu, dWd |
| B7 | gate/up dgrad, wgrad | CUTLASS / cuBLASLt | dm, dWgu |
| B6 | through F6 | `k_sandwich_tail_bwd` | dX_in-partial, doproj |
| B5 | o_proj dgrad with FA3's preprocess `rowsum(dO∘O)` per head in the epilogue; wgrad | CUTLASS | dattn_o, softmax_d, dWo |
| B4 | attention backward (no FA3 preprocess launch) | FA3 backward, fixed scheduler (§3.4) | dq_accum f32, dk, dv |
| B3 | dq postprocess fused with q-norm/RoPE backward (reads FA3's f32 dq accumulator, rounds as FA3's postprocess would, then dq *= scale in bf16, inverse rotation, norm backward); k side in FA3's dK epilogue (or `k_qknorm_rope_bwd` for k when the epilogue path is not built) | `k_dq_post_qknorm_rope_bwd` | q, k columns of dqkv, dgq/dgk partials |
| B2 | QKV dgrad, wgrad | CUTLASS / cuBLASLt | da, dWqkv |
| B1 | input-norm backward + residual | fused into B10 of the layer below; layer 0 uses `k_rmsnorm_bwd_add` | dX |

Weight-gradient GEMMs write f32 with beta = 1 in place across microbatches. Under `levanter` their epilogue computes
`D = C_f32 + f32(bf16(acc))` on the complete accumulator (after any split-K or stream-K fixup, which reduces in f32), so
the bf16 rounding of levanter's per-microbatch gradient costs no transient and no extra pass.

### 3.2 Input layer (embedding)

- Forward `k_embed_fwd`: bf16 gather `out[t] = E_bf16[token[t]]`, no scale (model.md §2).
- Backward `embed_bwd_sorted`: CUB radix sort of (token, position) pairs over `ceil(log2 V)` key bits (17 at V =
  128,256; 24 for 12.8M-row tables), run boundaries, one CTA per distinct token summing its rows in ascending position
  order, added to the f32 gradient. `exec.embed_accum=f32` sums in f32 (default); `bf16_seq` keeps a bf16 running sum
  in the same order (levanter's precision, deterministic). No atomics. The run list doubles as the touched-row mask for
  lazy embedding updates (§4.5). Cost ~0.2 ms at 131,072 tokens.
- The embedding is the default `InputLayer` (§9.3); composed rows, n-gram tables and byte inputs are other input layers.

### 3.3 RMSNorm, QK-norm, residual

- Statistics in f32, eps inside the rsqrt, gain after the normalization in both rounding modes.
- Levanter rounding points (numerics.md §3.2): `y = bf16(x32·r)`, `out = bf16(g_bf16·y)`; backward
  `a = bf16(g_bf16·dy)`, `dx = bf16(a32·r + c·2·x32)` with `c = (s·(-0.5·(r/(ms+eps))))/n`, in that operation order and
  with the contraction of §3.0.
- Gain gradients: per-CTA f32 partials, then `k_reduce_rows` in fixed order; under `levanter` the per-microbatch result
  is rounded to bf16 before it is added to the f32 gradient.
- QK-norm per (token, head) over hd with one gain per layer shared by all heads, before RoPE (model.md §2.1).
- The residual stream stays bf16 (class P; an f32 residual is an A/B candidate, not stage 1).

### 3.4 RoPE and attention

RoPE: NeoX half split; tables `cos, sin` bf16 `[S_max, hd]` built on the host from the f32 `inv_freq` of numerics.md
§3.3 and the f32 angle `f32(f32(pos)·inv_freq)`; cos/sin evaluated in double, rounded to f32, then bf16. T1.3 compares
the tables bitwise with levanter's; on a mismatch the run loads levanter's tables from the fixture (O4). Positions are
absolute window positions.

Backends behind `AttentionBackend` (attention.hpp):

| backend | what | status |
|---|---|---|
| FA3 (default) | vendored FA3 `hopper/` at `767386d` with CUTLASS `7127592`; torch-free entry points fill `Flash_fwd_params`/`Flash_bwd_params`; forward = FA3 varlen with `VarlenDynamicPersistentTileScheduler`; backward = one of the two fixed schedulers below | fwd and bwd instantiations for hd64 and hd128 compile with CUDA 12.8 and 13.0 on della-vis1 without torch |
| CudnnSdpa | cuDNN 9.19 SDPA via cudnn-frontend 1.32 with THD ragged offsets | candidate |
| LevRef | levanter's numerics (bf16 running statistics, block 1024, bf16(-1e10) fill, bf16-precision lse) | strict preset, tests |
| RefF32 | f32 math, block 1024, ascending key blocks | f32 policy, tests |

Document masking: `k_doc_meta` turns the microbatch rows into document segments (a segment starts after every EOS token
and at every row start). Each segment is causal inside. This equals levanter's mask (`k <= q` and same segment) for any
window length, so the `SEQ_LEN % 1024` constraint of the Tokenizer Size runs disappears.

The FA3 backward problem (perf review 1). FA3's causal varlen backward uses `SingleTileBwdLPTScheduler`: a
non-persistent grid of `ceil(max_seqlen_k / kBlockN) x h x num_batch` CTAs (`tile_scheduler.hpp:403`,
`flash_bwd_launch_template.h:176-179`), with `num_batch` and `max_seqlen` host integers fixed at capture. With version
1's padded batch (`rows x 24` segments, `max_seqlen` 4096), 130m launches 196,608 CTAs per call of which about 8,880 do
work (5.19 segments per row measured), and the GQA `dk/dv_accum` buffers carry `rows x 24 x 128` padding rows that must
be zeroed every call. Version 2 never runs that grid in training. Two fixed schedulers plug into the FA3 backend
(`exec.attn_bwd_sched`):

| scheduler | what | padding | launch shape |
|---|---|---|---|
| `persistent` | the backward gets a varlen persistent scheduler like the forward's: grid = number of SMs; a prepare kernel inside the graph reads `cu_seqlens` and the device segment count, builds the prefix sum of n-blocks per segment in LPT order; CTAs pull work items | accumulator rows still sized for the maximum segment count | fixed (SM count) |
| `doc_causal` | patch to FA3 forward and backward: batch = rows, a per-row segment-start table; tiles wholly outside a segment are skipped, boundary tiles masked by segment id | none: `dq_accum [H, T, hd]`, `dk/dv_accum [KV, T, hd]` | `(S/128) x H x rows` (8,192 CTAs at 130m), all useful |

- GQA without memsets: in deterministic mode the dk/dv accumulation over the G query heads of a group is ordered; the
  first head in that order stores instead of adding, so the accumulators need no zeroing.
- Every FA3 glue allocation of `flash_api.cpp` (semaphores, accumulators, `prepare_varlen_num_blocks` metadata) is an
  arena buffer; required zeroing is a memset node in the graph.
- The attention FLOPs of §12 use the measured computed/causal ratio 0.504 of the 192 x 128 forward tile on real rows
  (version 1 assumed 0.45).
- Decision rule (O1): E1b on H100 times, on the real segment rows (`bench/e1_attention/segments`), the padded
  version-1 grid, `persistent` and `doc_causal`, each with deterministic on and off, at the 130m, 520m and 1.2B shapes.
  Default = the faster of `persistent` and `doc_causal` summed over the shapes; ties within 3% go to `doc_causal` (no
  padding buffers). cuDNN replaces FA3 only if it beats the chosen FA3 path by more than 5% and passes T1.4. The E1
  results already queued (real `cu_seqlens`, `max_seqlen` = the real maximum, deterministic off) do not represent the
  trainer and do not decide D1 alone.

Scale: under `levanter`, the QKV epilogue multiplies q by the bf16 scale in bf16 and FA3 gets `softmax_scale = 1`; the
dq postprocess multiplies dq by the scale in bf16 (numerics.md §3.4). Under `accurate`, FA3 gets the bf16 scale.

GQA (520m, 1.2B): head h reads kv head h / G; `Wq` rows are ordered (kv, g, d) as levanter's `q_proj`; FA3 uses
PackGQA in the forward.

Depth keys (moda): attention with extra per-token depth keys and values in the same softmax needs an `extra_kv` input
in the FA3 forward and backward. The interface field is reserved (`AttnFwdArgs::extra_kv`); the knob value `moda` is
not accepted until that kernel exists and passes its gates.

### 3.5 SwiGLU

Levanter rounding (numerics.md §3.5): every op bf16, `dx = (dsilu·s) + ((x·dsilu)·sp)` with `sp = s·(1 - s)`;
`accurate`: one rounding. Stage 1 runs the fused forms of §3.1 (forward in the gate/up epilogue, backward in the
down-dgrad epilogue). The epilogue first rounds the accumulator to bf16 (the GEMM output levanter stores), then applies
the bf16 op chain, so the bits equal the unfused kernels.

### 3.6 Residual stream and sandwich norms

The 7 RMSNorm kinds (in, q, k, post_attn, pre_mlp, post_mlp, final) map to `k_rmsnorm_*`, the QKV epilogue / dq
postprocess, and `k_sandwich_tail_*` (or the D = 512 o_proj/down epilogue). Pre-LN blocks pass no post-branch gain.

### 3.7 GEMMs

- bf16 A/B, f32 accumulate, bf16 D for activations, f32 D with beta = 1 for weight gradients.
- Layout: activations `[T, features]` row-major, weights `[out, in]` row-major (levanter's out_first): forward NT,
  dgrad NN, wgrad TN.
- Algorithm choice (`GemmCache`): a timed search per shape at plan time is the default (about 40 shapes, a few seconds
  at startup), over every deterministic candidate: cuBLASLt algorithms with split-K reduction scheme NONE, INPLACE,
  COMPUTE_TYPE or OUTPUT_TYPE (never atomics), CUTLASS SM90 kernels including stream-K with deterministic fixup, and
  registered custom kernels. Under `levanter` with bf16 rounding of a split result, only schemes that reduce in f32
  before the rounding are candidates (COMPUTE_TYPE, INPLACE with f32 D, stream-K). The cuBLASLt workspace is sized from
  the chosen split counts. Choices are stored in `<run>/gemm_algos.bin`; resumed segments reuse them.
- Why the search matters (perf review 2): the weight-gradient GEMMs at D <= 1024 have tiny outputs and K = 131,072
  (130m: [1536,512], [512,512], [3584,512], [512,1792] = 24, 8, 56, 28 tiles of 128 x 256 on 132 SMs). Without split-K
  or stream-K each is one partial wave: 18-34 ms per step at 130m instead of about 10 ms. The `e1/gemm.py` wgrad sweep
  at M = 131,072 supplies the data.
- Custom CUTLASS kernels are first-class: every epilogue fusion of §3.1, the levanter wgrad epilogue, and K = 512
  projections where they beat cuBLASLt. A custom kernel takes a shape when it is the fastest deterministic candidate.
- Newton-Schulz GEMMs (§4.2): TF32, true f32 (FFMA), or 3xTF32 split precision (f32-level error at about twice the
  FFMA rate); the X·Xᵀ and A·A products compute only upper-triangle tiles (28-33% fewer FLOPs).

### 3.8 Fused linear cross-entropy

Goal: never materialize full-vocab f32 logits in the bf16 policy; one numeric path for every microbatch size; a
multi-head variant whose memory does not grow with the number of heads.

Per chunk of `ce_chunk_rows` rows:

1. `k_ce_logits_stats` (custom SM90 GEMM: persistent, TMA, warp-specialized with ping-pong consumer warpgroups so one
   warpgroup's epilogue overlaps the other's mainloop): `z = x Wᵀ` with f32 accumulators; the epilogue writes the logits
   store (bf16, or f32 when `exec.ce_logits=f32`) and, per row and 128-column tile, the f32 partial (max, sum exp(z -
   max)) from the f32 accumulators, plus the f32 `z[target]`.
2. `k_ce_finalize_rows`: merges the partials into `lse` and `CE = lse - z[target]` (levanter's forward from f32 logits),
   per-token loss and a per-chunk weighted sum.
3. The gradient `p = exp(z_s - lse)`, `dz = p·g`, `dz[target] -= g`, `dzb = bf16(dz)`, where `z_s` is the stored logit
   (bf16 or f32), then `dx = dzb W` (bf16, one GEMM over all of V) and `dW (+)= dzbᵀ x`. Three stages place this work:

| stage | dz | extra HBM traffic beyond GEMM operands | exps per logit |
|---|---|---|---|
| B1 | `k_ce_dlogits_inplace` rewrites the store as bf16 dz; dgrad and wgrad read dz | 6V B/token (store write, dz read + write) | 2 |
| B1.5 | the dgrad kernel computes dz from the store on the fly, uses it, and writes bf16 dz for the wgrad | 4V B/token | 2 |
| B2 | dgrad and wgrad both compute dz from the store in a transform warpgroup (store tile in shared memory -> bf16 dz -> shared memory -> WGMMA), as FA3 overlaps its softmax | 2V B/token (store write) | 3 |

Kernel requirements, all stages (perf review 3):
- Special-function limit: an H100 SM issues 4,096 bf16 tensor FLOP but 16 `exp2` per clock. At D = 512 one logit is
  0.25 clock of tensor work, its exp 0.0625 clock plus about 0.06 clock of f32 ops, so the non-tensor work is about 50%
  of the mainloop. It must overlap the mainloop (ping-pong or a transform warpgroup); part of the exps may run as an
  FMA polynomial on the f32 pipes to balance the two units.
- Output tile N = 512 for dgrad and wgrad (two consumer warpgroups sharing the A tile, or a 2-CTA cluster with TMA
  multicast), covering all of D at D = 512. With N = 256 every dz (or store) element is read D/256 times and the GEMM
  runs at 256 FLOP/B, below the 295 FLOP/B ridge.
- The dgrad and wgrad of B1 start as cuBLASLt calls (the first working version) and are replaced by the N = 512
  kernels.

Chunk size rule: each chunk does a read-modify-write of the f32 head gradient (2·V·D·4 B, 525 MB at D = 512). The
planner takes the smallest chunk (8,192 x 2^i rows, at most the microbatch) whose read-modify-write costs at most 0.5%
of the step, capped by what fits with 2 GiB headroom: 130m 32,768; 300m 32,768; 520m and 1.2B 16,384; pls option B
16,384 (cap). The strict preset fixes 8,192. A microbatch below 8,192 tokens runs as one chunk of its tokens rounded up
to a multiple of 128 (any `model.seq_len` resolves to a valid chunk). An explicit `exec.ce_chunk_rows` is the chunk
the planner's fit test uses while it chooses `micro_seqs`.

Logits store dtype (parity review 3): T1.0 item 2 reads whether XLA keeps the bf16 output of levanter's backward logit
dot. `exec.ce_logits=auto` resolves to that fact (bf16 if kept, f32 if XLA folded the convert into an f32 GEMM output);
until T1.0 has run, `auto` refuses to resolve in parity runs. The f32 store doubles the chunk buffer (+2.1 GB at 8,192
rows; every plan still fits, §2.5). A recompute variant (forward keeps only statistics; dgrad and wgrad recompute z in
f32 for +2DV FLOPs) is the fallback if the f32 store is slower than the recompute.

The gradient is computed in the forward pass; g = `weight[t] / (B_global x (S-1))` in f32 (the global token mean), so
the math does not depend on the microbatch split; only the summation order of dW does.

Implementations behind one call (`CeImpl`): A (cuBLASLt logits in f32 + row kernel; the reference, and the f32-policy
path), B1, B1.5, B2. Rule (O2): per D in {512, 768, 1024, 2048}, the fastest implementation that passes T1.5.

Multi-head (pls): the readout of layer l runs when the backward reaches layer l: recompute the readout norm into a
transient, run the fused CE (forward + gradient), accumulate `dW_l` into head l's f32 gradient, and add
`w_l · d(norm input)` into layer l's residual gradient. All heads share the one CE workspace. A readout whose backbone
gradient is statically off (probe, detached backbone, late start before its start step) passes `dx = nullptr`: the dX
GEMM does not run and the readout norm backward computes only its gain gradient (about 1/3 of the head FLOPs saved).
Late start is a regime (§2.4), not a device scalar multiplying a computed dX.

Head dW rounding: the fast default accumulates every chunk in f32 directly into the head gradient. The strict mode
copies levanter: each chunk's dW is rounded to bf16 in the wgrad epilogue and added into a per-microbatch f32
accumulator (one shared V x D buffer, reused by every head), which is rounded to bf16 once per microbatch and added to
the f32 gradient (one extra pass per head per microbatch, about 57 ms at pls option B). T1.5 decides whether
per-chunk rounding alone is acceptable.

Eval uses the forward-only path (steps 1-2, no store): GEMM-bound, 2DV FLOPs per token per readout.

### 3.9 Activation recompute policy per tensor

Default none (§2.3). remat=min recomputes exactly `a`, `qk_rot`, `m`, `h` in `Block::recompute()` immediately before
that layer's backward. remat=layer recomputes the whole layer. Chosen per size by `choose_plan`: every default plan is
remat=all; pls option A is remat=min; the strict plans are remat=layer.

## 4. Optimizer

All of it runs on device inside `g_step` and reads k from device memory (`optim_kernels.hpp`, `optim.hpp`).

### 4.1 Groups as data, clipping

- An optimizer group is a `GroupSpec`: rule (MuonH, AdamH, Adam, Frozen), lr table (main, adam, or its own), eps, b1,
  b2, clip group and max norm, per-tensor gradient scale and update scale (lns step equivalence), and an optional
  post-update kernel (attnres query clip). Metric arrays are sized from the number of groups at plan time.
- Baseline routing (optim.md §2.1): every transformer Linear weight (q, k, v, o, gate, up, down) -> `muonh`; `lm_head`
  and separate heads -> `adamh`; embedding and every RMSNorm gain -> `adam`. Variants add groups, e.g. `depth_adam`
  (hc/mhc/attnres parameters, eps 1e-8, as `DepthArchMuonHConfig`).
- Clip groups (optim.md §3.1): each group's norm = sqrt of the sum over its tensors. Sum-of-squares partials per 64 KiB
  slice are computed on the comm stream right after each bucket's reduction, over the slices this rank owns (§6); the
  per-group partials of all ranks are summed by one small all-reduce of `n_groups` floats; `k_clip_state` writes
  `{gn, clip}` with clip iff `!(gn < max_grad_norm)`. Updates apply `(g / gn) * max_grad_norm` while reading g.
- Each clip group starts its update as soon as its own buckets are reduced (§2.4); only the Adam group waits for the
  embedding bucket.
- Weight decay: none (optim.md §7). `optim.weight_decay` defaults to 0, applies only to groups that list it, and
  `validate()` rejects a nonzero value with a muonh_qwen3 preset unless `optim.wd_variant=true` (a config copied from
  levanter's logged 0.1 would otherwise apply it).

### 4.2 MuonH

1. `k_muon_momentum` (owner rank only): `buf = mu·buf + g'`, `u = mu·buf + g'` (Nesterov without the (1-mu) factor),
   written per layer matrix into the NS buffer in the wide orientation, plus per-matrix partial `||u||²`.
2. `k_ns_normalize`: `X = X / (||u|| + 1e-5)`.
3. Newton-Schulz, 5 iterations, "simple" coefficients (3.4445, -4.7750, 2.0315) as f32 constants (optim.md §5.3):
   `A = X Xᵀ; cA = c·A; B = b·A + cA·A; X = a·X + B·X`, batched by oriented shape over the matrices this rank owns.
4. Hyperball projection per layer matrix in three passes (optim.md §4.4); n recomputed from the current p every step;
   at eta = 0 it still runs (rounding-level renormalization, gate a7(ii)).

Precision (O3, T1.0 item 5): `optim.ns_precision=auto` resolves to TF32 if levanter ran TF32, else to f32. For f32 the
implementation is 3xTF32 with upper-triangle products unless T1.6 shows it outside `rel <= 1e-5`, in which case true
FFMA f32 is used. Budget at 1.2B with ownership over 4 ranks: TF32 43 ms, true f32 214 ms, 3xTF32 75 ms.

Distribution: NS always runs on the owner of each whole matrix (ZeRO-1, §6), so version 1's "distribute if > 1%" rule
is gone. Owners are assigned by LPT on NS cost plus elementwise bytes.

### 4.3 AdamH and Adam

- AdamH (optim.md §5.1), whole-matrix scope for `lm_head` and each separate head, three passes; for row-sharded
  tensors the whole-matrix norms are per-shard partial sums plus a scalar all-reduce (same math, class S order). Main
  lr eta.
- Adam (optim.md §5.2): one fused multi-tensor pass; `u = (-adam_lr_k)·((mu/bc1)/(sqrt(nu/bc2) + eps))`.
- Constants: `1-b1`, `1-b2` are the f64-then-f32 values (0x3DCCCCCD, 0x3CA3D70A) (optim.md §5.3).
- Fused multi-tensor kernels: one launch per pass over a prefix table of 64 KiB slices of the owned tensors.

### 4.4 Schedules

The schedule tables of record are a fixture: `tools/levref/schedules.py` runs the jitted optax schedules of the levanter
config on a GPU (XLA:GPU's f32 cos and pow, the same code on sm_80 and sm_90; T0.7 checks the H100 logged values) and
dumps `sched_main(k)`, `sched_adam(k)` for k = 0..N-1 and `1 - b^t` for t = 1..N, for each config in use. marincpp loads
it (`optim.schedule_fixture`, hashed by content). The C++ formula of optim.md §6 (`build_schedules`, host, exact f32
operation order, `-ffp-contract=off`) runs beside it as a cross-check that counts ulp differences; it never overrides a
fixture value. A run without a fixture uses the formula and records `schedule_source: formula`; parity runs
(`run.parity=true`: T2-T6; they also require the T1.0 facts file and a verified init) refuse to
start without one. T0.7 requires the fixture to equal every logged k bitwise, including 130m k = 4958 and 1.2B
k = 22887.

### 4.5 Per-row control

`RowPartition` (optim.hpp): a map `block_of_row[]` plus, per block, an lr-scale schedule, a frozen flag and its own
norm scope. It applies to Adam and AdamH. The mask is applied in `sumsq_partials` (frozen rows enter neither the clip
norm nor the AdamH norms) and in the moment updates (frozen rows get no update), as `freeze_rows_muonh.py` does: in
block mode the kept rows are [0, lo) plus the last 256 specials with their own norm, the rare rows another block with
their own norm and lr scale. Lazy embedding updates take the touched-row mask from the run list of `embed_bwd_sorted`.
Gate: one T1.7-style update against the JAX optimizer, rel <= 1e-6.

## 5. Data

### 5.1 Offline converter (Python, run once on della-vis1)

`tools/convert/levanter_to_flat.py`, built on `bench/n1_loader/nmloader.py` (verified: 103 of 103 checks):

- `$NM_DATA/fineweb-edu-10B-marin128k/tokens.u32`: the 10,000,000,738 tokens in cache order (40.0 GB).
- `segs_per_seq.u8` and `max_segs_per_seq` in meta.json.
- `docs.u64`: document start offsets, and per document its byte range in the source text when the source has it
  (needed to map resegmented and repacked sequences across tokenizers, §5.5).
- `meta.json`: counts, BOS/EOS, vocab, sha256 of every file, source path and its shard ledger.
- `paloma-<tokenizer>/`: the windows of the 16 domains in component order, tags, per-domain sha256, the levanter byte
  table, an exact UTF-8 byte table, and per domain the exact source bytes covered by the scored windows (§8.1).
- Tokenizer Size inputs: truncated-vocab caches, byte caches and materialized pass caches use the same layout.

### 5.2 C++ reader

`TokenFile` maps `tokens.u32` read-only (`MAP_SHARED`). A window is 16 KiB; a 128-row step reads 2 MiB at random
offsets; 130m needs about 11 MB/s on average.

### 5.3 Exact levanter order

`LevanterOrder` implements data.md §4.4 in C++: threefry2x32 (partitionable), `split`, `fold_in`, `bits32`, the two-round
stable-sort `permutation` of each 2048 block, the "restart" wrap, and the LCG with the copied constants. Tests: T0.1,
T0.2, T0.3, T0.4. At run time the loader logs H(B, k) at k = 0 and at every eval step (data.md §7 guard).

### 5.4 Prefetch ring, pinned slots, async H2D

- Data depth is separate from launch depth. The order is a pure function of (plan, step), so the reader pool fills a
  pageable ring of `data.prefetch_steps` = 256 future steps (512 MiB at B = 128) with 16-32 threads (`pread` into the
  ring; 16 threads gave about 20,000 random reads per second in `io_probe`). n1_loader measured cold-batch p90 latencies
  of 250-1,000 ms on GPFS; 256 steps cover 48 s at 130m.
- 3 pinned slots carry the next steps to the devices: worker r issues `cudaMemcpyAsync` from its rows of the slot and
  records an event; the slot is refilled after every rank's event completes. Launch lookahead stays 2 graphs, so the
  SIGTERM drain of §1.5 is unchanged.
- Targets, loss weights, positions and segments are computed on device by `k_doc_meta`; the host copies only tokens.

### 5.5 Data plans, forks and passes

The order of a run is a pure function of `(DataPlan, step)`; no loader state exists in a checkpoint. A plan is a list
of phases `[step_begin, step_end)`, each with its own dataset, global batch and sequence length (a change of batch or
length is a new regime, §2.4):

- `Levanter`: the baseline closed form with an `index_base`.
- `Repeat`: mixture index `j mod subset` of a Levanter order for a real-valued number of passes (`DATA_EPOCHS=2.4`
  repeats without reshuffle, as levanter's `max_train_batches` restart), optionally reshuffled per pass with
  `permutation(fold_in(Key(seed), pass))`.
- `Explicit`: sequence ids from a file (resegmented or repacked passes).

A fork sets `run.parent = <run_id>@<k>`, loads the parent's checkpoint at k, checks the parent's hash against the
parent's own stored manifest config, copies the parent's phases up to k and may append phases. `seq_of(step, row)` and
`times_seen(doc, step)` answer which data a step saw; `times_seen` counts documents (through `docs.u64`), so it stays
meaningful across resegmented and repacked datasets. `run.keep_pass_boundaries=true` keeps a checkpoint at every pass
boundary (fork points). Gate: `seq_of` equals, bitwise, the order the JAX runs read for DATA_EPOCHS 2.4, PASS_DATA rs
and repack.

## 6. Communication

### 6.1 DDP compute, ZeRO-1 state

- Compute is data parallel: every rank runs the whole model on its rows. Optimizer state is owned (perf review 7):
  every MuonH matrix has one owner rank (whole matrix, because NS needs it); every vocab-sized tensor (embedding,
  lm_head, separate heads) is split by rows over all ranks; small tensors (gains) have one owner. Owners are assigned
  by LPT at plan time.
- Gradients accumulate locally in f32 over the microbatches. During the last microbatch's backward, each bucket (64 MiB,
  contiguous, never splitting a parameter, laid out in reverse backward order) is reduced to its owners as soon as it
  is complete: one `ncclGroupStart/End` of `ncclReduce` (root = owner) for whole-owned tensors and
  `ncclReduceScatter` for row-sharded ones. The per-token scale already divides by `B_global x (S-1)`.
- The owner updates; the bf16 copies are then all-gathered (grouped `ncclBroadcast` from owners, `ncclAllGather` for
  row shards) at the start of the next `g_step` on the comm stream, per layer bucket in forward order; layer l's
  forward waits on its bucket's event. For pls heads the readout of layer l waits on its head's all-gather.
- Wire bytes per parameter: (N-1)/N x (4 + 2) = 6(N-1)/N against 8(N-1)/N for an f32 all-reduce (25% less). Optimizer
  work and state memory divide by N (1.2B: persistent memory 21.25 -> 11.78 GiB per GPU).
- Exposure: the embedding gradient comes from the last kernel of the backward. Its reduction overlaps the MuonH group's
  NS (§2.4) instead of preceding the whole optimizer.
- `optim.zero1=off` keeps the version-1 replicated path (all-reduce, replicated optimizer) for option A and for A/B.

### 6.2 Reduction dtype

Levanter's cross-GPU reductions were most likely bf16 (numerics.md §6.2, open until T1.0). marincpp reduces in f32 by
default (`exec.grad_reduce=f32`, class P upward); `bf16` (convert, reduce, convert back) is the strict-preset value.

### 6.3 Metrics reduction

`train/loss` is the mean over the whole global batch. Before the StepRecord, `g_step` all-reduces (sum, f32, fixed
order) one small vector per rank: loss sum, weight sum, per-readout loss sums, per-readout weight sums, non-finite
counts (loss, gradient, doc-meta overflow). A NaN loss on any rank therefore sets the flag in rank 0's ring. Test: the
step-0 loss on 1 GPU equals the loss on 4 GPUs to 1e-6 relative.

## 7. Checkpoint and resume

### 7.1 Format

`<run>/ckpt/step-<k:08d>/manifest.json` + `rank<r>.bin` (checkpoint.hpp). The manifest holds format version, run id,
config hash, the canonical config itself, code sha, `step_next = k`, data plan sha, and per tensor: name, dtype, shape,
rank file, offset, bytes, xxh3-64. Each rank writes what it owns (§6.1). The files hold f32 masters, MuonH momentum,
Adam/AdamH mu and nu (and EMA weights if `eval.ema_beta > 0`). bf16 copies, gradients and loader state are not saved.
The loader reads tensors by name, so a checkpoint loads on any rank count.

### 7.2 Sizes

| size | checkpoint bytes |
|---|---|
| 130m | 1.76 GB (masters 0.62 + momentum 0.09 + Adam/AdamH moments 1.05) |
| 300m | 3.19 GB |
| 1.2B | 13.6 GB |
| pls d48 | 40.1 GB |

### 7.3 Async save

At a periodic checkpoint after update k-1, each worker's copy stream waits on the end of `g_step(k-1)` and copies the
rank's owner slices D2H into pinned staging; the worker continues with `g_step(k)`, whose optimizer section waits on
`ckpt_copy_done`. Only the optimizer can stall, and only if the copy is slower than one forward/backward. Writer
threads write, fsync, then rank 0 commits the manifest. Interval `run.ckpt_every_min` = 30 min.

### 7.4 Atomic commit and retention

Write into `step-<k>.tmp/`; fsync every rank file; host barrier; rank 0 writes and fsyncs `manifest.json` and the
directory; `rename(step-<k>.tmp, step-<k>)`; fsync `ckpt/`; replace the `latest` symlink atomically. On start: delete
every `*.tmp`; resume from `latest`. Keep the newest `run.keep_ckpts` = 2, every multiple of `run.keep_every`, and the
pass boundaries when `run.keep_pass_boundaries`.

### 7.5 Exact resume and exit codes

Resume restores masters, optimizer state and the device step counter; bf16 copies are recast; the data cursor is k by
construction; the GEMM algorithm file is reused. In deterministic mode a run that stops at k and resumes equals an
uninterrupted run bitwise (test in S10). A resume whose config hash differs exits 1.

| exit | meaning | wq |
|---|---|---|
| 0 | all N updates and the final eval are durable; `DONE` written | finished |
| 1 | config error, hash mismatch, missing data | failed, not retried |
| 2 | non-finite loss or gradient norm detected | failed, not retried |
| 99 | voluntary stop (health probe below half the expected rate, or a configured deadline) after a checkpoint | requeued |
| 143 | TERM/INT received; state saved or last checkpoint judged sufficient | requeued |
| 124 | set by `timeout`, not by the trainer | requeued |

### 7.6 Loading into a different model, and importing levanter

- `init.mode=from` with `init.from=<run_id>@<k>` (or a manifest path), `init.map` (ordered regex rules: target tensor
  -> source tensor, or `init` for a fresh draw), `init.freeze` (regex of tensors in group Frozen) and `init.opt_state`
  (`fresh` or `copy` for mapped tensors). The source's hash is checked against the source's manifest. This serves
  PROBE_FROM, HEADS_FROM, LENS_FROM, INIT_FROM, vocab refits and transplants. A "heads only" model (`model.trunk=frozen`
  with a trunk loaded by `init.from`) drops trunk activations and trunk backward from the plan where the heads do not
  need them.
- `tools/levref/import_levanter.py` converts a levanter checkpoint (model, opt_state, step counts) to a marincpp
  manifest + `rank0.bin`. The row map is explicit: per layer, `Wqkv` rows = q rows in (kv, g, d) order from `q_proj
  [L, KV, G, hd, D]`, then k rows, then v rows; `Wgu` = gate and up rows interleaved in 64-row blocks (§3.1); `o_proj
  [L, D, H, hd]` -> `[D, H·hd]`. Round-trip test: convert, load, export per-leaf `.npy` (`nm_model_export_params`),
  compare bitwise with the levanter leaves. T3.2 loads levanter's 130m `step-4958` through it.

## 8. Eval and metrics

### 8.1 In-process eval

- Sources (`EvalSource`): Paloma (the converted 14,529 windows in component order, reading A, eval.md §2.3), a slice of
  the training DataPlan (repeated subset, seen/unseen), or an explicit list of sequence ids. Groupings (`Grouping`):
  domain tags, target-token group tables, position bins, `times_seen` bins. Per-group sums are device partials with
  fixed-order reductions.
- Batching: the estimator's batch stays exactly 128 sequences (`eval.batch`), the last Paloma batch 65 real + 63 zero
  rows. Execution packs several 128-row batches into one eval microbatch sized to memory (no saved activations: at
  least 4x the training microbatch) and one graph per batch group; per-batch sums are kept per 128-row batch.
- Per-position loss: the exact f32 CE of every scored position, optionally dumped (`eval.per_position`).
- Levanter's estimator (eval.md §4.1): per batch and tag, L and L_k are accumulated in f64 and rounded to f32 once; W, Y
  are exact integers. After the cross-rank sum, one host thread applies the f32 `RunningMean` updates in batch order
  with the contraction XLA used for `mean + (x - mean)·r` (explicit `std::fmaf` or none, from the eval PTX; host code
  is `-ffp-contract=off`). The np.mean / np.average keys use numpy's exact float32 algorithms (pairwise summation as
  numpy implements it), tested against numpy on the fixture. The 42 keys keep their names and formulas, so
  `scripts/della/objective_compare.py` reads marincpp runs unchanged.
- T3.1 is split into two tests: (i) given levanter's per-batch (L, W, Y, L_k, W_k, Y_k) (dumped by
  `tools/levref/paloma.py`), the RunningMean/key stage matches to 0-1 ulp; (ii) the sum stage passes the oracle rule
  `|nano - exact| <= |lev - exact| + 1 ulp`.
- Exact bpb: f64 sums of per-position loss and exact UTF-8 bytes give `eval_exact/paloma/<d>/bpb` and
  `eval_exact/macro_bpb`. Each domain keeps `floor(tokens/4096)` windows, so different tokenizers score different text;
  the converter records the exact source bytes covered per domain and tokenizer, and cross-tokenizer comparisons use
  `eval_exact_shared/...`, the bpb over the byte range every compared tokenizer covers. Gate: agreement with
  `bpb_exact_tables.py` on one checkpoint to rel 1e-6.
- Schedule (levanter's off-by-one, eval.md §1): after update s with s % 1000 == 0 and s > 0, logged at step s; the final
  eval after all N updates, logged at N-1, once. The init eval (gate a3) is logged under `eval_init/...` (and in
  `events.jsonl`), never under `eval/` at a step, so the `eval/` steps are exactly {1000k, N-1}. pls: every readout in
  the same forward (`eval/L{k}/...`).
- Cost per eval (forward share of the budget, §12): 130m 5.3 s, 300m 11.9 s, 520m 26.8 s, 1.2B 53.1 s, pls d48 with 48
  readouts 174 s (about 1,045 s per run, 3% of the run).

### 8.2 Train metrics and the logging contract

- Device ring (`metrics.hpp`): `g_step` writes one StepRecord per update after the metrics all-reduce of §6.3: loss sum
  and weight sum (global), lr, adam_lr, per-group pre-clip norms and clip flags, the per-step total gradient norm,
  per-readout losses, non-finite flags. The record layout is sized at plan time from the number of groups and readouts.
- Statistics step (regime, `k % 10 == 0 && k > 1`): `params/norm/<leaf>` of the f32 parameters before update k and
  `grad/norm/<leaf>` of the pre-clip mean gradient of batch k, per logical leaf: q, k, v, gate and up are stored fused
  but each gets its own norm from slice sums that end on leaf boundaries; plus `transformer.layers.stacked.<X>` and
  `total`: 13(L+1)+4 keys per prefix (95, 173, 329, 225 for 130m, 300m, 520m, 1.2B). Owners compute partials; one
  all-reduce of the per-leaf sums. `grad/norm/total` is logged only at this cadence; the per-step value is
  `grad/norm/total_every_step`.
- Host copies every `run.metrics_flush` = 32 steps by the copy stream; the writer thread waits on the event. Before an
  eval row at step s is written, the writer forces a flush of every train row <= s (W&B drops rows below the last
  logged step). A non-finite flag ends the run with exit 2 at most 32 steps later.
- Train rows at steps 2..N-1 (`train/loss`, `global_step`, `run_progress`, `optim/*`, `throughput/*`), hook and loading
  times at 0..N-1, nothing at a step >= N. Step times come from CUDA events; MFU uses levanter's formula.
- Record: `<run>/metrics.jsonl` (append-only, fsync at checkpoints, truncated to `step < k` on resume) and
  `events.jsonl`. W&B mirror: `tools/wandb_mirror.py` tails `metrics.jsonl`, run name
  `muonh-qwen3-<size>-della4xh100-nm<tags>`, nested config `{"trainer": {"num_train_steps": N}}`, `resume="allow"`,
  `finish()` only when `DONE` exists; the real GPU count goes into the config and the tags when the name says
  `della4x`.
- 520m history: levanter ran 1 x 64 sequences per GPU for updates 0-85, then 2 x 32; W&B rows 86-119 come from the
  discarded first trajectory (spec §C1). marincpp does not claim a bit-exact 520m replay; the T4 520m script masks
  rows 86-119 (reported apart).

## 9. Config and variants

### 9.1 One config struct, registries as the source of names

`include/marincpp/knobs.def` declares every knob once. `Config` (config.hpp) is generated from it, as are the TOML
parser, `--set section.name=value`, the canonical JSON, the run-id hash, the label and `marincpp knobs`. Knobs that
name an extension (`model.block`, `model.input`, `model.head`, `model.losses`) are checked against the
registries (`registry_has(kind, name)`), so a block name lives in one place: its `register_block` line.
`readout.kind` (none, shared, separate) is a closed set of trunk wirings with no factory, so it is a fixed list;
readouts that extend the model come through the readout table and the head registry (S13). A value without
a registered implementation (e.g. `moda` today) is rejected. Cross-knob rules live in `validate()`. No knob is read
from the environment (reliability R3).

### 9.2 Run identity

`run_id` = first 12 hex digits of sha256(canonical JSON of the hashed knobs whose value differs from the default, plus
`kKnobTableVersion`). Adding a knob whose default keeps the old behaviour leaves every existing run id unchanged;
changing any default bumps `kKnobTableVersion` (and so every id). File inputs that change numbers (init fixture,
schedule fixture, data plan, dataset and eval metadata) enter by content hash only: `resolve()` stores
`<path>#sha256=<hex>`, and the identity takes the hash, so the same content under another path or name gives the same
run id. A fork checks the parent's hash against the config stored in the parent's manifest, so a parent made by an
older binary still verifies. The identity is that of the RESOLVED config, and resolution depends on the binary
(planner formulas, the §12.1 estimates behind the CE chunk, "auto" decisions still open): resume and fork load the
stored resolved config and never re-resolve the original file.

### 9.3 How a research variant plugs in

Interfaces (model.hpp):

| interface | what it owns | notes |
|---|---|---|
| `Block` | one layer: declare (params with group/scope/init, activations with recompute class and lifetime, transients, stats, depth use), bind by name, forward, hand-written backward, recompute, `readout_state` / `readout_state_bwd` | default readout state = the residual |
| depth state | model-level `DepthDecl{n_sources, per_token, grad_dtype}` in the Microbatch region; `LayerCtx` gives a block `sources`, `d_sources`, `n_visible` | attnres: 2L+1 = 97 sources at d48 (1.6 GB at T = 16,384) plus gradient accumulators |
| `InputLayer` | tokens -> x0 with its own backward (default: gather + sorted segment sum) | composed rows (cv/cvd/cvr), n-gram tables, byte inputs, input noise |
| `OutputHead` | final hidden -> per-token loss, dx, its own dW | default: full-vocab fused CE; sampled softmax, composed heads, logit bias, label masks (rm), focal, OD heads |
| `LossTerm` | auxiliary loss; reads the head's lse, loss_tok and label mask; adds into d(final hidden) | logged as `train/aux/<name>` |
| readout table | several readouts per layer, each in mode train, probe or monitor, with its own group | probes pass `dx = nullptr` |
| `GroupSpec`, `RowPartition` | optimizer groups and per-row control as data (§4.1, §4.5) | |
| `StepRegime` | step ranges with static flags; one graph each (§2.4) | dropout from pass p, readout phases, sampled-softmax stages |
| RNG | `LayerCtx` carries `training`, a device step pointer, device schedules and an `RngKey`; device threefry with keys folded by (k, rank, microbatch, layer, purpose) | gate: device bits equal host bits, bitwise |
| statistics | `StatSpec` kinds Rms, AbsMax, SumSq, Mean, Dot(a, b), Vector(width), PosRangeMean(loss_tok); cross-rank reduce Sum, Max or PerRank; cadence by regime | attnres source weights, pls probe cosines, ICL score, worst-rank loss |
| diagnostics | `Model::record_backward(term_mask, grad_out)` in a diagnostic regime gives per-term gradients (pcgrad, probe cosines); costs one extra backward per term | |

Fit of the known experiments after these interfaces ("files" = files a variant author edits):

| experiment | fits | files | core edits |
|---|---|---|---|
| hc / mhc, d48, separate heads | yes (readout state, `depth_adam` group) | block .cu, its test, its levref config, registry line | none |
| attnres / attnres_block | yes (depth state) | 4 | none |
| moda | after the FA3 `extra_kv` patch | 4 + attention kernel | attention |
| probes without dX | yes | 0 | none |
| per-layer stats, hooks on residual streams | yes (Dot/Vector stats, readout states) | 1-2 | none |
| vocab 8K-128K, truncated tokenizers, bytes | yes | 0 (converted data) | none |
| SEQ_LEN 4224 / 4480 / 19456 | yes (varlen attention) | 0 | none |
| per-position eval on the training subset | yes (EvalSource, Grouping) | 0 | none |
| data passes, mid-run fork | yes | 0 | none |
| per-row freeze / LR (fo, fop, fip, slow, lazy) | yes (RowPartition) | 1-2 | none |
| sampled softmax | yes (OutputHead `sampled`: candidates = sort-unique targets of the device microbatch + golden-ratio stride negatives, gather W_P, fused CE, scatter dW_P rows without atomics; plan for P_max) | 3 | none |
| composed vocab cv/cvd | yes (InputLayer + OutputHead) | 3-4 | none |
| dropout / input noise from pass p | yes (RNG + regime) | 2 | none |
| hashed n-gram tables, 12.8M-class OD head | later stage: `ShardedTable` (rows sharded over ranks; lookup by all-gather of ids, local gather, reduce-scatter of rows; sparse gradients on touched rows; lazy Adam). Replicated it does not fit (39 GB of bf16 + f32 gradient per GPU at D = 512) | 4 | comm (sharded table) |

### 9.4 Graph-safety rules for variants

A variant may not read a host value that changes with k inside the graph (capture lint), may not allocate, and must
declare every buffer. Step-dependent behaviour is a device table, a regime flag, or a conditional node.

## 10. Correctness strategy for hand-written backward

### 10.1 References

1. Baseline kernels: a PyTorch or numpy-f64 reference in `tests/ref/`, written from the spec; the levanter fixtures of
   T1 are the independent second source.
2. Variants: the existing JAX code is the reference of record (`depth_arch_qwen3.py`, `per_layer_qwen3.py`,
   `sampled_softmax_qwen3.py`, `composed_vocab_qwen3.py`, `freeze_rows_muonh.py`, `dropout_qwen3.py`, with their CPU
   tests). `tools/levref` has a variant mode: it builds the JAX variant at small shapes, runs `jax.vjp` in f32 and f64,
   and dumps parameters, activations and gradients by name. A new reference is written only when no JAX version exists.

### 10.2 Kernel gates

- Gradient check of each kernel against autograd of its reference, f64, from the same bf16 inputs. The f32-storage
  instantiation must match to rel 1e-6 (math bugs); the bf16 instantiation passes the oracle rule (rounding).
- Error report per tensor and per row: the whole-tensor rel, plus per-row max error <= 10x the tensor rel, with the 10
  worst rows printed. Shapes include V in {512, 8192, 16384, 32768, 50257, 65536, 128256}, S in {4096, 4224, 4480,
  19456}, length-1 segments, a segment ending exactly on a tile edge, a forced `max_segs` overflow (the flag must be
  set), the last partial vocab tile, tokens absent from the batch, GQA group edges.
- Oracles are chunked: CE by row chunks (f64 logits at T = 131,072, V = 128,256 would be 134 GB), attention per segment.
  Each gate states its shapes.
- Fused kernels equal their unfused reference path bitwise in `levanter` mode.

### 10.3 Whole-model gates

| gate | what | bound |
|---|---|---|
| T2.x | baseline parity (step-0 loss and gradients, f32 and bf16 policies, group norms, optimizer steps) | parity plan |
| V1 | f32 policy, central finite differences on 16 random directions, every block, head, input layer and loss term, at perturbed parameters | rel error <= 1e-3 |
| V2 | per-tensor gradients vs JAX f64 (levref variant mode) | f32 storage: rel <= 1e-5 and cosine >= 1-1e-8; bf16: rel <= 2 x rel(JAX bf16, JAX f64) |
| V3 | per-tensor gradients after the reduction for n_micro in {1, 2, 4} and ranks in {1, 2, 4}, accurate mode | rel <= 1e-6 |
| V4 | 200 steps from the same exported init and data order vs the JAX run's `train/loss` and `train/pls/L{k}` | T4's 300m bounds; then one full run inside the JAX seed pool before any claim relies on the variant |
| arena | remat=min equals remat=all bitwise; ZeRO on equals off to rel 1e-6 (accurate); canary and poison runs clean; compute-sanitizer clean | exact |
| graph | eager equals graph at k0, k0+1 and every regime boundary; capture lint passes | bitwise |

Parameters for V1/V2 are perturbed from init by N(0, 0.1·scale), and one JAX checkpoint at about step 1000 is also
used, because init points hide terms (hc dynamic weight W = 0, mhc phi = 0 with saturated biases, attnres query = 0).
Gradients must be exactly 0 where the math says 0 (mhc identity-mode residual columns) and nonzero elsewhere (norm >
1e-3 x the reference norm). Variants implement `accurate` rounding and are compared to JAX bf16 under V2; levanter
rounding has no meaning for them.

### 10.4 Where tests run

`nm_tests` and the portable kernels (elementwise, cuBLASLt GEMMs, RefF32 and LevRef attention, CE A, optimizer, data,
eval) also build for sm_80 and sm_86, so V1-V3, the sanitizer runs and most kernel gates run on della-vis1 (A100) and
della-vis2 (A40) under `nice -n 19`. FA3, CE B1/B1.5/B2, the custom WGMMA kernels and every bitwise parity gate need
H100 jobs (smoke-test QoS quota).

### 10.5 C ABI

`capi.h` exposes each kernel, `nm_model_*` (create with any config including the f32 policy, load and export params,
loss and grads with per-readout losses, optimizer step, eval), and the variant entry points `nm_block_forward/backward`
(by registered name, config, layer and named buffers), `nm_model_dump_activations`, `nm_model_set_regime` and
`nm_model_set_rng`. pytest calls it with ctypes on torch device pointers; torch exists only in the test environment.

## 11. Determinism

`exec.deterministic=true` (default) makes a step bitwise repeatable for a fixed GPU count and microbatch split:

| source | fix |
|---|---|
| embedding backward atomics | sorted segment sum (§3.2) |
| norm-gain, loss, statistic reductions | per-CTA partials + fixed-order pass; no float atomics anywhere |
| FA3 backward dq and GQA dk/dv accumulation | FA3 deterministic mode (ordered accumulation; first head stores) |
| GEMMs | one stored algorithm per shape, deterministic split-K schemes and stream-K fixup only |
| NCCL | Ring/Simple pinned, fixed buckets and owners |
| graph vs eager | warm-up check of §2.4 |

Cost: the FA3 deterministic backward is the only known cost (serialized accumulation). Rule (O5): measure on/off at 130m
and 520m for 50 steps; if the cost exceeds 3% of the step, the default flips to off and a determinism smoke stays in
CI. Benefit: n = 1 A/B tests of single class-P items, bitwise resume.

## 12. Performance budget

### 12.1 Assumptions

H100 SXM: 989.5 TFLOP/s bf16 dense, 494.7 TF32, ~67 FP32 FFMA, 3.35 TB/s HBM, ~350 GB/s NCCL bus bandwidth. Efficiencies
(ESTIMATES until the benches land, roofline.md §0): GEMMs 0.60 (D = 512), 0.66 (768), 0.72 (1024), 0.78 (2048) of peak,
assuming the GEMM search of §3.7 removes partial waves and the CE kernels meet §3.8's requirements; FA3 0.40 fwd /
0.30 bwd at hd 64 and 0.55 / 0.42 at hd 128 with the fixed backward scheduler; memory-bound kernels 0.8 of HBM; NS TF32
at 0.5 of peak. Attention FLOPs use the measured computed/causal ratio 0.504. Elementwise bytes per layer per token,
fused stage 1: 40D + 16F + 10(H+KV)hd minus the fusions of §3.1 (-8F, -6(H+KV)hd, -2H·hd, -4D at D = 512), plus FA3's
backward side traffic 12·H·hd (accumulator clear, f32 dq accumulation, postprocess; the preprocess is folded).

### 12.2 Budget per step (ms, per GPU, 4 GPUs, reference global batch, CE B1 unless noted)

| op | 130m | 300m | 520m | 1.2B | pls d48 (opt. B) |
|---|---|---|---|---|---|
| tokens per GPU per step | 131,072 | 131,072 | 262,144 | 262,144 | 131,072 |
| trunk GEMMs (fwd + 2x bwd) | 30.2 | 123.6 | 750.0 | 1846.3 | 241.7 |
| attention fwd | 4.2 | 12.6 | 67.2 | 65.2 | 33.6 |
| attention bwd | 14.0 | 42.0 | 224.0 | 213.3 | 112.0 |
| LM head + CE GEMMs (6DV) | 87.0 | 118.6 | 289.9 | 535.3 | 4175.3 (48 heads) |
| CE extra traffic (B1 6V, B1.5 4V, B2 2V B/token) | 37.6 | 37.6 | 75.3 | 75.3 | 1806.5 |
| CE head-gradient read-modify-write per chunk | 0.8 | 1.2 | 6.3 | 12.5 | 75.3 |
| norms, RoPE, SwiGLU, residual, FA3 side traffic (fused) | 12.3 | 38.8 | 201.9 | 269.2 | 98.6 |
| embedding fwd + sorted bwd | 0.2 | 0.3 | 0.8 | 1.6 | 0.2 |
| optimizer elementwise (owner shards) | 0.6 | 1.2 | 2.4 | 5.7 | 17.4 |
| Newton-Schulz (TF32, owners) | 0.3 | 2.0 | 8.1 | 43.4 | 2.3 |
| exposed communication | 0.5 | 0.4 | 0.8 | 1.8 | 4.4 |
| **step (B1)** | **187.7** | **378.3** | **1626.8** | **3069.7** | **6567.2** |
| step (B1.5) | 175.2 | 365.7 | 1601.7 | 3044.6 | 5965.0 |
| step (B2) | 162.6 | 353.2 | 1576.6 | 3019.5 | 5362.8 |
| step, unfused elementwise (B1, version-1 schedule) | 194.9 | 398.1 | 1725.4 | 3201.1 | 6624.9 |
| tokens/s (B1) | 2,793,000 | 1,386,000 | 644,600 | 341,600 | 79,800 |
| levanter H100 tokens/s (measured) | 830,158 | 363,155 | 155,541 | 127,550 | 34,500 |
| speedup (B1 / B2) | 3.4 / 3.9 | 3.8 / 4.1 | 4.1 / 4.3 | 2.7 / 2.7 | 2.3 / 2.8 |
| levanter-formula MFU (B1) | 48% | 58% | 66% | 75% | 43% |
| eval per run (s) | 32 | 155 | 295 | 1,274 | 1,045 |
| levanter_strict step (B1) | — | — | 1950.3 | 3762.8 | — |

Notes:
- Levanter's MFU formula counts full non-causal attention, so these MFU numbers overstate hardware use; they are
  reported only because the gate and the W&B key use that formula.
- The stage-1 gate is tokens/s >= levanter at every size; the goal is 1.5x. The 130m budget is 2.2x above the goal.
- The risks the reviews priced, if left unfixed: FA3's padded backward grid (130m +4-13 ms, 520m +68-205 ms) and
  partial-wave weight gradients (130m +8-24 ms, 520m +55-120 ms): together +6-19% at 130m and +7-20% at 520m. Both
  fixes are stage 1 (S4).
- pls d48 is CE-bound: B1's memory term is 28% of its step, so pls needs B1.5 or B2 in stage 1 (O2 decides).
- Run wall clock, 130m: 4,959 x 0.188 s = 931 s of steps + 32 s of evals + ~30 s start = ~17 min (levanter 57 min).

### 12.3 Profiling method

1. `marincpp profile --profile-ops` runs 20 steps eagerly with CUDA events around every op group of §3.1 and prints
   measured ms next to the budget rows. Rule: each row within 1.25x of its budget, or the budget row is updated with the
   measured efficiency and the reason is written down.
2. nsys over steps 20-25 of the graphed run: `nsys profile --cuda-graph-trace=node --trace=cuda,nvtx,osrt
   --capture-range=cudaProfilerApi`. It gives the kernel timeline inside the graph, GPU idle between kernels (target
   < 2% of the step), the comm/compute overlap, and the count of empty CTAs per attention launch.
3. ncu on the top kernels: `ncu --set full -k regex:<name> --launch-skip 50 --launch-count 3`, reading achieved TFLOP/s,
   DRAM GB/s, SM occupancy per wave (weight gradients) and MUFU utilization (CE).
4. Every stage-1 run logs p50 tokens/s over steps >= 20 excluding eval steps (targets.md §4(d)) and the arena size.
5. Programmatic dependent launch on marincpp's own kernels (saves 1-2 µs gaps; at most 0.3% at 130m, ~15 ms at pls)
   is applied after the nsys idle measurement, if idle exceeds the 2% target.

## 13. Build, layout, staging

### 13.1 Dependencies on Della (checked on della-vis1, 2026-10-24)

| dependency | version | how it is found |
|---|---|---|
| CUDA toolkit | 12.8 (`module load cudatoolkit/12.8`, nvcc V12.8.93, cuBLAS/cuBLASLt 12.8.4); 12.9, 13.0-13.3 also installed | `find_package(CUDAToolkit)` |
| driver | 610.57 (vis node; supports CUDA 13.x) | — |
| host compiler | gcc 14.2 (`module load gcc-toolset/14`); gcc 11.5 default | C++20, `-ffp-contract=off` |
| CMake | 3.31.8 system, 3.30.8 module | `cmake_minimum_required(3.30)` |
| NCCL | no module; wheel `nvidia-nccl-cu12==2.28.9` unpacked to `$ROOT/nm_deps/nccl` | `-DNM_NCCL_ROOT` |
| cuDNN (CudnnSdpa only) | no module; wheel `nvidia-cudnn-cu12==9.19.0.56` to `$ROOT/nm_deps/cudnn`; cudnn-frontend 1.32 submodule | `-DNM_WITH_CUDNN=ON` |
| CUTLASS / CuTe | `7127592` (v4.2.1, FA3's pin), submodule | include path |
| FlashAttention-3 | `hopper/` at `767386d`, vendored subset with the backward scheduler and doc-causal patches as separate commits | compiled into `libmarincpp_kernels.so` |
| toml++ 3.4, nlohmann/json 3.11, xxHash 0.8 | single headers in `third_party/` | include path |

FA3 compile check (della-vis1, nice -n 19, `hopper/setup.py` flags): the hd64 and hd128 bf16 sm90 forward and backward
units compile with CUDA 12.8 and 13.0 against CUTLASS `7127592` without torch. `--use_fast_math` stays confined to the
FA3 units; marincpp kernels compile without fast math and with the contraction policy of §3.0. CUDA conditional graph
nodes (12.4+) are available in 12.8.

### 13.2 Layout and size

```
CMakeLists.txt            targets: marincpp (sm_90a), marincpp_kernels (shared lib + C ABI, sm_90a [+ sm_80/86 portable]),
                          nm_tests (sm_90a + sm_80/86)
include/marincpp/        the headers of this design (interfaces)
src/core/                 config, knob registry, registries, run id, run dir, signals, process/threads   ~2,000 LoC
src/memory/               planner, arena, bind-by-name, canary/poison, plan printing                    ~1,100
src/kernels/              doc_meta, embedding, norms, qknorm_rope, swiglu, casts, reductions, f32 forms ~3,000
src/kernels/gemm/         GemmCache search, cuBLASLt, CUTLASS kernels and fused epilogues               ~3,000
src/kernels/attention/    FA3 entry points + bwd scheduler + doc-causal patch (~2,000), cuDNN (~700), LevRef + RefF32 (~1,100)  ~3,800
src/kernels/ce/           impl A (~400), B1 (~1,500), B1.5 (~800), B2 (~1,600), readouts (~300)         ~4,600
src/kernels/optim/        multi-tensor framework, clip, MuonH, NS (TF32/f32/3xTF32), hyperball, AdamH, Adam, row partitions  ~1,900
src/model/                Model, baseline block, interfaces, readout table, regimes, RNG, graph capture  ~3,000
src/model/blocks|inputs|heads|losses/   variants (preln, lns, deepnorm, keel, hc, mhc, attnres(+block); sampled, composed)  ~5,000
src/optim/                plan, groups, owners, schedules (fixture + formula), NS batching               ~1,100
src/comm/                 GradReducer (reduce-to-owner), all-gather, metrics reduction, ShardedTable     ~1,200
src/data/                 meta, token file, threefry, levanter order, data plans, prefetch ring          ~1,300
src/eval/                 eval sources, groupings, accumulators, keys                                     ~1,100
src/ckpt/                 manifest, writer, reader, retention, init.from mapping                          ~1,200
src/train/ + main.cpp     trainer, CLI, profile mode                                                      ~1,300
src/capi/                 C ABI                                                                           ~1,200
tests/                    pytest: references, gradient checks, levref comparisons, gates V1-V4           ~6,500 Python
tools/convert/            levanter -> flat converter                                                      ~600
tools/levref/             fixtures, schedules, precision probe, variant mode, levanter importer            ~1,500
tools/wandb_mirror.py     W&B sidecar                                                                     ~300
third_party/              cutlass, fa3-hopper subset (+patches), cudnn-frontend, toml++, json, xxhash
```

Total own code: about 37,000 lines of C++/CUDA (24,000 for stage 1 without the variant family, B2 and the sharded table)
and 8,900 of Python.

### 13.3 Staged build order (each stage ends with a passing test)

| stage | builds | passing test (parity plan id) |
|---|---|---|
| S0 | CMake (sm_90a + sm_80/86 tests), core, knob registry with registry-checked names, hash of non-default knobs + table version, config resolution, memory planner with bind-by-name, `marincpp plan`, `marincpp knobs` | planner reproduces the §2.5 tables to the byte; parameter count and flops_per_example of the 4 sizes (T0.8); optimizer constants' bit patterns (T0.6) |
| S1 | converter (Python) + TokenFile, threefry, LevanterOrder, DataPlan, prefetch ring and pinned slots | T0.1, T0.2, T0.3, T0.4, T0.5 |
| S2 | schedule fixture (tools/levref) + loader + formula cross-check | T0.7 (all four sizes, every k) |
| S3 | unfused elementwise kernels, bf16 and f32 forms, + C ABI + references; FMA policy | per-kernel gradient checks (per-row report); T1.2 (RMSNorm), T1.3 (RoPE tables bitwise) |
| S4 | GemmCache timed search with deterministic split-K/stream-K and the levanter wgrad epilogue; FA3 entry points with both fixed backward schedulers, glue as arena buffers; LevRef, RefF32; cuDNN | GEMM checks; T1.4 (all backends, 130m/520m/1.2B shapes); E1b decision (O1) |
| S5 | CE A (bf16 and f32), B1 (bf16 and f32 store), B1.5, B2; readouts with `dx = nullptr` | T1.5 at D = 512/768/1024/2048; E2 decision per D (O2) |
| S6 | T1.0 read (precision facts); optimizer kernels with GroupSpec, RowPartition, owner plan; NS TF32/f32/3xTF32 | T1.6, T1.7 cases A, B, C; row-partition gate |
| S7 | Model (baseline block), init (fixture, then the threefry + XLA erf/erf_inv port), f32 policy eager; then DDP + ZeRO-1 reduce-to-owner, metrics reduction, single-graph capture, regimes, capture lint; canary/poison/sanitizer | T2.2, T2.4 (f32 policy); T2.1, T2.5, T2.6, T2.7 at 130m and 300m on 1 and 4 GPUs; T1.1 (init bitwise); 1-vs-4-GPU loss; V3; eager == graph at k0, k0+1, boundaries |
| S8 | stage-1 fusions: QK-norm/RoPE in the QKV epilogue, dq postprocess fusion, preprocess in the o_proj dgrad epilogue, SwiGLU epilogues, D = 512 sandwich-tail epilogues | each fused output equals the unfused path bitwise (levanter mode); `--profile-ops` rows within 1.25x |
| S9 | eval (sources, groupings, f64 per-batch sums, key formulas) and metrics (logging contract), JSONL, W&B sidecar | T3.1 (i) and (ii), T3.3; init eval under `eval_init/` |
| S10 | checkpoint, resume, SIGTERM, wq exit codes, health probe, `init.from`, levanter importer | stop-and-resume equals uninterrupted bitwise (130m, 50 steps); TERM test inside 60 s; importer round trip; T3.2; a 25-step wq `--partial` smoke on the smoke-test QoS |
| S11 | first runs | T4 at 300m with the fast config and the strict preset, then 130m (200 steps); T5.1, T5.2 (seeds 0-3, fast config), T5.3 (throughput); profile against §12.2 |
| S12 | larger sizes and pls | T6 (300m x4 seeds, 520m with rows 86-119 masked, 1.2B); pls d48 options A/B timing; readout tests (1 vs N GPUs, microbatch invariance); pls eval |
| S13 | variant framework: InputLayer, OutputHead, depth state, readout table, RNG, Dot/Vector stats, levref variant mode; then variants in owner priority, each through V1-V4 | V1-V4 per variant |

Stage 1 is S0-S12. Mapping to the parity gates: T0 in S0-S2, T1 in S3-S7, T2 in S7, T3 in S9-S10, T4-T5 in S11, T6
in S12.

## Appendix A. Review log

Perf review (perf-N), parity review (par-N), research review (res-N). "Applied" means the design text and headers now
carry the fix.

| point | verdict | what changed / why |
|---|---|---|
| perf-1 FA3 backward empty CTAs | applied, except the interim fix (b) | Checked: `SingleTileBwdLPTScheduler` grid is `num_blocks x h x b` and non-persistent (`tile_scheduler.hpp:403`), `dk/dv_accum` are zeroed per call (`flash_api.cpp:1496`). §3.4: two fixed schedulers (`persistent`, `doc_causal`) built in S4; GQA first-store removes the memsets; E1b arms (padded vs exact, deterministic on/off); O1 rule. Rejected (b), per-step `cudaGraphExecKernelNodeSetParams` from host-side `cu_seqlens`: the fixed schedulers exist before the first step graph (S4 < S7), so (b) would add host work per step and a code path that never runs |
| perf-2 partial-wave wgrad | applied | §3.7: timed search per shape by default over deterministic algorithms, split-K schemes NONE/INPLACE/COMPUTE_TYPE/OUTPUT_TYPE and stream-K allowed, workspace from split counts; levanter rounding restricts to schemes that reduce in f32 first |
| perf-3a B2 traffic | applied | B2 = 2V B/token (logits store write); budget and §0 corrected (pls B2 5.36 s, not 4.72 s) |
| perf-3b SFU limit | applied | §3.8 requirements: persistent warp-specialized kernels, ping-pong epilogue in B1, transform warpgroup in B2; FMA-polynomial exp offload added as an option |
| perf-3c dgrad tile | applied, generalized | intensity equals the N tile for any D, so N = 512 tiles are required for dgrad and wgrad at every D (covering D at D = 512) |
| perf-3d pls needs more than B1 | applied | `CeImpl::B1_5` added; O2 picks per D; §12 notes pls needs B1.5/B2 in stage 1 |
| perf-3e chunk size | applied | chunk rule of §3.8 (smallest chunk with RMW <= 0.5% of step, capped by fit); pls option B 16,384 (+2.1 GB, 67.25 GiB) |
| perf-4 fusions into stage 1 | applied | §3.1 fused schedule, S8 stage, bitwise-equal-to-unfused gate; FA3 side traffic added to the budget (12·H·hd after folding the preprocess) |
| perf-5 levanter wgrad pass, CE head buffer | applied | wgrad epilogue `C + f32(bf16(acc))`; head dW rounding only in strict mode with one shared per-microbatch accumulator (Microbatch region); T1.5 decides per-chunk rounding |
| perf-6 exposed embedding all-reduce | applied | single `g_step`; clip groups start on their own buckets; sum-of-squares partials on comm after each bucket; bf16 all-gather at the start of the next step (heads included) |
| perf-7 replicated optimizer, f32 NS | applied with a refinement | ZeRO-1 for every group (§6.1); vocab-sized tensors are row-sharded rather than whole-owned (one 65.7M-row table per rank would unbalance 130m); NS 3xTF32 + upper-triangle option; the contradictory 1% rule removed |
| perf-8 data lookahead | applied, except host `cu_seqlens` | 256-step pageable ring, 16-32 readers, 3 pinned slots, launch lookahead 2. Host-side `cu_seqlens` rejected: it only fed perf-1(b); `k_doc_meta` on device stays |
| perf-9 eval share, pls eval | applied | eval microbatch sized to memory, one graph per batch group; eval time per run in §12 (pls 174 s per eval) |
| perf-10 graph-capture items | applied | FA3 glue as arena buffers + memset nodes; `AttnFwdArgs::n_segs` claim removed (the `persistent` scheduler reads a device count; doc-causal needs none); stats via a regime graph; PDL as a measured follow-up |
| par-1 train/loss across ranks | applied | §6.3 metrics all-reduce inside `g_step`; 1-vs-4-GPU loss test |
| par-2 f32 compute policy | applied | `exec.policy=f32` (§3.0) with true-f32 GEMMs, RefF32 attention, CE A in f32; fixture at precision highest |
| par-3 bf16 logits while T1.0(2) open | applied | `exec.ce_logits` (auto = T1.0(2) fact; bf16 or f32 store); recompute fallback; D2/O2 now depend on T1.0 |
| par-4 levanter mode deviations | applied | per-item knobs and the `levanter_strict` preset (LevRef, bf16 reduce, bf16 sequential embedding sum, per-chunk bf16 dW, 2 x 32 with remat=layer); T4 300m runs both, T5.2 the fast one |
| par-5 T3.1 2-ulp vs order | applied with a change | f64 per-batch sums, per-batch levanter dumps, split T3.1, numpy algorithms; the RunningMean step runs on a host thread with explicit `std::fmaf` per the PTX instead of a one-thread device kernel (same bits, simpler to test against numpy) |
| par-6 FMA contraction | applied | §3.0 FMA policy: host `-ffp-contract=off`, `--fmad=false` units with explicit contractions per formula, `sum / n` |
| par-7 schedule holes | applied | schedule fixture from jitted optax (§4.4); formula only cross-checks |
| par-8 bit-exact init | applied | XLA f32 ErfInv/Erf port, `f*(b-a)+a` contraction per PTX, scale `(float)(1.0/sqrt((double)fan_in))`; `init.mode=fixture` until T1.1 passes (model.hpp comment fixed) |
| par-9 levanter checkpoint import | applied | §7.6 importer with explicit row map and round-trip test |
| par-10 logging contract | applied | cadence and per-leaf norms, stacked keys, flush before eval rows, `eval_init/`, nothing at >= N, real GPU count |
| par-11 weight decay guard | applied | `validate()` rule + `optim.wd_variant` |
| par-12 520m history | applied | §8.2: no bit-exact claim; T4 520m masks rows 86-119 |
| res-1 depth state | applied | `DepthDecl`, `Lifetime::Microbatch`, `LayerCtx::sources/d_sources/n_visible` |
| res-2 readout state | applied | `Block::readout_state` / `readout_state_bwd` |
| res-3 moda | applied | `moda` rejected until the `extra_kv` FA3 patch exists; field reserved |
| res-4 RNG, training flag, step | applied | `LayerCtx{training, step, sched, rng}`, device threefry, bitwise gate |
| res-5 one static graph | applied | `StepRegime` graphs + conditional nodes (§2.4) |
| res-6 probes compute dX | applied | `dx = nullptr` path; late start as a regime |
| res-7 input and output hard-coded | applied | `InputLayer`, `OutputHead`; `LossTerm` reads lse, loss_tok, mask |
| res-8 sampled softmax | applied | `OutputHead` `sampled` with candidate kernel, gathered W, row scatter; f64 masked-full-vocab gate |
| res-9 n-gram tables | applied as a later-stage interface | `ShardedTable` design; sort key bits now `ceil(log2 V)`; not stage 1 |
| res-10 per-row control | applied | `RowPartition` for Adam and AdamH, mask in sumsq and moments, touched-row mask |
| res-11 groups closed enum | applied | `GroupSpec` as data, metric arrays from the plan |
| res-12 load into a different model | applied | `init.from/map/freeze/opt_state`, heads-only model |
| res-13 eval only Paloma | applied | `EvalSource`, `Grouping` |
| res-14 bpb across tokenizers | applied | covered-bytes records, `eval_exact_shared/`, exact-table gate |
| res-15 DataPlan gaps | applied | real-valued passes, per-phase batch and length, doc map, pass-boundary checkpoints, order gate |
| res-16 statistics too narrow | applied | Dot, Vector, PosRangeMean; reduce Sum/Max/PerRank; cadence by regime |
| res-17 per-term gradients | applied | `Model::record_backward(term_mask)` in a diagnostic regime |
| res-18 several variants per trunk | applied | readout table; ZeRO-2 for vocab-sized gradients as a planner option |
| res-19 knob/registry duplication, hash churn | applied with a rule | names from registries; hash = non-default knobs + table version; a changed default bumps the version |
| res-20 references for variants | applied | JAX code is the reference of record; levref variant mode. Baseline kernels keep spec references because the levanter T1 fixtures already give an independent second source |
| res-21 whole-model gradient gate | applied | V2 |
| res-22 f32 storage beyond elementwise | applied | f32 policy (same as par-2); V1 |
| res-23 init-point tests | applied | perturbed parameters, step-1000 checkpoint, exact-zero and nonzero checks |
| res-24 whole-tensor rel hides bugs | applied | per-row bound and worst rows; the listed shapes and edge cases |
| res-25 arena silent corruption | applied | bind by name with checks, canary, poison, compute-sanitizer; remat and ZeRO equivalence gates |
| res-26 invariance on loss only | applied | V3 |
| res-27 eager vs graph at one step | applied | k0, k0+1, regime boundaries; capture lint |
| res-28 every test needs an H100 | applied | sm_80/86 test build for portable kernels; bitwise parity stays on H100 |
| res-29 f64 oracle infeasible; levanter mode for variants | applied | chunked oracles with stated shapes; variants use accurate + JAX bf16 parity |
| res-30 no dynamics gate for variants | applied | V4 |
| res-31 C ABI for variants | applied | `nm_block_*`, `nm_model_dump_activations`, `nm_model_set_regime`, `nm_model_set_rng`, per-readout losses, `nm_model_export_params` |

No point was rejected whole. Three parts were rejected or changed: perf-1(b) and perf-8's host `cu_seqlens` (no
consumer once the fixed schedulers exist), and par-5's device kernel for RunningMean (host thread with explicit FMA
instead).

## Open decisions (H100 measurements only)

| id | decision | measurement | rule |
|---|---|---|---|
| O1 | FA3 backward scheduler (`persistent` vs `doc_causal`), and FA3 vs cuDNN | E1b: real segment rows at 130m/520m/1.2B shapes; padded version-1 grid, `persistent`, `doc_causal`, cuDNN; deterministic on and off; fwd + bwd time | faster of the two FA3 schedulers summed over shapes; ties within 3% -> `doc_causal`; cuDNN only if > 5% faster than that and passes T1.4 |
| O2 | CE stage per D (B1, B1.5, B2) and the f32-store vs recompute variant | E2 + `bench/ce_micro` at T = 16,384 and 131,072, D = 512/768/1024/2048, V = 128,256, with the T1.0(2) store dtype | fastest passing T1.5 per D; f32 store unless recompute is faster |
| O3 | Newton-Schulz precision | T1.0 item 5 (optimized HLO of a levanter H100 step); T1.6 | TF32 if levanter ran TF32; else 3xTF32 if T1.6 holds rel <= 1e-5, else true f32 |
| O4 | CE logits store dtype; gradient reduction dtype; embedding and gain accumulator dtypes | T1.0 items 1-4 | `exec.ce_logits` = the fact; strict-preset values = the facts; fast defaults stay f32 unless the n >= 4 A/B fails |
| O5 | determinism default | 50 steps on/off at 130m and 520m | off only if > 3% of the step |
| O6 | pls d48 option A or B | 20-step timing of both | faster wins |
| O7 | each fast-default class-P deviation (FA3 statistics, f32 reduce, f32 embedding sum, f32 head dW, 8 x 8 microbatching) | T4 at 300m fast vs strict; T5.2 seeds 0-3 against the pool | keep fast if T5.2 passes; else bisect item by item with the strict preset |
| O8 | CUDA 12.8 or 13.x | T1 and step time on both | 12.8 unless 13.x is faster by > 3% |
| O9 | GEMM per shape (cuBLASLt split-K, CUTLASS stream-K, custom) | plan-time timed search on the job's GPU | fastest deterministic candidate; ties to cuBLASLt |
| O10 | checkpoint on SIGTERM | E11 write bandwidth with 4 writers from pinned staging | save if `elapsed + bytes/bandwidth < 50 s` |
| O11 | programmatic dependent launch | nsys GPU idle between kernels | apply if idle > 2% of the step |
| O12 | pls eval microbatch and graph grouping | eval timing at d48 with 48 readouts | largest eval microbatch that fits; group size that minimizes eval time |
