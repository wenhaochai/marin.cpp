# Speed findings handed over by the pretrain sessions

Owner 2026-10-24: pls and Tokenizer Size keep running the old code; they hand their speed findings to nanomarin, which turns them into the design and verifies what needs verifying.

## pls (received 2026-10-24)

Full material (read-only): $ROOT/project/marin-pls-speed/docs/della/pls_speed/ — HANDOFF.md (summary), plan.md (ranked plan, A/B protocols), results.json (every candidate, refuter verdicts, measured facts with log paths). Branch pls-speed (remote archive). All GPU A/Bs were cancelled before they started: the GPU numbers below come from production logs, not controlled A/Bs. CPU numerics tests ran on della-vis1.

Where the time goes (4×H100, width 512, 48 layers, vocab 128256, batch 128×4096):
- Pre-LN d48: 15.2 s/step; baseline without readouts 2.9 s. The 47 extra readouts take 12.3 s (81%), bound by HBM traffic of f32 logits (roofline predicts 15 s).
- hc/mhc d48 forced to microbatch 1: 24.3–25.4 s/step; microbatch 4 gives 16.3 s but OOMs at step 257 from BFC fragmentation. AttnRes 34.5 s, 32.5 s at microbatch 4.
- Wall clock of a finished run (hc d48, 37 h): train steps 90.1%, checkpoint 3.1% (20 s blocking per save, bound by host staging), eval 3.9%, smoke 1.6%, jaxpr logging + startup ~1%. Host overhead per step 0.2–0.6%.

Candidates (estimated saving, numerics class, location, test state):
- A1: label term in the b-tiled CE backward as select instead of scatter. ~10% on PDP≥2 runs. Bitwise identical on CPU (B 6000, V 128256, with/without soft cap). lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/batched_xla.py `_backward_b_tiled_from_lse`, knob LEVANTER_CE_BTILED_ONEHOT. GPU fusion/speed not measured.
- A2: probes drop the dX matmul (custom_vjp returns only dW and the norm-scale grad). ~6% on probe runs. dg rounding changes. per_layer_qwen3.py:454-476. Not implemented.
- A3: save a checkpoint before the known deadline, interval to 30 min. ~1.5–2%, numerics unchanged. checkpoint.py:578 on_step, :653 request_checkpoint. Not implemented.
- A4: turn off log_jaxprs/log_xla_hlo. ~4 min per segment, program unchanged. Implemented (LOG_JAXPR); no script reads these files.
- A5: use readout L-1 as the main eval, run eval once. ~0.9%. train_lm.py:362, per_layer_qwen3.py:775. Not implemented.
- A6: JAX_FLASH attention's data-dependent while loop -> static loop. ~3%, speculative, needs a profile. flash_attention.py:278-282, 416-419.
- A7: PDP=1 designs eval with 4 sequences per GPU. ~0.6–1.3%. Implemented (EVAL_PDP), memory not measured.
- B1: hc/mhc at microbatch 4 with a non-fragmenting allocator (vmm/cuda_async). ~30%, but switches the CE path, so not comparable with finished controls; not shipped.
- B2: fast CE backward for PDP=1 designs (LEVANTER_CE_XLA_FAST_BWD). Up to ~6% on AttnRes, changes numerics.
- Rejected: vocab-parallel CE, fused Pallas CE (GPU forward 8–9x slower than XLA), latency-hiding scheduler, AttnRes buffer slicing, pinned staging.

Lessons for the rewrite:
1. Microbatch silently switches the CE kernel. PDP=1 (4096 tokens per GPU) fails batched_xla's shared-memory check and falls back to XLA streaming CE (f32 delta, TF32 matmul, gradient accumulated in bf16 across vocab blocks). PDP≥2 takes b-tiled (bf16 dlogits, f32 accumulation). On CPU vs f32: grad_x relative error 5.3e-3 vs 7.6e-4. Only a RuntimeWarning reports it. The CE path must be explicit, logged and independent of microbatch.
2. Memory is bounded by per-readout copies. Every microbatch all-gathers 47 bf16 heads (5.8 GiB) and builds full f32 head gradients (11.5 GiB) before reduce-scatter. With b=8192 and one 8192 tile, 8 readouts need 64 GiB temp; tile 4096 gives 8 GiB. That is why microbatch 2 OOMs on every design.
3. Allocator fragmentation kills runs (hc mb4, step 257).
4. Checkpoint blocks 20 s per save. No SIGTERM handler saves, so each time limit re-runs ~6 min of work on average.
5. jaxpr/HLO debug dumps are on by default: 35–213 s idle GPU per process start.
6. The compile cache works only because ~/.bashrc sets JAX_COMPILATION_CACHE_DIR; job scripts do not.
7. On resume the smoke uses the same run id; W&B resumes the first segment's run and the gate reads a stale summary (fixed, pls 684b48ead8).

What pls needs most from the stack:
- Most changed: depth architecture (residual / routing / source mixing) and per-layer readout losses (separate heads, probes, visible coordinates, per-layer weights). One new knob touches 5 places (config field + validation, model, launcher env + run-id tag, two sbatch files) plus a CPU test. Wanted: a declarative variant registry — define once, run id, validation and CLI generated.
- Slowest: 48 full-vocab readouts. Wanted: a multi-head CE whose memory does not grow linearly with the number of readouts (head gradients accumulated in place in their shard, no full f32 copy) and that takes one numeric path for every microbatch size.
- Hard or impossible today: per-layer, per-tensor statistics (residual norms, source weights, grad norms) without host syncs; reading intermediate activations in hooks instead of separate probe programs; predictable memory (a temp estimate before compiling, not trial and error on microbatch and mem fraction); several variants in one program (separate heads and probes sharing one trunk forward); a resumable smoke gate consistent with the compile cache; segment-billed, interruptible training (auto checkpoint before the time limit).

## Tokenizer Size

Pending: summary at project/marin docs/della/speed-workflow.md when wf_cb88f9ae-72b ends; branches in $ROOT/tmp/speed_wt/.
