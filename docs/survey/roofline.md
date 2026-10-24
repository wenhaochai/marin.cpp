# Roofline ceiling for the owner's workloads on H100 SXM

Survey dossier, 2026-10-24. Script: `marincpp/scripts/roofline.py`, run on della-vis1 with no JAX. Its full output is in
`marincpp/scripts/roofline.json` and is reproduced below. No job was submitted and no W&B data was fetched; every
measured number comes from a log or doc cited inline. Paths without a prefix are relative to `project/marin`.

**ESTIMATE convention.** Hardware peaks are spec values: 989.5 TFLOP/s dense bf16 (the levanter MFU denominator,
`lib/fray/src/fray/device_flops.py:20-28`), 1979 FP8, 494.7 TF32, 3.35 TB/s HBM, and 350 GB/s NCCL bus bandwidth
(ESTIMATE; NVLink spec is 450 GB/s per direction). Every efficiency factor is an ESTIMATE:

| factor | value |
|---|---|
| GEMM, min(k, n) ≥ 2048 | 0.78 of peak |
| GEMM, min(k, n) ≥ 1024 | 0.72 of peak |
| GEMM, min(k, n) = 768 | 0.66 of peak |
| GEMM, min(k, n) = 512 | 0.60 of peak |
| FA3-class causal attention, hd 64 | 0.45 fwd, 0.35 bwd |
| FA3-class causal attention, hd 128 | 0.60 fwd, 0.45 bwd |
| fused elementwise kernels | 0.80 of HBM bandwidth |
| FP8 GEMMs | 0.60 of 1979 TFLOP/s |
| Newton-Schulz on TF32 | 0.50 of peak |
| exposed communication | 10% of the DDP all-reduce |

Anchors for these factors:
- torchtitan reports 33–42% MFU for Llama 3.1 8B in bf16 on 8–128 H100s (arXiv 2410.06511, §experiments; 6,258 tok/s/GPU with FSDP on 8 GPUs, Table 1).
- FlashAttention-3 peaks at about 740 TFLOP/s (75%) at hd 128 (arXiv 2407.08608). hd 64 with causal masking is lower. I could not read the per-shape figure values, so 0.45/0.35 is an ESTIMATE.

## 0. Workloads (exact configs)

Sources: `experiments/references/della_muonh_qwen3_scaling.py:427-435`, `:648-662`, `:776-816`; levanter `models/llama.py:60-81`; `models/qwen.py:299`.

| size | hidden | inter | layers | heads | kv | head_dim | batch (seq) | steps | microbatch/GPU (4 GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 130m | 512 | 1792 | 6 | 8 | 8 | 64 | 128 | 4959 | 32 (1 mb) |
| 300m | 768 | 2688 | 12 | 12 | 12 | 64 | 128 | 11444 | 32 (1 mb) |
| 520m | 1024 | 3584 | 24 | 16 | 8 | 64 | 256 | 9918 | 32 (2 mb) |
| 1_2b | 2048 | 7168 | 16 | 16 | 8 | 128 | 256 | 22888 | 32 (2 mb) |
| pls d48 | 512 | 1792 | 48 | 8 | 8 | 64 | 128 | 4959 | 4 (8 mb) |

Settings shared by all configs:
- Vocabulary 128256, seq 4096, untied embeddings (TIE=0 by default), hybrid_norm (4 RMSNorms per layer), and qk-norm.
- Precision policy `p=f32,c=bfloat16`. Attention backend JAX_FLASH with 1024-token blocks.
- Full per-layer remat (`gradient_checkpointing=True`, which saves carries only; haliax `_src/scan.py:94-163`).
- FSDP: `param_mapping {"embed": "data"}` (levanter `utils/mesh.py:40`).
- MuonH: Newton-Schulz in f32, 5 steps, "simple" coefficients (`lib/levanter/src/levanter/optim/util.py:244-297`). AdamH for lm_head, Adam for the embedding.

pls d48 settings:
- `LAYERS=48`, `PDP=4`, and `PARAM_MAP=vocab`, so layers are replicated and the vocab tensors are sharded (`marin-pls/.../della_muonh_qwen3_scaling.py:155,166-187`).
- With `PLS_SEP=1`, every layer gets its own RMSNorm and a 128K-vocab head (`per_layer_qwen3.py:451-476`).

## 1. FLOPs per token (forward; training = 3×)

| size | layer GEMM MFLOP | attn causal | attn as levanter counts it | LM head | head share of levanter count | levanter fwd GFLOP/token | non-embed params (M) | total params (M) |
|---|---|---|---|---|---|---|---|---|
| 130m | 45.6 | 25.2 | 50.9 | 131.3 | 58% | 0.228 | 23 | 154 |
| 300m | 205.3 | 75.5 | 152.8 | 197.0 | 35% | 0.555 | 103 | 300 |
| 520m | 679.5 | 201.3 | 407.4 | 262.7 | 19% | 1.350 | 340 | 603 |
| 1_2b | 1,811.9 | 268.4 | 540.0 | 525.3 | 18% | 2.877 | 906 | 1432 |
| pls d48, 1 head | 364.9 | 201.3 | 407.4 | 131.3 | 15% | 0.904 | 183 | 314 |
| pls d48, 48 heads | 364.9 | 201.3 | 407.4 | 6,304 | 89% | 7.076 | 183 | 3400 |

Facts that change how the measured MFU reads:
- **Levanter counts full non-causal attention.** It counts 4·nh·hd·S plus the softmax (`lib/levanter/src/levanter/utils/flop_utils.py:43-48`), so half of the counted attention FLOPs are masked. At 130m, attention is 22% of the counted FLOPs.
- **The JAX flash kernel computes 10 of 16 blocks** of the 4×4 block grid (`lib/levanter/src/levanter/models/flash_attention.py:279`, `j_end = i+1`). That is 1.25× the causal minimum.
- **Hardware FLOPs exceed model FLOPs by 1.33–1.4×.** Full remat runs the layer forward twice. The CE backward recomputes the logits, so the head costs 8hV instead of 6hV (b-tiled path `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/batched_xla.py:552-617`).
- **The LM head is the largest single term at 130m and 300m.** At pls d48 with separate heads it is 89% of the FLOPs.

## 2. Measured throughput, checked against the formula

| size | tok/s (4×H100) | s/step | implied MFU | logged MFU | true-causal model TFLOP/s/GPU | hardware TFLOP/s/GPU |
|---|---|---|---|---|---|---|
| 130m | 831.0k | 0.63 | 14.35% | 14.36% | 126 | 176 |
| 300m | 362.8k | 1.45 | 15.26% | 15.29% | 130 | 184 |
| 520m | 155.7k | 6.73 | 15.93% | 15.92% | 134 | 191 |
| 1_2b | 128.3k | 8.17 | 27.97% | 27.97% | 251 | 348 |

- Throughput sources: `logs/kaiyue_muonh_qwen3_h100x4.yaml:50` (130m), `:51` (300m), `:85` (520m) and `:97` (1.2B). The 300m MFU is from `:69`.
- The formula reproduces the logged MFU to within 0.03 points.
- 520m takes 82% of the 1.2B step time while doing 47% of its FLOPs per token. The 520m ladder is anomalously slow.

## 3. Per-token budget in the realistic ceiling model (µs per 1k tokens per GPU, remat on)

| size | layer GEMMs | attention | elementwise | LM head, fused CE | LM head, current CE | total, fused CE | total, current CE |
|---|---|---|---|---|---|---|---|
| 130m | 307 | 295 | 261 | 834 | 2,270 | 1,699 | 3,135 |
| 300m | 1,257 | 884 | 784 | 1,160 | 2,596 | 4,088 | 5,524 |
| 520m | 3,815 | 2,358 | 2,017 | 1,446 | 2,882 | 9,640 | 11,076 |
| 1_2b | 9,391 | 2,411 | 2,690 | 2,723 | 4,158 | 17,222 | 18,658 |

Elementwise bytes per layer per token, bf16, forward:
- 4 RMSNorms: 16h B.
- 2 residual adds: 12h B.
- Fused qk-norm + rotary: 4(nh+kv)·hd B.
- SwiGLU: 6I B.

The backward pass moves about 2× the forward bytes, and remat adds 1× more.

Softmax-CE bytes per token per head, excluding the GEMM operands (`roofline.py:ce_bytes`):

| CE path | bytes/token | notes |
|---|---|---|
| current b-tiled (f32 logits, scatter, convert) | 30V = 3.85 MB | fwd 12V (`batched_xla.py:317-366`), bwd 18V (`:579-617`; `tmp/pls_speed/plan.md:55-61`) |
| + A1 one-hot select (pls-speed branch) | 22V = 2.82 MB | plan.md:62 "12.6 → 4.2 GB" per 8192-row block |
| logits and dlogits in bf16 | 16V = 2.05 MB | |
| fused/chunked (logits stay on-chip) | ≈0 | GEMM-bound: 8hV FLOPs with recompute, 6hV without |

**Arithmetic intensity at small width.** H100's ridge point is 295 FLOP/B. A GEMM with contraction k = h that writes f32 output has intensity h/2, so it is bandwidth-bound for h ≤ 512. With bf16 output it is bandwidth-bound for the h×h projections at h = 512. The current 130m logits GEMM writes f32 at intensity 255, so its roofline sits below peak before any CE pass runs. Measured FP8 confirms that small-width GEMMs are not the bottleneck:

| size | FP8 / bf16 throughput |
|---|---|
| 130m | 0.99× |
| 300m | 1.01× |
| 520m | 1.01× |
| 1.2B | 1.14× |

Source: `logs/kaiyue_fp8_qwen3_h100x4.yaml:15-18`.

## 4. Ceiling per config (step compute only; evals, checkpoints, compile and queue excluded)

| config | GPUs | measured s/step | speed of light | realistic, remat on | realistic, remat off | remat off + FP8 GEMMs | realistic MFU, levanter conv. (on/off) | realistic MFU, true causal (on/off) | ceiling speedup (on / off / off+FP8) |
|---|---|---|---|---|---|---|---|---|---|
| 130m | 4 | 0.63 | 0.08 | 0.22 | 0.20 | 0.18 | 40% / 46% | 36% / 41% | **2.8× / 3.2× / 3.4×** |
| 300m | 4 | 1.45 | 0.19 | 0.54 | 0.45 | 0.40 | 41% / 49% | 35% / 42% | **2.7× / 3.2× / 3.7×** |
| 520m | 4 | 6.73 | 0.91 | 2.54 | 2.04 | 1.74 | 42% / 53% | 36% / 45% | **2.6× / 3.3× / 3.9×** |
| 1_2b | 4 | 8.17 | 2.07 | 4.59 | 3.68 | 3.03 | 50% / 62% | 45% / 56% | **1.8× / 2.2× / 2.7×** |
| 520m pls0-noev | 8 | 4.20 | 0.45 | 1.27 | 1.02 | 0.87 | 42% / 52% | 36% / 44% | 3.3× / 4.1× / 4.8× |
| 1_2b pls0-noev | 8 | 5.00 | 1.04 | 2.31 | 1.85 | 1.53 | 50% / 62% | 45% / 56% | 2.2× / 2.7× / 3.3× |

Notes:
- **Speed of light** runs the model FLOPs (causal attention, 6N) at 100% of bf16 peak, with no remat and no memory term.
- **Realistic** is the per-component roofline with the ESTIMATE efficiencies, plus the optimizer, Newton-Schulz and 10% exposed DDP communication.
- **8-GPU rows** are from `$ROOT/tmp/pls_speed/steptimes.txt`, `mtpl-520m-ntp_15591473` and `mtpl-1_2b-ntp_15591483`. They are median s/it over 664 and 1371 log lines. Their PLS_W=0 stride-16 stop-gradient readouts make them slightly slower than a plain baseline.
- **Remat off and FP8 are conditional.** Remat off needs smaller microbatches (§7). The FP8 gain beyond 1.2B is not demonstrated: levanter measured only 1.14×.
- **Levanter-convention MFU of 40–50% is about the top of what published stacks reach** for dense 1B-class models: torchtitan 8B reaches 33–42%. Small widths cannot do better, because K = 512 GEMMs and hd-64 attention cap the tensor-core fraction. The 130m/300m true-causal ceiling is therefore about 35–41%, against 13% today (126–130 TFLOP/s/GPU).

## 5. pls d48 (width 512, 48 layers, 4×H100, batch 128, 8 microbatches)

| config | measured s/step | realistic, current CE | realistic, CE + A1 | realistic, bf16 CE | realistic, fused CE | fused CE, remat off | ceiling speedup (fused) |
|---|---|---|---|---|---|---|---|
| 48 heads (47 separate + final) | 15.2 | **15.20** | 12.79 | 10.99 | **6.17** | 5.96 | **2.5×** |
| 1 head (backbone only) | 2.9 (15.2 − 12.3, inferred) | 1.21 | 1.16 | 1.12 | 1.02 | 0.81 | 2.8× |

- **Measured.** 15.2 s/step is from `marin-pls/logs/pls_depth_controls.yaml:151` and `$ROOT/tmp/pls_speed/steptimes.txt` (w89 15.2, w90 15.1). Of it, 12.3 s is the 47 extra heads (`marin-pls/docs/della/pls.md:640`).
- **Calibration.** The current-CE model predicts 15.20 s with no fitting. One head under the current CE costs 0.297 s/step per GPU: 0.19 s of HBM traffic and 0.11 s of GEMMs. 47 heads come to 14.0 s against the measured 12.3 s, so the model is within about 15%. The measured head rate is 263 TFLOP/s/GPU counting 4 GEMMs (8hV), the owner's "27% of peak", or 197 TFLOP/s at the model's 6hV.
- **Fused CE.** Each head then costs 0.109 s (GEMM-bound), so 47 heads take 5.1 s. The heads stay 83% of the step, and that is the floor for this design in bf16. Going lower needs one of these:
  - drop the logits recompute: 6hV instead of 8hV, 0.75×;
  - FP8 head GEMMs;
  - run the readouts through a sampled softmax (31% of V on average, §6) or a low-rank head;
  - share one head (1 head = 1.0 s).
- **Backbone.** The inferred 2.9 s is 2.4× the 1.21 s model. The extra time is unattributed (JAX flash at hd 64, 48 scan iterations, 8 microbatches of 16k tokens). The pls-speed workflow profile should split it.
- **Other designs.** hc/mhc run 25.4 s and attnres 34.5 s (`pls.md:640`). Both are forced to microbatch 1 by 30–53 GiB temp buffers (`pls_depth_controls.yaml:184`), and each microbatch-1 run takes the slower XLA CE fallback with f32 delta and TF32 GEMMs (`plan.md:202,278`). A rewrite with bounded-memory CE and per-design kernels would put them near the 6.2 s preln figure plus their extra mixing FLOPs, a 4–5× ceiling (ESTIMATE).

## 6. Sampled softmax and over-vocab

**Sampled softmax.** Schedule `0.57:24576, 0.81:36864, 0.93:65536`, then full softmax (`della_muonh_qwen3_scaling.py:189`). The time-weighted candidate width is 39.7k classes, 31% of V.

| size | fused full softmax s/step | fused ss s/step | ss gain ceiling | measured ss gain (`docs/della/sampled-softmax.md:3-4`) |
|---|---|---|---|---|
| 130m | 0.22 | 0.15 | 1.51× | 1.30× |
| 300m | 0.54 | 0.43 | 1.24× | 1.13× |
| 520m | 2.54 | 2.28 | 1.11× | 1.07× |
| 1_2b | 4.59 | 4.09 | 1.12× | 1.08× |

With a fused CE, the head share falls, so sampled softmax helps less in absolute time. It stays a 1.1–1.5× lever on top of the rewrite.

**Over-vocab hashed tables.** OE has 2k tables of 12.8M rows; k = 1 below 1.2B and k = 3 at 1.2B (`experiments/references/over_vocab_qwen3.py:99-112`).

| size | tables × dim | f32 p, g, m, v total | per GPU (4 / 8) | dense Adam pass per step (8 GPU) | OD second head + d×d, fused (8 GPU) |
|---|---|---|---|---|---|
| 130m | 2 × 170 | 65 GiB | 16 / 8 GiB | 6 ms | 0.05 s |
| 300m | 2 × 256 | 98 GiB | 24 / 12 GiB | 9 ms | 0.08 s |
| 520m | 2 × 341 | 130 GiB | 33 / 16 GiB | 11 ms | 0.19 s |
| 1_2b | 6 × 227 | 260 GiB | 65 / 32 GiB | 23 ms | 0.36 s |

- **The tables cost memory and checkpoint I/O, not FLOPs.** The checkpoint is 79–227 GB (launcher `:791`). The measured 30 GiB contiguous temp at 1.2B is the dense table gradient plus its microbatch accumulator (`docs/della/over-vocab.md:294-310`).
- **Sparse handling would remove that temp.** Row-sparse gradients and a lazy or row-sparse Adam would remove the temp and the allocator fragility.
- **The bandwidth floor is small.** It is the dense Adam pass, 6–23 ms per step.

## 7. Memory per GPU (GiB)

| config | tokens/microbatch/GPU | activations, remat on | no remat, save all | no remat, selective | state, FSDP over 4 | state, replicated | current CE temp (8192×V f32 block, ×3 live) |
|---|---|---|---|---|---|---|---|
| 130m | 131,072 | 3.6 | 16.9 | 10.5 | 0.55 | 2.21 | 11.7 |
| 300m | 131,072 | 6.5 | 50.7 | 31.6 | 1.02 | 4.08 | 11.7 |
| 520m | 131,072 | 11.3 | 126.2 | 78.2 | 1.93 | 7.71 | 11.7 |
| 1_2b | 131,072 | 18.5 | 168.1 | 104.1 | 4.49 | 17.96 | 11.7 |
| pls d48, 48 heads | 16,384 | 1.1 | 16.9 | 10.5 | 12.5 | 50.0 | 11.7 |

How the state is counted:
- Muon matrices: f32 param, grad and momentum, 12 B each.
- Embedding, heads and norms: f32 param, grad, m and v, 16 B each.

What fits:
- **No-remat fits.** 130m and 300m fit without remat at today's 32 sequences per microbatch. 520m and 1.2B fit at 8 sequences per microbatch, or at 16 with selective saving. The global batch does not change.
- **Replication is cheap.** Replicated state is at most 18 GiB at 1.2B, so plain DDP fits every dense config.
- **pls d48 must shard its heads.** Its 48 heads need 50 GiB if replicated, which is what PARAM_MAP=vocab already avoids by sharding the vocab tensors.
- **The CE temp is the largest single buffer** at the small sizes. A fused CE removes it.

## 8. Communication and optimizer per step

| config | params (M) | microbatches | DDP bf16 all-reduce GB/GPU | time at 350 GB/s | FSDP GB/GPU (4 / 8 GPU) | FSDP time (4 GPU) | Newton-Schulz TFLOP/step | NS s/step | realistic compute s/step |
|---|---|---|---|---|---|---|---|---|---|
| 130m | 154 | 1 | 0.46 | 0.001 | 0.9 / 1.1 | 0.00 | 0.27 | 0.000 | 0.22 |
| 300m | 300 | 1 | 0.90 | 0.003 | 1.8 / 2.1 | 0.01 | 1.85 | 0.003 | 0.54 |
| 520m | 603 | 2 | 1.81 | 0.005 | 7.2 / 4.2 | 0.02 | 8.25 | 0.012 | 2.53 |
| 1_2b | 1432 | 2 | 4.29 | 0.012 | 17.2 / 10.0 | 0.05 | 44.0 | 0.065 | 4.51 |
| pls d48, 48 heads | 3400 | 8 | 10.2 | 0.029 | 163 / 95 | 0.47 | 2.19 | 0.003 | 6.15 |

How the volumes are counted:
- **DDP:** one ring all-reduce of bf16 gradients per step, after local accumulation.
- **FSDP (ESTIMATE):** per microbatch, two bf16 all-gathers (forward and the remat recompute) plus one f32 reduce-scatter. It assumes XLA does not hoist the gathers out of the microbatch loop.
- **Newton-Schulz FLOPs:** per matrix with m ≤ n, 5 × (4m²n + 2m³). The X Xᵀ and B·X parts are sharded over data; A·A runs replicated on every GPU (`lib/levanter/src/levanter/optim/util.py:281-293`).

Conclusions:
- **Communication is not a ceiling term** on one NVLink node: at most 0.6% of step compute with DDP.
- **FSDP has no speed benefit at these sizes.** With many microbatches it costs up to about 8% (pls d48), on top of the numerics issue the pls session hit (`pls_depth_controls.yaml:100-110`).
- **Muon Newton-Schulz is at most 1.4% of a step** (1.2B), even in f32.

## 9. Sequence lengths off the 1024 grid (tokenizer session)

- The 300m model runs at SEQ_LEN 4224 or 4480, which forces flash block 128. It takes 2.5–2.7 s/step against 1.2–1.4 at 4096 (`needs_from_sessions.md` §Tokenizer 2).
- The FLOPs per token rise only 1–2.5%: 0.555 → 0.560 → 0.569 GFLOP/token (script §10).
- The ≥2× loss is therefore all kernel overhead. A FlashAttention kernel that takes any length (or pads to 128) recovers it.
- Ceiling for those runs: ≥2× from attention alone, about 5–6× overall against the 2.7× of §4.

## 10. What bounds a rewrite (summary)

1. **The step-compute ceiling is about 2.6–2.8× over today at 130m–520m and about 1.8× at 1.2B (4×H100).**
   - Dropping full remat where memory allows takes these to 3.2–3.3× and 2.2×.
   - FP8 adds about 1.2× more, but only at 1.2B (h = 2048).
   - True-causal MFU tops out near 36–41% (130m–520m) and 45–56% (1.2B).
2. **The LM head is the largest single lever.** Today's b-tiled CE moves 30V bytes per token (3.85 MB) in f32 logits traffic. A fused CE cuts the head from about 2.3 to 0.8 µs per token at 130m, which is about half the 130m ceiling gain. For pls d48 it is the whole story: 15.2 → 6.2 s/step.
3. **The 120–130 TFLOP/s/GPU gap at 130m–520m is not yet attributed past the CE.** The likely items are JAX blockwise flash at hd 64 (1.25× extra blocks, while-loop of small matmuls), unfused norms and rotary, and the 520m anomaly (§2). A profiler trace (pls-speed or wf_cb88f9ae) should split it before the design commits.
4. **These terms are noise:** communication, Newton-Schulz, the optimizer state pass and OV table Adam. Memory headroom is large once remat is selective and the CE is fused. The OV tables need row-sparse gradients, not more GPUs.
5. **Outside the step:** compile 70 s, per-segment smoke about 20 min, checkpoints 3.4%, per-layer eval 3.3% (`needs_from_sessions.md` §pls 2). These need their own fixes, because the per-step ceiling does not touch them.
