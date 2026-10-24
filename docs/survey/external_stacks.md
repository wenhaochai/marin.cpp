# External pretraining stacks and kernels: survey for the marin rewrite

Date: 2026-10-24. Scope: dense 100M–2B models on 4–8 H100 SXM (one node), seq 4096, 128K vocab, many readout heads, 48-layer narrow models, MuonH/AdamH.
Rules: every number has a source. `ESTIMATE` marks my arithmetic or judgment, with the reasoning next to it.
MFU below is against the H100 bf16 dense peak, 989 TFLOP/s, unless stated otherwise.

## 0. Baseline: where the owner stands today

| run | stack | measured | source |
|---|---|---|---|
| 300m Qwen3 (d768, 12 L, 128K vocab), 4×H100, batch 128×4096 | levanter, JAX 0.11.1, `JAX_FLASH` attention | 1.2–1.4 s/step at seq 4096; 2.5–2.7 s/step at seq 4224/4480; "MFU about 10%"; 12.6 of 71 GiB used | needs_from_sessions.md (Tokenizer Size session, item 2) |
| same, recomputed | — | 6N incl. head = 6 × 201M = 1.21e9 FLOP/token; 524,288 tok / 1.3 s / 4 GPU = 101k tok/s/GPU ⇒ 122 TFLOP/s = **12.3% MFU** (6N only, no attention) | ESTIMATE: params from `experiments/references/della_muonh_qwen3_scaling.py:428` (hidden 768, inter 2688, 12 layers) + vocab 128,256 |
| pls 130m d48 (width 512, 48 L, one 128K head per layer), 4×H100 | levanter | 15.2 s/step; 47 extra heads = 12.3 s (81%); ~263 TFLOP/s/GPU (27%); f32 logits are bandwidth-bound | needs_from_sessions.md (pls session, item 2) |
| head + CE cost at 300m | levanter `batched_xla` CE | 0.25–0.29 s/step at 128K (17–19% of step), 0.16 s at 64K, ~0.065 s at 32K | speed workflow wf_cb88f9ae-72b, journal.jsonl, agent "find:head-loss" |
| attention loop overhead | levanter `JAX_FLASH` (pure-JAX blocked flash with data-dependent while loops) | ~53–65 µs per inner iteration; 20,196 iterations/step at seq 4224 (block 128) vs 360 at 4096 (block 1024) ⇒ 42–56% of charwin step time | speed workflow, agent "find:parallelism-xla" |
| non-step overheads | levanter | Paloma eval 3.5%, host hooks 1.0% (grad-norm watch: 388 keys, one host transfer each, 43.5 ms every 10th step), collectives ~1% and 69/72 are `is_sync=true` (not overlapped) | speed workflow, agents "find:pipeline-overheads", "find:parallelism-xla" |
| remat | levanter default | `gradient_checkpointing=True` (full per-layer recompute) and `scan_layers=True` | `lib/levanter/src/levanter/models/llama.py:80-81` |
| attention backend | launcher | `attn_backend=AttentionBackend.JAX_FLASH` forced; levanter GPU default is `NVTE`, but Transformer Engine is not installed | `della_muonh_qwen3_scaling.py:661`; `lib/levanter/src/levanter/layers/attention.py:82-89`; marin issue #7013 |

Installed in the marin venv: jax 0.11.1 (cuda13 plugin), `flash_attn_4-4.0.0b16`, `jax_triton 0.3.1`, cuDNN 9.19, torch 2.11.0+cu128 (`.venv/lib/python*/site-packages`).

## 1. The owner's own PyTorch measurements (nanoswe = nanochat fork, Della H100)

Repo `project/nanoswe` (closed 2026-09-25). Stack: torch 2.9.1 (`pyproject.toml:15`), torch.compile, FA3 (varunneal/flash-attention-3 via `kernels`, `nanoswe/flash_attention.py:43-51`), FP8 tensorwise on 145/158 linears, DistMuonAdamW (`nanoswe/optim.py:299`), chunked fused linear-CE (`nanoswe/chunked_lce.py`).

| model | GPUs | tokens/step | step | tok/s/GPU | logged `bf16_mfu` | 6N-only MFU (ESTIMATE) | source |
|---|---|---|---|---|---|---|---|
| nanochat d24 (d1536, 24 L, 32K vocab, SSSL windows 8k/32k, smear, value embeds), FP8 | 1 | 131k (4×32k) | 1.85–1.88 s | 69.8–70.9k | 75.7–76.9 | 31.4% | `logs/q3lr_nanochat.log:574-575` |
| Qwen3 dense 0.78B (24×1536, 12/6 heads, MLP 4608, full attention all layers), FP8 | 1 | 131k | 2.44–2.55 s | 51.4–53.7k | 98–102 | 23.8% | `logs/q3lr_q3_emb0.01.log:577-578` |
| nanochat d24, record recipe, FP8 | 8 | 1,048,576 | 1.89–1.96 s | 67.0–69.3k | 72.6–75.1 | ~30% | `logs/slurm/nanoswe-train-14288402.out` (step 3000–3001) |
| nanochat d24, bf16 vs FP8 (300 steps, 1 GPU) | 1 | 131k | bf16 2,061 ms; FP8 fast-acc 1,883 ms; FP8 exact-acc 1,890 ms | — | — | FP8 = **−8.6% step time** | `logs/fp8acc_{bf16,fast,exact}.log`, mean over steps 100–299 (my awk) |
| compile + first step | 1 / 8 | — | 113 s (nanochat), 120 s (Qwen3), 130 s (8 GPU) | — | — | — | step-0 `dt` in the same logs |

How to read `bf16_mfu`: it uses 6N + 12·h·q·T per layer (full, not causal; window-capped) and divides FP8 work by the bf16 peak (`nanoswe/gpt.py:375-401`), so it overstates. 6N-only: nanochat 6N = 1.072e10 − attn 6.34e9 = 4.38e9 FLOP/token (`logs/q3lr_nanochat.log:60`), × 70,938 tok/s = 311 TFLOP/s. Qwen3 6N = 1.887e10 − 1.449e10 = 4.38e9 (`logs/q3lr_q3_emb0.01.log:63`), × 53,718 = 235 TFLOP/s.

Findings that matter for the rewrite:
- The Qwen3 variant was 1.36× slower per step than nanochat at equal tokens (2.54 vs 1.88 s), because it ran full 32k attention in all 24 layers, not windows (`docs/general_pretraining.md:254-256`; `docs/experiment_logs/test_qwen3_lr.yaml`). Attention at long context dominates. At seq 4096 it does not.
- Utilization gap (ESTIMATE, 6N only): nanochat d24 31% on 1 GPU against the owner's 300m 12%. Part of that gap comes from the smaller width (768 vs 1536). The rest comes from the stack: remat, the CE kernel, attention, and host syncs (section 0).
- The PyTorch stack trains on 8×H100 with micro-batch 1×32k plus 4-step accumulation. H100 80 GB could not hold the B200 record's 4×32k (`docs/REPRO.md:145`).

## 2. Stack-by-stack

### 2.1 modded-nanogpt (Keller Jordan speedrun), PyTorch
- Record #92: 0.665 min to val loss 3.28 on 8×H100, 2026-08-30 (sampled softmax, full FP8 MLP fwd+bwd, 84.6M-row hashed n-gram table, ANVIL optimizer, full CUDA-graph capture). https://github.com/KellerJordan/modded-nanogpt (README table, read 2026-10-24).
- Measured throughput of record #90 (2026-08-03, 11 layers, d768, 6 heads, 50K vocab): last stage 393,216 tok/step (`batch_size=24*2048*8`) at ≈75.3 ms/step ⇒ 5.2M tok/s per node = **~650k tok/s/GPU**. Source: log `records/track_1_short/2026-08-03_FP8MLPBackwardPackedQKV/candidate_h100/51c31db0-….txt` (step 996 at 45,882 ms, step 1315 at 69,918 ms; code at log lines 2164–2171, 2421–2424); torch 2.10.0+cu128, Triton 3.6.0. ESTIMATE for the arithmetic: (69,918 − 45,882) ms / 319 steps.
- Techniques worth taking: FP8 lm_head (#19) and FP8 MLP (#89, #90); Triton kernels for the fused softcapped CE (#60), the fused linear ReLU² MLP (#59), the symmetric matmul for Newton-Schulz (#27), and QK-norm+RoPE fusion (#90); FA3 with doc packing via `max_doc_len` (#29); distributed Muon (#6), Polar Express (#38), NorMuon (#41); batch and seq-length schedules; CUDA graphs.
- Hackability: one file (`train_gpt.py`, ~6,500 lines including kernels; it is the printed code at log lines 1–6535). Any change touches that one file, but the file is a tuned monolith with hard-coded shapes (`assert self.num_heads == 6`, log line 1592).
- For this owner: it is a technique catalogue, not a base. Its 124M/50K-vocab/seq-2048 regime is the closest public analogue to 130m–300m. Nothing in it targets 128K vocab or many heads.

### 2.2 nanochat (Karpathy), PyTorch
- Leaderboard "time to GPT-2" (CORE ≥ 0.2565) on 8×H100: d24 3.04 h (2026-01-29) → 1.65 h (2026-03-14, "autoresearch round 2"); FP8 since 2026-02-02 (d26). https://github.com/karpathy/nanochat (read 2026-10-24). The README reports no MFU.
- On Della: section 1 (nanoswe fork). ~6,400 lines of core code (`wc -l nanoswe/*.py scripts/base_train.py`); a new loss touches `gpt.py` and `base_train.py`; a new optimizer touches `optim.py`.
- Built in: FA3/FA4/SDPA fallback with varlen doc masking, sliding windows, torch.compile, FP8 tensorwise, DistMuonAdamW (ZeRO-2-style reduce-scatter, then gather), and a chunked weighted fused-linear CE written in nanoswe (`nanoswe/chunked_lce.py:1-20`: 3 cuBLAS matmuls per chunk, no recompute).
- Risk: single-node DDP-style; nanoswe found an H100-only seed instability at step 8 (`docs/REPRO.md:173-187`).

### 2.3 torchtitan (PyTorch-native), FSDP2 + compile + float8 + async TP
- v0.3.0 (2026-09-03): typed Python config, `spmd_types` default, Qwen3 / Qwen3.5 varlen, Kimi K2.7 with DistributedMuon overlap (#4060), GraphTrainer (whole-step graph, precompile once, rank-agnostic), CUDA graphs (#3559), MXFP8, NVFP4, bf16 optimizer states. Validated with torch 2.14.0 and torchao 0.18.0. https://github.com/pytorch/torchtitan/releases/tag/v0.3.0
- H100 numbers (end of 2024, Llama 3.1 8B, 8 GPUs, FSDP, selective AC): 5,762 → 6,667 (compile) → 8,532 tok/s/GPU (compile + float8), i.e. compile +16%, float8 +28% on top. Without float8 they report 33–39% MFU. `docs/benchmarks/llama3_h100_202412_torchtitan.md` in pytorch/torchtitan. They ran at a 500 W power limit, not 700 W.
- Async TP: 829 → 876 tok/s/GPU (+5.7%) at 70B on 256 GPUs (same file). It does not matter at 100M–2B, where FSDP/DDP within one node suffices.
- Hackability: a model lives in `torchtitan/models/<name>/` (model, args, parallelize, train spec); a new loss goes through the train spec; v0.3 adds an external override registry (#3396), so research code can stay outside the core. ESTIMATE: 2–4 files per new loss or architecture.
- For this owner: the most maintained PyTorch base for FSDP2+compile+float8+Muon. It is tuned for 8B+; small-model throughput is not published.

### 2.4 Megatron-Core / Transformer Engine (FP8)
- H100, Llama3 8B, 8 GPUs, FP8 current scaling, seq 8192: **14,451 tok/s/GPU, 744 model TFLOP/s/GPU** (NeMo 25.11 container); 14,079 / 725 in 25.09. `docs/performance-summary-archive.md:658,697` in NVIDIA-NeMo/Megatron-Bridge. ESTIMATE: 1.7× torchtitan's 2024 FP8 number on the same model. Recent releases (26.04–26.08) list H100 only for MoE models.
- Owner's prior attempt: megatron-pli (private `wenhaochai/megatron-pli`, branch `della-fp8-4gpu`, 11 commits, 2026-09-13). It ported MuonH + AdamH lm_head into `megatron/core/optimizer/emerging_optimizers.py` (`TensorParallelMuonHyperball`; +98 lines) and fixed a distrib-optimizer fp32/bf16 ordering bug. The project was dropped before the first H100 run (`~/docs/projects/INDEX.md:63`), so there is no throughput number.
- Hackability: low. New layers go through specs, configs, args and the TE modules (ESTIMATE 4–8 files). The NGC container is pinned. The K3 port needed a hook to keep params in fp32 under `--bf16`.
- For this owner: the highest raw FP8 throughput at ≥8B. For 130m–1.2B at d512–d2048, FP8 GEMMs are small and gain little (section 1: +8.6% at d1536). Not worth the rigidity.

### 2.5 MaxText (JAX on GPU)
- NVIDIA JAX-Toolbox: Llama2 7B, 1×DGX H100, BF16, seq 4096, BS/GPU 2, `cudnn_flash_te` + `NVTE_FUSED_ATTN=1`, remat `minimal_flash`: 0.721 s/step, 22.19 seq/s. https://docs.nvidia.com/jax-toolbox/frameworks/max-text/overview.md (container `nvcr.io/nvidia/jax:25.10-maxtext-py3`). ESTIMATE: 22.19 × 4096 / 8 = 11.4k tok/s/GPU; 6N+attn ≈ 4.67e10 FLOP/token ⇒ ~530 TFLOP/s ≈ **54% MFU**.
- Recommended XLA flags (same page): latency-hiding scheduler, pipelined all-gather/reduce-scatter/all-reduce, while-loop double buffering, combine thresholds of 1 GiB, `--xla_disable_hlo_passes=rematerialization`. Levanter on Della sets none of these (speed workflow: 69/72 collectives synchronous).
- For this owner: proof that JAX on H100 reaches ~50% MFU when attention goes through cuDNN/TE and remat is minimal. Hackability: flat `layers/*.py` + YAML; MuonH would need porting into optax form.

### 2.6 Levanter / Marin upstream since the fork (fork point marin `e9a1f53d6a`, 2026-09-11)
Searched with `mum search` on della-vis1 (2026-10-24) and `gh`.
- **Issue #6979 "GPU MFU Learning Path"** (2026-07-06 → 07-16, branch `dial_h100_dev`, 8×H100 CoreWeave): lm_head-only 0.43B → **50.3% MFU** with Liger chunked CE, one chunk per device (default fused CE 41%, earlier 35%); **full dense d2560/26L 2.70B → 56.2% MFU** with `gpu_fa4_cute` attention, FSDP, scan + full remat, Liger CE, and host sync every K steps (+~2.5 pts). Grad-only CE (skip the forward logits GEMM, loss every N steps): +31.5% throughput in a CE-bound regime. https://github.com/marin-community/marin/issues/6979 (comments of 2026-07-06, 07-13). This is JAX/levanter code: the gap to 12% at 300m is in the dense-LM path, not in JAX.
- Scan vs unrolled: compile ~26 min unrolled vs ~7 min scanned (MoE); full remat beat save policies at 2.7B on 8 GPUs, which was memory-bound. `experiments/grug/moe/throughput_tuning_h100.md` @ `27c763ac24`, "Tuning levers" and "Compile time".
- **Issue #7013** (2026-07-07, closed stale 2026-10-06): on GPU, levanter's dense-LM attention defaults to NVTE, which is not installed, so it falls back to the slow path. **PR #7015** wired FA4 into `dot_product_attention`: 3.4× faster than `jax_flash` (seq 8192, GQA 32/8, hd128: 11.3 vs 38.9 ms fwd+bwd), parity-tested. It was **closed unmerged** on 2026-08-14. The grug FA4 kernels are present in the fork (`lib/levanter/src/levanter/grug/attention/_fa4_cute*.py`) but cannot be reached from the Qwen3 model path. https://github.com/marin-community/marin/pull/7015
- Caveat for this owner: the fast native SM90 FA4 backward was gated to head_dim 128 (and 192 after the MLA work). Other shapes fall back to a generic segmented path that is "~40× slower per call" (#6979 comment of 2026-07-13). The owner's head_dim is 64 at 130m/300m/520m and 128 only at 1_2b (`della_muonh_qwen3_scaling.py:427-430`, hidden/heads).
- `jax.nn.dot_product_attention(implementation="cudnn")` was rejected upstream because it has "no packed segment_ids" (#7013). The private `jax._src.cudnn.fused_attention_stablehlo` in JAX 0.11.1 does take `q_offsets`/`kv_offsets` (THD packing; `fused_attention_stablehlo.py:120,277`).
- Since July the upstream GPU work has gone to MoE and B200/GB200 (hero run MFU 28.2 → 31.2%, PR #9708, 2026-10-06), AMD (#9462, #9812), and fused Pallas-Triton ShortConv (#8369). Nothing new for small dense models on H100.
- Grug (`lib/levanter/src/levanter/grug/`) is upstream's "plain JAX, explicit sharding" path. MuonH/AdamH, FA4, the CE kernels and stacked scan are already there.

### 2.7 llm.c (C/CUDA)
- GPT-2 1.6B on 8×H100, 2024-07-11: ~381k tok/s (47.7k tok/s/GPU), **~47.5% bf16 MFU**, 2.75 s/step at 2^20 tokens, seq 1024; 19% faster and 23 GB lighter than the PyTorch reference of the time. https://github.com/karpathy/llm.c/discussions/677
- Hackability: every new layer needs a hand-written backward kernel. Not a research base. It shows what fusing everything buys at 1.5B: ~48% MFU without FP8.

### 2.8 Attention kernels
| kernel | H100 perf | varlen / packing | head dims | FP8 | access from |
|---|---|---|---|---|---|
| FlashAttention-3 | 740 TFLOP/s fp16 (75%), ~1.2 PFLOP/s FP8; 1.5–2.0× FA2 (arXiv 2407.08608, 2024-07-11) | `cu_seqlens` varlen, sliding window | 64/96/128/256 (fork-dependent) | fwd FP8 | PyTorch (nanoswe uses it via `kernels`) |
| FlashAttention-4 (CuTe DSL) | Blackwell-first: 1613 TFLOP/s on B200 (arXiv 2603.05451, MLSys 2026); no published H100 number found | segment lower bounds (grug binding) | SM90 fast bwd only hd128/192 in grug | — | JAX via grug FFI; PyTorch via `flash_attn.cute` |
| cuDNN SDPA (9.x) | similar to FA3 on Hopper (no number found for 9.19) | THD ragged offsets; JAX public API exposes only seq lengths | ≤256 | FP8 fwd/bwd in cuDNN | JAX `jax.nn.dot_product_attention(implementation="cudnn")`; TE; PyTorch SDPA |
| tokamax (openxla) | Pallas Mosaic-GPU SM90 fwd+VJP kernels, Triton kernels (repo pushed 2026-10-24, 317 stars) | mask objects | — | — | JAX (`tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py`) |
| levanter `JAX_FLASH` (current) | 3.4× slower than FA4 on the #7015 shape; host-synchronized while loops | segment ids; block must divide SEQ_LEN | any | — | current runs |
| JAX Pallas `attention_mgpu.py` | Mosaic GPU Hopper example kernel | — | — | — | `jax/experimental/pallas/ops/gpu/` in jax 0.11.1 |

### 2.9 Fused linear cross-entropy
- Liger FLCE (arXiv 2410.10989, 2024-10): chunked logits, gradient computed in the forward, no recompute; +20% average throughput and −60% memory vs HF. In JAX (marin `dial_h100_dev`): lm_head-only 50.3% MFU on H100 with one chunk per device (#6979).
- Apple Cut Cross-Entropy (arXiv 2411.09009, ICLR 2025): never materializes logits; loss memory 24 GB → 1 MB at Gemma 2 2B. It recomputes in the backward, so it is slower than Liger-style when memory is not the limit (nanoswe `chunked_lce.py:5-6`: "the reason CCE is slow on B200").
- Levanter in the fork: `xla` / `batched_xla` / `pallas_tpu` only (`lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/`). No GPU Pallas or Liger kernel. Measured fast-backward env `LEVANTER_CE_XLA_FAST_BWD=1`: −24 to −33% head time at 8K–64K (speed workflow, A100/A40 microbenchmarks). The pls session found a Pallas GPU CE forward 8–9× slower than XLA (needs_from_sessions.md).
- tokamax `linear_softmax_cross_entropy_loss`: `chunked_xla` on GPU; the Mosaic kernel is TPU-only.

### 2.10 Muon / Newton-Schulz and distributed Muon
- Cost at this owner's sizes is small. ESTIMATE for 300m: 5 NS iterations over 12 layers × (3 MLP 768×2688 + 4 attn 768²) ≈ 2e12 FLOP per step, against ~8.7e14 FLOP of model work per step (0.2%). In f32 on CUDA cores (~67 TFLOP/s) that is ~30 ms replicated or ~8 ms sharded over 4 GPUs (≈1–2% of 1.3 s). Upstream measured real NS costing 3.7 MFU points (34.4 → 30.7%) on a 10B MoE (#6979, 2026-07-07).
- Levanter forces NS column-sharded in f32 over `data`: 35 all-reduces on [12,768,768] f32 per step (speed workflow, `util.py:282`).
- Implementations: modded-nanogpt distributed Muon (#6, owner-assigned matrices + all-gather) and the Triton symmetric matmul for X·Xᵀ (#27); nanochat `DistMuonAdamW`; torchtitan DistributedMuon with overlap (v0.3.0, #4060); Megatron `emerging_optimizers` (owner's MuonH port); DMuon (arXiv 2606.27153, 2026-06-25): optimizer step 6.85–163× faster than vanilla distributed Muon, end-to-end near AdamW; MatrixFSDP (arXiv 2607.05895): 4.2× faster optimizer step than FSDP2-Muon on one node.
- Porting MuonH/AdamH exactly: levanter `optim/muonh.py` 188 + `adamh.py` 161 + `util.py` 322 lines; Megatron port +98 lines. The details that must match: NS coefficients and step count, f32 NS, the hyperball renormalization `W ← R·normalize(W − lr·R·normalize(u))`, and AdamH's whole-table norm. The Tokenizer Size session needs per-row control that the whole-table norm prevents (needs_from_sessions.md, items 3–4). Verify a port by a one-step parity test on identical weights and gradients.

### 2.11 Compilers and compile time
| stack | measured | source |
|---|---|---|
| levanter 300m | startup to first step 45 s fresh (train_step compile 9.6 s), 92 s on resume | speed workflow, "find:pipeline-overheads" |
| levanter pls d48 | first-step compile 70 s; jaxpr/HLO logging 3.5–4 min per segment; 30-step smoke ~20 min per resume segment | needs_from_sessions.md |
| levanter grug MoE 24L | unrolled ~26 min, scanned ~7 min; FA4/CuTe JIT is not covered by the XLA cache; XLA autotune results not persisted (`JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`) | throughput_tuning_h100.md @27c763ac24 |
| torch.compile nanochat d24 | step 0 = 113 s (1 GPU), 130 s (8 GPU), including FA3 load | nanoswe logs, section 1 |
| torch regional compile | 67.4 s → 9.6 s cold, 2.4 s warm (Diffusers, H100) | https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/ |

Both sides fall to seconds with a warm persistent cache. A 48-layer model must compile one block (scan in JAX, regional/`nested_compile_region` in torch). Each pls switch currently costs a fresh compile per resume segment.

### 2.12 JAX Pallas / Mosaic GPU / Triton-in-JAX
- jax 0.11.1 ships Pallas Mosaic-GPU kernels for Hopper (`hopper_matmul_mgpu.py`, `attention_mgpu.py`, `ragged_dot_mgpu.py`, collective matmul). Pallas-Triton backs upstream `ragged_dot` and the ShortConv kernel (#8369). tokamax packages SM90 Mosaic attention.
- Risk: young APIs. One owner attempt (Pallas GPU CE) ran 8–9× slower than XLA. FA4 CuTe JIT needs its own cache. Kernel authorship in JAX on GPU is possible but costs more than Triton in PyTorch, where Liger, FA3 and modded-nanogpt kernels can be used as-is.

## 3. Data caches
Levanter caches are zarr v3 arrays: `input_ids/{data,offsets}`, int32, blosc-zstd in sharded chunks (`marin_store_big/fineweb-edu-10B-v32000/train/input_ids/data/zarr.json`; 18 GB on disk). tensorstore or zarr-python can read them from PyTorch as well. A one-time conversion to a flat uint32 memmap + offsets (nanochat/Megatron style) costs ~4 B/token, ≈40 GB for 10B tokens (ESTIMATE). The 128K vocab exceeds uint16. Exact data order (the Tokenizer Size session needs it) depends on levanter's permutation (linear permutation, seed 42; launcher line 218), which any new loader must replicate bit-exactly or replace with an explicit index file.

## 4. Comparison

| stack | H100 throughput at the nearest size | hackability (files per new loss/arch) | MuonH/AdamH exactness | existing caches | maturity / risk |
|---|---|---|---|---|---|
| levanter, current fork | 300m: 101k tok/s/GPU, ~12% (6N); pls d48 27% | 5 places + test (pls session) | native | native | known failure modes (OOM, while-loop attention, SEQ_LEN constraints) |
| levanter/grug best upstream | dense 2.7B 56.2%; lm_head-only 50.3% (8×H100, #6979) | grug: plain JAX, ~2–3 files (ESTIMATE) | native (grug has MuonH) | native | code on unmerged branches; FA4 SM90 bwd only hd128/192 |
| MaxText | Llama2 7B ~54% (ESTIMATE from 0.721 s/step) | 2–4 | port to optax | convert or reader | Google-maintained; GPU path depends on TE/cuDNN |
| nanochat / nanoswe | d24 0.73B: 70.9k tok/s/GPU, 31% 6N-only with FP8 (owner, Della) | 2 | port (~300 lines; owner did it for Megatron) | reader or convert | small code; single node; owner has run it |
| modded-nanogpt | 124M-class: ~650k tok/s/GPU (ESTIMATE from log) | 1 (monolith) | port | convert | tuned to a single benchmark |
| torchtitan | Llama 8B: 8,532 tok/s/GPU FP8 (2024); 33–39% MFU bf16 | 2–4 | port; DistributedMuon exists | reader | Meta-maintained; v0.3.0 2026-09-03; small-model numbers unpublished |
| Megatron-Core + TE | Llama3 8B FP8: 14,451 tok/s/GPU, 744 TFLOP/s | 4–8 | ported once (+98 lines), never run | convert to IndexedDataset | heavy; container-pinned; dropped once already |
| llm.c | GPT-2 1.6B ~47.5% | new backward kernel per layer | hand-write | convert | not for research |

## 5. The five techniques with the largest expected speedup for this owner

Ranked by step time saved on the owner's actual runs. All speedups are ESTIMATES unless a measurement is cited.

1. **A fused, chunked linear-CE kernel for all readout heads.** One chunk per device, bf16 logits, gradient computed in the forward (Liger style). For aux heads, compute the gradient only and the loss every N steps.
   - Evidence: in JAX on H100, an lm_head-only model went from 35–41% to 50.3% MFU (#6979). Grad-only CE gave +31.5% in a CE-bound model (#6979).
   - pls d48: the 47 heads do 47 × 6 × 512 × 128,256 × 524,288 ≈ 9.7e15 FLOP per step. In 12.3 s on 4 GPUs that is ~197 TFLOP/s/GPU (20%). At 50% MFU it would take ~4.9 s, so the step goes 15.2 → ~7.8 s (**~1.9×**).
   - 300m: the 128K head goes 0.25–0.29 s → ~0.12 s (**~10%**).
   - Also removes the f32-logits memory, which feeds item 3.
2. **A real flash-attention kernel that handles any length and packing.** Options: cuDNN via THD offsets or TE in JAX, FA4 for hd128, FA3 in PyTorch.
   - Removes the `SEQ_LEN % 1024` constraint and the host-synchronized loop overhead, which is 42–56% of step time at 4224/4480 (speed workflow).
   - Charwin runs: 2.5–2.7 s → ~1.3–1.4 s (**~1.9×**).
   - Seq 4096: only a few percent; the loops cost ~1.5% (speed workflow), and attention is ~16% of FLOPs at 300m.
   - Upstream measured FA4 3.4× faster than `jax_flash` (#7015). Check head_dim 64 coverage first.
3. **Memory-aware remat and buffers.** Turn off full per-layer recompute when there is headroom: 300m uses 12.6/71 GiB, and the levanter default is full remat (`llama.py:80`). That removes up to ~25% of trunk FLOPs (8N → 6N). Also cut the hc/mhc temp buffers (30–53 GiB) so those runs fit microbatch 4.
   - Measured: hc d48 at microbatch 1 vs 4 is 25.4 vs 16.0 s/step (**1.59×**, pls session).
   - Remat-off gain at 300m: ESTIMATE 10–20%; needs an A/B. Upstream found full remat best only when memory-bound at 2.7B/b256.
4. **Remove host syncs and overlap the rest.**
   - Run the loss `.item()`, NaN checks and watch hooks every K steps. Upstream measured +2.5 MFU pts. Locally the hooks cost 1.0% and the grad-norm watch takes 43.5 ms every 10th step.
   - Enable the latency-hiding scheduler and pipelined collectives; 69/72 collectives are synchronous now (~1%).
   - Use command buffers or CUDA graphs. `+WHILE,+CONDITIONAL` was ~2× on a contended A40 at seq 4224, a direction-only result.
   - Run Paloma eval asynchronously or on a separate GPU (3.5% of wall).
   - Expected at 4096: **~5–8%** combined.
5. **FP8 GEMMs (tensorwise or current scaling) only for widths ≥1024.**
   - Measured +8.6% step speed at d1536 (nanoswe bf16 2,061 → FP8 1,883 ms). torchtitan measured +28% at 8B.
   - Expect ≤5% at d512/d768: the GEMMs are small and the head/attention dominate (ESTIMATE).
   - It adds a numerics confound to research comparisons. Treat it as an opt-in for 520m/1_2b throughput runs, never for an A/B of a research variant.

Not in the top five:
- Distributed-Muon tricks: NS is ≈1–2% at these sizes (ESTIMATE, section 2.10).
- Async TP / context parallelism: no benefit on one node at ≤2B.
- Multi-node FSDP: upstream shows flat weak scaling (#6979), but the owner's jobs are single-node.

Bottom line: a JAX rewrite that adopts items 1–4 lands near upstream's own H100 numbers (50–56% MFU). A minimal PyTorch trainer (nanochat-style) gets items 1, 2 and 5 off the shelf (Liger/FA3/torchao), at the cost of porting MuonH/AdamH (~300 lines plus a parity test) and the cache reader. The kernel and remat choices matter more than the framework choice.
