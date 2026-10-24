# Research needs, as answered by the two live pretrain sessions (2026-10-24)

Questions sent by marincpp: (1) what you change most, which files each change touches; (2) what is slow, with numbers;
(3) what breaks; (4) what you cannot express today. Answers are verbatim.

## Tokenizer Size session (repo project/marin, branch vocab-overfitting)

1. Most frequent change: a new training variant switched by env flags. Each one touches the trainer experiments/references/della_muonh_qwen3_scaling.py (env parse, run-id tag, guard on allowed combinations, model or optimizer config), one model or optimizer module (e.g. dropout_qwen3.py, composed_vocab_qwen3.py, freeze_rows_muonh.py, embed_wd_muonh.py), one unit test under tests/, and a plan logs/plans/<name>.{txt,sbatch} checked by scripts/della/check_plans.py. Data changes use vocab_materialize_passes.py (pre-built pass caches under marin_store_big/materialized/) and convert_small_vocab_cache.py (truncated tokenizers). Evals are standalone modules: vocab_token_losses.py (per-position Paloma), vocab_temp_losses.py, vocab_subset_losses.py, vocab_refit.py.
2. Slow: SEQ_LEN 4224/4480 with flash block 128 steps at 2.5-2.7 s/step against 1.2-1.4 at 4096 (300m, 4x H100). MFU is about 10%, and memory is about 12.6 of 71 GiB. A charwin eval takes about 74 s per 1000 steps. A per-position Paloma eval takes about 4-5 min per checkpoint at 128K. Exact macro bpb needs byte tables offline: the W&B macro reads about 0.004 high. Re-reading about 200 W&B runs took about 15 min before I added a cache.
3. What breaks most often:
   - SEQ_LEN not a multiple of 1024 failed the flash kernel (smoke of w32-w34; fix 555a625346).
   - Zeroed input rows with hybrid_norm gave NaN at step 1 (w4; fix 70c4d0edfa).
   - AdamH sizes the step by the whole table, so freezing some rows silently sped up the rest (fop runs, Q5b).
   - Packed task limits were shorter than the run (w32).
   - The temp checkpoints that resume uses sit at marin_store_big/tmp/checkpoints-temp/<full path>, which is hard to find.
   - A stray XLA_PYTHON_CLIENT_MEM_FRACTION from the submit shell halved the GPUs.
   - The W&B proxy blocks uploads.
4. Hard or impossible now:
   - Per-position or per-group losses logged during training on the repeated subset and on Paloma. We need separate eval jobs for them.
   - Per-row optimizer control, such as a different LR or a freeze for some rows, without the whole-table AdamH norm interfering.
   - Rebuilding the exact data order and the unseen-sequence ranges. I reimplemented this twice: ov_memorization_eval.py and vocab_refit.py.
   - Forking a run from a mid-run checkpoint with an exact data-order match.
   - Fast attention for any SEQ_LEN.
   - Exact bpb inside the run.
5. Speed workflow wf_cb88f9ae-72b is still running. Raw agent outputs: /home/USER/.claude/projects/-home-USER/c574d901-7815-4be8-b851-ba33c8717449/subagents/workflows/wf_cb88f9ae-72b/journal.jsonl. Summary will go to docs/della/speed-workflow.md in the marin repo when it ends. H100 A/B timings come later.

## pls session (JAX/levanter, 130m width 512, 48 layers, 4x H100, one 128K-vocab readout head per layer)

1. 最常改的是深度结构和读出损失。结构在 experiments/references/depth_arch_qwen3.py（hc/mhc/attnres，扫描路径和循环路径各写一遍）。损失在 per_layer_qwen3.py（独立头、探针、可见坐标）。每个新开关要改 5 处：模型 config 字段和校验、per_layer_qwen3.py、della_muonh_qwen3_scaling.py（环境变量、run id 标签、model_extra）、pls_h100x8_single.sbatch 的 unset 列表、pls_arch3_fetchable.sbatch 的 design。另外还要写一个 CPU 测试，例如 scripts/della/pls_vnorm_test.py。
2. 慢的地方：
   - 每步时间。Pre-LN d48 是 15.2 s/step，其中 47 个额外读出头占 12.3 s（81%），约 263 TFLOP/s/GPU，是峰值的 27%。读出头受 f32 logits 的显存带宽限制。
   - microbatch。hc/mhc/attnres 因为 temp buffer 有 30–53 GiB，只能用 microbatch 1：hc d48 是 25.4 s/step，microbatch 4 是 16.0 s/step。
   - 每段开销。每个 resume 段先跑 30 步 smoke，约 20 min。第一步编译 70 s。jaxpr/HLO 日志每段 3.5–4 min。
   - checkpoint 每 10 min 存一次，每次 21 s，约占 3.4%。
   - eval。每 1000 步跑一次 per-layer eval，PDP=4 时 9 min，PDP=1 时 13.5 min，加起来约占 3.3%。
   - 排队。wq 作业通常等几小时到一天。
3. 常坏的地方：
   - OOM。显存碎片曾让两个 48 层 run 在第 ~260 步死掉（15266143、15266145）。
   - FSDP 分片下 48 个读出的损失随程序内容变化（15255032），所以改用 PARAM_MAP=vocab。
   - 8 GPU 下 HC/AttnRes 的 AllToAll rendezvous 卡住 40 s 后 XLA 中止（wq w16/w17）。
   - 一个 run 在 W&B 上显示 finished，实际只训练到第 259 步。
   - W&B 的 history 分页会丢一行、重复一行。
   - resume 段的 smoke 读到第一段的 wandb-summary，所以闸门检查的是旧值（刚发现）。
   - PDP=1 和 PDP>=2 走两条不同的 CE 反向代码，数值不同。代码里没有明说这一点。
4. 想做但很难的：
   - 按层、按张量读梯度和激活统计，又不增加 host 同步。
   - 对读出头写一个融合的 CE kernel：Pallas GPU 前向测出来比 XLA 慢 8–9 倍。
   - 显存可预测：现在只能靠试错设置 microbatch 和 mem fraction。
   - 一个 resume 安全、编译缓存能复用的 smoke 闸门。
   - 在 hook 里读每层的残差流，现在要另写 probe 程序。

## Owner directive (2026-10-24, in the marincpp session)

「记住我的想法是彻底重写 怎么快怎么来，可以不要python 甚至考虑不要繁重的jax部分 你看着办」: a complete rewrite, whatever is fastest; Python may be dropped; even the heavy JAX part may be dropped. Patching levanter in place is not the goal; it serves at most as the parity baseline.

## Owner scope (2026-10-24)

「只写完这两个实验组需要的范围，我的所有实验都是单机八卡及以下 尺寸你也能读到 我就是要在我的范围内定制化」: build only what the two experiment groups (pls and Tokenizer Size) need. Every run is one node, at most 8 GPUs. Model sizes are the ones in their launchers (130m..1.2B dense Qwen3-style; pls 48-layer width-512 with per-layer readout heads). Customize to this envelope: no multi-node, no general framework, no features neither group uses.

## Owner stage-1 goal (2026-10-24)

「你可以先从复现kaiyue baseline作为一个一阶段目标」: stage 1 of nanomarin = reproduce Kaiyue Wen's muonh_qwen3 baseline (della_muonh_qwen3_scaling.py, 130m first, then 300m/520m/1.2B) on the same data, steps and batch, matching the levanter runs within seed noise, at higher tokens/s.
