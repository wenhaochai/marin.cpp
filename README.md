# marin.cpp

A speed-first rewrite of the marin/levanter pretraining stack, built only for the owner's own experiments: one node, at most 8 H100s, dense Qwen3-style models from 130m to 1.2B, and the 48-layer per-layer-readout models of the pls study.

**Archived 2026-10-25.** The owner stopped the project during stage S7. State at archive:

- S0-S6 done (kernels, GEMM layer, FA3 attention, fused cross-entropy, MuonH/AdamH/Adam engine); see `docs/BUILD_LOG.md`.
- S7 partly done: the full 130m forward and hand-written backward (`src/model/baseline.cpp`), the optimizer binding, and a multi-GPU NCCL trainer with whole-step CUDA graph capture (`src/train/`).
- Parity against levanter (one A100, reference attention, 32 microbatches of 4 rows): step-0 loss diff 4.5e-5 (T2.1 pass); trunk gradient rel 2.6e-3 to 3.5e-3, cos 1.0000; step-1 loss diff 1.9e-4 to 5.4e-4 (T2.7 pass); step-2 loss diff +1.2e-2 (T2.7 fail, bound 2e-3; not yet isolated: the bf16 embedding accumulator does not close it).
- Open: the step-2 gap, the H100/FA3 run of the same test, ZeRO-1, throughput against levanter, S8-S11.

Stage 1: reproduce Kaiyue Wen's muonh_qwen3 baseline within seed noise of the levanter runs, at higher tokens/s.

- `docs/survey/` — research needs, measured bottlenecks, code paths, external stacks, roofline
- `docs/spec/` — exact specification of the Kaiyue baseline and the parity test plan

`$ROOT` in the docs is the owner's scratch directory on Della; `USER`/`GROUP` replace cluster identifiers.
