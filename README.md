# nanomarin

A speed-first rewrite of the marin/levanter pretraining stack, built only for the owner's own experiments: one node, at most 8 H100s, dense Qwen3-style models from 130m to 1.2B, and the 48-layer per-layer-readout models of the pls study.

Stage 1: reproduce Kaiyue Wen's muonh_qwen3 baseline within seed noise of the levanter runs, at higher tokens/s.

- `docs/survey/` — research needs, measured bottlenecks, code paths, external stacks, roofline
- `docs/spec/` — exact specification of the Kaiyue baseline and the parity test plan

`$ROOT` in the docs is the owner's scratch directory on Della; `USER`/`GROUP` replace cluster identifiers.
