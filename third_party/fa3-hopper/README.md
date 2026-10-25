# FlashAttention-3 (hopper/), vendored subset

Upstream: https://github.com/Dao-AILab/flash-attention, directory `hopper/` at commit `767386d` (2026-10-21), the 31
headers and `flash_prepare_scheduler.cu` that the sm90 forward and backward launch templates include, plus the
repository LICENSE (BSD-3-Clause). CUTLASS comes from `third_party/cutlass` at `7127592` (FA3's pin). No torch, no
`flash_api.cpp`: `src/kernels/attention/fa3_backend.cu` fills `Flash_fwd_params` / `Flash_bwd_params` itself and
`src/kernels/attention/fa3_inst_*.cu` instantiate the kernel variants marincpp uses (bf16, sm90, hd 64 MHA/GQA, hd 128
GQA; causal varlen and doc-causal batch).

`marincpp.patch` is the whole difference from upstream (`diff -ru a b` of the 31 source files, upstream in `a/`, the
vendored files in `b/`; regenerate it after any change and check that `patch -p1` on upstream reproduces the vendored
files).
Every changed line is marked with a `marincpp` comment. The patches:

1. Doc-causal masking (DESIGN.md §3.4 `doc_causal`). `flash.h`: `doc_lo` / `doc_hi`, per batch row and position the
   start of the query's segment and the end (exclusive) of the key's segment. With them set, FA3's `Is_local` kernels
   mask key `c` for query `r` unless `doc_lo[r] <= c <= r`: `block.h` bounds the forward n-block loop by the first
   query's segment start and the backward m-block loop by the last key's segment end, and gives the first unmasked
   blocks; `mask.h` uses the segment start as the left limit (forward) and the segment end as the bottom limit
   (backward, S transposed); the forward and backward mainloops pass the batch row's arrays to both, including the
   dQ writer warp of the backward, whose m-block range must equal the consumers'. Tiles wholly outside a segment are
   never visited; there are no padding rows.
2. Work-list backward (DESIGN.md §3.4 `persistent`). `tile_scheduler.hpp` `NmBwdListScheduler`: work item `i` is
   `items[i]` = (segment, n block, head) of a device list (segments longest first, then head, then n descending, so the
   CTAs running at once share one head's Q / dO in L2). `flash_bwd_launch_template.h` (`NmList`): main grid = the SM
   count (persistent) or one CTA per list slot; the preprocess and the dQ / dK / dV postprocess kernels take their
   (segment, block) tiles from lists too (`flash_bwd_preprocess_kernel.h`, `flash_bwd_postprocess_kernel.h`), so no
   grid covers empty segment slots. The lists are built in the graph by marincpp's `k_bwd_worklists`.
3. Persistent backward CTAs (`kv_sync`). Upstream's backward kernel is single-tile: the `KVEmpty` handoff between the
   dK/dV epilogue (whose shared memory aliases `smem_k` / `smem_v`) and the next tile's K/V load is commented out.
   `mainloop_bwd_sm90_tma_gmma_ws.hpp` and `flash_bwd_kernel_sm90.h` enable it at run time: consumers arrive once at
   start and after every stored tile, the producer warp waits before each K/V load.
4. Deterministic order. dQ: the doc-causal path accumulates, like FA3's causal path, the n blocks that visit an m
   block in descending order, and only they arrive (the visitors of m block `m` are exactly the n blocks from
   `doc_lo[first row of m] / kBlockN` to the causal limit); its deterministic grid is FA3's shortest-first
   `SingleTileBwdLPTScheduler` (highest n block first). Upstream's local path orders and pads over every n block of the
   row, ascending, which serializes the segments of a row. The work list holds the n blocks of a (segment, head) in
   descending order, so every item waits only for items earlier in the list and the persistent grid cannot deadlock.
   (Measured in E1b: ascending order costs 1.4x (work list) to 2.3x (doc-causal) of the nondeterministic backward,
   because each n block then waits for its predecessor's dQ write at every m block.) GQA dK/dV (`first_store`,
   `epilogue_bwd.hpp`, `copy_sm90_bulk_reduce.hpp`): in deterministic mode the first head of a kv group stores its
   tile with a plain bulk copy and the others add in head order, so the accumulators need no zeroing. Hand-off
   (`SM90_BULK_HANDOFF`, dQ writer and GQA dK/dV epilogue, deterministic only): upstream releases the semaphore after
   `tma_store_wait` (`cp.async.bulk.wait_group.read`), which waits only for the shared-memory reads, so the next
   writer's bulk add was not ordered after this one's by the PTX memory model. Each writer now waits for its global
   writes (`cp.async.bulk.wait_group 0`) and fences the async proxy (`fence.proxy.async.global`) before the release;
   the next writer fences after its acquire, before its own bulk add.
5. LSE layout override (`lse_head_stride`, `lse_batch_stride`): the batch-mode (doc-causal) LSE is written as one
   [H, T] tensor, the layout `AttnFwdArgs::lse` defines.
6. `cuda_check.h`: a failed CUDA or CUTLASS call throws `std::runtime_error` (marincpp reports it) instead of `exit(1)`.

Build flags of the FA3 units (from `hopper/setup.py`): `--use_fast_math --expt-relaxed-constexpr
-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DNDEBUG`, sm_90a only
(`src/kernels/attention/CMakeLists.txt`).
