# Data pipeline and exact data order

Stage-1 spec, section "data". Target: a new loader that produces, for every step k, the same token batch as the
levanter baseline runs `muonh-qwen3-{130m,300m,520m,1_2b}-della4xh100` (W&B `reself/marin-della`), and the same
Paloma eval set and eval arithmetic.

**Status of every claim.** Facts marked **[verified]** were checked on 2026-10-24 by
`cache/data/_tools/data_order_ref.py` (run on della-vis1, CPU, marin `.venv`, log `cache/data/_tools/run.log`). That
script builds the training set through levanter's own `LmDataConfig.train_set` with the logged config, rebuilds the same
order with an independent closed form (numpy Threefry-2x32, LCG, raw zarr3 reads through tensorstore), and asserts that
the two agree. Its outputs are in `cache/data/data_ref.json`. All other facts cite code.

**Abbreviations.** `M/` = `$ROOT/project/marin` (read-only reference). `L/` =
`M/lib/levanter/src/levanter`. `J/` = `M/.venv/lib/python3.12/site-packages/jax/_src`. `S/` =
`$ROOT/marin_store_big`. `N/` = `$ROOT/project/marin.cpp`.
Line numbers are for marin HEAD `1793624070`.

**Code version.** Each run is a chain of resume segments, and each segment logs its own `git_commit`; the run-level
W&B config keeps the last segment's values. The cached segment configs (`N/cache/wandb/config/*.yaml`) show 59ac0d647d
(130m, its only segment), 28fd599610 (300m, segment `run-20260914_123815`), cead09a259 (1_2b, last segment) and, for
520m, **two** commits: 59ac0d647d in the first segment (`run-20260914_063908`, steps 0..119) and ac6af86874 in the last
(`run-20260915_170105`). The first 520m segment ran with `per_device_parallelism` 64 (section 5). `git diff <commit> HEAD` over `L/data`, `L/store`,
`L/main/train_lm.py`, `L/eval.py`, `L/models/lm_model.py`, `L/tokenizers.py`, `M/lib/marin/src/marin/experiment/data.py`,
`M/experiments/datasets` and `M/uv.lock` is empty for all four. Only `L/trainer.py` differs (a contextvar for the
traced step, no data effect), plus the launcher (new env knobs whose defaults keep the baseline). So the HEAD code below
is the code that ran. Library versions: jax/jaxlib 0.11.1, numpy 2.3.5, tensorstore 0.1.84, tokenizers 0.22.2,
transformers 5.12.1.

---

## 1. Summary in one block

```
tokens  = int64 zarr3 array S/fineweb-edu-10B/2026.06.28/train/input_ids/data, length 10,000,000,738
N       = 10,000,000,738 // 4096 = 2,441,406 sequences; sequence s = tokens[4096*s : 4096*s + 4096]
mixture index j (global example counter) -> sequence id:
    blk  = j // 2048 ;  p = pi_blk[j % 2048]          # pi_blk: Threefry-keyed permutation of 0..2047
    r    = (blk*2048 + p) mod N                        # "restart": wrap, no reshuffle
    s(j) = (830293 * r + 746237) mod N                 # LCG permutation, data_seed 42
step k uses j = k*B .. k*B + B - 1 (B = 128 at 130m/300m, 256 at 520m/1_2b), rows in that order
```

The precomputed table `N/cache/data/fineweb_order_seed42.u32` holds s(j) for j = 0 .. 5,859,327 (all four sizes),
little-endian uint32, sha256 `dde0ca7b087499ae172baae940dc33c057356512b58d25a5dbffcf5989c7655b` **[verified]**.

---

## 2. Training corpus

### 2.1 Location and provenance

| item | value | source |
|---|---|---|
| dataset handle | `fineweb_edu_10B_dataset()` = `pretokenized("fineweb-edu-10B", repo_id="marin-community/fineweb-edu-pretokenized-10B", tokenizer=marin_tokenizer, version="2026.06.28")` | `M/experiments/datasets/prebuilt_caches.py:30-37` |
| launcher use | `train = {fineweb_edu_10B_dataset(): 1.0}` | `M/experiments/references/della_muonh_qwen3_scaling.py:544` |
| cache dir (logged) | `S/fineweb-edu-10B/2026.06.28` | config yaml `data.components.fineweb-edu-10B.cache_dir` |
| split dir | `S/fineweb-edu-10B/2026.06.28/train` | `L/data/text/datasets.py:569-574` (`cache_dir/<split>`) |
| upstream | HF dataset repo `marin-community/fineweb-edu-pretokenized-10B`, revision `null` (main); its README: subsample of fineweb-edu (`HuggingFaceFW/fineweb-edu@3c452cb`), "Seed 42" (the upstream subsampling seed, unrelated to `data_seed`), 10,000,000,738 tokens | `S/fineweb-edu-10B/2026.06.28/{.artifact.json,README.md}` |
| rows / tokens | 9,966,814 documents, 10,000,000,738 tokens **[verified]** | ledger `train/shard_ledger.json`, offsets array |

**Caveat (restore).** The files on disk were re-downloaded on 2026-09-28 (`.provenance.json` access_time), after the
09-24 store cleanup deleted the copy the 09-14 baseline runs read (`M/docs/della/sampled-softmax.md:94-101`). The path
is the same, the document and token counts equal the counts recorded before the cleanup, and the 09-28 restore
reproduced the 130m pool (restore baseline macro 4.1820, inside the pool; `sampled-softmax.md:122-123`). A token-level
identity with the 09-14 copy cannot be proven now; the batch hashes in section 7 pin the current copy.

### 2.2 Tokenizer

* Name `marin-community/marin-tokenizer` (`M/experiments/marin_tokenizer.py:23`): the Llama 3 tokenizer with a custom
  chat template; ids 128002/128003 renamed `<|start_think|>`/`<|end_think|>` (`marin_tokenizer.py:29-32`). Chat
  template does not affect pretraining data.
* Loaded by levanter as `load_marin_tokenizer(self.tokenizer)` (`L/data/text/datasets.py:705-710`,
  `L/tokenizers.py:733-747`); offline from the HF hub cache snapshot
  `$ROOT/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`.
* `len(tokenizer)` = 128256, BOS = `<|begin_of_text|>` = 128000, EOS = `<|end_of_text|>` = 128001 **[verified]**
  (`tokenizer_config.json`: `bos_token`, `eos_token`). The model vocabulary axis is 128256.
* The training data is pre-tokenized; the new loader never runs the tokenizer for training. The tokenizer is needed
  only for the bytes-per-token table of the bpb metric (section 8.4).

### 2.3 On-disk format

Levanter `TreeCache`, **consolidated** layout (the ledger has no `layout` key; default `consolidated`,
`L/store/cache.py:57,663`), one field `input_ids` stored as a `JaggedArrayStore` (`L/store/jagged_array.py:203-249`):

```
train/shard_ledger.json        {"total_num_rows": 9966814, "is_finished": true, ...}
train/input_ids/data/          zarr v3 array, int64, the concatenated tokens of all documents
train/input_ids/offsets/       zarr v3 array, int64
```

* `offsets[0]` = number of rows (9,966,814); `offsets[i+1]` = exclusive end of row i in `data`; row i spans
  `data[offsets[i] : offsets[i+1]]` with `offsets[0]` read as 0 for i = 0 (`jagged_array.py:217,276-305,542-554`).
  `offsets[1]` = 588 (first document has 588 tokens); `offsets[9966814]` = 10,000,000,738 = data size **[verified]**.
* zarr metadata (both arrays, `zarr.json`): `data_type` int64, logical shape 2^54 (resizable; the real length comes
  from the offsets), `chunk_grid` regular with `chunk_shape` [134217728], codec `sharding_indexed` with inner
  `chunk_shape` [262144], inner codecs `bytes` (little-endian) then `blosc` (`cname` lz4, `clevel` 5, `shuffle`
  byte-shuffle, `typesize` 8, `blocksize` 0), `index_codecs` `bytes` (little-endian) + `crc32c`, index location not
  given (zarr v3 default: end of shard). `fill_value` 0.
* Files: `data/c/0` … `data/c/74` (75 shard files, 134,217,728 elements each, the last one partial),
  `offsets/c/0` (one file). Each shard file holds 512 inner chunks and ends with the shard index: 512 entries of
  (offset uint64 LE, nbytes uint64 LE) followed by a 4-byte crc32c; an entry of (2^64-1, 2^64-1) is an empty chunk
  (fill value 0).
* Paloma caches use the **sharded** layout (section 8.1) and blosc `zstd` level 1 instead of lz4; the element layout
  is the same.

Reading from C++: tensorstore's C++ API opens these arrays directly with the spec
`{"driver": "zarr3", "kvstore": {"driver": "file", "path": ".../input_ids/data"}}` (the reference script reads them
this way, independent of levanter). Recommendation, not done: convert once to a flat uint32 file (40.0 GB) with a
sha256, because every id is < 2^17, and mmap it.

### 2.4 Document format

Each document is stored as `[BOS, text tokens …, EOS]`, with no padding between documents. On two 20M-token windows
(offset 0 and offset 5,000,000,369; 19,649 and 20,302 whole documents) **[verified]**:

* every document starts with 128000 and ends with 128001;
* 128000 occurs once elsewhere per window and 128001 occurs once (second window) not at a document end, i.e. these
  ids appear inside document text, rarely (about 1 in 20,000 documents);
* min id 0, max id 128001; shortest document 58 tokens; mean 1003.3 tokens over the whole corpus
  (10,000,000,738 / 9,966,814).

The cache ledger carries no preprocessor metadata (levanter logs "Metadata mismatch … preprocessor_metadata"; benign),
so `enforce_eos=True` in the config (`L/data/text/datasets.py:636`) does not alter the stored tokens; BOS/EOS come
from the upstream tokenization.

---

## 3. Documents to sequences (packing)

Component format `TextLmDatasetFormat(text_key="text")`, `pack: null` → `_effective_pack` returns False
(`L/data/text/datasets.py:362-370`) → `CausalLmDataset(TokenSeqDataset(cache, 4096))`
(`datasets.py:529-534`). `TokenSeqDataset` (`datasets.py:78-114`):

* length `N = data_size // seq_len` (`datasets.py:91-92`) = 10,000,000,738 // 4096 = **2,441,406**; the last 1,762
  tokens are never used **[verified]**;
* sequence s = `data[4096*s : 4096*s + 4096]` (`datasets.py:105-106`, `L/store/cache.py:573-580`).

So training is a plain concatenation of all documents in cache order, cut into fixed windows. No BOS is inserted at a
window start, no EOS is added, a document freely crosses window boundaries, and nothing is padded. Example: step 0,
row 0 starts `[4, 8834, 23246, 605, 4, 50585, 595, 1912]` (mid-document) **[verified]**.
`train_seq_len` 4096 (`della_muonh_qwen3_scaling.py:816`); `Pos = model.max_Pos.resize(4096)` (`L/main/train_lm.py:245`).

### 3.1 The training example built from one sequence

`GrugLmExample.causal(tokens, eos_id=128001, block_cross_document_attention=True)` (`datasets.py:157-168`,
`L/data/text/examples.py:123-175`):

| field | value | code |
|---|---|---|
| tokens | the 4096 ids, int32 inside jit (stored int64; JAX x64 is off) | **[verified]** dtype int32 |
| loss_weight | float32, 1 at positions 0..4094, 0 at position 4095 (no next token); sum 4095 per row | `examples.py:142-149,200-209` **[verified]** |
| segment ids | `seg[0] = 0`, `seg[i] = #{t < i : tokens[t] == 128001}` for i ≥ 1, i.e. `cumsum(roll(tokens,1)==EOS)` with position 0 forced to 0; int32; used as (q, kv) segment ids | `examples.py:166-171` **[verified]** for step 0 row 0 (EOS at 1380, 2370, 2578, 3243, 3888 → 6 segments) |
| attention | causal and same-segment only | `examples.py:165-173`; config `block_cross_document_attention: true` |
| positions | `arange(4096)` per row, **not** reset at document boundaries | `compute_next_token_loss` calls `self.activations(example.tokens, example.attn_mask, key=key)` with no `pos_ids` (`L/models/lm_model.py:337`); Qwen3 uses the standard `Attention` (`L/models/qwen.py:146,155`), whose default is `pos_ids = hax.arange(x.resolve_axis("position"))` (`L/layers/attention.py:1441-1442`) |

Consequences a reimplementation must copy:

1. Segments are derived from the **EOS token value**, not from the document offsets. An EOS inside text (section 2.4)
   starts a new segment; a BOS inside text does not.
2. The EOS token belongs to the segment it ends; the next token (BOS of the next document) opens the new segment.
3. The loss at the EOS position (predicting the next document's BOS) is **kept** (weight 1). Only position 4095 has
   weight 0.
4. RoPE positions run 0..4095 across documents inside a window, even though attention is blocked between them.

---

## 4. Exact data order

### 4.1 Keys

All randomness of the data order comes from `data_seed = 42` (`della_muonh_qwen3_scaling.py:820`). The trainer seed
(`seed: 0`, `:812`) does not touch the data order.

```
data_key            = PRNGKey(42)                       # L/main/train_lm.py:218-222  -> raw (0, 42)
mix_key, shuffle_key = split(data_key)                   # L/data/text/datasets.py:788
perm_key            = split(shuffle_key)[1]              # key_iterator, L/utils/jax_utils.py:316-321, used at datasets.py:839-841
```

Values (uint32 pairs) **[verified]**: `mix_key = (1832780943, 270669613)`, `shuffle_key = (64467757, 2916123636)`,
`perm_key = (2350016172, 1168365246)`.

Threefry, as JAX 0.11.1 runs it with `jax_threefry_partitionable = True` (logged `trainer.jax_config`; also the JAX
default, `J/config.py:1411-1413`; `L/trainer.py:98-99`):

* `T(k, (x0, x1))` = Threefry-2x32 with 20 rounds: `ks = (k0, k1, k0 ^ k1 ^ 0x1BD11BDA)`; `x0 += ks0; x1 += ks1`;
  for i = 0..4: four rounds with rotations `[13,15,26,6]` (i even) or `[17,29,16,24]` (i odd), each round
  `x0 += x1; x1 = rotl32(x1, r); x1 ^= x0`; then `x0 += ks[(i+1)%3]; x1 += ks[(i+2)%3] + (i+1)`. All uint32 wrapping
  (`J/random/threefry2x32.py:109-114,129-177`).
* `split(k)[i] = T(k, (0, i))` for i = 0, 1 (`threefry2x32.py:305-310`, `J/random/prng.py:798`).
* `fold_in(k, d) = T(k, (0, d))` (`threefry2x32.py:312-318`, seed `(d >> 32, d & 0xFFFFFFFF)` = `(0, d)`).
* `bits32(k, n)[i] = T(k, (0, i)).x0 ^ T(k, (0, i)).x1` for i = 0..n-1 (`threefry2x32.py:333-350`).

The reference script implements exactly these four lines in numpy and matches JAX on every key and permutation.

### 4.2 Stage A: LCG permutation of the 2,441,406 sequences

`shuffle: true`, `permutation_type: linear` (logged; set by `dataclasses.replace(mixture(..., shuffle=True),
permutation_type="linear")`, `della_muonh_qwen3_scaling.py:761`, overriding marin's Feistel default at
`M/lib/marin/src/marin/experiment/data.py:321-328`). `train_sets` shuffles the one nonzero-weight component
(`datasets.py:824-841`) with `PermutationDataset(ds, perm_key, "linear")` (`L/data/dataset.py:102-103,375-411`),
whose permutation is `LcgPermutation(N, perm_key)` (`L/data/_prp.py:61-78,88-146`):

```
seed = randint(perm_key, (), 0, 2**30)        # _prp.py:81-85
     = bits32(split(perm_key)[1], 1)[0] mod 2**30 = 937906096
       (randint: k1, k2 = split(key); multiplier = (2^16 mod 2^30)^2 mod 2^30 = 0, so only k2's bits count;
        J/random/core.py:709-740)
rng  = numpy.random.Generator(PCG64(937906096))
a    = rng.integers(1, N) repeated until gcd(a, N) == 1      -> 830293
b    = rng.integers(0, N)                                     -> 746237
sigma(r) = (a*r + b) mod N                                     # _prp.py:141
```

`N = 2,441,406`, `a = 830293`, `b = 746237` **[verified]**. Copy the two constants; do not reimplement numpy's bounded
integer sampler. Note: in levanter `a` is `np.int64` and the index array `uint64`, so numpy promotes `a*r + b` to
float64 (`_prp.py:141`); the product stays below 2^53 (max ≈ 2.0e12), so the result is exact and equal to integer
arithmetic.

### 4.3 Stage B: the mixture's within-block shuffle

The data then goes through `MixtureDataset(datasets={"fineweb-edu-10B": …}, weights={"fineweb-edu-10B": 1.0},
block_size=2048, key=mix_key, stop_strategy="restart")` (`datasets.py:781-801`; `mixture_block_size: 2048`,
`stop_strategy: restart` logged). The 16 Paloma components have weight 0 and are dropped
(`datasets.py:729`, `L/data/mixture.py:84-88`). **Even with one dataset, the mixture permutes indices inside every
block of 2048** (`randomize_blocks=True`, `mixture.py:58,210-216`). This step is easy to miss; the launcher docstring
mentions only the linear permutation.

With one dataset of weight 1: counts per block = [2048] (`mixture.py:143-160`), base ids = `0..2047`
(`mixture.py:162-169`; dataset id 0 in the high 16 bits), and for mixture index j (`mixture.py:231-269,218-229`):

```
blk = j // 2048
pi_blk = jax.random.permutation(fold_in(mix_key, blk), arange(2048))     # mixture.py:497-500
r = blk*2048 + pi_blk[j % 2048]
r = r mod N                                                               # restart, mixture.py:283-297
```

`jax.random.permutation` on a 1-D array is `_shuffle` (`J/random/core.py:744-820`): `rounds = ceil(3*ln(2048) /
ln(2^32-1)) = 2`; each round `key, sub = split(key)` (key ← `split(key)[0]`, sub ← `split(key)[1]`), draw
`bits32(sub, 2048)`, and **stable**-sort the current array by those keys ascending (`lax.sort_key_val`, `is_stable`
defaults True, `J/lax/lax.py:3555-3556`). Key i belongs to the element currently at position i. Block 0 starts
`[416, 814, 925, 86, 1936, 560, 614, 1951, …]` **[verified]**.

### 4.4 Closed form

```
s(j) = ( 830293 * ((blk*2048 + pi_blk[j mod 2048]) mod 2441406) + 746237 ) mod 2441406,   blk = j div 2048
```

and the token window of mixture index j is `tokens[4096*s(j) : 4096*s(j)+4096]`. First values:
`s(0..7) = 1909879, 335277, 2165778, 1350661, 1748337, 1843177, 292285, 1995702` **[verified]**.

### 4.5 Epochs and restart

* There is no reshuffle per epoch (`dataset.py:378`: "TODO: add epoch reshuffling"). Wrapping happens on r, after the
  block shuffle and before the LCG: pass 2 reads the same LCG order, but shifted by N mod 2048 = 190 positions relative
  to the block grid and with fresh block keys (blocks ≥ 1192), so its order inside each block differs from pass 1.
* Block 1192 (j = 2,441,216 … 2,443,263) is the first block with wrapped entries (those with `pi < 190` stay in pass 1,
  `pi ≥ 190` wrap) **[verified first wrapped j = 2,441,216]**. Pass 3 starts in block 2384 (j = 4,882,432).
* `max_train_batches`, `num_validation_sequences`, `experiment_budget`, `target_budget`: all null (logged), so no
  slicing or train/val split happens.

| size | B | steps | sequences consumed | passes over N | tokens | first step with repeated data |
|---|---|---|---|---|---|---|
| 130m | 128 | 4959 | 634,752 | 0.260 | 2,599,944,192 | none |
| 300m | 128 | 11444 | 1,464,832 | 0.600 | 5,999,951,872 | none |
| 520m | 256 | 9918 | 2,539,008 | 1.040 | 10,399,776,768 | 9536 (mixed batch), 9537–9917 more |
| 1_2b | 256 | 22888 | 5,859,328 | 2.400 | 23,999,807,488 | 9536 (pass 2), 19072 (pass 3) |

(B, steps: `della_muonh_qwen3_scaling.py:426-431`, logged `trainer.train_batch_size`, `num_train_steps`.)

Because s(j) depends only on j, 130m and 300m read the same stream (300m continues past 130m's end), and the B = 256
batch k is the concatenation of B = 128 batches 2k and 2k+1.

---

## 5. Global batch assembly

* Batch schedule constant: `global_data_offset_by_step(k) = k*B`, `batch_size_at_step(k) = B`
  (`L/schedule.py:103-115`). The loader asks for mixture indices `k*B … k*B+B-1` and keeps that order as batch rows
  (`L/data/loader.py:271-315,421-472`). `iter_from_step(state.step)` starts at batch index = number of completed steps
  (`L/main/train_lm.py:455-460`), so the update from step k to k+1 uses batch k.
* Mesh: one process, 4 GPUs, axes `data: -1 (=4), replica: 1, model: 1` (`della_muonh_qwen3_scaling.py:802-811`);
  the batch axis maps to `data`. Device d holds rows `[d*B/4, (d+1)*B/4)` (`loader.py:160-175,204-216`). B is a
  multiple of 4, so no padding rows (`allow_nondivisible_batch_size=True` is inert, `:814`).
* Microbatching: `microbatch_size = per_device_parallelism * 4` (`L/trainer.py:987-990`). Logged
  `per_device_parallelism`:
  * 130m, 300m: 32, from the default `B // data_axis = 128 // 4` (`L/trainer.py:1139-1141`). One microbatch of 128.
  * 1_2b: 32 in every segment (`PER_DEVICE_PARALLELISM = {"1_2b": 32}` already at 59ac0d647d, launcher line 79).
    Two microbatches of 128.
  * 520m: **64 for steps 0..119, then 32.** The first segment (`run-20260914_063908`, commit 59ac0d647d, config lines
    497-498: `per_device_parallelism: 64`, `per_device_eval_parallelism: 64`) ran at a commit whose launcher has
    `PER_DEVICE_PARALLELISM = {"1_2b": 32}` (`git show 59ac0d647d:experiments/references/della_muonh_qwen3_scaling.py:79`),
    so 520m took the default `256 // 4 = 64`: `microbatch_size = 256`, one microbatch, no gradient accumulation. That
    segment logged `global_step` 2..119; the launcher comment (`della_muonh_qwen3_scaling.py:432-434`) records the OOM
    at step ~115. Commit 28fd599610 ("Run 520m with two microbatches per step on 4x80GB", 2026-09-14 06:54) added
    `"520m": 32`. The next segment (`run-20260914_083457`, job 13852140, started 08:34) loaded the temporary checkpoint
    `step-85` and resumed from `state.step` 86 (`M/logs/slurm/muonh-qwen3-520m_13852140.out:110-111, 207`); its
    replayed steps 86..119 were not logged (W&B refuses steps below 120, `:215`), so its first logged `global_step` is
    120. So updates 0..85 (batches 0..85) used 64 and all later updates used 32; the W&B rows 86..119 come from the
    discarded first-segment trajectory. The 083457 config is not cached; 32 is read from the later segment config `run-20260915_170105` (lines 499-500) and
    from the commit order. The run-level W&B config shows 32 only because later segments overwrote it. No eval ran in
    the 64 window (the first segment has no `eval/` rows), so every logged 520m eval used EvalBatch 128.
  At two microbatches: `unflatten_axis(Batch, (AccumStep=2, Microbatch=128))` (`L/grad_accum.py:170-176`), so
  microbatch 0 = rows 0..127 and microbatch 1 = rows 128..255; loss and gradients are the mean of the two microbatch
  values (`grad_accum.py:153-162`). The token stream of a step does not depend on device count or microbatching; only
  the floating-point reduction order does. The training-loop section must carry the 520m switch at update 86
  (0-based).
* The original TPU runs (64 devices) and these 4-GPU runs map step k to the same mixture indices; the order is
  device-count independent.

---

## 6. Resume

* The data path holds no state. The loader is rebuilt from the config on every start, the LCG constants and block
  keys are recomputed from `data_seed`, and training resumes with `iter_from_step(state.step)`
  (`L/main/train_lm.py:455-460`). The checkpoint's step is the only data position; batch k after a resume is
  bit-identical to batch k in an uninterrupted run.
* Temporary checkpoints every 10 min, keep the last one; permanent every 10,000 steps (logged checkpointer). The 1_2b
  run went through 20+ resume segments (`N/cache/wandb/_tools/baseline_sacct.txt`) with this guarantee.
* A new implementation reproduces this with `next_j = step * B` stored (or recomputed) in its checkpoint, and no
  shuffle buffer or iterator state.

---

## 7. Check: batch hashes

**Definition.** `H(B, k) = sha256( tokens of mixture indices k*B .. k*B+B-1 as int32 little-endian, row-major
[B, 4096] )`. Both levanter (`mix.get_batch`) and the closed form give these values **[verified]**:

| B | step k | sha256 | s(kB) .. s(kB+3) |
|---|---|---|---|
| 128 | 0 | `4626638f4644adcaf66a9084c34ea3d4d423262a6473d03f6b95123f6bc9e449` | 1909879 335277 2165778 1350661 |
| 128 | 1 | `691e9e1ad524af5b1e3a772db5aceadc58bcf9c6e93506eaef508f37e7aaf50c` | 1339913 1849497 1944176 1892489 |
| 128 | 2 | `ffd0b0e8f29bece94a5888c79fba44a12e7bb6dbff34a6a7c4e45e227e2dc64f` | 1055876 1152769 1997916 2146496 |
| 128 | 1000 | `d1d0427ec3b89db162f09b57dd8ecea5c7053117c50c115fc1aea2a9c48104bf` | 1579833 633365 2410126 1061554 |
| 128 | 4958 (130m last) | `f8ba38d570bd33a16887fa5927f6ffe0550d9589b3ed44b9240228da2437bab1` | 1568248 699391 893016 1658660 |
| 128 | 11443 (300m last) | `6bc7e8d141df40d3f5c210d123e9bd81c2f3b45291fec053874c00d058378e2c` | 1329954 370524 1757982 2239911 |
| 256 | 0 | `813ff307c4b074a6ec62e51efa1cc95085a5b7c45dd31841d0fbd46895f1624f` | 1909879 335277 2165778 1350661 |
| 256 | 1 | `0778706834532ed1b912f18c990a7e72c323b26fe7734dc32c37050eb5282c0e` | 1055876 1152769 1997916 2146496 |
| 256 | 4767 | `154a347fbd2ab2800744e25b3ca92d383cb5bb34c30831bb9873484d6e2f8025` | 106570 2399396 2369366 775482 |
| 256 | 9536 (first wrap) | `3fc488330822e8a49e96ba54aa722f50f467a7fd3d4ac622661585c0751188ec` | 1316203 1236539 1292493 1128898 |
| 256 | 9917 (520m last) | `38211b426b26c5c7c1ce1092fe24eb329cc96b53d2a698180dcf969ea57730c4` | 736658 885238 618430 1068115 |
| 256 | 22887 (1_2b last) | `53942bac0515978ee19cdb0c606636d7e722900d4fadbfb74409cf773309a0e7` | 218635 2363203 1661886 1018737 |

Proposed acceptance test for the new loader: (1) the order table sha256 of section 1; (2) all twelve hashes above;
(3) for the same batches, segment ids and loss weights equal the formulas of section 3.1. A loader that passes these
produces the levanter token stream for every step of all four sizes. For a run-time guard, log `H(B, k)` at k = 0 and
at every eval step.

---

## 8. Paloma eval data

### 8.1 Caches and domains

`validation = list(paloma_datasets(tokenizer=marin_tokenizer).values())` (`della_muonh_qwen3_scaling.py:545`): 16
handles `tokenized("paloma/<subset>-marin-tokenizer", tokenizer="marin-community/marin-tokenizer",
version="2026.06.28", paths=["raw/paloma-fc6827/65cd6fc/<dir>/val/val*.jsonl.gz"], validation=True)`
(`M/experiments/datasets/paloma.py:17-57`), i.e. the Paloma **val** split of `allenai/paloma@65cd6fc`, tokenized with
the training tokenizer. Caches at `S/paloma/<subset>-marin-tokenizer/2026.06.28/validation/part-00000-of-00001/input_ids/{data,offsets}`
(sharded layout, one shard; int64; same element layout as section 2.3). Weight 0, so eval only.

Restored on 2026-09-28 like the training cache. All 16 are byte-identical (tokens and offsets, sha256) to the
Llama-3.1-tokenized caches that survived in `S/tokenized/paloma/` (`M/docs/della/sampled-softmax.md:102-105`).

Every document is `[BOS, …, EOS]` in all 16 domains (first/last token counts equal the document counts) **[verified]**.

| # | domain (tag `paloma/<d>-marin-tokenizer`) | docs | tokens | seqs = tokens//4096 | dropped tail | scored positions = seqs·4095 | target bytes | eval index range | eval batches | sha256 of the seqs·4096 tokens (int32 LE) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4chan | 523 | 929,479 | 226 | 3,783 | 925,470 | 3,108,620 | [0, 226) | 0–1 | `418421ca41b128b5a6e15b6d0f620d9a4e4215c0b26a6712acf04b12802caee0` |
| 2 | c4_100_domains | 14,059 | 9,464,109 | 2,310 | 2,349 | 9,459,450 | 43,843,015 | [226, 2536) | 1–19 | `167b9ea6e7dbb4b47d7691ba9931c76a3af1641d5a5d0130d2008326ef9767f4` |
| 3 | c4_en | 2,091 | 967,618 | 236 | 962 | 966,420 | 4,530,682 | [2536, 2772) | 19–21 | `5ef2fc5286e6a1fac552f34e9e48b3b664ddec8e3550dacba889de051d433b69` |
| 4 | dolma-v1_5 | 4,600 | 2,883,425 | 703 | 3,937 | 2,878,785 | 12,406,417 | [2772, 3475) | 21–27 | `ef6c458e93f48c42e725ee9d3f3807ac77aa3c52b2de1c6979179ae8e0e04080` |
| 5 | dolma_100_programing_languages | 8,123 | 8,502,810 | 2,075 | 3,610 | 8,497,125 | 28,826,435 | [3475, 5550) | 27–43 | `8bd793b5b43e0a1319ef70ea2a0567ee73cae27053b9fd6e5e4508f644272a8f` |
| 6 | dolma_100_subreddits | 46,347 | 9,435,238 | 2,303 | 2,150 | 9,430,785 | 41,467,459 | [5550, 7853) | 43–61 | `f0548824ea12008b094e09c55aead5a826843982561cd7d179dbe0dd940a2abc` |
| 7 | falcon-refinedweb | 1,553 | 970,564 | 236 | 3,908 | 966,420 | 4,371,173 | [7853, 8089) | 61–63 | `0f26d6b4aa56892d8194623578dae32260c1b3459e2dddf7867934c52a37b0c9` |
| 8 | gab | 23,249 | 958,643 | 234 | 179 | 958,230 | 3,354,865 | [8089, 8323) | 63–65 | `7ebf2640b9eba8524cc34631b5b4f5e3b67f9dbf21d4eb9e8de73458af00f850` |
| 9 | m2d2_s2orc_unsplit | 167 | 16,844,264 | 4,112 | 1,512 | 16,838,640 | 73,450,318 | [8323, 12435) | 65–97 | `f9e4608406c0bb372a6d93c94b52ec14162f6f22b3b5f344fff2ecff05f55df0` |
| 10 | m2d2_wikipedia_unsplit | 49 | 4,839,217 | 1,181 | 1,841 | 4,836,195 | 21,586,958 | [12435, 13616) | 97–106 | `4d8d5225bc27e7859e0ee980579ff431314803937acb6bfc9a8ae66c38d4ac77` |
| 11 | manosphere_meta_sep | 2,382 | 965,236 | 235 | 2,676 | 962,325 | 4,065,983 | [13616, 13851) | 106–108 | `09b6b8efed1a3847107a353540f51e501eb6b65922f3a130606983249de23f1e` |
| 12 | mc4 | 1,108 | 970,427 | 236 | 3,771 | 966,420 | 4,044,953 | [13851, 14087) | 108–110 | `081f5f67541a337845e3fd9f89b1589112f20eb90871ead87f5049e75030a6e3` |
| 13 | ptb | 1 | 87,135 | 21 | 1,119 | 85,995 | 394,630 | [14087, 14108) | 110 | `ffb45ee0e9a0b5f9f6ed16c51175cf6f7a7ca89756128727a704233dd7b374e7` |
| 14 | redpajama | 632 | 641,116 | 156 | 2,140 | 638,820 | 2,685,571 | [14108, 14264) | 110–111 | `8bc20349c6e0191412591b9f940c5b49d35f723bb70b8c0eac4e73ec99696e7e` |
| 15 | twitterAAE_HELM_fixed | 50,000 | 841,005 | 205 | 1,325 | 839,475 | 2,473,231 | [14264, 14469) | 111–113 | `9d110d797b7a233ab5b82fdcdd3c7202f4f5bb93ffd14b5701b28774ccf32d93` |
| 16 | wikitext_103 | 59 | 249,808 | 60 | 4,048 | 245,700 | 1,114,668 | [14469, 14529) | 113 | `a3cadf50589690a770299069a1a7f9f81f0bd532c180713f276f20e6dc91002a` |

Total 14,529 sequences, 114 eval batches of 128 **[verified]**. "Target bytes" = sum of `bytes_per_token[tokens[i+1]]`
over the scored positions i = 0..4094 of every sequence (section 8.4).

### 8.2 How eval examples are formed

* Same path as training, without any shuffle: `validation_sets` → `CausalLmDataset(TokenSeqDataset(cache, 4096))`
  per domain (`L/data/text/datasets.py:873-891,509-534`). Sequence i of a domain = its tokens `[4096 i, 4096 i + 4096)`;
  documents are concatenated in cache order, windows cross documents, the tail is dropped (column "dropped tail").
* Loss weights, segment ids from EOS, positions 0..4095: identical to section 3.1. Each sequence scores 4095
  positions.
* `tagged_eval_sets` gives each domain the single tag = its component name (`datasets.py:1064-1070`; component tags
  are `[]`). The order is the component order of section 8.1.
* `DomainTaggedDataset` concatenates the 16 domains in that order into one index space of 14,529 sequences
  (`L/eval.py:82-174`); the evaluator does not shuffle (`shuffle=False`, `eval.py:520-529`). The `DataLoader` reads
  batches of `EvalBatch = per_device_eval_parallelism · data_axis = 32 · 4 = 128` (logged 32 for all sizes, except 64 in the first 520m segment, which ran no eval;
  `L/trainer.py:1048-1049,1168-1177`) in index order, so one batch can hold two domains (column "eval batches").
  `per_device_eval_parallelism` defaults to `per_device_parallelism` (`L/trainer.py:1168-1177`), so EvalBatch depends
  on the GPU count and the microbatch setting: on 8 GPUs, 130m/300m keep 128 (16 · 8), but 520m/1_2b get 256 (32 · 8)
  unless the launcher's `PDP` knob changes it (`della_muonh_qwen3_scaling.py:103-104,787`); the first 520m segment
  had 64 · 4 = 256 but ran no eval. Because the per-domain bpb depends on the batching (trap 2 of section 8.3), a
  reimplementation must form eval batches of exactly 128 rows in this order whatever its own device count, and must
  not derive the eval batch size from its GPU count.
* The last batch (113) holds 65 real rows and 63 padding rows. Padding rows are all zeros (tokens 0, loss weight 0,
  tag vector 0) (`L/data/loader.py:349-372,646-648`), so they contribute nothing.
* `max_eval_batches: null` → the full set every time (`L/main/train_lm.py:350-352`).
* Cadence (off by one; read carefully). The hook test is `info.step > 1 and info.step % hook.every == 0`
  (`L/trainer.py:153`), `every = steps_per_eval = 1000` (`L/main/train_lm.py:372`; `della_muonh_qwen3_scaling.py:445`).
  `StepInfo` is built from `result.new_state` after the update (`L/trainer.py:557`), and `StepInfo.step =
  int(state.step) - 1` (`L/callbacks/_core.py:76`; "0-indexed"). `TrainerState` sets `step = self.step + 1` on each
  update (`L/trainer_state.py:152`). So the hook fires at `state.step` = 1001, 2001, 3001, …, and the eval callback logs
  at `step=step.step` = 1000, 2000, … (`L/eval.py:340,349`). **The W&B eval point at logged step S (S = 1000, 2000, …)
  evaluates the weights after S + 1 updates (batches 0..S consumed).** The forced final eval (`L/trainer.py:626`)
  evaluates the weights after all `num_train_steps` updates and logs at `num_train_steps - 1` = 4958 (130m), 11443
  (300m), 9917 (520m), 22887 (1_2b). The cached 130m history
  (`N/cache/wandb/targets/della__muonh-qwen3-130m-della4xh100.hist.json.gz`) has eval rows at `_step` 1000, 2000,
  3000, 4000 and 4958. A reimplementation must evaluate after updates 1001, 2001, … and tag those evals 1000, 2000, …;
  evaluating after S updates is one update early at every intermediate point. The final targets are unaffected.
* Weights and dtype: the current (non-EMA) weights, cast to compute dtype bf16 (`eval.py:225-239`,
  `mp p=f32,c=bfloat16`). The baseline logs `model_averaging: null`, and the `eval/ema/*` branch runs only when
  `step.state.model_averaging is not None` (`L/eval.py:357`). The `-ema<beta>` variant runs set
  `model_averaging=EmaModelAveragingConfig(beta=EMA_BETA)` (`della_muonh_qwen3_scaling.py:813`) and log extra
  `eval/ema/...` keys; those are not baseline targets.

### 8.3 Metric arithmetic (as logged)

Per batch b (`L/eval.py:546-594`), with per-position loss `l` (natural log), weight `w`, one-hot tag `t`:

```
L_b   = sum(l*w)                 W_b = sum(w)
L_b,d = sum over rows of tag d of l*w      W_b,d = same of w
Y_b,d = sum over rows of tag d of bytes_per_token[next token]*w
token_avg_loss  <- RunningMean.add(L_b / max(W_b,1), W_b)
loss_per_tag[d] <- RunningMean.add(L_b,d / W_b,d (0 if W_b,d = 0), W_b,d)
bpb_per_tag[d]  <- RunningMean.add(L_b,d / max(Y_b,d, 1) * log2(e), W_b,d)     # weighted by TOKENS
bpb             <- RunningMean.add(L_b / max(Y_b,1) * log2(e), W_b)
RunningMean.add(x, n): new_total = total + n; mean += (x - mean) * (n / new_total if new_total else 0)   # L/utils/stat_utils.py:24-30
```

Logged keys (`eval.py:444-485,596-660`), prefix `eval`:

All running means and totals are float32 (`RunningMean.zeros_like` of float32 zeros, `eval.py:597-600,837-841`), and
`W_d = loss_per_tag[d].total = bpb_per_tag[d].total` is a TOKEN count: both are accumulated with
`this_weights_per_tag` (`eval.py:564,573,589`). The name `total_bytes_per_tag_cpu` (`eval.py:629`) is misleading. NumPy
keeps float32 for float32 input: `np.mean` promotes only int/bool to float64 (`numpy/_core/_methods.py:127-129`), and
`np.average` uses `result_type(a, w)` = float32 (`numpy/lib/_function_base_impl.py:577-588`).

| key | code | value |
|---|---|---|
| `eval/paloma/<d>-marin-tokenizer/loss` | `float(loss_per_tag.mean[d])` (`eval.py:644`) | Σ_b W_b,d·(L_b,d/W_b,d) / Σ_b W_b,d = Σ_b L_b,d / Σ_b W_b,d up to float32 rounding |
| `eval/paloma/<d>-marin-tokenizer/bpb` | `float(bpb_per_tag.mean[d])` (`eval.py:646`) | Σ_b W_b,d·(L_b,d/max(Y_b,d,1))·log2 e / Σ_b W_b,d |
| `eval/loss` | `token_avg_loss.mean` (`eval.py:609`, logged `:448`), float32 on device | Σ_b W_b·(L_b/W_b) / Σ_b W_b |
| `eval/paloma/micro_loss` | `np.average(loss_per_tag.mean, weights=W·mask)` (`eval.py:637`), float32 | Σ_d W_d·loss_d / Σ_d W_d; equals `eval/loss` in exact arithmetic (130m step 1000: 4.7428202629089355 vs 4.742820739746094) |
| `eval/macro_loss` | `jnp.mean(loss_per_tag.mean)` (`eval.py:611`), float32, XLA reduction | mean of the 16 per-domain losses |
| `eval/paloma/macro_loss` | `np.mean(loss_per_tag.mean, where=mask)` (`eval.py:636`), mask = all 16 tags | float32 accumulator (NumPy float32 `add.reduce`, pairwise on contiguous runs; the exact masked order was not checked), then ÷ 16 (exact). Equal to `eval/macro_loss` up to ~1 float32 ulp (130m step 1000: both 4.8956499099731445, a float32 value) |
| `eval/bpb` | `bpb.mean` (`eval.py:614`, logged `:475`), float32 | Σ_b W_b·(L_b/max(Y_b,1))·log2 e / Σ_b W_b: per-batch bpb over all rows of the batch |
| `eval/paloma/bpb` | `np.average(bpb_per_tag.mean, weights=W·mask)` (`eval.py:641`, logged `:478-479`), float32 | Σ_d W_d·bpb_d / Σ_d W_d = Σ_b Σ_d W_b,d·(L_b,d/max(Y_b,d,1))·log2 e / Σ W. Differs from `eval/bpb` in exact arithmetic, because the ratio is per (batch, domain), not per batch (130m step 1000: 1.6721341609954834 vs 1.6692712306976318) |
| `eval/macro_bpb` | `jnp.mean(bpb_per_tag.mean)` (`eval.py:616`), float32 | mean of the 16 per-domain bpb |
| `eval/paloma/macro_bpb` | `np.mean(bpb_per_tag.mean, where=mask)` (`eval.py:640`), float32 | as `eval/paloma/macro_loss`; differs from `eval/macro_bpb` by one float32 ulp at 130m step 3000 (1.6098203659057617 vs 1.6098204851150513) |

Two traps:

1. **Key names.** On these della runs the per-domain keys carry the `-marin-tokenizer` suffix: the stage-1 target
   "eval/paloma/c4_en/bpb" is the W&B key `eval/paloma/c4_en-marin-tokenizer/bpb`. The 16-domain macro key has no
   suffix: `eval/paloma/macro_loss`.
2. **bpb is not total bits / total bytes.** It is a token-weighted mean of per-batch bpb values. It equals
   `Σ L / Σ Y · log2 e` only when a domain sits in one batch or its bytes per token are equal across its batches.
   c4_en spans batches 19–21 (24, 128 and 84 rows), so the logged value depends on the batch size 128 and on the
   concatenation order of section 8.1. To match logged numbers, use this exact batching and formula; also report the
   exact `Σ L / Σ Y · log2 e` beside it. (The vocabulary session notes that the W&B macro bpb reads about 0.004 high
   against an offline exact computation, `N/docs/survey/needs_from_sessions.md`, item 2; `eval.md` §4.3 measures the
   opposite sign and a smaller size: logged macro_bpb is 0.0004-0.0006 below the exact value. Open point.)

### 8.4 Bytes per token

`bytes_per_token[i] = byte_length_of_token(tokenizer, i)` for i in 0..127,255 (`L/eval.py:182-192`,
`L/utils/hf_utils.py:24-47`): 0 for the 256 special ids (128000–128255, including BOS and EOS); for byte-fallback
tokens `<0xNN>` the hex length; otherwise `len(decode([id('.'), i]).utf8) - 1`. Table saved as
`N/cache/data/bytes_per_token_marin.i32` (128256 int32 LE, sha256
`d3be74509077b5ba84a2822f0348c7574b685404178786b7e0d473c64a334e4d`) **[verified]**. Bytes are counted for the
**target** token (`per_pos_token_id = roll(tokens, -1)`, `eval.py:237`), with the same weights as the loss, so the BOS
target after an EOS adds 0 bytes but its loss counts.

**Do not recompute this table from the vocabulary.** The rule decodes through `tokenizer.decode`
(`L/utils/hf_utils.py:46-47`, `L/tokenizers.py:638-639`), i.e. the HF tokenizers ByteLevel decoder, which turns the
decoded bytes into a string with lossy UTF-8 conversion. A byte-level BPE token that holds an incomplete UTF-8 sequence
therefore counts the bytes of the U+FFFD replacement character(s) (3 bytes each), not its true 1–3 raw bytes. A loader
that sums raw byte lengths from the BPE byte map gets a different table for those ids. Load
`bytes_per_token_marin.i32` as is and check its sha256. (Mechanism from the decoder; the number of affected ids was
not counted here.)

---

## 9. Ambiguities and open points

1. **Cache identity across the 09-24 deletion** (section 2.1). Counts match and the 130m pool reproduced after the
   restore; no token hash of the 09-14 copy exists. All hashes here pin the current copy.
2. **Segments from EOS vs from offsets** (section 3.1). Levanter uses the EOS token value. A loader that builds
   segments from document offsets differs on the rare documents with an EOS id inside text. Reading 1 (match
   levanter): segment boundary after every 128001. Reading 2 (cleaner): boundaries at document offsets. Stage 1 must
   use reading 1.
3. **bpb aggregation** (section 8.3). Reading 1 (match W&B): token-weighted mean of per-batch bpb with EvalBatch 128 and
   the order of section 8.1. Reading 2 (exact): Σ bits / Σ bytes. Compare to the logged pool with reading 1.
4. **Original TPU runs.** The launcher reproduces Kaiyue Wen's `qwen3_<size>_muonh_4096_*` runs (2025-10-12, 64 TPU
   devices; docstring `della_muonh_qwen3_scaling.py:3-11`). Whether the 2025 levanter mixture also shuffled inside
   blocks was not checked; it does not matter for stage 1, whose target is the della runs.
5. **Microbatch reduction order** (section 5) is a numerics question for the training-loop section; the token stream
   is fixed by this section. That section must carry the 520m schedule: one microbatch of 256 for updates 0..85
   (batches 0..85), two microbatches of 128 from update 86 on (`optim.md` §8).

---

## Review log

Two skeptic reviews, checked against the code and cached W&B data on 2026-10-24.

| # | issue | verdict | reason |
|---|---|---|---|
| 1 | 520m `per_device_parallelism` 64 in the first segment; 520m spans two commits | applied (sections "Code version", 5, 9.5) | Confirmed: config `run-20260914_063908` lines 497-498 log 64/64 and commit 59ac0d647d; its launcher line 79 is `{"1_2b": 32}`; `"520m": 32` arrived in 28fd599610 (06:54); that segment logged `global_step` 2..119 and no eval; the next segment starts at 120. The 32 of segment 083457 is inferred from commit order and the later config, since its own config is not cached. |
| 2 | Eval cadence off by one (reviewer A) | applied (8.2) | Confirmed: `trainer.py:153`, `_core.py:76`, `trainer.py:557`, `trainer_state.py:152`, `eval.py:340,349`; 130m history evals at 1000..4000 and 4958. |
| 3 | `eval/paloma/macro_loss` is float32, not float64 | applied (8.3) | Confirmed: `np.mean` keeps float32 for float32 input (`_methods.py:127-129`). Logged values sit on the float32 grid, and `eval/paloma/macro_bpb` and `eval/macro_bpb` differ by one float32 ulp at step 3000. The exact NumPy summation order under `where=` was not checked, so the spec does not state it. |
| 4 | Positions citation points to MultiHeadLatentAttention | applied (3.1) | Confirmed: `attention.py:2104-2105` is in the class at `:1963`; Qwen3 uses `Attention` (`qwen.py:146,155`), default at `:1441-1442`; no `pos_ids` at `lm_model.py:337`. |
| 5 | `eval/paloma/bpb` missing; differs from `eval/bpb` (reviewer A) | applied (8.3) | Confirmed: `eval.py:641`, logged at `:478-479`; `bpb_per_tag.total` is a token count (`eval.py:589`). |
| 6 | Independent numpy-only re-derivation of all data-order constants | no change | It confirms the spec; nothing to fix. |
| 7 | Eval cadence off by one (reviewer B) | applied (8.2) | Duplicate of 2; same evidence. |
| 8 | `eval/paloma/bpb` missing (reviewer B) | applied, with one fix | Duplicate of 5. Its "float64 over float32 per-tag means" is wrong: `np.average` uses `result_type(float32, float32)` = float32 (`_function_base_impl.py:577-588`); the spec says float32. |
| 9 | bytes-per-token U+FFFD trap | applied (8.4) | The rule decodes through `tokenizer.decode` (`hf_utils.py:46-47`, `tokenizers.py:638-639`); the ByteLevel decoder is lossy on invalid UTF-8. Added as a warning, with the count of affected ids marked unverified. |
| 10 | Baseline evaluates only non-EMA weights | applied (8.2) | Confirmed: `model_averaging: null` logged; `eval.py:357`; variant knob at `della_muonh_qwen3_scaling.py:813`. |
| 11 | EvalBatch depends on device count; 8xH100 gives 256 | applied, narrowed (8.2) | `eval_batch_size = per_device_eval_parallelism · data_axis` (`trainer.py:1048-1049`), and it defaults to `per_device_parallelism` (`:1168-1177`). On 8 GPUs, 130m/300m still get 16 · 8 = 128; only 520m/1_2b get 32 · 8 = 256 (unless `PDP` is set). The configs of the 8xH100 runs were not read. The requirement to emulate 128-row batching was added. |

Merge cross-check (2026-10-24): values stated in two or more sections were compared; the corrections are listed in `kaiyue_baseline.md` §C.
