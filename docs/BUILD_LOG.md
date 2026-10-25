# marincpp build log

One section per stage of DESIGN.md §13.3: what was built, how to test it, results, known gaps.

Build and test on a vis node (never on a login node):

```bash
ssh della-vis1
cd $ROOT/project/marin.cpp && module load gcc-toolset/14 cudatoolkit/12.8
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++
nice -n 19 cmake --build build -j 8 && (cd build && nice -n 19 ctest --output-on-failure)
```

## S0: skeleton, config, memory planner (2026-10-24)

Built
- CMake (3.30+, CUDA 12.8, gcc 14, C++20). `nm_module()` (cmake/MarinModule.cmake) makes one OBJECT library per
  module: PORTABLE sources build for `NM_ARCHS_PORTABLE` (sm_90a + sm_80), SM90A sources for sm_90a only, FMAD_OFF
  sources get `--fmad=false`. Host flags `-Wall -Werror -ffp-contract=off`; nvcc `--Werror=all-warnings`;
  `__FILE__` is repo-relative. Targets: `libmarincpp.a`, `libmarincpp_kernels.so` (C ABI), `marincpp-train`,
  `nm_tests_s0`. Final targets link the module objects directly, so static registrations are kept.
- Stub `src/<module>/CMakeLists.txt` for data, sched, kernels/{elementwise,gemm,attention,cross_entropy}, optim,
  model, comm, eval, metrics, checkpoint, train. `tests/*/CMakeLists.txt` is picked up automatically.
- src/core: `nm::Fatal` (catchable `fatal()`), CUDA/NCCL/cuBLAS status checks, SHA-256, registries (baseline names
  sandwich/embed/full registered with null factories; `register_*` and `make_block`), C ABI basics
  (`nm_version`, `nm_last_error`, `nm_device_arch`, `capi_call` helper in core/capi_util.hpp), device probes.
- src/config: knob table from knobs.def, TOML subset reader and writer, presets (130m, 300m, 520m, 1_2b, pls_d48,
  custom, levanter_strict), `load_config`, `resolve` (plan choices, content hashes of file inputs, dataset meta
  hash, `exec.ce_logits` auto), `validate` (every knob check plus the cross-knob rules, all problems in one message),
  canonical JSON, config hash, run id, labels, `ModelShape`, `ExecShape`, `OptimConstants`.
- src/memory: `MemoryPlan` (256-byte layout, canary bands, groups), `build_plan` (every buffer of DESIGN.md §2.5, by
  name, dtype and shape), `choose_plan` (micro_seqs / remat rule of §2.3, CE chunk rule of §3.8), `step_budget`
  (port of design_budget.py perf()), `Arena` (one cudaMalloc, bind by name with dtype/shape checks, canary check,
  NaN poison).
- `marincpp-train plan | config | knobs` (plan prints the grouped plan and the step budget, `--json` for tools;
  exit 1 if nothing fits); `train` exits 1 until S7/S10.

Gates (ctest, della-vis1 A100, all pass; 0 compiler warnings from a clean build directory)

| ctest | checks | result |
|---|---|---|
| t0_6_constants | T0.6: f32 bits of every optimizer/schedule constant vs the hex column of optim.md §5.3, parsed from the spec | 17 rows, 67 size x constant comparisons equal |
| t0_8_params_flops | T0.8: n_params and flops_per_example vs the spec table and the W&B summaries (cache/wandb/api) | exact for 130m, 300m, 520m, 1_2b; pls d48 3,400,267,776 |
| s0_plan_vs_budget | planner vs scripts/design_budget.py for all 8 plans (CHOSEN on auto + STRICT as `--preset levanter_strict`): choices, every buffer group, P/A/T/W, totals, f32-logits variant; budget rows bitwise in double; DESIGN.md §2.5 and §12.2 numbers | 8 cases, 0 failures (Python 3.9 and 3.12) |
| s0_core, s0_config, s0_memory | SHA-256 vectors, bf16 RNE, registries, TOML, load order, 19 validation rules, identity (pinned resolved 130m canonical JSON), labels, file hashing, knob-defaults guard (tests/s0/knob_defaults_v1.json), layout, planner choices | pass |
| s0_arena, s0_device (GPU) | bind-by-name errors, canary catches a 1-byte overrun, poison fills transients only; portable kernel runs as sm_80, sm_90a-only kernel has no image on the A100 | pass; compute-sanitizer memcheck clean (the 2 reports are the intended no-image launch) |
| (build) nm_s0_headers | every public header compiles on its own | pass |

Known gaps and decisions for later stages
- `.gitignore` had `core.*`, which ignored `include/marincpp/core.hpp` (never committed in 42efe94) and would ignore
  `src/core/core.cpp`. Narrowed to `core.[0-9]*`; core.hpp must be added in the S0 commit.
- Owner shards in the plan are the even split ceil(n / ranks); S6 `plan_ownership` replaces them with the largest
  rank's LPT share. `max_segs_per_seq` is the bound 24 until S1 reads the dataset meta.
- Strict plans: fixed after review (section below): the gate now runs the preset, DESIGN.md §2.5 shows its plans.
- `step_budget` sums its rows like Python 3.12 `sum()` (Neumaier); older Pythons add plainly and differ by ulps.
- Knobs decided by H100 measurements stay `auto` (exec.ce, exec.attn_bwd_sched, optim.ns_precision; budget assumes
  B1 / TF32). `exec.ce_logits=auto` resolves to bf16 outside parity runs until the T1.0 facts file exists.
- `optim.ns_coeffs` other than `simple` is rejected by validate() and optim_constants(); S6 extends both.
- Binary name is `marincpp-train` (DESIGN.md says `marincpp`); `plan --create`, `order`, `eval`, `profile` belong
  to the train/data/eval stages.
- No pytest in the nanomarin env: Python gates are plain scripts run by ctest (`NM_PYTHON`).

## S0 review fixes (2026-10-24)

Built (fixes of the S0 review; every item verified on della-vis1 before and after)
- Arena poison filled saved activations under remat=all (filter `recompute != Never` matched act.L*.{a, qk_rot, m, h},
  24 buffers, 5.62 GiB at 130m). Now `poison_target()`: Transient and Workspace only (the recompute slots are
  Transient). Arena owns its allocations (destructor, no copy, move), `release()` resets every member, and canary mode
  puts a 4 KiB band before AND after every buffer (an underrun into alignment padding is now caught).
- Planner: a microbatch below 8,192 tokens takes one CE chunk of its tokens rounded up to 128 (`seq_len=2000, B=16`
  resolves to 8,064 and passes validate()); the fit test uses an explicit `exec.ce_chunk_rows` (300m with 65,536 rows
  now chooses micro 16 and fits); fit is decided on `device_bytes()` everywhere (choose_plan, `plan`, trainer), so a
  canary-mode plan between payload and arena is refused at choose time. CE head-gradient RMW counts
  n_micro x ceil(micro tokens / chunk) chunks (C++ and design_budget.py). Buffer sizes saturate at 1 PiB.
- Step budget: integer products in __int128 (were int64: `6 T D V heads` overflowed, ce_gemm < 0, MFU 456%). Knob ranges
  added (hidden <= 65536, heads/kv_heads <= 1024, mlp <= 2^20, global_batch and eval.batch <= 2^20) so every product
  stays exact; defaults unchanged (kKnobTableVersion stays 1).
- Strict plans: design_budget.py STRICT models `--preset levanter_strict` (CE chunk 8,192 + mb.ce_dw_accum); DESIGN.md
  §2.5 rows are 24.53 GiB (520m) and 44.76 GiB (1.2B), §12.2 strict step 1950.3 / 3762.8 ms; the gate runs the preset.
  ZeRO-1 shards in the script are ceil(n / 4), as in C++.
- Config: TOML reader follows TOML 1.0 inside the subset (number grammar: `0_1_2`, `1e_2`, `012` rejected; all string
  escapes incl. u/U code points; no raw control characters; redefined tables rejected); json_quote escapes DEL, so
  `load_config(full_toml(c)) == c` holds for every string. `--set model.size="300m"` works; size names in `--preset`
  are rejected. File inputs (NM_CONTENT knobs, incl. data.dataset and eval.paloma) enter the identity by content hash
  only (`./x` and an absolute path give one run id); `resolve()` takes the eval meta. config.hpp and DESIGN.md §9.2
  state that resume and fork reuse the stored resolved config and never re-resolve. validate(): tie_embeddings with
  separate readouts and `optim.zero2_vocab=on` are refused. Label tags are checked against the knob type at compile
  time and are never printf formats. DESIGN.md §9.1: readout.kind is a fixed list, not a registry.
- Errors and build: main catches std::exception (a filesystem_error is exit 1, not SIGABRT); the C ABI returns NM_OK /
  NM_ERR_INVALID / NM_ERR_DEVICE / NM_ERR_INTERNAL (nm::DeviceError for CUDA/NCCL/cuBLAS); nm_device_arch sets
  nm_last_error and clears the CUDA error; libmarincpp_kernels.so exports only nm_* (cmake/capi.map). NM_GIT_SHA is
  generated at build time and counts untracked files under src/, include/, cmake/. CMAKE_CUDA_ARCHITECTURES defaults to
  NM_ARCHS_PORTABLE; capi.h also compiles as C11 (-Wpedantic). Without Python, s0_plan_vs_budget is a failing test.
  The nmtest runner reports fail-then-skip as FAIL (ctest s0_runner_selftest). tools/scrub.py rejects unknown options
  and prints usage for -h/--help (it used to run in write mode).

Tests (della-vis1, clean configure + build: 0 warnings; ctest 9/9 pass; compute-sanitizer memcheck on s0_arena: 0 errors)
- New cases: s0_arena.{ownership, canary_detects_underrun}, poison test with Microbatch and Activation (Elementwise)
  buffers; s0_memory.{poison_never_touches_saved_tensors, ce_chunk_for_any_seq_len,
  fit_uses_the_requested_ce_chunk_and_the_arena, step_budget_has_no_integer_overflow}; s0_core.capi_status_codes;
  s0_config TOML/size/identity/rule cases; s0_runner_selftest.
- Reviewer commands rerun: `config --set model.seq_len=2000 --set data.global_batch=16` -> chunk 8064, exit 0;
  `plan --set model.seq_len=1000 --set data.global_batch=4` -> chunk 1024; `plan --size 300m --set
  exec.ce_chunk_rows=65536` -> micro 16, fits; the overflow plan prints ce_gemm 17101845.0 ms and MFU 60.0%; an
  unreadable fixture subdirectory -> exit 1 with the path.

Known gaps
- ZeRO-2 for vocab-sized gradients is not implemented: auto resolves to off, `on` is refused, build_plan and
  step_budget have no ZeRO-2 terms (the stage that adds the per-microbatch reduce-scatter adds both).
- Python's `int / int` in the comm and eval terms is correctly rounded; C++ divides doubles. They agree while the
  numerators stay below 2^53 (every realistic plan; the knob ranges allow larger only for absurd shapes).
- Outside S0 (controller): bench/e1_attention/{bench_jax.py, bench_torch.py, make_segments.py, run.sh,
  segments/segments_meta.json} still carry the literal `$ROOT` the reviewer's scrub.py run wrote (run.sh is broken);
  `scrub.py --check` also flags tools/devq/wq_runner.sh.

## S2: schedule tables (2026-10-24)

Built
- Fixture generator `tests/s2/schedule_fixture.py` (the `tools/levref/schedules.py` of DESIGN.md §4.4; it lives in
  tests/s2 because tools/levref belongs to another step). It builds the plain-baseline config with
  `tools.levref.config.build`, calls levanter's own `opt.lr_scheduler(N)` and `opt.lr_scheduler(N, override_lr=adam_lr)`
  (MuonHConfig.build), and evaluates them under `jax.jit` on the GPU with a traced int32 k, once vectorized and once
  through `lax.map`; the two must agree bitwise. Bias correction: `jax.jit(inline=True)` `1 - decay**count`, as
  optax's tree_bias_correction. Output `cache/levref/<size>/schedules/` = schedule.toml (levanter fields + sha256 per
  table; no host, time or version, so the directory hash is the identity of the tables) + sched_main, sched_adam
  (k = 0..N-1), bc1, bc2 (t = 1..N) as little-endian f32; provenance in `schedules_manifest/`. `--probes` writes 10
  synthetic configs to `cache/levref/s2_probe/` (min_lr_ratio > 0, warmup + linear, stable + cosine, decay as fraction,
  steps or 0, warmup over all N, other T_d and betas).
- `src/sched`: `load_schedules` (fixture of record: every [levanter] field checked against the config, table sizes,
  per-file sha256, the directory content hash against the one config resolution recorded; source
  `fixture:<sha256>`; the formula runs beside it and fills n_ulp_diffs / max_ulp_diff, never overriding a value;
  without a fixture it refuses run.parity and returns the formula), `build_schedules` (levanter's one-cycle shape:
  `_convert_frac_or_steps`, decay -1 = None, join boundaries), C ABI `nm_schedule_tables` (config_toml = path of a
  TOML config, NULL or "" = defaults; n must equal optim.num_steps).
- Finding: XLA:GPU does not compute optim.md §6 literally. Its algebraic simplifier rewrites `c / T` as
  `c * f32(1/T)` and folds `(pi * c) / T` into `c * f32(f32(pi) * f32(1/T))`; cos and pow are libdevice
  `__nv_cosf` / `__nv_powf` inlined into the fusion (final LLVM IR from `--xla_gpu_dump_llvmir`; the optimized HLO is
  stored since the review fixes below). The literal f32 formula (IEEE division) differs from the fixture's main table
  at 261 (130m), 3030 (300m), 3120 (520m) and 5584 (1_2b) of all k with math.cos (the f32 argument's double cos,
  rounded to f32), at 261 / 2786 / 2990 / 5103 with numpy's float32 cos, and from the logged values at 261 / 3028 /
  3118 / 5569 logged steps (math.cos); the first difference is at k = 58, inside the warmup, where no cos is involved.
  `build_schedules` therefore ports that arithmetic (src/sched/xla_math.cpp: `__nv_cosf` and `__nv_powf` op by op
  with std::fma, plain ops unfused as the IR shows; powf's `rcp.approx.ftz.f32` emulated as the correctly rounded
  reciprocal), and equals the fixture at every k.

Gates (della-vis1 A100, clean configure + build in a fresh directory: 0 warnings; ctest 14/14 pass, of them 5 new)

| ctest | checks | result |
|---|---|---|
| t0_7_schedules | T0.7: tables loaded through libmarincpp_kernels.so vs every logged `optim/learning_rate`, `optim/adam_lr` (W&B glob of the spec, dedup by `_step`, duplicates must agree, logged values must be exact f32); logged-step counts vs optim.md §6.2; holes and k = 0, 1 vs the formula and inside their logged neighbours; 130m k = 4958, 1_2b k = 22887; loader cross-check n_ulp_diffs = 0 | 130m 9914, 300m 22870, 520m 19800, 1_2b 45644 logged values: 0 differ; holes 2 / 9 / 18 / 66: fixture == formula at all; 130m k = 4958 5.041361e-06 / 2.016544e-06; 1_2b k = 22887 0 / 0. Passes under Python 3.12 and 3.9. A one-ulp change in one fixture entry fails it (2 failures) |
| s2_levanter | formula == fixture, all four tables, every k, four sizes | 0 differences (incl. bc1, bc2) |
| s2_probe | formula == fixture for the 10 probe configs | 0 differences in all 40 tables |
| s2_formula | spec values of optim.md §6, shape rules (warmup = 1 refused, decay fraction/steps/0/clip), xla_cosf and xla_powf within 1 ulp of correctly rounded, ulp distance, C ABI | pass |
| s2_fixture | synthetic fixtures: load, a 1-ulp edit counted not overridden, stale sha256, wrong config (lr, beta2, schedule, N), short file, path in [files], format, missing file, parity without fixture, content hash == resolve() and a change after resolution | pass |

Commands
```bash
# fixtures (vis-node GPU, marin venv; ~35 s for the sizes, ~30 s for the probes)
NANOMARIN_ROOT=$ROOT $ROOT/project/marin/.venv/bin/python tests/s2/schedule_fixture.py
NANOMARIN_ROOT=$ROOT $ROOT/project/marin/.venv/bin/python tests/s2/schedule_fixture.py --probes
# gates
ctest --test-dir build -R 's2_|t0_7' --output-on-failure
```

Known gaps
- The fixtures come from the vis node's A100 (sm_80). They equal every H100 logged value, so XLA:GPU's schedule code
  is the same on sm_80 and sm_90 for eta; bc1/bc2 have no logged values, and `rcp.approx.ftz` (MUFU) is the one
  hardware-approximate step: an H100 run of the generator through tools/devq would confirm the bc tables (expected
  identical).
- The rcp emulation: superseded by the measured rate in the review fixes below. xla_cosf covers |x| < 105615 only
  (the schedules reach [0, pi]); xla_powf 0 <= b < 1, integer t < 2^24 (b = 0 since the fixes).
- Spec and design text to update by their owners: the list in the review fixes below.
- Device tables (`DeviceSchedules`, the k-indexed lookup) and the `schedule_source` event belong to S7/S10.

## S2 review fixes (2026-10-24)

Built (every item reproduced on della-vis1 before the change)
- UB 1, a silent wrong schedule: `frac_or_steps` cast `optim.decay` >= 2^63 (a valid knob; levanter accepts it and
  clips) to int64. On x86 that gave INT64_MIN: decay = 1e19 or 1e300 became decay 0, a constant peak (UBSan at
  schedule.cpp:27; N = 100 cosine: lr[99] = 0.01, not 3.0458e-06). A step count >= N now returns N before the cast
  (both callers clip to N - warmup); inf and nan are refused, as levanter refuses them.
- UB 2: `xla_powf` converted round(t log2 b) to int32 before its |ph| > 152 select, which overflows for a subnormal
  beta and a large t (beta1 = 1e-45, N = 1.5e7; UBSan at xla_math.cpp:90). The select now returns 0 / inf before the
  conversion. Results are unchanged: the select used to overwrite them.
- The cross-check no longer blocks the record: when the formula refuses a config, `load_schedules` keeps the fixture
  and sets `n_ulp_diffs = max_ulp_diff = -1` (also from `nm_schedule_tables`). The formula refuses `optim.warmup = 1`,
  `optim.num_steps >= 2^24`, and a beta whose f32 is 1 (1 - b^t = 0). `xla_powf` now takes b = 0 (+-0 for odd t, +0
  for even t, 1 at t = 0), so beta = 0 gives 1 - 0^t = 1, as JAX does (GPU probe `beta0`).
- Probe gap: `cos_fold_1000` and `cos_fold_4000` have a T_d where the folded cosine constants differ, A =
  f32(f32(pi) * f32(1/T)) (XLA's and the port's) and B = f32(f32(pi) / f32(T)). No earlier fixture told them apart.
  B mismatches them at 1126 and 4274 entries, A at 0 (both 0 at the other 9 cosine fixtures).
  `s2_formula.folded_cosine_constant` pins 4 GPU values where B gives the next float, so the check runs without the
  cache. New also: `decay_huge` (decay = 1e19 through levanter's own conversion) and `beta0`. s2_probe lists all 14
  names and fails on a missing one.
- Evidence of the XLA rewrites: every `*_manifest/` now holds `vec.opt_hlo.txt` (`vec.lower(k, t).compile()`). 300m:
  no divide; 2 power ops, the bias corrections (the two `**1.0` are folded); constants 0.001 (warmup) and 0.0003008036
  (f32(pi) * f32(1/10444)). 130m: 0.000252079655 = f32(1/3967). cos_fold_1000: 0.00314159296 = 0x3B4DE32F (form A).
- Generator: an `XLA_FLAGS` or `NCCL_DEBUG` in the shell that differs from the run's (or that the run did not have)
  is refused before JAX starts; `os.environ.setdefault` used to let the shell win silently.
- test_t0_7.py: the temp directory is a `TemporaryDirectory`, removed on SystemExit too; no W&B history is a FAIL
  line, not an IndexError; the docstring says the holes are checked against marincpp's formula.
- One content hash: `sched::dir_content_hash` calls `resolve()` on a config that holds only `optim.schedule_fixture`,
  so the loader uses config resolution's own hash (the copy of config.cpp `hash_dir` is gone).
- Text: the libdevice is CUDA 13's (jax-cuda13-plugin 0.11.1, PJRT "cuda 13000", nvidia-nvvm 13.2.78), not CUDA 12;
  the literal-formula counts name their cos (S2 section corrected); sched.hpp states the measured rcp gap; s2_formula
  asserts <= 1 ulp for xla_cosf and xla_powf on its sweeps (measured 1; it asserted 2 while this log said 1).
- rcp gap (A100; the reviewer's numbers reproduced). rcp.approx.ftz.f32 differs from 1/x at 627,759 of 4,613,735 x
  in [1.7, 2.5]. 1 - b^t against XLA over every f32 b in [0.5, 1): 0 / 481 / 521 / 473 / 241 / 13 differences at
  t = 1 / 2 / 3 / 5 / 10 / 37, 0 at t = 100, 300, 1000, 4000. b in [0.25, 0.5): 112 / 68 / 15 at t = 2 / 3 / 5, 0
  from t = 10. With the measured rcp table in place of 1/x: 0 everywhere. 16 common betas: 0 for t = 1..2000.
  Decision: state it, do not emulate MUFU.RCP. The fixture is the record of parity runs; elsewhere the gap is 1 ulp
  of b^t at small t for rare betas; and a 4.6M-entry table measured on sm_80 is unverified on sm_90.

Tests (della-vis1, gcc 14, CUDA 12.8, fresh configure + build: 0 warnings)

| check | result |
|---|---|
| ctest s2_formula, s2_fixture, s2_levanter, s2_probe, t0_7_schedules | 5/5 pass (9 + 5 + 1 + 1 C++ cases); 4 sizes and 14 probes, 4 tables each: 0 differences; T0.7: 0 of 98,228 logged values differ, all holes equal the formula, the literal optim.md §6 differs at 0 / 4 / 4 / 28 hole entries; also under Python 3.9.25 |
| fixture regeneration (sizes, then `--probes`) | the 70 existing table and schedule.toml files are bitwise unchanged (so are the content hashes and run ids); 4 new probe directories; `vec.opt_hlo.txt` in all 18 manifests |
| clang 21 UBSan (float-cast-overflow, no recovery), the two reviewer inputs | before: reports at schedule.cpp:27 and xla_math.cpp:90; after: none, and decay 1e19 / 1e300 give lr[99] = 3.0458e-06 |
| clang ASan + UBSan build of nm_tests_s2, 4 suites | 0 reports |
| T0.7 failure paths | no W&B history: 4 FAIL lines, exit 1, no traceback; a corrupted fixture (SystemExit): exit 1; no `t0_7_*` directory left |
| generator with `XLA_FLAGS=--xla_gpu_autotune_level=0` | refused, nothing written |
| `tools/scrub.py --check src/sched tests/s2 docs/BUILD_LOG.md` | clean |

Other suites in the same build: S0 9/9 pass. The GPU tests s0_arena, s0_device and s1_loader each failed once with
`cudaMalloc` / `cudaHostAlloc` out of memory while other users filled the shared A100 (37 of 40 GB), and passed on
rerun (s0: 3 of 3; s1_loader: 6 of 7). S1 passed at 18:05; after 18:17 it fails on meta format 1 vs 2 because
src/data and tools/convert were being changed by another step. The S3 gates need `-DNM_TORCH_PYTHON` at configure.

```bash
NANOMARIN_ROOT=$ROOT $ROOT/project/marin/.venv/bin/python tests/s2/schedule_fixture.py            # 4 sizes
NANOMARIN_ROOT=$ROOT $ROOT/project/marin/.venv/bin/python tests/s2/schedule_fixture.py --probes   # 14 probes
ctest --test-dir build -R 's2_|t0_7' --output-on-failure
```

For the owners of the spec, DESIGN.md and the S0 headers (S2 does not edit them)
1. optim.md §6.1: `c/3967` is computed as `c * f32(1/3967)`, not an f32 division; the division differs from the log
   at 261 of 4959 k.
2. optim.md §6.2: `c/1000` is `c * f32(1/1000)`; `(f32(pi) * c) / T_d` is `c * f32(f32(pi) * f32(1/T_d))`, one rounded
   multiply at run time, not "two rounded f32 operations"; cos is libdevice `__nv_cosf`; `**1.0` is folded (verified).
   "Lookup table": the fixture replaces the logged table and the hole formulas.
3. optim.md §5.3 rule 2 and §10 item 7: XLA:GPU's f32 pow is libdevice `__nv_powf` (reciprocal rcp.approx.ftz.f32;
   measured gap in src/sched/sched.hpp); the schedule fixture is the record.
4. kaiyue_baseline.md T0.7 row: "equal to the formulas of optim.md §6" -> "equal to optim.md §6 as XLA:GPU computes it
   (marincpp build_schedules) and inside their logged neighbours".
5. DESIGN.md §4.4: "exact f32 operation order" -> "XLA:GPU's f32 operation order (x/C as x * f32(1/C), the folded
   cosine constant, libdevice cosf/powf)"; add "a config the formula refuses keeps the fixture (n_ulp_diffs = -1)";
   the generator is tests/s2/schedule_fixture.py. §13.2: schedules live in src/sched.
6. include/marincpp/optim.hpp: the build_schedules comment -> `frac = 1 - f32(c) * f32(1/T_d)` and
   `cos(f32(c) * f32(f32(pi) * f32(1/T_d)))`; `n_ulp_diffs = -1` = cross-check unavailable. capi.h: `config_toml` is
   a file path; `*n_ulp_diffs` can be -1.

Known gaps
- bc tables on sm_90: MUFU.RCP there is not measured (the eta tables equal every H100 logged value; bc has no logged
  values). An H100 run of the generator through tools/devq would settle it.
- Runs with `optim.warmup = 1`, `optim.num_steps >= 2^24` or a beta whose f32 is 1 need a fixture. It loads; the
  cross-check reports -1 without a reason (ScheduleTables has no field for one; build_schedules(c) states it).

## S1: data path (2026-10-24)

Built
- `tools/convert/levanter_to_flat.py` (`fineweb`, `paloma`, `verify`): levanter zarr3 caches -> flat files under
  `$NANOMARIN_ROOT/marin_store_big/marincpp/` (never in the repo); every file is written as `*.tmp`, checked, renamed (true since the review fixes: M1);
  `meta.json` last (counts, BOS/EOS, vocab, `max_segs_per_seq` + histogram, sha256 and size of every file, LCG
  constants per seed, source ledger, the checks it ran). Shard parsing and the numpy order come from
  `bench/n1_loader/nmloader.py`; bulk decode is whole inner chunks through c-blosc in 7 threads.
  - `fineweb-edu-10B-marin128k/`: `tokens.u32` (10,000,000,738 tokens, 40.0 GB, sha256 `26406ede...958b`), `docs.u64`,
    `doc_bytes.u64` (exact UTF-8 byte range of each document), `segs_per_seq.u16` (`.u8` before the review fixes),
    `order_seed42.u32` (T0.1 table),
    `bytes_levanter.i32`, `bytes_exact.i32`. 239 s on della-vis1 (45 Mtok/s). In-converter checks: T0.1 sha, the 12 T0.2
    hashes, 2,001 random windows + the tail against nmloader's independent block decoder: all equal.
  - `paloma-marin128k/`: the 14,529 windows of the 16 domains in component order, `tags.u16`, `docs.u64`,
    `segs_per_seq.u16`, both byte tables; per domain in meta.json: first_seq, n_seqs, scored_tokens, levanter target bytes,
    covered bytes (token stream and raw text), sha256; batch layout (128 rows, 114 batches, 65 real rows last). 16 s.
  - `tools/convert/bytes_levanter.py` (marin .venv) reran levanter's `byte_length_of_token` today: equal to
    `cache/data/bytes_per_token_marin.i32` (sha256 `d3be7450...4e4d`); the converter refuses a mismatch.
- `src/data`: threefry2x32 (one inline body for host and device, `threefry_impl.hpp`), `prng_key`/`split`/`fold_in`/
  `bits32`/`permutation`, `LevanterOrder`, a thread-safe `levanter_seq` with the restart wrap as a parameter, the LCG
  registry (built-in (42, 2441406) plus meta.json entries; unknown seed = fatal), `load_dataset_meta`, `TokenFile`
  (read-only `MAP_SHARED` mmap + an fd for pread), `DataPlan` (Levanter / Repeat / Explicit phases, TOML phase files,
  `fork_plan`, `pass_boundaries`, `times_seen`, `canonical`, `load_data_plan` with content-hash check), `BatchLoader`
  (pageable ring of `prefetch_steps` steps filled by `reader_threads` with one pread per 16 KiB window, a stager into 3
  pinned portable slots, per-rank copy events, `copy_rank_rows` = the worker's cudaMemcpyAsync + event, H(B, k) logged at
  k = 0 and every 1000 steps, wait / read statistics), host references of segment ids and loss weights, a small JSON
  reader, and the C ABI `nm_threefry2x32`, `nm_threefry2x32_device`, `nm_levanter_order`, `nm_plan_seq_of` (config
  path, "" = defaults, or TOML text). Internal API in `src/data/internal.hpp`.
- `tests/s1`: `nm_tests_s1` (8 suites), `test_convert_outputs.py` (stdlib, Python 3.9+), `nm_bench_s1`.

Tests (della-vis1, from a clean build directory)

```bash
export NANOMARIN_ROOT=$ROOT TMPDIR=$ROOT/tmp/marincpp_s1/tmp
nice -n 19 $ROOT/envs/nanomarin/bin/python -I tools/convert/levanter_to_flat.py paloma --levanter-bytes <rerun.i32>
nice -n 19 $ROOT/envs/nanomarin/bin/python -I tools/convert/levanter_to_flat.py fineweb --levanter-bytes <rerun.i32>
cmake ... && cmake --build build -j 7 && (cd build && nice -n 19 ctest)        # 23/23 pass (S0, S1, S2), 0 warnings
python3 tests/s1/test_convert_outputs.py --lib build/libmarincpp_kernels.so --src . --full   # + 40 GB re-hash, 54 s
nice -n 19 build/tests/s1/nm_bench_s1 --first 26000 --seconds 90 --threads 16 --pace 2793000
```

| ctest | checks | result |
|---|---|---|
| t0_1_order | T0.1: s(0..5,859,327) computed in C++ -> sha256 `dde0ca7b...655b`; element-wise equal to the levanter table through `nm_levanter_order`; `levanter_seq` on a stride through all passes; first wrapped index 2,441,216, pass 3 at 4,882,432; converted `order_seed42.u32` and its LCG entry | pass |
| t0_2_batches | T0.2: all 12 H(B, k) of data.md §7 through the loader (start at k, readers, stager, acquire; host slots, not pinned ones: corrected in the review fixes) and the loader's own hash log; also direct assembly; B 256 batch k = B 128 batches 2k, 2k+1 | 12/12 equal |
| t0_3_segments | T0.3: segment ids + loss weights of the 12 batches; only 4 of them (B 128 k 0, 1, 1000, 4958 of `levanter_ref.json`: tokens, segment ids, loss weights, each device's rows) and the batch-0 fixture against levanter arrays, the other 8 only through the row's segment count; every row's segment count equals the converter's numpy count; max 17 <= `max_segs_per_seq` | pass, 5 levanter references |
| t0_4_paloma | T0.4: the 16 domain sha256 read through `TokenFile`, positions, tags, 14,529 windows, 114 batches, 63 zero rows in batch 113; levanter's eval batches 19, 20, 21 (tokens and tags), 113 (tokens only; the zero-row check could not fail) and c4_en segment ids / loss weights; per-domain levanter target bytes recomputed in C++ = data.md §8.1 | pass |
| t0_5_bytes | T0.5: `bytes_levanter.i32` in both sets: sha256 `d3be7450...`, 128,256 entries, 256 zeros, equal to cache/data and to today's levanter rerun | pass |
| s1_threefry | Random123 known answers; data_key, mix/shuffle/perm keys, PCG64 seed 937906096; block-0 permutation head; s(0..7); device bits = host bits on 2^20 inputs (A100) | pass |
| s1_plan | default plan = LevanterOrder; pass boundaries 1_2b {9536, 19072}, 520m {9536}; Repeat = levanter's max_train_batches restart (closed form, = Levanter when subset = N); reshuffled passes are permutations of the subset; plan files (continuation of index_base, explicit ids, content hash, 7 malformed cases); fork keeps batches < k; times_seen vs brute force | pass |
| s1_loader | every step of a 2-phase plan (batch and length change) equals direct assembly in pread and mmap modes at ring depth 1, 4, 64; restart at 5 steps and on a fork; 4 slow consumer threads hold slots while the stager runs; a read error reaches acquire() as an exception; GPU: pinned slots + events + copy_rank_rows, device rows equal | pass; compute-sanitizer memcheck 0 errors; ThreadSanitizer (clang 21) 0 reports on s1_loader and s1_plan |
| s1_convert_outputs | sizes and sha256 of every converted file (`--full` includes tokens.u32), T0.1/T0.4/T0.5 from the files, C ABI (threefry, order, unknown seed error, `nm_plan_seq_of` 1_2b k 9536 -> 1316203, defaults (130m) k 4958 -> 1568248, both data.md §7) | pass (Python 3.12 and 3.9) |

Reader throughput (`nm_bench_s1`, della-vis1, 7-core cgroup, B 128, ring 256 steps, fresh steps from GPFS, every step
copied H2D to the A100). Target: the trainer's 130m rate, 2,793,000 tokens/s on 4 H100 (DESIGN.md §12.2); 8 GPUs at the
same per-GPU rate would need 5,586,000.

| mode | threads | consumer | tokens/s | stalls after the first step |
|---|---|---|---|---|
| pread | 8 / 16 / 16 / 32 | drains as fast as possible, 45 s each | 9.49 M / 7.51 M / 12.32 M / 10.94 M | n/a |
| mmap | 16 | drains, 45 s | 14.32 M | n/a |
| pread | 16 | paced at 2,793,000 tokens/s, 90 s | 2.78 M (= pace) | 0 (0.7 ms total) |
| pread | 16 | paced at 5,586,000 tokens/s, 90 s | 5.53 M (= pace) | 1 stall of 0.26 s while the ring was still filling |

The drain rate is 2.7-5.1x the 4-GPU target (1.3-2.6x the 8-GPU rate). Single steps take up to 0.76-3.6 s when GPFS
stalls; the 256-step ring (48 s at 130m) absorbs them. mmap and pread differ by less than the GPFS noise between windows;
pread stays the default (no page-fault readahead behaviour to depend on).

Findings
- Every fineweb document is `[BOS, text, EOS]` and no BOS or EOS occurs inside text anywhere: BOS and EOS counts are both
  9,966,814 = documents, and all document starts and ends were checked. data.md §2.4's "once per window" comes from its
  window-edge counting. So EOS-derived segments equal document segments on this corpus; `max_segs_per_seq` = 17.
- Paloma: marin's tokenize step appends one byte (a space, token 220 before EOS) to every document: in all 16 domains
  every document's token-stream bytes equal its raw text bytes + 1, in raw file order. meta.json keeps both the stream
  bytes the windows cover and `covered_text_bytes` (without the space); exact bpb (S9) must pick one on purpose.
- The exact UTF-8 table differs from levanter's at 1,346 ids (862 have the U+FFFD shape of data.md §8.4).
- DESIGN.md §5.5 describes Repeat as "mixture index (j mod subset) of a Levanter order". Levanter's max_train_batches
  slices the LCG-permuted set to its first `subset` sequences before the mixture (datasets.py:858-866), so the restart
  wraps the block-shuffled index mod subset and sigma stays over the full N. `Repeat` implements levanter's form
  (`levanter_seq(.., wrap = subset, j)`); the DESIGN text should be updated by its owner.

Known gaps
- data.hpp (S0) lacks members the implementation needs: `BatchLoader` has no pimpl and no destructor (its state lives in
  a table keyed by the loader's address; a started loader must be `stop()`ped, tests hold a guard), `TokenFile` has no fd
  and no destructor (fd in a side table; mappings are never unmapped), `Phase` has no mixture-block field (plans accept
  only data.mix_block = 2048). Suggested header change: `std::unique_ptr<struct LoaderImpl> impl_` + `~BatchLoader()`,
  `int fd_` + `~TokenFile()`, `int mix_block` in Phase.
- `src/main.cpp` (S0) still plans with `kDefaultMaxSegsPerSeq` = 24 and resolves without the dataset meta; the trainer
  (S7/S10) must call `load_dataset_meta` and pass it to `choose_plan`, `resolve` and `exec_shape` (gives 17 and the
  `#sha256=` dataset identity).
- `eval.hpp` `load_paloma` belongs to S9; the converted Paloma directory carries all its inputs.
- The DESIGN §5.5 gate "seq_of equals the order the JAX runs read for DATA_EPOCHS 2.4, PASS_DATA rs, repack" needs
  order dumps from those JAX runs; not run. PASS_DATA rs / repack are Explicit phases over materialized caches, which the
  converter does not convert yet. `times_seen` = sum over phases of the most-read window overlapping the document.
- A phase at another seq_len needs the LCG constants for its N: rerun the converter with `--seq-lens` (current meta:
  4096 only). load_dataset_meta checks file sizes, not hashes (the converter's `verify` and `--full` re-hash).
- Throughput is from the vis node (A100 PCIe, shared GPFS), not an H100 node; H100 numbers come with tools/devq.

## S1 review fixes (2026-10-24)

Built (every defect reproduced on della-vis1 before the change)
- B1, wrong converter output: `segs_per_window` cast the count to uint8. Paloma has 85 windows with 256-291 segments,
  all in twitterAAE_HELM_fixed (about 17 tokens per document); `segs_per_seq.u8` held count mod 256 and meta.json said
  `max_segs_per_seq` 255 (true 291; numpy recount reproduced). Now counts and histogram are int64, the max is taken
  before any narrowing, and the file is `segs_per_seq.u16` (exact: a 4096-token window holds at most 4096 segments;
  the converter refuses a count above 65535). Meta format 2; `load_dataset_meta` refuses format 1. commit() deletes
  files that only the old meta.json listed, so both sets lost their `segs_per_seq.u8`. Per-domain maxima: twitterAAE
  291, gab 122, dolma_100_programing_languages 47, the others 41 or less. Fineweb max stays 17.
- B2, use-after-free: `stop()` freed the slots while a worker blocked in `acquire()` woke and read its slot (ASan
  heap-use-after-free; the negative control below reproduces it). Now every `acquire` / `release` / `copy_rank_rows` is a
  counted caller: refused with "loader stopped" once `stop()` began, and `stop()` frees the slots only after the count
  is 0. The stop flag lives in the shared state, not in the BatchLoader. `stop()` is idempotent and safe on two threads.
  Every call after `stop()` throws "not started". `release` and `copy_rank_rows` check that the slot is this loader's.
- M1: the converter commits after its checks. Every file stays `<file>.tmp`. The T0.2 hashes, the spot windows (tokens
  and now also the stored segment counts against nmloader's block decoder) and, for Paloma, the 16 domain hashes and
  a full segment recount read the `.tmp` files. Only then does commit() remove the old meta.json, rename the files and
  write meta.json. A failed check leaves the old set intact; a crash during the commit leaves no meta.json.
- M2: both sets reconverted with this converter (Paloma 16 domains, then fineweb in 731 s on the loaded vis node).
  tokens.u32 sha256 is unchanged (`26406ede...958b`); the fineweb `lcg` entry now has `seq_len`.
- M3: datasets enter by content. The default phase keeps the config's `#sha256=` pin, a plan file's `dataset` and
  `explicit_file` keep theirs, and a wrong pin is fatal, also when the dataset was registered before (`find_dataset`
  compares it). `canonical()` names every dataset `sha256:<meta.json sha256>`, pinned or not and under any path, and
  every explicit file by its own sha256. Metas loaded by path are cached per path, so two directories whose metas share
  a name never answer for each other.
- M4: `check_token_file`: a TokenFile serves a phase only if its n_tokens and tokens_sha256 equal the phase dataset's
  (BatchLoader::start, assemble_batch). The reviewer's synth-other probe is now refused.
- M5: `explicit_file` checks `fstat`; `TokenFile::open` closes and unmaps everything on any failure. ReadMode::Mmap
  copies from a loader-private mapping with `madvise(MADV_RANDOM)` (other users of the TokenFile mapping keep the
  default) through a SIGBUS guard (sigsetjmp per window; the handler acts only on a kernel SIGBUS inside the armed
  range of the faulting thread, else it restores the previous action). A truncated file now reaches `acquire()` as
  "SIGBUS reading window ..." (mmap) or "EOF" (pread). `assemble_batch` / `batch_hash` read with pread. A/B in mmap
  mode, 16 readers, host only, fresh steps, alternating order, 40 s each: with MADV_RANDOM 4.53 / 4.40 / 9.99 / 4.02
  Mtok/s, without 3.39 / 1.47 / 1.72 / 7.82 (the node had load average 46-143, so the absolute rates are below the S1
  table). pread stays the default.
- M6: a reshuffled Repeat with `index_base != 0` is refused by `validate_plan`.
- M7: `first_index_of_pass` and `pass_boundaries` state that they return the start of the mixture block that holds
  the first index of a pass (subset 4767 x 128: step 4752; steps 4752-4767 mix passes 0 and 1).
- M8: see the tests below. The S1 section's t0_2, t0_3 and t0_4 rows are corrected in place.
- M9: `start()` runs `validate_plan` (a plan with a gap is refused), checks the TokenFiles, and with pinned slots
  checks for one CUDA device per rank, all before it allocates. An allocation failure frees what was allocated.
  Pending options are consumed only by a start() that succeeds. The reader's phase scan is bounded.
- M10: the first reader at a step computes its B sequence ids outside the loader lock (`step_seqs`, one phase and
  dataset lookup per step). A reshuffled pass's permutation is computed once per process (call_once; the newest 4
  passes stay cached) and shared by all threads. The LCG constants are cached per thread.
- The SIGBUS guard's TLS variables are volatile with signal fences around the copy. At -O3, gcc dropped the arming
  store as dead (a later store of nullptr, and a memcpy that does not read it), and the release build crashed with
  a bus error. The -O1 sanitizer builds had passed.

Tests (della-vis1, gcc 14 + CUDA 12.8, fresh configure and build: 0 warnings)

| check | result |
|---|---|
| ctest (all entries except s3_*, whose gates need `-DNM_TORCH_PYTHON` at configure) | 25/25 pass: S0 9, S1 11, S2 5 |
| t0_3_segments | the 12 batches element by element against data.md §3.1 (seg[0] = 0, seg[i] - seg[i-1] = [tok[i-1] = EOS], weight 0 only at 4095); row counts = `segs_per_seq.u16`; meta max 17 = file max = histogram max, histogram sums to N |
| t0_3_levanter (new ctest, "Skipped" when a reference is absent) | levanter_ref.json B 128 k 0, 1, 1000, 4958 (tokens, segment ids, loss weights, 4 device slices) and the batch-0 fixture: pass. The other 8 batches have no levanter array reference |
| t0_4_paloma.segment_counts (new) | all 14,529 windows recounted in C++ = `segs_per_seq.u16`; max 291 = meta; 85 windows above 255, all tag 14; the per-domain maxima above |
| t0_4_levanter (new ctest) | eval batches 19-21 tokens and tags, 113 tokens and real-row count, c4_en: pass. The tautological zero-row check is removed (eval_batch zero-fills) |
| t0_2_batches | with a GPU the 12 hashes now pass through pinned slots and `copy_rank_rows` to the A100 and back |
| s1_loader (9 cases) | new: stop_while_acquire_blocked (200 rounds), calls_race_stop (release/acquire racing two stop() calls, calls after stop), start_checks_inputs (wrong TokenFile, plan gap, ranks > devices; options kept), reshuffled_passes (both read modes), read_error_propagates (truncated file: pread EOF, mmap SIGBUS) |
| s1_plan (9 cases) | new: content pins (default, plan-file and explicit pins; canonical by content under two paths), reshuffle + index_base refused, pass boundary = block start (4767 -> 4752, 4768 -> 4768), step_seqs = seq_of for all kinds and across 8 threads |
| s1_convert_outputs `--full` (Python 3.12; 428 s) | 83 checks pass: format 2, no `.u8` left, every window of both sets recounted from tokens.u32 = `segs_per_seq.u16`, Paloma max 291 (85 windows above 255, all tag 14), fineweb histogram = file counts, `lcg` has `seq_len`, all sha256 incl. tokens.u32. Without `--full` also under Python 3.9.25 |
| clang 21 ASan + UBSan (src/{core,config,memory,data} + tests/s1; S1 files with -Wall -Wextra -Werror) | 0 reports: s1_loader (with CUDA, `protect_shadow_gap=0`), 3 single-core runs (`taskset -c 3`), s1_plan, t0_2, t0_3, t0_4 |
| clang 21 TSan | 0 reports: s1_loader (GPU cases included), s1_plan, t0_2, t0_3, t0_4 |
| compute-sanitizer memcheck, s1_loader | 0 errors |
| negative control | the stop-while-blocked scenario linked with the pre-fix loader: ASan heap-use-after-free in `acquire()`; with the fixed loader: 200/200 rounds end in an exception, 0 reports |
| `tools/scrub.py --check src/data tests/s1 tools/convert docs/BUILD_LOG.md` | clean |

The GPU cases failed some runs with `cudaHostAlloc` out of memory while other users held 39.7 of the A100's 40 GB.
The reruns passed.

```bash
export NANOMARIN_ROOT=$ROOT TMPDIR=$ROOT/tmp/<dir>
nice -n 19 $ROOT/envs/nanomarin/bin/python -I tools/convert/levanter_to_flat.py paloma --levanter-bytes <rerun.i32>
nice -n 19 $ROOT/envs/nanomarin/bin/python -I tools/convert/levanter_to_flat.py fineweb --levanter-bytes <rerun.i32>
ctest --test-dir build -E '^s3_' --output-on-failure
python3 tests/s1/test_convert_outputs.py --lib build/libmarincpp_kernels.so --src . --full
```

For the owners of the spec, DESIGN.md and the S0 headers (S1 does not edit them)
1. data.hpp comment and DESIGN.md §5.1: `segs_per_seq.u8` -> `segs_per_seq.u16` (exact counts; meta format 2).
2. DESIGN.md §5.5: Repeat is levanter's slice-then-restart (mixture index wrapped mod subset after the max_train_batches
   slice, sigma over the full N), not "j mod subset of a Levanter order" (della_muonh_qwen3_scaling.py:768-770,
   levanter datasets.py slice, mixture.py `_remap_indices`). A reshuffled Repeat needs index_base 0.
3. data.md §2.4: BOS and EOS do not occur inside text. A full-corpus check found BOS at all 9,966,814 document starts
   and EOS at all ends, with both counts equal to the document count (the reviewer confirmed it).
4. DESIGN.md §13.1: there is no `third_party/` nlohmann/json; src/data has its own JSON reader.
5. data.hpp: `BatchLoader` and `TokenFile` still have no destructor; a started loader that is destroyed without
   `stop()` calls std::terminate. Suggested: `std::unique_ptr<struct LoaderImpl> impl_` + `~BatchLoader()`,
   `int fd_` + `~TokenFile()`, `int mix_block` in Phase.

Known gaps
- The DESIGN §5.5 order gate: the reviewer's probe (levanter's own `train_set` with max_train_batches 4768 and 4767 vs
  `nm_plan_seq_of`, 11,598 rows around both pass wraps, 0 mismatches) covers DATA_EPOCHS. It is not a repo test yet.
  PASS_DATA rs and repack stay open.
- `load_dataset_meta` checks file sizes, not hashes. The converter's commit order prevents a stale pair; `verify` and
  `--full` re-hash.
- The SIGBUS guard owns the process's SIGBUS action: a handler installed later replaces it, and then mmap reads lose
  the guard (pread mode does not depend on it). `TokenFile::window()` still returns a raw pointer, and its callers
  get no guard.
- MADV_RANDOM was measured on the vis node's GPFS client only.

## S3: elementwise kernels (2026-10-24)

Built (src/kernels/elementwise, all portable: sm_90a + sm_80 cubins; levanter-mode units build with `--fmad=false`,
accurate-mode units keep nvcc's default contraction since the S3 review fixes)
- RMSNorm over D: `rmsnorm_fwd`, `rmsnorm_bwd_add`; sandwich tails `sandwich_tail_fwd/bwd` (sandwich placement with a
  post-norm; `g_post = nullptr` = pre-LN / hybrid placement: residual add + next pre-norm only). QK-norm + RoPE +
  q scale: `qknorm_rope_fwd/bwd` (MHA and GQA, hd 64 and 128, any qkv/dqkv row stride, q-only or k-only backward),
  `dq_post_qknorm_rope_bwd` (reads FA3's f32 dq accumulator [H, T, hd]), `rope_fwd_dt` (RoPE stage alone, for T1.3 and
  QK-norm-free variants). RoPE tables `build_rope_tables` (host, DESIGN.md §3.4) plus a device libdevice method.
  SwiGLU fwd/bwd on the 64-column interleaved gate/up layout. Embedding gather; deterministic backward: CUB radix sort
  of (token, position), then `EmbedAccum::F32` = fixed 256-entry chunks of the sorted list, one warp per (chunk,
  256-column slice) summing each run piece in ascending position order, runs inside a chunk added at once, pieces of
  runs that cross chunks merged in chunk order by the chunk where the run starts (no atomics, no warp walks a long run
  alone); `Bf16Sequential` = bf16 running sum in ascending position order (strict preset). `doc_meta` (targets,
  weights, positions, segment starts per row, packed cu_seqlens, overflow flag). reduce.hpp `reduce_partials_rows`,
  `sum_fixed`; `cast_f32_bf16_multi`. Residual adds live in the sandwich tails (x_out = x_in + y) and in
  `rmsnorm_bwd_add` / `sandwich_tail_bwd` (residual-gradient adds).
- Every kernel exists for bf16 storage and f32 storage (f32 compute policy) x levanter / accurate rounding:
  `ew::*_dt(DType, ...)` in `ew_api.hpp` (the public structs, pointers read as float* under DType::f32). C ABI: every
  elementwise entry point of capi.h (dtype 0 = f32 storage) plus `capi_ew.h` (rope method, rope-only, raw partials,
  reductions, cast, storage-generic embedding). In the C ABI the gain gradients are the reduced f32 [width] of the call
  (bf16-rounded under levanter + bf16), partials from cudaMallocAsync (the trainer passes planned slots).
- Levanter rounding = the rounding points of levanter's jaxpr (numerics.md §3.2-§3.5) op by op: RMSNorm
  `y = bf16(x r)`, `out = bf16(g y)`, backward `a = bf16(g dy)`, `dg += bf16(dy y)`, `dx = bf16(a r + c 2x)`; RoPE
  `bf16(bf16(n cos) + bf16(rot(n) sin))`, `q = bf16(q scale)`, backward the transposed ops; SwiGLU with XLA's
  logistic expansion `s = bf16(1 / bf16(1 + bf16(exp(-g))))`; residual and residual-gradient adds in bf16. rsqrt,
  exp, pow, cos, sin are libdevice (`rsqrtf`, `expf`, ...: the functions XLA:GPU emits), divisions IEEE. Accurate: f32
  chain, one rounding per stored output. The bf16 levanter instantiation computes on packed bf16x2 (`__hmul2_rn`,
  `__hadd2_rn`): a product of two bf16 values is exact in f32 and a sum of two is exact or returns the larger operand,
  so one RNE rounding equals the f32-then-bf16 chain; the tests below check it on every element.
- Norm partitions: T rows split into P = min(1024, ceil(T/16), 4 x SMs) parts of consecutive rows, one 256-thread CTA
  per part (`__launch_bounds__` keep 4 CTAs per SM resident: <= 64 registers), gain accumulators in shared memory
  (one private slot per thread), one partial row per part reduced in fixed order. P depends on T and the GPU's SM
  count (432 on the A100, 528 on the H100), never on timing.

Gates (della-vis1 A100-PCIE-40GB, clean configure + build in a fresh directory: 0 warnings; ctest 30 tests, of them 7
new: all 7 S3 tests pass; in the full run the S1 test s1_loader failed once on `cudaHostAlloc` out of memory and passes
when rerun; the GPU is shared with other jobs at 100 % utilization and up to 38 of 40 GiB)

| ctest | checks | result |
|---|---|---|
| s3_rmsnorm | RMSNorm D = 512 / 768 / 1024 / 2048 (130m, 300m, 520m, 1_2b, pls width 512), T = 8192 (4096 at D = 2048), per-row input scales 1e-3..1e2, gains 1 + 0.2 N(0,1); sandwich tails post-norm and pre-LN at 130m, 1_2b, pls; partial determinism and bounds; D = 64..4096 | f32 storage vs f64 autograd: every tensor rel <= 1.9e-7 (bound 1e-6). bf16 levanter vs the emulation: exact-sum inputs 100 % bitwise for out and dx at every width; realistic inputs >= 99.998 % bitwise (summation-order flips), forward driven by the kernel's rstd 100 %, rel(kernel, f64) = rel(emulation, f64) to 4 digits. bf16 accurate: rel 1.65e-3 vs levanter 2.3-2.8e-3 (out, dx). Per-row bound held everywhere |
| s3_qknorm_rope | 130m / pls (MHA hd 64), 520m (GQA hd 64, padded qkv_ld), 1_2b (GQA hd 128), T = 8192 at positions 0..4095 | f32 storage: rel <= 2.3e-7. bf16 levanter: sparse rows (exact sums) 100 % bitwise for q_rot, k_rot, dq, dk; realistic >= 99.999 %; kernel-rstd forward 100 %. Accurate rel 1.66e-3 vs levanter 3.3-3.7e-3. v columns untouched; k-only call equal; dq_post equals qknorm_rope_bwd on bf16(accumulator) bitwise (dgq too) |
| s3_swiglu | F = 1792 / 2688 / 3584 / 7168, gate values up to +-100 | f32 rel <= 6.6e-8; bf16 levanter 100 % bitwise vs the emulation (no reductions); accurate rel 1.65e-3 vs levanter 2.6e-3 (h), 4.1e-3 (dg) |
| s3_embed | V = 128,256: D = 512 at T = 131,072, D = 2048 at T = 32,768 (Zipf tokens, runs up to 12,303), f32 storage; chunk-edge cases | forward bitwise; backward rel(f32 sum, f64) <= 1.5e-7, absent rows bitwise untouched, touched flags exact, repeatable bitwise; Bf16Sequential bitwise equal to the ascending bf16 running sum |
| s3_misc | doc_meta at S = 4096 / 4224 / 4480 / 19456 (EOS at row start and end, length-1 segments, overflow, invalid row); reductions; cast | all outputs equal the numpy reference; cast bitwise RNE incl. inf, -0, subnormals |
| s3_t1_levref | T1.2, T1.3 against cache/levref (fixture 130m/units_cpu_vis, made by XLA:CPU) | see below |
| s3_bench | GB/s per kernel (label bench) | see below |

T1.2 (130m fixture, levanter's own RMSNorm modules, gains = 1 at init): block norm out 99.9997 % / dx 99.9995 %
bitwise, max 1 ulp; q_norm out 99.9998 % / dx 99.9986 %, max 1 ulp; k_norm out 99.9995 %, dx 99.9989 % bitwise with 4
of 524,288 elements at 2-4 ulp. Those 4 are cancelling sums (|dx| ~ 1e-5 of the addends a r and c 2x): nano and
levanter differ by 2.3 f32 ulps of the addends, and levanter itself is 5-10 bf16 ulps from f64 there. A row sum in any
other order than XLA:CPU's moves such an element by a few f32 ulps of the addends, which is many bf16 ulps of dx.
Corrected in the S3 review fixes: the spec text ("every element within 1 bf16 ulp") fails on these 4 elements, so
T1.2 is NOT passed; the test reports it as XFAIL until the owner amends T1.2 or records the failure. The fused QK kernel (identity RoPE
tables) gives the same bits as rmsnorm over hd. dg: rel(nano, f64) 2.8-3.1e-3 vs rel(levanter, f64) 1.2e-2 (block)
and 2.7e-2 (q, k): XLA:CPU accumulates the gain reduction in bf16.
T1.3: rotated q and k 100 % bitwise (rope_fwd on levanter's qn, kn with levanter's tables). Tables: the host-double
method equals the XLA:CPU fixture's inv_freq bitwise (32/32) and its cos table (100 %); sin differs at 2 of 262,144
entries (1 distinct angle: the tables repeat each half; XLA:CPU's sin); cos/sin from levanter's own angle in double -> f32 -> bf16: the same. The libdevice method:
inv_freq 30/32, cos 99.98 %, sin 99.97 % against XLA:CPU. The GPU table gate (`t1_3_tables_gpu`) skips until a
GPU-made fixture exists: the run computed inv_freq and the tables on the GPU (libdevice powf/cosf/sinf), so that
fixture decides between the two methods or O4 (load levanter's tables).

Bandwidth (A100-PCIE-40GB, peak 1555 GB/s from the memory clock and bus width; a device-to-device copy measured in the
same process gives 1369 GB/s (130m run) and 1402 GB/s (1_2b run) on the shared GPU; bytes = each input read once,
each output written once; median of 20 calls; levanter rounding, bf16; `tests/s3/bench_s3.py`)

| kernel | bytes per call | 130m / pls: T 131,072, D 512, H = KV = 8, hd 64, F 1792 | 1_2b: T 32,768, D 2048, H 16, KV 8, hd 128, F 7168 |
|---|---|---|---|
| rmsnorm_fwd | 4TD + 4T | 1233 GB/s (79 % of peak, 90 % of copy) | 1243 (80 %, 89 %) |
| rmsnorm_bwd_add (with residual add) | 8TD + 4T | 1256 (81 %, 92 %) | 1214 (78 %, 87 %) |
| sandwich_tail_fwd (post-norm) | 8TD + 8T | 1272 (82 %, 93 %) | 1272 (82 %, 91 %) |
| sandwich_tail_bwd (post-norm, + gain reductions) | 12TD + 8T | 1196 (77 %, 87 %) | 1185 (76 %, 85 %) |
| qknorm_rope_fwd | 4T(H+KV)hd + 4T(H+KV) + 4T | 1237 (80 %, 90 %) | 1180 (76 %, 84 %) |
| qknorm_rope_bwd (+ gain reductions) | 6T(H+KV)hd + 4T(H+KV) + 4T | 1068 (69 %, 78 %) | 1107 (71 %, 79 %) |
| dq_post_qknorm_rope_bwd (+ reduction) | 8THhd + 4TH + 4T | 1182 (76 %, 86 %) | 1153 (74 %, 82 %) |
| swiglu_fwd | 6TF | 1257 (81 %, 92 %) | 1283 (83 %, 92 %) |
| swiglu_bwd | 10TF | 1287 (83 %, 94 %) | 1299 (84 %, 93 %) |
| embed_fwd | 4TD + 4T | 1621 (frequent rows hit L2) | 1557 |
| embed_bwd F32, sort included | 2TD + 8 x rows x D + 24T | 505 us (641 GB/s effective, 45,677 rows) | 488 us (771, 14,958 rows) |
| doc_meta | 16T | 37 us (two small launches) | 29 us |

The first working version (same bench) reached 1064 (rmsnorm_bwd_add), 972 (sandwich_tail_bwd), 933
(qknorm_rope_fwd), 703 (qknorm_rope_bwd) and 890 GB/s (dq_post) at 130m. What moved them: one CTA per part with 4
CTAs per SM resident (no SM runs one part more than another), the packed bf16x2 levanter math (a quarter of the
instructions; the QK kernels were issue-bound), gain accumulators in shared memory instead of registers (no spills
under the 64-register bound; some instantiations still use 8-80 B of stack, see S3 review fixes), and no dynamically
indexed register arrays (they had gone to local memory).

Commands
```bash
ssh della-vis1; cd $ROOT/project/marin.cpp; module load gcc-toolset/14 cudatoolkit/12.8
export NANOMARIN_ROOT=$ROOT          # tests/s3 finds the torch interpreter $NANOMARIN_ROOT/envs/nanomarin/bin/python
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++        # or -DNM_TORCH_PYTHON=<python with torch>
nice -n 19 cmake --build build -j 8 && (cd build && nice -n 19 ctest -R 's3_' --output-on-failure)
# one script, selected cases:
python tests/s3/test_rmsnorm.py --lib build/libmarincpp_kernels.so --src . grad_rmsnorm_1_2b
# sanitizers (kernels of nm::ew only):
compute-sanitizer --tool memcheck --kernel-name kns=2nm2ew \
  python tests/s3/test_misc.py --lib build/libmarincpp_kernels.so --src .
```
compute-sanitizer on the S3 kernels only (`--kernel-name kns=2nm2ew`), cases: RMSNorm exact chains, D = 64..4096,
sandwich tails in both placements, partials; QK sparse rows (hd 64 and 128), k-only call, dq_post; SwiGLU; embedding
chunk edges, bf16 sequential, f32 storage; misc. memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors (cases
lost to an out-of-memory error from other jobs were covered by an earlier run on the same kernels). initcheck runs
without the kernel filter (a filtered run does not see torch's writes and flags every read of torch-made data): 0
errors.

Findings
- XLA:GPU on the A100 does not keep levanter's bf16 rounding points inside fusions: the optimized HLO of the 130m step
  (cache/levref/130m/precision_a100_vis) computes SwiGLU as `bf16(f32(g) * (1 / (f32(bf16(exp(-g))) + 1)) * f32(u))`
  (one rounding after exp, one at the end) and q_norm + RoPE with the unrounded f32 norm output in the `q cos` term
  (only the rotate_half branch is rounded), because `xla_allow_excess_precision=true` removes bf16 round trips when
  sm_80 upcasts bf16 arithmetic. Hopper has native bf16 add/mul, and XLA's GPU float support may keep those ops in
  bf16 there (not verified here), so the H100 program can differ from the A100 one. The levanter mode follows the
  jaxpr (DESIGN.md §3.0; the XLA:CPU fixture agrees: RoPE 100 % bitwise). T1.0 on the H100 (optimized HLO, item 5)
  decides whether any chain must change; every rounding point is one R8 op (ew_common.cuh), so a change is local.
- torch turns `tensor / python_scalar` into a multiply by the reciprocal (the emulation divides by a tensor).

Known gaps
- GPU-made T1.2/T1.3 fixtures (tools/levref/t1_gpu.sh units) have not landed: `t1_3_tables_gpu` skips; T1.2 ran on
  the XLA:CPU fixture only; 1_2b (hd 128) has no fixture yet (the kernels are tested at hd 128 against f64 and the
  emulation). The H100 numbers (and the sm_90a code paths) come later through tools/devq.
- Throughput on the shared vis A100 varies run to run by up to ~30 % (other tenants time-slice the GPU); the table
  is one run. Other jobs often hold all but a few MiB of the 40 GiB: the S3 runner reruns a case that dies with a CUDA
  out-of-memory error before any check failed (up to 4 attempts, 15 s apart) and the bench halves T for a section that
  does not fit; a check failure is never retried.
- The norm partial count (`ew::norm_partial_rows`) depends on the SM count (above): bitwise repeatable on one GPU
  type, different partials on A100 vs H100. reduce.hpp says "depends only on the shape"; S0 may want that line updated.
- Embedding F32 order: a run that crosses chunks is summed per chunk in ascending order and the chunk sums are added in
  chunk order (deterministic; not one sequential sum). The F32 path keeps 2 x ceil(T/256) x D f32 chunk partials in
  the cub temp region: the planner's 32 MiB `ws.embed_bwd.cub_temp` holds them up to T D ~ 9e8 (130m: 2 MiB);
  `embed_bwd_workspace_bytes(T, V)` sizes them for D <= 4096. `EmbedBwdWorkspace.run_start` / `n_runs` are unused.
- Under accurate rounding DESIGN.md §3.4 gives FA3 the bf16 scale: the caller then passes q_scale = 1 to
  qknorm_rope_fwd, and for dq_post_qknorm_rope_bwd q_scale is the whole scale applied to the accumulator. Under
  levanter + bf16 the q scale must be a bf16 value (checked).
- Not built here (no stage owns them yet in S3): reduce.hpp `stat_partials` / `stat_finalize` (statistics hooks), the
  fused GEMM-epilogue forms (S8), a standalone residual-add kernel (the adds are fused into the tails).
- `doc_meta` takes 28-34 us per call (two small launches; latency, not bandwidth); `embed_bwd` 0.5 ms at T = 131,072
  of which the f32 read-modify-write of the touched table rows is inherent.

## S3 review fixes (2026-10-24)

Built (every defect reproduced on della-vis1 before the change)
- Accurate-mode gates (blocking). Before, every bf16/acc check was `rel(k, f64) <= 1.25 x rel(levanter emulation,
  f64)`, so a kernel that rounds like levanter passed (reproduced: 130m RMSNorm out 1.6589e-3 = the single-rounding
  floor, levanter 2.3422e-3, bound 2.93e-3). Now `nmk.accurate_check` asserts `rel(k, f64) <= 1.05 x rel(bf16(f64),
  f64)` for every bf16 output of accurate mode: RMSNorm out, dx; sandwich x_out, a_next, dx_in, dbranch; QK q_rot,
  k_rot, dq, dk; dq_post dq; SwiGLU h, dg, du. The f32 outputs of accurate mode (gain gradients, rstd) are held to
  rel <= 1e-6. The same call asserts that the levanter emulation fails the bound on the same inputs, so each gate is
  shown to have power. The sandwich accurate oracle is now the f64 math of the kernel's stored inputs: a_next and the
  whole backward read the stored bf16 x_out (the old oracle used the unrounded x_in + y, which hid the floor). Pre-LN
  x_out = bf16(x_in + branch) is one rounding in both modes (no power check). Post-norm x_out under levanter is only
  1.04 x the floor over the tensor (x_in dominates y), so its power check runs on the rows with |x_in| < |y| / 4
  (levanter 1.73 x the floor there).
- T1.2 (blocking). The test now enforces the spec text. The 4 k_norm dx elements that are 2-4 ulp from levanter
  (fixture 130m/units_cpu_vis; elements 316818, 611964, 2321597, 3514961; the same bits from rmsnorm over hd and the
  fused QK kernel) make the case XFAIL: a new runner status, printed as "NOT PASSED", listed on the summary line, exit
  code 0. Any other violation, including an element beyond 1 ulp that is not a cancelling sum, is a FAIL. T1.2 is
  recorded as failing on those 4 elements. Owner decision: amend T1.2 in kaiyue_baseline.md with the cancellation rule
  (an element beyond 1 ulp passes when |nano - levanter| <= 4 x 2^-24 x (|a r| + |c 2x|); measured 2.32 f32 ulps of
  the addends), or keep T1.2 failing.
- doc_meta: rows = 13000 and 16384 failed with `doc_meta.cu:100: cudaErrorInvalidValue` (the pack kernel's
  (rows + 1) x 4 B of dynamic shared memory passed 48 KiB). The launcher now opts in with `cudaFuncSetAttribute`
  above 48 KiB (64 KiB at 16384 rows; a function attribute, legal during graph capture).
- Embedding backward, Bf16Sequential (strict preset): 14-15 ms per call at T = 131,072, D = 512, Zipf tokens
  (reproduced; longest run 12,626). New `k_embed_seq`: binary searches find the first own entry and the end of the
  last run; one warp per 64-column slice; batches of 32 entries, lane j loads entry j's whole row slice and the warp
  transposes the batch through shared memory; the next batch's rows and the keys / positions two batches ahead are in
  flight during the adds; run boundaries from one ballot; a batch inside one run is a branch-free chain. Under bf16
  storage each add is one packed `__hadd2_rn` (one HFMA2.BF16 per row): the rounded sum of two bf16 values equals
  bf16(f32 sum), bf16 subnormals included (tested at scale 1e-38). Bitwise equal to the ascending running sum in every
  case. The order of the sum did not change.
- Spill text corrected (below). No code change: the 64-register cap of `__launch_bounds__(512, 2)` / `(256, 4)` is
  what keeps 4 CTAs per SM, which the norm partition assumes.
- `sandwich_tail_bwd` reads `dx_out` with a coherent load (`load_z_cg`), as `rmsnorm_bwd_add` reads `dres`. ew_api.hpp
  states the aliasing contract: dres == dx and dx_out == dx_in (pre-LN also dbranch == dx_in) are allowed; no other
  output may alias an input. ew_api.hpp also states the q_scale contract (any f32 value applied as one multiply; a
  bf16 value under levanter + bf16, checked; the caller chooses it per DESIGN.md §3.4).
- Contraction as DESIGN.md §3.0 says: the kernels of RMSNorm / sandwich tails, QK-norm + RoPE and SwiGLU moved to
  `*_kernels.cuh`. The levanter instantiations are built in `rmsnorm.cu`, `qknorm_rope.cu`, `swiglu.cu`
  (`--fmad=false`). The accurate ones are built in `rmsnorm_acc.cu`, `qknorm_rope_acc.cu`, `swiglu_acc.cu` (nvcc's
  default contraction). Explicit instantiation (`NM_EW_*_LAUNCHERS`, `extern template` in the levanter unit) keeps
  each instantiation in one unit. The build's flags.make lists `--fmad=false` for the 3 levanter units plus embed.cu,
  doc_meta.cu, reduce_cast.cu only.
- T1.3 tables: `t1_3_tables_cpu_info` now checks the library's builders. The host-double tables must equal their
  definition at every entry (cos / sin in double of the f32 angle from the builder's own inv_freq, then f32, then
  bf16; numpy). The libdevice method's inv_freq must be within 4 f32 ulp of host double (measured 1), and its tables
  within 1 bf16 ulp of cos / sin in double of its own f32 angle (measured: cos 0; sin 1 ulp at 4 entries, 2 angles).
  Mismatches are counted as entries and distinct angles. Correction to the S3 section: sin differs from the XLA:CPU
  fixture at 2 of 262,144 entries, which is 1 distinct angle. The default `build_rope_tables` (host double) is still
  not confirmed against the GPU run: `t1_3_tables_gpu` skips (cache/levref/130m/units_a100_vis exists but is empty).

Tests (della-vis1 A100-PCIE-40GB, gcc-toolset 14, CUDA 12.8; fresh configure + build: 0 warnings; full ctest
excluding the bench: 31 of 31 pass, with S3 6 of 6; the shared GPU had other tenants at up to 100 % utilization)

| check | result |
|---|---|
| accurate mode, 70 bf16 outputs (s3_rmsnorm, s3_qknorm_rope, s3_swiglu; all shapes) | rel(k, f64) / single-rounding floor = 1.0000 for every output (bound 1.05); the levanter emulation is 1.35 x (sandwich dx_in) to 2.50 x (SwiGLU dg) the floor, so every gate fails a levanter-rounded kernel; f32 outputs of accurate mode: rel <= 2.3e-7 (bound 1e-6) |
| s3_rmsnorm in_place_residual_gradients | in-place dx_in (post-norm, pre-LN with dbranch == dx_in) and rmsnorm_bwd_add with dres == dx: bitwise equal to separate buffers, both roundings |
| s3_misc doc_meta_many_rows | rows 12,287 / 12,288 / 16,384 at S = 64: every output equals the numpy reference; rows 16,385 rejected |
| s3_embed (bf16_sequential_130m_microbatch, bf16_sequential_chunk_edges, earlier cases) | bitwise equal to the ascending running sum: T = 131,072 D = 512 (Zipf), D = 520 f32 storage (a partial slice), D = 192 chunk-edge cases (one run over 9 chunks, runs ending / starting on a chunk edge, through chunks, max token id, T = 1, 257), bf16 subnormals |
| s3_t1_levref | T1.2 XFAIL (above); T1.3 rotation 100 % bitwise; host-double and libdevice builder checks pass; GPU table gate skips |
| compute-sanitizer memcheck / racecheck / synccheck (`--kernel-name kns=2nm2ew`): Bf16Sequential cases, doc_meta_many_rows, in-place case | 0 errors, 0 hazards, 0 errors |
| s3_bench (levanter, bf16) | within run-to-run noise of the S3 table for the norm, QK and SwiGLU kernels; embed_bwd Bf16Sequential 966 us at 130m (45,466 rows; F32 path 507 us), 777 us at 1_2b (F32 492 us) |

Embedding backward timings (median of 10, sort included, D = 512 unless stated, T = 131,072; shared A100)

| tokens | F32 | Bf16Sequential before | Bf16Sequential now |
|---|---|---|---|
| Zipf 1.1 (longest run 12,626) | 0.51 ms | 14.4 ms (reviewer), 13.4-15.3 ms (here) | 0.97-1.03 ms |
| uniform | 0.53 ms | 0.95 ms | 0.62 ms |
| one run of ~4,096 plus uniform | 0.54 ms | 3.6 ms | 0.80 ms |
| one token (a single run of 131,072) | 0.58 ms | 127 ms | 28 ms (10 ms at D = 64) |

The single long run is the inherent critical path of a sequential sum: 1.25-1.44 us per 32 rows on this GPU when the
kernel is short (one-token sweep at D = 64, T <= 32,768). Kernels longer than about 2 ms measured 2 x slower per row
on the shared GPU (time-slicing with other tenants), so the long-run numbers are upper bounds.

Stack use (`cuobjdump --dump-resource-usage`, bytes per thread; corrects "no spills" in the S3 section)

| kernel | sm_90a | sm_80 |
|---|---|---|
| k_sandwich_bwd<bf16, levanter> (training path) | 8 | 8 |
| k_sandwich_bwd<bf16, accurate> | 48 | 40 |
| k_sandwich_bwd<f32, levanter / accurate> | 80 / 72 | 56 / 64 |
| k_qknorm_rope_bwd<bf16, levanter> (training path) | 0 | 8 |
| k_qknorm_rope_bwd<f32, ...> | 8-24 | 24-40 |
| k_qknorm_rope_fwd<f32, ...> | 24 | 24 |
| k_rmsnorm_bwd_add<f32, ...>, k_sandwich_fwd<f32, ...> | 0 | 16, 8 |
| k_embed_chunks<f32> | 24 | 24 |
| every other S3 kernel (k_embed_seq included) | 0 | 0 |

Commands
```bash
ssh della-vis1; cd $ROOT/project/marin.cpp; module load gcc-toolset/14 cudatoolkit/12.8; export NANOMARIN_ROOT=$ROOT
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++
nice -n 19 cmake --build build -j 7 && (cd build && nice -n 19 ctest -R 's3_' -E s3_bench --output-on-failure)
python tests/s3/test_embed.py --lib build/libmarincpp_kernels.so --src . bf16_sequential_130m_microbatch
compute-sanitizer --tool racecheck --kernel-name kns=2nm2ew \
  python tests/s3/test_embed.py --lib build/libmarincpp_kernels.so --src . bf16_sequential_chunk_edges
```

For the owners of the spec, DESIGN.md, the S0 headers and tools/levref (S3 does not edit them)
1. kaiyue_baseline.md T1.2: amend with the cancellation rule above, or record T1.2 as failing (4 of 524,288 k_norm dx
   elements on the XLA:CPU fixture).
2. include/marincpp/kernels/elementwise.hpp (embedding backward, lines 53-64) and DESIGN.md §3.2 describe
   `k_embed_bwd_runs` and one CTA per run with `run_start` / `n_runs`. The code is the chunked segment sum (F32) and
   the per-chunk sequential walk (Bf16Sequential), and leaves `run_start` / `n_runs` null. §3.2's "~0.2 ms at 131,072
   tokens" is 0.51 ms (F32) and ~1.0 ms (Bf16Sequential) on the A100, sort included.
3. elementwise.hpp `QkNormRopeFwd` comment: "Accurate: ... q_scale is still the bf16 constant (class M), folded into
   the rounding". The kernels apply the q_scale passed as one f32 multiply. Under accurate, DESIGN.md §3.4 gives FA3
   the scale, so the caller passes q_scale = 1 to qknorm_rope_fwd and the whole scale to dq_post_qknorm_rope_bwd. The
   value under the f32 policy (f32 1/sqrt(hd) or bf16(1/sqrt(hd)), class M) is for S0 / DESIGN to state.
4. tools/levref: a units fixture with non-unit gains (every gain of units_cpu_vis is 1.0, so the rounding points
   out = bf16(g y) and a = bf16(g dy) are compared only with the emulation, never with levanter), and the GPU-made
   units fixture (units_a100_vis has an empty norm/ directory).

Known gaps
- T1.2 is not passed (above). The levanter rounding points follow levanter's jaxpr; XLA:GPU on the A100 drops some of
  them (SwiGLU; the q cos term of QK). Until T1.0 on the H100 decides, the bitwise levanter-mode tests show agreement
  with the emulation of the jaxpr only, and no levanter-mode parity claim is made.
- H100 items for the tools/devq profile: the cost of the 8 B stack in k_sandwich_bwd<bf16, levanter>; widths 768
  (300m) and 1536, whose 192-thread CTAs give 768 threads per SM (A100: 3-5 % below D = 512 / 1024, reviewer's
  numbers). Rounding the CTA up to 384 threads does not help: 4 x 384 threads x 64 registers = 98,304 > 65,536
  registers per SM on both GPUs, so only 2 such CTAs fit, again 768 threads. A fix needs more bytes in flight per
  thread (two rows per row group), not a larger CTA.
- In a full ctest run on the shared GPU, s3_embed can exhaust its 4 out-of-memory retries while other tenants hold
  the memory (reviewer's run); run alone it passes.

## Docs: requests of S0-S3 applied (2026-10-24)

Built (DESIGN.md, docs/spec, include/marincpp only; no source file changed)
- S0: DESIGN §2.5 states that ZeRO-2 for vocab-sized gradients is not built (`auto` = off, `on` refused, no planner or
  budget terms). §13.2 names the binary `marincpp-train` (`marincpp <command>` in DESIGN means it) and lists the real
  module directories. reduce.hpp and DESIGN §11: the reduction order depends on the shape and the SM count, so a step
  is bitwise repeatable on one GPU type. readout.kind (§9.1) and §9.2 already carried the S0 text.
- S1 (items 1-5): `segs_per_seq.u16`, meta format 2 and the `.tmp`-then-rename commit (data.hpp, DESIGN §5.1). Repeat =
  levanter's slice-then-restart, `sigma_N((blk*B + pi_blk[j mod B]) mod subset)`, reshuffle needs index_base 0 (data.hpp,
  DESIGN §5.5). data.md §2.4, §3.1 item 1, §9 item 2 (and kaiyue_baseline.md Part 3): BOS and EOS never occur inside
  text (whole-corpus count); the old "once per window" came from documents cut by the window edges. DESIGN §13.1: no
  vendored TOML or JSON library (own readers in src/config and src/data); xxHash comes with S10.
  data.hpp: `BatchLoader` gets an inline destructor that calls `stop()` (idempotent, returns at once when never
  started), so destroying a started loader no longer calls std::terminate; `BatchLoader` and `TokenFile` are not
  copyable. Decision instead of the suggested `~TokenFile()` / `int fd_`: a TokenFile mapping lives until the process
  exits (one per dataset per run; src/data's fd table and `shared_token_file` rely on mappings never moving), stated in
  data.hpp and DESIGN §5.2. Phase gets no `mix_block` field yet; data.hpp states that plans accept only 2048 until a
  phase needs another value. The pimpl is left to src/data's owner (the address-keyed table works).
- S2 (items 1-6): optim.md §5.3 rule 2 (`__nv_powf`, rcp.approx gap, fixture is the record), §6.1 (`c * f32(1/3967)`;
  k = 4958 gives the same bits either way, checked in numpy f32), §6.2 (folded constant `c * f32(f32(pi) * f32(1/T_d))`,
  `__nv_cosf`, `**1.0` folded, warmup `c * f32(1/1000)`, literal-formula mismatch counts), "Lookup table" (the fixture
  replaces the logged table and the hole formulas), §10 item 7 resolved; the same text in kaiyue_baseline.md Part 2.
  T0.7 row: "equal to optim.md §6 as XLA:GPU computes it (marincpp build_schedules) and inside their logged neighbours".
  DESIGN §4.4 (XLA:GPU order, `n_ulp_diffs = -1`, generator tests/s2/schedule_fixture.py), §13.2 (src/sched), §13.3 S2
  row. optim.hpp `build_schedules` comment and `n_ulp_diffs = -1`; capi.h `config_toml` is a file path, `*n_ulp_diffs`
  can be -1; knobs.def help text of optim.schedule_fixture names the generator (help text only; defaults unchanged).
- S3 (items 1-3): elementwise.hpp embedding backward = `k_embed_chunks` + `k_embed_merge` (F32) and `k_embed_seq`
  (Bf16Sequential), `touched_rows` as the lazy mask, the measured A100 times, `run_start` / `n_runs` marked unused (null).
  DESIGN §3.2 and §4.5 likewise (0.51 ms F32, 0.62-1.0 ms Bf16Sequential at 131,072 tokens, not ~0.2 ms).
  `QkNormRopeFwd` q_scale comment: one multiply by the value passed; under accurate FA3 takes the scale (q_scale = 1
  here, the whole scale in dq_post). DESIGN §3.4 states the value: `bf16(1/sqrt(hd))` under the bf16 policy (both
  roundings), `f32(1/sqrt(hd))` under the f32 policy (levanter's Python-float constant in the compute dtype).
- T1.2 (controller decision): the T1.2 row of kaiyue_baseline.md passes an element of RMSNorm `dx` beyond 1 bf16 ulp
  when `|nano - lev| <= 4 x 2^-24 x (|a r| + |c 2x|)`, addends in f64 from the bf16 inputs with `a = bf16(g dy)`.
  Reason in the row: catastrophic cancellation in the k_norm dx sum, so any summation order of the row sum other than
  XLA's moves dx by a few f32 ulps of the addends; 4 of 524,288 k_norm dx elements of the 130m fixture. New §C12 row
  records it. targets.md and numerics.md do not restate T1.2 (checked). tests/s3 may switch the T1.2 XFAIL to PASS
  under the amended rule.

Tests (della-vis1, gcc-toolset 14, CUDA 12.8, fresh configure + build in a directory outside the repo)

| check | result |
|---|---|
| build (every target, incl. nm_s0_headers: each public header compiles on its own; capi.h as C11) | 0 warnings, 0 errors |
| ctest -E s3_bench | 31/31 pass (s1_loader exercises the new BatchLoader destructor; s3_t1_levref unchanged) |
| `python3 tools/scrub.py docs/DESIGN.md docs/spec include` | clean |

H100 gates: none (no kernel or source change).

Open for other owners
- src/data (S1's owner): optional pimpl in place of the address-keyed loader table; a `mix_block` Phase field when a
  plan needs a block other than 2048 (header change on request).
- src/kernels/elementwise (S3's owner): stop setting `EmbedBwdWorkspace::run_start` / `n_runs`, then ask for their
  removal from elementwise.hpp. tests/s3: T1.2 XFAIL -> PASS under the amended rule.
- tools/levref: S3 item 4 (a units fixture with non-unit gains; the GPU-made units fixture).
- src/main.cpp (trainer stage): pass the dataset meta to `choose_plan`, `resolve` and `exec_shape` (S1 gap).

## Decisions for S4-S6 (2026-10-25, from the H100 benches)

Binding for the S4 (GEMM + FA3 attention), S5 (fused CE) and S6 (optimizer) builders. Numbers: `docs/bench/RESULTS.md`;
reasons: DESIGN.md Appendix A part B and "Open decisions". A builder who cannot meet a bar reports it with the
measurement; the bar does not move without a DESIGN.md change.

All stages
1. Determinism stays on by default (O5). Every kernel has a deterministic path with no float atomics; a faster
   nondeterministic path may exist only behind `exec.deterministic=false`.
2. Time every kernel bar below on H100 through devq (1 GPU, 5 warm-up and 20 timed calls, median, CUDA events) and
   record the table in your BUILD_LOG section next to the bar.
3. `resolve()` (config owner) maps `optim.ns_precision=auto` to `tf32` and `exec.ce_logits=auto` to `bf16`; parity
   runs no longer refuse `auto` for want of T1.0.

S4: GEMMs
4. GemmCache searches cuBLASLt algorithms (deterministic split-K schemes only) and CUTLASS SM90 kernels per shape, as
   DESIGN §3.7 says. Bar per shape and orientation: at least the better of cuBLAS and XLA in RESULTS.md §2 (e.g. fwd
   M=131072 K=1024 N=3584: 760 TFLOP/s; fwd M=131072 K=512 N=128256: 671). Report every shape that misses it.
5. Build with CUDA 12.8 and keep the 13.0 build green. Time cuBLASLt 12.8 against 13.0 on the RESULTS.md §2 shapes and
   report it (input to O8).

S4: attention
6. FA3 (vendored `767386d`) is the backend. Build both fixed backward schedulers (`persistent`, `doc_causal`) with
   deterministic on and off; never ship the padded non-persistent varlen grid in a training path (E6: +25-33%).
7. Bar per scheduler, deterministic off, fwd+bwd per layer per 32 x 4096 tokens on the real segment rows: 130m 2.51 ms,
   300m 3.75 ms, 520m 5.68 ms, 1.2B 9.21 ms (fwd 0.66 / 0.95 / 1.27 / 1.73 ms). Then run E1b and apply the O1 rule:
   the faster scheduler summed over 130m, 520m, 1.2B; ties within 3% go to `doc_causal`.
8. Do not build the C++ `CudnnSdpa` backend unless E1b's chosen FA3 sum exceeds 20.56 ms (cuDNN THD 19.58 ms x 1.05).
   E1b times cuDNN with the existing Python bench. FA4 is not adopted (a tie).
9. Before S7 depends on one step graph: a 4-GPU H100 test (one process, a thread per GPU, `ncclCommInitAll`) that
   captures FA3 fwd+bwd and an NCCL reduce in one graph and replays it 100 times (E6's PyTorch graph arms crashed in
   this setting).

S5: fused CE
10. Gradient in the forward; logits store bf16; dW accumulated in f32 across chunks and microbatches; lse and the
    softmax statistics from the f32 accumulators. No recompute variant (dropped). The f32 store stays only as
    `exec.ce_logits=f32`, the fallback if the bf16 store fails T1.5.
11. Build B1 first, then B2, then B1.5; time all three at T = 16,384 and 131,072, D = 512 / 768 / 1024 / 2048,
    V = 128,256. The stage per D is the fastest one that passes T1.5 (O2).
12. Bar for B1 (Liger with f32 dW, the same structure): no slower than 13.57 ms per pls head (16,384 x 512 x 128,256)
    and 109.0 / 145.2 / 179.3 / 320.0 ms at 131,072 rows, D = 512 / 768 / 1024 / 2048. Targets: B1 12.8 ms, B2
    9.1-10.7 ms per pls head (GEMM floor 9.14 ms). The row kernel rewrites the store in place.
13. T1.5 oracle rule against levanter's own error (CE dx 1.72e-3, dW 1.73-1.76e-3, per-token loss 6.2e-8 at 130m).

S6: optimizer
14. Newton-Schulz runs in TF32 (O3 decided). TF32 operands are rounded to nearest, as XLA's are: on the 512^3 f32 GEMM
    probe, rel to the f64 product of TF32-RNE-rounded operands must be <= 1e-5 (levanter 1.2e-6 to 8.5e-6). 3xTF32 and
    true f32 stay for the f32 policy and A/B tests.
15. T1.6 and T1.7 use the oracle rule with levanter at HIGHEST as the oracle (levanter TF32 error 1.4e-4 to 1.0e-3).

## S6: optimizer (2026-10-25)

Built (src/kernels/optim and src/optim, one module `nm_optim`; the update kernels build with `--fmad=false` and spell
every formula with `__fmul_rn` / `__fadd_rn` / `__fdiv_rn` / `__fsqrt_rn`; all portable: sm_90a + sm_80)
- `src/optim/engine.hpp` `opt::Engine`: one rank's optimizer step from groups as data (`EngGroup`: rule, lr table, f64
  b1 / b2 / eps / momentum / max_grad_norm / weight decay, clip group) and the tensors it owns (`EngTensor`: [rows,
  cols] view, `base_row`, p32 / p16 / g / s0 / s1, lns grad and update scales, row partition, MuonH matrices with an
  optional row map for fused buffers, owned row ranges, optional `u_out`). Phases, each graph-capturable after
  `prepare()` (no allocation, no host sync, k read from device memory): `record_sumsq` (or per tensor:
  `record_sumsq_tensor` + `record_clip_sums`, for the per-bucket schedule of DESIGN.md §4.1) -> all-reduce
  `clip_sumsq` -> `record_update` (clip state, Adam, MuonH per NS chunk, AdamH pass 1) -> all-reduce `slot_pa` ->
  `record_adamh_w` -> all-reduce `slot_w` -> `record_finish` (AdamH pass 3, optional k++). `record_step` runs them with
  an all-reduce callback (skipped unless `sharded`). `baseline_groups(Config)` = muonh / adamh (main lr) / adam
  (adam_lr), no weight decay; `group_dev` makes the f32 constants (optim.md §5.3 rule 1).
- ZeRO-1 (DESIGN.md §6.1): `plan_ownership_specs(ParamSpec list, groups, ranks, zero1)` = every MuonH leaf matrix
  whole on one rank, vocab-sized tensors row-split over all ranks, other tensors whole, LPT on estimated H100 seconds
  (NS at 400 TFLOP/s + elementwise bytes at 3 TB/s); `zero1 = off` or 1 rank: owner -1 (everyone).
  `engine_layout` turns specs + ownership into one rank's EngTensor list (leaf row ranges -> row maps).
- Kernels (`src/kernels/optim/optim_dev.hpp`, internal): per-slice (16,384 floats) sums of squares in f64 of f32
  squares, fixed-order segment sums (one f32 rounding at use), `k_clip_state` (`!(gn < max)`, `(g / gn) * max` applied
  while reading g); Adam in one pass; AdamH in three (moments + ||p||^2, ||a||^2; ||w||^2; update); MuonH per 64 x 64
  tile: Nesterov momentum into the NS buffer (wide orientation, shared-memory transpose for OUT > IN), `X / (||X|| +
  1e-5)`, the three hyperball passes; float4 paths with a scalar fallback. Newton-Schulz batched by oriented shape
  per chunk (`EngOptions::ns_chunk_bytes`, default 1 GiB of X, X', A, cA, B): `A = X X^T`; `cA = c * A` (f32
  elementwise, as XLA's operand); `B = cA A + b A`; `X' = B X + a X` (b A and a X in the cuBLASLt epilogue, beta).
  Two options, both off by default after the H100 bench: symmetric products (`EngOptions::ns_symmetric`; capi.h
  `symmetric` bit 0): A and B are symmetric, so their GEMMs compute the blocks 11, 12, 22 of a 2 x 2 split (m1 = m / 2
  rounded up to 32) and `k_ns_mirror` copies the transpose of block 12 into block 21 (3/4 of the X X^T and cA A
  FLOPs; not with 3xTF32; m < 64 runs full); fold c (`ns_fold_c`; bit 1): `B = c (A A) + b A` with alpha = c, no cA
  pass (the TF32 operand is A instead of c A).
- `src/kernels/optim/ns_gemm.{hpp,cpp}` `NsGemm`: cuBLASLt strided-batched f32 GEMMs, TF32
  (`CUBLAS_COMPUTE_32F_FAST_TF32`), true f32 (`_PEDANTIC`), 3xTF32 (operands split into tf32_rna(x) and the f32
  remainder, small x big + big x small + big x big). Algorithm per shape: candidates without split-K (reduction scheme
  NONE) from the heuristic of a fixed batch-2 layout, kept if valid for the real layout, first one or the fastest
  (timed); `save()` / `load()` for a resumed run. The heuristic choice is a function of the matrix shape alone, so a
  matrix runs the same kernel in any batch (gate: s6_nsalgo, review fixes below). (With per-batch heuristics, batch 1
  picked split-K and batch 6 another tile; split-K validity depends on the batch through its workspace.) What training
  needs, all ranks agreeing, comes from the all-reduced sums; 1 rank == 4 ranks bitwise (s6_zero1) holds on its data
  but is not guaranteed in general: the f64 partials of row-split tensors add in another order, and the one f32
  rounding of a sum can flip at a rounding boundary.
- C ABI: capi.h `nm_newton_schulz`, `nm_muonh_update`, `nm_adamh_update`, `nm_adam_update` (one-tensor engines, test
  path); `src/optim/capi_optim.h`: `nm_opt_create / set_tables / prepare / phase / step / buffers / destroy` (the engine
  over a tensor list; sharded engines driven by phases) and `nm_ns_gemm` (the probe).

Operation order. optim.md §3-§5, with one change read from the optimized HLO of levanter's 130m H100 train step
(cache/levref/130m/t1_precision): XLA's algebraic simplifier rewrites the Adam direction `(mu / bc1) / (sqrt(nu / bc2)
+ eps)` into `mu / (bc1 * (sqrt(nu / bc2) + eps))` for lm_head's AdamH and every Adam leaf (a `multiply` with no jaxpr
frame, `op_name="multiply.4x"`); the kernels compute that order (Adam updates moved closer to levanter: max rel 8.6e-8
-> 8.4e-8 at 130m case A). The hyperball keeps `(w / ||w||) * n - p` (the HLO divides, then multiplies), the momentum
`mu*buf + g` twice, the moments `(1-b)*x + b*m`, all without contraction; whether XLA's PTX contracts any of them is
not known (no PTX in the dump); every tolerance below holds either way.

Gates (della-vis1 A100-PCIE-40GB, shared: fresh full build, 0 warnings, `ctest -R s6_ -LE bench` 6 / 6 pass; H100:
devq t49, t50, t52, t59, t61 on snapshots of the S6 build in bench/s6/snap, 1 GPU of a della H100 node; t61 = the
final sources: every gate below passes there, T1.7 with the same 3 pass / 2 specfail; superseded by the review
fixes below)

| test | checks | A100 | H100 |
|---|---|---|---|
| s6_ownership | 130m and 1.2B fused layouts at 2 / 4 / 8 ranks: every row owned once, MuonH leaves whole, vocab tensors on every rank, LPT max/mean load <= 1.0028; zero1 = off; f32 constants as optim.md §5.3 bit patterns and == `optim_constants` | pass | pass |
| s6_zero1 | 4 emulated sharded ranks (phases in lockstep, sums added between them) vs 1 rank on 130m shapes (2 layers, V 16,384, 64 MiB NS chunks): p, buf, mu, nu | bitwise | bitwise |
| s6_graph | the step captured (global capture mode) and replayed 3 x == 3 eager steps (per-tensor clip sums in reverse order) | bitwise | bitwise |
| s6_rowpart | frozen block (no update, no norm), lr-multiplier table, own AdamH norm scope vs a host f64 reference; scalar fallback vs float4 path | Adam within 0.25 of 1e-6 max(\|u\|, lr) + ulp; AdamH 7.5e-7; paths bitwise | pass |
| s6_ns probe | 512^3 TF32 GEMM: rel to the f64 product of TF32-RNE / RNA / truncated operands | 7.6e-6 / 1.7e-6 / 8.7e-4 (A100 rounds RNA) | 1.2e-6 / 7.5e-6 / 8.7e-4 (H100 rounds RNE) |
| s6_ns ns_ref | NS on random matrices (square, wide, tall, 100x37, 64x300, 33x65; batched) vs f64 | f32 <= 3.2e-6, 3xTF32 <= 4.1e-5, TF32 0.79-1.01 x the error of an f64 emulation with TF32-RNA operands; repeat bitwise | pass |
| s6_ns t1_6 | T1.6, below | pass | pass |
| s6_t1_7 | T1.7, below | 3 pass, 2 specfail (NOT PASSED) | 3 pass, 2 specfail (NOT PASSED) |

T1.6 (levanter's u of update 0 per leaf; rule: rel(nano TF32, levanter HIGHEST) <= 1.25 x rel(levanter DEFAULT,
HIGHEST)), H100:

| leaf | levanter DEFAULT | nano TF32 (ratio) | nano 3xTF32 | nano f32 |
|---|---|---|---|---|
| 130m q / k / v / o (512 x 512) | 9.7e-4 / 1.0e-3 / 2.6e-4 / 3.5e-4 | 9.8e-4 (1.012) / 1.0e-3 (1.003) / 2.6e-4 (1.001) / 3.5e-4 (1.000) | 3.5e-6 .. 4.8e-6 | 2.1e-6 .. 6.6e-6 |
| 130m gate / up (1792 x 512), down (512 x 1792) | 4.8e-4 / 4.4e-4 / 2.6e-4 | 0.994 / 1.001 / 1.004 | 7.1e-6 .. 7.8e-6 | 2.9e-6 .. 3.0e-6 |
| 1_2b q, o (2048^2); k, v (1024 x 2048) | 5.9e-4, 1.4e-4; 8.2e-4, 1.6e-4 | 0.998, 1.000; 0.995, 0.998 | 8.6e-6 .. 1.6e-5 | 1.6e-6 .. 3.0e-6 |
| 1_2b gate, up (7168 x 2048), down (2048 x 7168) | 2.1e-4, 1.8e-4, 1.4e-4 | 0.997, 0.996, 1.000 | 1.7e-5 .. 2.0e-5 | 1.5e-6 .. 3.2e-6 |

TF32 passes at every leaf (ratio 0.994-1.012 H100, 0.997-1.032 A100). Two TF32 computations are independent samples
of the same error, not near-copies: nano vs levanter DEFAULT is as far apart as either is from HIGHEST (the iteration
amplifies single operand-rounding flips), so the ratio sits at 1, not below. True f32 holds the f32-dot bound (<= 6.6e-6
<= 1e-5). 3xTF32 does not at 1_2b (1.6e-5 to 2.0e-5 on the H100; A100: 3.5e-5, and 1.2e-5 at 130m): its small parts
are rounded to TF32 again and the tensor cores accumulate in f32 (probe: 1.2e-6 vs 4.1e-7 for true f32). By DESIGN.md
§4.2 the f32 policy therefore uses true f32 (`optim.ns_precision = f32` -> `CUBLAS_COMPUTE_32F_PEDANTIC`); 3xTF32 stays
an explicit A/B choice.

T1.7 (one update on G0; cache/levref/<size>/step0_f32; case C inputs exported from the step-4958 checkpoint by
tests/s6/export_case_c.py into cache/s6/caseC_in, all 16 parameters equal caseC/pin's sha256; params after the update
rebuilt as f32(p_in + u) and checked against the fixture's sha256). H100 and A100 give the same numbers:

| case | k, lr | group norms, clip | MuonH buf, mu, nu | Adam u | MuonH u: rel to HIGHEST (levanter DEFAULT) | AdamH u | norms kept | verdict |
|---|---|---|---|---|---|---|---|---|
| 130m A | 0, 0.02 | rel <= 6.5e-8, same clip | <= 1.2e-7 | <= 8.4e-8 | 2.4e-4..8.3e-4 (ratio <= 1.006) | 2.7e-6 | <= 9.0e-8 | pass |
| 130m B | 1, 0.02 | bitwise, same | <= 7.5e-8 | <= 7.5e-8 | 2.4e-4..8.3e-4 (ratio <= 1.001) | 2.6e-6 | <= 8.6e-8 | pass |
| 130m C | 4958, 5.04e-6 | bitwise, same | <= 8.5e-8 | <= 7.0e-8 | 1.1e-2..2.2e-2 (levanter 2.7e-3..8.5e-3) | 9.3e-3 | <= 8.3e-8 | specfail |
| 300m A | 0, lr 0 | <= 1.1e-7, same | <= 3.3e-7 | 0 (exact) | (rounding noise at lr 0) | | <= 8.2e-11; params within 1 ulp | pass |
| 300m B | 1, 1.0e-5 | <= 6.8e-8, same | <= 4.4e-8 | <= 7.1e-8 | 5.5e-3..7.7e-3 (levanter 1.2e-3..3.4e-3) | 3.3e-3 | <= 6.9e-8 | specfail |

f32 policy (true-f32 NS) MuonH updates at 130m A / B: rel to HIGHEST <= 6.0e-6 (bound 1e-5). Two TF32 runs: bitwise.
The symmetric-block option gives the same verdicts (item 15 ratio <= 1.004 at A / B; f64 ratio 0.97-1.00 at C, 300m B).

Why C and 300m B miss item 15, and why that is not a defect of the port: at lr ~1e-5 the update `target - p` is about
lr x |p| (5e-6 relative), while the f32 renormalization `(w / ||w||) * n` leaves about one f32 ulp of p per element
(6e-8 relative), i.e. ~1 % of the update. Levanter's DEFAULT and HIGHEST programs share that rounding (same reduction
order of n and ||w||), so their distance (the denominator of item 15) is only the TF32 part; marincpp's norms round
independently. The test therefore also computes the spec's own oracle (kaiyue_baseline.md "Conventions": the whole
update chain in f64 from the same f32 inputs and constants) and reports item 15 misses as SPECFAIL only where the oracle
rule holds: rel(nano, f64) / rel(levanter, f64) for MuonH and AdamH updates is 0.71-0.97 (case C) and 0.61-1.09
(300m B) with TF32, and the same with true f32 against levanter HIGHEST: marincpp is as close to the exact update as
levanter or closer. At lr 0.02 (cases A, B) the same ratio is 0.995-1.005 for MuonH (TF32; true f32 vs levanter
HIGHEST 0.82-1.09) and 1.10 / 1.35 for lm_head (AdamH; 3.1e-6 vs 2.3e-6 at case B: levanter's f32 sums of ||p||^2
and ||w||^2 share one reduction order, so the rounding of n / ||w|| partly cancels; marincpp's correctly rounded
norms do not; item 15 passes there).

Bench (H100, `tests/s6/bench_s6.py`: one step on synthetic data in levanter's layout, median of 20 after 5 warm-up,
CUDA events; "rank 0/4" = rank 0's ZeRO-1 share of 4 without the all-reduces; NS GEMM choice heuristic / timed)

| size | TF32, one GPU holds all | TF32 rank 0/4 | TF32 symmetric rank 0/4 | true f32 rank 0/4 | DESIGN.md §4.2 budget (rank 0/4) |
|---|---|---|---|---|---|
| 130m | 5.22 / 4.53 | 1.55 / 1.38 | 1.84 / 1.68 | 2.70 |  |
| 300m | 15.08 / 13.87 | 4.19 / 3.95 | 4.87 / 4.52 | 14.15 |  |
| 520m | 48.91 / 43.06 | 12.58 / 11.24 | 14.04 / 11.77 | 47.21 |  |
| 1_2b | 183.30 / 178.01 | 46.03 / 44.15 | 45.66 / 43.18 | 223.97 | 43 (TF32), 214 (true f32) |

ms per step, heuristic / timed NS GEMM choice. 1.2B rank 0/4 with the defaults: 46.0 ms against the 43 ms budget
(+7 %; the timed choice: 44.2 ms, +3 %); true f32 224 ms against 214 (+5 %). Where it goes (nsys, 1.2B rank 0/4,
TF32): NS GEMMs 35.9 ms (cuBLASLt sm90 TF32 kernels, 10.7 TFLOP per step at ~300 TFLOP/s; the large shapes at ~390),
MuonH elementwise 7.5 ms (momentum 1.9, hyperball 1.0 + 1.4 + 1.3, cA = c A 1.2, normalize 0.6), Adam / AdamH / clip
sums 2.3 ms (at ~95 % of HBM bandwidth after the float4 paths; first version 48.9 ms in total: 64 x 64 tiles, a 2-D
normalize grid without a 64-bit division per element and float4 paths took 3 ms off). The two NS options lose on the
H100: symmetric blocks are slower at 130m-520m (130m rank 0/4: 1.85 vs 1.55 ms) and 1 % faster at 1.2B (the 1024 x
1024 block GEMMs fill 1.45 waves of 132 SMs, against 5.8 for one 2048 x 2048 GEMM); folding c saves the 1.2 ms cA pass
but T1.6's ratio rises to 1.01-1.11 (0.99-1.03 with the explicit cA operand), so both stay off.

Requests (headers and docs outside src/optim, src/kernels/optim, tests/s6)
1. optim.hpp `GroupSpec`: b1, b2, eps, momentum as double. f32 fields lose the decimal that rule 1 of optim.md §5.3
   needs (`f32(1 - 0.9)` from the f64 difference is 0x3DCCCCCD; from f32(0.9) it is 0x3DCCCCD0). The engine keeps f64
   (`EngGroup`).
2. optim.hpp `plan_ownership`, `build_optimizer_plan`, `declare_optimizer`, `record_optimizer_step` take Model /
   DeviceContext / MemoryPlan, which S7 builds. S6 implements the parts on `std::vector<ParamSpec>`
   (`opt::plan_ownership_specs`, `opt::engine_layout`) and the step as `opt::Engine`; proposal: those four
   declarations become thin S7 wrappers over the engine (or take `params()` directly).
3. kernels/optim_kernels.hpp: its kernel list (MultiTensor, MuonMatrices, per-kernel launchers with ClipState) is not
   what was built (src/kernels/optim/optim_dev.hpp: TensorTab / SliceTab / MatTab / RowPartTab, f64 partials, the
   engine's phases). Proposal: reduce it to a pointer to src/optim/engine.hpp.
4. capi.h: `nm_newton_schulz` needs `workspace = NULL` (it allocates); its `symmetric` argument carries two flags
   (1: the 2 x 2 block form, 2: c folded into alpha); the block arrays of `nm_adamh_update` / `nm_adam_update` are
   device pointers. capi_optim.h `nm_opt_create` takes the same options as `ns_precision` + 4 / + 8.
5. optim.md §5.1 / §5.2 (and kaiyue_baseline.md Part 2): state the H100 program's `a = mu / (bc1 * (sqrt(nu / bc2) +
   eps))` next to the jaxpr form.
6. DESIGN.md §4.2 and O3 row: record T1.6's 3xTF32 result (outside 1e-5 at 1_2b on the H100) and that the f32 policy
   runs true f32; `resolve()`: `optim.ns_precision = f32` means true f32 (3xtf32 an explicit value).
7. kaiyue_baseline.md T1.7 row / decision item 15: at small lr the update is dominated by the f32 renormalization's
   rounding, which levanter DEFAULT and HIGHEST share; proposal: MuonH and AdamH updates use the oracle rule against
   the f64 oracle of the whole update chain (as the Conventions define it), with levanter DEFAULT (TF32) and levanter
   HIGHEST (f32 policy) as "lev"; item 15's form stays as a report. Until then C and 300m B are NOT PASSED
   (s6_t1_7_small_lr: SPECFAIL with both numbers, exit 1).

Commands
```bash
ssh della-vis1; cd $ROOT/project/marin.cpp; module load gcc-toolset/14 cudatoolkit/12.8; export NANOMARIN_ROOT=$ROOT
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++ && nice -n 19 cmake --build build -j 8
$ROOT/project/marin/.venv/bin/python tests/s6/export_case_c.py --src .     # once: T1.7 case C inputs (tensorstore)
(cd build && nice -n 19 ctest -R 's6_' -LE bench --output-on-failure)
python tests/s6/test_t1_7.py --lib build/libmarincpp_kernels.so --src . --only 130m_C --json out.json
```

Known gaps
- Integration is S7's: reduce-to-owner and the bf16 all-gather (comm), the all-reduces between the phases (a callback
  here), the per-clip-group start of DESIGN.md §4.1 (the phases are global; per-tensor clip sums exist), the metrics
  ring and `step_advance`'s lr record, the memory plan entry for `Engine::workspace_bytes()`.
- The MuonH tile kernels (momentum, hyperball) run at ~65 % of HBM bandwidth on the H100 (scalar 4-byte accesses
  per thread, 32 x 32 sub-tiles with a shared-memory transpose); Adam, AdamH and the clip sums are at ~95 %.
- Lazy (touched-row) Adam updates and decoupled weight decay: gated since the review fixes (baseline: neither); lazy
  AdamH is refused.
- XLA's FMA contraction inside the optimizer fusions is unverified (no PTX in the T1.0 dump).
- T1.7 exists for 130m and 300m only (no 520m / 1_2b fixtures; spec).
- A timed NS GEMM choice (`timed_gemm`) can differ per batch and so per rank; the default (heuristic, shape-only) is
  rank-invariant; `save_gemm_algos()` keeps either across a resume.

## S6 review fixes (2026-10-25)

Blocking findings (both real; fixed)
1. C ABI constants (optim.md §5.3 rule 1). capi.h passes b1 / b2 / eps (and MuonH mu) as float; `nm_adam_update` /
   `nm_adamh_update` stored double(f32(0.9)), so `group_dev` made f32(1 - 0.89999997615814209) = 0x3DCCCCD0 and
   0x3CA3D700 instead of 0x3DCCCCCD / 0x3CA3D70A (one step at g = 0.5: mu 0x3d4cccd0, nu 0x3ba3d700). Fix:
   `decimal_of` (src/optim/capi_optim.cpp) recovers the decimal of each f32 argument (shortest round-trip digits by
   `std::to_chars`, then `strtod`) before the engine forms 1 - b. Gate s6_rowpart.capi_one_tensor_bits: one step of
   each one-tensor entry point; mu / nu bits 0x3d4ccccd / 0x3ba3d70a, Adam's p bitwise against host f32 in the kernels'
   order, the MuonH buffer bitwise, AdamH and MuonH norms kept to 1e-6. Mutation (no `decimal_of`): the case fails with
   the reviewer's bits.
2. Rank-invariance gate. New s6_nsalgo: every oriented NS shape of the four sizes (10 shapes), TF32 and true f32, full
   and symmetric products, planned at batch 1, 2, 3, 6 and the engine's 1 GiB chunk capacity: the chosen cuBLASLt
   algorithm configuration of every GEMM of the iteration equals batch 1's, reduction scheme NONE (new
   `NsGemm::algo_config`: the nine CUBLASLT_ALGO_CONFIG_* attributes, kernel id, tile, stages, split-K, reduction,
   swizzle, custom option, inner shape, cluster shape; `reduction_scheme`; `batch_fallback`). 200 plans, 0 fallbacks
   to the per-layout heuristic, on the A100 and the H100. Mutation (`make_layouts(e->d, e->d.batch, two)`): fails on
   the A100 (e.g. 1_2b 1024 x 2048: every batch other than 1 picks another configuration), where s6_zero1 and s6_ns
   still passed. A first version compared the opaque algorithm bytes, as the review proposed; it failed on the H100
   (t76: 130m 512 x 512 TF32, batch 1 vs 2 / 3 / 6 / 204) with the same configuration: on the H100 the first heuristic
   call of a process leaves non-zero garbage in bits 16..31 of algo.data[4], outside every configuration attribute
   (bench/s6/dbg, t77 / t78: three processes, same kernel, word 0x0001d845bf330001 in the first call and
   0x0001d84500000001 after). The bytes are not a rank-invariant key; the configuration is.

Minor findings
3. T1.7 at small lr: the binding rule decides the exit code. s6_t1_7 (130m A, B; 300m A) passes; the new ctest
   s6_t1_7_small_lr (130m C, 300m B) prints SPECFAIL with both numbers and exits 1 until the owner rules on request 7.
   (S3's T1.2 prints NOT PASSED with exit 0; here the ctest status itself shows the failure, so `ctest -R s6_ -LE bench`
   is 8 / 9 by design.) Case C is mandatory when the 130m fixture exists (a missing caseC directory is a FAIL, no
   longer a SKIP): case A has little power against momentum-form bugs (buf = 0), B and C carry it.
4. One bias-correction table pair: the engine refuses Adam / AdamH groups whose b1 / b2 differ (s6_rowpart.refusals).
5. Table bounds: `optk::Tables::n_steps`; `Engine::set_tables` and `nm_opt_set_tables` take N. Every kernel that reads
   k (`cur_step`, `k_step_advance`) traps when k is outside [0, N), so the stream reports a launch failure instead of
   reading past the lr and bias-correction tables (one compare per thread). Gate s6_bounds: a step at k = N - 1 runs
   and advances k to N; the next step fails with cudaErrorLaunchFailure.
6. Lazy AdamH is refused (the whole-tensor norms are not defined over touched rows only). Lazy Adam rows and decoupled
   weight decay now have a gate: s6_rowpart.lazy_adam_weight_decay (untouched rows keep p, mu, nu bitwise; touched rows
   bitwise equal to host f32 with a = mu / (bc1 (sqrt(nu / bc2) + eps)) + wd p, u = (-lr) a).
7. MuonH edge tiles in the engine path: s6_rowpart.muonh_edge_tiles, [2 x 100, 37] (OUT > IN: transposed NS
   orientation) and [2 x 33, 65], true-f32 NS: buffer bitwise against host f32, update rel to an f64 reference of the
   whole chain 2.6e-6 .. 3.3e-6 (bound 2e-5), every element written (u_out pre-filled with NaN). Mutations: dropping the
   last partial column sub-tile of the non-transposed X load (rel 5.9) or the last partial row of the transposed X
   store (rel 9.4) both fail.
8. s6_zero1 wording (S6 section above): 1 rank == 4 ranks bitwise holds on its data, not in general.
9. bench_s6.py: the ZeRO-1 rows used a round-robin split. Each rank's share now comes from the LPT rule of
   `plan_ownership_specs` (same units, costs and parameter order; per-layer gains as separate units); every rank is
   timed; the row reports the slowest rank (the step waits for it) and rank 0.
10. T1.7 on the shared A100: `retry_oom` (tests/s6/s6lib.py) retries a Run or the f64 oracle after a CUDA
    out-of-memory error (8 tries, 60 s apart); other errors and the last OOM propagate.
11. H100 rerun from the integrated full build (S4, S5 and S6 modules in one library): devq t76, below.
Also: s6_ownership's 1.2B layout had 28 layers; the preset has 16 (fixed; balance unchanged, max / mean <= 1.0028).

Not changed: the Adam order (the H100 HLO is the only evidence; T1.7 cannot tell the two orders apart, 8.4e-8 vs
8.6e-8); the 1.2B budget miss (no kernel change in this pass; H100 numbers below); the "verified correct" items.

Gates (della-vis1 A100-PCIE-40GB, shared; incremental build of the full tree with the fixes, -Werror, 0 warnings)

| test | A100 |
|---|---|
| s6_ownership | pass (max / mean load <= 1.0028 at 2 / 4 / 8 ranks) |
| s6_zero1, s6_graph | pass (bitwise) |
| s6_rowpart (6 cases: + muonh_edge_tiles, capi_one_tensor_bits, lazy_adam_weight_decay, refusals) | pass |
| s6_nsalgo | pass (200 plans, 0 fallbacks) |
| s6_bounds | pass |
| s6_ns | pass |
| s6_t1_7 (130m A, B; 300m A) | pass |
| s6_t1_7_small_lr (130m C, 300m B) | SPECFAIL (exit 1): MuonH item 15 ratio <= 6.39, f64 oracle ratio <= 0.974 (C) and <= 1.090 (300m B) |

H100 (devq, 1 GPU of a della H100 node; snapshots of the integrated build): t76 (bench/s6/snap/v6) ran every gate;
all pass except s6_nsalgo's byte comparison (item 2), and s6_t1_7_small_lr is SPECFAIL as on the A100. t79
(bench/s6/snap/v7: the configuration comparison) reran every gate: rc 0, every gate passes, s6_t1_7_small_lr
SPECFAIL with the same numbers; bench within 1 % of t76 (1_2b TF32 4 ranks: 46.50 / 44.82 ms).

| test | H100 (t76 unless noted) |
|---|---|
| s6_ownership, s6_zero1, s6_graph, s6_bounds | pass |
| s6_rowpart | pass: edge tiles rel 2.5e-6 .. 3.3e-6; C ABI mu / nu bits 0x3d4ccccd / 0x3ba3d70a; lazy Adam + decay bitwise |
| s6_nsalgo | t76 FAIL (bytes, item 2); t79 pass (configurations: 200 plans, 0 fallbacks) |
| s6_ns | pass (probe, ns_ref, T1.6 130m, T1.6 1_2b) |
| s6_t1_7 | pass: 130m A (MuonH ratio <= 1.006, Adam 8.4e-8, f32 policy 6.0e-6), 130m B (ratio <= 1.001), 300m A (<= 1 ulp) |
| s6_t1_7_small_lr | SPECFAIL: MuonH item 15 ratio <= 6.39; f64 oracle ratio <= 0.974 (130m C), <= 1.090 (300m B) |

Bench (H100, t76; ZeRO-1 rows: every rank of the LPT plan timed, the slowest reported; ms per step, NS GEMM choice
heuristic / timed)

| size | TF32, one GPU holds all | TF32, 4 ranks (slowest) | TF32 symmetric, 4 ranks | true f32, 4 ranks | DESIGN.md §4.2 budget |
|---|---|---|---|---|---|
| 130m | 5.20 / 4.53 | 1.63 / 1.50 | 1.92 / 1.77 | 3.23 | |
| 300m | 15.15 / 13.96 | 4.21 / 3.93 | 4.88 / 4.51 | 14.09 | |
| 520m | 49.15 / 43.66 | 12.99 / 11.20 | 14.13 / 11.77 | 47.18 | |
| 1_2b | 183.59 / 180.06 | 46.36 / 44.72 | 45.59 / 43.54 | 224.24 | 43 (TF32), 214 (true f32) |

The slowest planned share is 0.5-5 % above the round-robin rank 0 of the S6 table (1_2b TF32: 46.36 vs 46.03 ms). The
1_2b miss stands: TF32 +7.8 % (timed choice +4.0 %), true f32 +4.8 %; within DESIGN.md §12.3's 1.25 x row rule.

## S4: GEMM layer and attention (2026-10-25)

Built
- GEMM layer (src/kernels/gemm). `GemmCache` (gemm.hpp): `plan()` records a `GemmDesc`, `choose_all()` times every
  deterministic candidate per shape and keeps the fastest. Candidates: cuBLASLt's heuristic algorithms (up to 24) and
  each of them with split-K 2-16 under every allowed reduction scheme (bf16 D: NONE and COMPUTE_TYPE only, so a split
  result is reduced in f32 and rounded once; f32 D: also INPLACE and OUTPUT_TYPE); the same set again on a transposed B
  (`ltT:`, for op_b N: the weight is transposed into the workspace inside the timed call, then a K-major GEMM); and 16
  CUTLASS SM90 kernels registered at static init (`gemm_sm90_*.cu`: fwd, dgrad, wgrad, Levanter wgrad x {128x256
  cooperative cluster 2x1, 256x128 cooperative, 128x128 ping-pong, 128x256 stream-K with the deterministic fixup}; the
  Levanter wgrad epilogue `D = C + f32(bf16(acc))` is an epilogue visitor tree, no transient). Timing: all candidates 2
  warm-up + 5 timed calls (median, CUDA events), the 4 fastest again 5 + 20; a custom kernel wins only when > 1% faster
  than the best cuBLASLt one (O9). Candidates that fail to launch are dropped. The workspace is the largest the chosen
  algorithms need plus the transients. `save()` writes `gemm_algos.bin` (key = desc + GPU name + SM count + cuBLASLt
  version -> kernel name, `.tmp` then rename); a later `choose_all` with that file executes the saved kernels (bitwise
  resume). Portable Levanter wgrad path: cuBLASLt writes bf16(acc) to a transient, `bf16_round_add` adds it to the f32
  gradient. `linear_{fwd,dgrad,wgrad}_desc`. Fused epilogues other than Bf16RoundAddF32 are refused (S8).
- C ABI: `nm_linear_fwd/dgrad/wgrad`, `nm_gemm_f32` (TF32 and pedantic f32; 3xTF32 refused), and
  src/kernels/gemm/capi_gemm.h (`nm_gemm_search` -> JSON report, `nm_gemm_chosen`, `nm_gemm_custom_*`,
  `nm_linear_custom`). The C ABI searches a shape on its first use.
- FA3 (third_party/fa3-hopper): the 31 sources of `hopper/` at `767386d` that the sm90 launch templates include, the
  LICENSE, and the marincpp patches (README.md there; `marincpp.patch` = the whole diff; it was stale until the
  review fixes below regenerated and verified it). CUTLASS stays a git checkout at `7127592`; `.gitmodules` declares it as the submodule
  `third_party/cutlass` (the controller adds it with `git submodule add`; the checkout is a blob-less clone). Patches:
  doc-causal masking on FA3's `Is_local` kernels (per-position segment start / end tables bound the n / m loops and
  the masks); a device work-list backward scheduler (`NmBwdListScheduler`; pre/postprocess over list tiles); the
  commented-out KVEmpty handoff enabled for persistent backward CTAs; descending deterministic dQ order over exactly
  the visiting n blocks for doc-causal (upstream local pads over the whole row); GQA first-head store in deterministic
  mode (no dK/dV accumulator memsets); [H, T] LSE in batch mode; errors throw instead of `exit(1)`.
- src/kernels/attention: `fa3_backend.cu` (torch-free parameter fill replacing flash_api.cpp, workspace layout, the
  in-graph kernels `k_bwd_worklists` and `k_doc_tables`), 17 instantiation units (`fa3_inst_*.cu`: fwd hd64 varlen /
  varlen PackGQA / doc, hd128 varlen PackGQA / doc; bwd {hd64 MHA+GQA, hd128 GQA} x {stock, list, doc} x
  deterministic on/off), FA3's `flash_prepare_scheduler.cu`. Modes (attn_internal.hpp): `doc_causal` and
  `persistent` (= the work list on a grid of one CTA per list slot, `ListGrid`), and the E1b arms `ListPersistent`
  (grid = SM count, static striding), `StockPadded`, `StockExact`. Every workspace buffer is caller memory; required
  zeroing is a memset node (GQA dK/dV semaphores; accumulators only when deterministic is off); the forward and the
  backward of one shape share one training workspace. `AttnBwdArgs::softmax_d` and `dq_accum` (S8 fusions) are refused.
- Reference attention (`ref_attention.cu`, sm_80 and sm_90a): the RefF32 backend and the oracle of the FA3 gates.
  Causal same-segment softmax, f32 math and statistics over 1024-key blocks in ascending order, bf16 or f32 storage,
  segments from cu_seqlens or row_seg_start, every output element one warp's fixed-order sum (no atomics). `make_attention`
  builds FA3 and RefF32; LevRef and CudnnSdpa are refused with the reason.
- C ABI: `nm_attention_workspace_bytes/fwd/bwd` (capi.h) and src/kernels/attention/capi_attn.h (`nm_fa3_mode_*` for
  every mode, `nm_ref_attention_fwd/bwd` with explicit storage).

Gates

| gate | where | result |
|---|---|---|
| build, CUDA 12.8 (gcc 14, -Werror, sm_90a + sm_80) | della-vis1 | every target, 0 warnings (CUTLASS as a SYSTEM include: its sm100 collective headers warn under C++20) |
| build, CUDA 13.0 | della-vis1 | every S4 object library and nm_bench_s4_gemm: 0 warnings. The full link stops at S5's `ce_sm90.cu` (`cudaGetDriverEntryPoint` is deprecated in 13.0, -Werror) |
| s4_attention: reference vs f64 (synthetic rows with length-1 segments and tile-edge ends; 130m, 300m, 520m GQA, 1.2B hd128) | A100, H100 | f32 storage rel <= 5.5e-7 (bound 1e-6); bf16 storage O at 1.000x the single-rounding floor, dq / dk 1.14-1.18x, dv 1.00x (bounds 1.05 / 1.25), L rel <= 3.5e-8; bitwise repeatable; cu_seqlens and row_seg_start tables give the same bits |
| s4_attention: every FA3 mode, deterministic off and on, vs the reference in f32 storage, at 130m / 300m / 520m / 1.2B on the 32 real segment rows (bench/e1_attention/segments; 166 segments, longest 4096), on 2 synthetic edge-case rows per shape, and on rows with 24 segments each | H100 | every O / dq / dk / dv within 1.59x the bf16 floor (bound 2.5; E1 measured 1.2-1.9x), L rel <= 1e-5, rows within 10x; deterministic mode bitwise repeatable in all 50 checks; persistent and list_grid bitwise equal (same order) |
| s4_t1_4 (T1.4, oracle rule, levanter fixtures 130m / 520m / 1_2b, both roundings) | A100 (reference), H100 (all) | pass; rel(nano, f64) / rel(levanter, f64), worst over sizes: reference O 0.52, dq 0.23, dk 0.24, dv 0.22, L 0.03; FA3 (persistent and doc_causal, det on and off) O 0.57, dq 0.26, dk 0.28, dv 0.27, L 0.03 (bound 1.25) |
| s4_gemm: linear fwd / dgrad vs f64 (owner layers, T 8192, lm head T 2048) | A100, H100 | 1.000x the single-rounding floor (bound 1.05); two calls bitwise equal |
| s4_gemm: wgrad accurate (T 16,384, beta = 1 in place) | A100, H100 | rel(D - C, f64) <= 2.4e-5 (bound 1e-4: tensor-core f32 accumulation); bitwise repeatable |
| s4_gemm: wgrad Levanter | A100, H100 | D(0) bf16-valued, 1.00x floor; D(C) == C + D(0) bitwise |
| s4_gemm: every CUTLASS kernel on every orientation it takes | H100 | all pass the same bounds; stream-K deterministic: bitwise repeatable |
| s4_gemm: nm_gemm_f32 | A100, H100 | pedantic rel <= 3.5e-7, TF32 2.9e-4 |
| s4_graph_nccl (decision 9): 4 H100, one process, a thread per GPU, ncclCommInitAll; FA3 fwd + bwd of both training schedulers (deterministic, 520m GQA shape, 32k tokens per GPU) + an NCCL all-reduce in one graph | H100 (devq t42; final code devq t56) | 22-node graph per rank, 100 replays, every checkpoint bitwise equal to the eager run on all 4 ranks, both runs |

E1b (decisions 6-8, O1): fwd+bwd per layer per 32 x 4096 tokens on the real segment rows, ms (median of 20, CUDA
events, 1 H100, devq t63, `tests/s4/bench_attention.py`); E1 bar = FA3 varlen of E1 with the real `cu_seqlens`
(deterministic off).

| shape | E1 bar | stock_exact det 0 / 1 | stock_padded 0 / 1 | persistent (SM grid) 0 / 1 | list_grid 0 / 1 | doc_causal 0 / 1 |
|---|---|---|---|---|---|---|
| 130m | 2.51 | 2.45 / 2.61 | 3.65 / 3.71 | 3.11 / 3.27 | 2.55 / 2.63 | **2.37 / 2.46** |
| 300m | 3.75 | 3.78 / 3.91 | 5.43 / 5.49 | 4.64 / 4.86 | 3.79 / 3.91 | **3.63 / 3.70** |
| 520m | 5.68 | 5.64 / 6.10 | 8.24 / 8.63 | 7.71 / 6.85 | 5.79 / 5.70 | **5.36 / 5.42** |
| 1.2B | 9.21 | 9.46 / 10.24 | 12.33 / 13.27 | 14.44 / 11.22 | 9.59 / 9.49 | **9.05 / 9.17** |
| 300m S 4224 | 3.72 | 3.73 / 3.92 | 5.47 / 5.57 | 5.03 / 4.78 | 3.81 / 3.90 | 3.58 / 3.48 |
| 300m S 4480 | 4.71 | 4.80 / 4.83 | 6.57 / 6.69 | 5.61 / 6.36 | 4.87 / 5.12 | 4.96 / 4.93 |
| sum 130m + 520m + 1.2B | 17.40 | 17.56 / 18.94 | 24.22 / 25.61 | 25.26 / 21.33 | 17.93 / 17.82 | **16.78 / 17.06** |

Forward (ms, doc_causal / list modes): 130m 0.68 / 0.62, 300m 1.00 / 0.92, 520m 1.31 / 1.22, 1.2B 1.72 / 1.75 (bar
0.66 / 0.95 / 1.27 / 1.73). Full table with the error of every arm: bench/s4/results/v5-e1b/e1b_summary.md.
- O1 rule: doc_causal (16.78 ms against list_grid 17.93 and the SM-count grid 25.26, deterministic off; 17.06 against
  17.82 and 21.33 on). doc_causal meets the bar at every 4096-token shape with deterministic off (0.94-0.98) and on
  (0.95-1.00); its deterministic cost is +1.7% (FA3's own varlen backward: +8%). cuDNN (decision 8): not built, the
  chosen FA3 sum 16.78 ms is far below 20.56 ms. Request: resolve `exec.attn_bwd_sched=auto` to `doc_causal`.
- The padded version-1 grid costs +39% (24.22 vs 17.40), E6's finding. The work list (list_grid) removes it (+3%
  over E1); the SM-count grid with static striding is 1.2-1.6x slower than the same list on a full grid (no dynamic
  load balance), so `persistent` runs the full-grid list.
- What moved the numbers (earlier E1b runs, bench/s4/results/v3-e1b, v4-e1b): ascending deterministic dQ order cost
  2.2-2.5x (doc_causal) and 1.4x (list) because every n block then waits for its predecessor's dQ write at every m
  block; descending order (FA3's causal order: the higher n blocks reach an m block first) removed it. Ordering the
  work list head-innermost cost 1.3-1.6x (poor L2 reuse of Q / dO); segment, then head, then n fixed it.
- Misses: doc_causal at S 4480 (4.96 vs 4.71, +5%; its forward 1.37 vs 1.23: 4480 is not a multiple of the 192-row
  forward tile); list_grid +1-4% at the 4096 shapes. S 19456 (pls bytes) was not run: its segment file is the bytes
  tokenizer's, which the bench does not load yet.

GEMM search (decision 4; bench/s4/results/gemm_combined.md): `nm_bench_s4_gemm --set all` on 1 H100 in three runs
(devq t41, t55, t64), TFLOP/s of the chosen kernel (search median, and a second timing through `GemmCache::run`).
The same shape differs by up to 8% between runs on the shared node, so the table takes the best of the three. 29 of the
44 E1 shape-orientations reach the bar (best of cuBLAS default and XLA, RESULTS.md §2).

| M x K x N | fwd | dgrad | wgrad | wgrad Levanter |
|---|---|---|---|---|
| 131072 x 512 x 1536 | 751 / 606 | 760 / 707 | 751 / 728 | 737 / 728 |
| 131072 x 512 x 1792 | 753 / 645 | 718 / 715 | 717 / 712 | **707 / 712** |
| 131072 x 768 x 2304 | 794 / 692 | 771 / 754 | 712 / 671 | 708 / 671 |
| 131072 x 768 x 2688 | 772 / 690 | **737 / 760** | 749 / 712 | 768 / 712 |
| 131072 x 1024 x 3584 | 769 / 760 | **717 / 756** | 728 / 718 | 750 / 718 |
| 131072 x 2048 x 7168 | **712 / 733** | 740 / 733 | 714 / 700 | 710 / 700 |
| 131072 x 512 x 128256 | **658 / 671** | 691 / 685 | 706 / 696 | 698 / 696 |
| 131072 x 768 x 128256 | 684 / 668 | **699 / 723** | 712 / 709 | 713 / 709 |
| 16384 x 512 x 128256 | 718 / 692 | **691 / 724** | **703 / 704** | **691 / 704** |
| 16384 x 768 x 128256 | 690 / 665 | **736 / 740** | **693 / 708** | **694 / 708** |
| 8192 x 768 x 128256 | 747 / 743 | **712 / 759** | **670 / 778** | **698 / 778** |

(chosen / bar; bold = below the bar.) Trunk misses: dgrad K 1024 N 3584 0.948, dgrad K 768 N 2688 0.970, fwd K 2048
N 7168 0.971, Levanter wgrad K 512 N 1792 0.994. The other 11 misses are LM-head shapes (S5's CE uses its own GEMMs).
Chosen kernels: cuBLASLt algorithm 66 (sm90) with various tiles and split-K (COMPUTE_TYPE) for the small-output
wgrads, CUTLASS stream-K for some wgrads and the Levanter wgrad at K 768-1024, CUTLASS cooperative for dgrad K 2048.
The transposed-B dgrad variant never won. Per-shape candidates: 12 (bf16 out) to 65 (f32 out with split-K).
- O8 (decision 5): cuBLASLt 13.0 / 12.8, best of three runs each, over the 44 E1 shape-orientations: median 0.976,
  0.90-1.02. 13.x is not faster by > 3%: stay on 12.8.

Commands
```bash
ssh della-vis1; cd $ROOT/project/marin.cpp; module load gcc-toolset/14 cudatoolkit/12.8; export NANOMARIN_ROOT=$ROOT
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++
nice -n 19 cmake --build build -j 14 && (cd build && nice -n 19 ctest -R 's4_' --output-on-failure)
# H100 (devq): snapshot the build (lib, nm_bench_s4_gemm for 12.8 and 13.0, nm_test_s4_graph_nccl, tests/s4, tests/s3/nmk.py)
# into bench/s4/snap/<v>/ with bench/s4/snap/v1/task.sh, then
python3 tools/devq/devq.py submit --gpus 1 --arch h100 --name s4-tests --cwd <snap> -- bash <snap>/task.sh tests  # e1b, gemm
python3 tools/devq/devq.py submit --gpus 4 --arch h100 --name s4-graph --cwd <snap> -- bash <snap>/task.sh graph
python3 tests/s4/gemm_report.py bench/s4/results/<v>-gemm/gemm_cu128.jsonl bench/s4/results/<v>-gemm/gemm_cu130.jsonl
```

For other owners (S4 does not edit them)
1. config (`resolve`, config.hpp `AttnBwdSched`): `exec.attn_bwd_sched=auto` -> `doc_causal` (O1 decided above).
2. DESIGN.md §3.4 / O1 / attention.hpp: `persistent` is the device work list on a grid of one CTA per list slot (the
   SM-count grid measured 1.2-1.6x slower); O1 decided: doc_causal. The FA3 deterministic cost in §11 / §12.2b
   (x1.14) is x1.017 for doc_causal.
3. attention.hpp `AttnBwdArgs::dq_accum` and S3's `k_dq_post_qknorm_rope_bwd` (DESIGN §3.1 B3): FA3's dQ accumulator
   is not row-major [H, T, hd]: each kBlockM x hd tile is stored in the dQ MMA's fragment order (FA3's postprocess
   reads it through TiledMmadQ). The S8 fusion must read that layout (or FA3 must store rows); until then the backend
   refuses `dq_accum` and writes dq. `softmax_d` input likewise refused until S8 also writes lse_log2.
4. gemm.hpp: `GemmCache` has no destructor and no report accessor; S4 keeps the search reports and chosen names in
   a side table (`gemm::report`, `gemm::chosen_name` in gemm_internal.hpp). Request: `~GemmCache()`, non-copyable,
   `std::vector<ShapeReport> report() const`.
5. S5 (`ce_sm90.cu`): `cudaGetDriverEntryPoint` is deprecated in CUDA 13.0 (`cudaGetDriverEntryPointByVersion`),
   which breaks the 13.0 build under -Werror.
6. S6: `nm_gemm_f32` (capi.h GEMM block) is S4's; it does not prove TF32 rounding to nearest (decision 14 is S6's
   T1.6 probe).

Known gaps
- LevRef (levanter's bf16 running statistics, the strict preset's attention) is not built. Levanter's jaxpr
  (flash_attention.py) rounds attn, max, P, the running sums and o to bf16 at each 1024-key block, and its backward
  mixes an f32 P with bf16 dP / D; XLA:GPU on the A100 dropped bf16 rounding points inside fusions (S3 finding), so a
  bit-faithful port needs the H100 HLO first. T1.4 passes with FA3 and the reference.
- 3xTF32 GEMMs and upper-triangle-only (`symmetric_output`) tiles are not built (NS: S6 uses its own path);
  `symmetric_output` computes the full product.
- The work list's SM-count grid (`ListPersistent`) has no dynamic load balance (1.2-1.6x slower); kept as an E1b arm.
- GQA accumulators in deterministic-off mode are still zeroed by memsets each call (FA3's order is not fixed there).
- GEMM bars: 15 of 44 shape-orientations below on the best of three runs (above); the run-to-run spread (up to 8%) is
  as large as most gaps. A CUTLASS dgrad tuned for N-major B, or more cuBLASLt configurations than the heuristic
  returns (tile / stage enumeration), are the next candidates.
- E1b at S 19456 (pls bytes) not run.

## S4 review fixes (2026-10-25)

Blocking findings (all three real; fixed)
1. FA3 deterministic hand-off. The dQ writer (`store_dq`) and the GQA dK / dV epilogue released the semaphore after
   `tma_store_wait` (`cp.async.bulk.wait_group.read`: the shared-memory reads are done, not the global add), so the
   next writer's bulk add to the same tile was not ordered by the PTX memory model. Fix (`SM90_BULK_HANDOFF`,
   third_party/fa3-hopper/copy_sm90_bulk_reduce.hpp; deterministic instantiations only): each writer runs
   `cp.async.bulk.wait_group 0` + `fence.proxy.async.global` before `arrive_inc`; the next writer runs
   `fence.proxy.async.global` after its acquire, before its bulk add or store. Covers dQ (doc-causal, work list, stock)
   and every non-first GQA head (G >= 3 included). Cost, E1b v7 (below): doc_causal deterministic +2.6% summed
   (16.46 -> 16.88 ms; it was +1.7% before the fix).
2. GEMM gate covers every candidate. New C ABI (capi_gemm.h): `nm_gemm_candidates` (every candidate of one search that
   launched, fastest first) and `nm_linear_named` (one call of a named candidate through the cache's own code path;
   `gemm::force_choice`). New case s4_gemm.every_candidate: every candidate of every owner layer (11) and orientation
   (fwd, dgrad, wgrad accurate, wgrad levanter), two calls bitwise equal, the same bounds; wgrad on a random nonzero C
   (accurate rel(D - C, f64) <= 1e-4; levanter D(0) bf16-valued, D(0) <= 1.05x floor, D(C) == C + D(0) bitwise). H100:
   3,532 candidates, 0 failing (fwd / dgrad <= 1.004x floor, wgrad accurate <= 1.86e-5). A100 (one layer per size +
   head): 1,294 candidates, 0 failing. Mutation (the reviewer's: `transpose_bf16` skips the last k row): the old
   linear_fwd_dgrad still passes; every_candidate fails 604 checks (every `ltT:` candidate, dgrad up to 22x floor,
   wgrad rel 7.9e-3).
3. custom_kernels runs every CUTLASS wgrad on a random nonzero C (in place, beta = 1): accurate rel(D - C, f64) <= 1.9e-5;
   Levanter D(C) == C + D(0) bitwise for all 22 kernel x shape pairs (H100, devq t83).

Minor findings
4. `marincpp.patch` regenerated (`diff -ru` against upstream `767386d` extracted with `git show`, equal to the
   reviewer's copy); `patch -p1` on upstream reproduces all 31 vendored files. README.md states the check.
5. doc_causal forward at hd 64 = the varlen forward (`fwd_varlen`, fa3_backend.cu): it writes O [T, H*hd] and lse
   [H, T], the layout the doc backward reads; hd 128 keeps the doc forward (faster there). doc_causal now needs
   cu_seqlens at hd 64 (k_doc_meta writes both tables). Forward, ms, E1b v7: 130m 0.62, 300m 0.91, 520m 1.22, 1.2B 1.72
   (bars 0.66 / 0.95 / 1.27 / 1.73): every bar met; S 4480 fwd+bwd 4.37 (bar 4.71, was 4.96).
6. Production dispatch tested. fa3_owner_shapes and the new fa3_gqa_group4 run `nm_attention_workspace_bytes / fwd /
   bwd` (make_attention, fa3::fwd / bwd, fa3_mode_of) for both schedulers: deterministic on, bitwise equal to the mode
   it must run (12 of 12); off, within the FA3 bounds. production_ref_f32: ref_f32 through make_attention ==
   nm_ref_attention_* bitwise (A100 and H100). T1.4 calls FA3 through nm_attention_*. Test mode names now match the
   training knob: `persistent` = ListGrid (what exec.attn_bwd_sched=persistent runs), `sm_grid` = ListPersistent (the
   E1b arm); the old T1.4 / gate rows labelled "persistent" ran the SM-count grid.
7. test_graph_nccl: each scheduler writes its own O / LSE / dQ / dK / dV and all-reduced dK (23 graph nodes per rank);
   100 replays bitwise equal to the eager run on 4 ranks for both (devq t82).
8. GemmCache: candidate names carry every config attribute read with its own type (`:is` inner shape added; the
   uint16 cluster id had been read as uint32, the read failed and every name said `cl0`). On the E1 shapes the
   candidate counts are unchanged (no two heuristic results had collided), but saved names now identify the kernel
   exactly. The heuristic is asked with the fixed
   256 MiB limit, so the candidate set (and a resume's saved name) no longer depends on the scratch size; candidates
   needing more than the scratch are not timed. `save()` writes `<path>.tmp.<pid>.<thread>`. `gemm::forget` drops a
   cache's side-table records (the C ABI calls it before destroying a cache). The C ABI reuses one search scratch per
   device. Still open: the Entry leak (needs `~GemmCache`, request below) and sharing choices across ranks (S7).
9. GEMM table without selection bias: tests/s4/gemm_report.py `--runs` prints every run, the median and the best.
   Three independent searches of the final code (devq t84, t85, t86; bench/s4/results/v8_gemm_runs.md): the median
   meets the bar at 15 of 44 E1 shape-orientations (best of three: 24). Trunk medians below the bar: dgrad K 768 N 2688
   0.869, K 768 N 2304 0.890, K 1024 N 3584 0.931, K 512 N 1792 0.937, K 2048 N 7168 0.989; fwd K 2048 N 7168 0.968,
   K 1024 N 3584 0.967; Levanter wgrad K 768 N 2688 0.963, K 1024 N 3584 0.967, K 2048 N 7168 0.994, K 512 N 1536
   0.997, K 512 N 1792 0.999; wgrad K 2048 N 7168 0.990. The other 16 are LM-head shapes (S5 owns the CE GEMMs). The
   earlier "29 of 44" was the best of three. O8: cuBLASLt 13.0 / 12.8 (t84) median 0.995 (0.94-1.10): stay on 12.8.
10. GQA G = 4 (H 16, KV 4, hd 64 and 128) is tested: reference vs f64 on the A100; on H100 every FA3 mode, the work
    list, doc_causal and the SM-count grid, deterministic on and off, on the real rows: <= 1.56x floor, bitwise
    repeatable.
11. fa3_padded_slots_overflow_guard renamed fa3_full_slots (24 segments in 24 slots); it states the data-path contract
    instead of claiming a refusal.

Gates (della-vis1 A100, CUDA 12.8, -Werror, 0 warnings; H100 through devq from bench/s4/snap/v7 and v8)

| gate | A100 | H100 |
|---|---|---|
| s4_gemm (6 cases incl. every_candidate, custom_kernels with nonzero C) | pass (custom_kernels skips) | pass (t80, t83) |
| s4_attention (8 cases) | pass (FA3 cases skip) | pass (t80, t83): FA3 <= 1.59x floor, 58 of 58 deterministic runs bitwise repeatable, 12 of 12 production == mode |
| s4_t1_4 (oracle rule, FA3 via nm_attention_*) | pass (reference) | pass: worst ratio 0.57 (bound 1.25) |
| s4_graph_nccl (4 GPUs, 100 replays) | skip | pass (t82) |
| build, CUDA 13.0: S4 object libraries and nm_bench_s4_gemm | 0 warnings | |
| scrub check (src/kernels/gemm, src/kernels/attention, third_party/fa3-hopper, tests/s4, .gitmodules) | clean | |

E1b after the fixes (devq t81, bench/s4/results/v7-e1b; fwd+bwd ms, deterministic off / on; mode names as in 6)

| shape | E1 bar | stock_exact | stock_padded | sm_grid | persistent | doc_causal |
|---|---|---|---|---|---|---|
| 130m | 2.51 | 2.45 / 2.51 | 3.64 / 3.70 | 3.12 / 3.29 | 2.60 / 2.64 | **2.37 / 2.36** |
| 300m | 3.75 | 3.67 / 3.83 | 5.42 / 5.53 | 4.61 / 4.88 | 3.84 / 3.93 | **3.48 / 3.64** |
| 520m | 5.68 | 5.62 / 6.09 | 8.19 / 8.62 | 7.70 / 6.82 | 5.77 / 5.67 | **5.15 / 5.27** |
| 1.2B | 9.21 | 9.34 / 10.18 | 12.34 / 13.26 | 14.43 / 11.40 | 9.49 / 9.45 | **8.94 / 9.25** |
| 300m S 4480 | 4.71 | 4.74 / 4.98 | 6.59 / 6.72 | 5.52 / 6.48 | 4.94 / 5.08 | 4.37 / 4.77 |
| sum 130m + 520m + 1.2B | 17.40 | 17.41 / 18.78 | 24.17 / 25.58 | 25.24 / 21.51 | 17.85 / 17.76 | **16.46 / 16.88** |

O1 rule: doc_causal (persistent / doc_causal 1.085 off, 1.052 on). doc_causal meets the bar at every 4096 shape with
deterministic off (0.91-0.97) and on (0.94-1.00). Errors <= 1.52x floor.

For other owners (S4 does not edit them; adds to the S4 requests above)
7. attention.hpp: the FA3 entry file is src/kernels/attention/fa3_backend.cu (not fa3_entry.cu); `Persistent` is the
   device work list on one CTA per list slot (not "grid = SM count; CTAs pull work items"); `AttnFwdArgs::cu_seqlens`
   is also read by DocCausal's forward at hd 64. State the contract: each row has at most max_segs_per_seq segments
   (k_doc_meta keeps the first max_segs starts and flags the overflow), so n_segs <= rows * max_segs_per_seq and
   cu_seqlens[rows * max_segs_per_seq] = total_q; FA3's varlen forward reads every slot and the reference reads
   cu[slot + 1].
8. DESIGN §11 / §12.2b: the FA3 deterministic cost for doc_causal is x1.026 (E1b v7, with the hand-off fix).
9. gemm.hpp (replaces request 4): `~GemmCache()` that frees the entries' descriptors and calls `gemm::forget(*this)`;
   non-copyable; `std::vector<ShapeReport> report() const`.
10. S7: GemmCache choices are timing-dependent per process. Ranks of one run and A/B twins (DESIGN §11) must load one
    gemm_algos.bin: rank 0 searches and saves, the others construct with that path after a barrier.

## S5: fused linear cross-entropy (2026-10-25)

Built (src/kernels/cross_entropy, tests/s5; no header outside the module changed)
- `ce::run` (behind `fused_linear_ce` and the C ABI `nm_fused_linear_ce`) walks the call in chunks of
  `CeShape::chunk_rows`; per chunk, as decided (items 10-13): gradient in the forward, bf16 logits store, lse and the
  softmax statistics from the f32 accumulators, dW accumulated in f32 across chunks and microbatches. Stages:
  - A: cuBLASLt f32 logits + `k_ce_row_a` (lse, loss, dz in place: bf16 dz over the first half of each f32 row, p
    from f32(bf16(z)) under the bf16 store; f32 dz under the f32 policy), cuBLASLt dX / dW. The reference, the
    f32-policy path (true f32 GEMMs, f32 x, W, dX, dW) and the only stage with `logit_bias`.
  - B1: `k_ce_logits_stats` -> `k_ce_finalize_rows` -> `k_ce_dlogits` (rewrites the store as bf16 dz in place) -> dX
    and dW GEMMs on the plain dz (cuBLASLt by default, the own kernels on request).
  - B1.5: the dX kernel computes dz from the store on the fly and writes it in place (its CTA owns a row block over all
    of D); dW reads the plain dz.
  - B2: dX and dW both compute dz from the store on the fly.
  - logits store `exec.ce_logits=f32` (the fallback) works in every stage (sm_90a kernels fall back to the portable
    ones for it); per-chunk bf16 dW (`CeDwRound::PerChunkBf16` + `ce_finish_microbatch`) in every stage;
    `accumulate_dx` with the levanter form bf16(old + f32(bf16(acc))) and the accurate form bf16(old + acc); dX may
    alias x (readouts write d(norm out) over the norm output).
- sm_90a kernels (ce_sm90.cu; dispatch on compute capability 9.0, portable mma.sync kernels on sm_80 and as fallback):
  - logits + statistics: persistent, TMA, one producer warpgroup and two consumer warpgroups in ping-pong (named-barrier
    order: a warpgroup's mainloop starts after the other one has waited on every stage of the previous tile, which also
    keeps every mbarrier wait within one ring phase), 128 x 128 tile per warpgroup (two wgmma.m64n128k16 per k16 step,
    128 f32 accumulators per thread), 4-stage ring of 64-wide K blocks, statistics from the accumulators (row max,
    then sum of libdevice expf, two rows at a time with the exponentials as independent chains), z[target], the bf16
    store through a 128B-swizzled staging tile and two TMA stores per tile.
  - dX / dW: one kernel template, 64 output rows x 2 x 256 columns of D (2 x 192 when D is a multiple of 384 but not
    512), K blocks of 32, 5-stage TMA ring; MN-major operands (descriptor layout confirmed on the H100 by
    bench/s5_ce/probe_mn.cu); for dz-from-store a transform (warps 1-3 of the producer warpgroup) writes bf16 dz into
    one of 4 shared tiles once per CTA tile, with lse, g, target loaded by TMA next to each stage; B1.5's in-place
    dz write from the transform. Bitwise equal to the portable kernels (same K order): dX and dW do not depend on the
    GPU type.
- Numerics shared by every stage: one `ce_dz` (p = expf(z_s - lse), dz = p g, dz[target] -= g, explicit roundings,
  `--fmad=false`), so A (bf16 store emulation), B1, B1.5 and B2 produce the same dz bits; the stage and the GEMM
  backend change only the GEMM summation order. A target outside [0, V) gets no loss and no gradient.
- One numeric path for every microbatch size: a row's loss, lse and dX do not depend on the chunk or call size
  (no split-K in the own kernels; cuBLASLt plans fix one algorithm per (kind, D, V, ld) chosen for 16,384 rows, and
  a split-K algorithm with a bf16 output must reduce in f32 (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) or not split:
  the A100 heuristic's bf16-output reduction raised the dX error to 1.33x levanter's, DESIGN §3.7). Determinism:
  no float atomics anywhere; repeat runs are bitwise equal.
- Loss reduction: `loss_partial[c]` = fixed-order sum of weight x CE per chunk; `ce::grad_scale` (C ABI
  `nm_ce_grad_scale`) = mult / sum(w) in f32, the spec's denominator (numerics.md §3.6; the trainer passes
  1/(B_global (S-1)) x readout weight as `grad_scale`).
- Readouts (`run_readout`, `ce::readout`, C ABI `nm_ce_readout`): readout norm forward into the transient, columns
  past `visible_dims` zeroed, CE with dW added in place into the head's f32 gradient, d(norm out) written over the
  norm output, its invisible columns zeroed, then `rmsnorm_bwd_add` adds d(state) into the layer's residual gradient
  and writes the gain partials. No backbone and no gain gradient: no dX GEMM (the ~1/3 saving). Gain gradient only:
  the dX GEMM runs (the gain gradient needs it) and the discarded norm dX goes to the free logits store.
- Options (`ce::Backends`, process-wide, C ABI `nm_ce_set_backends`): plain-dz GEMMs on cuBLASLt (default: 740-820
  TFLOP/s on H100 against 450-550 for the own kernels) or the own kernels; which sm_90a kernels run (default both);
  `fast_stats_exp` (`nm_ce_set_fast_stats_exp`, default off): ex2.approx of the rounded d log2(e) for the statistics
  only, 3 instructions instead of 9 per logit, off until the numerics.md §1 A/B allows a fast-math exponential.
- `ce_workspace_bytes` = the planner's ws.ce.* layout plus a 32 MiB cuBLASLt workspace region (Hopper cuBLASLt offers
  no algorithm without one at these shapes) plus dxtmp when V < 32 D - 128 (see requests).

Gates (della-vis1 A100-PCIE-40GB: fresh configure + build, 0 warnings; H100 through devq: t69 every case and T1.5,
t71 sm90 / bitwise / fast-exp cases on the kernels of this commit; on the final code: t72 every case and T1.5,
t74 the two-GPU case with sm90 / bitwise / strict / readout)

| gate | checks | A100 | H100 |
|---|---|---|---|
| s5_ce grad_V512 / V8256 / V64256 / V128256 | A, B1 (cuBLASLt and own GEMMs), B1.5 (both), B2 at h = 512 / 768 / 1024 / 2048, T = 2,125 (2 chunks + tail), f64 torch autograd oracle: loss and lse within max(2e-5, 1.25 x the emulation's error), dX and dW by the oracle rule against the levanter emulation (numerics.md §5.2 with tensor-core f32 GEMMs), weight-0 loss and dX exactly 0 | pass | pass |
| s5_ce f32_store | the f32-store fallback, every stage, both GEMM backends | pass | pass |
| s5_ce bitwise | repeat; chunk 128 / 512 / 1024 / 4096: loss, lse, dX bitwise equal, dW rel < 1e-5; forward-only = the loss of the gradient call; B1 = B1.5 = B2 bitwise with the own GEMMs | pass | pass |
| s5_ce inplace_dx, accumulate_dx | dX over x equals a separate dX; levanter accumulate bitwise, accurate within one bf16 ulp and not worse | pass | pass |
| s5_ce strict_dw | PerChunkBf16 + finish_microbatch over two microbatches, bitwise | pass | pass |
| s5_ce f32_io, odd_shapes | f32 policy (rel <= 1e-5) with and without bias; V = 1000, 8257, 130; D = 520, 136, 64; T = 1; 1-row tail chunk | pass | pass |
| s5_ce readout | readout = the composed C ABI calls bitwise (both roundings, partial width, gain-only, no backbone, forward-only); f64 autograd sanity | pass | pass |
| s5_ce sm90_vs_portable, sm90_gemm | sm_90a vs portable: store bitwise equal, loss/lse within 4e-6; own sm_90a GEMMs bitwise equal to the portable ones; repeat and chunk bitwise | skip | pass |
| s5_ce fast_stats_exp | the opt-in exponential within the grad-case bars; lse moves by <= 1.9e-6, 0.001-0.002 % of dX bits | pass | pass |
| s5_ce two_devices | the same calls on GPU 0 and GPU 1 of one process give the same bits (per-device cuBLASLt handles and plans, per-device kernel attributes) | skip (1 GPU) | pass (t74) |
| s5_t15_levref | T1.5, below | pass | pass |
| compute-sanitizer (CE kernels only, `--kernel-name kns=2nm2ce`) | memcheck on odd_shapes, readout, inplace_dx, strict_dw, f32_io; racecheck and synccheck on every stage and store at V = 1000 / 8257 (racecheck on the V = 128,256 cases did not finish in an hour) | memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors (portable kernels) | |

T1.5 (cache/levref/<size>/t1_units/ce and t1_oracle/ce; 130m and 1_2b; levanter's own error to the f64 oracle:
grad_x 1.717e-3 / 1.715e-3, grad_w 1.725e-3 / 1.759e-3). Every row passes on both GPUs (t69 on the H100): A, B1
(cuBLASLt, own), B1.5 (both), B2; chunk 8192 / 16384 (130m) and 32768 (130m, GPUs with >= 60 GiB); bf16 and f32
store; strict dW. The fast-exponential row passes on both GPUs (t72 on the H100).

| H100 | per-token loss max abs diff (bar 2e-5) | weighted mean rel (bar 1e-6) | grad_x rel / levanter's | grad_w rel / levanter's | bitwise equal to levanter: loss / grad_x / grad_w |
|---|---|---|---|---|---|
| 130m, B stages, f32 dW | 1.9e-6 | 2.7e-10 | 1.000 | 0.11-0.13 | 68 % / 99.994 % / 83 % |
| 130m, A | 1.9e-6 | 1.2e-11 | 1.000 | 0.11-0.13 | 97 % / 100.000 % / 83 % |
| 130m, strict dW (bf16 per 8192 rows) | 1.9e-6 | 2.7e-10 | 1.000 | 1.000 | 68 % / 99.994 % / 99.97 % |
| 1_2b, B stages | 1.9e-6 | 3.4e-10 | 1.000 | 0.126 | 68 % / 99.995 % / 80 % |
| 1_2b, strict dW | 1.9e-6 | 3.4e-10 | 1.000 | 1.000 | 68 % / 99.995 % / 99.96 % |

On the A100 the same rows pass with grad_x at 1.014 x levanter's error (98.5 % of its bits). With the bf16 store the
port reproduces levanter's batched_xla backward: on the H100, grad_x is levanter's to the bit except 0.006 % of the
elements and the strict dW is levanter's bf16 grad_w except 0.03 %; the f32 dW is 8x closer to the f64 oracle. T1.5
passes with the bf16 store, so `exec.ce_logits=auto` = bf16 stands and the f32 store stays only as a knob.

H100 timing (devq, 1 GPU, 5 warm-up + 20 timed calls, median of CUDA events; forward + dX + dW, bf16 store, f32 dW,
levanter rounding; tests/s5/bench_ce.py focus; other jobs shared the node, run-to-run spread about 3 %):

| rows x h x V | Liger bar (E2) | B1 | B1 / bar | B1, fast statistics exp | B1.5 | B2 | logits + statistics alone |
|---|---|---|---|---|---|---|---|
| 16,384 x 512 x 128,256 (pls head) | 13.57 | 13.57 | 1.000 | 13.39 | 16.28 | 19.15 | 4.23 (fast exp 3.74) |
| 16,384 x 768 x 128,256 | 18.1 | 18.64 | 1.030 | 18.71 | 24.21 | 31.23 | 5.91 |
| 16,384 x 1,024 x 128,256 | 22.6 | 23.83 | 1.054 | 23.09 | 28.61 | 36.14 | 7.63 |
| 16,384 x 2,048 x 128,256 | 40.8 | 42.27 | 1.036 | 42.29 | 55.38 | 71.06 | 15.17 |
| 131,072 x 512 x 128,256 | 109.0 | 113.66 | 1.043 | 109.09 | 123.85 | 152.18 | 35.01 (fast exp 30.84) |
| 131,072 x 768 x 128,256 | 145.2 | 151.45 | 1.043 | 149.00 | 196.15 | 251.67 | 47.22 |
| 131,072 x 1,024 x 128,256 | 179.3 | 189.00 | 1.054 | 185.84 | 229.39 | 290.01 | 61.25 |
| 131,072 x 2,048 x 128,256 | 320.0 | 340.50 | 1.064 | 337.51 | 445.58 | 568.65 | 124.20 |
| 131,072 x 512 x 64,256 | 53.7 | 56.98 | 1.061 | 54.85 | 62.01 | 75.58 | |
| 131,072 x 2,048 x 64,256 | 162.4 | 173.43 | 1.068 | 171.47 | 228.20 (t69) | 291.85 (t69) | |
| 131,072 x 512 x 8,256 (t69) | 8.2 | 7.70 | 0.939 | | 8.43 | 9.95 | |
| 131,072 x 2,048 x 8,256 (t69) | 21.8 | 22.43 | 1.029 | | 29.41 | 37.72 | |

ms; chunk 16,384 rows; devq t71 (the code of this commit; its runner was cancelled by the controller before the
V = 8,256 rows), rows marked t69 from the run before the last mainloop change (descriptor arithmetic, -0.15 to
-0.5 ms per 16,384 rows). B1.5 = the faster of its cuBLASLt / own dW. B1 with the own sm_90a GEMMs: 15.36 / 22.46 /
27.36 / 54.60 ms at 16,384 rows. B1's dlogits pass alone: 3.06-3.09 ms per 16,384 x 128,256 (2,725-2,746 GB/s). The
portable logits kernel on the H100: 8.3 ms forward at 16,384 x 512 (the sm_90a kernel 4.23).

O2 per D: B1 (sm_90a logits, dlogits in place, cuBLASLt dX / dW) is the fastest stage that passes T1.5 at every D and
row count. B1.5 and B2 pass T1.5 and are slower: their dX / dW run on the own GEMM kernels, which reach 450-550
TFLOP/s on a plain dz against cuBLASLt's 740-820, and the dz transform (3 warps, one per scheduler) takes longer than
the 512-clock tensor work of a K block, so B2's GEMMs run at ~300 TFLOP/s.

Findings
- The statistics epilogue first cost 3.3 ms per 2.1e9 logits on top of a 3.3 ms mainloop (D = 512): a conditional
  libdevice expf compiles to a branch per element (BSSY/BRA/BSYNC), which serializes the exponentials of the one
  epilogue warp per scheduler. Branch-free selects, two rows at a time and independent exponential chains, plus
  st.shared for the staging tile, brought the D = 512 forward from 6.8 to 4.2 ms; the dz transform had the same
  pattern (fixed the same way).
- setmaxnreg.inc only takes registers a setmaxnreg.dec released: the warpgroup split must sum to the launch
  allocation (40 + 2 x 232 = 88 + 2 x 208 = 3 x 168). The first dX / dW kernel asked for 96 + 2 x 208 and hung.
- The hardware applies the shared-memory swizzle by absolute address bits: a dz tile placed 768 B past a 1024-B
  boundary gave uncorrelated dX (rel 1.41) while the dz written to global memory was right.
- A 2-CTA cluster variant of the logits kernel (W tiles multicast, half the W L2 traffic) was correct but slower
  (4.66 vs 4.00 ms at D = 512, 14.7 vs 14.0 ms at D = 2048); with .release.cluster stage releases it was 2.7-3.2x
  slower. L2 traffic is not what limits the mainloop; the variant was removed.
- ncu (D = 512 logits): L2 throughput 82 %, tensor pipe 61 %, issue slots 45 %; top stalls fixed-latency waits,
  short scoreboard, math pipe throttle; the most executed mainloop instructions were the per-k-block descriptor
  construction and its R2UR moves (now base + offset in the descriptor's low word).
- Implementation A's row kernel under f32 io read z[target] for the loss while other threads already wrote dz over
  the row (no barrier on that path): an intermittent wrong per-token loss with right gradients, seen once in the
  final A100 ctest; fixed with a barrier. The sm_90a launchers first set the large-shared-memory attribute once per
  process; it is per device (one thread per GPU in the trainer), now set on every launch.
- A 5-stage ring in the logits kernel (t75; forward only, 16,384 rows, two runs each) gains nothing at D = 512 and
  1,024 (4.32 / 7.04-7.27 ms against 3.89-4.30 / 7.27-7.31 with 4 stages) and is slower at D = 2,048 (16.1-16.4
  against 14.1-14.6 ms); the kernel keeps 4 stages.
- Hopper cuBLASLt offers no algorithm without workspace for these shapes (every cuBLASLt call failed with workspace
  0); the A100 heuristic picks a split-K whose bf16-output reduction adds a rounding per part.

Known gaps
- B1 meets the E2 Liger bars only at some shapes: 1.000x at the pls head shape (16,384 x 512 x 128,256), 1.03-1.07x
  at the other V = 128,256 and 64,256 shapes, 0.94x at 131,072 x 512 x 8,256. The remaining gap is the logits kernel
  at large D
  (580-650 TFLOP/s mainloop against cuBLASLt's 740-820 for the same GEMM shape) and its epilogue at D = 512 (4.2 ms
  against a 3.3 ms mainloop). The fast statistics exponential takes the D = 512 forward to 3.7 ms but is off until its
  A/B. B2 does not reach its 9.1-10.7 ms target (B2 needs the own GEMM core at cuBLASLt speed and a cheaper transform).
- The partial-tile statistics path of the sm_90a logits kernel spills one register (4 B; runs only for the last
  vocab block when V % 128 != 0).
- `fused_linear_ce` ignores its GemmCache argument (the CE GEMMs are this module's kernels and its own cuBLASLt plans).
- `run_readout` uses eps = 1e-5 (ReadoutCall carries no eps).

Requests
- memory planner (src/memory, S0 owner): ws.ce.* lacks the 32 MiB cuBLASLt region (`ce::Ws::ltws`) and, when
  V < 32 D - 128 (e.g. V = 8,256 at D = 2,048), the [rows, D] dX scratch; `ce_workspace_bytes` is the contract.
  cross_entropy.hpp: its "B1 at 16384 rows: 4,334,288,896 B" becomes 4,367,843,328 B.
- cross_entropy.hpp: `ReadoutCall` gets `float eps` (model.norm_eps).
- DESIGN §3.8: "a readout with dx = nullptr ... computes only its gain gradient" cannot hold (the gain gradient needs
  d(norm out), the dX GEMM's output); the implemented rule is above.
- knobs.def / config owner: the GEMM backend of the plain-dz GEMMs and the fast statistics exponential are
  process-wide options today; if they should be run knobs, `exec.ce_gemm=lt|own` and `exec.ce_stats_exp=accurate|fast`
  (the latter needs the numerics.md §1 A/B before it can default to fast).
- DESIGN §3.8 / O2: record B1 as the stage for every D, and the measured B1 margins against the Liger bars.

Commands
```bash
ssh della-vis1; cd $ROOT/project/marin.cpp; module load gcc-toolset/14 cudatoolkit/12.8
export NANOMARIN_ROOT=$ROOT
nice -n 19 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++
nice -n 19 cmake --build build -j 8 && (cd build && nice -n 19 ctest -R 's5_' -LE bench --output-on-failure)
python tests/s5/test_ce.py --lib build/libmarincpp_kernels.so --src . grad_V128256 bitwise    # selected cases
python tests/s5/bench_ce.py --lib build/libmarincpp_kernels.so --src . focus                   # H100, through devq
```

## S5 review fixes (2026-10-25)

Blocking findings (all three real; fixed)
1. `loss_partial` (the per-chunk weighted loss sum behind train/loss) had no test. tests/s5 now passes it in every
   call of `run` and checks it in grad_V*, bitwise (every chunk size, repeat, forward-only, B1 = B1.5 = B2) and
   odd_shapes (tail chunks): bitwise equal to a torch emulation of `k_ce_chunk_loss` (1,024 thread-strided f32 sums,
   then the fixed tree; `nmce.chunk_loss_ref`) and rel <= 1e-6 to the f64 per-chunk sum of w x loss_tok. Mutation M8
   (`*out = red[1]`) now fails grad_V512, bitwise and odd_shapes (rel 0.51).
2. The accurate `accumulate_dx` was not told apart from the levanter form. The stronger test found a real defect:
   on the H100 (t72 log), A/lt and B1/lt gave the levanter bits for the accurate form (0.00 operand ulp from
   bf16(old + bf16(acc)), the same rel 1.706e-3), so Hopper cuBLASLt with beta = 1 and a bf16 C / D rounds the product
   before the add. Fix (ce_api.cpp `dgrad_plain`): an accurate accumulation always runs the own dgrad kernel (both
   forms are explicit in its epilogue); cuBLASLt keeps beta = 0 only. accumulate_dx now also requires that the accurate
   result differs from the levanter form in > 0.1 % of the elements, that its rel to f64 is strictly lower, and that
   B1, B1.5 and B2 on both GEMM backends give the same accurate bits. A100: 11.37 % of the elements differ, rel
   1.659e-3 against 1.702e-3. Mutation M12 (the own dgrad always rounds first) now fails every row.
3. `nm_ce_workspace_bytes` sized the bf16-io layout, but `carve` placed an f32 dX scratch for io = f32, so the
   cuBLASLt region could run past the caller's buffer. The f32 policy never uses dxtmp (cuBLASLt beta, f32 output), so
   `carve` no longer places it for io = f32: the f32-io layout is never larger than the bf16-io one that the C ABI
   reports. The workaround in tests/s5/nmce.py is removed; f32_io runs in a workspace of exactly
   nm_ce_workspace_bytes followed by a guard band that must stay untouched.

Minor findings
- H100 provenance: the reviewer matched the sm_90a SASS of the t72 / t74 snapshot (built before the last
  ce_sm90.cu edit) to the final S5 tree, so t72 / t74 held for S5. This fix changes ce_sm90.cu again; t87 reruns
  every gate on the library of this fix (bench/s5_ce/snap/fix1, built after the last source edit).
- Fast statistics exp on the ragged last vocab tile: fixed (`stats_rows<false, true>` when `fast_stats_exp`), so the
  opt-in mode applies to every tile on both kernels. cuobjdump: LOCAL 0 for every sm_90a CE kernel (also in the
  t72 snapshot; the 4-B spill of the S5 known gaps is not in these builds).
- sm90_gemm docstring: B1.5's dgrad runs the sm_90a kernel (write_dz); corrected.
- `launch_dgrad_sm90` / `launch_wgrad_sm90` check 16-byte alignment of lse, gscale and targets (the 1-D TMA maps)
  and return false (portable kernel) instead of throwing. odd_shapes adds V = 1,000, D = 512, chunks of 1,022 rows
  (chunk offsets not a multiple of 4 rows), which takes this path on the H100.
- Test margins: check_against adds a 1.05x bar against the emulation of the same rounding policy: dX against the
  levanter (or f32-store) emulation, dW against a new f32-dW emulation of the same dz (`levanter_emulation(f32_dw=True)`;
  the port's dW is 0.999-1.000x it on the A100, against 0.82-0.97x the levanter-dW emulation). Worst dX on the A100:
  1.039x (V = 128,256, D = 768, f32 store; t72 H100: 1.025x). odd_shapes drops the absolute floors (`or rel < 4e-3`):
  every shape passes the plain oracle rule, worst 1.000x the emulation.
- cuBLASLt plans (ce_lt.cpp): at most 256 plans per device; past that, the device's plans are dropped (the canonical
  algorithms stay). Safe under one host thread per device.
- grad_scale: the spec's 128 x 4096 weights (last position 0) give f32(1 / 524160) bitwise, 256 x 4096 give
  f32(1 / 1048320), and f32(1 / 524160) = 2 f32(1 / 1048320).
- Not changed: run_readout eps = 1e-5 and the planner's ws.ce.* sizes (requests in S5 stand).

Gates
- della-vis1 A100, fresh configure and full build: 0 warnings, 0 errors. s5_ce: 14 pass, 3 skip (sm90_vs_portable,
  sm90_gemm, two_devices). s5_t15_levref: pass (130m, 1_2b; grad_x 1.014x levanter's error).
- Full ctest (46 entries): 43 pass; s4_graph_nccl skips (no NCCL); s3_t1_levref fails on S3's open items and
  s6_t1_7_small_lr reports S6's SPECFAIL (item 15), neither in S5 code.
- Mutants built from a copy of the tree: M8 and M12 fail (above).
- compute-sanitizer memcheck (`--kernel-name kns=2nm2ce`) on f32_io, accumulate_dx, inplace_dx, strict_dw, readout,
  grad_scale: 0 errors. Racecheck and synccheck were not rerun: no portable kernel changed.
- Scrub check: clean on src/kernels/cross_entropy, tests/s5, docs/BUILD_LOG.md.
- H100, devq t87 (2 GPUs, the library and tests of this fix): s5_ce 17 of 17 pass (two_devices, sm90_vs_portable,
  sm90_gemm and the chunk-1,022 shape included); accumulate_dx: accurate differs from the levanter form in 11.39 %
  of the elements on every stage and backend, rel 1.663e-3 against 1.706e-3; worst dX 1.025x the emulation, dW
  1.000x the f32-dW emulation. s5_t15_levref passes (130m, 1_2b; grad_x 1.000x levanter's error). Quick bench,
  16,384 x 512 x 128,256: B1 13.40 ms (0.988x the 13.57 ms bar; S5: 13.57), so the fixes cost no time.
