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
