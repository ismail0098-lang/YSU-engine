# SASS Reverse Engineering Results — RTX 4070 Ti Super (SM 8.9, Ada Lovelace)

First-party measurements taken with CUDA 13.1 on Windows.

## 2026-03-19 Recursive Sweep Addendum

The SASS RE toolkit is no longer limited to the legacy top-level probe list.
The current automation walks the full recursive probe tree via
[`probe_manifest.py`](scripts/probe_manifest.py).

Current manifest-backed corpus:

- 349 recursive probe files
- 347 non-skip manifest entries
- 343 compile-enabled manifest entries
- manifest runner kinds: 333 `plain`, 2 `texture_surface`,
  1 `cp_async_zfill`, 1 `mbarrier`, 1 `barrier_arrive_wait`,
  1 `barrier_coop_groups`, 1 `cooperative_launch`, 1 `tiling_hierarchical`,
  1 `depbar_explicit`, 1 `optical_flow`, 1 `optix_pipeline`,
  1 `optix_callable_pipeline`, 1 `video_codec`, 1 `cudnn`, and 2 `skip`

## 2026-03-20 Full Recursive Refresh

Canonical aggregate run:
[`src/sass_re/results/runs/full_recursive_20260320_142438`](results/runs/full_recursive_20260320_142438)

Note:
- The current manifest is slightly larger than this run. Two final-cluster
  follow-up probes were added after `full_recursive_20260320_142438`, so the
  run-level compile-enabled total is `341` while the current manifest total is
  `343`.

| Phase / lane | Pass | Fail | Mnemonics | New | Spills | Notes |
|---|---|---|---|---|---|---|
| Disassembly | 341 | 0 | -- | -- | -- | Full compile-enabled corpus |
| baseline | 341 | 0 | 379 | 0 | 16 | Canonical optimized reference |
| `-O2` | 341 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O2 -Xptxas -O3` | 341 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O3` | 341 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O3 -Xptxas -O3` | 341 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O0 -G` | 341 | 0 | 351 | 87 | 3728 | Debug-only support and control-flow forms |
| `-O0 -G -Xptxas -O3` | 0 | 341 | -- | -- | -- | Expected ptxas rejection on this toolchain |
| `--maxrregcount={64,128,255}` | 341 | 0 | 380 | 2 | 16 | `UISETP.GE.U32.AND`, `UISETP.GT.AND` |
| `--restrict` | 341 | 0 | 381 | 4 | 40 | `I2F.S8`, `LDG.E.U16.CONSTANT`, `LDG.E.U8.CONSTANT`, `LDL.LU` |
| `-O3 --use_fast_math --restrict` | 341 | 0 | 375 | 15 | 16 | FTZ-family variants plus `MUFU.SQRT` |
| compile/profile lane | 341 | 0 | -- | -- | -- | Full `O3` compile-profile pass complete |
| `ncu` | 314 | 16 | -- | -- | -- | 16 expected/runner skips remain |

Newly folded into the checked-in canonical inventory by this refresh:

- `UISETP.GE.U32.AND`
- `UISETP.GT.AND`
- `I2F.S8`
- `LDG.E.U16.CONSTANT`
- `LDG.E.U8.CONSTANT`
- `LDL.LU`

The final refresh also re-confirms that the strongest direct-local source/IR
frontier is now bounded by `P2R.B1`, `P2R.B2`, `P2R.B3`, and `UPLOP3.LUT`.
Base `R2P` is already directly observed in the local corpus via the
transcendental compile-profile path, the newer exact follow-up
`predicate_uniform_frontier_20260321_024500` directly confirms
`USHF.L.U64.HI`, and direct local `ULOP3.LUT` is already present in the
uniform-path corpus. Cubin-side substitution now materializes and runs
`P2R.B1/B2/B3`, and a newer `PLOP3 -> UPLOP3` cubin sketch now decodes and
runs cleanly in multiple local contexts on `sm_89`, even though
`UPLOP3.LUT` still does not appear verbatim in direct source/IR output. The
normalized `UPLOP3` runtime matrix now shows a real semantic split: `0x80`
stays inert across all validated local contexts so far, while the
cooperative-groups `0x8` site is the first stable-but-different local
`UPLOP3` case. The denser `probe_uplop3_uniform_predicates` branch now also
shows semantically live `UPLOP3(0x80)` and `UPLOP3(0xfe)` behavior, while
`UPLOP3(0xa8)` remains inert and the mixed `0xa8/0x2a` sliding-window cluster
stays fully inert. A newer occurrence and combo sweep in that same uniform-
predicate kernel sharpens the boundary again: `occ1(0xfe)`, `occ2(0xfe)`, and
`occ5(0x80)` are live, `occ3(0xfe)` and `occ4(0xa8)` are inert, and the live
sites compose into multiple distinct output families rather than a single
additive effect. A matching occurrence sweep on `probe_cg_coalesced` then
shows the cooperative branch is site-specific too: only the first `0x8` site
is live, the later `0x80` site is inert, and patching both sites collapses to
the exact same behavior as patching the first site alone. A newer CUTLASS-like
predicate-pipeline sweep now adds a second strong live family:
`occ2(0xa8)`, `occ4(0x80)`, and `occ5(0xa8)` are all semantically live in
`probe_cutlass_predicate_pipeline`, and `cutlass_occ5` now ties `uniform_occ1`
for the strongest exact live local match to a mined library `UPLOP3` motif
(`best_jaccard = 0.375`). The CUTLASS combo law is also sharper than the
cooperative case: `occ2_occ4` collapses to the `occ2` family on patterns
`0/2/3` and to the `occ4` family on pattern `1`, while `occ4_occ5` and
`occ2_occ4_occ5` retain the `occ5`-like visible outputs on patterns `0/2/3`
but still diverge in total sum. A first targeted `ncu` comparison shows only
weak perf-side separation for these live CUTLASS cases, so `ncu` is useful as
a sanity check but not the main discovery tool; `compute-sanitizer` is the
stronger sidecar here, and now clears both the baseline and the richest live
combo `occ2_occ4_occ5` with `0 errors`. A newer tandem workflow is now wired in
for `UPLOP3` frontier work:
- semantic anchor or baseline run
- `ncu`
- `compute-sanitizer`
- `nsys`
- seeded differential fuzzing

The first tandem anchor runs are:
- [`uplop3_uniform_tandem_20260323_092500`](results/runs/uplop3_uniform_tandem_20260323_092500)
- [`uplop3_cutlass_tandem_20260323_092500`](results/runs/uplop3_cutlass_tandem_20260323_092500)

These sharpen the tool split:
- seeded differential fuzzing is the strongest semantic discriminator
- `compute-sanitizer` is the safety gate
- `ncu` is a weak but still useful perf sanity check
- `nsys` is now captured by default for later timeline mining

They also sharpen the ranking of the live local branches:
- uniform anchor `occ1` is relatively stable, and `occ1_occ5` stays close to it
  under seeded fuzz (`21/24` unchanged)
- CUTLASS anchor `occ5` broadens more when combined with the other live sites,
  and `occ2_occ4_occ5` is the broadest semantic widener so far
  (`18/24` seeded cases with diffs)

A follow-up second-tier tandem tranche then refined the live-site ranking:
- [`uplop3_uniform_occ2_tandem_20260323_094000`](results/runs/uplop3_uniform_occ2_tandem_20260323_094000)
- [`uplop3_cutlass_occ4_tandem_20260323_094000`](results/runs/uplop3_cutlass_occ4_tandem_20260323_094000)

That result says:
- `uniform_occ2` behaves like a credible secondary anchor:
  `occ2_occ5` stays close to it (`21/24` unchanged)
- `cutlass_occ4` is semantically live but modifier-heavy:
  `occ4_occ5` is extremely broad (`23/24` seeded cases with diffs)

A richer anchor tranche then tightened the top of the ranking:
- [`uplop3_uniform_occ1_rich_tandem_20260323_095500`](results/runs/uplop3_uniform_occ1_rich_tandem_20260323_095500)
- [`uplop3_uniform_occ2_rich_tandem_20260323_095500`](results/runs/uplop3_uniform_occ2_rich_tandem_20260323_095500)
- [`uplop3_cutlass_occ5_rich_tandem_20260323_095500`](results/runs/uplop3_cutlass_occ5_rich_tandem_20260323_095500)

Those runs show:
- `uniform_occ1` remains a real anchor, but `occ1_occ2_occ5` broadens it more
  than `occ1_occ2`
- `uniform_occ2` remains unexpectedly stable with `occ2_occ5`
  (`21/24` unchanged)
- `cutlass_occ5` remains the strongest CUTLASS-like anchor
- `occ2_occ4_occ5` remains the broadest CUTLASS widener (`18/24` diffs)

A partner-centered follow-up then clarified the causal roles:
- [`uplop3_uniform_occ5_tandem_20260323_101500`](results/runs/uplop3_uniform_occ5_tandem_20260323_101500)
- [`uplop3_cutlass_occ2_tandem_20260323_101500`](results/runs/uplop3_cutlass_occ2_tandem_20260323_101500)

That result says:
- on the uniform branch, `occ5` behaves more like a sensitizer than an anchor
- on the CUTLASS branch, `occ5` is the main trigger of the semantic explosion
- `cutlass_occ4` is better understood as an amplifier than the root cause

A sharper pair-baseline tranche then treated `occ2_occ5` itself as the baseline
state on both branches:
- [`uplop3_uniform_pair_baseline_20260323_094600`](results/runs/uplop3_uniform_pair_baseline_20260323_094600)
- [`uplop3_cutlass_pair_baseline_20260323_094718`](results/runs/uplop3_cutlass_pair_baseline_20260323_094718)
- [`uplop3_uniform_pair_baseline_20260323_094600__uplop3_cutlass_pair_baseline_20260323_094718__pair_summary.txt`](results/runs/uplop3_uniform_pair_baseline_20260323_094600__uplop3_cutlass_pair_baseline_20260323_094718__pair_summary.txt)

That result says:
- on the uniform branch, `occ1` remains the true extra widener, but the full
  triple `occ1_occ2_occ5` is less disruptive than `occ1` or `occ1_occ5` alone
- on the CUTLASS branch, `occ4` is the strongest visible widener against the
  stable pair baseline, while `occ4_occ5` and `occ2_occ4_occ5` more often act
  as aggregate-state wideners than prefix-level wideners
- `compute-sanitizer` remains clean on the pair-baseline cases, and `ncu`
  remains a secondary regime check rather than the main ranking signal

## 2026-03-20 Postfix Recursive Refresh

Postfix aggregate run:
[`src/sass_re/results/runs/full_recursive_20260320_182500`](results/runs/full_recursive_20260320_182500)

This rerun folds in the two late final-cluster probes plus the generic-runner
fixes.

| Phase / lane | Pass | Fail | Mnemonics | New | Spills | Notes |
|---|---|---|---|---|---|---|
| Disassembly | 343 | 0 | -- | -- | -- | Current compile-enabled corpus snapshot |
| baseline | 343 | 0 | 379 | 0 | 16 | Canonical optimized reference still stable |
| `--maxrregcount=32` | 343 | 0 | 382 | 3 | 16 | `UISETP.EQ.U32.XOR`, `UISETP.GE.U32.AND`, `UISETP.GT.AND` |
| `--maxrregcount=64/128/255` | 343 | 0 | 380 | 2 | 16 | `UISETP.GE.U32.AND`, `UISETP.GT.AND` |
| `--restrict` | 343 | 0 | 381 | 4 | 40 | `I2F.S8`, `LDG.E.U16.CONSTANT`, `LDG.E.U8.CONSTANT`, `LDL.LU` |
| postfix `ncu` | 331 | 3 | -- | -- | -- | Major coverage improvement over prior run |

Postfix `ncu` cleanup after the aggregate run:

- [`src/sass_re/results/runs/final_ncu_tail_retest_20260320`](results/runs/final_ncu_tail_retest_20260320)
  clears all three remaining direct failures from the postfix run:
  `probe_control_flow.cu`, `probe_tiling_2d_stencil.cu`, and
  `probe_uniform_stage_toggle_pipeline.cu`.
- That means the postfix aggregate summary `331 profiled / 3 failed / 14 skipped`
  is now known to be reducible to an all-clear on those three probes with the
  current local code and runner fixes.

Post-refresh cleanup and final direct follow-ups:

- Runner-fix validation under
  [`src/sass_re/results/runs/runner_fix_validation_20260320/summary.txt`](results/runs/runner_fix_validation_20260320/summary.txt)
  converts the old `UNSUPPORTED RUNNER` cases for `probe_mx_microscaling.cu`,
  `probe_uniform_exotic.cu`, `probe_uniform_strict_address.cu`,
  `probe_uniform_u64_strict.cu`, `probe_uplop3_uniform_predicates.cu`, and the
  `data_movement2` local-width probes into clean generic-runner `ncu` passes.
- Custom-runner validation under
  [`src/sass_re/results/runs/custom_runner_validation_20260321_012900/summary.txt`](results/runs/custom_runner_validation_20260321_012900/summary.txt)
  now converts `probe_barrier_arrive_wait.cu`,
  `probe_barrier_coop_groups_sync.cu`, and
  `probe_tiling_hierarchical.cu` into clean dedicated-runner `ncu` passes.
  The barrier probe was rewritten to use balanced named barriers, and the
  cooperative-groups runner intentionally omits the still-unsafe multilevel
  kernel from runtime execution while preserving it for disassembly.
- Runner-tail validation under
  [`src/sass_re/results/runs/runner_tail_validation_20260321_014800/summary.txt`](results/runs/runner_tail_validation_20260321_014800/summary.txt)
  now converts `probe_cooperative_launch.cu` and
  `barrier_sync2/probe_depbar_explicit.cu` into clean dedicated-runner `ncu`
  passes. The cooperative runner launches the grid-sync kernel via
  `cudaLaunchCooperativeKernel` when the device reports support.
- Failure-retest validation under
  [`src/sass_re/results/runs/ncu_fail_retest_20260320/summary.txt`](results/runs/ncu_fail_retest_20260320/summary.txt)
  now clears representative old `ncu FAIL` cases including
  `probe_barrier_membar_variants.cu`, `probe_tiling_scatter_gather.cu`,
  `shared_atomics/probe_satom_int32_add.cu`,
  `data_movement/probe_dm_gather_scatter_idx.cu`, and
  `edge_atomics2/probe_edge2_local_atomic.cu`.
- The final direct predicate/uniform tranche under
  [`src/sass_re/results/runs/final_cluster_followup_20260320/summary.txt`](results/runs/final_cluster_followup_20260320/summary.txt)
  strengthens the negative evidence: the optimized local path still does not
  reproduce `P2R.B1/B2/B3`, `UPLOP3.LUT`, or `USHF.L.U64.HI`. The
  debug lane again shows only the nearby neighborhood `P2R`, `PLOP3.LUT`, and
  `SHF.L.U64.HI`.
- An exact predicate/uniform frontier tranche under
  [`src/sass_re/results/runs/predicate_uniform_frontier_20260321_031500/summary.txt`](results/runs/predicate_uniform_frontier_20260321_031500/summary.txt)
  reconciles an important erratum and narrows the true remaining gap. `R2P`
  is already directly observed locally in `probe_transcendentals.sass` and
  `probe_fp64_transcendentals.sass`, while the new exact probes still do not
  surface `P2R.B1/B2/B3`, `UPLOP3.LUT`, or `USHF.L.U64.HI`. The tranche does
  reproduce stronger nearby local forms such as `ULOP3.LUT`, `USEL`,
  `UISETP.NE.U32.AND`, `ULDC.64`, `HMMA.1688.F32.TF32`, and
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`.
- A targeted documented-flag hunt on the highest-yield subset under
  [`src/sass_re/results/runs/high_yield_flag_hunt_20260320/summary.txt`](results/runs/high_yield_flag_hunt_20260320/summary.txt)
  shows that:
  - `-O0 -G -dopt=on`, `--extra-device-vectorization`,
    `--extra-device-vectorization --restrict`, and
    `-Xptxas -disable-optimizer-consts` do not move the local frontier on that
    subset
  - `-O3 --use_fast_math --extra-device-vectorization --restrict` only adds the
    expected FTZ-family spellings
  - `-Xptxas -dlcm=cg` is the only new documented knob in that subset to expose
    additional raw load spellings, specifically `LDG.E.STRONG.GPU`,
    `LDG.E.S8.STRONG.GPU`, and `LDG.E.U8.STRONG.GPU`
- A dedicated multi-translation-unit `-dlto` tranche under
  [`src/sass_re/results/runs/dlto_tranche_20260320_213954/summary.txt`](results/runs/dlto_tranche_20260320_213954/summary.txt)
  now validates real cross-TU device link-time optimization on the local Ada
  toolchain. Relative to the no-LTO build, the with-LTO build:
  - removes helper call plumbing from the kernel body
    (`CALL.ABS.NOINC`, `RET.ABS.NODEC`)
  - collapses the helper functions into the main `probe_dlto_cross_tu` kernel
  - surfaces `IMAD.X` in the fused with-LTO kernel body
  - drops debug-style reconvergence helpers such as `BSSY` and `BSYNC` from
    the no-LTO split-function path
- A focused runtime-performance tranche under
  [`src/sass_re/results/runs/runtime_perf_tranche_20260320_214330/run.log`](results/runs/runtime_perf_tranche_20260320_214330/run.log)
  now turns three Ada-side execution levers into measured local evidence:
  - `cudaAccessPolicyWindow` / persisting L2:
    `l2_persist_off_ms=0.006321`, `l2_persist_on_ms=0.005586`,
    `speedup=1.131645`
  - CUDA Graph replay vs plain launches:
    `graph_plain_launch_us=6.526640`,
    `graph_replay_launch_us=5.958656`,
    `speedup=1.095321`
  - synchronous staged shared-memory copy vs `cp.async` overlap:
    `pipeline_sync_ms=0.019738`,
    `pipeline_async_ms=0.024704`,
    `speedup=0.798964`
  - Interpretation:
    the local hot-window and launch-overhead levers are immediately useful,
    while the current async overlap microkernel is not yet compute-heavy enough
    to amortize its staging overhead
- A focused HMM / Unified Memory tranche under
  [`src/sass_re/results/runs/runtime_hmm_tranche_20260321_020217/summary.txt`](results/runs/runtime_hmm_tranche_20260321_020217/summary.txt)
  now turns the already-enabled desktop HMM path into measured local evidence:
  - platform capability:
    `managed_memory=1`, `concurrent_managed_access=1`,
    `pageable_memory_access=1`, `direct_managed_mem_access_from_host=0`
  - cold managed-memory GPU access after host residency:
    `gpu_cold_ms=23.549952`
  - explicit prefetch to GPU on the same 64 MiB working set:
    `gpu_prefetch_ms=0.641024`, `speedup=36.738019`
  - `cudaMemAdviseSetPreferredLocation + cudaMemAdviseSetReadMostly` plus
    prefetch on this write-heavy kernel:
    `gpu_advise_prefetch_ms=5145.206055`, `speedup=0.004577`
  - host touch after GPU residency:
    `host_touch_cold_ms=342.847000`
  - explicit prefetch back to CPU before the same host touch:
    `host_touch_prefetch_ms=103.657000`, `speedup=3.307514`
  - Interpretation:
    explicit prefetch is highly effective on this local HMM path, while the
    chosen advice mix is catastrophically wrong for a write-heavy managed
    kernel and should not be generalized as a default optimization
- A focused GreenBoost runtime tranche under
  [`src/sass_re/results/runs/runtime_greenboost_tranche_20260321_082925/summary.txt`](results/runs/runtime_greenboost_tranche_20260321_082925/summary.txt)
  now turns the upstream March 2026 GreenBoost release into measured local
  evidence on this Ada desktop:
  - local packaging/install path:
    `~/Github/pkgbuilds/greenboost-dkms`
  - install result:
    DKMS builds and installs cleanly for both local kernels, the shim is
    available as `/usr/lib/libgreenboost_cuda.so`, and `modprobe greenboost`
    successfully creates `/dev/greenboost`
  - baseline 512 MiB `cudaMalloc`:
    `alloc_ms=0.153111`, `first_kernel_ms=11.966464`,
    `second_kernel_ms=13.340544`
  - forced Path A (`DMA-BUF + greenboost.ko`):
    `alloc_ms=916.656588`, `resolved_path=DMA_BUF_FAILED`
  - forced Path B (`HostReg`):
    `alloc_ms=728.332168`, `resolved_path=HOSTREG_TO_UVM`
  - forced Path C (`UVM`):
    `alloc_ms=0.126451`, `resolved_path=UVM`
  - Local path interpretation:
    the shim does interpose and inflate the CUDA/NVML memory view, but on
    this host Path A currently fails at `cuMemHostGetDevicePointer` with CUDA
    driver error `201` (`CUDA_ERROR_INVALID_CONTEXT`), Path B degrades into
    UVM after a large setup penalty, and Path C is the only cleanly resolved
    overflow path in this controlled local configuration
- A follow-up driver-API-native tranche under
  [`src/sass_re/results/runs/runtime_greenboost_driver_tranche_20260321_083427/summary.txt`](results/runs/runtime_greenboost_driver_tranche_20260321_083427/summary.txt)
  sharpens that interpretation:
  - baseline driver allocation:
    `alloc_ms=0.147430`
  - forced Path A:
    `alloc_ms=232.111265`, `resolved_path=DMA_BUF_FAILED`
  - forced Path B:
    `alloc_ms=190.237577`, `resolved_path=HOSTREG_TO_UVM`
  - forced Path C:
    `alloc_ms=0.174110`, `resolved_path=UVM`
  - Refined interpretation:
    the initial device-node permission issue was real, but once Path A is run
    with sufficient privilege the true local failure still remains
    `cuMemHostGetDevicePointer(...)=201`. So the current local GreenBoost
    frontier is not “DMA-BUF permissions,” but a deeper Path A interop/context
    problem after successful overflow routing into the DMA-BUF registration
    path
- A newer context-mode and raw-hostreg follow-up under
  [`src/sass_re/results/runs/runtime_greenboost_driver_tranche_ctxmodes_20260321_085750/summary.txt`](results/runs/runtime_greenboost_driver_tranche_ctxmodes_20260321_085750/summary.txt)
  and
  [`src/sass_re/results/runs/runtime_hostreg_probe_20260321_085842/summary.txt`](results/runs/runtime_hostreg_probe_20260321_085842/summary.txt)
  tightens that diagnosis:
  - the driver runner reports `ctx_flags=0x8`, `ctx_has_map_host=1`, and
    `can_map_host_memory=1` in both the primary-context and explicit
    user-context cases
  - Path A still fails at `cuMemHostGetDevicePointer(...)=201` in both
    context modes
  - a raw host-registration probe outside the shim succeeds for
    `mmap(MAP_ANONYMOUS)`, `malloc`, and `mlock`-pinned host memory on the
    same host
  - Refined interpretation:
    the unresolved GreenBoost Path A / Path B failure is now strongly bounded
    to the shim's overflow implementation or its DMA-BUF / pin-user-ptr
    handling, not to a general local inability to map registered host memory
    into a CUDA device pointer
- A symbol-level follow-up under
  [`src/sass_re/results/runs/runtime_hostreg_symbol_probe_20260321_090753/summary.txt`](results/runs/runtime_hostreg_symbol_probe_20260321_090753/summary.txt)
  then isolates the exact compatibility fault:
  - the legacy unsuffixed `cuMemHostGetDevicePointer` symbol returns
    `CUDA_ERROR_INVALID_CONTEXT` on this local driver stack
  - `cuMemHostGetDevicePointer_v2` succeeds on the same registered host
    allocation
  - Interpretation:
    GreenBoost's failing local Path A / Path B behavior came from using the
    legacy symbol path rather than the `_v2` entry point that matches the
    current CUDA header/runtime mapping
- `greenboost-dkms 2.5-3` now carries a local patch that prefers
  `cuMemHostRegister_v2` and `cuMemHostGetDevicePointer_v2`.
- With that patch installed, both GreenBoost overflow paths now work locally:
  - driver-native validation:
    [`src/sass_re/results/runs/runtime_greenboost_driver_tranche_v2pref_20260321_091139/summary.txt`](results/runs/runtime_greenboost_driver_tranche_v2pref_20260321_091139/summary.txt)
    shows `resolved_path=DMA_BUF` for Path A and `resolved_path=HOSTREG` for
    Path B
  - runtime-API validation:
    [`src/sass_re/results/runs/runtime_greenboost_tranche_v2pref_20260321_091159/summary.txt`](results/runs/runtime_greenboost_tranche_v2pref_20260321_091159/summary.txt)
    shows the same successful path resolution
  - current 512 MiB runtime-path measurements after the fix:
    - Path A (`DMA_BUF`):
      `alloc_ms=124.330160`, `first_kernel_ms=164.898819`,
      `second_kernel_ms=163.566589`
    - Path B (`HOSTREG`):
      `alloc_ms=112.992360`, `first_kernel_ms=159.618149`,
      `second_kernel_ms=161.120895`
    - Path C (`UVM`):
      `alloc_ms=0.181220`, `first_kernel_ms=11.945984`,
      `second_kernel_ms=12.315648`
  - Interpretation:
    the compatibility bug is fixed, but the DDR-backed GreenBoost tiers remain
    far slower than real VRAM residency for this bandwidth-heavy test, so the
    next phase is performance characterization rather than basic bring-up.
- A stronger multi-pattern GreenBoost performance tranche under
  [`src/sass_re/results/runs/runtime_greenboost_perf_tranche_20260321_094142/summary.txt`](results/runs/runtime_greenboost_perf_tranche_20260321_094142/summary.txt)
  now shows how that penalty changes with access regime:
  - baseline VRAM path at 256 MiB:
    - `stream_rw`: `5.863424 ms` (`341.10 GiB/s`)
    - `read_reduce`: `2.906112 ms` (`344.10 GiB/s`)
    - `stride_rw`: `2.011136 ms` (`31.08 GiB/s`)
    - `compute_heavy`: `5.997696 ms` (`333.46 GiB/s`)
    - `compute_very_heavy`: `57.416962 ms` (`34.83 GiB/s`)
  - Path A (`DMA_BUF`):
    - `stream_rw`: `86.643715 ms` (`23.08 GiB/s`, `14.78x` slower)
    - `read_reduce`: `50.264065 ms` (`19.89 GiB/s`, `17.30x` slower)
    - `stride_rw`: `58.052608 ms` (`1.08 GiB/s`, `28.87x` slower)
    - `compute_heavy`: `85.265411 ms` (`23.46 GiB/s`, `14.22x` slower)
    - `compute_very_heavy`: `88.094719 ms` (`22.70 GiB/s`, `1.53x` slower)
  - Path B (`HOSTREG`):
    - `stream_rw`: `86.907906 ms` (`23.01 GiB/s`, `14.82x` slower)
    - `read_reduce`: `53.073921 ms` (`18.84 GiB/s`, `18.26x` slower)
    - `stride_rw`: `62.979073 ms` (`0.99 GiB/s`, `31.32x` slower)
    - `compute_heavy`: `91.686653 ms` (`21.81 GiB/s`, `15.29x` slower)
    - `compute_very_heavy`: `84.875267 ms` (`23.56 GiB/s`, `1.48x` slower)
  - Path C (`UVM`):
    - `stream_rw`: `6.427648 ms` (`311.16 GiB/s`, `1.10x` slower)
    - `read_reduce`: `2.907136 ms` (`343.98 GiB/s`, `1.00x`)
    - `stride_rw`: `1.943552 ms` (`32.16 GiB/s`, `0.97x`)
    - `compute_heavy`: `5.878784 ms` (`340.21 GiB/s`, `0.98x`)
    - `compute_very_heavy`: `52.577278 ms` (`38.04 GiB/s`, `0.92x`)
  - Interpretation:
    on this Ada desktop, Path A and Path B now behave much more like two
    flavors of the same DDR-backed PCIe tier than like meaningfully distinct
    performance classes. The large penalty persists for streaming, sparse, and
    moderately compute-amortized kernels, but collapses to about `1.5x` once
    arithmetic intensity is high enough. Path C remains the best local
    overflow path for the tested patterns.
- A first oversubscription-focused tranche under
  [`src/sass_re/results/runs/runtime_greenboost_oversub_tranche_20260321_094432/summary.txt`](results/runs/runtime_greenboost_oversub_tranche_20260321_094432/summary.txt)
  then pushed the working set to `14 GiB`, beyond current free VRAM, and
  exposed a new GreenBoost-specific frontier:
  - requested Path A (`GREENBOOST_USE_DMA_BUF=1`) does not stay on DMA-BUF at
    this size; the shim logs
    `GB_IOCTL_PIN_USER_PTR failed for 14336 MB: Invalid argument` and falls
    back to `HOSTREG`
  - measured local behavior at `14 GiB`:
    - Path A fallback (`HOSTREG`):
      `alloc_ms=10616.608059`, `first_touch_ms=99.994621`,
      `second_touch_ms=78.517250`, `hot_window_first_ms=158.814209`,
      `hot_window_second_ms=161.812485`
    - Path B (`HOSTREG`):
      `alloc_ms=8600.712234`, `first_touch_ms=107.915260`,
      `second_touch_ms=76.322655`, `hot_window_first_ms=152.241150`,
      `hot_window_second_ms=171.457504`
    - Path C (`UVM`):
      `alloc_ms=0.219720`, `first_touch_ms=2631.634766`,
      `second_touch_ms=36085.496094`, `hot_window_first_ms=133.934082`,
      `hot_window_second_ms=11.553792`
  - Interpretation:
    under real oversubscription, the GreenBoost DDR tiers are still slow but
    predictable, while UVM enters a sharply different migration regime:
    first-touch is expensive, full-range revisits can become disastrous, but a
    repeatedly reused hot window can recover strongly after residency settles.
    The next GreenBoost frontier is therefore twofold:
    large-allocation DMA-BUF eligibility, and controlled hot-window versus
    full-range oversubscription behavior.
- A dedicated DMA-BUF size sweep under
  [`src/sass_re/results/runs/runtime_greenboost_dmabuf_size_sweep_20260321_094708/summary.txt`](results/runs/runtime_greenboost_dmabuf_size_sweep_20260321_094708/summary.txt)
  now turns that first question into measured local evidence:
  - `4 GiB` remains real `DMA_BUF`
  - `8 GiB` fails `GB_IOCTL_PIN_USER_PTR` with `Invalid argument` and falls
    back to `HOSTREG`
  - `10 GiB`, `12 GiB`, and `14 GiB` behave the same way
  - Interpretation:
    the local GreenBoost DMA-BUF path currently has a size eligibility limit
    somewhere between `4 GiB` and `8 GiB`. Above that, the practical behavior
    is HostReg fallback, not persistent DMA-BUF-backed overflow.
- A tighter bisect under
  [`src/sass_re/results/runs/runtime_greenboost_dmabuf_size_sweep_20260321_104547/summary.txt`](results/runs/runtime_greenboost_dmabuf_size_sweep_20260321_104547/summary.txt)
  sharpens that bound:
  - `4.125 GiB`, `4.25 GiB`, and `4.375 GiB` all fail
    `GB_IOCTL_PIN_USER_PTR` with `Invalid argument`
  - Interpretation:
    the current local DMA-BUF eligibility limit is now tightly bounded to
    `4.0 GiB` passing and `4.125 GiB` failing.
- Local source inspection then resolves the cause directly:
  in `/usr/src/greenboost-2.5/greenboost.c`, the `GB_IOCTL_PIN_USER_PTR` path
  explicitly returns `-EINVAL` when
  `req.size > (u64)virtual_vram_gb * (1ULL << 30)`.
  - Because the sweep loaded the module with `virtual_vram_gb=4`, the
    apparent DMA-BUF cliff was really a module configuration cap.
  - A targeted rerun with `virtual_vram_gb=8` and an `8 GiB` request stayed on
    real `DMA_BUF`, confirming that the earlier `8 GiB` fallback was policy,
    not a fundamental DMA-BUF limitation.
- A two-dimensional policy sweep under
  `src/sass_re/results/runs/runtime_greenboost_policy_surface_20260321_112937`
  now makes that envelope explicit:
  - `virtual_vram_gb=4`: `4.0 GiB` -> `DMA_BUF`, `4.125 GiB+` -> `HOSTREG`
  - `virtual_vram_gb=6`: `6.0 GiB` -> `DMA_BUF`, `8.0 GiB` -> `HOSTREG`
  - `virtual_vram_gb=8`: `8.0 GiB` -> `DMA_BUF`
  So the local Path A limit is policy-bounded, not hardware-bounded.
- A first Nsight Systems tranche under
  `src/sass_re/results/runs/runtime_greenboost_nsys_tranche_20260321_114548`
  then distinguishes the runtime cost modes:
  - Path A (`DMA_BUF`, `14 GiB`) is dominated by `cuMemHostRegister_v2`
    (`12.826 s`) and shows no UVM-style migration entries in the GPU memory
    summary.
  - Path C (`UVM`, `14 GiB`) is dominated by `page_touch_kernel`
    (`10.786 s` average) plus many unified-memory migration operations:
    `401106` unified host-to-device copies and `10929` unified
    device-to-host copies.
- A newer three-way tranche under
  `src/sass_re/results/runs/runtime_greenboost_nsys_tranche_20260321_115053`
  refines that split:
  - Path A (`DMA_BUF`, `14 GiB`) is dominated by front-loaded
    `cuMemHostRegister_v2` (`6.186 s`) plus `cuMemHostUnregister`
    (`314.446 ms`), while `page_touch_kernel` itself stays near `80.55 ms`
    and the GPU memory summary still shows no UVM-style migration storm.
  - Path B (`HOSTREG`, `14 GiB`) looks very similar:
    `cuMemHostRegister_v2` (`6.562 s`), `cuMemHostUnregister`
    (`250.275 ms`), `page_touch_kernel` (`79.58 ms`), and again no
    migration-heavy GPU memory summary.
  - Path C (`UVM`, `14 GiB`) remains the distinct migration-driven regime:
    much cheaper allocation/setup, but extremely expensive faulting kernel
    time and hundreds of thousands of unified-memory transfer operations.
  So the local GreenBoost DDR-backed tiers and the managed-memory tier are now
  clearly separated by where they pay their cost: registration up front for
  Path A / Path B, migration-heavy kernel execution for Path C.
- An expanded access-pattern tranche under
  `src/sass_re/results/runs/runtime_greenboost_nsys_tranche_20260321_115747`
  then shows that the two DDR-backed paths are not identical once the active
  subset shape changes:
  - fixed hot-window reuse over `256 MiB` favors `DMA_BUF` over `HOSTREG`
    (`cuMemHostRegister_v2` `6.303 s` vs `20.275 s`, and
    `hot_window_kernel` `890.902 ms` vs `1223.494 ms`)
  - hopping-window pressure with a `64 MiB` window stepping by `512 MiB` makes
    the two DDR-backed paths much closer in registration cost
    (`15.399 s` vs `15.033 s`), though `DMA_BUF` still keeps the smaller
    kernel cost (`92.381 ms` vs `136.769 ms`)
  So the local GreenBoost story is now sharper: Path A and Path B both avoid
  the UVM migration storm, but `DMA_BUF` gains more on concentrated hot-set
  reuse while the gap narrows under colder hopping-window pressure.
- A new reusable chain miner under `src/sass_re/scripts/sass_chain_mine.py`
  now computes mnemonic bigrams, trigrams, and anchor neighborhoods over SASS
  corpora. The first direct-local vs cuDNN comparison under
  `src/sass_re/results/runs/chain_mine_compare_20260321_121200` found that the
  strongest cuDNN anchor windows are already present somewhere in the local run
  corpus. That suggests the remaining frontier is about form selection and
  scheduling pressure within known neighborhoods, not about entirely absent
  hidden neighborhoods.
- A pivot-mining pass under
  `src/sass_re/results/runs/chain_pivot_ldg_uisept_ulop3_20260321_130000`
  now points to the next likely "wombo combo" frontier. The strongest anchor
  windows concentrate around:
  - `ULOP3.LUT` + `UIADD3` + `ULDC(.64/.U8/.S8)` + `USHF.L.U32`
  - `LDGSTS.E.BYPASS.LTC128B.128` + `LDGDEPBAR` + `DEPBAR.LE` +
    `UISETP.*`
  That is a stronger next pivot than broader generic predicate-pack probing.
- A first direct combo probe under
  `src/sass_re/results/runs/combo_wombo_frontier_20260321_131500` already
  lands most of that pivot in local raw SASS:
  - `probe_combo_ulop3_uiadd3_uldc` emits
    `ULDC(.64/.U8) + UIADD3 + USHF.L.U32 + ULOP3.LUT`
  - `probe_combo_ldgsts_depbar_uisept` emits
    `LDGSTS.E.BYPASS.LTC128B.128 + LDGDEPBAR + DEPBAR.LE + ISETP.*`
  The next refinement there is to pull `UISETP.*` into the async/cache half
  of the same neighborhood.
- A newer cache-policy combo under
  `src/sass_re/results/runs/combo_cache_policy_wombo_20260321_130151` then
  lands a stronger direct local hit. In the default lane it emits
  `LDG.E.U8/U16 + LDGSTS.E.BYPASS.LTC128B.128 + LDGDEPBAR + DEPBAR.LE`; under
  `-Xptxas -dlcm=cg` the same kernel upgrades those loads to
  `LDG.E.U8.STRONG.GPU` and `LDG.E.U16.STRONG.GPU` while keeping the full
  async-depbar chain intact.
  That makes the load/cache-policy combo family the strongest next novelty
  frontier.
- A further refinement under
  `src/sass_re/results/runs/combo_cache_policy_wombo_20260321_125711` extends
  that family again. The new `probe_combo_cache_policy_uniform_uisept` kernel
  still does not flip the async/cache half into `UISETP.*`, but it does land a
  richer direct-local chain:
  - default lane:
    `LDG.E + LDG.E.U8 + LDG.E.U16 + LDGSTS.E.BYPASS.LTC128B.128.ZFILL +
    LDGSTS.E.BYPASS.LTC128B.128 + LDGDEPBAR + DEPBAR.LE`
  - `-dlcm=cg` lane:
    `LDG.E.STRONG.GPU + LDG.E.U8.STRONG.GPU + LDG.E.U16.STRONG.GPU +
    LDGSTS.E.BYPASS.LTC128B.128.ZFILL + LDGSTS.E.BYPASS.LTC128B.128 +
    LDGDEPBAR + DEPBAR.LE`
  So the load/cache-policy combo frontier keeps producing direct local wins,
  while the `UISETP` sub-problem now looks like a stricter uniform-domain
  scheduling requirement rather than a missing neighborhood.
- The frontier widens again under
  `src/sass_re/results/runs/combo_warp_atomic_cache_wombo_20260321_131200`
  and `src/sass_re/results/runs/combo_atomic_cache_wombo_20260321_131350`.
  Those runs show that the same async/cache backbone can coexist with:
  `MATCH.ANY`, `REDUX.MIN.S32`, `REDUX.MAX.S32`, `REDUX.SUM(.S32)`,
  `VOTE.ALL`, `VOTE.ANY`, `VOTEU.ANY`, `POPC`, `UFLO.U32`,
  `ATOMG.E.ADD.STRONG.GPU`, and `RED.E.MIN/MAX.S32.STRONG.GPU`.
- A block-reduction extension under
  `src/sass_re/results/runs/combo_blockred_cache_wombo_20260321_131500`
  widens the same family again. It shows that `BAR.RED.POPC/AND/OR` plus
  `B2R.RESULT` can coexist with
  `LDG(.STRONG.GPU) + LDGSTS(.ZFILL) + LDGDEPBAR + DEPBAR.LE + RED.E.ADD`.
- A divergence/reconvergence extension under
  `src/sass_re/results/runs/combo_divergence_cache_wombo_20260321_132800`
  then clarifies the control-flow side of the frontier:
  - optimized `-O3` keeps the async/cache/warp/reduction core
  - `-G` widens that neighborhood with `WARPSYNC`,
    `WARPSYNC.EXCLUSIVE`, `BSSY`, `BSYNC`, `CALL.ABS.NOINC`, and
    `RET.ABS.NODEC`
  This makes reconvergence pressure a real side frontier, but currently a
  debug-lane-weighted one rather than the main optimized novelty path.
- A mixed `ATOMG` non-add extension under
  `src/sass_re/results/runs/combo_atomg_nonadd_cache_wombo_20260321_133400`
  widens the same family again. In one emitted kernel it keeps the
  async/cache backbone while surfacing
  `ATOMG.E.MIN/MAX/AND/OR/XOR/EXCH/CAS.STRONG.GPU`.
- A scope-mix extension under
  `src/sass_re/results/runs/combo_scope_mix_cache_wombo_20260321_133900`
  shows that the same family can also cross scope domains without collapsing:
  the emitted kernel keeps the `LDGSTS/LDGDEPBAR/DEPBAR` backbone while also
  surfacing `ATOMG.E.EXCH.STRONG.SYS`, `ATOMG.E.CAS.STRONG.SYS`, and
  `ATOMG.E.ADD.F32.FTZ.RN.STRONG.SYS`.
- A newer block-red plus system-scope reduction extension under
  `src/sass_re/results/runs/combo_redsys_blockred_cache_wombo_20260321_140200`
  closes the next adjacency question. The emitted kernel keeps the same
  `LDG(.STRONG.GPU) + LDGSTS(.ZFILL) + LDGDEPBAR + DEPBAR.LE` backbone while
  also surfacing:
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `RED.E.MIN.S32.STRONG.SYS`, `RED.E.MAX.S32.STRONG.SYS`,
  `RED.E.ADD.STRONG.SYS`, and `RED.E.ADD.F32.FTZ.RN.STRONG.SYS`.
  That means the combo frontier now spans both block-wide reductions and
  system-scope `RED` inside one direct-local emitted family.
- A follow-up uniform-helper plus system-`RED` extension under
  `src/sass_re/results/runs/combo_uniform_redsys_async_wombo_20260321_140900`
  sharpens the remaining subproblem. It preserves
  `ULDC.64`, `UIADD3`, `ULOP3.LUT`, `USHF.L.U32`, `USHF.L.U64.HI`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`, `LDGDEPBAR`, `DEPBAR.LE`, and
  `RED.E.MIN/MAX.S32.STRONG.SYS` plus `RED.E.ADD(.F32).STRONG.SYS` in one
  emitted kernel, but still does not surface `UISETP.*`.
- A still more literal stage-mask control follow-up under
  `src/sass_re/results/runs/combo_uniform_stage_redsys_wombo_20260321_142200`
  pushes the negative evidence further. This probe copies the strongest known
  local `UISETP`-producing control shape from the uniform HMMA-toggle path,
  but once the body is replaced with the async/cache + system-`RED` family the
  compare side still lowers to ordinary `ISETP.*`. The emitted kernel keeps
  `ULDC(.64)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`, `LDGDEPBAR`,
  `DEPBAR.LE`, and the system-scope `RED.E.*` family, but no `UISETP.*`
  survives in the mixed combo body.
- A 64-bit scope-mix pivot under
  `src/sass_re/results/runs/combo_red64sys_cache_wombo_20260321_144500` and
  `src/sass_re/results/runs/combo_atomg64_cache_wombo_20260321_144500`
  moves the combo frontier again. These probes directly land:
  - `LDG.E.64`
  - `LDG.E.64.STRONG.GPU` under `-Xptxas -dlcm=cg`
  - `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`
  - `LDGDEPBAR`
  - `DEPBAR.LE`
  - `RED.E.ADD.64.STRONG.SYS`
  - `ATOMG.E.CAS.64.STRONG.GPU`
  in the same established async/cache family.
- The chain-mined follow-up under
  `src/sass_re/results/runs/chain_mine_64bit_combo_20260321_145000`
  shows that this is also a real adjacency extension, not just raw mnemonic
  reuse. The strongest new windows are variants of
  `IMAD.X -> LDG.E.64(.STRONG.GPU) -> LDGSTS.E.BYPASS.LTC128B.128.ZFILL ->
  LDGSTS.E.BYPASS.LTC128B.128 -> LDGDEPBAR -> DEPBAR.LE`, plus distinct
  side windows for `RED.E.ADD.64.STRONG.SYS` and
  `ATOMG.E.CAS.64.STRONG.GPU`.
- A block-red plus 64-bit system-`RED` follow-up under
  `src/sass_re/results/runs/combo_blockred_red64sys_cache_wombo_20260321_150000`
  widens the same 64-bit scope-mix family again. In one emitted kernel it now
  keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`, and
  `RED.E.ADD.64.STRONG.SYS`.
- The chain-mined note under
  `src/sass_re/results/runs/chain_mine_blockred_red64sys_combo_20260321_150300`
  confirms that this is also a real adjacency extension: the new windows link
  the 64-bit load/dependency-barrier path with the block-reduction side rather
  than merely placing them in the same file.
- A further 64-bit atomic follow-up under
  `src/sass_re/results/runs/combo_atomg64_exch_cache_wombo_20260321_151200`
  extends the same branch again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`, and `ATOMG.E.EXCH.64.STRONG.GPU`.
  That means the 64-bit scope-mix branch now has both `CAS` and `EXCH`
  represented inside the direct-local async/cache family.
- A still newer 64-bit system-scope atomic follow-up under
  `src/sass_re/results/runs/combo_atomg64sys_cache_wombo_20260321_153200`
  widens that branch again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `ATOMG.E.CAS.64.STRONG.SYS`, `ATOMG.E.EXCH.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  That means the local async/cache family now spans:
  - 64-bit system-scope `RED`
  - 64-bit GPU-scope `ATOMG.CAS`
  - 64-bit GPU-scope `ATOMG.EXCH`
  - 64-bit system-scope `ATOMG.CAS`
  - 64-bit system-scope `ATOMG.EXCH`
  - explicit system-scope fence/control helpers in the same body
- The chain-mined note under
  `src/sass_re/results/runs/chain_mine_atomg64sys_combo_20260321_153600`
  confirms that this newer widening still preserves the same anchor window:
  `IMAD.X -> LDG.E.64(.STRONG.GPU) ->
  LDGSTS.E.BYPASS.LTC128B.128.ZFILL ->
  LDGSTS.E.BYPASS.LTC128B.128 -> LDGDEPBAR -> DEPBAR.LE`,
  with the new downstream extension into
  `ATOMG.E.CAS.64.STRONG.SYS`, `ATOMG.E.EXCH.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
- A direct 64-bit system load/store follow-up under
  `src/sass_re/results/runs/combo_store64sys_cache_wombo_20260321_154200`
  widens the same branch again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `LDG.E.64.STRONG.SYS`, `STG.E.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
- The chain-mined note under
  `src/sass_re/results/runs/chain_mine_store64sys_combo_20260321_154500`
  confirms that this store-side widening preserves the same
  `IMAD.X -> LDG.E.64(.STRONG.GPU) -> LDGSTS -> LDGDEPBAR -> DEPBAR.LE`
  anchor while extending the downstream system side with
  `LDG.E.64.STRONG.SYS`, `STG.E.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
- A 64-bit system-scope atomic op-matrix follow-up under
  `src/sass_re/results/runs/combo_atomg64sys_ops_cache_wombo_20260321_155400`
  widens the same branch substantially. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `ATOMG.E.ADD.64.STRONG.SYS`, `ATOMG.E.MIN.64.STRONG.SYS`,
  `ATOMG.E.MAX.64.STRONG.SYS`, `ATOMG.E.AND.64.STRONG.SYS`,
  `ATOMG.E.OR.64.STRONG.SYS`, `ATOMG.E.XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
- The chain-mined note under
  `src/sass_re/results/runs/chain_mine_atomg64sys_ops_combo_20260321_155800`
  confirms that this op-matrix widening still preserves the same
  `IMAD.X -> LDG.E.64(.STRONG.GPU) -> LDGSTS -> LDGDEPBAR -> DEPBAR.LE`
  anchor, but now feeds a dense downstream 64-bit system-scope `ATOMG` block
  rather than only one or two individual atomic forms.
- A block-red plus 64-bit system-op-matrix follow-up under
  `src/sass_re/results/runs/combo_blockred_atomg64sys_ops_cache_wombo_20260321_160900`
  widens the same branch again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  This is the strongest local scope-mix "wombo combo" family so far.
- A still newer fused warp-side follow-up under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_cache_wombo_20260321_162300`
  widens that family again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `MATCH.ANY`,
  `REDUX.MIN/MAX/SUM(.S32)`,
  `VOTE.ALL`, `VOTE.ANY`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  That is the strongest direct-local combined family in the local corpus so
  far.
- A reconvergence-shaped follow-up under
  `src/sass_re/results/runs/combo_divergent_blockred_warp_atomg64sys_ops_cache_wombo_20260321_163500`
  moves the family again. The optimized lanes keep the same fused async/cache
  + block-red + warp vote/match/redux + dense 64-bit system-ATOMG/control
  body, and now also carry `BSSY` and `BSYNC`.
  The debug lane for the same probe adds `WARPSYNC` and
  `WARPSYNC.EXCLUSIVE`, plus the expected `CALL.ABS.NOINC` /
  `RET.ABS.NODEC` structure.
  So the current boundary is sharper:
  - `BSSY` / `BSYNC` are now part of the optimized fused family
  - `WARPSYNC` remains debug-lane weighted on this source shape
- A narrower optimized `WARPSYNC` chase under
  `src/sass_re/results/runs/combo_warpsync_fused_narrow_20260321_165100`
  tightens that boundary further. Even the least-disruptive full-mask and
  ballot-mask `__syncwarp` variants still keep optimized `BSSY` / `BSYNC`
  rather than selecting raw `WARPSYNC` in the fused family, while the debug
  lane continues to expose `WARPSYNC` and `WARPSYNC.EXCLUSIVE` in helper
  bodies.
  So the current optimized reconvergence endpoint for this branch is:
  - positive: `BSSY` / `BSYNC`
  - negative: no direct optimized `WARPSYNC` yet
- A still newer store-side fused follow-up under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_store_cache_wombo_20260321_170300`
  widens the same family again. In one emitted kernel it now keeps:
  `LDG.E.64(.STRONG.GPU)`, `LDG.E.64.STRONG.SYS`,
  `STG.E.64.STRONG.SYS`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `MATCH.ANY`, `REDUX.MIN/MAX/SUM(.S32)`,
  `VOTE.ALL`, `VOTE.ANY`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  The raw SASS result is positive, but the first dedicated runners for this
  dense SYS branch still hit a runtime `misaligned address` boundary during
  warmup, so this kernel is currently a symbolic/disassembly-positive frontier
  rather than a trustworthy runtime-profile anchor.
- A runtime-safe profiling surrogate under
  `src/sass_re/results/runs/combo_warp_atomic_cache_profile_safe_tranche_20260322_000230`
  now provides the first trustworthy `ncu` anchor for the combo family with
  aligned `cp.async`.
  Its executable SASS keeps:
  `LDGSTS.E.BYPASS.LTC128B.128`, `LDGDEPBAR`, `DEPBAR.LE`,
  `MATCH.ANY`, `REDUX.MIN/MAX/SUM(.S32)`,
  `VOTE.ALL`, `VOTE.ANY`, `VOTEU.ANY`,
  `POPC`, `UFLO.U32`,
  `RED.E.ADD.F32.FTZ.RN.STRONG.GPU`, and `RED.E.ADD.STRONG.GPU`.
  The `ncu` readout shows:
  - `smsp__cycles_elapsed.avg` about `4342.48`
  - `smsp__inst_executed.sum` fixed at `70`
  - `launch__registers_per_thread = 16`
  - `launch__shared_mem_per_block_static = 1024 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.20%`
  - median `lts__t_sector_hit_rate.pct` about `75.7%`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed` about `0.02%`
  A second stall-focused `ncu` pass shows the runtime-safe family is
  primarily long-scoreboard limited, not barrier- or membar-limited:
  - barrier stall `0%`
  - membar stall `0%`
  - short scoreboard stall about `3.76%`
  - long scoreboard stall about `32.51%`
  - wait stall about `4.90%`
- A direct policy comparison under
  `src/sass_re/results/runs/combo_warp_atomic_cache_profile_safe_tranche_dlcm_cg_20260322_001200`
  sharpens that interpretation. The safe surrogate flips its load spellings to
  `LDG.E.U8/U16.STRONG.GPU` while preserving the same
  `LDGSTS/LDGDEPBAR/DEPBAR + MATCH/VOTE/REDUX + RED.E.ADD` family, but the
  runtime profile stays essentially the same:
  - `smsp__cycles_elapsed.avg` about `4348.29`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.09%`
  - median `lts__t_sector_hit_rate.pct` about `72.46%`
  - short scoreboard stall about `3.76%`
  - long scoreboard stall about `32.45%`
  - wait stall about `4.90%`
  - barrier stall `0%`
  - membar stall `0%`
  So on this runtime-safe branch, forcing `STRONG.GPU` changes the emitted load
  spellings but does not materially change the measured dependency profile.
- A 4-way safe-anchor matrix under
  `src/sass_re/results/runs/combo_family_safe_anchor_batch_20260322_001544`
  then confirms how the smaller family scales with dependency depth:
  - shallow safe default:
    `70` instructions, `16` registers/thread, `1024 B` static shared memory,
    `short_scoreboard ~3.58%`, `long_scoreboard ~33.55%`,
    `wait ~4.66%`, `barrier = 0%`, `membar = 0%`
  - shallow safe `-dlcm=cg`:
    effectively the same runtime shape
  - deeper safe default:
    `94` instructions, `17` registers/thread, `2048 B` static shared memory,
    `short_scoreboard ~5.28%`, `long_scoreboard ~31.21%`,
    `wait ~6.53%`, `barrier = 0%`, `membar = 0%`
  - deeper safe `-dlcm=cg`:
    same qualitative picture, with `long_scoreboard ~32.18%`
  So the smaller safe branch scales mostly through dependency depth and fixed
  execution latency, not through barrier or membar pressure.
- A runtime-safe 64-bit SYS surrogate under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_profile_safe_tranche_20260322_002817`
  closes the next execution gap. Its executable SASS still keeps:
  `LDG.E.64`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `MATCH.ANY`, `REDUX.MIN/MAX/SUM.S32`,
  `VOTE.ALL`, `VOTE.ANY`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, `CCTL.IVALL`,
  and `STG.E.64`.
  Its `ncu` profile is qualitatively stronger than the smaller safe family:
  - `smsp__cycles_elapsed.avg` about `11043.89`
  - `smsp__inst_executed.sum` fixed at `532`
  - `launch__registers_per_thread = 22`
  - `launch__shared_mem_per_block_static = 2048 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.23%`
  - median `lts__t_sector_hit_rate.pct` about `65.4%`
  - long scoreboard about `53.41%`
  - membar about `24.84%`
  - short scoreboard about `0.96%`
  - barrier about `1.01%`
  - wait about `4.61%`
  This means the dense 64-bit SYS branch is now both symbolic-positive and
  runtime-profile-positive, and it crosses into a materially stronger memory-
  barrier plus dependency-latency regime than the smaller safe family.
- A direct policy comparison under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_profile_safe_tranche_dlcm_cg_20260322_003600`
  sharpens that result further. The main load flips to
  `LDG.E.64.STRONG.GPU`, but the runtime shape stays essentially unchanged:
  - `smsp__cycles_elapsed.avg` about `10918.09`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.17%`
  - median `lts__t_sector_hit_rate.pct` about `66.58%`
  - long scoreboard about `53.30%`
  - membar about `25.35%`
  - short scoreboard about `0.97%`
  - barrier about `0.93%`
  - wait about `4.63%`
  So even the heavier 64-bit SYS-safe branch remains structure-limited first
  and cache-policy-limited second.
- A deeper 64-bit SYS-safe follow-up under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche_20260322_005633`
  preserves the same fused family while increasing dependency depth and adding
  a second round of `ATOMG.E.*.64.STRONG.SYS`. The executable SASS still keeps:
  `LDG.E.64`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `MATCH.ANY`, `REDUX.MIN/MAX/SUM.S32`,
  `VOTE.ALL`, `VOTE.ANY`,
  two rounds of `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, `CCTL.IVALL`,
  and `STG.E.64`.
  Its runtime profile shifts in a useful way:
  - `smsp__cycles_elapsed.avg` about `10036.76`
  - `smsp__inst_executed.sum = 740`
  - `launch__registers_per_thread = 38`
  - `launch__shared_mem_per_block_static = 4096 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `1.88%`
  - long scoreboard about `45.94%`
  - membar about `26.67%`
  - short scoreboard about `1.74%`
  - barrier about `1.23%`
  - wait about `5.97%`
  Relative to the shallower 64-bit SYS-safe branch, deeper structure raises
  membar, short-scoreboard, and wait pressure while long-scoreboard remains
  the largest single stall class.
- A runtime-safe store-side SYS surrogate under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_20260322_005938`
  closes the old symbolic/runtime split for the direct store-side family.
  Its executable SASS now keeps:
  `LDG.E.64`,
  `LDG.E.64.STRONG.SYS`,
  `STG.E.64.STRONG.SYS`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.POPC/AND/OR.DEFER_BLOCKING`, `B2R.RESULT`,
  `MATCH.ANY`, `REDUX.MIN/MAX/SUM.S32`,
  `VOTE.ALL`, `VOTE.ANY`,
  two direct `LDG.E.64.STRONG.SYS` reads, two direct
  `STG.E.64.STRONG.SYS` writes,
  and the dense `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS` body.
  Its first trustworthy `ncu` profile is:
  - `smsp__cycles_elapsed.avg` about `10705.61`
  - `smsp__inst_executed.sum = 828`
  - `launch__registers_per_thread = 40`
  - `launch__shared_mem_per_block_static = 4096 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.40%`
  - median `lts__t_sector_hit_rate.pct` about `69.28%`
  - long scoreboard about `46.17%`
  - membar about `25.74%`
  - short scoreboard about `1.70%`
  - barrier about `1.19%`
  - wait about `6.23%`
- A direct `-dlcm=cg` comparison on that same store-side surrogate under
  `src/sass_re/results/runs/combo_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_dlcm_cg_20260322_010500`
  follows the same rule as the other runtime-safe branches:
  - the leading load flips to `LDG.E.64.STRONG.GPU`
  - the direct `LDG.E.64.STRONG.SYS` / `STG.E.64.STRONG.SYS` pair remains
  - `lts__t_sector_hit_rate.pct` improves to about `75.20%`
  - `smsp__cycles_elapsed.avg` stays essentially unchanged at about `10649.02`
  - long scoreboard stays about `45.52%`
  - membar stays about `25.38%`
  - short scoreboard stays about `1.69%`
  - wait stays about `6.22%`
  So the store-side SYS branch is now both emitted-SASS-positive and
  runtime-profile-positive, and it also remains structure-limited first and
  cache-policy-limited second.
- A normalized batch matrix under
  `src/sass_re/results/runs/combo_family_ncu_batch_20260322_022222`
  now puts the main runtime-safe combo branches onto one comparison surface:
  - small safe shallow:
    `70` instructions, `16` regs, `1024 B` shared,
    `long_scoreboard ~32.95%`, `short_scoreboard ~3.61%`,
    `wait ~4.70%`, `barrier = 0%`, `membar = 0%`
  - small safe deep:
    `94` instructions, `17` regs, `2048 B` shared,
    `long_scoreboard ~32.76%`, `short_scoreboard ~5.33%`,
    `wait ~6.61%`, `barrier = 0%`, `membar = 0%`
  - 64-bit SYS-safe shallow:
    `532` instructions, `22` regs, `2048 B` shared,
    `long_scoreboard ~53.74%`, `membar ~25.27%`
  - 64-bit SYS-safe deep:
    `740` instructions, `38` regs, `4096 B` shared,
    `long_scoreboard ~45.33%`, `membar ~26.92%`
  - store-side SYS-safe default:
    `828` instructions, `40` regs, `4096 B` shared,
    `long_scoreboard ~45.60%`, `membar ~24.84%`
  - store-side SYS-safe `-dlcm=cg`:
    L2 hit improves from about `66.29%` to about `69.07%`, but cycles and the
    stall mix stay nearly unchanged
  This makes the runtime split explicit:
  - the small safe family is a dependency-depth / scoreboard study case
  - the 64-bit SYS-safe and store-side SYS-safe families are the heavy
    long-scoreboard + membar regime
- A runtime-safe uniform-helper/system-`RED` follow-up under
  `src/sass_re/results/runs/combo_uniform_redsys_async_profile_safe_tranche_20260322_133653`
  closes the next missing runtime branch. Its executable SASS preserves:
  `ULDC.64`, `UIADD3`, `ULOP3.LUT`, `USHF.L.U32`, `USHF.L.U64.HI`,
  `LDG.E.U8`, `LDG.E.U16`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  and `RED.E.MIN/MAX/ADD(.F32).STRONG.SYS`.
  Its first `ncu` profile is:
  - `smsp__cycles_elapsed.avg` about `4801.74`
  - `launch__registers_per_thread = 24`
  - `launch__shared_mem_per_block_static = 1024 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `2.28%`
  - median `lts__t_sector_hit_rate.pct` about `74.73%`
  - long scoreboard about `29.33%`
  - short scoreboard about `2.89%`
  - wait about `7.09%`
  - barrier about `0%`
  - membar about `0%`
  So this uniform-helper/system-`RED` branch joins the runtime matrix as a
  lighter dependency-latency branch rather than a memory-barrier branch.
- A narrow 64-bit SYS atomic matrix runtime-safe surrogate under
  `src/sass_re/results/runs/combo_atomg64sys_ops_profile_safe_tranche_20260322_135833`
  closes the next symbolic/runtime split for the SYS-side family. Its
  executable SASS preserves:
  `LDG.E.64`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  Its first `ncu` profile is:
  - `smsp__cycles_elapsed.avg` about `8442.60`
  - `launch__registers_per_thread = 32`
  - `launch__shared_mem_per_block_static = 1024 B`
  - median `lts__t_sector_hit_rate.pct` about `72.99%`
  - long scoreboard about `23.84%`
  - membar about `37.84%`
  - short scoreboard about `0.05%`
  - wait about `5.78%`
  So the narrow SYS matrix is the cleanest membar-dominated executable branch
  in the current runtime corpus.
- A divergent fused runtime-safe surrogate under
  `src/sass_re/results/runs/combo_divergent_blockred_warp_atomg64sys_ops_profile_safe_tranche_20260322_140111`
  now preserves optimized `BSSY` and `BSYNC` in executable code alongside
  the same `BAR.RED.*`, `B2R.RESULT`, `MATCH.ANY`, `REDUX.*`, `VOTE.*`, and
  dense `ATOMG.E.*.64.STRONG.SYS` body. Its first `ncu` profile is:
  - `smsp__cycles_elapsed.avg` about `9096.39`
  - `launch__registers_per_thread = 30`
  - `launch__shared_mem_per_block_static = 4096 B`
  - median `lts__t_sector_hit_rate.pct` about `68.86%`
  - barrier about `1.31%`
  - long scoreboard about `23.99%`
  - membar about `33.74%`
  - wait about `6.39%`
  So optimized reconvergence is now confirmed in a safe executable branch,
  and it stays in the same broad SYS-side latency class.
- A Python auto-explorer report under
  `src/sass_re/results/runs/auto_explorer_20260322_141500`
  now ingests the combo-family `.sass` and `ncu` corpus through:
  - `src/sass_re/scripts/auto_explorer.py`
  - `src/sass_re/auto_explorer_search_space.toml`
  - `src/sass_re/scripts/auto_explorer_queue.py`
  - `src/sass_re/AUTO_EXPLORER.md`
  The first explorer run matches the hand-built frontier:
  - top runtime continuations:
    `uniform_blockred_sys64_store`,
    `uniform_blockred_sys64_depth`,
    `uniform_blockred_sys64_store_dlcm_cg`
  - residual symbolic-only raw-SASS boundary:
    `P2R.B1`, `P2R.B2`, `P2R.B3`
  The current Python stack already available on this workstation is enough
  for a serious first explorer:
  `numpy`, `pandas`, `networkx`, `scikit-learn`, `optuna`, `onnx`,
  `onnxruntime`, `torch`, `typer`, and `pydantic`.
- A queue artifact under
  `src/sass_re/results/runs/auto_explorer_queue_20260322_143000`
  now turns the proposal table into a simple execution order while reusing the
  same search-space registry.
- The first executed explorer pick under
  `src/sass_re/results/runs/combo_uniform_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_20260322_143100`
  validates the top-ranked runtime branch and widens it further in practice.
  Its emitted SASS preserves:
  `ULDC(.64)`, `UIADD3`, `ULOP3.LUT`, `USHF.L.U32`, `USHF.L.U64.HI`,
  `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`,
  `LDGDEPBAR`, `DEPBAR.LE`,
  `BAR.RED.*`, `B2R.RESULT`,
  `MATCH.ANY`, `REDUX.*`, `VOTE.ALL`, `VOTE.ANY`,
  `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`,
  `LDG.E.64.STRONG.SYS`, `STG.E.64.STRONG.SYS`,
  `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL`.
  It also directly emits optimized `BSSY` and `BSYNC`, so the realized branch
  is stronger than the original non-divergent proposal.
  Its first `ncu` profile is:
  - `smsp__cycles_elapsed.avg` about `12028.98`
  - `smsp__inst_executed.sum = 1152`
  - `launch__registers_per_thread = 38`
  - `launch__shared_mem_per_block_static = 4096 B`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` about `1.74%`
  - `lts__t_sector_hit_rate.pct` about `75.72%`
  - barrier about `0.88%`
  - short scoreboard about `1.09%`
  - long scoreboard about `47.40%`
  - membar about `24.97%`
  - wait about `5.96%`
  So this fused uniform + block-red + warp + SYS64 + direct SYS store branch
  now joins the heavy executable SYS-side regime as one of the richest
  runtime-profile-positive families in the local corpus.
- Two immediate follow-ups then clarify how to push that new fused family:
  - `src/sass_re/results/runs/combo_uniform_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_dlcm_cg_20260322_145200`
    shows that `-dlcm=cg` is a modest improvement, not a new regime:
    - `smsp__cycles_elapsed.avg` improves from about `12028.98` to `11878.29`
      (`-1.25%`)
    - `long_scoreboard` improves from about `47.40%` to `46.73%`
    - `membar` improves from about `24.97%` to `23.00%`
    - instruction count and register count stay flat
  - `src/sass_re/results/runs/combo_uniform_blockred_warp_atomg64sys_ops_store_profile_depth_safe_tranche_20260322_145200`
    shows that more structural depth pushes the branch into a much stronger
    long-scoreboard regime:
    - `smsp__cycles_elapsed.avg` rises to about `13805.60` (`+14.77%`)
    - `smsp__inst_executed.sum` rises to `1356`
    - `launch__registers_per_thread` rises to `40`
    - `long_scoreboard` rises to about `58.42%`
    - `membar` falls to about `19.81%`
  So the fused branch is now well-characterized:
  - cache policy is a small refinement lever
  - added depth is mainly a long-scoreboard stress lever
- The refreshed explorer and queue under
  `src/sass_re/results/runs/auto_explorer_20260322_150000` and
  `src/sass_re/results/runs/auto_explorer_queue_20260322_150100`
  now re-rank the next runtime frontier around those measured outcomes:
  - `uniform_blockred_sys64_depth`
  - `uniform_blockred_sys64_store`
  - `uniform_blockred_sys64_store_dlcm_cg`
  - `narrow_atomg_sys64_depth`
  - `narrow_atomg_sys64_dlcm_cg`
  while the residual symbolic-only raw-SASS boundary still remains
  `P2R.B1/B2/B3`.
- `src/sass_re/scripts/auto_explorer.py` now supports both the older wide
  Nsight Compute CSV layout and the newer long `Metric Name` / `Metric Value`
  form. That parser fix was required so the newest bridge tranches would be
  scored instead of silently undercounted.
- The refreshed explorer/queue passes under
  `src/sass_re/results/runs/auto_explorer_20260322_171047`,
  `src/sass_re/results/runs/auto_explorer_queue_20260322_171047`,
  `src/sass_re/results/runs/auto_explorer_20260322_171324`, and
  `src/sass_re/results/runs/auto_explorer_queue_20260322_171324`
  now fold in two bridge branches.
- A new runtime-safe uniform + divergent + SYS64 bridge under
  `src/sass_re/results/runs/combo_uniform_divergent_atomg64sys_profile_safe_tranche_20260322_170855`
  now keeps `ULDC.64`, `UIADD3`, `ULOP3.LUT`, `USHF`, the
  `LDGSTS/LDGDEPBAR/DEPBAR` backbone, and a warp/divergence shape in one
  executable body. Its runtime profile is:
  `cycles ~9553.06`, `inst 1012`, `regs 30`, `shared 4096 B`,
  `long_scoreboard ~20.56%`, `membar ~31.67%`,
  `short_scoreboard ~10.76%`, `wait ~6.78%`.
- A new runtime-safe divergent + SYS64 store bridge under
  `src/sass_re/results/runs/combo_divergent_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_20260322_171218`
  now keeps `BSSY`, `BSYNC`, `BAR.RED.*`, `B2R.RESULT`, `MATCH/REDUX/VOTE`,
  direct `LDG.E.64.STRONG.SYS` / `STG.E.64.STRONG.SYS`, the dense
  `ATOMG.E.*.64.STRONG.SYS` matrix, and the
  `MEMBAR.SC.SYS` / `ERRBAR` / `CCTL.IVALL` tail in one executable body.
  Its runtime profile is:
  `cycles ~11159.66`, `inst 880`, `regs 38`, `shared 4096 B`,
  `long_scoreboard ~44.26%`, `membar ~25.15%`,
  `short_scoreboard ~1.99%`, `barrier ~1.32%`, `wait ~5.94%`.
  Interpretation:
  direct SYS load/store, not divergence alone, is what pulls the divergent
  branch back toward the heavier fused long-scoreboard + membar regime.
- The direct `-dlcm=cg` comparison for that same bridge under
  `src/sass_re/results/runs/combo_divergent_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_20260322_171344`
  flips the leading load to `LDG.E.64.STRONG.GPU`, but it is mildly harmful on
  this branch:
  `cycles ~11381.39` (`+1.99%`),
  `lts__t_sector_hit_rate.pct ~61.63%` (`-9.39` points),
  `long_scoreboard ~44.00%` (`-0.26` points),
  `membar ~26.10%` (`+0.96` points).
  Interpretation:
  the divergent SYS-store bridge is even less policy-sensitive than the
  earlier fused branch, and forcing the stronger GPU-side cache spelling here
  mainly hurts L2 behavior without improving the dominant stall regime.
- A new runtime-safe uniform + block-red + SYS64 depth bridge under
  `src/sass_re/results/runs/combo_uniform_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche_20260322_172135`
  now closes the non-store depth gap on the uniform side.
  The executable SASS keeps `ULDC(.64)`, `UIADD3`, `ULOP3.LUT`,
  `USHF.L.U32`, `USHF.L.U64.HI`, the
  `LDGSTS/LDGDEPBAR/DEPBAR` backbone, `BAR.RED.*`, `B2R.RESULT`,
  `MATCH/REDUX/VOTE`, and the dense `ATOMG.E.*.64.STRONG.SYS` matrix.
  Its runtime profile is:
  `cycles ~12529.48`, `inst 1236`, `regs 36`, `shared 4096 B`,
  `long_scoreboard ~54.97%`, `membar ~22.40%`,
  `short_scoreboard ~0.99%`, `barrier ~0.83%`, `wait ~5.42%`.
  Interpretation:
  the deeper uniform+blockred SYS64 branch is now strongly
  long-scoreboard-driven, even without direct SYS store.
- A new runtime-safe uniform + divergent + block-red + SYS64 + direct SYS
  store bridge under
  `src/sass_re/results/runs/combo_uniform_divergent_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_20260322_172130`
  closes the next aggressive fused gap.
  The executable SASS keeps `ULDC(.64)`, `UIADD3`, `ULOP3.LUT`, `USHF`,
  `BSSY`, `BSYNC`, `BAR.RED.*`, `B2R.RESULT`, `MATCH/REDUX/VOTE`,
  direct `LDG.E.64.STRONG.SYS` / `STG.E.64.STRONG.SYS`, and the dense
  `ATOMG.E.*.64.STRONG.SYS` matrix in one body.
  Its runtime profile is:
  `cycles ~12042.67`, `inst 1284`, `regs 38`, `shared 4096 B`,
  `long_scoreboard ~42.45%`, `membar ~25.24%`,
  `short_scoreboard ~8.03%`, `barrier ~1.16%`, `wait ~5.98%`.
  Interpretation:
  compared with the non-uniform divergent SYS-store bridge, the uniform
  front-end trims long-scoreboard pressure somewhat but introduces a much
  larger short-scoreboard term.
- The direct `-dlcm=cg` comparison for that fully fused branch under
  `src/sass_re/results/runs/combo_uniform_divergent_blockred_warp_atomg64sys_ops_store_profile_safe_tranche_dlcm_cg_20260322_172203`
  is also mildly negative overall:
  `cycles ~12170.44` (`+1.06%`),
  `lts__t_sector_hit_rate.pct ~67.50%` (`-2.34` points),
  `long_scoreboard ~42.26%` (`-0.20` points),
  `membar ~22.69%` (`-2.55` points).
  Interpretation:
  on this heavier uniform+divergent SYS-store body, `-dlcm=cg` lowers
  `membar` but still fails to deliver a net runtime win.
- The lighter uniform+divergent SYS64 midpoint branch also now has a direct
  `-dlcm=cg` comparison under
  `src/sass_re/results/runs/combo_uniform_divergent_atomg64sys_profile_safe_tranche_dlcm_cg_20260322_173400`.
  There the policy shift improves
  `lts__t_sector_hit_rate.pct ~73.69%` (up from about `64.28%`), but cycles
  still worsen to `~9701.17` (up from about `9553.06`).
  Interpretation:
  even on the lighter midpoint branch, `-dlcm=cg` changes cache behavior much
  more than it improves the dominant runtime regime.
- The non-store uniform+blockred SYS64 depth branch also now has a direct
  `-dlcm=cg` comparison under
  `src/sass_re/results/runs/combo_uniform_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche_dlcm_cg_20260322_174100`.
  There, cycles improve only marginally to `~12483.15` (from `~12529.48`),
  while `lts__t_sector_hit_rate.pct` falls to about `71.25%` and
  `membar` rises to about `24.28%`.
  Interpretation:
  even where `-dlcm=cg` is slightly favorable on cycles, it is still not a
  clean first-order lever on this fused family.
- The latest refreshed explorer/queue under
  `src/sass_re/results/runs/auto_explorer_20260322_174400` and
  `src/sass_re/results/runs/auto_explorer_queue_20260322_174400`
  now fold in both of those new uniform+SYS64 runtime branches. The
  conceptual frontier is still stable:
  richer uniform+SYS64 runtime-safe variants remain highest-yield, while the
  residual symbolic-only direct-local raw-SASS boundary stays pinned to
  `P2R.B1/B2/B3`.
- A broader explorer refresh under
  `src/sass_re/results/runs/auto_explorer_20260322_190600` and
  `src/sass_re/results/runs/auto_explorer_queue_20260322_190600`
  then clears the remaining runtime queue entirely once the realized runtime
  corpus is fully represented. That leaves only the symbolic
  `P2R.B1/B2/B3` boundary in the current search space.
- A fresh symbolic rerun under
  `src/sass_re/results/runs/p2r_symbolic_refresh_20260322_190301`
  then revalidates that remaining boundary on the current local toolchain.
  Both refreshed lanes, `-O3 -Xptxas -O3` and
  `-O3 -Xptxas -O3 --maxrregcount=32`, still produce only:
  `P2R R0, PR, R0, 0x7f`
  while the literal `B1/B2/B3` and split-seed carriers still lower through
  `ISETP` + `SEL` + `LOP3.LUT` glue, with occasional `PRMT`.
  Interpretation:
  the residual `P2R.B1/B2/B3` boundary is now not only narrow, but rigid
  across rerun and register-pressure control.
- A wider symbolic matrix under
  `src/sass_re/results/runs/p2r_symbolic_matrix_20260322_194108`
  then extends that ranking again with three tighter `B1` retries:
  `probe_p2r_b1_secondbank_halfword_exact`,
  `probe_p2r_b1_samecarrier_late4_exact`,
  `probe_p2r_b1_samecarrier_r7style_exact`,
  `probe_p2r_b1_dualpack_transition_exact`,
  `probe_p2r_b1_nibble_exact`,
  `probe_p2r_b1_regmask_transition_exact`,
  `probe_p2r_b2_split_seed_exact`, and
  `probe_p2r_b2_nibble_exact`,
  `probe_p2r_b2_regmask_transition_exact`,
  `probe_p2r_b3_split_seed_exact`,
  `probe_p2r_b3_nibble_exact`,
  `probe_p2r_b3_regmask_transition_exact`,
  `probe_p2r_b2_tripack_prefix_exact`, and
  `probe_p2r_b3_tripack_prefix_exact`.
  The scorer
  `src/sass_re/scripts/score_p2r_symbolic_boundary.py`
  compares the emitted opcode sets against the cuDNN-mined `P2R.B*` windows.
  That ranking is stable across both lanes:
  `probe_p2r_b1_samecarrier_r7style_exact` is now the closest local `B1`
  approximation at `jaccard_vs_ref = 0.258065`,
  `probe_p2r_b1_dualpack_transition_exact` is next at `0.225806`,
  `probe_p2r_b1_split_seed_exact` follows at `0.20`,
  `probe_p2r_b1_samecarrier_late4_exact` follows at `0.193548`,
  and `probe_p2r_b1_secondbank_halfword_exact` is next at `0.166667`,
  `probe_p2r_b2_literal_cudnn_exact`, `probe_p2r_b2_split_seed_exact`, and
  `probe_p2r_b3_split_seed_exact` all cluster at `0.16`,
  and `probe_p2r_b3_literal_cudnn_exact` is still the weakest branch at
  `0.115385` while uniquely retaining `PRMT`.
  Interpretation:
  the symbolic frontier is now ordered more sharply than before, and `B1`
  remains the most promising unresolved byte-qualified form. Even the tighter
  same-carrier, dual-pack, nibble, and regmask retries still do not surface
  `P2R.B1/B2/B3`, and the only surviving direct local `P2R` hit remains the plain
  `P2R R0, PR, R0, 0x7f` path.
- A nibble-sized symbolic follow-up inside that same matrix then adds one more
  useful boundary result. `probe_p2r_b2_nibble_exact` and
  `probe_p2r_b3_nibble_exact` still do not emit `P2R.B2/B3`, but they do
  directly reproduce plain `P2R ... 0xf` plus `PRMT` on local SM89. That
  means the remaining byte-qualified gap is no longer about small-mask
  predicate packing in general; it is specifically about byte-qualified form
  selection on top of a pack that the compiler can already express as plain
  `P2R`.
- A final regmask-transition retry inside that same matrix then closes the
  last obvious carrier-update question. `probe_p2r_b1_regmask_transition_exact`,
  `probe_p2r_b2_regmask_transition_exact`, and
  `probe_p2r_b3_regmask_transition_exact` replace direct byte-field writes
  with whole-register masked rewrites on the same live carrier, but they
  lower exactly the same way: plain `P2R ... 0xf` plus `LOP3`, with `B3`
  still picking up `PRMT`. That is strong negative evidence that the last
  missing detail is not a missed byte-store vs masked-rewrite style; it is
  byte-qualified `P2R.B*` form selection itself.
- A final staged same-carrier tripack then closes the last obvious
  higher-byte-liveness question. `probe_p2r_b2_tripack_prefix_exact` and
  `probe_p2r_b3_tripack_prefix_exact` keep successive `0xf` packs alive in
  the same carrier in the order suggested by the cuDNN `R231` pocket. The
  `B2` tripack improves local neighborhood closeness to `jaccard_vs_ref =
  0.192308`, but both tripack variants still lower to plain `P2R ... 0xf`,
  with `B3` still picking up `PRMT`. That is strong evidence that even
  prior packed higher-byte state is not sufficient to trigger byte-qualified
  `P2R.B*` selection on this local source-space path.
- A deeper fully fused uniform + divergent + block-red + SYS64 + direct SYS
  store branch under
  `src/sass_re/results/runs/combo_uniform_divergent_blockred_warp_atomg64sys_ops_store_profile_depth_safe_tranche_20260322_184634`
  now closes the next runtime-safe depth point. Its executable profile is:
  `cycles ~14812.33`, `inst 1636`, `regs 38`, `shared 4096 B`,
  `long_scoreboard ~53.64%`, `membar ~17.37%`,
  `short_scoreboard ~6.44%`, `barrier ~0.83%`, `wait ~5.75%`.
  Interpretation:
  deeper direct SYS-store fusion pushes the family back toward strong
  long-scoreboard dominance, even though the membar share falls relative to
  the shallower store branch.
- The matching deeper non-store branch under
  `src/sass_re/results/runs/combo_uniform_divergent_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche_20260322_185002`
  then closes the missing comparison point. It preserves the same uniform /
  divergent / block-red / warp / SYS64 family without direct SYS load/store.
  Its runtime profile is:
  `cycles ~13264.42`, `inst 1540`, `regs 38`, `shared 4096 B`,
  `long_scoreboard ~49.96%`, `membar ~19.66%`,
  `short_scoreboard ~6.40%`, `barrier ~1.01%`, `wait ~5.86%`.
  Interpretation:
  removing direct SYS store lowers cycles and trims long-scoreboard pressure,
  but the branch still remains structure-limited and does not revert to a
  pure membar-free regime.
- The direct `-dlcm=cg` comparison on that new non-store depth branch under
  `src/sass_re/results/runs/combo_uniform_divergent_blockred_warp_atomg64sys_ops_profile_depth_safe_tranche_20260322_185120`
  sharpens the policy story again:
  `cycles ~13224.93` (`-0.30%`),
  `lts__t_sector_hit_rate.pct ~64.77%` (`-4.05` points),
  `long_scoreboard ~50.29%` (`+0.33` points),
  `membar ~19.05%` (`-0.61` points).
  Interpretation:
  even on this newly closed branch, `-dlcm=cg` remains a spelling/cache
  modifier more than a first-order runtime lever.
- A latest combo-only chain-mining pass under
  `src/sass_re/results/runs/chain_mine_combo_latest_20260321_134300`
  strengthens the same reading. Its dominant anchor windows are all variants
  of the async/cache backbone, especially
  `LDG.E.U16(.STRONG.GPU) + LDGSTS.E.BYPASS.LTC128B.128.ZFILL +
  LDGSTS.E.BYPASS.LTC128B.128 + LDGDEPBAR + LDG.E.U8(.STRONG.GPU) +
  DEPBAR.LE`, with newer local adjacency around `BAR.RED.*`, `B2R.RESULT`,
  `RED.E.*.STRONG.SYS`, and the debug-lane reconvergence helpers.
  That re-ranks the frontier again: widening the cache-policy combo family is
  now higher-yield than more broad predicate-bank probing, while the residual
  direct-local gap remains tightly bounded to `P2R.B1/B2/B3`.
- An OptiX/callable follow-up note under
  `src/sass_re/results/runs/chain_mine_optix_callable_20260321_123500`
  records that the captured callable bundles on this workstation contain PTX
  and runtime logs but not emitted raw `.sass`, so they do not currently add a
  third raw-SASS chain corpus beyond the local direct corpus and the cuDNN-
  mined library corpus.

## 2026-03-20 Full-Corpus 6x4 Flag Sweep

Final full-corpus flag sweep:
[`src/sass_re/results/runs/flag_sweep_postfix_parallel6x4_20260320_233059`](results/runs/flag_sweep_postfix_parallel6x4_20260320_233059)

This run replays the full 343-probe compile-enabled corpus through the
manifest-backed flag matrix with bounded parallelism:

- `FLAG_SWEEP_JOBS=6`
- `FLAG_SWEEP_EXTRACT_JOBS=4`

| Phase / lane | Pass | Fail | Mnemonics | New | Spills | Notes |
|---|---|---|---|---|---|---|
| baseline | 343 | 0 | 379 | 0 | 16 | Canonical optimized reference |
| `-O2` | 343 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O2 -Xptxas -O3` | 343 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O3` | 343 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O3 -Xptxas -O3` | 343 | 0 | 379 | 0 | 16 | Same frontier as baseline |
| `-O0 -G` | 343 | 0 | 351 | 87 | 3728 | Debug/control-flow-heavy lane |
| `-O0 -G -Xptxas -O3` | 0 | 343 | -- | -- | -- | Expected ptxas rejection on this toolchain |
| `-fmad=false` | 343 | 0 | 378 | 0 | 16 | Slightly perturbs code shape, no new frontier |
| `-prec-div=true` | 343 | 0 | 379 | 0 | 16 | No frontier move |
| `-prec-sqrt=true` | 343 | 0 | 379 | 0 | 16 | No frontier move |
| `-ftz=false` | 343 | 0 | 379 | 0 | 16 | No frontier move |
| `--use_fast_math` | 343 | 0 | 374 | 12 | 16 | FTZ-family plus `MUFU.SQRT` |
| `--maxrregcount=32` | 343 | 0 | 382 | 3 | 16 | `UISETP.EQ.U32.XOR`, `UISETP.GE.U32.AND`, `UISETP.GT.AND` |
| `--maxrregcount=64/128/255` | 343 | 0 | 380 | 2 | 16 | `UISETP.GE.U32.AND`, `UISETP.GT.AND` |
| `--restrict` | 343 | 0 | 381 | 4 | 40 | `I2F.S8`, `LDG.E.U16.CONSTANT`, `LDG.E.U8.CONSTANT`, `LDL.LU` |
| `--default-stream per-thread` | 343 | 0 | 379 | 0 | 16 | Runtime semantic, not a mnemonic mover |
| `-G -dopt=on` | 343 | 0 | 379 | 0 | 16 | No frontier move on full corpus |
| `--extra-device-vectorization` | 343 | 0 | 379 | 0 | 16 | No frontier move |
| `--extra-device-vectorization --restrict` | 343 | 0 | 379 | 0 | 16 | No frontier move |
| `--use_fast_math --extra-device-vectorization --restrict` | 343 | 0 | 374 | 12 | 16 | Same FTZ-family pattern as `--use_fast_math` |
| `-Xptxas -dlcm=cg` | 343 | 0 | 377 | 7 | 16 | Load-family spelling lever |
| `-Xptxas -disable-optimizer-consts` | 343 | 0 | 378 | 0 | 16 | No new raw spellings |
| `-O3 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false` | 343 | 0 | 378 | 0 | 16 | Precision bundle does not move frontier |
| `-O3 --use_fast_math --restrict` | 343 | 0 | 375 | 15 | 16 | FTZ-family plus `I2F.S8`, constant sub-byte loads, `MUFU.SQRT` |
| `-Xptxas -warn-spills` | 343 | 0 | 379 | 0 | 16 | Diagnostic only |
| `-Xptxas -warn-double-usage` | 343 | 0 | 379 | 0 | 16 | Diagnostic only |

Full-corpus takeaways from this sweep:

- `--maxrregcount=32` remains the strongest frontier mover, pushing the local
  maximum to `382` via `UISETP.EQ.U32.XOR`, `UISETP.GE.U32.AND`, and
  `UISETP.GT.AND`.
- `--restrict` remains the highest-yield semantic flag for the full corpus,
  exposing `I2F.S8`, `LDG.E.U16.CONSTANT`, `LDG.E.U8.CONSTANT`, and `LDL.LU`.
- `-Xptxas -dlcm=cg` is now confirmed on the full corpus as a real load-family
  spelling lever, surfacing `LD.E.64.STRONG.GPU`,
  `LDG.E.128.STRONG.GPU`, `LDG.E.64.STRONG.GPU`,
  `LDG.E.S16.STRONG.GPU`, `LDG.E.S8.STRONG.GPU`,
  `LDG.E.U16.STRONG.GPU`, and `LDG.E.U8.STRONG.GPU`.
- `-G -dopt=on`, `--extra-device-vectorization`,
  `--extra-device-vectorization --restrict`, and
  `-Xptxas -disable-optimizer-consts` do not move the local frontier on the
  full corpus.
- `--use_fast_math` and
  `--use_fast_math --extra-device-vectorization --restrict` add the expected
  FTZ-family variants plus `MUFU.SQRT`, but do not expose new structural
  instruction families beyond that cluster.

Focused validation artifacts for this implementation pass are under
[`src/sass_re/results/plan_impl_20260319_230717`](results/plan_impl_20260319_230717).

The dedicated accelerator tranche artifacts are under
[`src/sass_re/results/runs/tranche_accel_20260319_235626`](results/runs/tranche_accel_20260319_235626).

The scripted revalidation pass is under
[`src/sass_re/results/runs/tranche_accel_20260320_065036`](results/runs/tranche_accel_20260320_065036).

The forced inline-PTX packed-video tranche is under
[`src/sass_re/results/runs/video_isa_inline_ptx_20260320_084601`](results/runs/video_isa_inline_ptx_20260320_084601).

The debug-vs-optimized packed-video comparison run is under
[`src/sass_re/results/runs/video_flag_compare_20260320_090000`](results/runs/video_flag_compare_20260320_090000).

The scalar-video and variant inline-PTX tranche is under
[`src/sass_re/results/runs/video_scalar_variant_ptx_20260320_092500`](results/runs/video_scalar_variant_ptx_20260320_092500).

The scalar-video debug-vs-optimized comparison run is under
[`src/sass_re/results/runs/video_extended_flag_compare_20260320_091500`](results/runs/video_extended_flag_compare_20260320_091500).

The dedicated `R2UR` / `MEMBAR.SC.VC` / selector-video follow-up run is under
[`src/sass_re/results/runs/debug_followups_20260320_090904`](results/runs/debug_followups_20260320_090904).

The completed deep-OptiX + cuDNN tranche bundle is under
[`src/sass_re/results/runs/tranche_ml_optix_20260320_104244`](results/runs/tranche_ml_optix_20260320_104244).

The architecture-filtered cuDNN library mining bundle is under
[`src/sass_re/results/runs/cudnn_library_sm86_mining_20260320_103900`](results/runs/cudnn_library_sm86_mining_20260320_103900).

The latest strict `UPLOP3` / `P2R` follow-up bundle is under
[`src/sass_re/results/runs/uplop3_p2r_followup_20260320`](results/runs/uplop3_p2r_followup_20260320).

Confirmed additions from that bundle:

- `IDP.4A` signedness closure:
  `IDP.4A.U8.U8`, `IDP.4A.S8.S8`, `IDP.4A.S8.U8`, `IDP.4A.U8.S8`
- `cp.async` lowering closure:
  `LDGSTS.E.ZFILL`, `LDGSTS.E.64.ZFILL`,
  `LDGSTS.E.BYPASS.128.ZFILL`, and predicated
  `LDGSTS.E.BYPASS.128` for ignore-src
- Existing nested recursive probes promoted into the public inventory:
  `BAR.RED.AND.DEFER_BLOCKING`, `BAR.RED.OR.DEFER_BLOCKING`,
  `BAR.RED.POPC.DEFER_BLOCKING`, `B2R.RESULT`,
  `ATOMS.INC`, `ATOMS.DEC`,
  `RED.E.INC.STRONG.GPU`, `RED.E.DEC.STRONG.GPU`, `REDUX.XOR`
- Inline-PTX packed-video closure:
  forced PTX `vadd2`, `vsub2`, `vavrg2`, `vmin2`, `vmax2`, `vset2`,
  `vadd4`, `vsub4`, `vavrg4`, `vmin4`, `vmax4`, `vset4`, and
  `vabsdiff4...add`, plus signed `vmin/vmax/vset` controls, confirm that only
  the explicit accumulate form survives as raw video SASS
  (`VABSDIFF4.U8.ACC`). The other packed-video PTX operations lower to
  `IADD3`, `LOP3.LUT`, `PRMT`, `IMNMX`, `ISETP`, `SHF`, and `IMAD.IADD`.
- Debug-vs-optimized packed-video diff:
  rerunning the full `simd_video` slice under `-O0 -G` and
  `-O3 -Xptxas -O3` does not reveal any additional raw packed-video `V*`
  mnemonics. The debug lane introduces expected instrumentation/control-flow
  spellings such as `BPT.TRAP`, `BSSY`, `BSYNC`, `PLOP3.LUT`, `LDC`, `LDL`,
  and `STL`, but the packed-video boundary stays the same:
  `VABSDIFF4.U8` and `VABSDIFF4.U8.ACC` only.
- Scalar-video closure:
  forced PTX `vadd`, `vsub`, `vabsdiff`, `vmin`, `vmax`, `vshl`, `vshr`,
  `vmad`, and `vset` all compile and lower, but none surface new raw `V*`
  SASS spellings on Ada. They synthesize into ordinary integer SASS families
  built from `PRMT`, `IMAD.IADD`, `IADD3`, `IADD3.X`, `IMAD.X`, `IMNMX`,
  `ISETP.*`, `SHF.*`, `IABS`, and `LOP3.LUT`.
- Variant closure:
  forcing `.sat`, `.add`, merge forms, `vabsdiff2`, `vadd2.sat`,
  `vadd4.sat`, `vset2.add`, and `vset4.add` still does not expose additional
  raw packed-video `V*` SASS. The only raw packed-video `V*` spellings remain
  `VABSDIFF4.U8` and `VABSDIFF4.U8.ACC`.
- Debug-vs-optimized scalar-video diff:
  rerunning the new scalar/variant tranches under `-O0 -G` and
  `-O3 -Xptxas -O3` again does not reveal any new raw `V*` spellings.
  The `-G` lane contributes `ERRBAR`, `MEMBAR.SC.VC`, `R2UR`, `PLOP3.LUT`,
  extra `LDC*`/`LD.E`/`ST.E`, and related debug support code, while the
  optimized lane contributes the leaner global-memory forms and `IMAD.*`
  helpers.
- Selector-heavy video closure:
  forcing selector and merge variants such as `.b0`, `.h1`, `.h10`,
  `.b3210`, `vset*.add`, `vabsdiff2`, and selector-biased `vmad` forms still
  does not expose any additional raw packed-video `V*` SASS. The dedicated
  selector run confirms the same Ada boundary: `VABSDIFF4.U8` and
  `VABSDIFF4.U8.ACC` only.
- Dedicated `R2UR` follow-up:
  `probe_r2ur_debug_path.cu` now provides a stable reproducer. In the `-O0 -G`
  lane it emits `R2UR`, `MEMBAR.SC.VC`, `ERRBAR`, `PLOP3.LUT`, and `IADD3.X`
  around the debug-heavy load/store path. In the `-O3 -Xptxas -O3` lane the
  `R2UR` and `MEMBAR.SC.VC` path disappears.
- Dedicated `MEMBAR.SC.VC` follow-up:
  `probe_membar_sc_vc_debug.cu` shows that `MEMBAR.SC.VC` is reproducible as a
  debug-lane fence form. The optimized lane keeps the architectural fence
  spellings `MEMBAR.SC.GPU` and `MEMBAR.SC.SYS`, while the `-G` lane adds
  `R2UR`, `MEMBAR.SC.VC`, `ERRBAR`, `PLOP3.LUT`, and `IADD3.X`.

One-off mnemonic classification after the video/debug sweeps:

- `PLOP3.LUT`: not actually a debug-only one-off. It is already reproducible in
  optimized code paths that do predicate-heavy control flow or TMU boundary
  logic, so it should stay in the main inventory rather than a debug appendix.
  The dedicated predicate-logic follow-up bundle at
  [`src/sass_re/results/runs/predicate_logic_followup_20260320_091700`](results/runs/predicate_logic_followup_20260320_091700)
  confirms an important nuance: the minimal direct `-O3 -Xptxas -O3` build of
  `probe_predicate_pressure.cu` does not surface `PLOP3.LUT`, but the same
  probe compiled with the full high-signal compile-profile lane
  (`--use_fast_math --restrict --extra-device-vectorization` with the
  resolver-selected `nvcc` language mode; `-std=c++20` locally on CUDA 13.1)
  does reproduce `PLOP3.LUT` and `P2R` in optimized code.
- `IADD3.X`: also not a one-off. It is the carry-propagating half of multiword
  integer arithmetic and already appears in the int64/int128/bignum probes.
- `R2UR`: dedicated follow-up complete. The new repro bundle shows it is
  stable in the `-G` lane of simple global-memory probes, but absent from the
  optimized lane for the same kernels on this CUDA 13.1 setup.
- `MEMBAR.SC.VC`: dedicated follow-up complete. The new repro bundle shows it
  is a reproducible debug/instrumentation fence spelling, while optimized
  memory-ordering probes keep `MEMBAR.SC.GPU` and `MEMBAR.SC.SYS`.
- `ERRBAR`: currently explained well enough by debug/instrumentation and some
  cooperative or memory-scope paths; a dedicated probe is lower priority.
- `CALL.ABS.NOINC` and `RET.ABS.NODEC`: classify as debug/helper call plumbing
  for now, not primary SASS targets.
- TMU behavior confirmation:
  point vs linear filtering, clamp/border/wrap/mirror address modes,
  1D/2D/3D filtered fetches, and surface boundary handling all pass the
  dedicated CPU-oracle runner
- TMU offload evidence from disassembly:
  `tmu_2d_linear` shows 17 `FFMA` vs 30 in `tmu_2d_manual_bilerp`, and
  `tmu_3d_linear` shows 22 `FFMA` vs 44 in `tmu_3d_manual_trilerp`
- `mbarrier` lowering confirmation:
  `probe_mbarrier_core.cu` emits `ATOMS.ARRIVE.64` and
  `ATOMS.POPC.INC.32` around the barrier object lifecycle. The safe
  init/arrive/wait and arrive-drop kernels run correctly; the `try_wait`
  variant disassembles but triggers a trap-like launch failure on this local
  Ada toolkit path and is therefore kept as a disassembly-only negative control.
- Outboard accelerator tranche:
  `optix_real_pipeline_runner.cu` builds and launches a real triangle GAS/SBT
  OptiX pipeline and returns `0x00000100`. `ofa_pipeline_runner.cu` confirms
  OFA API availability, output grids `1,2,4`, and a measured center flow of
  about `(+3.906, 0.000)` on a synthetic translated patch. The
  `nvenc_nvdec_pipeline_runner.cu` confirms NVDEC H.264 support, NVENC session
  open, preset enumeration, and input-format enumeration, but full NVENC
  encoder initialization still returns `NV_ENC_ERR_UNSUPPORTED_PARAM` or
  `NV_ENC_ERR_INVALID_PARAM` for the current minimal H.264 setup, so the probe
  reports `nvenc_session_initialized=0` and remains profile-safe.
- Deeper OptiX callable confirmation:
  `optix_callable_pipeline_runner.cu` now validates a direct-callable plus
  continuation-callable path around `optixTrace()`. The expected payload
  `0x00000117` is reproduced in the dedicated quick bundle, confirming that the
  callable symbols were preserved correctly in PTX via `--keep-device-functions`.
- cuDNN runtime + library-mined provisional mnemonic expansion:
  `cudnn_conv_mining_runner.cu` launches a profile-safe forward convolution on
  the local cuDNN 9.20 stack (`algo=6`, workspace `52352` bytes, checksum
  `247689.875`). In addition, architecture-filtered library mining over
  `libcudnn_cnn.so.9.20.0`, `libcudnn_engines_runtime_compiled.so.9.20.0`, and
  `libcudnn_engines_precompiled.so.9.20.0` at `sm_86` surfaces provisional
  mnemonic candidates absent from the checked-in census. A follow-up direct
  local `sm_89` confirmation tranche has now promoted `HFMA2.RELU`,
  `HMMA.1688.F32.TF32`, `LDSM.16.MT88.4`, `LDGSTS.E.LTC128B.128`,
  `LDGSTS.E.LTC128B.128.ZFILL`, `LDGSTS.E.BYPASS.LTC128B.128`, and
  `LDGSTS.E.BYPASS.LTC128B.128.ZFILL`, `HMNMX2.NAN`,
  `F2FP.BF16.F32.PACK_AB`, `ULDC.U8`, and `ULDC.S8` into the checked-in Ada
  inventory. The same tranche also adds the ReLU-clamped direct local spelling
  `F2FP.RELU.BF16.F32.PACK_AB`. Follow-up direct probes
  `probe_p2r_mov_pack_inline_ptx.cu` and
  `probe_tensor_uniform_predicate.cu` tighten the remaining gap rather than
  closing it: `P2R.B1`, `P2R.B2`, `P2R.B3`, and `UPLOP3.LUT` still come only
  from packaged cuDNN cubins nearest to Ada (`sm_86`), not from the direct
  local `sm_89` probe path. A third uniform-path follow-up,
  `probe_uniform_exotic.cu`, directly reproduces `ULDC.U8` and `ULDC.S8` but
  still bottoms out at regular `LEA.HI.X.SX32` and `SHF.L.U64.HI` rather than
  the fully uniform `ULEA.HI.X.SX32` and `USHF.L.U64.HI` spellings. Two
  tighter follow-ups, `probe_uniform_strict_address.cu` and
  `probe_uniform_sx32_mix.cu`, keep the relevant shift/load or mixed signed-
  offset patterns closer to the uniform/coalescing source shapes, but still do
  not surface `ULEA.HI.X.SX32` or `USHF.L.U64.HI` on the local toolchain. A
  dedicated alias-pack follow-up, `probe_f2fp_alias_pack_inline_ptx.cu`, then
  tested the nearest cuDNN-like scalar-pack source shape directly: zero high
  lane plus immediate `STG.E.U16`, with both `cuobjdump` and `nvdisasm`, under
  both `-O3 -Xptxas -O3` and `-O0 -G`. That still rendered only the typed local
  forms `F2FP.F16.F32.PACK_AB` and `F2FP.BF16.F32.PACK_AB`, sharpening the
  interpretation that shorter `F2FP.PACK_AB` and `F2FP.BF16.PACK_AB` are more
  likely library/disassembler aliases adjacent to the confirmed local typed
  forms than simple missing direct `sm_89` source shapes. A stricter follow-up
  pair in `p2r_uniform_strict_20260320` then split the remaining uniform gap:
  `probe_uniform_u64_strict.cu` now directly confirms `ULEA.HI.X.SX32` on local
  `sm_89`, while `probe_p2r_bytepack_strict_inline_ptx.cu` still does not
  surface `P2R.B1/B2/B3`, and the tighter uniform shift shapes still only reach
  `USHF.L.U32` plus GPR-space `SHF.L.U64.HI`, not `USHF.L.U64.HI`. A newer
  follow-up pair, `probe_p2r_vector_pack_inline_ptx.cu` and
  `probe_uniform_async_tensor_pipeline.cu`, then tried two closer source
  approximations to the cuDNN library neighborhoods. The vector-pack probe uses
  predicate-derived byte lanes plus `mov.b32 {b0,b1,b2,b3}` before merging into
  a live destination register; it still lowers to `ISETP`/`SEL`/`PRMT`/`LOP3`
  glue rather than `P2R.B1/B2/B3`. The async+tensor probe does reproduce the
  local neighborhood around the remaining `UPLOP3` gap by combining
  `LDGSTS.E.BYPASS.LTC128B.128`, `HMMA.1688.F32.TF32`, `@!UPT UIADD3`, and
  dense `PLOP3.LUT` with `0x40`/`0x80` immediates, but it still does not emit
  raw `UPLOP3.LUT` on local `sm_89`.
  A stricter banked-reload and stage-toggle follow-up under
  `p2r_uplop3_stage_followup_20260320` then pushed both remaining predicate
  motifs harder. `probe_p2r_banked_reload.cu` mirrors the cuDNN-like
  long-lived packed-mask lifecycle more closely, but still lowers to
  `ISETP`/`SEL`/`LOP3`/`PRMT`/`IMAD` glue rather than surfacing `P2R.B1/B2/B3`.
  `probe_uniform_stage_toggle_pipeline.cu` rebuilds a tighter
  software-pipelined tensor mainloop and directly reproduces
  `HMMA.1688.F32.TF32`, `LDGSTS.E.BYPASS.LTC128B.128`,
  `LDGSTS.E.BYPASS.LTC128B.128.ZFILL`, dense `PLOP3.LUT`, and repeated
  `@!UPT UIADD3` / `R2UR` scaffolding, but still does not emit raw
  `UPLOP3.LUT` on the local direct `sm_89` path.
  A final uniform-u64 follow-up under `ushf_u64_hi_final_20260320` then tests
  cleaner constant-backed and parameter-backed 64-bit shift families. That
  tranche directly confirms `ULEA` and `ULEA.HI.X` in the uniform address
  path, but still does not reproduce `USHF.L.U64.HI`; the compiler continues
  to select GPR-space `SHF.L.U64.HI` or `SHF.R.U64` for the actual 64-bit-high
  shift.
  A direct prefix cleanup tranche under `prefix_direct_followup_20260320` then
  promotes `LDC.U8` and `LDC.S8` into the direct local inventory, while a
  separate `UISETP` sweep still lowers to ordinary `ISETP` plus `SEL` instead
  of surfacing the wider library-mined `UISETP.*` family. Finally, a
  CUTLASS-like software-pipelined tensor probe under
  `cutlass_predicate_pipeline_20260320` produces the strongest direct local
  optimized `P2R` neighborhood yet by combining base `P2R`, `PLOP3.LUT`,
  `HMMA.1688.F32.TF32`, `LDGSTS.E.BYPASS.LTC128B.128*`, and `@!UPT UIADD3`,
  but it still does not emit `P2R.B1/B2/B3` or `UPLOP3.LUT`.

## Instruction Latency (dependent chains, 512 deep)

| Instruction | Latency (cycles) | Notes |
|---|---|---|
| FADD | 4.53 | FP32 add |
| FMUL | 4.53 | FP32 multiply |
| FFMA | 4.54 | FP32 fused multiply-add |
| IADD3 | 2.51 | 3-input integer add (**fastest ALU op measured**) |
| IMAD | 4.52 | Integer multiply-add |
| MUFU.RCP | 41.55 | Reciprocal (SFU, value-dependent convergence) |
| MUFU.RSQ | 39.55 | Reciprocal square root (SFU) |
| MUFU.SIN | 23.51 | Sine approximation (SFU) |
| MUFU.EX2 | 17.56 | Base-2 exponential (SFU) |
| MUFU.LG2 | 39.55 | Base-2 logarithm (SFU) |
| LOP3 | 4.52 | 3-input logic operation |
| SHF | 4.55 | Funnel shift |
| PRMT | 4.51 | Byte permute |
| F2I+I2F | 12.05 | Float-to-int + int-to-float round-trip |
| SHFL.BFLY | 24.96 | Warp shuffle (butterfly) |
| LDG chase | 92.29 | Global memory pointer chase (L1/L2 hit) |
| LDS chase | 28.03 | Shared memory pointer chase |

### Tensor Core Latency -- ALL Precisions (first-ever on Ada Lovelace)

| Format | Input -> Accum | Latency (cy) | SASS Instruction | Ops/WMMA | Eff FMA/cy |
|---|---|---|---|---|---|
| INT4 S4 | INT4 -> INT32 | **28.05** | IMMA.8832.S4.S4 | 256 | **9.13** |
| UINT4 U4 | UINT4 -> INT32 | **28.05** | IMMA.8832.U4.U4 | 256 | **9.13** |
| INT8 S8 | INT8 -> INT32 | **34.06** | IMMA.16816.S8.S8 | 256 | **7.52** |
| FP16 | FP16 -> FP16 | **42.14** | HMMA.16816.F16 | 256 | **6.07** |
| FP16 | FP16 -> FP32 | 66.28 | HMMA.16816.F32 | 256 | 3.86 |
| BF16 | BF16 -> FP32 | 66.33 | HMMA.16816.F32.BF16 | 256 | 3.86 |
| TF32 | TF32 -> FP32 | 66.66 | 2x HMMA.1684.F32.TF32 | 256 | 3.84 |

Key TC findings:
- **INT4 is fastest per-instruction** at 28 cy (9.13 effective FMA/cy)
- **FP16 -> FP16 accum is 36% faster** than FP16 -> FP32 (42 vs 66 cy)
- **BF16 TC = FP16 TC** at FP32 accumulator (66.33 vs 66.28 cy, identical)
- **TF32's 2x HMMA decomposition does NOT double latency** (pipelines overlap)
- **TC + FP32 ALU run in parallel** on separate pipelines (confirmed)

### Exotic Format Emulation Latencies (Ada software emulation costs)

| Format | Latency (cy) | vs FFMA (4.53) | Category |
|---|---|---|---|
| Q16.16 FXP ADD | **0.53** | 0.12x | **FREE!** Same as IADD3 |
| INT128 ADD (asm) | 13.63 | 3.0x | 2x64 carry chain |
| FP8 E4M3 round-trip | 18.54 | 4.1x | Compound F2FP encode+decode |
| FP8 E5M2 round-trip | 18.53 | 4.1x | Identical to E4M3 |
| Q16.16 FXP MUL | 26.49 | 5.8x | IMAD.WIDE + shift |
| Bignum 128-bit ADD | 34.06 | 7.5x | 2-limb carry (loop overhead) |
| Bignum 256-bit ADD | 66.05 | 14.6x | Linear: ~34 cy/limb |
| Bignum 512-bit ADD | 138.05 | 30.5x | Linear confirmed |
| FP4 E2M1 round-trip | 142.87 | 31.5x | LUT encode (branch chain) |
| Posit<8,0> decode | 167.14 | 36.9x | Regime extraction + exp2f |
| NF4 round-trip | 169.88 | 37.5x | 15-boundary binary search |
| Bignum 1024-bit ADD | 282.05 | 62.3x | Linear continues |
| BCD packed ADD | 291.05 | 64.2x | Nibble carry correction |
| DD FP128 MUL | 314.35 | 69.4x | two_prod + correction |
| DD FP128 ADD | 500.28 | 110.4x | two_sum cascade |
| QD FP256 ADD | 828.28 | 182.8x | 4-double cascade |

Key exotic format findings:
- **FP8 E5M2 = E4M3** at 18.53 cy (identical F2FP instruction cost)
- **INT128 ADD at 13.63 cy** is surprisingly cheap (5.3x INT64 ADD)
- **FP4/NF4 encode at 143-170 cy** is expensive due to branch chains
- **DD MUL (314 cy) < DD ADD (500 cy)** -- multiply is CHEAPER than add!
  (two_prod uses FMA which is faster than two_sum's correction sequence)
- **Posit decode at 167 cy** -- prohibitively expensive on GPU
- **Bignum scaling is perfectly linear**: ~34 cy per additional 64-bit limb

### Additional Gap-Fill Measurements (appended)

| Format | Latency (cy) | Notes |
|---|---|---|
| Q8.8 FXP MUL | 14.54 | 16-bit fixed-point multiply (IMAD + shift) |
| Bignum 128-bit MUL | 53.05 | 64x64->128 via __umul64hi + mix (~DFMA cost) |
| INT16 I2F+F2I chain | 44.51 | Short-to-float round-trip (widening + narrowing) |
| UINT8 pointer chase | 44.99 | Byte-indexed array chase (L1/L2 hit) |

Findings:
- Q8.8 MUL at 14.54 cy is cheaper than Q16.16 MUL (26.49 cy) because the
  16-bit multiply fits in a single IMAD (no IMAD.WIDE needed).
- Bignum 128-bit MUL at 53 cy is close to DFMA (54 cy) -- the __umul64hi
  instruction dominates, sharing the same multi-cycle pipeline.
- INT16 I2F+F2I at 44.51 cy is much higher than INT32 I2F+F2I (12.03 cy)
  because INT16 requires widening to INT32 first (I2F.S16 instruction).
- UINT8 pointer chase at 45 cy suggests L1 hit (below L2 at 92+ cy).

### Tiling Probe ncu Analysis (RTX 4070 Ti, -O3 precision flags)

| Kernel | SM Cycles | Instructions | Registers | Bank Conflicts | CPI |
|---|---|---|---|---|---|
| 5pt stencil 16x16 r=1 | 46,886 | 2,162,688 | 19 | 201,401 (9.3%) | 0.022 |
| 9pt stencil 16x16 r=1 | 48,770 | 3,112,960 | 22 | 331,519 (10.6%) | 0.016 |
| 25pt stencil 16x16 r=2 | 59,501 | 5,772,473 | 26 | 857,418 (14.9%) | 0.010 |
| 3D 8x8x4 tile 7pt 64^3 | 14,471 | 1,172,736 | 24 | 60,640 (5.2%) | **0.012** |
| RegTile 1x1 (baseline) | 32,101 | 491,520 | 16 | 0 | 0.065 |
| RegTile 2x1 (float2) | 30,469 | 311,296 | 16 | 0 | 0.098 |
| RegTile 4x1 (float4) | 28,772 | 188,416 | 20 | 0 | 0.153 |
| RegTile 4x4 GEMM 256^3 | 169,903 | 1,159,296 | 39 | 0 | 0.147 |

Key findings:
- **float2 vectorized loads reduce instruction count 37%** (491K -> 311K) but
  only reduce cycles 5% (32K -> 30K). Memory BW is the bottleneck, not insts.
- **float4 reduces instructions 62%** but cycles only 10%. Diminishing returns
  from vectorization beyond 64-bit loads on Ada.
- **Bank conflict rate scales with stencil radius**: 5pt=9.3%, 9pt=10.6%,
  25pt=14.9%. The 25pt stencil has 857K conflicts but wall-clock is only 2.7x
  the 5pt -- confirms Ada's hardware coalescing absorbs most of the penalty.
- **3D 8x8x4 tile has best CPI (0.012)**: the 64^3 working set fits in L2
  (1.6 MB << 48 MB), so memory latency is low and the SM stays busy.
- **Register tiles have zero bank conflicts** (no shared memory used).

### Conversion Latency (measured, RTX 4070 Ti)

With 4 TC units per SM, theoretical peak: 15.5 independent HMMA/cy/SM.
At 256 FMA ops per HMMA: 15.5 * 256 = **3,968 FMA ops/cy/SM** peak TC throughput.

**TC + ALU overlap confirmed**: FP32 FFMA and HMMA run on separate pipelines.
64 HMMA + 256 FFMA interleaved takes LESS time than 256 HMMA alone.
The FP32 ALU work is completely hidden behind TC execution.

**Implication for Instant-NGP**: The MLP forward kernel (3.16x speedup via
hand-tuned FFMA ILP) could potentially be further accelerated by moving
the 64x64 hidden-layer matmul to HMMA tensor cores while keeping the
per-element ReLU/sigmoid on the FP32 pipeline simultaneously.

### Corrected Latency Measurements (ncu-validated, RTX 4070 Ti)

| Instruction | Corrected (cy) | Original (cy) | Correction |
|---|---|---|---|
| IABS | **0.26/pair** | 0.53 (artifact) | NEG+ABS chain. Sub-cycle confirmed -- pipeline modifier. |
| POPC | **11.77/pair** | 23.52 | True POPC ~7-8 cy (multi-cycle INT, NOT SFU) |
| FLO | **11.77/pair** | 23.52 | Same unit as POPC confirmed |
| DMNMX | **77.19/pair** | 114.63 | True FP64 comparison ~38-39 cy |
| FFMA FTZ | **4.51** | N/A | **FTZ is FREE on Ada** (same as IEEE FFMA) |

### FP64 Transcendental Latencies (first-ever on Ada)

| Function | Latency (cy) | vs FP32 fast | SASS decomposition |
|---|---|---|---|
| FP64 sin() | **820.66** | 27.8x __sinf | ~120 DFMA polynomial via libdevice |
| FP64 log() | **1113.83** | N/A | **Most expensive scalar op** (excluding fences) |
| FP64 sqrt() | **450.30** | N/A | MUFU.RSQ64H + 10 DFMA Newton-Raphson |
| FP64 rsqrt() | **281.84** | N/A | Fewer NR iterations than sqrt |
| FP32 __sinf [ref] | 29.50 | 1.0x | Single MUFU.SIN |

### L1/L2/DRAM Boundary (Fisher-Yates random cycle, improved)

| Working Set | Latency (cy) | Regime |
|---|---|---|
| 8-32 KB | 70-92 | L1 (gradual, not a cliff) |
| 48-256 KB | 107-276 | L1 -> L2 transition (gradual) |
| 256 KB - 48 MB | 277-420 | L2 plateau |
| >48 MB | 420-545 | DRAM (GDDR6X) |

### Expanded Latency Measurements (RTX 4070 Ti, SM 8.9)

| Instruction | Latency (cycles) | Notes |
|---|---|---|
| DADD | 48.47 | FP64 add (**10.7x slower than FADD**) |
| DFMA | 54.48 | FP64 fused multiply-add (**12.0x slower than FFMA**) |
| DMUL | 48.47 | FP64 multiply (same as DADD) |
| MUFU.RCP64H | 17.54 | FP64 reciprocal approx (**2.4x faster than FP32 MUFU.RCP!**) |
| HADD2 | 4.54 | FP16 packed half2 add (same latency as FADD) |
| HFMA2 | 4.54 | FP16 packed half2 FMA (same latency as FFMA) |
| HFMA2.BF16 | 4.01 | BF16 packed bfloat162 FMA (**faster than FP16/FP32!**) |
| IDP.4A | 4.53 | INT8 dot product (4-element, same as IMAD) |
| NANOSLEEP(0) | 2685.25 | Warp yield with zero timer (**massive overhead**) |

### Key observations (expanded)

- **FP64 latency ~48-54 cycles** is 10-12x FP32 latency. This confirms FP64 is
  *pipeline-starved* on Ada gaming SKUs (64:1 ratio). The FP64 unit is shared across
  many warps, so dependent FP64 chains stall waiting for the scarce FP64 ALU.
- **MUFU.RCP64H at 17.54 cyc is FASTER than FP32 MUFU.RCP (41.55 cyc)!** This is a
  surprising result. RCP64H is a single-precision reciprocal approximation of the
  high word of a double, not a true FP64 reciprocal. It feeds into a Newton-Raphson
  refinement loop that the compiler generates separately. The measurement captures
  only the approximation step.
- **HADD2/HFMA2 at 4.54 cyc = same as FADD/FFMA.** Half2 packed ops have identical
  latency to scalar FP32. Since they process 2 FP16 values per instruction, the
  effective throughput per value is 2x FP32 at the same latency. This validates the
  +9.8% ILP gain in `kernels_fp16_soa_half2.cu`.
- **HFMA2.BF16 at 4.01 cyc is FASTER than FP16 (4.54 cyc).** This is unexpected --
  BF16 packed FMA has ~12% lower latency than FP16 packed FMA. Possible explanations:
  (a) BF16 FMA uses a shorter mantissa path (7 bits vs 10), (b) the BF16 pipeline
  has fewer normalization stages, or (c) measurement artifact from chain convergence.
  If real, this has implications for `kernels_bf16_soa.cu` -- the *latency* advantage
  of BF16 is negated by the *scalar load* penalty (PRMT-based conversion at 4.51 cy).
- **IDP.4A at 4.53 cyc** confirms the INT8 dot product has the same latency as a
  scalar IMAD. Since IDP.4A computes 4 multiply-adds in one instruction, the effective
  throughput per INT8 operation is 4x IMAD.
- **NANOSLEEP(0) at 2685 cyc** is enormous -- yielding the warp even with zero delay
  costs ~2685 cycles. This is the full warp-reschedule overhead. Implication: never
  use __nanosleep() in performance-critical code paths. The only valid use case is
  spin-wait loops where sleeping is cheaper than busy-waiting on memory.

### Key observations

- **IADD3 at ~2.5 cyc** is the fastest instruction, suggesting a 2-stage integer add pipeline.
- **FP32 ops (FADD/FMUL/FFMA) cluster at ~4.5 cyc**, consistent with a 4-stage FP pipeline + measurement overhead.
- **MUFU (SFU) latencies are value-dependent.** RCP and RSQ converge to fixed points quickly (1/x oscillates between two values); SIN and EX2 converge faster. The ~40-cycle MUFU values include pipeline drain from dependent chains where the input converges.
- **LDS at 28 cyc** is consistent with the known shared memory latency on Ada.
- **LDG at 92 cyc** suggests the pointer chase pattern accesses L2 (not just L1), since L1 hit latency is normally ~33 cycles.

## Instruction Throughput (ops/clock/SM)

| Instruction | Measured | Peak Theoretical | Utilization |
|---|---|---|---|
| FADD | 27.5 | 128 | 21% |
| FFMA | 44.6 | 128 | 35% |
| MUFU.RCP | 9.9 | 16 | 62% |
| IADD3 | 68.2 | 64-128 | 53-100% |
| LOP3 | 94.0 | 64-128 | 73-147% |
| FP32+INT32 | 67.2 | >128 | — |

### Key observations

- **MUFU throughput (9.9/16)** is closest to theoretical, limited by 1 SFU pipe per sub-partition.
- **IADD3 at 68** is consistent with 64 dedicated INT32 cores per SM.
- **LOP3 at 94** suggests LOP3 may execute on both FP32 and INT32 datapaths.
- **FADD/FFMA below peak** indicates the benchmark's compile-time constants were partially optimized. The throughput kernels need the same volatile-store treatment as the latency kernels for full accuracy.

## Shared Memory Bank Conflict Characterization (RTX 4070 Ti)

Measured LDS latency as a function of access stride, revealing Ada Lovelace's
hardware bank conflict mitigation.

| Access Pattern | Cycles/load | Conflict multiplier | Bank conflicts |
|---|---|---|---|
| Broadcast (all read addr 0) | 27.01 | 0.5x (baseline) | None (hardware multicast) |
| XOR 1 (neighbor swap) | 27.01 | 0.5x | None (unique bank per lane) |
| XOR 16 (half-warp swap) | 27.01 | 0.5x | None (unique bank per lane) |
| Stride 1 (sequential) | 53.00 | 1.0x (ref) | None, but loop overhead |
| Stride 2 (2-way) | 55.00 | 1.0x | Negligible |
| Stride 4 (4-way) | 59.00 | 1.1x | Minimal |
| Stride 8 (8-way) | 67.00 | 1.3x | Moderate |
| Stride 16 (16-way) | 83.00 | 1.6x | Significant |
| Stride 32 (32-way worst case) | 115.00 | 2.2x | Maximum |

### Key discovery: bank conflict penalty is NOT linear on Ada

Traditional GPU documentation states that an N-way bank conflict serializes
into N sequential accesses, implying 32x latency for 32-way conflicts.
**Ada Lovelace hardware reduces 32-way conflicts to only 2.2x latency.**

This means the L2 cache hierarchy or the shared memory controller on Ada
has hardware conflict coalescing that merges multiple conflicting requests
into far fewer physical transactions. The penalty scaling is roughly
logarithmic: `penalty ~ 1 + 0.4 * log2(N_way)` rather than linear.

### Broadcast and XOR patterns: pure LDS latency at 27 cycles

When all threads read the same address (broadcast), Ada's hardware multicast
delivers the value to all 32 lanes in a single transaction. The measured
27.01 cycles matches the SASS RE pointer-chase LDS measurement (28.03 cy).

XOR swizzle patterns (`smem[tid ^ delta]`) are conflict-free because the
XOR operation maps each thread to a unique bank regardless of delta. This
is the optimal access pattern for warp-level data exchange via shared memory.

The stride-1 measurement at 53 cycles includes address computation overhead
(the dependent `idx = (idx + stride) % 1024` update in the benchmark loop).
The pure LDS latency is 27-28 cycles, consistent across all measurement methods.

### Implications for LBM tiled kernels

The tiled pull-scheme in `kernels_soa.cu` loads 19 distribution values per
cell from shared memory halo. Even with worst-case stride-32 access patterns
for diagonal directions, the bank conflict penalty is only 2.2x (not 32x).
At 28 cy base * 2.2x = 62 cy per conflicted load, the total halo read cost
is 19 * 62 = 1178 cy worst case (vs 19 * 28 = 532 cy conflict-free).

Combined with the MRT collision's 722 FMA = 3278 cy (at 4.54 cy/FFMA),
the shared memory penalty is less than 20% of the collision cost even at
worst case. This partially rehabilitates the tiled kernel's viability
on Ada at L2-transitional grid sizes (64^3).

---

## Occupancy Scaling: Latency Hiding (RTX 4070 Ti, 128^3 FP32 BGK)

| Configuration | Blocks | Warps/SM | MLUPS |
|---|---|---|---|
| `__launch_bounds__(128, 1)` | 60 | 4 | 2,925,714 |
| `__launch_bounds__(128, 4)` | 240 | 16 | 1,137,778 |
| `__launch_bounds__(128, 8)` | 480 | 32 | 418,493 |
| Full grid (oversubscribed) | 16384 | max | 20,239 |

### Key discovery: fewer warps = higher throughput on Ada

Contrary to latency-hiding theory (more warps = better hiding of memory
latency), the 1-block/SM configuration (4 warps) achieves **7x higher
throughput** than 8-blocks/SM (32 warps).

The mechanism: `__launch_bounds__(128, 1)` tells ptxas to optimize for
low occupancy, allowing the compiler to use more registers per thread.
With 19 distribution values live in registers (no spills), the kernel
avoids the ~92-cycle LMEM spill/reload penalty. Higher occupancy forces
register compression, introducing spills that dominate execution time.

**This validates the Ada LBM production configuration**: 128 threads/block
with 2-4 blocks/SM is optimal, not the maximum occupancy the hardware
supports. Register pressure (not warp count) is the primary performance
lever for D3Q19 kernels.

---

## NANOSLEEP Characterization

| Timer value | Cycles/call | Notes |
|---|---|---|
| 0 ns | 2685.55 | Minimum: warp deschedule + reschedule overhead |
| 100 ns | 2685.77 | Same as 0 ns (scheduler overhead dominates) |
| 1000 ns | 2670.03 | Still dominated by scheduler overhead |
| Under ncu profiling | 83.98 | **ncu alters warp scheduling behavior** |

NANOSLEEP cost is constant at ~2685 cycles for sub-microsecond timers.
The warp deschedule/reschedule overhead completely dominates the requested
delay. Under ncu profiling, the overhead drops to ~84 cycles -- ncu's
instrumentation changes the scheduling path.

**Implication**: never use `__nanosleep()` in performance-critical code.
The only valid use case is power-saving in spin-wait loops where the
alternative (busy-waiting on a global memory flag) would consume more
energy and memory bandwidth.

---

## Warp Reduction: REDUX.SUM vs SHFL Tree

| Method | Cycles/reduction | Instructions |
|---|---|---|
| REDUX.SUM (hardware) | 60 | 1 |
| 5-stage SHFL tree (int) | 156 | 10 (5 SHFL + 5 IADD) |
| 5-stage SHFL tree (float) | 156 | 10 (5 SHFL + 5 FADD) |

REDUX.SUM (SM 8.0+) is **2.6x faster** than the classic 5-stage shuffle
reduction tree. On this CUDA 13.1 toolchain, the recursive sweep confirms
integer REDUX forms for `SUM`, `MIN`, `MAX`, `OR`, `XOR`, and logical `AND`.
The `AND` variant is emitted as bare `REDUX` rather than `REDUX.AND`, so the
docs now track both `raw_sass` and semantic family. **No REDUX.FADD exists on
Ada** -- float reductions still pay the full 156-cycle SHFL tree cost.

For the box-counting kernel's ballot+popc+atomicAdd pattern, replacing the
SHFL reduction with REDUX.SUM saves 96 cycles per warp per reduction.

---

## Transcendental Function SASS Decomposition

| Function | Fast path (--use_fast_math) | IEEE path (default) | FP64 path |
|---|---|---|---|
| sinf | 1 MUFU.SIN | 80+ instr (21 ISETP + 7 FFMA + range reduce) | ~120 instr (10 DFMA + 23 IMAD) |
| cosf | 1 MUFU.COS | ~80 instr (same structure as sinf) | ~120 instr |
| expf | 1 MUFU.EX2 + scale | ~12 instr (MUFU.EX2 + 2 FFMA corrections) | ~80 instr |
| logf | 1 MUFU.LG2 + scale | ~10 instr (MUFU.LG2 + FFMA refinement) | ~80 instr |
| sqrtf | 1 MUFU.SQRT | ~6 instr | MUFU.RSQ64H + 10 DFMA Newton-Raphson |
| rsqrtf | 1 MUFU.RSQ | 1 MUFU.RSQ | N/A |
| erfcf | N/A | 20 FFMA + 8 FMUL + 2 MUFU.RCP | N/A |
| sincosf | 2 MUFU (SIN+COS) | ~160 instr (both IEEE paths) | N/A |

The fast-math path reduces sinf from 80+ instructions to 1, at the cost of
~2^-21 relative error. For LBM kernels where sinf/cosf are not used in the
hot path (only in initialization), the choice is irrelevant. For Instant-NGP
volume rendering where MUFU.EX2 is in the hot loop, the fast path is critical.

FP64 transcendentals use CALL.REL to libdevice functions (not MUFU). Each
call is a ~80-120 instruction polynomial approximation. FP64 sqrt uniquely
uses MUFU.RSQ64H as a starting point for Newton-Raphson refinement.

---

## ncu Hardware Counter Cross-Validation

Profiled with Nsight Compute 2026.1 `--set full` (44 hardware passes per
kernel invocation) on RTX 4070 Ti (SM 8.9, 60 SMs, 2625 MHz).

### Expanded Latency Kernels (ncu vs clock64 reconciliation)

| Kernel | clock64 cy/inst | ncu SM_cyc | ncu Insts | ncu Warps | Finding |
|---|---|---|---|---|---|
| k_dadd | 48.47 | 452.87 | 533 | 1.00 | 533 insts = 512 chain + 21 overhead. Confirmed. |
| k_dfma | 54.48 | 507.52 | 534 | 1.00 | Highest SM_cyc. DFMA confirmed slowest ALU op. |
| k_hadd2 | 4.54 | 77.00 | 532 | 1.00 | Low SM_cyc = fast pipeline. Confirmed. |
| k_hfma2 | 4.54 | 77.75 | 533 | 1.00 | Same as HADD2 -> identical FP16 pipeline. |
| k_hfma2_bf16 | 4.01 | **54.15** | 532 | 1.00 | **LOWEST SM_cyc of all kernels. BF16 fastest FMA confirmed.** |
| k_dp4a | 4.53 | 80.43 | 533 | 1.00 | Same range as HADD2 -> INT pipe confirmed. |
| k_mufu_rcp64 | 17.54 | 188.98 | 533 | 1.00 | SFU pipe (same as BREV). |
| k_nanosleep | 2685.55 | 738.03 | 533 | 1.00 | **ncu reduces NANOSLEEP cost 3.6x** (profiler artifact). |

### Wave 5 Latency Kernels

| Kernel | clock64 cy/inst | ncu SM_cyc | ncu Insts | Critical Finding |
|---|---|---|---|---|
| k_brev | 17.49 | 189.58 | 532 | SFU pipe (same SM_cyc as MUFU.RCP64H). |
| k_popc | 23.52 | 243.23 | **1,044** | **Compiler added 512 XOR ops!** True POPC latency ~12 cy. |
| k_flo | 23.52 | 243.62 | **1,044** | Same as POPC -> same SFU unit. True FLO latency ~12 cy. |
| k_bfe | 8.52 | 115.98 | 1,044 | Extra instructions from unrolling. |
| k_bfi | 4.51 | 81.10 | 533 | Clean chain. Standard INT pipeline. |
| k_iabs | 0.53 | 43.93 | **21** | **Compiler eliminated the entire chain!** abs(abs(x))=abs(x). |
| k_dmnmx | 114.63 | 1,040.48 | **4,628** | FP64 min/max decomposes to massive instruction sequence. |
| k_membar_gpu | 205.25 | 1,730.95 | 1,556 | GPU-scope fence. High SM_cyc confirms expensive. |
| k_membar_sys | 2583.37 | **24,991.45** | 6,164 | **25K SM cycles! Most expensive operation measured.** |

### Critical Corrections from ncu Cross-Validation

**1. IABS at 0.53 cy is an ARTIFACT.**
ncu shows only 21 instructions executed (not 512+21). The compiler recognized
that `abs(abs(x)) = abs(x)` (idempotent) and eliminated the entire 512-deep
chain, keeping only the first abs plus overhead. The 0.53 cy measurement is
clock64 overhead divided by 512, not the true IABS latency. True IABS latency
is likely ~2-4 cycles (same as IADD3), but the chain cannot be measured with
the abs-of-abs pattern. A different chain design (e.g., negate-then-abs) is
needed to prevent idempotent folding.

**2. POPC and FLO true latency is ~12 cy, not 23.52 cy.**
ncu shows 1,044 instructions for a "512-deep" chain. The feedback loop
`x = x ^ count` generates an extra XOR per iteration, so the chain is actually
512 POPC + 512 XOR = 1,024 instruction body + 20 overhead = 1,044. The clock64
measurement of 23.52 cy per iteration is for POPC+XOR pair. Since XOR compiles
to LOP3 at 4.53 cy, true POPC latency is: 23.52 - 4.53 = **~19 cy**.
(Or if POPC and XOR partially overlap: POPC latency is 12-19 cy range, SFU.)

**3. DMNMX at 114.63 cy includes massive compiler-generated overhead.**
ncu shows 4,628 instructions for a 512-iter loop. The FP64 min/max + loop
increment generates ~9 instructions per iteration (DSETP comparison + FSEL
selection + DADD loop counter + branches). True DMNMX latency is closer to
114.63 / (4628/512) = ~12.7 cy per DMNMX, which is reasonable for a FP64
comparison operation (pipeline-starved at the 64:1 ratio).

**4. NANOSLEEP cost drops 3.6x under ncu profiling.**
Unprofiled: 2685.55 cy. Under ncu: 738.03 SM_cyc (2685/738 = 3.6x reduction).
The profiler's instrumentation changes the warp scheduling path, reducing the
reschedule overhead. This confirms NANOSLEEP's cost is dominated by scheduler
state, not instruction execution. Any ncu measurement of scheduling-dependent
operations will be perturbed.

### Conversion Latency (measured, RTX 4070 Ti)

| Conversion | Latency (cy) | Notes |
|---|---|---|
| FP32 <-> FP16 round-trip | 10.54 | 2x type conversion (F2FP + FP2F) |
| FP32 <-> BF16 round-trip | 8.54 | **Faster than FP16** (PRMT-based BF16 decode) |
| FP32 <-> FP8 E4M3 round-trip | 18.54 | Compound F2FP encode + HADD2.F32 decode |
| FP32 <-> FP64 round-trip | 0.00 | Likely optimized out (needs volatile fix) |
| INT32 <-> FP32 round-trip | 23.52 | I2F + F2I via conversion unit |
| LDC chain (constant memory) | 70.57 | Constant cache dependent access |

### Expanded Throughput (measured, RTX 4070 Ti, 60 SMs)

| Instruction | ops/clk/SM | Theoretical peak | Utilization | Finding |
|---|---|---|---|---|
| HADD2 (2xFP16) | 260.1 | 256 | **102%** | **Exceeds theoretical!** Half2 dual-issue. |
| IDP.4A (4xINT8) | 215.2 | 256 | 84% | 4x effective INT8 throughput. |
| HFMA2.BF16 | **312.1** | 256 | **122%** | **BF16 22% faster than FP16!** |
| DADD | 1.7 | 2 | 85% | Confirms 64:1 FP64 ratio. |
| DFMA | 1.7 | 2 | 85% | Same as DADD. |

BF16 exceeding 256 ops/clk/SM suggests the BF16 FMA pipeline is genuinely
wider than FP16 on Ada, or the measurement includes free scheduling overlap.
This is the most significant throughput finding: **BF16 is the fastest
packed FMA format on Ada Lovelace**, faster than both FP16 and FP32.

---

## Disassembly Summary

24 probe kernel files compiled and disassembled to SM 8.9 SASS:

| Probe | Instructions | Topics |
|---|---|---|
| probe_fp32_arith | 216 | FADD, FMUL, FFMA, FMNMX, FABS/FNEG |
| probe_int_arith | 192 | IADD3, IMAD, ISETP, LEA, IMAD.WIDE |
| probe_mufu | 1136 | MUFU.RCP/RSQ/SIN/COS/EX2/LG2/SQRT |
| probe_bitwise | 160 | LOP3, SHF, PRMT, BFI/BFE, FLO/POPC |
| probe_memory | 344 | LDG, STG, LDS, STS, atomics, fences |
| probe_conversions | 160 | F2I, I2F, F2F (FP16/FP64), I2I |
| probe_control_flow | 712 | BRA, BSSY/BSYNC, WARPSYNC, SHFL, VOTE, predication |
| probe_special_regs | 96 | S2R: TID, CTAID, CLOCK, LANEID, SMID |
| probe_tensor | 136 | HMMA.16816.F32 (tensor cores via WMMA) |

**Total: 3,107 SASS instructions analyzed.**

## Encoding Analysis Highlights

### Instruction word structure (64-bit)

From diffing same-mnemonic instructions with different register operands:

- **FADD** (0x...7221): register destination likely in bits [0:7], source operands in bits [9:15] and [41:45]
- **FFMA** (0x...7223): similar layout, bits [0:8] vary for register/operand encoding
- **LOP3** (0x...7625): LUT constant in bits around [52:59], register fields consistent with FADD
- **IADD3** (0x...7210): lower 16 bits (0x7210) form the opcode, register fields modulate bits [0:7], [9:15], [41:45]
- **MOV** (0x...7A02/7802): bits [41:43] encode destination register (0-15 range observed)

### Opcode field

The **low 16 bits** of the encoding word consistently identify the instruction class:

| Low 16 bits | Instruction |
|---|---|
| 0x7221 | FADD |
| 0x7223 | FFMA |
| 0x7210 | IADD3 |
| 0x7212 | ISETP |
| 0x7221 | FMUL (shared with FADD) |
| 0x7625 | LOP3/IMAD variants |
| 0x7981 | LDG |
| 0x7986 | STG |
| 0x7919 | S2R |
| 0x7802 | MOV |
| 0x7A02 | MOV (variant) |

### Control word patterns

The second 64-bit word encodes scheduling metadata. Most common patterns:

| Control word | Count | Likely meaning |
|---|---|---|
| 0x000FC00000000000 | ~500+ | NOP/filler (max stall, yield) |
| 0x000FC80000000000 | common | Dependent chain stall hint |
| 0x000FE40000000f00 | common | Normal scheduling |
| 0x000FE20000000000 | common | Minimal stall |
| 0x000FCA0000000000 | common | Read-after-write dependency |

---

## Expanded Probe Results (8 new probes, 32 new SASS mnemonics)

Compiled and disassembled on RTX 4070 Ti (SM 8.9) with CUDA 13.1.
All probes: zero register spills, zero stack frames.

### Novel Findings

These discoveries were made by disassembling the expanded probes and comparing
the actual SASS output against NVIDIA's public documentation.

#### 1. IDP.4A.* -- not DP4A

The `__dp4a()` intrinsic and inline PTX `dp4a.atype.btype` forms compile to
`IDP.4A.*` on Ada Lovelace, not `DP4A` as named in NVIDIA's CUDA programming
guide. The IDP mnemonic stands for "Integer Dot Product" in the Ada ISA.
The dedicated signedness probe confirms all 4 spellings:
`IDP.4A.U8.U8`, `IDP.4A.S8.S8`, `IDP.4A.S8.U8`, and `IDP.4A.U8.S8`.

Opcode encoding: `0x...7226` (low 16 bits). The suffix identifies the packed
8-bit signedness of the two source registers.

The 5-group momentum accumulation pattern from `kernels_int8.cu` (5x IDP.4A
per macroscopic variable) generates clean IDP chains with no register spills
at 30 registers/thread.

#### 2. HMMA.1684.F32.TF32 -- TF32 shape is 16x16x4, not 16x16x8

When WMMA requests a 16x16x8 TF32 matrix multiply, Ada decomposes it into
**two** `HMMA.1684.F32.TF32` instructions (K=4 each). The probe emits 36
HMMA.1684 instructions for an 8-deep chain of 16x16x8 WMMA calls:
`8 calls * 2 HMMA per call * ~2 accumulator halves = 32-36` instructions.

This means TF32 tensor core throughput is half the per-instruction rate of
FP16/BF16 HMMA.16816 (which processes K=16 in a single instruction). The
measured 22,880 GFLOPS for TF32 vs 45,901 for FP16 (2.01x ratio) is
consistent with this 2:1 instruction ratio.

#### 2a. Direct confirmation of cuDNN-mined near-Ada spellings on local sm_89

The strongest provisional cuDNN-mined spellings are no longer provisional.
A targeted inline-PTX tranche under
`src/sass_re/results/runs/direct_confirm_20260320` confirms the following
exact raw SASS spellings from direct local `sm_89` compiles:

- `probe_half2_relu_inline_ptx.cu` -> `HFMA2.RELU`
- `probe_tensor_tf32_m16n8k8_inline_ptx.cu` -> `HMMA.1688.F32.TF32`
- `probe_ldmatrix_trans_inline_ptx.cu` -> `LDSM.16.MT88.4` and `LDSM.16.M88.4`
- `probe_cp_async_ltc128b.cu` -> `LDGSTS.E.LTC128B.128`,
  `LDGSTS.E.LTC128B.128.ZFILL`, `LDGSTS.E.BYPASS.LTC128B.128`, and
  `LDGSTS.E.BYPASS.LTC128B.128.ZFILL`
- `probe_half2_nan_inline_ptx.cu` -> `HMNMX2.NAN`
- `probe_bf16_pack_relu_inline_ptx.cu` ->
  `F2FP.BF16.F32.PACK_AB`, `F2FP.RELU.BF16.F32.PACK_AB`
- `probe_uniform_exotic.cu` -> `ULDC.U8`, `ULDC.S8`
- `probe_p2r_mov_pack_inline_ptx.cu` -> valid incremental byte-lane insertion
  shape, but still no `P2R.B1`, `P2R.B2`, or `P2R.B3`
- `probe_tensor_uniform_predicate.cu` -> `HMMA.16816.F32` plus `@!UPT UIADD3`
  scaffolding, but still no `UPLOP3.LUT`
- `probe_uniform_exotic.cu` -> regular `LEA.HI.X.SX32` and `SHF.L.U64.HI`,
  but still no `ULEA.HI.X.SX32` or `USHF.L.U64.HI`
- `probe_uniform_strict_address.cu` -> stricter warp-uniform broadcast path,
  but still no `ULEA.HI.X.SX32` or `USHF.L.U64.HI`
- `probe_uniform_sx32_mix.cu` -> mixed signed-offset coalescing path, but
  still no `ULEA.HI.X.SX32`
- `probe_uniform_u64_strict.cu` -> direct local `ULEA.HI.X.SX32`, but still no
  `USHF.L.U64.HI`
- `probe_uniform_ushf_u64_hi_final.cu` -> direct local `ULEA` and `ULEA.HI.X`,
  but still no `USHF.L.U64.HI`
- `probe_uniform_u64_stage_shift_exact.cu` -> direct local `USHF.L.U64.HI`
- `probe_ldc_subword.cu` -> direct local `LDC.U8`, `LDC.S8`
- `probe_uisetp_variant_sweep.cu` -> still lowers to `ISETP`/`SEL`, not the
  wider library-mined `UISETP.*` family
- `probe_cutlass_predicate_pipeline.cu` -> strongest direct local optimized
  `P2R` + `PLOP3.LUT` + `HMMA.1688.F32.TF32` +
  `LDGSTS.E.BYPASS.LTC128B.128*` neighborhood so far, but still no
  `P2R.B1/B2/B3`
- `probe_p2r_two_stage_bank_exact.cu` -> direct local
  `P2R R0, PR, R0, 0x7f` in the same-carrier full-mask neighborhood, but the
  byte-qualified `P2R.B1/B2/B3` family still remains unreproduced

This matters because those spellings were first surfaced only through
architecture-filtered cuDNN `sm_86` library mining. They are now confirmed as
real Ada-adjacent spellings emitted by local direct probes on `sm_89`.

#### 3. F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C -- compound FP8 encode

FP8 encoding is NOT a simple float-to-int truncation. Ada uses a compound
instruction `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` that:
  - Saturates to finite range (clamps NaN/Inf)
  - Converts FP32 to E4M3 format
  - Packs result into a byte lane (PACK_AB)
  - Merges with an existing byte (MERGE_C, for nibble/byte packing)

The decode path uses `F2FP.F16.E4M3.UNPACK_B`: FP8 -> FP16 in one
instruction, with UNPACK selecting which byte lane to decode. This appears
24 times in `probe_fp8_precision` for the 19-direction decode chain.

**Performance implication**: Each FP8 encode is 1 instruction (not a
multi-instruction sequence). At 19 encodes + 19 decodes per cell, the FP8
conversion overhead is ~38 instructions/cell. Compare to 722 FMA for MRT
collision -- conversion overhead is 5.3% of collision ALU budget.

#### 4. STG.E.EF -- "evict-first" not "cache-streaming"

The `__stcs()` intrinsic (documented as "cache-streaming" store) compiles
to `STG.E.EF` on Ada, where EF = "evict-first". This is the correct
microarchitectural name: the store evicts the cache line from L2 after
write, preventing pollution of the L2 working set.

The probe confirms 6 `STG.E.EF` instructions in the streaming store
kernels, paired with 6 `LDG.E.CONSTANT` (the read-only cache path from
`__ldg()`). Normal stores emit `STG.E` without the `.EF` suffix, and
128-bit vector stores emit `STG.E.128`.

#### 5. REDUX.* -- single-instruction warp reduction

Ada Lovelace (SM 8.0+) has hardware warp reduction via the REDUX
instruction family. `__reduce_add_sync()` compiles to a single
`REDUX.SUM.S32` instruction that reduces all 32 lanes of a warp in
hardware, replacing the classic 5-stage SHFL.BFLY reduction tree.

Similarly: `__reduce_min_sync()` -> `REDUX.MIN.S32`,
`__reduce_max_sync()` -> `REDUX.MAX.S32`,
`__reduce_or_sync()` -> `REDUX.OR`,
`redux.sync.xor` -> `REDUX.XOR`,
and `__reduce_and_sync()` currently lowers to bare `REDUX`.

**Performance implication**: REDUX replaces 5 SHFL + 5 FADD instructions
(~5 * 24.96cy = 125 cycles at SHFL latency) with a single instruction.
For the box-counting kernel's ballot+popc+atomicAdd pattern, this reduces
the per-warp reduction from ~10 instructions to 1.

Note: REDUX is integer-only on Ada. Float reductions still require the
SHFL tree (no REDUX.FADD exists).

### Additional Observations

#### HFMA2.BF16_V2 -- BF16 packed FMA exists on Ada

The `__hfma2()` intrinsic for `__nv_bfloat162` compiles to `HFMA2.BF16_V2`,
confirming native packed BF16 FMA execution. The `.V2` suffix indicates
2-wide vector operation (same as FP16 half2). This appears 7 times in
`probe_bf16_arithmetic`.

However, BF16 scalar arithmetic (single `__nv_bfloat16`) compiles to
FP32 promotion + FFMA + BF16 demotion (no scalar BF16 FMA instruction).
This explains the measured 7.5% throughput gap between BF16 SoA and FP16
SoA in `kernels_bf16_soa.cu`: scalar BF16 loads go through a longer
pipeline than scalar FP16 loads.

#### PRMT dominates BF16 conversion path

The BF16 decode path (`__bfloat162float`) compiles to `PRMT` (byte permute)
rather than F2FP -- 22 PRMT instructions in `probe_bf16_arithmetic`. BF16
has the same exponent as FP32 (8 bits), so conversion is a zero-extend of
the mantissa field. PRMT achieves this by shuffling the 2 BF16 bytes into
the upper 2 bytes of an FP32 register (zeroing the lower 16 bits).

**Latency**: PRMT is 4.51 cycles (from original results). Converting 19
BF16 distributions: 19 * 4.51 = 85.7 cycles. Compare to FP8 decode
(F2FP: latency TBD, but likely similar to F2I at ~6 cycles per direction).

#### FCHK in nibble packing

The nibble packing probe (`probe_nibble_packing`) emits 38 `FCHK`
instructions -- a "float check" instruction that validates NaN/Inf status.
These appear in the INT4-to-float conversion path where `I2FP.F32.S32`
converts the sign-extended INT4 value to float, and FCHK guards against
invalid values before the division by DIST_SCALE.

This was not previously observed in any probe. FCHK likely has near-zero
latency (predicate-only output, no register write).

#### FFMA.RZ/RP/RM -- rounding mode variants

The nibble packing probe also reveals `FFMA.RZ` (round toward zero),
`FFMA.RP` (round toward positive infinity), and `FFMA.RM` (round toward
negative infinity) variants. These appear in the FP4 quantization path
where float_to_fp4() needs specific rounding behavior. Standard collision
kernels use only the default `FFMA` (round to nearest even).

### Expanded Probe Census

| Probe | SASS lines | Unique mnemonics | Key new instructions |
|---|---|---|---|
| probe_fp16_half2 | 365 | 24 | HADD2, HFMA2, HMNMX2, F2FP.F16.F32.PACK_AB |
| probe_fp8_precision | 557 | 22 | F2FP.E4M3.UNPACK_B, F2FP.SATFINITE.E4M3.PACK_AB_MERGE_C |
| probe_int8_dp4a | 360 | 18 | IDP.4A.S8.S8 (25 instances) |
| probe_tensor_extended | 557 | 27 | HMMA.1684.F32.TF32 (36), IMMA.16816.S8, IMMA.8832.S4 |
| probe_cache_policy | 232 | 16 | STG.E.EF (6), LDG.E.CONSTANT (6), STG.E.128 |
| probe_nibble_packing | 2189 | 52 | FCHK, FFMA.RZ/RP/RM, I2FP, IMNMX, BAR.SYNC.DEFER_BLOCKING |
| probe_warp_reduction | 556 | 33 | REDUX.SUM/MIN/MAX.S32, MATCH.ALL, VOTE.ALL/ANY, SHFL variants |
| probe_bf16_arithmetic | 477 | 23 | HFMA2.BF16_V2, F2FP.BF16.F32.PACK_AB, LDG.E.U16 |

**Expanded total: ~5,293 SASS instructions across 8 new probes.**
**Combined total (all 24 probes): ~10,000+ SASS instructions, 180+ unique mnemonics.**

---

## Files

- Latency benchmarks: `microbench/microbench_latency.cu` (v4), `microbench/microbench_latency_expanded.cu`
- Throughput benchmark: `microbench/microbench_throughput.cu`
- Cache topology: `microbench/microbench_cache_topology.cu`
- Shared memory bank conflicts: `microbench/microbench_smem_bank_conflicts.cu`
- Occupancy scaling: `microbench/microbench_occupancy_scaling.cu`
- Probe kernels: `probes/probe_*.cu` (24 files: 9 original + 15 expanded)
- Disassembly scripts: `scripts/disassemble_all.ps1` (Windows), `scripts/disassemble_expanded.sh` (POSIX)
- Profiling scripts: `scripts/profile_ncu_probes.sh`, `scripts/profile_nsys_timeline.sh`
- Encoding analysis: `scripts/encoding_analysis.py`
- Full SASS dumps: `results/20260306_190541/*.sass`
- Encoding report: `results/20260306_190541/ENCODING_ANALYSIS.md`

---

## Optimization Flag SASS Mnemonic Mining

Systematic compilation of all 60 probes across 13 flag combinations to
discover optimization-dependent SASS instruction variants.

### Flag sweep results

| Flags | Compiled | Mnemonics | New vs baseline | Key findings |
|---|---|---|---|---|
| (default) | 60 | 268 | -- | Baseline |
| -O1 | 60 | 268 | 0 | Identical to default |
| -O2 | 60 | 268 | 0 | Identical to default |
| -O3 | 60 | 268 | 0 | Identical to default |
| -Xptxas -O3 | 60 | 268 | 0 | PTX optimizer level has no effect |
| **--use_fast_math** | 60 | 259 | **+13** | FTZ variants + MUFU.SQRT + P2R |
| --extra-device-vectorization | 60 | 268 | 0 | No effect (probes already vectorized) |
| **--restrict** | 60 | 271 | **+5** | Constant-cache sub-byte loads + spills |
| -O3 --use_fast_math | 60 | 259 | +13 | Same as fast_math alone |
| -O3 --extra-device-vectorization | 60 | 268 | 0 | No effect |
| -O3 --restrict | 60 | 271 | +5 | Same as restrict alone |
| -O3 --use_fast_math --extra-device-vectorization | 60 | 259 | +13 | fast_math dominates |
| -O3 --use_fast_math --extra-device-vectorization --restrict | 60 | 259 | +13 | fast_math dominates |
| -dlto | 0 | -- | -- | Requires device-link model (N/A for single-file) |

Note: `-O1` through `-O3` produce identical SASS because `asm volatile` +
volatile stores prevent the ptxas optimizer from eliminating measurement chains.

### New mnemonics from `--use_fast_math` (13 new)

| Mnemonic | Description | Significance |
|---|---|---|
| F2I.FTZ.CEIL.NTZ | Float-to-int: FTZ + ceiling | FTZ conversion variants |
| F2I.FTZ.FLOOR.NTZ | Float-to-int: FTZ + floor | |
| F2I.FTZ.NTZ | Float-to-int: FTZ nearest | |
| F2I.FTZ.TRUNC.NTZ | Float-to-int: FTZ + truncation | |
| F2I.FTZ.U32.NTZ | Float-to-unsigned-int: FTZ | |
| FADD.FTZ.RZ | FP32 add: FTZ + round-toward-zero | Combined modifier |
| FMNMX.FTZ | FP32 min/max with FTZ | |
| FMUL.FTZ.D8 | FP32 multiply: FTZ + D8 scale | |
| FSET.BF.GT.FTZ.AND | Float set (boolean): FTZ | |
| FSETP.GE.FTZ.AND | Float set-predicate: FTZ | |
| FSETP.GT.FTZ.AND | Float set-predicate: FTZ | |
| **MUFU.SQRT** | **Hardware square root SFU path** | **Only emitted with fast_math!** |
| **P2R** | **Predicate-to-register (spill)** | **Proves predicate spilling CAN occur** |

**MUFU.SQRT discovery**: Without `--use_fast_math`, `sqrtf()` compiles to
`MUFU.RSQ` + `FMUL` (reciprocal sqrt * x). Fast math enables a direct
`MUFU.SQRT` instruction. This is a different SFU opcode from `MUFU.RSQ`.

**P2R discovery**: We previously reported that predicate spilling never occurs
(probe_predicate_pressure.cu). This is true at default optimization, but
`--use_fast_math` generates FTZ predicate chains complex enough to trigger
P2R (predicate-to-register spill). The complementary R2P (register-to-predicate
reload) is now also directly observed locally in the transcendental
compile-profile path, even though the byte-bank-qualified `P2R.B*` family and
`P2R.B*` still remain unreproduced on the direct local `sm_89` probe path.
A newer exact same-carrier tranche also directly confirms
`P2R R0, PR, R0, 0x7f`, which closes the broader full-mask same-carrier shape
and leaves the byte-qualified family as the real remaining gap.
An even more literal cuDNN-shaped follow-up under
`src/sass_re/results/runs/p2r_b1_literal_cudnn_20260321_115500` still lowers
through `ISETP` + `SEL` + `LOP3.LUT` glue and does not emit `P2R.B1/B2/B3`,
even under `--maxrregcount=32`.
Two newer literal byte-two and byte-three follow-ups under
`src/sass_re/results/runs/p2r_b23_literal_cudnn_20260321_122400` then extend
the same boundary: neither surfaces `P2R.B2` or `P2R.B3`, and both still
lower through `ISETP` + `SEL` + `LOP3.LUT` glue. That means the remaining
direct-local gap is now evidence-backed across the full byte-qualified
`P2R.B*` family, not just the byte-one form.
A corpus-wide re-scan under
`src/sass_re/results/runs/p2r_boundary_rescan_20260321_124500` then closes the
loop: recursive local raw `.sass` contains `P2R ... 0x7f` and many
`ULOP3.LUT` instances, but still no direct-local raw `P2R.B1`, `P2R.B2`, or
`P2R.B3`.
One final split-seed carrier attempt under
`src/sass_re/results/runs/p2r_b1_split_seed_20260321_125300` then mirrors the
cuDNN-style `0x80` and `0x8000` split-carrier setup more literally. It still
re-confirms `P2R ... 0x7f` in both optimized lanes and does not emit
`P2R.B1`.
The cubin-side phase then moved that boundary from symbolic to semantic.
Under `src/sass_re/results/runs/p2r_cubin_patch_trial_20260322_233700`, the
top local `probe_p2r_two_stage_bank_exact_O3.cubin` site re-disassembles
directly as `P2R.B2` and `P2R.B3`. The first one-pattern runtime check looked
equivalent, but the corrected multi-pattern matrix under
`src/sass_re/results/runs/p2r_cubin_pattern_matrix_20260322_234513` shows the
real boundary: `two_stage` is runtime-stable, yet only patterns `0` and `1`
remain inert, while patterns `2` and `3` diverge. Additional local candidates
under `src/sass_re/results/runs/p2r_cubin_pattern_matrix_20260322_234517`,
`src/sass_re/results/runs/p2r_cubin_pattern_matrix_20260322_234522`, and
`src/sass_re/results/runs/p2r_cubin_pattern_matrix_20260322_234632` establish
the fuller semantic map: some patched contexts are runtime-stable but
deterministically different, and the earlier `byteview` branch remains
runtime-unstable. A final follow-up under
`src/sass_re/results/runs/p2r_cubin_pattern_matrix_20260322_235427` then adds
direct local cubin-side `P2R.B1` on that same top `two_stage` site. So direct
local optimized source/IR still does not emit `P2R.B1/B2/B3`, but local
cubin-side substitution can now materialize the full `P2R.B1/B2/B3` family and
test their semantics in-context.
The final `PLOP3`-fed symbolic strip-mine under
`src/sass_re/results/runs/p2r_plop3_source_20260322_202900`,
`src/sass_re/results/runs/p2r_plop3_samecarrier_20260322_202951`, and
`src/sass_re/results/runs/p2r_plop3_selpack_20260322_203047` then closes the
last meaningful source-space axis. Predicate-source kind really does alter
the optimized neighborhood: these O3 probes emit dense `LOP3.LUT P*` and
`PLOP3.LUT`. But even with same-carrier lifetime and explicit `SEL`-weighted
packing restored, the compiler still does not select `P2R.B1/B2/B3`. It
either keeps plain `P2R ..., RZ, 0x1` in the older tripack kernel or removes
`P2R` entirely and rebuilds the bytes in GPRs with `SEL` and `LOP3`. That is
the strongest local evidence yet that byte-qualified `P2R.B*` selection is
not reachable from ordinary CUDA C++ source shaping on this local toolchain.

### New mnemonics from `--restrict` (5 new)

| Mnemonic | Description | Significance |
|---|---|---|
| I2F.S8 | Direct signed-byte to float | Skips INT32 widening step |
| LDG.E.U16.CONSTANT | 16-bit unsigned read-only load | __restrict__ enables constant cache |
| LDG.E.U8.CONSTANT | 8-bit unsigned read-only load | __restrict__ enables constant cache |
| **LDL.LU** | **Local memory load (uniform hint)** | **Register spill to LMEM** |
| **STL** | **Local memory store** | **Register spill to LMEM** |

**LDL.LU + STL discovery**: `--restrict` changes pointer aliasing, causing
the compiler to allocate more registers for pointer disambiguation. This
pushes some kernels over the register limit, triggering LMEM spills.
LDL.LU (load local, uniform) is the spill reload path; STL is the spill store.
These are the instructions that `check_spills.sh` warns about.

### Combined total (flag sweep only: fast_math + restrict)

**286 unique SASS mnemonics** across fast_math and restrict flag combinations
(268 baseline + 13 from fast_math + 5 from restrict = 286 total).

### Legacy focused flag-matrix slice (20 combos x 60 probes = 1200 compilations)

| Flags | Compiled | Mnemonics | New | Spills | Key discovery |
|---|---|---|---|---|---|
| baseline (resolver-selected std flag, `-std=c++20` locally) | 60 | 268 | -- | 0 | Reference |
| -O1 / -O2 / -O3 | 60 | 268 | 0 | 0 | Identical (asm volatile proof) |
| -fmad=false | 60 | 268 | **+1** | 0 | **HMUL2** (FMA decomposed to MUL+ADD) |
| --use_fast_math | 60 | 259 | **+13** | 0 | MUFU.SQRT, P2R, 11 FTZ variants |
| --maxrregcount=32-255 | 60 | 270 | **+2** | 0 | UISETP.GE.U32.AND, UISETP.GT.AND |
| **-G (debug)** | 60 | 255 | **+76** | **892** | **76 new mnemonics + massive spills** |
| --restrict | 60 | 271 | **+5** | 24 | LDL.LU, STL, constant-cache sub-byte |
| -O3 precision flags | 60 | 268 | +1 | 0 | HMUL2 |
| -O3 --fast_math --restrict | 60 | 260 | +16 | 0 | Combined set |
| -Xptxas -warn-spills | 60 | 268 | 0 | 0 | Warning only, no SASS change |
| -Xptxas -warn-double-usage | 60 | 268 | 0 | 0 | Warning only, no SASS change |

### Debug build (-G) reveals 76 hidden instruction variants

The `-G` (device debug) flag disables optimization entirely, exposing
instructions that ptxas normally optimizes away. 76 new mnemonics appear:

- **ATOMS.CAST.SPIN / .SPIN.64**: Shared-memory spin-lock atomics (debug CAS loop)
- **BMOV.32 / BMOV.32.CLEAR**: Convergence barrier move/clear
- **BMSK**: Bit mask generation instruction (normally folded into LOP3)
- **IMMA.*.SAT**: Saturating tensor MMA variants (IMMA.16816.S8.SAT, IMMA.8832.S4/U4.SAT)
- **LDGSTS.E.*.ZFILL**: Async copy with zero-fill (64-bit, 128-bit bypass)
- **MOVM.16.MT88**: Matrix move via shared memory (tensor core data staging)
- **QSPC.E.G / QSPC.E.S**: Query address space (global / shared)
- **R2UR**: Register to uniform-register (reverse of S2UR)
- **WARPSYNC.EXCLUSIVE**: Exclusive warp synchronization mode
- **MEMBAR.SC.VC**: Sequential consistency fence with virtual channel
- **F2F.BF16.F32 / F2F.F16.F32**: Direct format-to-format conversions
- **76 total new mnemonics** (full list in results/flag_sweep/debug_G/)

### -fmad=false reveals HMUL2

When FMA fusion is disabled, `a*b+c` decomposes to separate multiply + add:
FP16: `HFMA2` -> `HMUL2` + `HADD2` (2 instructions instead of 1).
This is the only way to observe the standalone `HMUL2` instruction.

### --maxrregcount reveals UISETP

Register pressure constraints expose unsigned uniform integer set-predicate
variants (UISETP.GE.U32.AND, UISETP.GT.AND) in the uniform register datapath.

**Grand total: 443 unique SASS mnemonics** across all flag combinations,
262 probe kernels, and 15+ microbenchmarks. See `SM89_INSTRUCTION_CATALOG.md`
for the complete catalog with CUDA/PTX intrinsic mappings.

### CRITICAL: --use_fast_math breaks throughput benchmarks

`--use_fast_math` enables associativity optimizations that allow the
compiler to constant-fold throughput measurement chains:

| Benchmark | Without fast_math | With fast_math | Cause |
|---|---|---|---|
| HFMA2.BF16 throughput | 312.1 ops/clk/SM | **12.9** (24x regression!) | Chain folded to 1 instruction |
| FFMA throughput | 44.6 ops/clk/SM | 85.0 (inflated) | Partial folding |

The `asm volatile` pattern protects **latency** chains but NOT **throughput**
chains (which use C++ operators for 8 independent accumulators). The compiler
can fold `a = a * scale + bias` across iterations when fast_math grants
associativity permission.

**Rule: compare throughput benchmarks compiled WITHOUT --use_fast_math.**
Use `--use_fast_math` only for SASS mnemonic hunting, not measurement.

---

## Definitive Summary (2026-03-20)

| Metric | Value |
|---|---|
| Recursive probe files | 349 |
| Non-skip manifest entries | 344 |
| Compile-enabled probe files in current manifest | 343 |
| Compile-enabled probe files in first full refresh | 341 |
| Compile-enabled probe files in postfix refresh | 343 |
| Microbenchmarks | 15 |
| Canonical optimized mnemonic frontier | **379** |
| Strongest discovery-lane frontier | **382** (`--maxrregcount=32`) |
| Checked-in SM89 catalog rows | **470** |
| Total SASS instructions disassembled | 26,000+ |
| Latency measurements | 70+ (ncu cross-validated) |
| Throughput measurements | 10+ |
| Tensor core precisions measured | 7 (all available on Ada) |
| Exotic formats characterized | 16 (NF4, MX, FXP, Posit, BCD, DD, QD, bignum) |
| Novel discoveries | 30+ |
| `ncu` hardware counter validations | 331 profiled in postfix run, 3 failed, 14 skipped |
| Production kernels created | 4 (BF16 bf162, INT8 MRT A-A, inv_tau, REDUX box count) |
| Zero-cost modifiers confirmed | 5 (FTZ, SAT, FABS, FNEG, denormals) |
| Remaining unreproduced direct-local source/IR cluster | 4 mnemonics (`P2R.B1/B2/B3`, `UPLOP3.LUT`) |

### Fastest instruction per pipeline (Ada Lovelace SM 8.9)

| Pipeline | Fastest instruction | Latency |
|---|---|---|
| Integer | IABS (modifier) | 0.26 cy/pair |
| Integer standalone | IADD3 | 2.52 cy |
| FP32 | FFMA | 4.53 cy |
| FP16 packed | HADD2 / HFMA2 | 4.54 cy |
| **BF16 packed** | **HFMA2.BF16** | **4.01 cy** |
| INT8 dot product | IDP.4A | 4.53 cy |
| Tensor core | IMMA.8832 (INT4) | 28.05 cy |
| Tensor core float | HMMA.16816 (FP16->FP16) | 42.14 cy |
| Warp reduction | REDUX.SUM | 60.01 cy |
| SFU | MUFU.EX2 | 17.55 cy |
| FP64 | DADD | 48.47 cy |
| Shared memory | LDS | 28.03 cy |

### Most expensive operations (Ada Lovelace SM 8.9)

| Operation | Latency | Category |
|---|---|---|
| NANOSLEEP(0) | 2685.55 cy | Warp reschedule |
| MEMBAR.SYS | 2583.37 cy | System-scope fence |
| FP64 log() | 1113.83 cy | Transcendental |
| FP64 sin() | 820.66 cy | Transcendental |
| QD FP256 ADD | 828.28 cy | Arbitrary precision |
| DD FP128 ADD | 500.28 cy | Double-double |
| FP64 sqrt() | 450.30 cy | Transcendental |
| LDGSTS | 363.28 cy | Async copy overhead |
| DD FP128 MUL | 314.35 cy | Double-double |
| BCD ADD | 291.05 cy | Decimal arithmetic |
