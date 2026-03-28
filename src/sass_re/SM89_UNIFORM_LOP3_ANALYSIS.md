# UPLOP3 Frontier Analysis

## Current State

The remaining non-cubin frontier after `P2R.B1/B2/B3` is `UPLOP3.LUT`.

Direct local source / IR still does not emit `UPLOP3.LUT`, but the library-side
motif space is now normalized enough to support cubin-side RCA and semantic
validation. The local boundary is now richer than a pure decode check: we have
multiple runnable `PLOP3 -> UPLOP3` contexts, and the first stable-but-
different semantic class has appeared in cooperative-groups code.

## Library Census

From the cached cuDNN SM86 dumps in
`results/runs/cudnn_library_sm86_mining_20260320_103900` and the mined window
set in
`results/runs/library_uplop3_text_windows_20260323_000500/summary.txt`:

- `UPLOP3.LUT`: 26 windows
- `ULOP3.LUT`: 358 windows
- `PLOP3.LUT`: 1950 windows

The strongest `UPLOP3` windows come from precompiled cuDNN engines. The current
runtime-compiled cuDNN dump contributed no `UPLOP3.LUT`.

## Dominant Motifs

From
`results/runs/library_uplop3_text_window_summary_20260323_000500/summary.txt`:

- tensor-core heavy:
  `HMMA.16816.F32 -> HMMA.16816.F32 -> HMMA.16816.F32 -> UPLOP3.LUT -> HMMA.16816.F32 -> HMMA.16816.F32 -> HMMA.16816.F32`
- integer/address shaping:
  `LOP3.LUT -> IMAD.WIDE -> SHF.R/U32 or SHF.L/U32 -> UPLOP3.LUT -> LEA.HI / IADD3 -> IMAD.WIDE -> ...`
- mixed async/tensor neighborhood:
  `LOP3.LUT -> HMMA.16816.F32 -> LDGSTS.E.BYPASS.LTC128B.128 -> UPLOP3.LUT -> SHF.R.U32.HI -> HMMA.16816.F32 -> LDGSTS...`

This means `UPLOP3` is not just a uniform-datapath curiosity. It lives in both
HMMA-wrapped and address-shaping neighborhoods.

## Local Anchors

The best direct local `ULOP3` anchors remain:

- `results/precision_build/probe_async_copy.sass`
- `results/precision_build/probe_warp_matrix_shared.sass`

These are runtime-safe and compact, but their overlap with the strongest library
`UPLOP3` windows is weak. The current patch ranking in
`results/runs/uplop3_patchpoint_rankings_20260323_000900/summary.txt` tops out
around Jaccard `0.286`.

## Encoding / Patch Findings

Two cubin-side substitution experiments now define the boundary:

1. `ULOP3 -> UPLOP3`
   - patched local `probe_async_copy` cubin in
     `results/runs/uplop3_patch_sketch_20260323_001400`
   - result: patched cubin is rejected by `nvdisasm` as an illegal instruction
   - interpretation: `UPLOP3` is not a trivial opcode-nibble swap on the `UR`
     operand `ULOP3` form

2. `PLOP3 -> UPLOP3`
   - patched local `probe_uniform_datapath` cubin in
     `results/runs/uplop3_patch_sketch_20260323_001900`
   - result: both `0x80` and `0x8` predicate forms re-disassemble cleanly as:
     - `UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x80, 0x0`
     - `UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x8, 0x0`
   - interpretation: the structurally correct direct-local patch target is
     `PLOP3`, not `ULOP3`

## Runtime Status

A guarded runtime matrix was added via a dedicated cubin driver runner
and a shell script that validates the UPLOP3 cubin pattern matrix (both
provided by the probe domain PRs).

The first runtime attempt at
`results/runs/uplop3_cubin_pattern_matrix_20260323_002300/summary.txt` was
blocked before launch by a transient primary-context
`CUDA_ERROR_OUT_OF_MEMORY`. That issue cleared on retry.

Two guarded runtime matrices now complete cleanly:

- `results/runs/uplop3_cubin_pattern_matrix_20260323_003500/summary.txt`
- `results/runs/uplop3_sliding_window_matrix_20260323_005300/summary.txt`

Results so far:

- `probe_uniform_loop`
  - `PLOP3(0x80) -> UPLOP3(0x80)` is runtime-inert on patterns `0..3`
  - `PLOP3(0x8) -> UPLOP3(0x8)` is runtime-inert on patterns `0..3`
- `probe_smem_sliding_window`
  - `PLOP3(0x80) -> UPLOP3(0x80)` is runtime-inert on patterns `0..3`
  - `PLOP3(0x8) -> UPLOP3(0x8)` is runtime-inert on patterns `0..3`
- `probe_cg_coalesced`
  - `PLOP3(0x80) -> UPLOP3(0x80)` is runtime-inert on patterns `0..3`
  - `PLOP3(0x8) -> UPLOP3(0x8)` is stable-but-different:
    different on patterns `0/2/3`, same on pattern `1`
  - the occurrence sweep in
    `results/runs/uplop3_cooperative_occ_sweep_20260323_081500/summary.txt`
    now shows that this is also site-specific inside the same kernel:
    `occ1(0x8)` is the only live site, `occ2(0x80)` is inert, and the double
    patch `occ1_occ2` collapses to the exact same behavior as `occ1` alone
- `probe_hmma_tiled_loop`
  - `PLOP3(0x80) -> UPLOP3(0x80)` is runtime-inert on patterns `0..3`
  - `PLOP3(0x8) -> UPLOP3(0x8)` is runtime-inert on patterns `0..3`
- `probe_uplop3_uniform_predicates`
  - `PLOP3(0x80) -> UPLOP3(0x80)` is stable-but-different:
    different on patterns `1/2`, same on `0/3`
  - mixed-predicate `PLOP3(0xa8) -> UPLOP3(0xa8)` decodes cleanly as
    `UPLOP3.LUT UP3, UPT, UP0, UP3, UPT, 0xa8, 0x0` and is runtime-inert
  - mixed-predicate `PLOP3(0xfe) -> UPLOP3(0xfe)` decodes cleanly as
    `UPLOP3.LUT UP4, UPT, UP4, UP1, UP2, 0xfe, 0x0` and is stable-but-different:
    different on patterns `0/2`, same on `1/3`
  - the occurrence sweep in
    `results/runs/uplop3_uniform_predicates_occ_sweep_20260323_030000/summary.txt`
    now shows that semantic liveness is site-specific inside the same kernel:
    `occ1(0xfe)`, `occ2(0xfe)`, and `occ5(0x80)` are live, while `occ3(0xfe)`
    and `occ4(0xa8)` are inert
  - the combo sweep in
    `results/runs/uplop3_uniform_predicates_combo_sweep_20260323_031500/summary.txt`
    and
    `results/runs/uplop3_uniform_predicates_combo_analysis_20260323_032200/summary.txt`
    shows nontrivial composition rather than simple additivity:
    `occ1` dominates pattern `0`, `occ5` dominates pattern `1`, `occ2` controls
    the pattern `3` divergence family, and pattern `2` splits into two distinct
    live outputs depending on whether `occ1` or `occ2` is paired with `occ5`
- `probe_smem_sliding_window` mixed cluster
  - `PLOP3(0xa8) -> UPLOP3(0xa8)` at both local sites is runtime-inert
  - `PLOP3(0x2a) -> UPLOP3(0x2a)` at both local sites is runtime-inert

The normalized runtime-class matrix now lives at
`results/runs/uplop3_runtime_class_matrix_20260323_024000/summary.txt`:

- inert:
  - `probe_uniform_loop` (`0x80`, `0x8`)
  - `probe_smem_sliding_window` PT-only (`0x80`, `0x8`)
  - `probe_smem_sliding_window` mixed (`0xa8`, `0x2a`, both local sites)
  - `probe_hmma_tiled_loop` (`0x80`, `0x8`)
  - `probe_cg_coalesced` (`0x80`)
  - `probe_uplop3_uniform_predicates` (`0xa8`)
- stable-but-different:
  - `probe_cg_coalesced` (`0x8`)
  - `probe_uplop3_uniform_predicates` (`0x80`, `0xfe`)

That is materially stronger than the current `P2R.B*` cubin-side result: most
validated `UPLOP3` contexts are inert, but the cooperative-groups `0x8` site
and the uniform-predicate `0x80` / `0xfe` sites show that `UPLOP3` can also be
semantically live in local runnable code.

The live-context motif ranking at
`results/runs/live_uplop3_context_rankings_20260323_024500/summary.txt`
now suggests the strongest library-side resemblance comes from the denser
uniform-predicate `0xfe` case (`best_jaccard=0.375000`), followed by the
cooperative `0x8` and uniform-predicate `0x80` sites (`0.300000` each).

The local composition law is now much sharper in both live kernels:

- `probe_cg_coalesced`:
  only the first `0x8` site is live; the second `0x80` site is inert; and the
  pair behaves exactly like the first site alone
- `probe_uplop3_uniform_predicates`:
  liveness is split across `occ1(0xfe)`, `occ2(0xfe)`, and `occ5(0x80)`, with
  nontrivial composition rather than a single dominating effect across every
  pattern
- `probe_cutlass_predicate_pipeline`:
  `occ2(0xa8)`, `occ4(0x80)`, and `occ5(0xa8)` are all semantically live, but
  they do not compose in a trivial overwrite order:
  `occ2_occ4` collapses to the `occ2` family on patterns `0/2/3` and to the
  `occ4` family on pattern `1`, while `occ4_occ5` and `occ2_occ4_occ5` share
  the `occ5`-like visible output prefix on patterns `0/2/3` but still diverge
  in total sum, especially on patterns `1/2/3`

The exact live-occurrence ranking is now:

- `uniform_occ1`: `best_jaccard = 0.375`
- `cutlass_occ5`: `best_jaccard = 0.375`
- `cutlass_occ4`: `best_jaccard = 0.333333`
- `uniform_occ2`: `best_jaccard = 0.300`
- `uniform_occ5`: `best_jaccard = 0.300`
- `cooperative_occ1`: `best_jaccard = 0.300`

Repo hygiene also improved along the way:

- PMD proper on the supported HTML surface is now clean after reducing
  inline styles in `src/editor/ysu_scene_editor.html`
- PMD CPD on the cubin runner/script tranche is clean after factoring shared
  driver boilerplate into `runners/cubin_driver_common.h`

The new mixed-predicate decode result also sharpens the structural boundary:

- `ULOP3 -> UPLOP3`: still structurally invalid
- PT-only `PLOP3 -> UPLOP3`: structurally valid and runtime-inert in two local contexts
- mixed-predicate `PLOP3 -> UPLOP3`: structurally valid in at least one denser
  local context, with multiple live local sites now validated in both the
  uniform-predicate and CUTLASS-like families

Profiling and trace guidance is also clearer now:

- `ncu` does help, but only as a secondary signal on this frontier:
  for the CUTLASS pattern-2 comparison, baseline, `occ4`, `occ5`, and
  `occ2_occ4_occ5` all keep the same executed instruction count (`381`), the
  same register footprint (`34`), and nearly the same stall mix
  (`long_scoreboard` stays in the `34-36%` band), while cycles move only
  modestly (`6304.60` baseline, `6309.60` for `occ4`, `6457.80` for `occ5`,
  `6321.60` for `occ2_occ4_occ5`)
- that means `ncu` is useful for ruling out a hidden performance-regime shift,
  but not for discovering semantic liveness by itself
- `compute-sanitizer` is the stronger sidecar tool here:
  both the CUTLASS baseline and the richest live combo `occ2_occ4_occ5` pass
  memcheck with `0 errors` and `0 bytes leaked`, so the semantic differences
  are not just obvious invalid memory behavior
- however, `compute-sanitizer` perturbs aggregate sums, so it should be used as
  a safety gate, not as the semantic oracle

The default workflow is now tandem rather than ad hoc:

- semantic anchor or baseline run
- `ncu` metrics + stall pass
- `compute-sanitizer` memcheck
- `nsys` timeline capture
- seeded differential fuzzing

The reusable scripts are:

- `scripts/validate_uplop3_tandem_site.sh`
- `scripts/uplop3_diff_fuzz.py`
- `scripts/summarize_uplop3_tandem.py`

The first tandem anchor runs are:

- `results/runs/uplop3_uniform_tandem_20260323_092500`
- `results/runs/uplop3_cutlass_tandem_20260323_092500`

These use the strongest live local sites as anchors:

- uniform anchor: `uniform_occ1`
- CUTLASS anchor: `cutlass_occ5`

Seeded differential fuzzing is currently the strongest semantic discriminator:

- uniform branch:
  `occ1_occ5` stays close to the `uniform_occ1` anchor
  (`21/24` seeded cases unchanged, `3/24` with diffs),
  while control `occ4` diverges much more broadly
  (`7/24` unchanged, `17/24` with diffs)
- CUTLASS branch:
  `occ2_occ4_occ5` is the broadest semantic widener so far
  (`6/24` unchanged, `18/24` with diffs),
  while `occ4` is intermediate
  (`11/24` unchanged, `13/24` with diffs)

That means differential fuzzing now separates stable anchors from broadening
variants better than raw perf counters do.

The next tandem tranche sharpened the second-tier sites too:

- `results/runs/uplop3_uniform_occ2_tandem_20260323_094000`
- `results/runs/uplop3_cutlass_occ4_tandem_20260323_094000`

Those runs show:

- `uniform_occ2` behaves more like a stable secondary anchor than a broad
  widener:
  `occ2_occ5` stays very close to the `uniform_occ2` anchor
  (`21/24` seeded cases unchanged, `3/24` with diffs),
  while `occ1_occ2` is broader but still mixed
  (`13/24` unchanged, `11/24` with diffs)
- `cutlass_occ4` behaves more like a modifier than an anchor:
  `occ2_occ4` is only moderately broader
  (`16/24` unchanged, `8/24` with diffs),
  but `occ4_occ5` is highly expansive
  (`1/24` unchanged, `23/24` with diffs)

The richer anchor tranche then tightened the top of the ranking:

- `results/runs/uplop3_uniform_occ1_rich_tandem_20260323_095500`
- `results/runs/uplop3_uniform_occ2_rich_tandem_20260323_095500`
- `results/runs/uplop3_cutlass_occ5_rich_tandem_20260323_095500`

Those runs show:

- `uniform_occ1` is still a real anchor, but the triple combo
  `occ1_occ2_occ5` broadens it more than the pair `occ1_occ2`
  (`10/24` unchanged vs `13/24` unchanged)
- `uniform_occ2` stays surprisingly stable when paired with `occ5`:
  `occ2_occ5` remains close to the anchor
  (`21/24` unchanged, `3/24` with diffs),
  while `occ1_occ2_occ5` broadens it
  (`13/24` unchanged, `11/24` with diffs)
- `cutlass_occ5` remains the strongest CUTLASS-like anchor:
  `occ2_occ5` stays close to it
  (`21/24` unchanged, `3/24` with diffs),
  `occ4_occ5` is moderately broader
  (`11/24` unchanged, `13/24` with diffs),
  and `occ2_occ4_occ5` is still the broadest CUTLASS widener
  (`6/24` unchanged, `18/24` with diffs)

The next partner-centered tranche then clarified the causal roles:

- `results/runs/uplop3_uniform_occ5_tandem_20260323_101500`
- `results/runs/uplop3_cutlass_occ2_tandem_20260323_101500`

Those runs show:

- uniform side:
  `occ5` is not a strong anchor by itself.
  From the `occ5` viewpoint, both `occ1_occ5` and `occ2_occ5` are only
  moderately stable (`13/24` unchanged each), while the triple
  `occ1_occ2_occ5` broadens further (`7/24` unchanged, `17/24` with diffs).
  So on the uniform branch, `occ5` behaves more like a sensitizer/mediator
  than a primary anchor.
- CUTLASS side:
  `occ2` is not a stable anchor.
  From the `occ2` viewpoint, `occ2_occ5` diverges on all `24/24` seeded cases,
  `occ2_occ4` is only moderately broad (`9/24` unchanged), and
  `occ2_occ4_occ5` also diverges on all `24/24` seeded cases.
  So the main trigger of the CUTLASS semantic explosion is `occ5`, while
  `occ4` acts more like an amplifier than the root cause.

The next pair-baseline tranche then asked the sharper causal question:

- `results/runs/uplop3_uniform_pair_baseline_20260323_094600`
- `results/runs/uplop3_cutlass_pair_baseline_20260323_094718`
- `results/runs/uplop3_uniform_pair_baseline_20260323_094600__uplop3_cutlass_pair_baseline_20260323_094718__pair_summary.txt`

Those runs treat the stable pair `occ2_occ5` as the baseline state rather than
the original unpatched kernel.

- uniform pair-baseline:
  `occ1` is still the extra live ingredient, but the triple
  `occ1_occ2_occ5` is actually less disruptive than `occ1` or `occ1_occ5`
  alone (`13/24` unchanged vs `7/24`). So on the uniform branch, `occ1`
  remains the true widener, while the full triple partially re-stabilizes the
  pair baseline.
- CUTLASS pair-baseline:
  `occ4` is the strongest visible widener (`0/24` unchanged), while
  `occ4_occ5` and `occ2_occ4_occ5` often preserve the visible output prefix
  and instead change aggregate sums (`8/24` and `9/24` unchanged,
  respectively). So on the CUTLASS branch, `occ4` is the clean visible
  widener, while the richer combos act more like aggregate-state wideners than
  prefix-level wideners.

The tool split stayed consistent in this tranche too:

- `compute-sanitizer`: clean across the tested pair-baseline cases
- `ncu`: useful as a secondary regime check, but not the main semantic signal
- `nsys`: preserved in the run dirs for timeline inspection
- differential fuzzing: still the strongest semantic discriminator

So the current live-site ranking by tandem evidence is:

1. `uniform_occ1` - strongest exact library match and relatively stable anchor
2. `cutlass_occ5` - strongest CUTLASS-like anchor and broadens strongly when combined
3. `uniform_occ2` - credible secondary anchor
4. `cutlass_occ4` - semantically live, but primarily an amplifier/modifier
5. `uniform_occ5` - semantically live, but behaves more like a sensitizer than
   an anchor
6. `cooperative_occ1` - still live, but now a lower-priority widening lane
   widening lane

## Best Next Move

The next highest-yield tranche is now:

1. continue expanding from the now-validated `PLOP3` substrates into the next
   mixed-predicate or async/tensor families, rather than revisiting `ULOP3`
2. widen first from `uniform_occ1`, `cutlass_occ5`, and `uniform_occ2`
3. treat `uniform_occ5` as a sensitizer lane and `cutlass_occ4` as an
   amplifier lane rather than standalone anchors
4. use the stable pair anchors `uniform_occ2+occ5` and `cutlass_occ5+occ2`
   as reference points, and compare all future broad wideners against:
   - uniform `occ1` as the main pair-baseline widener
   - CUTLASS `occ4` as the main visible pair-baseline widener
   - CUTLASS `occ2_occ4_occ5` as the richer aggregate-state explosion family
5. keep using `ncu` as a weak perf-side sanity check and `compute-sanitizer`
   as the stronger safety-side gate when a new local `UPLOP3` site turns live
6. keep differential fuzzing in the default loop, because it is currently the
   strongest ranking signal for semantic broadening
7. keep treating `ULOP3` anchors as secondary, since `ULOP3 -> UPLOP3` still
   decodes as illegal while `PLOP3 -> UPLOP3` is now structurally validated in
   PT-only, mixed-predicate, and CUTLASS-like forms
