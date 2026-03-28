# SM89 SASS RE Knowledge Synthesis

This document is the current single-page synthesis of what the repo has
demonstrated, what it has ruled out, what only works cubin-side, and what the
live frontiers actually are now.

It integrates the recursive corpus refreshes, the `P2R.B*` RCA work, the
`UPLOP3.LUT` cubin-side semantic validation work, the runtime-safe `ncu`/`nsys`
surrogate work, and the tandem differential-fuzzing workflow.

## 1. Inventory Snapshot

Current inventory state:

- canonical optimized frontier: `379` raw mnemonics
- strongest discovery-lane frontier: `382` raw mnemonics under
  `--maxrregcount=32`
- checked-in SM89 catalog rows: `470`

Primary references:

- [SM89_INSTRUCTION_CATALOG.md](SM89_INSTRUCTION_CATALOG.md)
- [SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md](SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md)

Important consequence:

- the frontier is no longer broad mnemonic discovery
- it is now a narrow form-selection and semantic-validation problem centered on
  a few unreproduced direct source/IR spellings

## 2. What Has Been Fully Closed

### 2.1 Async/cache and SYS64 combo families

The repo has already closed a very large direct-local emitted-neighborhood
frontier around:

- `LDG(.STRONG.GPU/.STRONG.SYS)`
- `LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)`
- `LDGDEPBAR`
- `DEPBAR.LE`
- `MATCH.ANY`
- `REDUX.MIN/MAX/SUM`
- `VOTE.ALL/ANY/VOTEU.ANY`
- `BAR.RED.*`
- `B2R.RESULT`
- dense `ATOMG.E.*.64.STRONG.SYS`
- `MEMBAR.SC.SYS`
- `ERRBAR`
- `CCTL.IVALL`

This is no longer a speculative family. It exists in both symbolic and
runtime-safe surrogate form and is profiled with `ncu`.

Interpretation:

- the async/cache neighborhood is locally reproducible
- the main remaining questions in that area are semantic/perf-side, not opcode
  discovery

### 2.2 Direct local cubin-side `P2R.B1/B2/B3`

Direct source/IR still does not emit `P2R.B1/B2/B3`, but local cubin-side
substitution now materializes and runs all three:

- `P2R.B1`
- `P2R.B2`
- `P2R.B3`

Interpretation:

- the opcodes themselves are not impossible on local `sm_89`
- what remains unreproduced is source/IR-driven form selection, not the raw
  existence of the encodings

## 3. What Has Been Exhaustively Ruled Out For `P2R.B*`

The `P2R` target was very specific:

- direct local optimized `sm_89`
- emitted from local source or local frontend/IR experiments
- containing `P2R.B1`, `P2R.B2`, or `P2R.B3`
- not just library evidence
- not plain `P2R`
- not debug-only neighbors

### 3.1 Source-space mutations that do not unlock `P2R.B*`

Exhausted families:

- same-carrier full-mask `0x7f`
- split-seed carriers
- regmask rewrites
- dualpack transitions
- second-bank halfword staging
- staged tripack prefix lifetimes
- nibble-sized `0x0f` paths
- dense `PLOP3`-fed and `LOP3.LUT P*`-fed variants
- explicit `SEL`-weighted pack variants
- optimization pressure changes such as `--maxrregcount=32`

What these proved:

- local source can emit plain `P2R ... 0x7f`
- local source can emit plain `P2R ... 0x0f`
- local source can recreate much of the surrounding predicate neighborhood
- but local source still does not select byte-qualified `P2R.B1/B2/B3`

### 3.2 PTX, frontend, and version sweeps that do not unlock `P2R.B*`

Tried and normalized:

- direct PTX search
- clang CUDA -> PTX -> local `ptxas`
- Triton -> PTX -> cubin
- side-by-side `ptxas 11.8`, `12.6`, and `13.1`

What these proved:

- simple direct PTX is version-insensitive across the tested `ptxas` range
- clang changes the IR basin, but still not the last form-selection step
- Triton changes the basin more strongly, but still does not surface `P2R.B*`
- the missing rule is not explained by a simple frontend or `ptxas` version
  swap

### 3.3 Library mining did not show a broader localizable ecosystem

The richest `P2R.B*` pockets remain dominated by cuDNN precompiled engines.

What was learned from library mining:

- cuDNN precompiled engines are the real source of the strongest `P2R.B*`
  windows
- cuDNN runtime-compiled engines contributed none in the tested sweep
- cuBLAS contributed none in the tested sweep
- TensorRT builder resources were not directly `cuobjdump`-readable as normal
  device-code windows

Interpretation:

- `P2R.B*` is not a broadly exposed user-space pattern in the installed stack
- it behaves more like a library/compiler-internal lowering path

## 4. Current `P2R` Conclusion

The `P2R` search changed from a discovery problem into a control problem.

What is proven:

- local source/IR can reproduce the neighborhood
- local source/IR can reproduce plain `P2R` pack widths like `0x7f` and `0x0f`
- cubin-side substitution can materialize runnable `P2R.B1/B2/B3`

What is disproven:

- simple carrier-shape mutation is enough
- simple pack-width mutation is enough
- simple predicate-source-kind mutation is enough
- simple PTX/clang/Triton/version sweep is enough

Current open question:

- which compiler-internal or deeper IR condition selects byte-qualified
  `P2R.B*` instead of plain `P2R` plus glue

So the remaining `P2R` frontier is no longer "get the opcode locally at all."
It is:

- recover the source/IR-level form-selection rule, or
- accept cubin-side semantic validation as the reproducible proof layer

## 5. `UPLOP3.LUT`: What Changed

`UPLOP3.LUT` is now the stronger remaining non-cubin frontier.

Direct source/IR still does not emit `UPLOP3.LUT`, but the repo has now shown:

- library-side `UPLOP3` motifs can be mined and normalized
- local `PLOP3 -> UPLOP3` cubin substitution is structurally valid
- local `ULOP3 -> UPLOP3` substitution is structurally invalid
- multiple local `UPLOP3` sites are runtime-safe
- some are inert, and some are semantically live

This is a deeper result than the original `P2R` story because it crosses all
the way from disassembly to validated runtime semantic classes.

## 6. `UPLOP3` Library Census and Motif Read

Current mined library census:

- `UPLOP3.LUT`: `26` windows
- `ULOP3.LUT`: `358` windows
- `PLOP3.LUT`: `1950` windows

Dominant motif families:

- HMMA-wrapped tensor-core blocks
- integer/address-shaping blocks around `IMAD.WIDE`, `SHF`, `LEA`, `IADD3`
- mixed async/tensor neighborhoods

Interpretation:

- `UPLOP3` is not a weird one-off uniform datapath artifact
- it appears in real tensor and address-shaping motifs
- the local patch work therefore needs motif-sensitive substrates, not just any
  `ULOP3` or `PLOP3` site

## 7. `UPLOP3` Structural Boundary

Two patch rules now define the structural boundary:

### 7.1 `ULOP3 -> UPLOP3`

Result:

- illegal under disassembly

Interpretation:

- `UPLOP3` is not just a trivial `UR`-form opcode swap from `ULOP3`

### 7.2 `PLOP3 -> UPLOP3`

Result:

- decodes cleanly as real `UPLOP3.LUT`
- launches cleanly in multiple local contexts

Interpretation:

- `PLOP3` is the correct local cubin substrate

This was the crucial structural breakthrough for the second major frontier.

## 8. `UPLOP3` Runtime Classes

The runtime matrix now separates local `UPLOP3` substitutions into two classes:

### 8.1 Inert contexts

These decode and run, but match baseline on the tested semantic patterns:

- `probe_uniform_loop`
- `probe_smem_sliding_window`
- `probe_hmma_tiled_loop`
- many mixed `0xa8` / `0x2a` sliding-window cases
- several weaker single-site cases

### 8.2 Stable-but-different contexts

These decode and run, and produce semantic deltas:

- cooperative-groups `0x8` site
- several `probe_uplop3_uniform_predicates` sites
- several `probe_cutlass_predicate_pipeline` sites

Interpretation:

- local `UPLOP3` is not merely syntactic or disassembly-only
- it can be semantically active in runnable code
- liveness is highly site-specific

## 9. `UPLOP3` Live-Site Hierarchy

Current best local live sites by tandem evidence:

1. `uniform_occ1`
2. `cutlass_occ5`
3. `uniform_occ2`
4. `cutlass_occ4`
5. `uniform_occ5`
6. `cooperative_occ1`

Important refinement:

- `uniform_occ5` is live, but behaves more like a sensitizer than a primary
  anchor
- `cutlass_occ4` is live, but behaves more like an amplifier/modifier than the
  root cause of the broadest semantic explosions

## 10. Pair-Baseline Interpretation

The pair-baseline tranche sharpened the causal story:

- uniform pair baseline: `occ2_occ5`
- CUTLASS pair baseline: `occ2_occ5`

What it proved:

- on the uniform branch, `occ1` is still the true extra widener
- the full triple `occ1_occ2_occ5` is actually less disruptive than `occ1` or
  `occ1_occ5` alone, so the triple partially re-stabilizes the pair baseline
- on the CUTLASS branch, `occ4` is the strongest visible widener against the
  stable pair baseline
- richer CUTLASS combos often preserve the visible output prefix and instead
  perturb aggregate sums, so they behave more like aggregate-state wideners
  than prefix-level wideners

This matters because it tells us which sites are true anchors and which ones
mainly act through interaction with another live site.

## 11. Tool Effectiveness Synthesis

The repo now has enough evidence to rank the tools themselves.

### 11.1 Strongest semantic discriminator

- differential fuzzing

Why:

- it cleanly separates stable anchors from broad semantic wideners
- it exposed the pair-baseline re-stabilization effect
- it distinguishes visible output-prefix changes from broader aggregate-sum
  changes

### 11.2 Strongest safety gate

- `compute-sanitizer`

Why:

- live patched cubins can be validated as free of obvious memory errors
- semantic differences are therefore not explained away as invalid memory
  behavior

Caveat:

- sanitizer perturbs aggregate sums, so it is a safety check, not the semantic
  oracle

### 11.3 Best perf sanity tool

- `ncu`

Why:

- it is good at ruling out hidden regime shifts
- it showed the CUTLASS live cases keep the same basic execution footprint even
  when semantics change

Caveat:

- it is not the main semantic discovery tool for the current frontier

### 11.4 Trace sidecar

- `nsys`

Why:

- it is worth collecting for timeline provenance and future driver/context
  analysis

Caveat:

- it has been lower-yield than fuzzing and sanitizer for the current opcode
  frontier

## 12. What Is Proven, Disproven, and Open

### Proven

- direct local async/cache and SYS64 combo neighborhoods are real and profiled
- local cubin-side `P2R.B1/B2/B3` materialization is real and runnable
- local cubin-side `PLOP3 -> UPLOP3` is structurally valid
- local `UPLOP3` can be runtime-safe and semantically live
- differential fuzzing is the best semantic ranking tool for the current work

### Disproven

- broad CUDA-C++ source mutation alone is enough to surface `P2R.B*`
- simple PTX, clang, Triton, or `ptxas` version sweeps are enough to surface
  `P2R.B*`
- `ULOP3 -> UPLOP3` is the correct structural local patch path
- raw perf counters alone can explain the semantic liveness hierarchy

### Open

- the source/IR-level form-selection rule for `P2R.B1/B2/B3`
- whether an older sysroot/containerized older distro would unlock meaningful
  older-`nvcc -G` behavior
- which additional `PLOP3` sites can be promoted into the live `UPLOP3`
  hierarchy
- whether the next live `UPLOP3` sites better match HMMA-wrapped library motifs
  or address-shaping motifs

## 13. Current Frontier Ranking

Current highest-yield frontiers:

1. `UPLOP3.LUT` local live-site widening from the best current anchors:
   - `uniform_occ1`
   - `cutlass_occ5`
   - `uniform_occ2`
2. broader `PLOP3` motif mining and local site ranking against mined library
   `UPLOP3` windows
3. deeper `P2R.B*` work only if the attack layer changes enough to be
   meaningfully new:
   - older sysroot/container
   - deeper IR lowering
   - cubin-side semantics

Current lower-priority lanes:

- more generic CUDA source mutation for `P2R.B*`
- more generic PTX mutagenesis without a new structural hypothesis
- treating `uniform_occ5` or `cutlass_occ4` as standalone top anchors

## 14. Bottom Line

The repo is no longer in an early discovery phase.

For `P2R`:

- source/IR-level direct emission is effectively exhausted on the current local
  toolchain stack
- cubin-side validation has already proven the byte-qualified opcodes are real
  and runnable locally

For `UPLOP3`:

- direct source/IR still does not emit the mnemonic
- but the frontier is active and productive because local cubin-side
  `PLOP3 -> UPLOP3` has already crossed from structural validity into
  multi-context runtime semantics

That means the current work is not "can we find something at all?"
It is:

- identify which local patched sites are true semantic anchors
- identify which ones are modifiers or sensitizers
- keep ranking those against real library motif families
- only revisit broader `P2R` work when the attack layer changes enough to be
  meaningfully new
