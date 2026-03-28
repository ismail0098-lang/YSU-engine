# SM89 Frontier Monograph Synthesis

This document is the repository's current monograph-grade synthesis of the
SM89 reverse-engineering frontier. It does not attempt to pretend that the full
Pascal-vs-Ada comparison is already closed. Instead, it consolidates the part
of the story that the repository presently proves, derives, falsifies, and
operationalizes.

The synthesis is built from the current claims ledger, generated tables, live
runtime-class artifacts, pair-baseline summaries, and the processed data archive
under `processed/monograph_20260323/`.

Primary sources:

- [SM89_ARCHITECTURAL_FINDINGS.md](SM89_ARCHITECTURAL_FINDINGS.md)
- [PAPER_CLAIMS_MATRIX.md](PAPER_CLAIMS_MATRIX.md)
- [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- [MONOGRAPH_BUILD.md](MONOGRAPH_BUILD.md)
- [MONOGRAPH_THEOREM_APPENDIX.md](MONOGRAPH_THEOREM_APPENDIX.md)
- [SM89_PREDICATE_REGISTER_ANALYSIS.md](SM89_PREDICATE_REGISTER_ANALYSIS.md)
- [SM89_UNIFORM_LOP3_ANALYSIS.md](SM89_UNIFORM_LOP3_ANALYSIS.md)

## 1. Problem Statement

The modern SM89 frontier is no longer a broad search over the whole mnemonic
space. The wide inventory question has mostly closed. The remaining hard
questions are narrower and more mathematical:

1. When does a local source or IR graph induce a specific low-level opcode form
   rather than a semantically equivalent neighborhood?
2. When a cubin-side substitution is structurally valid, when is it semantically
   inert, and when is it semantically live?
3. Which observed failures are existence failures, and which are selection
   failures?

These questions are instantiated in two main frontiers:

- `P2R.B1/B2/B3`
- `UPLOP3.LUT`

The central methodological theme is a decomposition into three layers:

- existence: does the opcode exist and run locally?
- selection: does direct source/IR induce that opcode?
- semantics: when the opcode is present, what behavioral class does it belong
  to?

## 2. Prerequisites, Definitions, And Axioms

### 2.1 Primitive objects

Let `S` denote the space of local source programs, `I` the space of intermediate
representations, `B` the space of cubin-level binaries, and `D` the space of
disassembled SASS neighborhoods. Let `R` be the space of observable runtime
behaviors over a chosen pattern family.

We define four maps:

1. `C : S -> I`, the frontend lowering map.
2. `A : I -> B`, the assembler and packer map.
3. `N : B -> D`, the disassembly neighborhood extractor.
4. `E : B x P -> R`, the runtime execution map under pattern family `P`.

The reverse-engineering task is not to invert these maps exactly. It is to
characterize the fibers and failure modes induced by them.

### 2.2 Frontier definition

For a target opcode family `T`, define the direct-local frontier indicator

`F_T(s) = 1` if and only if `T` appears in `N(A(C(s)))`.

Define the cubin-side validity indicator

`G_T(b) = 1` if and only if a patched binary `b_T` both disassembles with `T`
and executes without immediate structural failure.

Define the semantic deviation indicator under a pattern family `P`

`H_T(b, P) = 1` if and only if `E(b_T, P) != E(b, P)`.

These three indicators separate the questions:

- `F_T = 1` answers source/IR selection.
- `G_T = 1` answers local opcode validity.
- `H_T = 1` answers semantic liveness.

### 2.3 Runtime-class definition

For a patched context `k`, define its class under test family `P` as:

- inert if `H_T(k, P_i) = 0` for all tested `P_i`
- stable-but-different if `H_T(k, P_i) = 1` for at least one tested `P_i` and
  the binary still executes cleanly

This partition is central to the `UPLOP3` result.

## 3. Inventory Closure And Problem Compression

### 3.1 Empirical inventory state

The broad corpus now stabilizes at:

- canonical optimized frontier: `379`
- strongest discovery lane: `382`
- checked-in catalog rows: `470`

Mathematically, this means the unresolved set

`U = Catalog - Canonical`

is small enough that the problem has compressed from a high-entropy discovery
phase into a low-entropy control phase. The unresolved direct-local source/IR
cluster is only:

- `P2R.B1`
- `P2R.B2`
- `P2R.B3`
- `UPLOP3.LUT`

### 3.2 Consequence

This compression changes the right proof strategy. When the unresolved set is
large, breadth-first probe expansion is rational. When the unresolved set is
tiny, one should switch to structural RCA, neighborhood mining, cubin-side
validity checks, and runtime semantic classification. That switch has already
occurred in this repository.

## 4. The P2R Frontier

### 4.1 Target statement

The exact target was direct local optimized `sm_89` emission of:

- `P2R.B1`
- `P2R.B2`
- `P2R.B3`

subject to the constraints:

- not library-only evidence
- not plain `P2R`
- not debug-only neighborhood artifacts

### 4.2 What was exhausted

Source-space families exhausted include:

- same-carrier full-mask `0x7f`
- split-seed carriers
- regmask rewrites
- dualpack transitions
- second-bank halfword staging
- staged tripack prefix lifetimes
- nibble-sized `0x0f` paths
- dense `PLOP3`-fed variants
- dense `LOP3.LUT P*`-fed variants
- explicit `SEL`-weighted pack variants
- pressure changes such as `--maxrregcount=32`

Frontend and assembler basins exhausted include:

- direct PTX
- clang CUDA to PTX to `ptxas`
- Triton to PTX to cubin
- tested `ptxas 11.8`, `12.6`, and `13.1`

### 4.3 Lemma: neighborhood reproduction is insufficient

If a source family reproduces the `P2R` neighborhood and plain `P2R` pack
widths, but still fails to induce `P2R.B*`, then neighborhood reproduction is
not the missing condition.

### 4.4 Proof sketch

The repository reproduces:

- plain `P2R ... 0x7f`
- plain `P2R ... 0x0f`
- much of the surrounding predicate neighborhood

But it does not reproduce:

- `P2R.B1`
- `P2R.B2`
- `P2R.B3`

Therefore the missing condition is not raw neighborhood availability. It is a
selection rule over that neighborhood.

### 4.5 Theorem: the live P2R problem is form selection

`P2R.B*` on current SM89 is not blocked by opcode nonexistence. It is blocked by
source/IR-level form selection.

### 4.6 Justification

This follows from the conjunction:

1. `F_T = 0` for all tested direct source/IR basins.
2. plain `P2R` neighborhoods are reproducible.
3. cubin-side substitution materializes runnable `P2R.B1/B2/B3`, so `G_T = 1`.

Hence the failure lies in the map `A(C(s))`, not in the local validity of `T`.

### 4.7 Corollary

Further generic source mutation is lower-yield than either:

- recovering the hidden selection rule
- or accepting cubin-side semantic validation as the reproducible proof layer

## 5. The UPLOP3 Frontier

### 5.1 Structural split

The `UPLOP3` program produced a sharper structural result than the `P2R`
program:

- `ULOP3 -> UPLOP3` is structurally invalid
- `PLOP3 -> UPLOP3` is structurally valid

This already rules out a naive "uniform register form swap" hypothesis.

### 5.2 Lemma: substrate choice matters

Not all semantically related source opcodes are equally valid cubin substrates
for a target opcode family.

### 5.3 Interpretation

`UPLOP3` is not merely a renamed or trivially rebased `ULOP3` form. The valid
local substrate is `PLOP3`, which implies that predicate-logic carrier
structure matters at the encoding level.

### 5.4 Runtime classes

The current runtime-class counts are archived in:

- [uplop3_runtime_class_counts.csv](processed/monograph_20260323/uplop3_runtime_class_counts.csv)

The coarse partition is:

- inert: `12`
- stable-but-different: `3`

This proves that the valid structural path is not only decodable but
semantically stratified.

### 5.5 Live-site hierarchy

The best current live local sites, ranked by library overlap, are:

1. `uniform_occ1`, jaccard `0.375`
2. `cutlass_occ5`, jaccard `0.375`
3. `cutlass_occ4`, jaccard `0.333333`
4. `uniform_occ2`, jaccard `0.300`
5. `uniform_occ5`, jaccard `0.300`
6. `cooperative_occ1`, jaccard `0.300`

This ranking is archived in:

- [uplop3_live_site_numeric.csv](processed/monograph_20260323/uplop3_live_site_numeric.csv)

### 5.6 Pair-baseline law

The pair-baseline results show a causal asymmetry:

- on the uniform branch, `occ2_occ5` is comparatively stable and `uniform_occ1`
  is the main extra widener
- on the CUTLASS branch, `occ2_occ5` is again the stable pair and `cutlass_occ4`
  is the main visible widener

The processed ratios are archived in:

- [uplop3_pair_baseline_numeric.csv](processed/monograph_20260323/uplop3_pair_baseline_numeric.csv)

This suggests a three-role vocabulary:

- anchor
- sensitizer
- amplifier

The anchor supplies the most stable live baseline, the sensitizer perturbs
conditions under which widening occurs, and the amplifier broadens a previously
live branch into stronger visible divergence.

## 6. Toolchain And Measurement Law

The empirical workflow also converged on a practical law:

- differential fuzzing is the strongest semantic discriminator
- `compute-sanitizer` is the safety gate
- `ncu` is a perf-side sanity check
- `nsys` is a trace sidecar

This is archived in:

- [tool_effectiveness_numeric.csv](processed/monograph_20260323/tool_effectiveness_numeric.csv)

The mathematical implication is that semantic information and performance
information are not interchangeable observables. A patch can be semantically
live while remaining perf-regime stable. Therefore one must avoid treating
performance counters as proxies for semantic divergence.

## 7. Visual And Organizational Synthesis

The processed archive under
[processed/monograph_20260323](processed/monograph_20260323)
normalizes the evidence into plot-ready tables. The corresponding monograph
LaTeX file is:

- [sm89_monograph.tex](tex/sm89_monograph.tex)

The visualization package is designed to support:

1. inventory closure plots
2. `P2R` frontier state diagrams
3. `UPLOP3` runtime-class histograms
4. live-site ranking charts
5. pair-baseline divergence-ratio charts

These are not cosmetic figures. Each one corresponds to a specific claim family
and compresses an otherwise diffuse set of artifact families into one falsifiable
visual statement.

## 8. Open Gaps And Next Falsifiable Experiments

### 8.1 Remaining `P2R` gap

Open question:

- which hidden compiler-internal or deeper IR condition chooses `P2R.B*`
  instead of plain `P2R` plus glue?

Falsifiable next experiments:

1. older-sysroot or containerized `nvcc` sweeps
2. deeper NVVM or MLIR perturbation
3. library-delta-guided cubin semantic transplantation

### 8.2 Remaining `UPLOP3` gap

Open question:

- can any direct source/IR basin induce `UPLOP3.LUT` without cubin-side help?

Falsifiable next experiments:

1. motif-sensitive widening from `uniform_occ1`
2. motif-sensitive widening from `cutlass_occ5`
3. mixed tensor/address contexts nearer to the mined library motifs

## 9. Conclusion

The repository has already crossed the line from exploratory notebook work into
a partially axiomatized empirical program. The wide inventory frontier is mostly
closed. `P2R.B*` is now understood as a form-selection problem rather than an
existence problem. `UPLOP3` is now understood as a structurally valid and
semantically stratified cubin-side frontier. The remaining work is no longer
"search harder everywhere." It is "identify the hidden selection laws and the
minimal live motifs that control them."
