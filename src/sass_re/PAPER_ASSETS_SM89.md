# SM89 Paper Assets

This document instantiates the current paper-ready SM89 assets that can be
supported from the present Ada-only evidence base.

It is intentionally bounded. It does not claim cross-architecture completion,
and it does not overstate direct source/IR closure for `P2R.B*` or
`UPLOP3.LUT`.

Monograph-level synthesis and plot-ready archive:

- [MONOGRAPH_SM89_SYNTHESIS.md](MONOGRAPH_SM89_SYNTHESIS.md)
- [sm89_monograph.tex](tex/sm89_monograph.tex)
- [build_monograph_pdf.sh](scripts/build_monograph_pdf.sh)
- [verify_monograph_pdf.py](scripts/verify_monograph_pdf.py)
- [sm89_monograph.pdf](tex/build/sm89_monograph.pdf)
- [MONOGRAPH_BUILD.md](MONOGRAPH_BUILD.md)
- [MONOGRAPH_THEOREM_APPENDIX.md](MONOGRAPH_THEOREM_APPENDIX.md)
- [processed/monograph_20260323/README.md](processed/monograph_20260323/README.md)

## Table A1. SM89 Inventory Summary

Claims:

- `C01`
- `C02`

| Metric | Value | Evidence |
|---|---:|---|
| Recursive probe files | 349 | [RESULTS.md](RESULTS.md) |
| Compile-enabled probe files | 343 | [RESULTS.md](RESULTS.md) |
| Canonical optimized mnemonic frontier | 379 | [RESULTS.md](RESULTS.md) |
| Strongest discovery-lane frontier | 382 | [RESULTS.md](RESULTS.md) |
| Checked-in SM89 catalog rows | 470 | [SM89_SASS_INSTRUCTION_REFERENCE.md](SM89_SASS_INSTRUCTION_REFERENCE.md) |
| Remaining unreproduced direct-local source/IR cluster | `P2R.B1/B2/B3`, `UPLOP3.LUT` | [RESULTS.md](RESULTS.md) |

Bounded interpretation:

- `379` is the current stable optimized frontier.
- `382` is a lane-specific discovery maximum, not the canonical optimized
  baseline.
- Generated table artifact:
  [table_a1_sm89_inventory_summary.csv](tables/table_a1_sm89_inventory_summary.csv)

## Table A2. `P2R` Frontier Status

Claims:

- `C04`
- `C05`
- `C06`
- `C07`
- `C08`

| Axis | Current result | Evidence |
|---|---|---|
| Direct local source/IR emission of `P2R.B1/B2/B3` | not reproduced | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md) |
| Local plain `P2R ... 0x7f` | reproduced | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md) |
| Local plain `P2R ... 0x0f` | reproduced | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md) |
| PTX/clang/Triton/simple `ptxas` version sweep | no direct `P2R.B*` unlock | [Thought_Processes.md](Thought_Processes.md) |
| Cubin-side local `P2R.B1/B2/B3` | materialized and runnable | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) |
| Best current interpretation | source/IR form-selection problem, not opcode-existence problem | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md) |

Bounded interpretation:

- This table is intentionally about the local SM89 `P2R` frontier.
- It does not claim that no other older sysroot or deeper IR layer could ever
  unlock direct source/IR `P2R.B*`.
- Generated table artifact:
  [table_a2_p2r_frontier_status.csv](tables/table_a2_p2r_frontier_status.csv)

## Table A3. `UPLOP3` Structural Boundary

Claims:

- `C09`
- `C10`
- `C11`

| Patch path / class | Current result | Evidence |
|---|---|---|
| `ULOP3 -> UPLOP3` | structurally invalid | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md) |
| `PLOP3 -> UPLOP3` | structurally valid | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md) |
| Local runtime classes | inert and stable-but-different | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md) |
| Direct local source/IR emission of `UPLOP3.LUT` | not reproduced | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) |

Bounded interpretation:

- The breakthrough is structural and semantic, not source-level emission.
- The local substrate is `PLOP3`, not `ULOP3`.
- Generated table artifact:
  [table_a3_uplop3_structural_boundary.csv](tables/table_a3_uplop3_structural_boundary.csv)

## Figure A2. `UPLOP3` Runtime Class Map

Claims:

- `C11`
- `C12`
- `C13`
- `C14`

Recommended visual structure:

1. Left panel: runtime class buckets
   - inert
   - stable-but-different
2. Middle panel: live-site hierarchy
   - `uniform_occ1`
   - `cutlass_occ5`
   - `uniform_occ2`
   - `cutlass_occ4`
   - `uniform_occ5`
   - `cooperative_occ1`
3. Right panel: causal roles
   - anchor
   - secondary anchor
   - sensitizer
   - amplifier

Suggested caption:

> Local `UPLOP3` cubin substitutions separate into inert and semantically live
> classes. The strongest current anchors are `uniform_occ1` and
> `cutlass_occ5`, while `uniform_occ5` behaves more like a sensitizer and
> `cutlass_occ4` more like an amplifier. In the pair-baseline framing,
> `uniform_occ1` is the main extra widener over the uniform `occ2_occ5` pair,
> while `cutlass_occ4` is the main visible widener over the CUTLASS
> `occ2_occ5` pair.

Primary sources:

- [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md)
- [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md)
- Rendered artifact:
  [uplop3_runtime_class_map.svg](figures/uplop3_runtime_class_map.svg)
- Generated artifact:
  [uplop3_runtime_class_map.generated.svg](figures/uplop3_runtime_class_map.generated.svg)

## Table A4. Live `UPLOP3` Site Ranking

Claims:

- `C12`
- `C13`
- `C14`

| Rank | Site | Current role | Interpretation |
|---|---|---|---|
| 1 | `uniform_occ1` | anchor | strongest exact library-aligned live local anchor |
| 2 | `cutlass_occ5` | anchor | strongest CUTLASS-like live anchor |
| 3 | `uniform_occ2` | secondary anchor | stable with `occ2_occ5`, but weaker than `uniform_occ1` |
| 4 | `cutlass_occ4` | amplifier | semantically live, but mainly broadens another live branch |
| 5 | `uniform_occ5` | sensitizer | semantically live, but not a strong standalone anchor |
| 6 | `cooperative_occ1` | lower-priority live site | valid live site, but weaker widening lane |

Primary sources:

- [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md)
- Generated table artifact:
  [table_a4_live_uplop3_site_ranking.csv](tables/table_a4_live_uplop3_site_ranking.csv)

## Table A5. Tool Effectiveness Matrix

Claims:

- `C15`
- `C16`

| Tool | Current role | Why it matters | Caveat |
|---|---|---|---|
| Differential fuzzing | primary semantic discriminator | separates anchors, sensitizers, amplifiers, and broadeners | not a safety tool |
| `compute-sanitizer` | safety gate | shows live cubins are not merely obvious invalid memory behavior | perturbs aggregate sums |
| `ncu` | perf sanity check | helps rule out hidden perf-regime shifts | weak semantic discriminator |
| `nsys` | trace sidecar | preserves timelines and host/driver context for later analysis | lower immediate yield than fuzzing/sanitizer |

Primary sources:

- [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md)
- [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md)
- Generated table artifact:
  [table_a5_tool_effectiveness_matrix.csv](tables/table_a5_tool_effectiveness_matrix.csv)

## Usage Note

These assets are ready for Ada-only bounded insertion into the paper now. They
should not be used to imply that the cross-architecture Pascal-vs-Ada paper is
fully evidenced yet.
