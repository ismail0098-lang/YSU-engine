# SM89 Paper Figure and Table Plan

This plan maps the current paper claims to concrete figures, tables, and
sections in the paper outline.

It separates:

- items that are already supportable from current SM89 evidence
- items that still depend on future Pascal-side or cross-arch work

## A. Immediately Supportable From Current SM89 Evidence

### Table A1: SM89 inventory summary

Claims:

- `C01`
- `C02`

Content:

- canonical optimized frontier: `379`
- strongest discovery lane: `382`
- checked-in catalog rows: `470`

Likely placement:

- paper Section 4.1 or a short Ada-only inventory subsection before the full
  cross-arch ISA comparison

Primary sources:

- [SM89_SASS_INSTRUCTION_REFERENCE.md](SM89_SASS_INSTRUCTION_REFERENCE.md)
- [RESULTS.md](RESULTS.md)

### Table A2: `P2R` frontier status

Claims:

- `C04`
- `C05`
- `C06`
- `C07`
- `C08`

Content:

- source-space status: no direct local `P2R.B1/B2/B3`
- local plain `P2R` outcomes: `0x7f`, `0x0f`
- tested frontend/version basins
- cubin-side materialization status for `P2R.B1/B2/B3`

Likely placement:

- paper Section 7 or an Ada-only frontier subsection after the core binary
  encoding material

Primary sources:

- [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md)
- [Thought_Processes.md](Thought_Processes.md)
- [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md)

### Figure A1: `P2R` evidence ladder

Claims:

- `C05`
- `C07`
- `C08`

Content:

- source-space attempts -> plain `P2R`
- PTX/frontend/version sweeps -> still no `P2R.B*`
- cubin-side substitution -> runnable `P2R.B1/B2/B3`

Likely placement:

- discussion-facing figure in Section 8

### Table A3: `UPLOP3` structural boundary

Claims:

- `C09`
- `C10`
- `C11`

Content:

- `ULOP3 -> UPLOP3`: illegal
- `PLOP3 -> UPLOP3`: decodes cleanly
- runtime classes: inert vs stable-but-different

Likely placement:

- paper Section 7 or a dedicated Ada-only encoding/semantic subsection

Primary sources:

- [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md)
- [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md)

### Figure A2: `UPLOP3` runtime class map

Claims:

- `C11`
- `C12`
- `C13`
- `C14`

Content:

- inert sites
- stable-but-different sites
- anchor sites
- sensitizer/amplifier interpretation
- pair-baseline widening summary

Likely placement:

- Section 8 discussion figure or Ada-only results subsection

### Table A4: Live `UPLOP3` site ranking

Claims:

- `C12`
- `C13`
- `C14`

Content:

- rank
- site
- class: anchor / secondary anchor / sensitizer / amplifier
- best motif overlap
- pair-baseline interpretation

Likely placement:

- Section 8

### Table A5: Tool effectiveness matrix

Claims:

- `C15`
- `C16`

Content:

- differential fuzzing
- `compute-sanitizer`
- `ncu`
- `nsys`
- role: semantic discriminator / safety gate / perf sanity / trace sidecar

Likely placement:

- methodology subsection or discussion subsection

## B. Supportable But Better Kept Bounded In Prose

These are current synthesis claims that should usually stay in prose rather
than becoming headline tables:

- `C03`
- `C08`
- `C17`

Reason:

- they are real and useful, but they are framing claims over multiple artifact
  families rather than single-measurement results

## C. Evidence Gaps In The Current Paper Outline

The current paper outline is cross-architecture, but much of the new frontier
work is Ada-only.

Sections still blocked on future evidence:

- Section 4 cross-arch ISA delta tables
- Section 5 latency comparison tables
- Section 6 throughput comparison tables
- Section 7 direct Pascal-vs-Ada encoding comparison tables

What we can already add safely:

- an Ada-only subsection or boxed result in Sections 4, 7, or 8 for:
  - the `P2R` frontier
  - the `UPLOP3` structural boundary
  - the live-site semantics

## D. Recommended Near-Term Paper Edits

1. Keep the main title and outline cross-architecture.
2. Add an Ada-only subsection in the results/discussion path for current
   frontier findings.
3. Use the claims matrix as the bounded source of truth for wording.
4. Do not let the paper prose imply direct source/IR emission of `P2R.B*` or
   `UPLOP3.LUT`; that would overstate the evidence.

## E. Best Current Figure/Table Set

If we only add a minimal paper-ready tranche right now, the best set is:

1. Table A1: SM89 inventory summary
2. Table A2: `P2R` frontier status
3. Table A3: `UPLOP3` structural boundary
4. Figure A2: `UPLOP3` runtime class map
5. Table A5: tool effectiveness matrix

That set gives one inventory result, one negative frontier result, one
structural breakthrough result, one semantic ranking result, and one methods
result.
