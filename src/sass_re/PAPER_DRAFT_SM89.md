# Bounded Ada-Only Frontier Draft For SM89

This document is a standalone manuscript fragment derived from the current
SM89 paper assets, claims ledger, and outline. It is intentionally narrower
than the full Pascal-vs-Ada paper. Its job is to capture the part of the
story that is already evidence-backed today.

Primary bounded source of truth:

- [PAPER_CLAIMS_MATRIX.md](PAPER_CLAIMS_MATRIX.md)
- [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md)

## Draft Abstract

We present a bounded Ada-only SASS reverse-engineering result set for NVIDIA
SM89 that is already closed enough for paper use, even though the broader
Pascal-vs-Ada comparison remains in progress. Using custom CUDA probes,
systematic disassembly, cubin-side patch validation, runtime differential
fuzzing, and tandem `compute-sanitizer`, `ncu`, and `nsys` passes, we map both
broad instruction inventory coverage and two narrower opcode frontiers:
byte-qualified `P2R.B*` and `UPLOP3.LUT`.

The current Ada-only findings are threefold. First, the stable optimized SM89
inventory reaches `379` canonical raw mnemonics, with a lane-specific maximum
of `382` and a checked-in catalog of `470` rows. Second, direct local
source/IR still does not emit `P2R.B1/B2/B3`, but local cubin-side
substitution proves that all three are valid and runnable on the same target,
which reframes the remaining gap as a form-selection problem rather than an
opcode-existence problem. Third, `PLOP3 -> UPLOP3` is structurally valid while
`ULOP3 -> UPLOP3` is not, and local `UPLOP3` substitutions partition into
inert and semantically live runtime classes. These results are already strong
enough for bounded paper claims, while stronger cross-architecture conclusions
remain reserved for future Pascal-side measurement.

Relevant claims:

- `C01`
- `C02`
- `C04`-`C17`

Relevant assets:

- [table_a1_sm89_inventory_summary.csv](tables/table_a1_sm89_inventory_summary.csv)
- [table_a2_p2r_frontier_status.csv](tables/table_a2_p2r_frontier_status.csv)
- [table_a3_uplop3_structural_boundary.csv](tables/table_a3_uplop3_structural_boundary.csv)
- [uplop3_runtime_class_map.generated.svg](figures/uplop3_runtime_class_map.generated.svg)
- [table_a4_live_uplop3_site_ranking.csv](tables/table_a4_live_uplop3_site_ranking.csv)
- [table_a5_tool_effectiveness_matrix.csv](tables/table_a5_tool_effectiveness_matrix.csv)

## Draft Section 7.4: Ada-Only Frontier Status

On current local SM89, the `P2R` and `UPLOP3` frontiers are now bounded in a
way that is strong enough for declarative paper use, even though neither
frontier is source-level closed in the same way. Table A2 captures the `P2R`
state: direct local source and frontend/IR search still do not emit
`P2R.B1/B2/B3`, despite extensive CUDA-source mutation, PTX search, clang and
Triton frontend variation, and tested `ptxas 11.8/12.6/13.1` sweeps. At the
same time, the local compiler does reproduce the surrounding neighborhood,
including plain `P2R ... 0x7f` and `P2R ... 0x0f`. This combination is the
important result. The failure is no longer best interpreted as a missing
opcode or missing neighborhood. It is better described as a source/IR-level
form-selection problem. Cubin-side substitution then closes the
opcode-validity question directly by materializing and running `P2R.B1`,
`P2R.B2`, and `P2R.B3` on the same local SM89 target.

Table A3 captures the parallel `UPLOP3` boundary. Direct local source/IR still
does not emit `UPLOP3.LUT`, but cubin-side substitution now exposes a sharp
structural rule: `ULOP3 -> UPLOP3` is invalid, whereas `PLOP3 -> UPLOP3` is
structurally valid. That result matters because it does more than prove a
decode spelling. The valid `PLOP3 -> UPLOP3` substitutions launch and execute
in multiple local contexts, and they separate into inert and
stable-but-different runtime classes. In other words, the local `UPLOP3`
frontier has already crossed from structural decode validation into semantic
classification. Figure A2 and Table A4 make this shift explicit: some patched
sites behave like true live local anchors, while others remain semantically
neutral even though they decode and run.

Relevant claims:

- `C04`-`C11`

## Draft Section 8.5: Ada-Only Frontier Synthesis

The current SM89 frontier is best understood as two different bounded stories
with two different levels of closure. For `P2R`, the key contribution is a
negative source-level result paired with a positive opcode-validity result.
Direct local source/IR still does not select `P2R.B*`, but local cubin-side
substitution proves that the byte-qualified opcodes are valid and runnable on
the same machine. That changes the question from "does SM89 support these
forms?" to "what compiler-internal condition selects them?" This is why the
remaining `P2R` problem is best framed as a form-selection problem, not an
opcode-existence problem.

For `UPLOP3`, the frontier is further along. The repo now has a structurally
valid cubin-side path, runtime-safe execution, and a ranked semantic map of
live sites. Figure A2 and Table A4 show that the live set is not homogeneous.
`uniform_occ1` and `cutlass_occ5` are the strongest current anchors.
`uniform_occ2` behaves like a secondary anchor. `uniform_occ5` behaves more
like a sensitizer, and `cutlass_occ4` behaves more like an amplifier than a
primary anchor. This is a stronger statement than merely saying that patched
`UPLOP3` sites can differ from baseline; it means the frontier already has a
usable causal vocabulary.

The pair-baseline framing makes that causal vocabulary more concrete. On the
uniform branch, `occ2_occ5` is comparatively stable, and `uniform_occ1` is the
main extra widener over that stable pair. On the CUTLASS branch, `occ2_occ5`
again behaves like the stable anchor pair, while `cutlass_occ4` is the main
visible widener. Richer CUTLASS combinations often preserve visible output
prefixes while still perturbing aggregate sums, which is why they cannot be
understood from disassembly alone. Differential fuzzing was the decisive tool
for separating those behaviors. Table A5 summarizes the resulting workflow:
differential fuzzing is the primary semantic discriminator,
`compute-sanitizer` is the safety gate, `ncu` is a perf-side sanity check, and
`nsys` is a lower-yield but still useful trace sidecar.

Relevant claims:

- `C12`-`C18`

## Draft Conclusion

The current Ada-only evidence supports a bounded but substantial paper result.
On SM89, the broad inventory question is largely closed, while the remaining
interesting work has narrowed to form-selection and semantic-validation
frontiers rather than generic opcode discovery. The `P2R` program established a
clean negative source/IR result and a positive cubin-side opcode-validity
result on the same target, which sharply localizes the remaining open problem.
The `UPLOP3` program went further by establishing a structurally valid
cubin-side path, runtime-safe execution, and a ranked set of semantically live
local sites with distinct causal roles.

Methodologically, the work also clarifies which tools matter at each stage.
Source mutation, PTX/front-end variation, and `ptxas` version sweeps are useful
for bounding negative space. Cubin-side substitution resolves opcode-validity
questions. Differential fuzzing is the strongest semantic discriminator once a
patched form is runnable, with `compute-sanitizer` acting as the safety gate
and `ncu`/`nsys` serving as secondary performance and trace sidecars. This
Ada-only draft should therefore be read as an evidence-backed frontier study
embedded in a larger Pascal-vs-Ada comparison scaffold that remains ready for
future cross-architecture completion.

## Bounded-Language Reminder

This fragment must not be revised to claim that:

- direct source-level `P2R.B*` has been reproduced
- direct source-level `UPLOP3.LUT` has been reproduced

It may safely claim that:

- local cubin-side `P2R.B1/B2/B3` materialization is proven
- `PLOP3 -> UPLOP3` is structurally valid
- local `UPLOP3` substitutions separate into inert and stable-but-different
  runtime classes
