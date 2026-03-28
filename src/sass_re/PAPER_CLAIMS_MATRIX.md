# SM89 SASS RE Paper Claims Matrix

This matrix turns the current SM89 reverse-engineering state into paper-ready
claims with explicit evidence and scope.

Status meanings:

- `proven`: directly supported by reproducible repo artifacts
- `supported`: strongly indicated, but narrower than a universal claim
- `bounded-negative`: the repo strongly supports a negative result or failure
  boundary
- `open`: not yet closed by current evidence

| Claim ID | Claim | Status | Evidence | Interpretation / Scope |
|---|---|---|---|---|
| C01 | The recursive SM89 probe corpus stabilizes at `379` canonical optimized raw mnemonics. | proven | [RESULTS.md](RESULTS.md), [SM89_SASS_INSTRUCTION_REFERENCE.md](SM89_SASS_INSTRUCTION_REFERENCE.md) | This is the stable baseline frontier across the canonical optimized flag lanes. |
| C02 | The strongest current discovery lane reaches `382` raw mnemonics under `--maxrregcount=32`. | proven | [RESULTS.md](RESULTS.md), [SM89_SASS_INSTRUCTION_REFERENCE.md](SM89_SASS_INSTRUCTION_REFERENCE.md) | This is a bounded lane-specific bump, not the new canonical optimized baseline. |
| C03 | The async/cache plus SYS64 combo family is directly reproducible in local emitted SM89 code. | proven | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md), [RESULTS.md](RESULTS.md) | This includes `LDGSTS`, `LDGDEPBAR`, `DEPBAR`, `BAR.RED`, `MATCH`, `REDUX`, `VOTE`, and dense `ATOMG.E.*.64.STRONG.SYS` neighborhoods. |
| C04 | Direct local source/IR on the current toolchain stack does not emit `P2R.B1`, `P2R.B2`, or `P2R.B3`. | bounded-negative | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md), [Thought_Processes.md](Thought_Processes.md) | This is a strong negative result over the tested CUDA source, PTX, clang, Triton, and `ptxas` sweep space. |
| C05 | The local compiler can reproduce the `P2R` neighborhood, including plain `P2R ... 0x7f` and `P2R ... 0x0f`, without selecting byte-qualified `P2R.B*`. | proven | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md) | This shows the frontier is a form-selection problem, not a missing-neighborhood problem. |
| C06 | Simple PTX-level, clang, Triton, and tested `ptxas 11.8/12.6/13.1` sweeps do not unlock direct local `P2R.B*` emission. | bounded-negative | [Thought_Processes.md](Thought_Processes.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This is a bounded statement over the tested basins, not a proof that no older sysroot or deeper IR layer could do it. |
| C07 | Local cubin-side substitution can materialize and run `P2R.B1`, `P2R.B2`, and `P2R.B3` on SM89. | proven | [RESULTS.md](RESULTS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This proves the opcodes are locally valid and runnable even though source/IR still does not select them. |
| C08 | The remaining `P2R` frontier is source/IR-level form selection, not opcode existence. | supported | [P2R_FRONTIER_ANALYSIS.md](P2R_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This is the main synthesis claim from the `P2R` RCA work. |
| C09 | `ULOP3 -> UPLOP3` is structurally invalid in local cubin-side substitution. | proven | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | The patched cubin is rejected as illegal under disassembly. |
| C10 | `PLOP3 -> UPLOP3` is the structurally valid local cubin-side path for `UPLOP3.LUT`. | proven | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This is the key structural breakthrough for the second major frontier. |
| C11 | Local `UPLOP3.LUT` substitutions partition into inert and stable-but-different runtime classes. | proven | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [RESULTS.md](RESULTS.md) | This is stronger than a decode-only claim: the repo now has runtime semantic classes. |
| C12 | `uniform_occ1` and `cutlass_occ5` are the strongest current live local `UPLOP3` anchors. | supported | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This is a ranking claim based on tandem evidence, not a universal theorem. |
| C13 | `uniform_occ5` acts more like a sensitizer, and `cutlass_occ4` more like an amplifier/modifier, than as primary anchors. | supported | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [RESULTS.md](RESULTS.md) | This is the causal interpretation from the partner-centered and pair-baseline tranches. |
| C14 | In the pair-baseline framing, `uniform_occ1` is the main extra widener over `occ2_occ5`, while `cutlass_occ4` is the main visible widener over the CUTLASS `occ2_occ5` pair. | supported | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md), [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md) | This is the cleanest current causal statement for the live `UPLOP3` sites. |
| C15 | Differential fuzzing is the strongest current semantic discriminator for the `UPLOP3` frontier. | proven | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | It separates anchors, sensitizers, amplifiers, and broad wideners more clearly than the perf tools. |
| C16 | `compute-sanitizer` is the right safety gate, while `ncu` is mainly a perf-side sanity check rather than the primary semantic discovery tool for `UPLOP3`. | supported | [UPLOP3_FRONTIER_ANALYSIS.md](UPLOP3_FRONTIER_ANALYSIS.md), [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md) | This is a workflow claim grounded in repeated tandem runs. |
| C17 | The strongest remaining non-cubin frontier is `UPLOP3.LUT`, not `P2R.B*`. | supported | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md), [SM89_SASS_INSTRUCTION_REFERENCE.md](SM89_SASS_INSTRUCTION_REFERENCE.md) | `P2R.B*` is now cubin-side closed, while `UPLOP3` still lacks direct source/IR emission. |
| C18 | The best next paper-worthy frontier is motif-sensitive widening from the live `UPLOP3` anchors rather than more generic `P2R` source mutation. | open | [KNOWLEDGE_SYNTHESIS.md](KNOWLEDGE_SYNTHESIS.md), [Thought_Processes.md](Thought_Processes.md) | This is the current forward-looking research hypothesis, not a closed result. |

## Notes For Paper Use

- Claims `C01`-`C03`, `C07`, `C09`-`C11`, and `C15` are the safest current
  candidates for strong declarative paper prose.
- Claims `C08`, `C12`-`C14`, and `C17` are strong synthesis claims, but should
  be phrased with bounded language.
- Claim `C18` is a forward-looking research direction and should not be stated
  as a finding.
