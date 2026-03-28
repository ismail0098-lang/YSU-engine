# Thought Processes: SM89 SASS Reverse Engineering

This document is a retrospective analytical narrative of the intellectual
process behind the SM89 SASS reverse engineering effort. It captures why
each direction was chosen, where the research branched, what was learned
at each stage, and what remains unresolved. Think lab notebook, not
project plan.


## 1. Research Methodology

### The core problem

The goal was systematic enumeration and characterization of NVIDIA Ada
Lovelace (SM89) SASS instructions -- the native ISA that sits below PTX
and is not publicly documented. The approach was empirical: write probe
kernels, compile them, disassemble the resulting cubins, profile them
under Nsight Compute, and iteratively widen coverage until the frontier
stopped yielding new mnemonics.

### Why probes, not static analysis alone

Static disassembly of installed libraries (cuDNN, cuBLAS, TensorRT) can
show what instructions exist, but it cannot explain what triggers them or
how they interact under scheduling. Probes let us control the source
shape and observe what the compiler selects, which gives causal
information that pure mining cannot.

### Strategy evolution

The strategy evolved through three distinct phases:

1. **Broad enumeration** (early probes through ~379 mnemonics): Write
   many small kernels targeting different functional families (arithmetic,
   memory, control, synchronization, atomics, reductions, uniform
   operations, tensor cores). Each probe is a hypothesis about what
   source patterns the compiler will lower into which SASS forms.

2. **Combo frontier** (~379 to ~448 mnemonics): After diminishing returns
   from single-family probes, shift to multi-family "combo" kernels that
   force coexistence of instructions from different domains (async/cache
   + atomics + reductions + warp intrinsics + uniform helpers) in single
   kernels. This turned out to be the highest-yield strategy.

3. **Residual frontier** (448+ mnemonics): The remaining unreproduced
   instructions required escalation beyond ordinary CUDA C++ source:
   PTX-level mutation, library binary mining, cross-toolchain sweeps,
   and ultimately cubin surgery.

### Why the auto-explorer

By the combo frontier phase, the search space was structured enough that
a classical registry-driven Python explorer was justified. The key
insight was that the frontier was still sparse, discrete, and
interpretable -- not continuous geometry. A TOML search-space registry
with sklearn surrogate models and optuna for bounded discrete search was
the right first architecture, not a neural policy model.

The useful pattern borrowed from the sibling `open_gororoba` project was
the explicit registry and lineage style, not the Cayley-Dickson algebra.
The frontier is a discrete experiment graph (source-shape toggles, flag
toggles, mnemonic-chain neighborhoods, stall-regime transitions), not a
continuous manifold.


## 2. Key Decision Points

### Decision: Combo kernels over single-family probes

After reaching ~379 raw mnemonics from individual probes, the
single-family approach flattened. The chain-mining pass showed that the
dominant anchor windows in cuDNN and local code were all async/cache
variants, not predicate-byte outliers. The remaining frontier was about
sub-variant selection inside multi-family neighborhoods, not about
discovering absent neighborhoods.

This led to the combo probe strategy: force `LDG(.STRONG.GPU) + LDGSTS +
LDGDEPBAR + DEPBAR.LE` to coexist with warp intrinsics, block
reductions, system-scope atomics, and uniform helpers in single kernels.
Each combo was a controlled experiment about which instruction families
could survive together under the compiler's optimizer.

### Decision: Runtime safety before performance characterization

Early combo probes produced rich symbolic SASS but hit runtime errors
(misaligned address, illegal instruction) because the source shapes
needed for instruction coverage were not always safe to execute. This
created a split between "symbolic-only" probes (useful for disassembly)
and "runtime-safe" probes (safe for ncu profiling).

The decision was to systematically create runtime-safe surrogates for
each symbolic frontier, preserving the emitted instruction neighborhoods
while making the execution path safe for profiling. This was the right
call: it separated the "does this instruction exist?" question from the
"what does it cost?" question and avoided false conclusions from
unstable execution.

### Decision: `-dlcm=cg` as a refinement lever, not a regime changer

Repeated controlled experiments across small, medium, and heavy combo
families consistently showed: forcing `STRONG.GPU` load policy via
`-dlcm=cg` changes emitted load spellings and can improve L2 hit rates,
but does not materially change cycle counts or stall distributions. This
was true across the lighter uniform-helper family, the heavy 64-bit
SYS-safe family, and the fully fused store-side family.

The conclusion: cache-policy mutation is not a primary driver once the
fused SYS64 body exists. It is a secondary refinement at best.

### Decision: Stop source-space P2R.B* retries, escalate to binary methods

After exhaustive source-space exploration of the `P2R.B1/B2/B3`
frontier (mask width, carrier lifetime, byte-store vs masked-register
rewrite, higher-byte prefix state, predicate-source kind via PLOP3-fed
carriers), all axes were strip-mined with the same result: the compiler
consistently lowers to plain `P2R` plus GPR glue or drops `P2R` entirely
and rebuilds with `SEL + LOP3`. The decision to stop and escalate was
based on diminishing returns from a well-mapped search space.

### Decision: Cubin surgery for semantic validation

When PTX-level search, frontend variation (clang, Triton), and cross-
version ptxas sweeps (CUDA 11.8, 12.6, 13.1) all converged on the same
negative result for `P2R.B*`, the next meaningful escalation was to
operate directly on cubins. This was not a shortcut -- it was the only
remaining path to answer "are these instructions semantically valid in
locally compiled code?" The answer: yes, through substitution, but
source/IR selection remains closed.


## 3. Discovery Timeline

### Phase 1: Initial enumeration (probes 1-~200, ~0 to ~350 mnemonics)

Individual probe kernels targeting specific instruction families:
arithmetic (integer, floating-point, half-precision), memory (loads,
stores, atomics at various scopes and sizes), control flow (branches,
barriers, synchronization), uniform operations (ULDC, UIADD3, ULOP3),
and tensor core operations (HMMA).

Key early discoveries:
- `USHF.L.U64.HI` via the predicate uniform frontier
- `ULOP3.LUT` in the uniform-path corpus
- `R2P` via the transcendental compile-profile path
- `P2R Rn, PR, Rn, 0x7f` via the two-stage bank probe

### Phase 2: Flag sweeps and specialized probes (~350 to ~382 mnemonics)

Systematic flag sweeps across the corpus:
- `--maxrregcount=32` as the strongest discovery lane (382 mnemonics)
- `--restrict` (381 mnemonics)
- `-Xptxas -dlcm=cg` as the strongest load-policy lane

Specialized runner infrastructure for probes that could not use the
generic launcher: barrier arrive/wait, cooperative launch, depbar
explicit, tiling hierarchical.

### Phase 3: Combo frontier (~382 to ~448 mnemonics)

The combo strategy produced the largest single-phase jump in coverage.
The progression was:

1. **Async/cache backbone**: Establish `LDGSTS + LDGDEPBAR + DEPBAR.LE`
   as the anchor.
2. **Warp helpers**: Add `MATCH.ANY`, `REDUX.*`, `VOTE.*` alongside
   block reductions (`BAR.RED.*`, `B2R.RESULT`).
3. **System-scope atomics**: Widen to `ATOMG.E.*.STRONG.SYS` and
   `RED.E.*.STRONG.SYS` families.
4. **64-bit system scope**: Push to
   `ATOMG.E.ADD/MIN/MAX/AND/OR/XOR.64.STRONG.SYS`, with `MEMBAR.SC.SYS`,
   `ERRBAR`, and `CCTL.IVALL`.
5. **Direct SYS load/store**: Add `LDG.E.64.STRONG.SYS` and
   `STG.E.64.STRONG.SYS`.
6. **Uniform helper integration**: Fuse `ULDC/UIADD3/ULOP3/USHF` with
   the full SYS64 atomic body.
7. **Divergence/reconvergence**: Add `BSSY` and `BSYNC` in optimized
   code (distinct from `WARPSYNC`, which remained debug-lane weighted).

### Phase 4: P2R residual frontier (parallel to Phase 3)

This was the systematic attempt to close the last three direct-local
unreproduced mnemonics: `P2R.B1`, `P2R.B2`, `P2R.B3`.

The progression:
1. Source-space retries (mask width, carrier lifetime, byte-store vs
   register rewrite, higher-byte prefix, PLOP3-fed carriers)
2. Library binary mining (cuDNN precompiled engines dominant; cuBLAS and
   TensorRT negative)
3. PTX-level search (direct predicate-logic variants, still negative)
4. Frontend variation (clang, Triton -- distinct basins, still negative)
5. Cross-version ptxas sweep (CUDA 11.8, 12.6, 13.1 -- version-invariant
   negative)
6. Motif-seeded constrained PTX mutation (still negative)
7. Cubin-side substitution (breakthrough: byte-qualified forms materialize
   and disassemble correctly)
8. Semantic validation matrix (runtime outcomes in three classes: inert
   on some patterns, stable-but-different on others, non-terminating in
   some contexts)


## 4. Frontier Evolution

### The combo frontier families

The runtime-safe combo families organized into a clear performance
hierarchy:

**Small safe family** (dependency-depth / scoreboard study):
- Long-scoreboard ~32.5%, no barrier or membar stalls
- Clean dependency-depth study case
- Deeper structure raises instruction count, register count, shared
  memory, short-scoreboard, and wait, but never introduces barrier or
  membar

**Uniform-helper / system-RED family** (lightweight SYS variant):
- Long-scoreboard ~29.3%, membar = 0%
- Closer to the small safe family than to the heavy SYS branches

**Narrow 64-bit SYS atomic family** (cleanest membar-dominated branch):
- Long-scoreboard ~23.8%, membar ~37.8%
- The purest membar-dominated executable branch observed

**Uniform + divergent SYS64 bridge** (midpoint regime):
- Long-scoreboard ~20.56%, membar ~31.67%, short-scoreboard ~10.76%
- A genuine midpoint between lighter uniform helpers and heavier SYS store

**64-bit SYS-safe family** (heavy long-scoreboard + membar):
- Long-scoreboard ~53.4%, membar ~24.8%
- Qualitatively distinct from the lighter async/cache combos: a genuine
  memory-barrier plus dependency-latency regime

**Store-side SYS-safe family** (heaviest fused regime):
- Long-scoreboard ~46.2%, membar ~25.7%
- Direct SYS load/store alive in executable code on top of the full
  block-red, warp, and dense 64-bit ATOMG body

**Fully fused uniform + divergent + block-red + SYS64 + store**:
- The most aggressive branch: uniform front-end trims long-scoreboard
  but introduces larger short-scoreboard term
- Deeper variants push long-scoreboard to ~53.64% while reducing membar
  to ~17.37%

### Causal insights from the frontier

The controlled bridge experiments produced several clear causal
findings:

1. **Direct SYS load/store, not divergence, drives the heavy regime.**
   The divergent SYS-store bridge (~44.26% long-scoreboard, ~25.15%
   membar) tracked much closer to the heavy store-side family than to the
   lighter uniform-divergent midpoint. Divergence alone does not pull the
   branch into the heavy regime.

2. **The uniform front-end is a short-scoreboard lever.** Adding
   `ULDC/UIADD3/ULOP3/USHF` to the SYS-store body softens long-scoreboard
   pressure but introduces short-scoreboard terms that were absent
   without it.

3. **Deeper structure amplifies membar and secondary latency terms.**
   Deepening the 64-bit SYS body raises membar, short-scoreboard, and
   wait, while long-scoreboard remains the single largest stall class.

4. **The membar term is not solely from direct SYS store.** Removing
   direct SYS load/store from the deeper fused branch reduces but does not
   eliminate the membar term. The broader SYS64 fused body itself carries
   it.

### The P2R frontier

The P2R investigation produced a progressively sharpened root-cause
analysis:

1. **Carrier lifetime is not the issue.** Multiple same-carrier and
   tripack experiments showed that even when prior higher-byte packs are
   alive in the same carrier, the compiler still prefers plain `P2R` plus
   GPR glue.

2. **Mask width is not the issue.** Nibble-sized `0xf` variants reproduce
   plain `P2R ... 0xf` but still miss byte-qualified `P2R.B2/B3`.

3. **Byte-store vs masked-register rewrite is not the issue.** Whole-
   register masked rewrites produce the same result as direct byte-store
   variants.

4. **Predicate-source kind changes the neighborhood but not the form
   selection.** PLOP3-fed carriers reproduce dense `LOP3.LUT P*` and
   `PLOP3.LUT` neighborhoods, but the compiler still does not select
   `P2R.B1/B2/B3`.

5. **The gap is version-invariant.** Three generations of ptxas
   (CUDA 11.8, 12.6, 13.1) all produce the same negative result on the
   same source shapes.

6. **The gap is frontend-invariant.** nvcc, clang, and Triton all fail
   to produce `P2R.B*`, though each produces a distinct instruction
   basin.

7. **Cubin-side substitution confirms the instructions are valid.**
   Patching plain-`P2R` sites to byte-qualified forms produces correct
   disassembly and (in favorable contexts) runtime-stable execution.

The surviving conclusion: `P2R.B1/B2/B3` is likely compiler-internal,
library-only, or dependent on a lower-level IR pattern that ordinary
CUDA C++ source does not expose on the tested toolchains.

### The UPLOP3 frontier

The UPLOP3 investigation established a structural law:

- `ULOP3` is the wrong patch substrate for reaching `UPLOP3`
- `PLOP3` is the right one
- The result crosses from decode-table reasoning into runtime semantic
  classes: some substitutions are semantically inert, others produce
  stable-but-different outputs, and the distinction is reproducible

The pair-baseline analysis refined the live-site hierarchy:
- Strongest stable pair anchors: uniform `occ2_occ5` and CUTLASS
  `occ2_occ5`
- On the uniform branch, `occ1` acts as the main widener once the pair
  baseline is established
- On the CUTLASS branch, `occ4` acts as the principal amplifier while
  `occ5` remains the trigger


## 5. What Was Learned

### Architectural insights

1. **The SM89 async/cache backbone is the central organizing structure.**
   `LDGSTS + LDGDEPBAR + DEPBAR.LE` is not just one instruction family
   among many -- it is the anchor that enables coexistence of warp
   intrinsics, block reductions, system-scope atomics, and uniform
   helpers in a single kernel.

2. **System-scope operations create qualitatively distinct performance
   regimes.** The transition from `STRONG.GPU` to `STRONG.SYS` is not
   just a scope change; it introduces membar as a major stall source and
   pushes the kernel into a different scheduling regime. This is visible
   in the data, not just the ISA encoding.

3. **The compiler's form selection is more complex than instruction
   availability.** Many instructions (P2R.B*, WARPSYNC in optimized code)
   exist in the ISA and execute correctly when manually substituted, but
   the compiler consistently selects different lowerings from source code.
   This is not a toolchain bug; it is a form-selection policy that
   reflects the compiler's cost model.

4. **Divergence and reconvergence have distinct representation in
   optimized vs debug code.** `BSSY/BSYNC` appear in optimized code;
   `WARPSYNC/WARPSYNC.EXCLUSIVE` remain debug-lane weighted. The source
   pattern (`__syncwarp`) is the same in both cases; the compiler's
   optimization level determines which SASS form is selected.

5. **The uniform instruction family (ULDC, UIADD3, ULOP3, USHF) is a
   scheduling-sensitive domain.** Getting `UISETP` to appear requires
   that compare operands remain in uniform registers through scheduling,
   not merely that they are warp-uniform in source semantics. The compiler
   rewrites the compare chain before the async/cache half unless the
   entire predicate path stays in the pure uniform corpus.

### Methodological insights

1. **Registry-driven exploration beats random search.** The TOML
   search-space registry with explicit candidate derivation metadata
   outperformed ad hoc exploration because it made coverage gaps visible
   and made the explorer's state inspectable.

2. **Controlled comparisons require bridge experiments.** To understand
   what causes the heavy SYS-side regime, we needed branches that
   systematically added and removed individual features (uniform helpers,
   divergence, direct SYS store, depth). Each bridge was a controlled
   experiment, not just another probe.

3. **Cubin surgery is the last resort but also the strongest evidence.**
   When source-space, PTX-level, and frontend approaches all converge
   on the same boundary, direct cubin manipulation provides the definitive
   answer about whether the boundary is encoding-level or selection-level.

4. **Surrogate models are useful even when simple.** A
   `RandomForestRegressor` over cycles, long-scoreboard pressure, and
   membar pressure was already enough to rank candidate branches and
   predict which would yield new information. The dataset is too small
   and too interpretable for heavier ML.

### Tool-role separation

The research established that different observability tools answer
different questions and should not be conflated:

- **cuobjdump / nvdisasm**: What instructions exist (static analysis)
- **ncu (Nsight Compute)**: What the performance regime looks like
  (profiling)
- **compute-sanitizer**: Whether execution is safe (correctness)
- **nsys (Nsight Systems)**: Temporal execution trace (scheduling)
- **Differential fuzzing**: Whether semantic substitutions change
  observable behavior (semantic validation)


## 6. Open Questions

### Source/IR form selection

Why does the SM89 ptxas consistently select plain `P2R` plus GPR glue
(or `SEL + LOP3` reconstruction) instead of byte-qualified `P2R.B*`,
even when the source pattern closely matches the cuDNN neighborhoods
where `P2R.B*` appears? Is this a cost-model decision, an IR-level
representation gap, or a feature gated by an internal compiler flag?

### WARPSYNC in optimized code

Can any source pattern produce `WARPSYNC` or `WARPSYNC.EXCLUSIVE` in
optimized (`-O3`) code, or are these forms strictly debug-lane artifacts
on SM89? The current evidence says optimized code always selects
`BSSY/BSYNC` instead, but the search space for source patterns that
might trigger the alternative is not fully exhausted.

### UISETP in mixed bodies

Can `UISETP` survive in a kernel body that also contains async/cache
and system-scope operations, or does the compiler always rewrite the
uniform compare chain when the body becomes sufficiently complex? The
current evidence suggests the latter, but a more literal control-
skeleton transplant from known UISETP-producing probes has not been
tried in the latest combo family.

### Pascal-side comparison

The full Pascal-vs-Ada microarchitecture comparison paper remains
scaffolded but not evidenced. It is blocked on Pascal-side latency
tables, throughput tables, and cross-architecture encoding field
comparisons. The Ada-only monograph can stand alone, but the
comparative story is incomplete.

### Deeper cubin-side semantic validation

The cubin-side `P2R.B*` substitution matrix revealed three semantic
outcome classes (inert, stable-but-different, non-terminating), but the
mapping from source context to semantic outcome class is not yet
understood. Which structural properties of the surrounding code
determine whether a byte-qualified substitution preserves semantics?

### Combo family ceiling

The auto-explorer's runtime queue was exhausted after the bridge
experiments. Are there genuinely new runtime-safe combo families beyond
the current matrix, or has the combinatorial frontier been effectively
mapped? The symbolic-only `P2R.B1/B2/B3` boundary remains, but no
runtime-safe branch is known to still be missing.

### TensorRT sm89 builder resources

TensorRT's sm89 builder resource was not directly `cuobjdump`-readable
as device code. A deeper inspection of its sections and packaging might
reveal additional `P2R.B*` or other frontier instruction windows, but
this path was deprioritized in favor of cuDNN mining, which was
productive.

---

*This document consolidates the research reasoning from 14 planning and
roadmap documents that tracked the SM89 SASS reverse engineering effort
from initial probes through 448 mnemonics. It is a record of decisions
and discoveries, not a plan for future work.*
