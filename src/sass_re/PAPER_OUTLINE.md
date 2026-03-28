# A Decade of GPU Microarchitecture: Empirical SASS-Level Comparison of Pascal and Ada Lovelace

**Authors:** Umut Korkmaz  
**Affiliation:** Independent Researcher  
**Date:** 2026

---

Current claim ledger:
[PAPER_CLAIMS_MATRIX.md](PAPER_CLAIMS_MATRIX.md)

Current figure/table coverage plan:
[PAPER_FIGURE_TABLE_PLAN.md](PAPER_FIGURE_TABLE_PLAN.md)

Current instantiated SM89 paper assets:
[PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)

Current standalone Ada-only manuscript fragment:
[PAPER_DRAFT_SM89.md](PAPER_DRAFT_SM89.md)

Current monograph-level synthesis and LaTeX/PGFPlots package:
[MONOGRAPH_SM89_SYNTHESIS.md](MONOGRAPH_SM89_SYNTHESIS.md)
[sm89_monograph.tex](tex/sm89_monograph.tex)
[sm89_monograph.pdf](tex/build/sm89_monograph.pdf)

Current section coverage map:
[PAPER_SECTION_COVERAGE.md](PAPER_SECTION_COVERAGE.md)

## Abstract

We present an open and reproducible SASS reverse-engineering workflow for
current NVIDIA consumer GPUs, with a bounded Ada Lovelace result set already
closed on SM89 and a broader Pascal-vs-Ada comparison scaffolded but not yet
fully populated. Using custom CUDA probes, systematic disassembly, cubin-side
patch validation, runtime differential fuzzing, and tandem `compute-sanitizer`,
`ncu`, and `nsys` passes, we map both broad instruction inventory coverage and
two narrower opcode frontiers: byte-qualified `P2R.B*` and `UPLOP3.LUT`.

The current Ada-only findings are threefold. First, the stable optimized SM89
inventory reaches `379` canonical raw mnemonics, with a lane-specific maximum
of `382`. Second, direct local source/IR still does not emit
`P2R.B1/B2/B3`, but local cubin-side substitution proves that all three are
valid and runnable on the same target, making the remaining gap a form-
selection problem rather than an opcode-existence problem. Third,
`PLOP3 -> UPLOP3` is structurally valid while `ULOP3 -> UPLOP3` is not, and
local `UPLOP3` substitutions partition into inert and semantically live
runtime classes. We frame these results as bounded Ada-only claims that are
already paper-safe, while leaving stronger cross-architecture conclusions for
future Pascal-side measurement.

---

## 1. Introduction

### 1.1 Motivation

NVIDIA's GPU instruction set architecture (ISA) is proprietary and undocumented.
While NVIDIA publishes high-level architecture whitepapers, the actual machine-level
instruction encoding, latencies, and scheduling behavior are not officially specified.
This information is critical for:

- Compiler writers targeting PTX → SASS translation
- Performance engineers hand-tuning GPU kernels  
- Architecture researchers studying microarchitectural evolution
- Students learning about real-world processor design

### 1.2 Scope

We compare two consumer GPUs spanning the Pascal-to-Ada Lovelace evolution:

| Property | GTX 1050 Ti | RTX 4070 Ti Super |
|---|---|---|
| Architecture code | GP107 | AD104 |
| SM version | 6.1 | 8.9 |
| Release year | 2016 | 2024 |
| Process node | 14nm (Samsung) | 5nm (TSMC) |
| CUDA cores | 768 | 8,448 |
| SMs | 6 | 66 |
| Tensor cores | None | 264 (4th gen) |
| RT cores | None | 66 (3rd gen) |
| Memory | 4 GB GDDR5 | 16 GB GDDR6X |
| TDP | 75W | 285W |

These represent six years and five architectural generations (Pascal → Volta → Turing → Ampere → Ada).

### 1.3 Contributions

1. **Open-source SASS reverse engineering toolkit** with 9 probe kernels, latency/throughput
   microbenchmarks, automated disassembly pipeline, and encoding analysis tools
2. **First-party instruction latency measurements** for both Pascal SM 6.1 and Ada SM 8.9
3. **ISA evolution catalog**: instructions added, removed, and modified between the two architectures
4. **Binary encoding format comparison**: instruction word structure, opcode field layout,
   register encoding, and control word changes
5. **Fully reproducible methodology** — all tools, scripts, and raw data published

---

## 2. Background

### 2.1 NVIDIA GPU Architecture Overview

Brief overview of SM structure: warp schedulers, dispatch units, FP32/INT32/SFU/LDST
execution units, register file, shared memory, L1/L2 cache hierarchy.

### 2.2 SASS Instruction Format

- Pascal: 64-bit instruction words with separate control words (1 control word per 3 instructions)
- Ada Lovelace: 128-bit instruction "bundles" with embedded control/scheduling information
- Both architectures use predicated execution (guard predicates @P0–@P6, @!P0–@!P6)

### 2.3 Related Work

- **Jia et al. (2018)**: "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" —
  the gold standard for GPU microbenchmarking methodology; we extend their approach to
  compare two architectures
- **NVIDIA CUDA PTX ISA documentation**: official but intentionally one level above SASS
- **Citadel GPU benchmark suite** and similar microarchitecture probes
- **MaxAs** (Scott Gray): early SASS assembler for Maxwell/Pascal; demonstrated encoding
  structure but focused on a single architecture
- Community efforts on NVIDIA's binary format (CuAssembler, TuringAs)

### 2.4 Challenges

- ptxas optimizer aggressively eliminates benchmark chains (our v1–v3 attempts failed;
  v4 uses volatile global stores to keep chains live)
- Undocumented encoding means bit field positions must be inferred by differential analysis
- Instruction latency is context-dependent (pipeline depth, operand forwarding, value-dependent
  convergence for SFU operations)

---

## 3. Methodology

### 3.1 Measurement Platform

| Component | Details |
|---|---|
| OS | Windows 11 |
| CUDA Toolkit | 13.1 |
| Compiler | nvcc + MSVC v18 (VS2025) |
| Optimization | `-O1` (prevent instruction reordering while allowing basic scheduling) |
| Profiler | Nsight Compute 2025.4.1 (validation) |

### 3.2 Probe Kernels

Nine minimal CUDA kernels, each isolating a specific instruction class:

1. **FP32 arithmetic**: FADD, FMUL, FFMA, FMNMX, FABS/FNEG
2. **Integer arithmetic**: IADD3/IADD, IMAD/XMAD, ISETP, LEA
3. **Special function unit**: MUFU.RCP, .RSQ, .SIN, .COS, .EX2, .LG2, .SQRT
4. **Bitwise/shift**: LOP3/LOP, SHF, PRMT, BFI/BFE, FLO, POPC
5. **Memory**: LDG (32/64/128-bit), STG, LDS, STS, atomics, fences
6. **Conversions**: F2I, I2F, F2F (FP16/FP64), I2I
7. **Control flow**: BRA, BSSY/BSYNC (or SSY/SYNC), divergence, predication
8. **Special registers**: S2R (tid, ctaid, clock, laneid, smid)
9. **Tensor core**: HMMA via WMMA API (Ada only — Pascal lacks tensor cores)

### 3.3 Latency Measurement

- **Chain method**: 512 dependent instructions of the same type
- **Timing**: `clock64` register read before and after chain
- **Anti-optimization**: Initial values loaded from `volatile float*` global memory;
  chain result stored back to prevent dead code elimination
- **Statistical rigor**: 20 repetitions with 1 warmup; report mean cycles/instruction

### 3.4 Throughput Measurement

- **ILP method**: 8 independent accumulator streams per thread
- **Saturation**: 1024 threads per block, enough warps to saturate the SM
- **Timing**: `cudaEventRecord` for wall-clock; normalized to ops/clock/SM

### 3.5 Disassembly and Encoding Analysis

- `cuobjdump -sass` for human-readable SASS
- `nvdisasm -hex -raw` for binary encoding extraction
- `nvdisasm -cfg` for control flow graphs
- Custom Python analysis: opcode field identification via constant-bit masking,
  register field isolation via XOR-diffing same-opcode instructions with different operands,
  control word pattern extraction

---

## 4. Results: ISA Comparison

Current Ada-only claim coverage already available:
- inventory: `C01`, `C02`
- `P2R` frontier status: `C04`-`C08`
- `UPLOP3` structural and runtime status: `C09`-`C17`

Current Ada-only assets available:
- Table A1 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)

### 4.1 Instruction Set Changes

[Table: Instructions unique to each architecture]

Key changes from Pascal to Ada Lovelace:
- **IADD → IADD3**: Pascal uses 2-input IADD; Ada uses 3-input IADD3 (frequently combines
  address calculations that previously required 2 instructions into 1)
- **XMAD → IMAD**: Pascal's XMAD (extended multiply-add, 16×16-bit with mode bits) replaced
  by unified IMAD (32×32-bit integer multiply-add)
- **LOP/LOP32I → LOP3**: Pascal has 2-input logic ops; Ada has 3-input with programmable
  LUT byte, enabling any 3-input Boolean function in a single instruction
- **SSY/SYNC → BSSY/BSYNC**: Divergence handling renamed/redesigned for Independent Thread
  Scheduling (ITS) introduced in Volta
- **New on Ada**: WARPSYNC, MATCH, NANOSLEEP, BAR.ARRIVE, LDGSTS (async global→shared copy),
  HMMA/IMMA (tensor core)
- **Removed from Ada**: XMAD, DEPBAR, TEXDEPBAR

[Detailed delta table to be generated by compare_architectures.py]

### 4.2 Instruction Counts per Probe

[Table: same kernel compiled for both architectures — how many SASS instructions result]

This reveals compiler strategy differences: e.g., does Ada's 3-input IADD3 reduce total
instruction count for address arithmetic?

---

## 5. Results: Latency

### 5.1 FP32 Pipeline

[Table with measured latencies on both GPUs]

### 5.2 Integer Pipeline

[Table]

### 5.3 SFU (MUFU) Pipeline

[Table — particularly interesting because SFU is value-dependent]

### 5.4 Memory Subsystem

[Table: LDG, LDS, shared memory, L1/L2 latencies]

### 5.5 Discussion

[Analysis of changes: What got faster? What stayed the same? What got slower?
Correlate with known microarchitectural changes (wider pipes, deeper caches, etc.)]

---

## 6. Results: Throughput

[Table with measured throughputs on both GPUs, normalized to ops/clk/SM]

### 6.1 FP32 Throughput Scaling

Pascal: 128 FP32 cores / 6 SMs. Ada: 128 FP32 cores / 66 SMs (but each SM has doubled FP32).
How does per-SM throughput compare?

### 6.2 INT32 Throughput

Pascal: INT32 shares the FP32 datapath (can't issue both simultaneously).
Ada: Separate INT32 datapath. Measured concurrent FP32+INT32 throughput.

### 6.3 SFU Throughput

Both architectures have 1 SFU pipe per sub-partition (4 per SM).
Expect similar per-SM throughput; verify with measurements.

---

## 7. Results: Binary Encoding

Current Ada-only frontier claims that can be inserted here now:
- `P2R` bounded-negative/source-vs-cubin claims: `C04`-`C08`
- `UPLOP3` structural boundary claims: `C09`-`C11`

Current Ada-only assets available:
- Table A2 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- Table A3 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)

### 7.1 Instruction Word Format

- **Pascal (SM 6.1)**: 64-bit instructions, grouped in bundles of 3 + 1 control word.
  Control word encodes stall counts, yield hints, read/write barriers for all 3 instructions.
  (Total: 256 bits per 3-instruction bundle = ~85.3 bits/instr effective)

- **Ada (SM 8.9)**: 128-bit per instruction (64-bit instruction word + 64-bit control word).
  Each instruction has its own scheduling info.

### 7.2 Opcode Field Analysis

[Table: Low-bits comparison for same instruction across architectures]

Bit field positions for opcode, register operands, immediate constants, and modifiers.
Differential analysis (XOR between instructions with different register operands but same
opcode) reveals which bit ranges encode register fields.

### 7.3 Control Word / Scheduling

Pascal's compact 21-bit control word (per instruction, packed 3 at a time) vs.
Ada's 64-bit per-instruction control word. What does the extra space encode?

### 7.4 Ada-only frontier status (draft)

On current local SM89, the `P2R` and `UPLOP3` frontiers are now bounded in a
way that is strong enough for declarative paper use, even though neither
frontier is source-level closed in the same way. Table A2 captures the `P2R`
state: direct local source and frontend/IR search still do not emit
`P2R.B1/B2/B3`, despite extensive CUDA-source mutation, PTX search, clang and
Triton frontend variation, and tested `ptxas 11.8/12.6/13.1` sweeps. At the
same time, the local compiler does reproduce the surrounding neighborhood,
including plain `P2R ... 0x7f` and `P2R ... 0x0f`. This combination is the
important result. The failure is no longer best interpreted as a missing opcode
or missing neighborhood. It is better described as a source/IR-level
form-selection problem. Cubin-side substitution then closes the opcode-validity
question directly by materializing and running `P2R.B1`, `P2R.B2`, and
`P2R.B3` on the same local SM89 target.

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

---

## 8. Discussion

Current Ada-only synthesis claims that fit here now:
- live `UPLOP3` site hierarchy: `C12`-`C14`
- tool effectiveness and workflow claims: `C15`-`C16`
- frontier ranking and next-step framing: `C17`-`C18`

Current Ada-only assets available:
- Figure A2 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- Table A4 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- Table A5 in [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)

### 8.1 ISA Design Philosophy Evolution

From "simple instructions, pack more" (Pascal) to "powerful instructions, schedule independently" (Ada).

- 3-input instructions (IADD3, LOP3) reduce instruction count
- Independent Thread Scheduling (BSSY/BSYNC) replaces warp-lockstep SSY/SYNC
- Control word expansion enables finer-grained hardware scheduling

### 8.2 Process Scaling vs. Architectural Improvements

14nm → 5nm provides ~3x density. But many latency improvements are architectural
(deeper forwarding, better caching). Separate the contributions.

### 8.3 What the Encoding Reveals About the Hardware

The instruction encoding is a window into the hardware's internal organization:
- Number of register file read/write ports (inferred from max operands per instruction)
- Scoreboard depth (inferred from barrier bit count in control words)
- Pipeline depth (inferred from stall count range)

### 8.4 Limitations

- Only two GPUs; intermediate architectures (Volta, Turing, Ampere) would show
  the evolution more gradually
- Consumer GPUs may differ from datacenter variants (V100, A100, H100)
- CUDA 13.1's optimizer influences the SASS output — different CUDA versions may
  produce different results for the same kernel
- ptxas may apply transformations not visible at the PTX level
- Single-warp latency measurements don't capture pipeline overlap effects

### 8.5 Ada-only frontier synthesis (draft)

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

---

## 9. Reproducibility

For current Ada-only frontier claims, the bounded evidence source of truth is:
- [PAPER_CLAIMS_MATRIX.md](PAPER_CLAIMS_MATRIX.md)
- [PAPER_FIGURE_TABLE_PLAN.md](PAPER_FIGURE_TABLE_PLAN.md)
- [PAPER_ASSETS_SM89.md](PAPER_ASSETS_SM89.md)
- [PAPER_SECTION_COVERAGE.md](PAPER_SECTION_COVERAGE.md)

All code, scripts, and raw data are available at:
**https://github.com/ismail0098-lang/YSU-engine** (`src/sass_re/`)

### To reproduce:

```powershell
# Ada Lovelace (RTX 4070 Ti Super)
cd src/sass_re
.\scripts\disassemble_all.ps1 -Arch sm_89 -GpuTag Ada_RTX4070TiS
.\scripts\build_and_run_latency.ps1
.\scripts\build_and_run_throughput.ps1

# Pascal (GTX 1050 Ti)
.\scripts\disassemble_all.ps1 -Arch sm_61 -GpuTag Pascal_GTX1050Ti
.\scripts\build_and_run_latency.ps1
.\scripts\build_and_run_throughput.ps1

# Compare
python scripts/compare_architectures.py results/Ada_RTX4070TiS_* results/Pascal_GTX1050Ti_*
```

---

## 10. Conclusion

This repo now supports a bounded but genuinely substantive Ada-only paper story.
On SM89, the broad inventory question is largely closed, while the remaining
interesting work has narrowed to form-selection and semantic-validation
frontiers rather than generic opcode discovery. The `P2R` program established a
clean negative source/IR result and a positive cubin-side opcode-validity
result on the same target, which sharply localizes the remaining open problem.
The `UPLOP3` program went further by establishing a structurally valid cubin-
side path, runtime-safe execution, and a ranked set of semantically live local
sites with distinct causal roles.

Methodologically, the work also clarifies which tools matter at each stage.
Source mutation, PTX/front-end variation, and `ptxas` version sweeps are useful
for bounding negative space. Cubin-side substitution resolves opcode-validity
questions. Differential fuzzing is the strongest semantic discriminator once a
patched form is runnable, with `compute-sanitizer` acting as the safety gate
and `ncu`/`nsys` serving as secondary performance and trace sidecars. The
current paper is therefore best read as an Ada-only evidence-backed frontier
study embedded in a larger Pascal-vs-Ada comparison scaffold that remains ready
for future cross-architecture completion.

---

## References

1. Jia, Z., et al. "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking."
   arXiv:1804.06826, 2018.
2. NVIDIA Corporation. "CUDA C++ Programming Guide." v13.1, 2026.
3. NVIDIA Corporation. "PTX ISA Reference." v8.7, 2026.
4. NVIDIA Corporation. "Ada GPU Architecture Whitepaper." 2022.
5. Gray, S. "MaxAs: SASS Assembler for Maxwell." github.com/NervanaSystems/maxas, 2015.
6. Luo, Y., et al. "CuAssembler: An assembler for NVIDIA CUDA binary." 2021.
7. NVIDIA Corporation. "Pascal Architecture Whitepaper." 2016.
8. Lopes, N., et al. "GPUOcelot: A dynamic compilation framework for GPU computing." 2010.
9. Volkov, V. "Better performance at lower occupancy." GTC 2010.
10. Wong, H., et al. "Demystifying GPU microarchitecture through microbenchmarking." ISPASS 2010.

---

## Appendix A: Measured Data Tables

[Full raw tables to be inserted after 1050 Ti measurement runs]

## Appendix B: Complete Instruction Census

[Per-probe instruction frequency tables for both GPUs]

## Appendix C: Encoding Bit Field Maps

[Visual diagrams of instruction word bit assignments for key opcodes on both architectures]
