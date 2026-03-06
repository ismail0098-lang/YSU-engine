# A Decade of GPU Microarchitecture: Empirical SASS-Level Comparison of Pascal and Ada Lovelace

**Authors:** Umut Korkmaz  
**Affiliation:** Independent Researcher  
**Date:** 2026

---

## Abstract

We present a first-party empirical comparison of NVIDIA's GPU microarchitecture across
two generations separated by six years: Pascal (GTX 1050 Ti, SM 6.1, 2016) and
Ada Lovelace (RTX 4070 Ti Super, SM 8.9, 2022). Using custom CUDA microbenchmarks
and systematic SASS (Streaming ASSembler) disassembly, we measure instruction latencies,
throughputs, and analyze binary encoding evolution for both architectures. Our toolkit is
open-source and fully reproducible. Key findings include [to be filled after 1050 Ti runs]:
instruction set additions/removals, latency improvements, encoding format changes, and
scheduler behavior differences revealed through control word analysis.

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

---

## 8. Discussion

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

---

## 9. Reproducibility

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

We presented the first open-source, side-by-side empirical comparison of NVIDIA GPU
microarchitecture at the SASS instruction level, spanning six years of evolution from
Pascal (SM 6.1) to Ada Lovelace (SM 8.9). Our measurements quantify improvements in
instruction latency, throughput, and ISA expressiveness, while our encoding analysis
reveals how the binary format evolved to support wider scheduling and more powerful
instructions. The complete toolkit is published for the community to extend to additional
GPU generations.

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
