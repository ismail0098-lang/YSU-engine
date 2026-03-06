# NVIDIA SASS Reference — Ada Lovelace (SM 8.9)

Target: RTX 4070 Ti Super (AD103, compute capability 8.9)

This document is a technical reference for NVIDIA Shader Assembly (SASS) on the Ada Lovelace microarchitecture. It is intended for engine developers writing low-level GPU code and researchers building custom compilers or GPU simulators. Information is derived from `cuobjdump`, `nvdisasm`, NVIDIA whitepapers, PTX ISA documentation, and public reverse-engineering research.

Where exact internal details are not publicly confirmed, inferences are marked as such and are based on observable behavior, patent filings, and continuity with prior architectures (Turing SM 7.5, Ampere SM 8.0/8.6).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [SASS Instruction Reference](#2-sass-instruction-reference)
3. [Predicate and Control Flow](#3-predicate-and-control-flow)
4. [Memory Operations](#4-memory-operations)
5. [PTX to SASS Mapping](#5-ptx-to-sass-mapping)
6. [Reverse Engineering Sources and Methodology](#6-reverse-engineering-sources-and-methodology)

---

## 1. Architecture Overview

### 1.1 SM Layout — Ada Lovelace (SM 8.9)

Each Streaming Multiprocessor (SM) in Ada Lovelace contains **128 CUDA cores** organized into **4 processing blocks** (also called sub-partitions or quadrants). Each processing block contains:

| Component | Per Sub-Partition | Per SM (4 sub-partitions) |
|---|---|---|
| Warp Scheduler | 1 | 4 |
| Dispatch Units | 1 | 4 |
| FP32 CUDA Cores (primary datapath) | 16 | 64 |
| FP32/INT32 CUDA Cores (secondary datapath) | 16 | 64 |
| FP64 Cores | 1 (inferred) | 4 (1:32 ratio) |
| Load/Store Units | 8 | 32 |
| Special Function Units (SFU) | 4 | 16 |
| Tensor Core (4th gen) | 1 | 4 |
| Register File (32-bit) | 16,384 | 65,536 (256 KB) |
| Warp Slots | 12 (inferred) | 48 |

The **RTX 4070 Ti Super** (AD103) has **66 active SMs** of 80 physical SMs on the die.

### 1.2 Warp Schedulers

Each sub-partition has one warp scheduler managing up to 12 resident warps (48 per SM, inferred upper bound — actual limit depends on register and shared memory pressure). Each scheduler:

- Tracks instruction dependencies via a **scoreboard** per warp.
- Issues **one instruction per clock** to its execution pipelines.
- Selects which warp to issue from among its eligible (non-stalled) warps using a **greedy-then-oldest** scheduling policy (inferred from Turing/Ampere continuity; the scheduler picks the warp that has been eligible longest, with tie-breaking favoring instruction-level parallelism).
- Does **not** dual-issue two independent instructions from the same warp in the same clock (dual-issue was removed after Kepler). Each warp gets at most one instruction dispatched per clock per scheduler.

The 4 schedulers operate **independently and in parallel**. In a single clock, the SM can issue up to 4 instructions total (one per scheduler), potentially to 4 different execution pipelines.

### 1.3 Execution Pipelines

Each sub-partition has access to these functional units:

| Pipeline | Width (lanes) | Notes |
|---|---|---|
| **FP32 Primary** | 16 | Executes FP32 arithmetic (FADD, FMUL, FFMA) |
| **FP32/INT32 Secondary** | 16 | Executes FP32 **or** INT32 (IADD3, IMAD, LOP3) per cycle; cannot do both simultaneously in one sub-partition |
| **INT32** | — | Shared with secondary FP32 datapath; when the secondary path executes FP32, INT32 cannot execute on that path in the same cycle |
| **SFU** | 4 | Transcendentals: MUFU (sin, cos, rcp, rsq, exp2, lg2). 4 lanes → a warp takes 8 cycles to retire through SFU |
| **Load/Store** | 8 | LDG, STG, LDS, STS, LDGSTS, atomics. 8 lanes → 4 cycles per warp for a simple load |
| **Tensor Core** | 1 | HMMA, IMMA, DMMA. Matrix ops: 16×16 outer products per clock |
| **FP64** | 1 | DADD, DFMA. 1 lane → 32 cycles per warp; 1:32 FP32 throughput |
| **Uniform Datapath** | — | Warp-uniform operations (ULDC, UIADD3, etc.) execute on a scalar unit, result broadcast to all lanes |

**Ada Lovelace FP32 doubling**: When a warp executes an FP32 instruction, *both* the primary FP32 datapath and the secondary FP32/INT32 datapath can process FP32 work simultaneously, yielding 32 FP32 operations per sub-partition per clock (128 per SM). This only applies when the secondary path is not needed for INT32 work. When a kernel mixes FP32 and INT32, the secondary path must timeshare between them, and effective FP32 throughput drops toward 16 per sub-partition.

### 1.4 Register File

| Property | Value |
|---|---|
| Total registers per SM | 65,536 × 32-bit |
| Total register file size per SM | 256 KB |
| Max registers per thread | 255 |
| Register banking | 4 banks (inferred), 1 read port per bank per cycle per sub-partition |
| Register allocation granularity | Per-warp, in multiples of 8 registers (inferred from occupancy calculator behavior) |

Registers are named `R0`–`R255` in SASS. `RZ` is the zero register (always reads as 0, writes are discarded). Register pairs (`R0`–`R1`) are used for 64-bit operations (FP64, 64-bit integer, 64-bit addresses).

**Bank conflicts**: The register file is banked (likely 4 banks, with register number modulo 4 determining bank assignment). When an instruction's source operands fall in the same bank, a 1-cycle stall is incurred. The compiler aggressively assigns registers to minimize bank conflicts. The operand collector (see below) can buffer operands across multiple cycles to hide single-bank-conflict stalls.

### 1.5 Shared Memory

| Property | Value |
|---|---|
| Combined L1 / Shared Memory per SM | 128 KB |
| Maximum Shared Memory per SM | 100 KB (Ada; configured via `cudaFuncSetAttribute`) |
| Default Shared Memory per SM | 48 KB |
| Shared Memory Banking | 32 banks, 4 bytes per bank |
| Bank access width | 32-bit or 64-bit mode (configurable) |
| Shared memory latency | ~23–30 cycles (inferred from microbenchmarks; varies with bank conflicts) |

Shared memory addresses are 32-bit offsets within the per-block allocation. SASS instructions use `STS` (store shared) and `LDS` (load shared) with byte addressing. Bank conflicts cause serialized accesses at 1 additional cycle per conflict degree. Broadcast occurs when all lanes in a warp access the same address.

### 1.6 Instruction Issue Model

SASS instructions on Ada Lovelace are **128 bits (16 bytes)** wide. Each 128-bit instruction word contains:

- **Opcode and operand fields** (variable bit allocation per instruction class).
- **Stall count** (4 bits): Number of cycles the scheduler should stall *before* issuing the next instruction from the same warp. Range 0–15. This is a compiler-inserted hint/directive, not a hardware interlock override. The hardware scoreboard still enforces true data dependencies regardless of stall count.
- **Yield hint** (1 bit): When set, suggests the scheduler yield to another warp after issuing this instruction, even if the current warp is still eligible. Used to improve latency hiding.
- **Write barrier index** (3 bits): Identifies which of 6 write barriers this instruction's result will signal upon completion.
- **Read barrier mask** (6 bits): Identifies which write barriers must have completed before this instruction can issue.
- **Wait barrier mask** (6 bits): Similar to read barrier, used for memory-ordering synchronization.

The stall/yield/barrier fields occupy the **control word** portion of the 128-bit encoding. In prior architectures (Maxwell/Pascal), control information was packed into a separate "control word" covering 3 instructions. Starting with Volta (SM 7.0), each instruction carries its own control bits inline.

### 1.7 Dependency Resolution

Ada Lovelace uses a **dual dependency tracking** system:

1. **Scoreboard (hardware)**: Tracks register write-after-read (WAR) and read-after-write (RAW) hazards per warp. When an instruction is issued, its destination register is marked "pending" on the scoreboard. Any subsequent instruction reading that register will stall until the write retires. The scoreboard has a fixed number of slots (6 per warp, inferred from the 6 write barriers).

2. **Compiler-inserted barriers (software)**: The `ptxas` compiler statically analyzes dependencies and inserts stall counts and barrier annotations into the instruction stream. This reduces the hardware scoreboard's burden: if the compiler guarantees sufficient distance between a producer and consumer, the scoreboard slot is freed earlier. The 6 write barrier indices (`{0..5}`) plus 6-bit read/wait masks represent the compiler's dependency graph. A `DEPBAR` instruction can explicitly wait on specific barriers.

**Scoreboard stall behavior**: When a warp's next instruction has an unsatisfied dependency, the scheduler skips that warp and attempts to issue from another eligible warp. The stalled warp remains in the scheduler's pool. If no warps are eligible, the sub-partition idles (a "stall cycle"). High occupancy (many warps per SM) hides this latency.

**Reuse flags**: SASS instructions can mark source operands with a "reuse" flag (`.reuse`), indicating the value should be held in the operand reuse cache (a small buffer in the operand collector) rather than re-read from the register file. This reduces register file read port pressure and bank conflict probability.

---

## 2. SASS Instruction Reference

### 2.1 Notation

```
OPCODE{.modifier}  Rd, Ra, Rb, Rc {, Pd}
```

| Symbol | Meaning |
|---|---|
| `Rd` | Destination register (R0–R255, or RZ for discard) |
| `Ra`, `Rb`, `Rc` | Source operands (registers, immediates, or special registers) |
| `Pd`, `Ps` | Predicate destination / source (P0–P6, PT = true) |
| `RZ` | Zero register (reads 0, writes discarded) |
| `PT` | True predicate (always true) |
| `UR0`–`UR63` | Uniform registers (warp-uniform values) |
| `UP0`–`UP6` | Uniform predicates |
| `c[bank][offset]` | Constant memory reference |
| `desc[URx][Ry]` | Descriptor-based memory access |
| `0x1234` | Immediate constant |
| `.reuse` | Operand reuse hint (cached in operand collector) |

### 2.2 Floating-Point Arithmetic

#### FADD — Floating-Point Add

| Field | Detail |
|---|---|
| **Syntax** | `FADD Rd, Ra, Rb ;` |
| **Purpose** | IEEE 754 FP32 addition: `Rd = Ra + Rb` |
| **Operands** | Rd: R, Ra: R or c[][], Rb: R or 20-bit immediate or c[][] |
| **Modifiers** | `.FTZ` (flush denorms to zero), `.SAT` (clamp to [0,1]), `.rounding` (`.RN`/`.RZ`/`.RM`/`.RP`) |
| **Negation** | Source operands can be negated: `FADD Rd, -Ra, Rb` flips the sign bit |
| **Absolute** | Source operands can be absolute-valued: `FADD Rd, |Ra|, Rb` |
| **Pipeline** | FP32 (primary or secondary datapath) |
| **Latency** | 4 cycles (register-to-register, inferred) |
| **Throughput** | 32 ops/clock/sub-partition (both datapaths), 128 ops/clock/SM |
| **Encoding** | 128 bits. Opcode in bits [127:118] (inferred). Ra in [39:32], Rb varies by operand type, Rd in [7:0] |

```
// Example: add two registers
FADD R5, R2, R3 ;

// With flush-to-zero and saturation
FADD.FTZ.SAT R5, R2, R3 ;

// Negate second operand (equivalent to subtraction)
FADD R5, R2, -R3 ;

// Immediate operand (20-bit float, limited precision)
FADD R5, R2, 1.0 ;

// Constant bank operand
FADD R5, R2, c[0x0][0x10] ;
```

#### FMUL — Floating-Point Multiply

| Field | Detail |
|---|---|
| **Syntax** | `FMUL Rd, Ra, Rb ;` |
| **Purpose** | IEEE 754 FP32 multiply: `Rd = Ra × Rb` |
| **Operands** | Rd: R, Ra: R or c[][], Rb: R or 20-bit imm or c[][] |
| **Modifiers** | `.FTZ`, `.SAT`, `.rounding` |
| **Pipeline** | FP32 |
| **Latency** | 4 cycles |
| **Throughput** | 128 ops/clock/SM |

```
FMUL R4, R1, R2 ;
FMUL.FTZ R4, R1, 0.5 ;
```

#### FFMA — Floating-Point Fused Multiply-Add

| Field | Detail |
|---|---|
| **Syntax** | `FFMA Rd, Ra, Rb, Rc ;` |
| **Purpose** | FP32 fused multiply-add: `Rd = Ra × Rb + Rc` with single rounding |
| **Operands** | Rd: R, Ra: R, Rb: R or 20-bit imm or c[][], Rc: R or c[][] |
| **Modifiers** | `.FTZ`, `.SAT`, `.rounding` |
| **Pipeline** | FP32 |
| **Latency** | 4 cycles |
| **Throughput** | 128 ops/clock/SM |
| **Notes** | The compiler's workhorse instruction. Single rounding means `(a*b)+c` is computed with one rounding step, not two, giving better precision than a separate FMUL+FADD sequence. Nearly all FP32 multiply-add patterns in PTX compile to FFMA. When the addend is `RZ`, this is a multiply. When one multiplicand is `1.0`, this is an add. |

```
// Standard FMA
FFMA R5, R1, R2, R3 ;

// FMA as multiply (addend = RZ = 0)
FFMA R5, R1, R2, RZ ;

// FMA as add (multiply by 1.0)
FFMA R5, R1, 1.0, R3 ;

// With negation on the addend (multiply-subtract)
FFMA R5, R1, R2, -R3 ;
```

#### MUFU — Multi-Function Unit (SFU)

| Field | Detail |
|---|---|
| **Syntax** | `MUFU.op Rd, Ra ;` |
| **Purpose** | Transcendental and special functions via the Special Function Unit |
| **Operations** | `.COS` (cosine), `.SIN` (sine), `.EX2` (2^x), `.LG2` (log2), `.RCP` (reciprocal 1/x), `.RSQ` (reciprocal square root 1/sqrt(x)), `.RCP64H` (FP64 reciprocal high), `.RSQ64H` (FP64 rsqrt high), `.SQRT` (square root, Ada+) |
| **Pipeline** | SFU (4 lanes per sub-partition) |
| **Latency** | ~8 cycles (for full warp: 32 lanes / 4 SFU lanes = 8 issue cycles, plus pipeline depth) |
| **Throughput** | 16 ops/clock/SM (4 per sub-partition, one lane per clock per SFU pipe, 8 clocks per warp) |
| **Precision** | Approximately 22-bit mantissa (~6 decimal digits). Not fully IEEE 754. Applications requiring full precision must use software minimax polynomial sequences, not MUFU. |

```
MUFU.RCP R3, R1 ;       // R3 = 1.0 / R1
MUFU.RSQ R3, R1 ;       // R3 = 1.0 / sqrt(R1)
MUFU.EX2 R3, R1 ;       // R3 = 2^R1
MUFU.LG2 R3, R1 ;       // R3 = log2(R1)
MUFU.SIN R3, R1 ;       // R3 = sin(R1)  (input in radians / 2*pi, i.e., range [0,1) maps to [0, 2*pi))
MUFU.COS R3, R1 ;       // R3 = cos(R1)
```

### 2.3 Integer Arithmetic

#### IADD3 — 3-Input Integer Add

| Field | Detail |
|---|---|
| **Syntax** | `IADD3 Rd, Ra, Rb, Rc ;` |
| **Purpose** | 32-bit integer 3-input add: `Rd = Ra + Rb + Rc` |
| **Operands** | Rd: R, Ra: R, Rb: R or 32-bit imm or c[][], Rc: R |
| **Modifiers** | `.X` (extended precision — add carry-in from predicate) |
| **Pipeline** | INT32 (secondary datapath) |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM (secondary datapath only when not doing FP32) |
| **Notes** | Replaces `IADD` from older architectures. The third input enables address arithmetic (base + index + offset) in one instruction. With `.X` modifier and predicate operands, supports multi-word (64-bit, 128-bit) addition chains. |

```
IADD3 R5, R1, R2, RZ ;         // R5 = R1 + R2 + 0 (simple 2-input add)
IADD3 R5, R1, R2, R3 ;         // R5 = R1 + R2 + R3
IADD3 R5, R1, 0x100, RZ ;      // R5 = R1 + 256

// 64-bit add using carry chain:
IADD3   R4, P0, R0, R2, RZ ;   // R4 = lo(R0+R2), P0 = carry
IADD3.X R5, RZ, R1, R3, P0 ;   // R5 = R1 + R3 + carry
```

#### IMAD — Integer Multiply-Add

| Field | Detail |
|---|---|
| **Syntax** | `IMAD Rd, Ra, Rb, Rc ;` |
| **Purpose** | 32-bit integer multiply-add: `Rd = Ra × Rb + Rc` (low 32 bits of result) |
| **Operands** | Rd: R, Ra: R, Rb: R or 32-bit imm or c[][], Rc: R or c[][] |
| **Modifiers** | `.MOV` (move; when Ra=RZ, acts as move of Rc), `.WIDE` (produces 64-bit result in Rd:Rd+1), `.HI` (returns high 32 bits of 64-bit multiply), `.U32` / `.S32` (unsigned/signed) |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles (inferred; `.WIDE` may be higher) |
| **Throughput** | 64 ops/clock/SM |
| **Notes** | The compiler uses `IMAD.MOV` as a general-purpose MOV for integers when it can fold into an IMAD slot. `IMAD.WIDE` is used for 64-bit address computation (pointer arithmetic). |

```
IMAD R5, R1, R2, R3 ;          // R5 = R1 * R2 + R3 (low 32 bits)
IMAD.MOV.U32 R5, RZ, RZ, R3 ;  // R5 = R3 (move)
IMAD.WIDE.U32 R4, R1, R2, R6 ; // R4:R5 = R1 * R2 + R6:R7 (64-bit result)
IMAD.HI.U32 R5, R1, R2, RZ ;   // R5 = hi32(R1 * R2)

// Address calculation: ptr = base + index * stride + offset
IMAD.WIDE.U32 R4, R1, 4, R2 ;  // R4:R5 = R1 * 4 + R2:R3  (array indexing)
```

#### ISETP — Integer Set Predicate

| Field | Detail |
|---|---|
| **Syntax** | `ISETP.cmp.logic Pd, Pt, Ra, Rb, Ps ;` |
| **Purpose** | Compare two integers and set a predicate register |
| **Comparisons** | `.LT`, `.LE`, `.GT`, `.GE`, `.EQ`, `.NE` |
| **Logic** | `.AND`, `.OR`, `.XOR` — combines the comparison result with source predicate `Ps` |
| **Operands** | Pd: predicate dest, Pt: secondary predicate dest (complement), Ra: R, Rb: R or imm or c[][], Ps: predicate source (default `PT` = true) |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM |

```
// Set P0 if R1 < R2
ISETP.LT.AND P0, PT, R1, R2, PT ;

// Set P0 if R1 >= 100
ISETP.GE.AND P0, PT, R1, 0x64, PT ;

// Set P0 if R1 == R2 AND P1 is true
ISETP.EQ.AND P0, PT, R1, R2, P1 ;

// Conditional execution using predicate:
@P0 IADD3 R5, R1, R2, RZ ;     // only executes if P0 is true
```

### 2.4 Bit Manipulation

#### LOP3 — 3-Input Logic Operation

| Field | Detail |
|---|---|
| **Syntax** | `LOP3.LUT Rd, Ra, Rb, Rc, immLUT, Pd ;` |
| **Purpose** | 3-input bitwise logic operation defined by an 8-bit lookup table |
| **LUT encoding** | The 8-bit immediate `immLUT` encodes the truth table for `f(a,b,c)`. Bit `i` of the LUT corresponds to input pattern `(a_bit, b_bit, c_bit)` = binary representation of `i`. For example: AND(a,b) ignoring c uses LUT `0xC0`; OR(a,b) uses `0xFC`; XOR(a,b) uses `0x3C`; a AND (NOT b) uses `0x30`. |
| **Operands** | Rd: R, Ra: R, Rb: R or imm, Rc: R, immLUT: 8-bit immediate |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM |
| **Notes** | Replaces `AND`, `OR`, `XOR`, `NOT` from older ISAs. One LOP3 can express any 3-input boolean function, eliminating multi-instruction sequences. The compiler computes the LUT constant at compile time. |

```
// AND: R5 = R1 & R2  (LUT = 0xC0, Rc = RZ, effectively 2-input)
LOP3.LUT R5, R1, R2, RZ, 0xC0, !PT ;

// OR: R5 = R1 | R2
LOP3.LUT R5, R1, R2, RZ, 0xFC, !PT ;

// XOR: R5 = R1 ^ R2
LOP3.LUT R5, R1, R2, RZ, 0x3C, !PT ;

// Complex: R5 = (R1 & R2) | (~R1 & R3)  — bitwise select / mux
// LUT for (a AND b) OR (NOT a AND c) = 0xCA
LOP3.LUT R5, R1, R2, R3, 0xCA, !PT ;
```

**LUT computation reference**:

| Inputs (a,b,c) | Bit index | Example: a AND b (0xC0) | Example: a XOR b (0x3C) | Example: mux (0xCA) |
|---|---|---|---|---|
| 0,0,0 | 0 | 0 | 0 | 0 |
| 0,0,1 | 1 | 0 | 0 | 1 |
| 0,1,0 | 2 | 0 | 1 | 0 |
| 0,1,1 | 3 | 0 | 1 | 1 |
| 1,0,0 | 4 | 0 | 1 | 0 |
| 1,0,1 | 5 | 0 | 1 | 0 |
| 1,1,0 | 6 | 1 | 0 | 1 |
| 1,1,1 | 7 | 1 | 0 | 1 |

#### SHF — Funnel Shift

| Field | Detail |
|---|---|
| **Syntax** | `SHF.dir.type Rd, Ra, Rb, Rc ;` |
| **Purpose** | Funnel shift: concatenates two 32-bit values and extracts a 32-bit window shifted by a variable or immediate amount |
| **Direction** | `.L` (left), `.R` (right) |
| **Type** | `.U32` (unsigned/logical), `.S32` (signed/arithmetic for right shifts) |
| **Modifiers** | `.W` (wrap shift amount mod 32), `.HI` (high 32 bits of concatenation) |
| **Operands** | Rd: R, Ra: R (low word), Rb: R or 5-bit imm (shift amount), Rc: R (high word) |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM |
| **Notes** | Replaces SHL, SHR from older ISAs. By providing `RZ` as one of the concatenation inputs, standard left/right shifts are achieved. With two non-zero inputs, performs a 64-bit funnel shift in a single instruction. |

```
// Simple left shift: R5 = R1 << 2
SHF.L.U32 R5, R1, 2, RZ ;

// Simple right shift: R5 = R1 >> 4 (logical)
SHF.R.U32 R5, RZ, 4, R1 ;

// Arithmetic right shift: R5 = R1 >> 4 (sign-extending)
SHF.R.S32 R5, RZ, 4, R1 ;

// Funnel shift: extract 32 bits from {R2:R1} >> R3
SHF.R.U32 R5, R1, R3, R2 ;
```

#### PRMT — Permute Bytes

| Field | Detail |
|---|---|
| **Syntax** | `PRMT Rd, Ra, Rb, Rc ;` |
| **Purpose** | Byte permutation: selects 4 bytes from the concatenation of Ra and Rc, using Rb as the selection control |
| **Control** | Each nibble of Rb (4 bits) selects one of 8 source bytes: bytes 0–3 from Ra, bytes 4–7 from Rc. Nibble values 0–7 select the corresponding byte. |
| **Modifiers** | `.F4E` (forward 4-element permute — byte rotate), `.B4E` (backward), `.RC8` (replicate 8-bit), `.ECL` (edge clamp left), `.ECR` (edge clamp right), `.RC16` (replicate 16-bit) |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM |

```
// Extract byte 2 of R1 and broadcast to all bytes of R5
// Control: 0x00000202 means byte 2 for all positions
PRMT R5, R1, 0x2222, RZ ;

// Swap bytes 0 and 3 of R1 (endian swap + middle bytes from R1)
PRMT R5, R1, 0x0123, RZ ;      // identity permutation
PRMT R5, R1, 0x3210, RZ ;      // full byte reverse (endian swap)

// Select bytes from two registers: byte 0 from R1, bytes 1-3 from R3
PRMT R5, R1, 0x6540, R3 ;
```

### 2.5 Data Movement

#### MOV — Move Register

| Field | Detail |
|---|---|
| **Syntax** | `MOV Rd, Ra ;` |
| **Purpose** | Copy value: `Rd = Ra` |
| **Operands** | Rd: R, Ra: R or 32-bit immediate or c[][] |
| **Pipeline** | INT32 |
| **Latency** | 4 cycles |
| **Throughput** | 64 ops/clock/SM |
| **Notes** | In practice, `ptxas` often encodes moves as `IMAD.MOV.U32 Rd, RZ, RZ, Ra` to use the IMAD pipeline, or as `MOV` when a dedicated slot is available. A `MOV R5, 0x3f800000` loads the FP32 representation of 1.0 into R5 as a 32-bit immediate. |

```
MOV R5, R1 ;                    // register-to-register move
MOV R5, 0x100 ;                 // load 32-bit immediate
MOV R5, c[0x0][0x160] ;        // load from constant memory
```

#### MOV (Large Immediate) — via LOP3 or IMAD

For 32-bit immediates that exceed the inline immediate field, the compiler materializes constants using multi-instruction sequences or constant memory loads:

```
// If immediate fits in 20-bit field:
MOV R5, 0x3F ;

// If immediate requires full 32 bits, common pattern:
MOV R5, 0x3f800000 ;           // compiler may use a dedicated MOV encoding
                                // or split into upper/lower halfword loads
```

#### S2R — Special Register to Register

| Field | Detail |
|---|---|
| **Syntax** | `S2R Rd, SR_name ;` |
| **Purpose** | Read a special hardware register into a general register |
| **Common SRs** | `SR_TID.X/Y/Z` (threadIdx), `SR_CTAID.X/Y/Z` (blockIdx), `SR_NTID.X/Y/Z` (blockDim), `SR_LANEID` (lane within warp), `SR_CLOCK` (cycle counter), `SR_GLOBALTIMERLO/HI` (global timer) |
| **Pipeline** | INT32 |
| **Latency** | ~5 cycles (inferred; slightly higher than standard ALU) |

```
S2R R0, SR_TID.X ;             // R0 = threadIdx.x
S2R R1, SR_CTAID.X ;           // R1 = blockIdx.x
S2R R2, SR_LANEID ;            // R2 = lane ID within warp (0..31)
```

#### CS2R — Composite Special Register Read

| Field | Detail |
|---|---|
| **Syntax** | `CS2R Rd, SR_name ;` |
| **Purpose** | Read a 64-bit special register into a register pair `Rd:Rd+1` |
| **Common SRs** | `SR_CLOCKLO` (64-bit cycle counter) |

```
CS2R R2, SR_CLOCKLO ;          // R2:R3 = 64-bit clock counter
```

### 2.6 Floating-Point Comparison

#### FSETP — Floating-Point Set Predicate

| Field | Detail |
|---|---|
| **Syntax** | `FSETP.cmp.logic Pd, Pt, Ra, Rb, Ps ;` |
| **Purpose** | Compare two FP32 values and set predicate |
| **Comparisons** | `.LT`, `.LE`, `.GT`, `.GE`, `.EQ`, `.NE`, `.LTU`, `.LEU`, `.GTU`, `.GEU`, `.EQU`, `.NEU`, `.NUM`, `.NAN` |
| **Logic** | `.AND`, `.OR`, `.XOR` — combines with Ps |
| **Pipeline** | FP32 |
| **Latency** | 4 cycles |
| **Notes** | Unordered variants (`.LTU`, etc.) are true when either operand is NaN. `.NUM` is true when neither is NaN. `.NAN` is true when either is NaN. |

```
FSETP.LT.AND P0, PT, R1, R2, PT ;   // P0 = (R1 < R2)
FSETP.GE.AND P0, PT, R1, 0.0, PT ;  // P0 = (R1 >= 0.0)
```

### 2.7 Conversion

#### F2I — Float to Integer

| Field | Detail |
|---|---|
| **Syntax** | `F2I.rounding.desttype.srctype Rd, Ra ;` |
| **Purpose** | Convert FP32 to integer |
| **Modifiers** | `.TRUNC`/`.FLOOR`/`.CEIL`/`.ROUND` (rounding mode), `.S32`/`.U32`/`.S16`/`.U16`/`.S8`/`.U8` (dest type) |

#### I2F — Integer to Float

| Field | Detail |
|---|---|
| **Syntax** | `I2F.desttype.srctype Rd, Ra ;` |
| **Purpose** | Convert integer to float |

#### F2F — Float to Float

| Field | Detail |
|---|---|
| **Syntax** | `F2F.desttype.srctype Rd, Ra ;` |
| **Purpose** | Convert between float precisions (FP16 ↔ FP32 ↔ FP64) |
| **Modifiers** | `.F16`/`.F32`/`.F64` for source and dest types |

```
F2I.TRUNC.S32.F32 R5, R1 ;     // R5 = (int32_t)truncf(R1)
I2F.F32.S32 R5, R1 ;           // R5 = (float)R1
F2F.F16.F32 R5, R1 ;           // R5 = (half)R1  (stored in low 16 bits)
```

### 2.8 Summary Table — Common Instructions

| Instruction | Purpose | Pipeline | Latency (cycles) | Throughput (ops/clock/SM) |
|---|---|---|---|---|
| `FADD` | FP32 add | FP32 | 4 | 128 |
| `FMUL` | FP32 multiply | FP32 | 4 | 128 |
| `FFMA` | FP32 fused multiply-add | FP32 | 4 | 128 |
| `MUFU` | Transcendental (SFU) | SFU | 8 (warp) | 16 |
| `IADD3` | 3-input INT32 add | INT32 | 4 | 64 |
| `IMAD` | INT32 multiply-add | INT32 | 4 | 64 |
| `ISETP` | Integer compare → predicate | INT32 | 4 | 64 |
| `FSETP` | Float compare → predicate | FP32 | 4 | 128 |
| `LOP3` | 3-input bitwise logic | INT32 | 4 | 64 |
| `SHF` | Funnel shift | INT32 | 4 | 64 |
| `PRMT` | Byte permute | INT32 | 4 | 64 |
| `MOV` | Register move | INT32 | 4 | 64 |
| `S2R` | Special register read | INT32 | ~5 | 64 |
| `LDG` | Global memory load | LD/ST | ~200–400+ | 32 |
| `STG` | Global memory store | LD/ST | ~200–400+ | 32 |
| `LDS` | Shared memory load | LD/ST | ~23–30 | 32 |
| `STS` | Shared memory store | LD/ST | ~23–30 | 32 |
| `BAR` | Barrier synchronization | Special | variable | n/a |
| `BRA` | Branch | Special | 0 (non-divergent) | n/a |
| `DEPBAR` | Dependency barrier wait | Control | 0 (stall) | n/a |
| `NOP` | No operation (encodes stall cycles) | — | per stall count | n/a |

**Latency notes**: ALU latencies are register-to-register pipeline latencies (issue to result available). Memory latencies are highly variable: L1 hit ~30 cycles, L2 hit ~200 cycles, DRAM ~400+ cycles. All latencies are approximate and measured via microbenchmark or inferred from NVIDIA documentation.

---

## 3. Predicate and Control Flow

### 3.1 Predicate Registers

Ada Lovelace provides **7 predicate registers per thread**: `P0`–`P6`, plus the hardwired `PT` (predicate true, always 1). Each predicate is a single bit per lane.

There are additionally **7 uniform predicates** `UP0`–`UP6` (warp-uniform; one bit shared across all lanes).

Predicates are set by comparison instructions (`ISETP`, `FSETP`, `DSETP`, `HSETP2`, `PLOP3`) and consumed by conditional execution prefixes or branch instructions.

### 3.2 Conditional Execution (Predication)

Any SASS instruction can be predicated by prefixing it with `@Pn` or `@!Pn`:

```
@P0  FFMA R5, R1, R2, R3 ;     // executes only in lanes where P0 is true
@!P0 FADD R5, R1, R3 ;         // executes only in lanes where P0 is false
```

Predicated instructions that are "off" in a lane still consume pipeline cycles but do not write their destination register in that lane. The instruction issues for the full warp; the predicate mask suppresses writes on a per-lane basis. This means predication has **zero branch penalty** but consumes execution resources even for inactive lanes.

**When to use predication vs. branching**: Predication is optimal for short conditional sequences (1–3 instructions) where both paths are cheap. For longer sequences, divergent branching is preferable as it avoids executing both paths.

### 3.3 Predicate Logic

#### PLOP3 — Predicate 3-Input Logic

| Field | Detail |
|---|---|
| **Syntax** | `PLOP3.LUT Pd, Pt, Pa, Pb, Pc, immLUT ;` |
| **Purpose** | Combine three predicate inputs using an 8-bit LUT (analogous to LOP3 for predicates) |
| **Notes** | Enables arbitrary boolean combinations of predicates in a single instruction |

```
// P3 = P0 AND P1
PLOP3.LUT P3, PT, P0, P1, PT, 0xC0 ;

// P3 = P0 OR (NOT P1)
PLOP3.LUT P3, PT, P0, P1, PT, 0xF3 ;
```

### 3.4 Branch Instructions

#### BRA — Branch

| Field | Detail |
|---|---|
| **Syntax** | `BRA target ;` or `@Pn BRA target ;` |
| **Purpose** | Unconditional or conditional branch to a PC-relative target |
| **Encoding** | target is a signed 24-bit PC-relative offset (in units of 16 bytes, i.e., instruction-aligned) |
| **Pipeline** | Control flow unit (not a standard ALU pipe) |
| **Latency** | 0 additional cycles for non-divergent warps (branch resolved in issue stage). Divergent branches incur serialization overhead (see below). |

```
BRA 0x1a0 ;                     // unconditional jump
@P0 BRA 0x1a0 ;                 // branch if P0 is true
@!P0 BRA 0x1a0 ;                // branch if P0 is false
```

#### BSSY — Branch Synchronization Barrier Set

| Field | Detail |
|---|---|
| **Syntax** | `BSSY B0, target ;` |
| **Purpose** | Set a synchronization point (reconvergence point) for a subsequent divergent branch. Pushes the reconvergence PC onto the convergence barrier stack indexed by the barrier register (B0–B5). |
| **Notes** | Introduced in Volta. Replaces the pre-Volta implicit stack-based reconvergence (SSY instruction). Ada continues this model. |

#### BSYNC — Branch Synchronization

| Field | Detail |
|---|---|
| **Syntax** | `BSYNC B0 ;` |
| **Purpose** | Wait at the reconvergence point for all diverged lanes to arrive. Pops the convergence barrier. |
| **Notes** | Execution pauses for lanes that arrive early until all lanes in the warp reach this point. |

#### WARPSYNC — Warp Synchronize

| Field | Detail |
|---|---|
| **Syntax** | `WARPSYNC mask ;` |
| **Purpose** | Synchronize a subset of lanes in a warp. `mask` is a 32-bit immediate specifying which lanes must synchronize. Equivalent to PTX `bar.warp.sync`. |

### 3.5 Divergence Handling — Independent Thread Scheduling

Starting with Volta (SM 7.0) and continuing through Ada Lovelace, NVIDIA implements **Independent Thread Scheduling (ITS)**. Each thread in a warp has its own program counter and call stack. This means:

1. **Divergent branches do not require lockstep execution.** When a warp diverges at a conditional branch, the hardware can interleave execution of the two paths rather than strictly serializing them (taken-first, then not-taken).

2. **Reconvergence is explicit**, not implicit. The compiler inserts `BSSY` / `BSYNC` pairs to define reconvergence points. This enables structured divergence where the compiler knows that both paths will merge, as well as unstructured control flow (goto, early return) without corrupting a hardware reconvergence stack.

3. **Starvation freedom**: ITS guarantees forward progress for all threads. One path of a divergent branch cannot indefinitely starve the other. The scheduler round-robins among active thread groups within a warp.

**Performance implications of divergence**:
- A fully divergent warp (16 lanes take path A, 16 take path B) runs both paths, effectively halving throughput for that warp.
- Memory operations from divergent lanes can still coalesce if addresses happen to be adjacent.
- Predication avoids the BSSY/BSYNC overhead but executes all lanes regardless.

**Typical SASS sequence for an if/else**:

```
    ISETP.LT.AND P0, PT, R1, R2, PT ;   // P0 = (R1 < R2)
    BSSY B0, RECONVERGE ;                // set reconvergence point
    @!P0 BRA ELSE_BLOCK ;                // if P0 is false, jump to else

    // --- then block ---
    FFMA R5, R1, R2, R3 ;
    BRA RECONVERGE ;                     // skip else block

ELSE_BLOCK:
    // --- else block ---
    FADD R5, R1, R3 ;

RECONVERGE:
    BSYNC B0 ;                           // all lanes reconverge here
    // --- continuation ---
```

### 3.6 Loops

Loop constructs compile to backward branches. The compiler often places a `BSSY` at the loop header with the convergence point at the loop exit:

```
LOOP_HEADER:
    BSSY B0, LOOP_EXIT ;
    // ... loop body ...
    ISETP.LT.AND P0, PT, R1, R10, PT ;  // loop condition
    @P0 BRA LOOP_HEADER ;               // repeat if condition met
    BSYNC B0 ;
LOOP_EXIT:
```

### 3.7 YIELD and EXIT

| Instruction | Purpose |
|---|---|
| `YIELD` | Hint to scheduler: yield execution to another warp. Does not block; just a scheduling suggestion. |
| `EXIT` | Terminate the current thread. Reduces active lane mask in the warp. When all lanes have exited, the warp retires. |
| `RET` | Return from a function call (pops PC from call stack). |
| `CALL` | Call a device function (pushes PC onto call stack). |
| `BREAK` | Break out of a convergence barrier region (used for loop break/continue). |

---

## 4. Memory Operations

### 4.1 Global Memory

#### LDG — Load Global

| Field | Detail |
|---|---|
| **Syntax** | `LDG.size.cache Rd, [Ra + imm] ;` |
| **Purpose** | Load from global memory into register(s) |
| **Sizes** | `.U8`, `.S8`, `.U16`, `.S16`, `.32`, `.64`, `.128` |
| **Cache** | `.E` (evict-first / streaming), `.EF` (evict-first), `.EL` (evict-last / LRU), `.LU` (last use), `.EU` (evict-first, uncached), `.NA` (non-allocating) — controls L1/L2 caching policy |
| **Addressing** | 64-bit virtual address formed from register pair `[Ra:Ra+1]`, optionally with signed 32-bit immediate offset. For 64-bit addresses: `LDG.E.64 R4, [R2.64 + 0x10]` |
| **Pipeline** | Load/Store unit (8 lanes per sub-partition) |
| **Latency** | L1 hit: ~34 cycles; L2 hit: ~200 cycles; DRAM: ~400-600+ cycles (highly variable with memory controller contention) |
| **Throughput** | 32 bytes/clock/sub-partition for L1 hit (128 bytes/clock/SM); limited by memory bandwidth for L2/DRAM |
| **Coalescing** | The hardware coalesces accesses from all 32 lanes in a warp into as few 128-byte cache line transactions as possible. Ideal access pattern: 32 consecutive 4-byte addresses → 1 transaction. Scattered accesses generate multiple transactions (worst case: 32 transactions for 32 random addresses). |

```
// Load 32-bit from global memory
LDG.E.32 R5, [R2.64] ;

// Load 32-bit with offset
LDG.E.32 R5, [R2.64 + 0x100] ;

// Load 128-bit (4 consecutive registers)
LDG.E.128 R4, [R2.64] ;        // loads R4, R5, R6, R7

// Load with cache evict-first (streaming, don't pollute L1)
LDG.E.EF.32 R5, [R2.64] ;

// Predicated load
@P0 LDG.E.32 R5, [R2.64] ;     // only active lanes perform the load
```

#### STG — Store Global

| Field | Detail |
|---|---|
| **Syntax** | `STG.size.cache [Ra + imm], Rb ;` |
| **Purpose** | Store register(s) to global memory |
| **Sizes** | `.U8`, `.S8`, `.U16`, `.S16`, `.32`, `.64`, `.128` |
| **Cache** | `.E`, `.EF`, `.EL`, `.EU`, `.NA` |
| **Pipeline** | Load/Store unit |
| **Notes** | Stores are buffered in the store queue and do not block the pipeline (fire-and-forget from the thread's perspective). Write-combining may occur in the L2 cache. |

```
STG.E.32 [R2.64], R5 ;                 // store 32-bit
STG.E.128 [R2.64], R4 ;                // store R4,R5,R6,R7 (128 bits)
STG.E.32 [R2.64 + 0x100], R5 ;         // store with offset
```

#### LDGSTS — Asynchronous Global-to-Shared Load

| Field | Detail |
|---|---|
| **Syntax** | `LDGSTS.size [Rshared], [Rglobal.64] ;` |
| **Purpose** | Asynchronous copy from global memory directly to shared memory, bypassing registers. Part of the `cp.async` mechanism introduced in Ampere. |
| **Sizes** | `.32`, `.64`, `.128` |
| **Notes** | Enables pipelining: the copy proceeds in the background while the warp continues executing other instructions. Completion is tracked via `DEPBAR` / `BAR` instructions. Essential for efficient GEMM-style algorithms with multi-stage data loading. |

```
// Async copy 128 bits from global to shared
LDGSTS.E.128 [R0], [R2.64] ;   // R0 = shared mem addr, R2:R3 = global addr
```

### 4.2 Shared Memory

#### LDS — Load Shared

| Field | Detail |
|---|---|
| **Syntax** | `LDS.size Rd, [Ra + imm] ;` |
| **Purpose** | Load from shared memory |
| **Addressing** | 32-bit byte offset in the per-block shared memory allocation |
| **Latency** | ~23–30 cycles (no bank conflict); +1 cycle per degree of bank conflict |
| **Banking** | 32 banks, each 4 bytes wide. Address `addr` maps to bank `(addr / 4) % 32`. If two lanes in a warp access different addresses in the same bank, a bank conflict occurs and accesses are serialized. Identical addresses broadcast (no conflict). |

```
LDS.U.32 R5, [R1] ;            // load 32-bit from shared[R1]
LDS.U.64 R4, [R1] ;            // load 64-bit (R4:R5) from shared[R1]
LDS.U.128 R4, [R1] ;           // load 128-bit from shared[R1]
LDS.U.32 R5, [R1 + 0x40] ;     // load with immediate offset
```

#### STS — Store Shared

| Field | Detail |
|---|---|
| **Syntax** | `STS.size [Ra + imm], Rb ;` |
| **Purpose** | Store to shared memory |
| **Latency** | ~23–30 cycles |

```
STS.32 [R1], R5 ;
STS.128 [R1], R4 ;             // store R4,R5,R6,R7
```

### 4.3 Local Memory

Local memory (per-thread stack/spill) is backed by global (DRAM) but cached in L1. SASS uses `LDL` and `STL`:

```
LDL.32 R5, [R1] ;              // load from local memory
STL.32 [R1], R5 ;              // store to local memory
```

Latency similar to global memory (L1 cached ~34 cycles, L2 ~200 cycles). Spills to local memory are a major performance hazard — high register pressure causes `ptxas` to spill, and each spill adds a load/store to the instruction stream.

### 4.4 Constant Memory

Constant memory is read-only, cached in a dedicated constant cache per SM:

```
// Constant memory is addressed as c[bank][byte_offset]
// Bank 0 (c[0x0]) is kernel arguments
// Bank 2 (c[0x2]) is user __constant__ variables

MOV R5, c[0x0][0x160] ;        // load kernel argument at byte offset 0x160
FADD R5, R1, c[0x0][0x160] ;   // use constant memory directly as an operand

// Uniform register load from constant memory:
ULDC.64 UR4, c[0x0][0x118] ;   // load 64-bit constant into uniform register pair
```

| Property | Value |
|---|---|
| Constant cache per SM | 64 KB (inferred; dedicated cache, separate from L1) |
| Access latency (cache hit) | ~4 cycles (broadcast to all lanes) |
| Access pattern | All lanes in a warp **must** access the same address. If lanes access different constant addresses, the accesses are serialized (worst case: 32 serial reads). This is by design — constant memory is for warp-uniform data. |

### 4.5 Texture and Surface Memory

Texture loads via `TLD` / `TLD4` / `TEX` instructions route through the texture cache and **texture mapping units (TMU)**. These are separate from the standard LD/ST path:

```
TEX.1D.R32F R5, R1, t0, s0 ;   // pseudo-syntax; actual SASS is opaque descriptor-based
```

Texture operations are outside the scope of general compute SASS but are used by the graphics pipeline and `tex1D`/`tex2D` CUDA intrinsics.

### 4.6 Atomic Operations

#### ATOMS — Shared Memory Atomic

```
ATOMS.ADD.32 R5, [R1], R2 ;    // shared[R1] += R2; R5 = old value
ATOMS.CAS.32 R5, [R1], R2, R3; // CAS: if shared[R1]==R2, shared[R1]=R3; R5=old
ATOMS.EXCH.32 R5, [R1], R2 ;   // exchange
ATOMS.MIN.S32 R5, [R1], R2 ;   // signed min
ATOMS.MAX.U32 R5, [R1], R2 ;   // unsigned max
```

#### ATOMG — Global Memory Atomic

```
ATOMG.E.ADD.32 R5, [R2.64], R3 ;       // global atomic add
ATOMG.E.CAS.32 R5, [R2.64], R3, R4 ;   // global CAS
ATOMG.E.ADD.F32 R5, [R2.64], R3 ;      // global atomic float add (hardware-accelerated on Ada)
```

Ada Lovelace has hardware-accelerated FP32 atomicAdd via the L2 cache (introduced in Turing for FP16/BF16, extended to FP32 in Ada).

### 4.7 Memory Fences and Barriers

#### BAR — Barrier Synchronization

| Field | Detail |
|---|---|
| **Syntax** | `BAR.SYNC barrier_id ;` |
| **Purpose** | Block synchronization — all threads in the block must reach this barrier before any thread proceeds. PTX equivalent: `bar.sync`. |
| **barrier_id** | Integer 0–15. Barrier 0 is `__syncthreads()`. |

```
BAR.SYNC 0x0 ;                 // __syncthreads()
```

#### MEMBAR — Memory Barrier

```
MEMBAR.CTA ;                    // memory fence: all prior memory operations from this thread are
                                // visible to all threads in the same CTA before subsequent ops
MEMBAR.GPU ;                    // fence visible to all threads on the GPU
MEMBAR.SYS ;                    // fence visible system-wide (including CPU / other GPUs)
```

#### DEPBAR — Dependency Barrier

```
DEPBAR.LE SB0, 0x0 ;           // wait until all dependencies on scoreboard barrier 0
                                // have at most 0 outstanding. Used for async copy completion.
```

### 4.8 Addressing Modes and Alignment

| Pattern | Encoding | Notes |
|---|---|---|
| Register | `[Ra.64]` | 64-bit address from register pair |
| Register + Immediate | `[Ra.64 + 0x100]` | Signed 32-bit immediate offset |
| Shared (register) | `[Ra]` | 32-bit shared memory offset |
| Shared (register + imm) | `[Ra + 0x40]` | 32-bit offset + immediate |
| Constant | `c[bank][offset]` | Bank index + byte offset (both immediate) |
| Uniform register | `desc[URx][Ry]` | Descriptor-based (texture/surface) |

**Alignment constraints**:

| Access Size | Required Alignment |
|---|---|
| 8-bit | 1 byte |
| 16-bit | 2 bytes |
| 32-bit | 4 bytes |
| 64-bit | 8 bytes |
| 128-bit | 16 bytes |

Misaligned accesses are not hardware-faulted on Ada but may decompose into multiple smaller transactions, reducing performance. The compiler ensures proper alignment for statically-known access patterns. Dynamic misalignment (e.g., from casting pointer types) silently degrades performance.

---

## 5. PTX to SASS Mapping

### 5.1 Compiler Pipeline

```
CUDA C/C++  →  nvcc (frontend)  →  PTX (virtual ISA)  →  ptxas (backend)  →  SASS (machine code)
```

PTX is a stable virtual ISA. SASS is the actual hardware instruction set, specific to each GPU architecture. `ptxas` performs:

1. **Register allocation** (graph-coloring allocator with heuristic spill decisions)
2. **Instruction selection** (PTX → SASS opcode mapping, often 1:N expansion)
3. **Instruction scheduling** (latency-aware, scoreboard-aware ordering)
4. **Dependency annotation** (stall counts, yield hints, barrier assignments)
5. **Peephole optimization** (strength reduction, constant folding, dead code elimination)

### 5.2 Mapping Examples

#### Simple FP32 Add

**PTX:**
```
add.f32 %f3, %f1, %f2;
```

**SASS (possible outputs):**
```
// Direct mapping:
FADD R3, R1, R2 ;

// If compiler can fold into a multiply-add chain:
FFMA R3, R1, 1.0, R2 ;     // equivalent: R1 * 1.0 + R2

// If one operand is already needed for an FMA:
// (may be absorbed into a surrounding FFMA)
```

The compiler prefers `FFMA` over separate `FADD`/`FMUL` whenever a multiply-add pattern exists, even generating `FFMA R, R, 1.0, R` for standalone adds when it improves scheduling.

#### FP32 Multiply-Add

**PTX:**
```
fma.rn.f32 %f4, %f1, %f2, %f3;
```

**SASS:**
```
FFMA R4, R1, R2, R3 ;
```

Direct 1:1 mapping. `ptxas` preserves the fused semantics (single rounding).

#### Integer Multiply-Add with 64-bit Result

**PTX:**
```
mad.wide.u32 %rd3, %r1, %r2, %rd1;
```

**SASS:**
```
IMAD.WIDE.U32 R6, R1, R2, R4 ;    // R6:R7 = R1 * R2 + R4:R5
```

#### Global Memory Load

**PTX:**
```
ld.global.f32 %f1, [%rd1];
```

**SASS:**
```
LDG.E.32 R1, [R2.64] ;            // R2:R3 holds the 64-bit pointer %rd1
```

**PTX (vectorized):**
```
ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
```

**SASS:**
```
LDG.E.128 R4, [R2.64] ;           // loads R4, R5, R6, R7
```

The compiler coalesces four 32-bit loads into a single 128-bit transaction.

#### Global Memory Store

**PTX:**
```
st.global.f32 [%rd1], %f1;
```

**SASS:**
```
STG.E.32 [R2.64], R1 ;
```

#### Shared Memory Load/Store

**PTX:**
```
ld.shared.f32 %f1, [%r1];
st.shared.f32 [%r1], %f2;
```

**SASS:**
```
LDS.U.32 R1, [R0] ;
STS.32 [R0], R2 ;
```

#### Predicated If/Else

**PTX:**
```
setp.lt.f32 %p1, %f1, %f2;
@%p1 add.f32 %f3, %f1, %f2;
@!%p1 mul.f32 %f3, %f1, %f2;
```

**SASS (short sequence — predication):**
```
FSETP.LT.AND P0, PT, R1, R2, PT ;
@P0  FADD R3, R1, R2 ;
@!P0 FMUL R3, R1, R2 ;
```

For short if/else, the compiler prefers predication (no branch overhead). Both instructions issue unconditionally; the predicate mask suppresses the write in inactive lanes.

**SASS (long sequence — divergent branch):**
```
    FSETP.LT.AND P0, PT, R1, R2, PT ;
    BSSY B0, RECONVERGE ;
    @!P0 BRA ELSE_BLOCK ;

    // then-block (many instructions)
    ...
    BRA RECONVERGE ;

ELSE_BLOCK:
    // else-block (many instructions)
    ...

RECONVERGE:
    BSYNC B0 ;
```

#### Reduction to Thread ID Calculation

**PTX:**
```
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.u32 %r4, %r2, %r3, %r1;  // globalIdx = blockIdx.x * blockDim.x + threadIdx.x
```

**SASS:**
```
S2R R0, SR_TID.X ;
S2R R1, SR_CTAID.X ;
S2R R2, SR_NTID.X ;
IMAD R3, R1, R2, R0 ;            // R3 = blockIdx.x * blockDim.x + threadIdx.x
```

Or, if `blockDim.x` is a compile-time constant (e.g., 256):

```
S2R R0, SR_TID.X ;
S2R R1, SR_CTAID.X ;
IMAD R3, R1, 0x100, R0 ;         // R3 = blockIdx.x * 256 + threadIdx.x
                                   // compiler replaces ntid.x with immediate
```

#### Transcendental Functions

**PTX:**
```
rcp.approx.f32 %f2, %f1;
```

**SASS:**
```
MUFU.RCP R2, R1 ;
```

**PTX (full precision, compiler-generated minimax polynomial):**
```
rcp.rn.f32 %f2, %f1;
```

**SASS (expanded — compiler generates a Newton-Raphson refinement):**
```
MUFU.RCP R3, R1 ;                //  R3 ≈ 1/R1 (approx, ~22-bit mantissa)
FFMA R4, R1, -R3, 1.0 ;          //  R4 = 1.0 - R1 * R3  (error term)
FFMA R2, R3, R4, R3 ;            //  R2 = R3 + R3 * R4    (Newton-Raphson step)
```

The compiler inserts the refinement to achieve full FP32 precision (23-bit mantissa) from the SFU's approximate result.

#### Loop with Adaptive Exit

**PTX:**
```
loop:
    add.f32 %f1, %f1, %f2;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, %r10;
    @%p1 bra loop;
```

**SASS:**
```
LOOP:
    FADD R1, R1, R2 ;
    IADD3 R3, R3, 0x1, RZ ;
    ISETP.LT.AND P0, PT, R3, R10, PT ;
    @P0 BRA LOOP ;
```

Note: the compiler may unroll the loop by a factor of 2–8 and software-pipeline the iterations, interleaving loads from iteration N+1 with computation from iteration N to hide memory latency.

### 5.3 Compiler Optimizations Visible in SASS

| Optimization | Description | SASS Signature |
|---|---|---|
| **FMA folding** | Separate multiply and add merged into single FFMA | `FFMA` where PTX had `mul` + `add` |
| **Constant folding** | Expressions with compile-time constants pre-evaluated | Immediates in instructions, absent arithmetic |
| **Strength reduction** | Multiply by power-of-2 → shift | `SHF.L` instead of `IMAD` |
| **Dead code elimination** | Unused computations removed | Fewer instructions than PTX |
| **Register coalescing** | MOVs eliminated by allocating source and dest to same register | Absent MOV instructions |
| **Instruction reordering** | Independent instructions interleaved to hide latency | Load issued many instructions before the first use of its result |
| **Loop unrolling** | Small loops expanded N times | Duplicated instruction sequences without branch |
| **Software pipelining** | Loads from next iteration issued before current iteration completes | LDG interspersed with computation for previous data |
| **Operand reuse** | Frequently-read operands cached in the operand reuse buffer | `.reuse` suffix on source operands |
| **Predication** | Short if/else converted to predicated instructions | `@P0` / `@!P0` prefixed instructions instead of BSSY/BRA/BSYNC |
| **Address arithmetic fusion** | Base + index * stride folded into IMAD.WIDE | Single `IMAD.WIDE` for pointer computation |
| **Uniform hoisting** | Warp-uniform computations moved to the uniform datapath | `UIADD3`, `UIMAD`, `ULDC` prefixed instructions |

---

## 6. Reverse Engineering Sources and Methodology

### 6.1 Primary Tools

#### cuobjdump

Extracts embedded SASS or PTX from compiled `.cubin` / `.fatbin` / `.o` files:

```bash
# Dump SASS assembly
cuobjdump --dump-sass my_kernel.cubin

# Dump PTX
cuobjdump --dump-ptx my_kernel.cubin

# List all kernels in a binary
cuobjdump --list-elf my_kernel.cubin
```

#### nvdisasm

Disassembles raw SASS binary into human-readable assembly. More detailed than `cuobjdump`:

```bash
# Basic disassembly
nvdisasm my_kernel.cubin

# With control flow graph
nvdisasm --print-cfg my_kernel.cubin > cfg.dot

# With register liveness analysis
nvdisasm --print-life-ranges my_kernel.cubin

# With instruction encoding (hex dump alongside assembly)
nvdisasm --print-code my_kernel.cubin

# Filter to specific kernel
nvdisasm --function my_kernel_name my_kernel.cubin
```

`nvdisasm --print-code` is the most useful for encoding analysis — it shows the 128-bit hex encoding alongside each instruction, enabling byte-level opcode extraction.

#### cuasm / CuAssembler

Third-party SASS assembler (open source) that can modify individual instructions in compiled cubins. Useful for:
- Testing instruction encoding hypotheses
- Microbenchmarking specific instruction sequences
- Binary patching for performance experiments

### 6.2 Microbenchmarking Methodology

To determine instruction latency and throughput:

1. **Latency measurement**: Create a dependent chain of N identical instructions (e.g., `FFMA R0, R0, R1, R0` repeated 1000 times). Measure total cycles via `clock64()` or `RDTSC`-equivalent (`CS2R SR_CLOCKLO`). Divide by N.

2. **Throughput measurement**: Create N independent identical instructions (different destination registers, no dependencies). Measure total cycles. Throughput = N / cycles.

3. **Memory latency**: Pointer-chasing (linked list traversal) through global memory to measure L1/L2/DRAM latencies, defeating prefetchers.

4. **Shared memory banking**: Access patterns designed to create 2-way, 4-way, ..., 32-way bank conflicts, measuring cycle penalty per conflict degree.

### 6.3 Encoding Inference

SASS instruction encoding is not publicly documented by NVIDIA. The following is inferred from `nvdisasm --print-code` output analysis and continuity with prior architectures:

**128-bit instruction word layout (Ada / SM 8.9, inferred)**:

```
[127:118]  Opcode (10–12 bits, exact width varies by instruction class)
[117:112]  Modifier / subopcode bits
[111:106]  Barrier / scheduling control (stall count, yield, read/write barriers)
[105:96]   Additional modifiers, predicate source, reuse flags
[95:64]    Source operand C / immediate field (32 bits for full immediates)
[63:56]    Source operand B
[55:48]    Source operand A
[47:40]    Predicate target / secondary destination
[39:32]    Destination register
[31:24]    Predicate guard (@Pn prefix)
[23:0]     Extended encoding / condition codes / type specifiers
```

**Caveats**: Bit positions are approximate and instruction-class-dependent. The encoding is not fully orthogonal — different instruction classes (ALU, memory, control flow) use overlapping bit fields for different purposes. The above layout is a composite approximation.

**Opcode space partitioning (representative, inferred)**:

| Opcode Range (approximate) | Instruction Class |
|---|---|
| `0x000–0x0FF` | FP32 arithmetic (FADD, FMUL, FFMA, FMNMX, ...) |
| `0x100–0x1FF` | FP64 arithmetic (DADD, DMUL, DFMA, ...) |
| `0x200–0x2FF` | Integer arithmetic (IADD3, IMAD, ISETP, ...) |
| `0x300–0x3FF` | Bit manipulation (LOP3, SHF, PRMT, BFI, ...) |
| `0x400–0x4FF` | Conversion (F2I, I2F, F2F, ...) |
| `0x500–0x5FF` | Load/Store (LDG, STG, LDS, STS, LDL, STL, ...) |
| `0x600–0x6FF` | Texture / Surface |
| `0x700–0x7FF` | Special (MUFU, S2R, BAR, VOTE, SHFL, ...) |
| `0x800–0x8FF` | Tensor Core (HMMA, IMMA, ...) |
| `0x900–0x9FF` | Control flow (BRA, BSSY, BSYNC, EXIT, ...) |

### 6.4 Key Public References

1. **NVIDIA Ada Lovelace Architecture Whitepaper** (2022): SM layout, cache hierarchy, CUDA core counts, Tensor Core gen-4 specs.

2. **NVIDIA PTX ISA Reference** (latest version): Documents the PTX virtual ISA. While not SASS directly, the PTX → SASS mapping is largely mechanical, and PTX documents semantics that SASS implements. Available at NVIDIA's developer documentation site.

3. **NVIDIA CUDA Binary Utilities** (cuobjdump, nvdisasm): Official tools for disassembling SASS. Man pages document output format.

4. **CuAssembler** (GitHub: cloudcores/CuAssembler): Open-source SASS assembler with partial encoding documentation for Maxwell through Ampere. Ada encoding follows the same general structure.

5. **Scott Gray's MaxAs** (GitHub: NervanaSystems/maxas): The original Maxwell SASS assembler. While dated (Maxwell era), it established the instruction encoding analysis methodology used by subsequent projects.

6. **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking** (Jia et al., 2018): Detailed latency and throughput measurements for Volta, establishing the microbenchmarking methodology. Many structural parameters (scoreboard depth, register bank count, issue model) carry forward to Ada.

7. **Dissecting the Ampere GPU Architecture through Microbenchmarking** (Jia et al., 2021): Updated measurements for Ampere (SM 8.0), directly preceding Ada.

8. **NVIDIA Patent Filings**: US patents on warp scheduling, independent thread scheduling, operand collector design, and memory hierarchy behavior provide architectural insights that complement empirical measurements.

### 6.5 Verification Approach

When using this reference:

1. **Compile a minimal kernel** targeting `sm_89` (Ada):
   ```bash
   nvcc -arch=sm_89 -cubin kernel.cu -o kernel.cubin
   ```

2. **Disassemble**:
   ```bash
   nvdisasm --print-code kernel.cubin
   ```

3. **Compare** the disassembly against this document. Instruction mnemonics should match. Encoding details may shift between CUDA toolkit versions as NVIDIA updates `ptxas` without changing the ISA.

4. **Benchmark** critical instruction sequences using the microbenchmarking methodology described above to verify latency/throughput numbers for your specific workload.

---

## Appendix A: Register Usage Conventions

The `ptxas` register allocator follows these observable conventions:

| Register | Common Usage |
|---|---|
| `R0`–`R3` | Thread identity (threadIdx, blockIdx — from S2R early in kernel) |
| `R4`–`Rn` | General computation (allocated by graph coloring) |
| `R255` | Rarely used (high register pressure indicator) |
| `RZ` | Zero register (hardwired) |

The number of registers per thread is reported by `cuobjdump --dump-resource-usage` and directly impacts occupancy. Ada Lovelace has 65,536 registers per SM; with 48 warps × 32 threads = 1,536 potential threads, the maximum is 42 registers/thread at full occupancy. In practice, `ptxas` often uses 32–128 registers per thread, trading occupancy for ILP.

## Appendix B: Uniform Datapath

Ada Lovelace continues the Turing/Ampere **uniform datapath** for warp-uniform operations. Instructions prefixed with `U` operate on uniform registers (`UR0`–`UR63`) and execute once per warp on a scalar unit:

```
ULDC.64 UR4, c[0x0][0x118] ;    // uniform load from constant memory
UIADD3 UR6, UR4, UR5, URZ ;     // uniform integer add
UIMAD UR8, UR6, UR7, URZ ;      // uniform integer multiply-add
UISETP.LT.AND UP0, UPT, UR4, UR5, UPT ;  // uniform predicate set
UMOV UR10, 0x100 ;               // uniform move immediate
```

Benefits:
- Saves register file bandwidth — one read serves all 32 lanes.
- Saves ALU cycles — one execution instead of 32 redundant lane executions.
- Reduces power consumption for warp-uniform control flow, addressing, and constants.

The compiler identifies warp-uniform values (kernel parameters, blockIdx-derived constants, loop bounds) and routes them to the uniform datapath automatically.

## Appendix C: Warp Shuffle and Vote

#### SHFL — Warp Shuffle

| Field | Detail |
|---|---|
| **Syntax** | `SHFL.mode Rd, Pd, Ra, Rb, Rc ;` |
| **Purpose** | Exchange register values between lanes within a warp without shared memory |
| **Modes** | `.IDX` (indexed), `.UP` (delta decrease), `.DOWN` (delta increase), `.BFLY` (butterfly / XOR) |
| **Operands** | Ra: source value, Rb: source lane / delta, Rc: clamp mask |

```
SHFL.IDX R1, P0, R0, R2, 0x1f ;    // R1 = value of R0 from lane R2 (indexed)
SHFL.BFLY R1, P0, R0, 0x1, 0x1f ;  // R1 = R0 from lane (laneid ^ 1) (butterfly)
SHFL.DOWN R1, P0, R0, 0x1, 0x1f ;  // R1 = R0 from lane (laneid + 1)
```

Warp shuffles have ~4 cycle latency and enable efficient reductions, scans, and inter-lane communication without shared memory round-trips.

#### VOTE — Warp Vote

```
VOTE.ALL Rd, Pd, Ps ;          // Rd = ballot where all lanes have Ps true
VOTE.ANY Rd, Pd, Ps ;          // Rd = ballot where any lane has Ps true
VOTE.UNI Rd, Pd, Ps ;          // Rd = 1 if all active lanes have same Ps value
```

`VOTE` returns a 32-bit ballot mask (one bit per lane) in `Rd` and optionally sets a predicate `Pd`.

---

*End of reference. This document reflects publicly available and inferable information as of early 2026. Exact internal encoding details are NVIDIA proprietary and subject to change between CUDA toolkit versions. When precision matters, verify against nvdisasm output for your specific toolkit/driver combination.*
