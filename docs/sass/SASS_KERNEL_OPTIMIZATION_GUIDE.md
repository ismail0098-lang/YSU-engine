# SASS-Informed Kernel Optimization Guide

How the SASS reverse engineering measurements in `src/sass_re/` connect to
concrete optimization decisions in `src/cuda_lbm/` kernels.

---

## Latency Budget Per LBM Cell

From `src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md` (RTX 4070 Ti, SM 8.9, ncu-validated):

| Instruction | Latency (cy) | Throughput (ops/clk/SM) | LBM role |
|-------------|-------------|------------------------|----------|
| HFMA2.BF16 | **4.01** | **312.1** | **Fastest FMA on Ada.** BF16 bf162 moment accum. |
| FFMA | 4.53 | 44.6 | Equilibrium, MRT collision, Guo forcing |
| HADD2 | 4.54 | 260.1 | FP16 half2 moment accumulation |
| IDP.4A | 4.53 | 215.2 | INT8 dp4a momentum (4x effective) |
| IADD3 | 2.52 | 68.2 | SoA index arithmetic |
| MUFU.RCP | 41.53 | -- | `1.0f / tau` in BGK and MRT |
| MUFU.EX2 | 17.55 | 9.9 | Instant-NGP volume render (not LBM) |
| POPC | ~7-8 | -- | Box counting ballot reduction (SFU corrected) |
| REDUX.SUM | 60 | -- | Warp-level integer reduction (2.6x faster than SHFL) |
| LDG | 92-123 | -- | Distribution loads (19 per cell) |
| LDS | 28.03 | -- | Tiled kernel shared-memory reads |
| LDC | 70.57 | -- | Constant memory (D3Q19 weights) |

### Key corrections from ncu cross-validation

- **POPC/FLO are ~7-8 cy (multi-cycle INT), NOT 23 cy (SFU).**
  Original measurement had hidden XOR instruction doubling the chain.
- **Denormals are FREE on Ada.** No throughput penalty for denormalized
  float inputs (ratio = 1.00x denormal vs normal).
- **FTZ mode is FREE.** FFMA.FTZ = 4.51 cy = FFMA IEEE. Zero overhead.
- **IABS is sub-cycle (0.26 cy/pair).** Integer abs is a pipeline modifier.
- **Bank conflicts max 2.2x penalty** (not 32x). Hardware coalescing.
- **BF16 is faster than FP16** (312 vs 260 ops/clk/SM throughput,
  4.01 vs 4.54 cy latency, 8.54 vs 10.54 cy conversion).

### BGK collision: ~57 FMA ops/cell

At 4.54 cy/FFMA with no ILP, a single-warp BGK cell takes ~259 cycles of
pure ALU. But LDG latency is 92 cy * 19 loads = ~1750 cycles if serialized.
This is why BGK is *memory-bound* at 128^3: the warp scheduler fills ALU
pipeline bubbles with other warps' memory requests.

### MRT collision: ~722 FMA ops/cell

At 4.54 cy/FFMA, single-warp MRT takes ~3278 cycles of pure ALU. With 4+
warps resident per SM, the scheduler overlaps MRT ALU with memory latency
from other warps. This is why MRT is *free* on Ada (C-1391): the 722 FMA
ops hide within the 19 * 92 = 1748 cycles of memory latency per cell.

**Key insight**: MRT becomes costly only when memory latency drops below
~3278/4 = 820 cycles per cell. This happens at 32^3 (L2-resident, LDG
drops to ~33 cycles, so 19 * 33 = 627 cycles -- MRT can no longer hide).
At 32^3, prefer BGK or coarsened variants.

---

## MUFU.RCP Optimization: The tau Reciprocal

`1.0f / tau` compiles to MUFU.RCP (41.55 cycles). In the MRT collision,
this appears once per cell for `inv_tau = 1.0f / tau_local`. Since MRT
has 722 FMA ops, one 41-cycle RCP is negligible (5.7% of ALU budget).

However, in BGK with only 57 FMA ops, the single RCP is 73% of ALU time.
Pre-computing `inv_tau` as a per-cell field (stored as FP32) would eliminate
this. Trade-off: 4 extra bytes/cell VRAM vs 41 cycles saved. At 128^3:
- Extra VRAM: 2M cells * 4 bytes = 8 MB (vs 304 MB FP32 distributions)
- Cycles saved: 41 / ~259 = ~16% of BGK ALU time

**Recommendation**: Pre-compute `inv_tau` for BGK kernels. Not needed for
MRT (already hidden by 722 FMA).

---

## SoA Index Arithmetic: IADD3 vs IMAD

The SoA access pattern `f[dir * n_cells + cell]` compiles to:
```
IMAD.IADD  R_addr, R_dir, R_n_cells, R_cell    // dir*n_cells + cell
```

IMAD has the same 4.54 cy latency as FFMA but shares the INT32 datapath
(64 ops/clk/SM on Ada vs 128 ops/clk/SM for FP32). With 19 directions,
19 IMAD instructions per cell take ~86 cycles.

**Optimization**: For pull-scheme kernels, the base address
`base = dir * n_cells` is loop-invariant. Hoisting it outside the per-cell
computation reduces 19 IMADs to 19 IADD3s (2.51 cy each = ~48 cycles).
The compiler usually does this, but `asm volatile` in hand-tuned kernels
can prevent it.

---

## LDG Latency Hiding: Why 128 Threads/Block Works

At 128 threads/block = 4 warps/block. With 2-4 blocks/SM (occupancy
depends on register pressure), we get 8-16 warps/SM.

LDG latency: 92 cycles. Warp scheduler needs `ceil(92 / 4.54) = 21` FFMA
instructions from other warps to fully hide one LDG. With 8 warps resident
and ~57 FMA/cell (BGK) or ~722 FMA/cell (MRT), there is ample ALU work
to fill the pipeline.

**Anti-pattern**: Increasing threads/block to 256 or 512 does *not* help
at 128^3. It reduces blocks/SM (register pressure), lowering occupancy
and potentially hurting latency hiding. The measured optimum is 128 tpb.

---

## Tiled Pull-Scheme: When Shared Memory Helps (and When It Hurts)

From `src/cuda_lbm/README.md` (C-1389): Tiled pull-scheme is an
**anti-pattern** on Ada at 128^3.

Why: LDS latency is 28 cycles (from SASS RE) vs LDG at 92 cycles. The
tiled kernel replaces 19 scattered LDG reads with 19 LDS reads, saving
64 cycles per read * 19 = 1216 cycles. But it adds:
- Shared memory halo load phase: 19 LDG reads to fill smem (1748 cy)
- __syncthreads() barrier: ~20 cycles
- Indexing overhead in shared memory: ~19 * 2.51 = 48 cycles

Net savings per cell: 1216 - 48 - 20 = ~1148 cycles.
But the halo phase serializes all threads on LDG, reducing overlap.

At 128^3 (GDDR6X-bound), the L2 hit rate is <10% for FP32, so both
tiled and pull-scheme are DRAM-limited. The extra shared memory overhead
(45.6 KB/block) limits occupancy to 1 block/SM, reducing latency hiding.

**When tiled helps**: 64^3 (L2-transitional). The working set partially
fits in L2, so halo reads often hit L2 (~33 cy) instead of DRAM (92 cy).
At this regime, tiled saves ~19 * (33 - 28) = 95 cycles/cell plus better
L2 utilization from structured access patterns.

---

## Instant-NGP Techniques Applicable to LBM

### 1. 8-wide ILP FFMA chains (MLP kernel, 3.16x speedup)

The MLP kernel proved that maintaining 8 independent FFMA accumulator
chains saturates the FP32 pipeline. In LBM, the MRT collision operator
has natural ILP: the 19 moment transforms are independent until the
final relaxation step. Currently, the compiler serializes some of these.

**Opportunity**: Hand-schedule the MRT moment computation to maintain
6-8 independent FFMA chains. Expected gain: 5-15% on MRT kernels at
grids where the kernel is compute-bound (32^3, 64^3).

### 2. Software pipelining (hash grid, 1.11x speedup)

The hash grid kernel overlaps loads for level N+1 with computation for
level N. In pull-scheme LBM, the 19 direction reads are independent.
Issue all 19 LDG instructions at the top of the cell computation, then
process them as they return.

The compiler already does this for simple pull-scheme kernels. But for
tiled kernels with explicit __syncthreads(), software pipelining across
multiple halo phases could reduce barrier stall time.

### 3. LOP3.LUT for multi-input masking (hash grid, 1.11x)

LOP3 performs arbitrary 3-input boolean logic in one instruction (4.5 cy).
In LBM, bounce-back boundary conditions use:
```
if (solid[neighbor]) f_out[opposite_dir] = f_in[dir];
```
This typically compiles to ISETP + predicated MOV. With packed solid masks
(32 cells per uint32), LOP3 could compute the bounce-back predicate for
an entire warp in one instruction.

### 4. FMNMX for branchless clamping (MLP ReLU, 3.16x)

FMNMX(a, b, predicate) selects min or max without branching. In LBM,
density clamping (`rho = max(rho, rho_floor)`) compiles to ISETP + BRA.
Using FMNMX via `fmaxf()` eliminates the branch penalty. The compiler
usually emits FMNMX for `fmaxf`/`fminf`, but verify in SASS dumps.

---

## Encoding Analysis Implications

From `src/sass_re/results/*/ENCODING_ANALYSIS.md`:

The 64-bit instruction word places the opcode in bits [48:63]. The
control word (second 64-bit word) encodes stall counts in bits [0:3]
and yield hints in bits [4:5].

**Implication for hand-tuned PTX**: When writing inline PTX for LBM
kernels, the assembler's scheduling decisions (stall counts, yield hints)
can be overridden via `.reuse` and `.uniform` modifiers. The SASS RE
data shows that:
- Memory ops (LDG, LDS) should have yield hints set (switch warps during latency)
- FFMA chains should have minimal stall counts (0-1) when independent
- MUFU ops need stall counts >= 4 (pipeline depth of SFU unit)

---

## Priority Optimization Targets (Updated)

Based on the SASS measurements, ncu cross-validation, and the kernel
performance table in `src/cuda_lbm/README.md`:

1. **BF16 SoA bf162 kernel** (IMPLEMENTED: `kernels_bf16_soa_bf162.cu`)
   Uses HFMA2.BF16_V2 packed FMA (fastest FMA on Ada: 4.01 cy, 312 ops/clk/SM).
   76 HFMA2.BF16 instructions for velocity moment accumulation, 90 registers,
   zero spills. Expected +20-25% MLUPS over scalar BF16 SoA based on measured
   throughput advantage. **This is the highest-priority win from SASS RE.**

2. **REDUX.SUM for integer reductions** (IMPLEMENTED: `kernels_box_counting.cu`)
   Replaced ballot+popc with `__reduce_add_sync` (REDUX.SUM.S32) on SM 8.0+.
   Single instruction at 60 cy vs 156 cy SHFL tree (2.6x faster).

3. **INT8 SoA MRT + A-A kernel** (not yet implemented)
   Expected: ~5100 MLUPS (combining INT8 SoA's 5643 MLUPS with A-A's
   VRAM halving). This is the Pareto-optimal production kernel.

4. **FP8 e4m3 SoA MRT + A-A kernel** (not yet implemented)
   Same logic: combine FP8 SoA throughput with A-A VRAM efficiency.

5. **FP16 SoA Half2 MRT variant** (not yet implemented)
   Half2 ILP (+9.8%) combined with MRT stability.

6. **Pre-computed inv_tau field for BGK kernels**
   Eliminates one MUFU.RCP (41 cy) per cell. ~16% BGK ALU reduction.

7. **Warp-level bounce-back via LOP3**
   Packed solid masks + LOP3 for 32-cell-wide boundary detection.

### Findings that change previous assumptions

- **Bank conflicts are cheap**: Max 2.2x penalty (not 32x). Tiled kernels
  are more viable at 64^3 than previously assumed (C-1389).
- **Denormals are free**: No need for FTZ mode in LBM kernels. IEEE
  compliance costs nothing on Ada.
- **BF16 > FP16 for packed FMA**: If the 7-bit mantissa is acceptable,
  BF16 bf162 is strictly faster than FP16 half2 on Ada.
- **FP64 transcendentals are massive**: sin() = 821 cy, log() = 1114 cy.
  Avoid FP64 math functions in any production kernel.

---

## Cross-References

- SASS measurements: `src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md`
- SM 8.9 architecture: `docs/sass/SM89_ARCHITECTURE_REFERENCE.md`
- Instant-NGP optimizations: `src/sass_re/instant_ngp/docs/TECHNICAL_REFERENCE.md`
- Kernel performance table: `src/cuda_lbm/README.md`
- Kernel selector: `src/cuda_lbm/include/lbm_kernel_selector.h`
- Encoding analysis: `src/sass_re/results/*/ENCODING_ANALYSIS.md`
