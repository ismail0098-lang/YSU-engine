# SM89 Architecture Reference -- Ada Lovelace

Hardware reference for CUDA LBM kernel optimization on Ada Lovelace (SM 8.9).
Synthesized from NVIDIA whitepapers, first-party SASS RE measurements
(`src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md`), and measured kernel benchmarks (`src/cuda_lbm/README.md`).

---

## 1. Die and SM Configuration

### AD10x Die Family

| SKU | Die | Active SMs | CUDA Cores | L2 Cache (MB) | Peak BW (GB/s) | FP32 TFLOPS | TDP (W) |
|-----|-----|-----------|------------|---------------|----------------|-------------|---------|
| RTX 4090 | AD102 | 128 | 16384 | 72 | 1008 | 82.6 | 450 |
| RTX 4080 Super | AD103 | 80 | 10240 | 64 | 736 | 52.2 | 320 |
| RTX 4070 Ti Super | AD103 | 66 | 8448 | 48 | 504 | 44.1 | 285 |
| RTX 4070 Ti | AD104 | 60 | 7680 | 48 | 504 | 40.1 | 285 |
| RTX 4070 | AD104 | 46 | 5888 | 36 | 504 | 29.1 | 200 |
| RTX 4060 Ti | AD106 | 34 | 4352 | 32 | 288 | 22.1 | 160 |

### Per-SM Structure (SM 8.9)

| Component | Per Sub-Partition | Per SM (4 sub-partitions) |
|-----------|-------------------|---------------------------|
| Warp Scheduler | 1 | 4 |
| FP32 CUDA Cores (primary) | 16 | 64 |
| FP32/INT32 CUDA Cores (secondary) | 16 | 64 |
| Combined FP32 (when no INT32 contention) | 32 | 128 |
| FP64 Cores | 1 (inferred) | 4 (1:32 FP32 ratio) |
| Load/Store Units | 8 | 32 |
| SFU (Special Function Units) | 4 | 16 |
| Tensor Core (4th gen) | 1 | 4 |
| Register File (32-bit regs) | 16384 | 65536 (256 KB) |
| Max Warp Slots | 12 (inferred) | 48 |

---

## 2. Compute Throughput

### Per-SM Peak Throughput

| Precision | Ops/Clock/SM | Peak TFLOPS (RTX 4070 Ti Super @ 2610 MHz boost) |
|-----------|-------------|---------------------------------------------------|
| FP32 (dual-issue) | 128 | 44.1 |
| FP32 (INT32 contention) | 64 | 22.1 |
| FP16 (native scalar) | 128 | 44.1 |
| BF16 (native scalar) | 128 | 44.1 |
| INT32 | 64 | 22.1 |
| FP64 | 4 | 0.69 |
| FP32 Tensor (TF32) | -- | ~165 (theoretical) |
| FP16 Tensor | -- | ~330 (theoretical) |
| INT8 Tensor | -- | ~330 (theoretical TOPS) |

### Measured Tensor Core Throughput (RTX 4070 Ti Super)

From `kernels_tensor_core.cu` WMMA proxy benchmarks:

| Tier | Shape (MxNxK) | Measured | Unit | % of Theoretical |
|------|---------------|----------|------|-------------------|
| TF32 | 16x16x8 | 22880 | GFLOPS | ~14% |
| FP16 | 16x16x16 | 45901 | GFLOPS | ~14% |
| BF16 | 16x16x16 | 45954 | GFLOPS | ~14% |
| INT8 | 16x16x16 | 166189 | TOPS | ~50% |
| INT4 | 8x8x32 | 189103 | TOPS | ~29% |

The ~14% utilization for FP16/BF16/TF32 reflects single-warp-per-block launch
geometry (deliberate -- measures per-warp throughput, not SM saturation).
BF16 parity with FP16 confirmed on Ada (both ~45.9 TFLOPS in proxy).

### FP64 Throughput Limitation

Ada gaming SKUs have a 64:1 FP32:FP64 ratio (4 FP64 cores per SM). At 66 SMs
and 2610 MHz boost: peak FP64 = 66 * 4 * 2 * 2.610 GHz = ~0.69 TFLOPS.

Implication: FP64 and DD (double-double) LBM kernels are COMPUTE-BOUND on Ada
gaming SKUs, not bandwidth-bound. SoA coalescing does not help FP64 performance
(measured: 406 MLUPS SoA vs 461 MLUPS AoS -- AoS wins due to lower instruction
overhead at the compute bottleneck).

---

## 3. Memory Hierarchy

### DRAM (GDDR6X)

| Property | RTX 4070 Ti Super (AD103) |
|----------|---------------------------|
| Type | GDDR6X (Micron) |
| Bus Width | 256-bit |
| Data Rate | 21 Gbps |
| Peak Bandwidth | 504 GB/s |
| VRAM Capacity | 16 GB |
| ECC | None (consumer SKU) |

### L2 Cache

| Property | AD102 | AD103 | AD104 | AD106 |
|----------|-------|-------|-------|-------|
| L2 Size | 72 MB | 48 MB | 36 MB | 32 MB |
| L2 Bandwidth | ~4x DRAM (est.) | ~2016 GB/s (est.) | -- | -- |
| L2 Line Size | 128 bytes | 128 bytes | 128 bytes | 128 bytes |
| L2 Sectors | 32 bytes | 32 bytes | 32 bytes | 32 bytes |

Ada Lovelace introduced L2 residency control via `cudaAccessPolicyWindow`:
allows pinning hot data regions in L2. Useful for auxiliary arrays (tau, force)
that are reused across multiple kernel launches.

### L1 / Shared Memory (per SM)

| Property | Value |
|----------|-------|
| Combined L1 + Shared Memory | 128 KB |
| Max Shared Memory per SM | 100 KB (configurable) |
| Default Shared Memory per SM | 48 KB |
| L1 Cache (when smem = 48 KB) | 80 KB |
| L1 Line Size | 128 bytes |
| L1 Latency | ~33 cycles (inferred from SASS RE) |
| Shared Memory Latency | ~23-30 cycles (measured: 28.03 cyc LDS chase) |
| Shared Memory Banks | 32 banks, 4 bytes/bank |

### Constant Memory

| Property | Value |
|----------|-------|
| Total Constant Memory | 64 KB (global, cached) |
| Constant Cache per SM | 8 KB (inferred) |
| Constant Cache Latency | ~4 cycles (register-like, warp-uniform) |

All LBM kernels store D3Q19 lattice velocities (CX, CY, CZ) and weights (W)
in `__constant__` memory. 19 * 4 * 4 = 304 bytes -- fits easily in constant
cache with no eviction pressure.

### Register File

| Property | Value |
|----------|-------|
| Registers per SM | 65536 x 32-bit (256 KB) |
| Max Registers per Thread | 255 |
| Allocation Granularity | 8 registers per warp |
| Register Latency | 0 cycles (operand collector) |

---

## 4. Measured Instruction Latencies (SASS RE)

First-party measurements from `src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md` on RTX 4070 Ti Super
(SM 8.9, CUDA 13.1). Dependent-chain methodology, 512-deep chains.

### Arithmetic Latencies

| Instruction | Latency (cycles) | Pipeline | Notes |
|-------------|-------------------|----------|-------|
| IADD3 | 2.51 | INT32 | Fastest ALU op; 2-stage pipeline |
| FADD | 4.53 | FP32 | 4-stage FP pipeline |
| FMUL | 4.53 | FP32 | Same as FADD |
| FFMA | 4.54 | FP32 | Workhorse; single-rounding |
| IMAD | 4.52 | INT32 | Integer multiply-add |
| LOP3 | 4.52 | INT32/FP32 | 3-input logic; may dual-issue |
| SHF | 4.55 | INT32 | Funnel shift |
| PRMT | 4.51 | INT32 | Byte permute |

### SFU (MUFU) Latencies

| Instruction | Latency (cycles) | Notes |
|-------------|-------------------|-------|
| MUFU.EX2 | 17.56 | Base-2 exponential |
| MUFU.SIN | 23.51 | Sine approximation |
| MUFU.RSQ | 39.55 | Reciprocal square root |
| MUFU.RCP | 41.55 | Reciprocal (1/x) |
| MUFU.LG2 | 39.55 | Base-2 logarithm |

### Memory Latencies

| Operation | Latency (cycles) | Notes |
|-----------|-------------------|-------|
| LDS chase | 28.03 | Shared memory pointer chase |
| LDG chase | 92.29 | Global memory (L1/L2 hit path) |
| F2I + I2F | 12.05 | Float-to-int round-trip |
| SHFL.BFLY | 24.96 | Warp shuffle (butterfly) |

### LBM Implications

- **BGK collision**: Dominated by FFMA (4.54 cyc). 19 directions * ~5 FMAs
  per direction = ~95 dependent FFMA chains (but most are independent across
  directions). The 1/tau reciprocal (MUFU.RCP, 41.55 cyc) is the single
  longest-latency instruction per cell.
- **MRT collision**: 722 FFMA/cell. The M^-1 * S * M matrix product creates
  longer dependent chains. MUFU.RCP still dominates per-cell latency.
- **Thread coarsening**: float2 (2 cells/thread) hides MUFU.RCP latency by
  interleaving two independent collision chains. float4 pushes register
  pressure toward the 255 limit.

---

## 5. Occupancy Limits and Register-Occupancy Trade-off

### Occupancy Bounds

| Limiter | Max Warps/SM | Max Blocks/SM | Notes |
|---------|-------------|---------------|-------|
| Warp Slots | 48 | -- | Hardware maximum |
| Block Slots | -- | 24 | SM 8.9 limit |
| Registers | 65536 / (regs_per_thread * 32) | -- | Per allocation |
| Shared Memory | 128 KB / smem_per_block | -- | Per allocation |

### Register-Occupancy Table

| Regs/Thread | Warps/SM | Occupancy | Typical Kernel Type |
|-------------|----------|-----------|---------------------|
| 32 | 48 | 100% | Simple init kernels |
| 48 | 42 | 87.5% | Basic BGK SoA |
| 64 | 32 | 66.7% | BGK + Guo forcing |
| 80 | 24 | 50.0% | BGK coarsened (float2) |
| 96 | 20 | 41.7% | MRT collision |
| 128 | 16 | 33.3% | MRT coarsened / tiled |
| 160 | 12 | 25.0% | MRT tiled + Smagorinsky |
| 192 | 10 | 20.8% | DD (double-double) |
| 255 | 8 | 16.7% | Maximum register pressure |

For bandwidth-bound LBM kernels, occupancy below ~33% (16 warps) still
saturates the memory subsystem if each warp issues enough outstanding loads.
The `__launch_bounds__(128, 4)` hint on most kernels targets ~50% occupancy
as a balance between register spilling and latency hiding.

---

## 6. Ada-Specific Features

### FP8 Conversion (SM 8.9+)

Ada Lovelace (SM 8.9) introduced hardware FP8 conversion instructions:
- `__nv_cvt_fp8_to_halfraw()`: FP8 -> FP16 in ~1 cycle
- `__nv_cvt_halfraw_to_fp8()`: FP16 -> FP8 with saturation

Two formats available:
- **E4M3**: Range +-448, ~12.5% relative error. Best for LBM distributions.
- **E5M2**: Range +-57344, ~25% relative error. Use when flow exceeds E4M3 range.

FP8 storage with FP32 compute is the dominant pattern: load FP8, promote to
FP32 via FP16 intermediate, compute BGK/MRT in FP32, demote to FP8, store.

### 4th-Generation Tensor Cores

New on Ada:
- **FP8 Tensor (E4M3/E5M2)**: Available on SM 8.9 for inference workloads
- **Hopper-style TMA**: Not available on Ada (SM 9.0+ only)
- **Sparsity**: 2:4 structured sparsity for Tensor Core ops

Not directly useful for LBM (bandwidth-bound, not compute-bound), but the
proxy benchmark (`kernels_tensor_core.cu`) measures the headroom.

### L2 Residency Control

`cudaAccessPolicyWindow` allows applications to mark memory regions as
"persistent" in L2. For LBM:
- **tau array** (4 bytes * N cells): Good candidate for L2 pinning (read-only,
  reused every step)
- **force array** (12 bytes * N cells): Read-only per step if forces are static
- Distribution arrays: Too large at 128^3+ to benefit from pinning

### Cache-Streaming Stores (__stcs)

PTX `st.global.cs` (cache streaming): Marks writes as low-priority for L2
eviction. Prevents cold output writes from displacing hot input data.

Measured impact (FP32 SoA CS kernel at 128^3): <3% improvement. At 128^3 the
304 MB working set far exceeds the 48 MB L2, so there is no hot data to
protect. Only beneficial at grid sizes where working set fits in L2.

---

## 7. Roofline Model Parameters

### Arithmetic Intensity Breakevens

The roofline ridge point (AI_ridge) is where the kernel transitions from
bandwidth-bound to compute-bound:

```
AI_ridge = Peak FLOPS / Peak BW

RTX 4070 Ti Super:
  FP32:  44.1 TFLOPS / 504 GB/s = 87.5 FLOP/byte
  FP16:  44.1 TFLOPS / 504 GB/s = 87.5 FLOP/byte (scalar FP16)
  FP64:  0.69 TFLOPS / 504 GB/s = 1.37 FLOP/byte
  INT32: 22.1 TFLOPS / 504 GB/s = 43.8 OP/byte
```

### LBM Arithmetic Intensity

```
BGK collision:
  ~95 FMA = ~190 FLOP per cell
  Bytes: 46 * bytes_per_dist (read+write+aux)

  FP32: 190 / (46*4) = 1.03 FLOP/byte  << 87.5 -> BANDWIDTH-BOUND
  FP16: 190 / (46*2) = 2.07 FLOP/byte  << 87.5 -> BANDWIDTH-BOUND
  INT8: 190 / (46*1) = 4.13 FLOP/byte  << 87.5 -> BANDWIDTH-BOUND
  FP64: 190 / (46*8) = 0.52 FLOP/byte  <  1.37 -> BANDWIDTH-BOUND
  DD:   190 / (46*16) = 0.26 FLOP/byte <  1.37 -> COMPUTE-BOUND (DD arithmetic overhead)

MRT collision:
  ~722 FMA = ~1444 FLOP per cell

  FP32: 1444 / (46*4) = 7.85 FLOP/byte << 87.5 -> BANDWIDTH-BOUND
  FP64: 1444 / (46*8) = 3.92 FLOP/byte >  1.37 -> COMPUTE-BOUND
```

All scalar LBM kernels (FP32 and below) are firmly bandwidth-bound on Ada.
FP64 MRT crosses into compute-bound territory. DD is always compute-bound.

---

## 8. L2 Cache Hit Rate vs Grid Size (Critical Section)

### Working Set Sizes per Grid Size

Working set = 19 directions * 2 buffers (ping+pong) * N^3 cells * bytes_per_dist
+ auxiliary arrays (rho, u[3], tau, force[3]) * N^3 * 4 bytes.

Auxiliary = 8 * N^3 * 4 = 32 * N^3 bytes.

| Grid | N^3 Cells | INT8 (1B) | FP16 (2B) | FP32 (4B) | FP64 (8B) | DD (16B) |
|------|-----------|-----------|-----------|-----------|-----------|----------|
| 32^3 | 32768 | 2.4 MB | 3.6 MB | 6.0 MB | 10.8 MB | 20.4 MB |
| 64^3 | 262144 | 19.0 MB | 28.5 MB | 47.5 MB | 86.0 MB | 163.0 MB |
| 128^3 | 2097152 | 152 MB | 228 MB | 380 MB | 688 MB | 1304 MB |
| 256^3 | 16777216 | 1216 MB | 1824 MB | 3040 MB | 5504 MB | -- |

### Bandwidth Regime Classification

RTX 4070 Ti Super L2 = 48 MB.

| Grid | INT8 | FP16 | FP32 | FP64 | Regime |
|------|------|------|------|------|--------|
| 32^3 | 2.4 MB | 3.6 MB | 6.0 MB | 10.8 MB | **L2-resident**: Entire working set fits in L2. Effective BW >> DRAM peak. MLUPS inflated by L2 hit rate. |
| 64^3 | 19.0 MB | 28.5 MB | 47.5 MB | 86.0 MB | **L2-transitional**: INT8/FP16 partially fit; FP32 at boundary; FP64 spills. Mixed hit rate. |
| 128^3 | 152 MB | 228 MB | 380 MB | 688 MB | **GDDR6X-bound**: All tiers exceed L2 capacity and spill to DRAM. L2 acts as throughput amplifier for reuse only. |
| 256^3 | 1216 MB | 1824 MB | 3040 MB | 5504 MB | **GDDR6X-bound**: Pure DRAM streaming. L2 provides zero capacity benefit. |

### L2 Hit Rate Expectations

| Grid | INT8 | FP16 | FP32 | FP64 |
|------|------|------|------|------|
| 32^3 | >95% | >90% | ~80% | ~50% |
| 64^3 | ~60% | ~40% | ~20% | <10% |
| 128^3 | <10% | <5% | <3% | <2% |
| 256^3 | <2% | <1% | <1% | <1% |

### Evidence from Measured Data

- **FP32 SoA CS at 128^3**: 2027 MLUPS, 74.0% BW utilization. The __stcs
  evict-first writes provide <3% gain because the 380 MB working set >> 48 MB
  L2. There is no hot data to protect.
- **FP32 SoA CS at 32^3/64^3**: "Apparent bandwidth exceeds 200% of device
  peak" -- this is L2 reuse artifact, not real DRAM bandwidth. The working set
  fits in L2, so reads are served at L2 bandwidth (~2x DRAM peak).
- **FP32 MRT AA at 64^3**: 5541 MLUPS -- suspiciously high. At 64^3 the FP32
  A-A working set is ~24 MB (single buffer), well within 48 MB L2.

### Benchmark Implications

1. **Baselines MUST be tagged with bandwidth regime**: "L2-bound" at 64^3 vs
   "GDDR6X-bound" at 128^3. Comparing MLUPS across regimes is meaningless.
2. **Regression thresholds differ by regime**:
   - 128^3 (GDDR6X-bound): Stable, repeatable. Threshold: 10%.
   - 64^3 (L2-transitional): L2 hit rate sensitive to co-tenancy. Threshold: 20%.
   - 32^3 (L2-resident): Dominated by kernel launch overhead. Not useful for
     bandwidth benchmarking.
3. **The canonical benchmark grid is 128^3**: All production kernels are
   GDDR6X-bound here. MLUPS is directly proportional to achieved DRAM bandwidth.
4. **64^3 benchmarks measure L2 effectiveness**, not DRAM bandwidth. Useful for
   evaluating cache-streaming stores and L2 residency control, not for
   production MLUPS baselines.

---

## 9. Optimization Implications for LBM Kernels

### Memory Access Patterns

1. **SoA > AoS for all BW-bound tiers**: i-major SoA with pull-scheme achieves
   1.81x-2.22x over AoS at 128^3. The AoS push-scheme scatter-writes to
   non-contiguous addresses; SoA pull-scheme reads from non-contiguous but
   writes to contiguous (coalesced).

2. **Coalescing is everything**: At 128^3, the kernel is DRAM-bound. Every
   non-coalesced access wastes a 128-byte cache line fetch for <32 useful bytes.
   The AoS scatter penalty is the single largest performance cliff in the kernel
   suite.

3. **Vectorized loads matter less than layout**: INT8 SoA (no vectorization)
   beats INT8 AoS (dp4a vectorized) by 2.22x. Layout dominates over ALU tricks.

### Compute Patterns

4. **MUFU.RCP is the critical-path bottleneck**: At 41.55 cycles, the 1/tau
   reciprocal dominates per-cell latency. Thread coarsening (float2, float4)
   hides this by interleaving independent reciprocals.

5. **MRT collision register pressure**: 722 FMA/cell requires ~96-128 registers
   per thread, dropping occupancy to 33-42%. This is acceptable for BW-bound
   kernels -- 16 warps can still saturate the memory subsystem.

6. **INT32 and FP32 contention**: When a kernel mixes INT32 index arithmetic
   (IADD3, IMAD) with FP32 collision (FFMA), the secondary datapath must
   timeshare. Minimizing index computation in the hot loop helps.

### Precision Trade-offs

7. **INT8 SoA is Pareto-optimal**: 5643 MLUPS, 76 MB VRAM, 2.85x FP32.
   Sufficient for qualitative flow visualization and moderate-Re (tau >= 0.6).

8. **FP16 SoA H2 is the precision sweet spot**: 3802 MLUPS with 10-bit
   mantissa. The half2 ILP trick gains +9.8% via Ada dual-issue scheduling.

9. **BF16 is slower than FP16 despite equal size**: 3204 vs 3463 MLUPS.
   BF16 scalar load latency on Ada SM 8.9 is higher than FP16 scalar load
   latency. Use BF16 only when the 8-bit exponent range is needed.

### What NOT to Optimize

10. **Shared memory tiling at 128^3**: The tiled pull-scheme (8x8x4 smem halo)
    does NOT help at 128^3 where the working set >> L2. The smem overhead
    (45.6 KB/block, limiting occupancy) outweighs the halo benefit. Tiling
    only helps at 64^3 where L2 is hot.

11. **Cache-streaming stores at 128^3**: __stcs provides <3% gain. Not worth
    the code complexity.

12. **FP64 SoA optimization**: FP64 is compute-bound on Ada gaming SKUs.
    Memory layout optimization does not help. Use FP64 for validation only.

---

## 10. References

- NVIDIA Ada Lovelace GPU Architecture Whitepaper (2022)
- `docs/sass/NVIDIA_SASS_ADA_LOVELACE_REFERENCE.md` -- Full SASS ISA reference
- `src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md` -- First-party instruction latency/throughput measurements
- `src/cuda_lbm/README.md` -- Measured kernel performance tables
- `src/cuda_lbm/kernels_tensor_core.cu` -- Tensor Core WMMA proxy measurements
- NVIDIA CUDA C++ Programming Guide, Appendix H: Compute Capabilities
- NVIDIA PTX ISA Reference v8.x
