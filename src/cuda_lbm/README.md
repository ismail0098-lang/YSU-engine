# cuda_lbm -- D3Q19 Lattice Boltzmann GPU Kernels

Precision-tier CUDA kernels implementing the D3Q19 Lattice Boltzmann Method (LBM)
for YSU fluid simulation and physics subsystems. All kernels follow the
storage-compute split pattern: distributions are stored in a compressed precision
format, but arithmetic is promoted to FP32 before any computation.

Two layout families exist:

- **AoS ping-pong**: `f[cell * stride + dir]`, stride padded to 20 for alignment.
  Push-scheme scatter-writes; diagonal directions are non-coalesced.
- **i-major SoA pull-scheme**: `f[dir * n_cells + cell]`, stride = 19 (no padding).
  Pull gather-reads from backward neighbor; writes to own cell = fully coalesced.

---

## Kernel File Index

| File                        | Storage              | Bytes/dist | Layout        | Key trick                                          | Min arch |
|-----------------------------|----------------------|------------|---------------|----------------------------------------------------|----------|
| kernels_fp16.cu             | `__half`             | 2          | AoS stride=20 | 10x half2 vectorized loads (padded stride)         | SM 7.0   |
| kernels_fp8.cu              | FP8 e4m3             | 1          | AoS stride=20 | 5x uchar4 vectorized loads; e4m3 range +-448       | SM 8.9   |
| kernels_fp8_e5m2.cu         | FP8 e5m2             | 1          | AoS stride=20 | 5x uchar4 vectorized loads; e5m2 range +-57344     | SM 8.9   |
| kernels_int8.cu             | `signed char`        | 1          | AoS stride=20 | 5x `__dp4a` momentum groups (padded stride)        | SM 6.1   |
| kernels_int16.cu            | `short`              | 2          | AoS stride=20 | 5x int2 vectorized loads; DIST_SCALE=16384         | SM 5.0   |
| kernels_bf16.cu             | `__nv_bfloat16`      | 2          | AoS stride=20 | 8-bit exponent = FP32 range; 7-bit mantissa        | SM 8.0   |
| kernels_fp64.cu             | `double`             | 8          | AoS stride=20 | Full IEEE-754 double; 53-bit mantissa              | SM 6.0   |
| kernels_fp16_soa.cu         | `__half`             | 2          | SoA pull    | Coalesced gather+scatter; no AoS padding           | SM 7.0   |
| kernels_fp16_soa_half2.cu   | `__half` + `__half2` | 2          | SoA pull    | 2 cells/thread; __half2 moment accum; +9.8% ILP    | SM 7.0   |
| kernels_fp8_soa.cu          | FP8 e4m3             | 1          | SoA pull    | Coalesced gather+scatter; SM 8.9+                  | SM 8.9   |
| kernels_fp8_e5m2_soa.cu     | FP8 e5m2             | 1          | SoA pull    | e5m2 variant; SM 8.9+                             | SM 8.9   |
| kernels_int8_soa.cu         | `signed char`        | 1          | SoA pull    | Coalesced gather+scatter; int momentum accum       | SM 6.1   |
| kernels_int16_soa.cu        | `short`              | 2          | SoA pull    | Integer load path; 3% faster than FP16 SoA        | SM 5.0   |
| kernels_bf16_soa.cu         | `__nv_bfloat16`      | 2          | SoA pull    | 2x BW vs FP32; SM 8.0+                            | SM 8.0   |
| kernels_bf16_soa_bf162.cu   | `__nv_bfloat16`+`bf162` | 2       | SoA pull    | 2 cells/thread; HFMA2.BF16_V2 moment accum         | SM 8.0   |
| kernels_fp32_soa_cs.cu      | `float`              | 4          | SoA pull    | __ldg ping reads + __stcs (L2 evict-first) writes  | SM 8.0   |
| kernels_fp64_soa.cu         | `double`             | 8          | SoA pull    | FP64 coalesced; compute-bound (not BW-bound)       | SM 6.0   |
| kernels_int4.cu             | nibble (4-bit)       | 0.5        | SoA nibble  | BW ceiling only; 2 cells/thread; NOT physics       | SM 6.1   |
| kernels_fp4.cu              | FP4 E2M1 nibble      | 0.5        | SoA nibble  | BW ceiling only; FP4_DECODE[16] lookup; NOT physics| SM 8.9** |
| kernels_dd.cu               | `double[2]`          | 16         | SoA pull    | Knuth 2-sum + Veltkamp/Dekker FMA; coalesced       | SM 6.0   |
| kernels_tensor_core.cu      | WMMA frags           | N/A        | N/A           | WMMA proxy: TF32/FP16/BF16/INT8/INT4              | SM 7.0   |

(**) FP4 E2M1 is a Blackwell SM 10.0+ hardware format; emulated via nibble packing on Ada.

---

## Measured Performance (RTX 4070 Ti, Ada SM 8.9, 504 GB/s peak, 128^3 grid)

Results from `cuda-precision-bench` (open_gororoba, 2026-03-15). 30 timing steps,
5 warmup steps. Bandwidth model: (19_read + 19_write + 8_aux) * N^3 * elem_bytes.

### Physics-valid tiers -- MLUPS ranking at 128^3

| Rank | Tier            | Layout  | MLUPS  | BW_GBS | BW_PCT | VRAM_MB | Notes                            |
|------|-----------------|---------|--------|--------|--------|---------|----------------------------------|
| 1    | INT8 SoA        | SoA     | 5643   | 259.6  | 51.5%  | 76      | Pareto-optimal production tier   |
| 2    | FP8_e4m3 SoA    | SoA     | 5408   | 248.8  | 49.4%  | 76      | SM 8.9+; best FP8 variant        |
| 3    | FP8_e5m2 SoA    | SoA     | 5280   | 242.9  | 48.2%  | 76      | SM 8.9+; wider dynamic range     |
| 4    | FP16 SoA H2     | SoA     | 3802   | 349.7  | 69.4%  | 152     | 2 cells/thread; +9.8% vs FP16SoA |
| 5    | INT16 SoA       | SoA     | 3569   | 328.3  | 65.1%  | 152     | +3% vs FP16 SoA; integer path    |
| 6    | FP16 SoA        | SoA     | 3463   | 318.6  | 63.2%  | 152     | Standard FP16 pull-scheme        |
| 7    | BF16 SoA        | SoA     | 3204   | 294.7  | 58.5%  | 152     | SM 8.0+; 7.5% below FP16 SoA    |
| 8    | FP32 coarsened  | AoS     | 2107   | 387.8  | 76.9%  | 304     | Best FP32 single-kernel          |
| 9    | FP32 aa (A-A)   | AoS     | 2062   | 379.3  | 75.3%  | 304     | Halves VRAM vs ping-pong         |
| 10   | FP32 SoA CS     | SoA     | 2027   | 372.9  | 74.0%  | 304     | __stcs: <3% gain at 128^3        |
| 11   | FP32 standard   | AoS     | 1984   | 365.0  | 72.4%  | 304     | Baseline reference               |
| 12   | INT16 AoS       | AoS     | 1904   | 182.8  | 36.3%  | 160     | Equal to FP16 AoS (scatter limit)|
| 13   | FP16 AoS        | AoS     | 1912   | 183.5  | 36.4%  | 160     | AoS scatter-write limit          |
| 14   | FP8_e4m3 AoS    | AoS     | 3202   | 153.7  | 30.5%  | 80      | AoS scatter erases 4x BW gain   |
| 15   | FP8_e5m2 AoS    | AoS     | 3149   | 151.1  | 30.0%  | 80      | Same as e4m3 AoS; format=physics |
| 16   | INT8 AoS        | AoS     | 2541   | 122.0  | 24.2%  | 80      | dp4a momentum accel              |
| 17   | BF16 AoS        | AoS     | 1278   | 117.6  | 23.3%  | 152     | AoS + BF16 scalar latency        |
| 18   | FP64 AoS        | AoS     | 461    | 169.5  | 33.6%  | 608     | Compute-bound (not BW-bound)     |
| 19   | FP64 SoA        | SoA     | 406    | 149.6  | 29.7%  | 608     | FP64 SoA slower: compute-bound  |
| 20   | DD FP128        | SoA     | 58     | 42.5   | 8.4%   | 152*    | 64^3 only; research tier         |

(*) DD VRAM at 64^3 = 152 MB; at 128^3 would require ~1215 MB (capped at 64^3 in benchmark).

### Peak bandwidth reference (non-physics)

| Tier        | MLUPS  | BW_GBS | BW_PCT | VRAM_MB | Notes                          |
|-------------|--------|--------|--------|---------|--------------------------------|
| INT4 nibble | 6169   | 314.6  | 62.4%  | 38      | Edge weights collapse to zero  |
| FP4 E2M1    | 4727   | 241.0  | 47.8%  | 38      | 23% slower than INT4 (decode)  |

### Tensor Core WMMA proxy (not LBM)

| Tier  | Shape (MxNxK) | Measured GFLOPS/TOPS |
|-------|---------------|----------------------|
| TF32  | 16x16x8       | 22,880 GFLOPS        |
| FP16  | 16x16x16      | 45,901 GFLOPS        |
| BF16  | 16x16x16      | 45,954 GFLOPS        |
| INT8  | 16x16x16      | 166,189 TOPS         |
| INT4  | 8x8x32        | 189,103 TOPS         |

---

## Tier Selection Guide

```
Production (bandwidth-limited, 128^3+):
  Highest MLUPS    -> INT8 SoA (5643 MLUPS, 76 MB, 2.85x FP32)
  Best precision   -> FP16 SoA H2 (3802 MLUPS; 10-bit mantissa, +9.8% vs plain FP16 SoA)
  Moderate Re, INT -> INT16 SoA (3569 MLUPS; DIST_SCALE=16384, LSB=6.1e-5 vs INT8 LSB=0.016)
  VRAM-critical    -> INT8 SoA (76 MB) or INT4 bw-ceiling (38 MB, physics broken)

AoS vs SoA delta at 128^3:
  FP16:  SoA 3463 vs AoS 1912 -> 1.81x (scatter penalty erases BW gain in AoS)
  INT8:  SoA 5643 vs AoS 2541 -> 2.22x (largest gain from SoA at 1-byte tier)
  INT16: SoA 3569 vs AoS 1904 -> 1.87x
  FP32:  SoA ~1967 vs AoS standard 1984 -> essentially equal (both BW-bound, no scatter issue)

FP32 variant decision (AoS):
  grid <= 32^3  -> coarsened or mrt_coarsened (instruction-bound, not BW-bound)
  grid = 64^3   -> mrt_aa (5541 MLUPS at 64^3; L2 hot due to small working set)
  grid = 128^3  -> coarsened (2107 MLUPS) or aa (2062 MLUPS; halves VRAM)
  grid >= 256^3 -> aa (A-A halves ping-pong VRAM; same MLUPS as standard)

Cache policy notes:
  __stcs (FP32 SoA CS): <3% gain at 128^3 (304 MB >> 48 MB L2); not worth the complexity
  Only beneficial at <= 64^3 where working set fits in L2

FP64:
  Compute-bound on Ada gaming SKU (64:1 ratio); SoA does NOT help (406 vs 461 MLUPS AoS)
  Use FP64 for validation only; FP32 for production
```

---

## FP32 Production SoA Kernels (kernels_soa.cu)

FP32 SoA production kernels ported from open_gororoba. All kernels use the
i-major SoA layout (`f[dir * N + cell]`) with FP32 storage and arithmetic.
Standalone file, no external headers.

### Kernel Entry Points

| Kernel                              | Collision | Streaming           | Coarsening | Notes                                              |
|-------------------------------------|-----------|---------------------|------------|----------------------------------------------------|
| `lbm_step_soa_fused`               | BGK       | Push                | None       | Baseline SoA kernel                                |
| `lbm_step_soa_mrt_fused`           | MRT       | Push                | None       | d'Humieres 5-rate MRT (722 FMA/cell)               |
| `lbm_step_soa_pull`                | BGK       | Pull                | None       | Coalesced writes, scattered reads                  |
| `lbm_step_soa_mrt_pull`            | MRT       | Pull                | None       | MRT + pull streaming                               |
| `lbm_step_soa_tiled`               | BGK       | Tiled pull (8x8x4)  | None       | Shared-memory halo; 45.6 KB smem/block             |
| `lbm_step_soa_mrt_tiled`           | MRT       | Tiled pull (8x8x4)  | None       | MRT + tiled pull                                   |
| `lbm_step_soa_coarsened`           | BGK       | Push                | float2     | 2 cells/thread; ILP via independent chains          |
| `lbm_step_soa_mrt_coarsened`       | MRT       | Push                | float2     | MRT + float2 coarsening (1444 FMA/thread)          |
| `lbm_step_soa_coarsened_float4`    | BGK       | Push                | float4     | 4 cells/thread; 128-bit bus saturation              |
| `lbm_step_soa_aa`                  | BGK       | A-A (single buffer) | None       | Halves VRAM; parity-driven direction swap           |
| `lbm_step_soa_mrt_aa`              | MRT       | A-A (single buffer) | None       | MRT + A-A streaming                                |
| `lbm_step_soa_batch_kernel`        | BGK       | Push                | None       | 4D batch: multiple galaxies per launch              |
| `initialize_uniform_soa_kernel`    | --        | --                  | --         | Uniform rho/u init                                 |
| `initialize_custom_soa_kernel`     | --        | --                  | --         | Per-cell AoS input -> SoA storage                  |
| `initialize_custom_soa_batch_kernel` | --      | --                  | --         | Batch init (multiple galaxies)                     |
| `compute_smagorinsky_tau_kernel`   | --        | --                  | --         | LES Smagorinsky tau from velocity gradients        |
| `compute_smagorinsky_tau_tiled`    | --        | --                  | --         | Tiled Smagorinsky (smem halo, 7.2 KB/block)       |
| `reduce_max_speed_f32`             | --        | --                  | --         | GPU max-speed reduction for Mach telemetry         |

### Features

- **MRT collision**: d'Humieres (2002) orthogonal basis, 5 distinct relaxation
  rates (s_nu, s_e, s_eps, s_q, s_ghost). Ghost moment damping (s_ghost=1.0)
  extends Mach stability from ~0.3 (BGK) to ~1.5.
- **Tiled pull-scheme (8x8x4)**: Shared-memory halo loads replace 19 scattered
  global reads with 19 LDS reads (~20 cycles vs ~200 cycles L2 miss).
  45.6 KB smem/block fits within Ada's 48 KB default partition.
- **A-A streaming**: Single-buffer in-place scheme with parity-driven direction
  remapping. Halves VRAM (76 MB saved at 128^3, 608 MB at 256^3). Enables
  256^3 on 12 GB GPUs.
- **Thread coarsening**: float2 (2 cells/thread) and float4 (4 cells/thread)
  vectorized loads. ILP from independent collision chains hides MUFU.RCP latency.
- **Smagorinsky LES**: Per-cell turbulent viscosity via strain rate tensor.
  Tiled variant uses 8x8x4 shared-memory halo for velocity gradients.
- **Batch kernel**: 4D dispatch for multiple galaxies in a single launch.
  CUDA Graph compatible (fixed grid geometry across steps).
- **Guo forcing**: Unconditional force path on all step kernels; compiler culls
  zero-force branch.

### Kernel Variant Decision Table

```
grid <= 32^3  -> coarsened or mrt_coarsened (instruction-bound, not BW-bound)
grid = 64^3   -> mrt_tiled or coarsened (L2 hot; tiled benefits from smem)
grid = 128^3  -> aa or coarsened (aa halves VRAM; coarsened: +2% MLUPS)
grid >= 256^3 -> aa (A-A halves ping-pong VRAM; same MLUPS as standard)
```

### Reference Claims

- **C-1388**: CPU LBM compute-bound at 128^3 (GPU required for production)
- **C-1389**: Tiled pull-scheme is an anti-pattern on Ada Lovelace at 128^3
  (L2 bandwidth sufficient; smem overhead > halo benefit)
- **C-1390**: A-A streaming at 79.7% peak memory bandwidth (RTX 4070 Ti)

---

## GPU Box-Counting Fractal Dimension (kernels_box_counting.cu)

GPU box-counting kernels for computing fractal dimension of density fields.
One thread per box at each scale; warp-level ballot reduction aggregates
occupancy with 32x fewer atomics than naive per-thread atomicAdd.

### Kernel Entry Points

| Kernel                | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `box_count_at_scale`  | Count occupied boxes at a given scale (early-exit)   |
| `zero_u32`            | Zero a single u32 counter                            |
| `reduce_minmax_f32`   | Two-pass min/max reduction (eliminates 8 MB readback)|
| `build_histogram_f32` | 256-bin shared-memory histogram (1 KB host transfer) |
| `zero_histogram`      | Zero a 256-bin histogram                             |

### Algorithm

1. Host dispatches `box_count_at_scale` once per scale (log2 box sizes 1..N/2).
2. Each thread scans cells within its box, early-exits on first occupied cell.
3. `__activemask()` -> `__ballot_sync()` -> `__popc()` -> lane-0 `atomicAdd`:
   warp-level reduction cuts atomic contention 32x.
4. GPU Otsu threshold: `reduce_minmax_f32` + `build_histogram_f32` eliminate
   the 8 MB PCIe readback; only 1 KB histogram is copied to host.

---

## Sparse Brick Map Kernels (sparse/)

Sparse A-A D3Q19 LBM for high-sparsity domains (e.g., 1024^3 in 12 GB VRAM).
Uses a Brick Map (indirect table) to allocate only active 8x8x8 bricks.

### sparse/kernels_sparse_lbm.cu

| Kernel                | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `lbm_step_sparse_aa`  | Sparse A-A fused collision + streaming (512 thr/blk)|

- A-A single-buffer pattern with parity-driven direction swap.
- Brick-local addressing via indirect table lookup.
- Bounce-back at solid (inactive) brick boundaries.

### sparse/kernels_sparse_map.cu

| Kernel                       | Purpose                                         |
|------------------------------|-------------------------------------------------|
| `generate_occupancy_bitmask` | Sweep geometry mask -> packed occupancy bits     |
| `expand_bitmask_to_counts`   | Unpack bitmask to per-brick 0/1 array            |
| `build_indirect_table`       | Exclusive-scan offsets -> indirect table + IDs   |

---

## Layout Details

### AoS (Array of Structures)

```
Memory: [f0_dir0, f0_dir1, ..., f0_dir18, f0_pad | f1_dir0, f1_dir1, ...]
Access: f_in[cell * 20 + dir]
```

Push-scheme writes to next cell `f_out[(cell+delta)*20 + dir]`:
- x-direction streaming: delta=1, stride 1 -- coalesced.
- y-direction streaming: delta=nx, stride 20*nx -- non-coalesced.
- z-diagonal streaming: delta=nx*ny+nx+1 -- scattered across cache lines.

### i-major SoA (Structure of Arrays)

```
Memory: [f_dir0_cell0, f_dir0_cell1, ..., f_dir0_cellN | f_dir1_cell0, ...]
Access: f_in[dir * n_cells + cell]
```

Pull-scheme reads from backward neighbor `f_in[dir*n_cells + back_cell]`:
- Warp threads have consecutive cell indices -- all reads/writes hit consecutive addresses.
- All 19 directions produce coalesced 128-byte transactions.

---

## Kernel Descriptions

### FP16 AoS (kernels_fp16.cu)
- AoS stride=20 (padded from 19 for 4-byte half2 alignment).
- 10x `half2` vectorized loads cover indices 0-19 with a bounds guard on slot 19.
- `__ldg()` read-only cache path. Horner FMA equilibrium.
- VRAM at 128^3: 20 * 2,097,152 * 2 * 2 (ping+pong) = ~160 MB.
- Measured: 1912 MLUPS at 128^3 (scatter-write non-coalescing caps throughput).

### FP8 e4m3 AoS (kernels_fp8.cu)
- `__nv_fp8_storage_t`, SM 8.9+ (Ada Lovelace / CUDA 11.8+).
- AoS stride=20; 5x `uchar4` aligned loads. Conversion: `__nv_cvt_fp8_to_halfraw`.
- VRAM at 128^3: 20 * 2,097,152 * 1 * 2 (ping+pong) = ~80 MB.
- Measured: 3202 MLUPS (1.61x FP32; AoS scatter prevents full 4x gain).

### FP8 e5m2 AoS (kernels_fp8_e5m2.cu)
- Same layout as e4m3 AoS; uses `__NV_E5M2` format tag.
- 5-bit exponent (range +-57344 vs e4m3's +-448), 2-bit mantissa (~25% rel. error).
- Measured: 3149 MLUPS -- within 2% of e4m3 AoS; format choice is physics-only.

### INT8 AoS (kernels_int8.cu)
- DIST_SCALE = 64; range [-128, 127] maps to f_i in [-2.0, 1.984].
- AoS stride=20; 5x int32 loads with signed-byte unpacking.
- 5x `__dp4a(a_i8x4, b_i8x4, acc)` momentum groups.
- Measured: 2541 MLUPS at 128^3. dp4a accel partially offsets AoS scatter penalty.

### INT16 AoS (kernels_int16.cu)
- DIST_SCALE = 16384; range [-32768, 32767] = f_i in [-2.0, 1.9999].
  LSB = 6.1e-5 (vs INT8 LSB = 0.016). For moderate-Re flows where INT8 saturates.
- AoS stride=20; 5x `int2` vectorized loads (4 shorts per load = 8 bytes).
- Argument order: force (arg 4), tau (arg 5) -- matches BenchKernelRunner convention.
- Measured: 1904 MLUPS at 128^3 (equal to FP16 AoS; scatter dominates at 2 bytes).

### BF16 AoS (kernels_bf16.cu)
- `__nv_bfloat16`, SM 8.0+. Same dynamic range as FP32 (8-bit exponent), 7-bit mantissa.
- AoS stride=20; same vectorized load pattern as FP16 AoS.
- Measured: 1278 MLUPS at 128^3 (below FP16 AoS due to BF16 scalar load latency on Ada).
- Risk: tau < 0.55 instability at shear boundaries (7-bit mantissa insufficient).

### FP64 AoS (kernels_fp64.cu)
- Full IEEE-754 double; 53-bit mantissa. Compute-bound on Ada gaming SKU.
- AoS stride=20 (8 bytes/dist; no padding needed for 8-byte alignment).
- Measured: 461 MLUPS at 128^3 (~0.6 TFLOPS FP64 throughput limits Ada gaming SKU).
- Use for numerical validation only.

### FP16 i-major SoA (kernels_fp16_soa.cu)
- i-major layout: `f[dir * n_cells + cell]`, `__half` elements, no padding.
- Pull-scheme: reads `f_in[dir*n_cells + back_cell]`, writes `f_out[dir*n_cells + cell]`.
- VRAM at 128^3: 19 * 2,097,152 * 2 * 2 (ping+pong) = ~152 MB (5% less than AoS FP16).
- Measured: 3463 MLUPS at 128^3 (1.81x over FP16 AoS; scatter penalty eliminated).

### FP16 SoA half2 ILP (kernels_fp16_soa_half2.cu)
- Thread k handles cells 2k and 2k+1.
- `__half2` velocity moment accumulation: `__hadd2`/`__hmul2` = 2 FP16 FMAs/instruction.
- BGK collision stays FP32 for numerical stability.
- Grid: `ceil(n_cells / 2) / 128` blocks; init kernel uses 1 thread/cell.
- Measured: 3802 MLUPS at 128^3 (+9.8% vs plain FP16 SoA via Ada dual-issue scheduling).

### FP8 e4m3 i-major SoA (kernels_fp8_soa.cu)
- i-major layout with `__nv_fp8_storage_t` elements (1 byte each). SM 8.9+ required.
- VRAM at 128^3: 19 * 2,097,152 * 1 * 2 = ~76 MB (vs AoS FP8 ~80 MB).
- Measured: 5408 MLUPS at 128^3 (1.69x over FP8 e4m3 AoS).

### FP8 e5m2 i-major SoA (kernels_fp8_e5m2_soa.cu)
- Same pull-scheme as FP8 e4m3 SoA; uses e5m2 encoding.
- Measured: 5280 MLUPS at 128^3 (2.4% below e4m3 SoA; quantization path marginally slower).

### INT8 i-major SoA (kernels_int8_soa.cu)
- i-major layout with `signed char` elements. DIST_SCALE = 64 (same as AoS INT8).
- Direct int32 momentum accumulation (cannot use dp4a for non-sequential SoA loads).
- Measured: 5643 MLUPS at 128^3 (2.85x FP32 baseline; Pareto-optimal production tier).

### INT16 i-major SoA (kernels_int16_soa.cu)
- i-major layout with `short` elements. DIST_SCALE = 16384 (same as INT16 AoS).
- Integer load/store path avoids FP16 half-conversion pipeline per direction.
- Measured: 3569 MLUPS at 128^3 (+3.0% over FP16 SoA; same element size, integer path wins).

### BF16 i-major SoA (kernels_bf16_soa.cu)
- i-major layout with `__nv_bfloat16` elements. SM 8.0+.
- Measured: 3204 MLUPS at 128^3 (7.5% below FP16 SoA despite equal element size;
  BF16 scalar load latency on Ada SM 8.9 is higher than FP16 scalar load latency).

### FP32 SoA cache-streaming stores (kernels_fp32_soa_cs.cu)
- Ping reads via `__ldg()` (read-only cache path).
- Pong writes via `__stcs()` (PTX `st.global.cs`): L2 evict-first, does not pollute L1.
- Prevents cold pong writes from evicting hot ping data from L2.
- Measured: 2027 MLUPS at 128^3 (+3.1% vs FP32 SoA baseline -- negligible).
  At 128^3 the 304 MB ping buffer far exceeds the 48 MB Ada L2; no hot data to protect.
  At 32^3/64^3, apparent bandwidth exceeds 200% of device peak (L2 reuse artifact).

### FP64 i-major SoA (kernels_fp64_soa.cu)
- Full double precision; i-major pull-scheme for coalesced memory access.
- Measured: 406 MLUPS at 128^3 (COMPUTE-BOUND; slower than FP64 AoS at 461 MLUPS).
  SoA coalescing does not help when FP64 arithmetic throughput is the bottleneck.
  Ada SM 8.9 gaming SKU: ~0.6 TFLOPS FP64 (64:1 ratio vs FP32).

### INT4 nibble-packed i-major SoA (kernels_int4.cu)
**BANDWIDTH CEILING ONLY -- NOT physics-viable.**
- DIST_SCALE = 14; range [-8, 7].
  - Rest weight (1/3*14 = 4.67 -> 5): OK.
  - Face weight (1/18*14 = 0.78 -> 1): marginally OK.
  - Edge weight (1/36*14 = 0.39 -> 0): ZERO. Physics corrupted.
- 2 cells per thread; eliminates concurrent nibble read-modify-write races.
- Measured: 6169 MLUPS, 38 MB VRAM at 128^3 (bandwidth ceiling reference).

### FP4 E2M1 nibble-packed (kernels_fp4.cu)
**BANDWIDTH CEILING ONLY -- NOT physics-viable. Emulated on Ada.**
- E2M1 format: values {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Rest weight 1/3 -> 0.5 (50% error).
- `FP4_DECODE[16]` lookup table for decode; `float_to_fp4()` quantizer for encode.
- Same nibble layout as INT4; 2 cells per thread.
- Measured: 4727 MLUPS at 128^3 (23% below INT4 due to decode table overhead).
- Blackwell SM 10.0+ required for hardware FP4; this is an Ada emulation.

### Double-Double FP128 (kernels_dd.cu)
- `(hi: double, lo: double)` pairs per distribution -- 16 bytes each.
- i-major SoA pull: `f_hi[dir * n_cells + cell]` -- fully coalesced.
- Four device buffers: `f_hi_a/f_lo_a` (ping), `f_hi_b/f_lo_b` (pong).
- ~106-bit mantissa via Knuth 2-sum (`two_sum`) and Veltkamp/Dekker (`two_prod` + FMA residual).
- Measured: 58 MLUPS at 64^3 (capped; ~1215 MB VRAM at 128^3).
- Research tier; measures DD arithmetic overhead vs FP64.

### Tensor Core WMMA Proxy (kernels_tensor_core.cu)
NOT an LBM kernel -- measures raw Tensor Core GFLOPS as design headroom reference.

| Tier  | Shape (MxNxK)  | Precision   | Measured (RTX 4070 Ti) | Accumulator |
|-------|----------------|-------------|------------------------|-------------|
| TF32  | 16x16x8        | tf32        | 22,880 GFLOPS          | FP32        |
| FP16  | 16x16x16       | half        | 45,901 GFLOPS          | FP32        |
| BF16  | 16x16x16       | bfloat16    | 45,954 GFLOPS          | FP32        |
| INT8  | 16x16x16       | signed char | 166,189 TOPS           | INT32       |
| INT4  | 8x8x32         | s4          | 189,103 TOPS           | INT32       |

BF16 TC parity with FP16 on Ada confirmed (both ~45.9 TFLOPS).
Quantifies the gap between bandwidth-bound custom LBM and TC peak throughput.

---

## Launch Configuration

| Kernel family                  | Threads/block | Grid formula                          |
|-------------------------------|---------------|---------------------------------------|
| AoS kernels (all)              | 128           | `ceil(n_cells / 128)`                 |
| SoA pull kernels (all)         | 128           | `ceil(n_cells / 128)`                 |
| FP16 SoA half2 ILP (step only) | 128           | `ceil(n_cells / 2 / 128)` (2 cells/T) |
| INT4 / FP4 nibble (step only)  | 128           | `ceil(n_cells / 2 / 128)` (2 cells/T) |
| DD FP128                       | 128           | `ceil(n_cells / 128)`                 |
| TC WMMA proxy                  | 32 (1 warp)   | fixed; `__launch_bounds__(32,1)`       |

Note: FP16 SoA H2 / INT4 / FP4 init kernels use 1 thread/cell (not 2 cells/thread).

---

## Key Physical Constraints

- **INT4/FP4**: Edge-velocity populations (directions 7-18, weight 1/36) quantize to 0.
  Mass not conserved. Use only for bandwidth measurement.
- **FP8 e4m3**: ~12.5% relative error per distribution value. tau >= 0.6 recommended.
- **FP8 e5m2**: ~25% relative error. Use only when flow exceeds e4m3's +-448 range.
- **BF16**: 7-bit mantissa, ~0.8% relative error. Instability risk for tau < 0.55.
- **INT8**: Saturation at rho >> 1 (DIST_SCALE=64, max f_i = 1.984).
- **INT16**: Better than INT8 for moderate-Re (DIST_SCALE=16384, LSB=6.1e-5 vs 0.016).
  Useful when INT8 saturation causes instability but FP32 bandwidth is excessive.
- **FP64 / DD**: Compute-bound on Ada gaming (64:1 FP64 ratio). Validation/research only.

---

## Infrastructure Headers

| Header | Purpose |
|--------|---------|
| `include/lbm_kernels.h` | Kernel variant enum, metadata table, VRAM/bandwidth helpers |
| `include/lbm_kernel_selector.h` | Auto-selects optimal kernel for grid size and precision requirement (C-1392 decision table) |
| `include/lbm_managed_memory.h` | CUDA Unified Memory config and PCIe overhead estimation for 1024^3+ grids |
| `include/lbm_metrics.h` | Performance measurement utilities |

## SASS-Informed Optimization

The SASS reverse engineering measurements in `src/sass_re/` feed directly
into kernel optimization decisions. See
[`docs/sass/SASS_KERNEL_OPTIMIZATION_GUIDE.md`](../../docs/sass/SASS_KERNEL_OPTIMIZATION_GUIDE.md)
for the full analysis connecting measured instruction latencies to LBM
kernel design choices.
