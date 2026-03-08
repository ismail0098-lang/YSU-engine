# Instant-NGP Hot Loops in SASS-Level PTX

Hand-written SASS-level implementations of the three critical kernels in
NVIDIA's [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/).
Written in inline PTX assembly for Ada Lovelace (SM 8.9) with every instruction
hand-chosen to match the optimal SASS output.

## Why inline PTX instead of raw SASS?

There is no official NVIDIA SASS assembler. Community tools (CuAssembler, MaxAs)
have limited SM 8.9 support. Inline PTX gives us:
- **1:1 mapping** to SASS instructions (ptxas translates each PTX op to exactly one SASS op)
- **Full control** over register allocation, instruction selection, and data movement
- **Compilable** with standard nvcc — no binary patching needed
- **Verifiable** — we disassemble the output and confirm every instruction matches intent

## Kernels

### 1. Hash Grid Encoding (`hashgrid_encode.cu`)
The signature kernel of instant-NGP. For each 3D point:
- Compute grid coordinates at L resolution levels (12–16 levels)
- Hash 8 voxel corners per level using spatial hash (prime XOR)
- Load 2 features per corner from hash table (16 loads per level)
- Trilinear interpolation → 2L output features

**SASS-level optimizations:**
- Integer hashing via IMAD (fused multiply-add) instead of separate MUL+XOR
- LOP3 for 3-input XOR in a single instruction
- Coalesced LDG.128 where possible, LDG.64 for feature pairs
- FFMA chains for trilinear interpolation with careful scheduling
- Register-resident intermediate features (no spills to local memory)

### 2. Tiny MLP Forward (`mlp_forward.cu`)
64-wide fully-connected network: 27→64→64→4 with ReLU.

**SASS-level optimizations:**
- FFMA dot-product chains with 8-wide ILP (8 independent accumulators)
- MUFU.EX2 for fast sigmoid approximation
- Predicated ReLU via FMNMX (no branch, no predicate register)
- Shared memory weight tiling for the 64×64 hidden layer
- Register blocking: 4×4 output tile per thread

### 3. Volume Rendering (`volume_render.cu`)
Front-to-back alpha compositing along each ray.

**SASS-level optimizations:**
- Early ray termination via predicated exit (ISETP + @P BRA)
- FFMA for T*(1-alpha) accumulation
- MUFU.RCP for 1/(1+exp(-sigma*dt)) (fast sigmoid of density)
- Warp-level ray coherence: SHFL for neighbor sample sharing

## Build & Verify

```powershell
cd src/sass_re/instant_ngp

# One-click: compile, disassemble, validate, benchmark:
powershell -ExecutionPolicy Bypass -File build_and_verify.ps1

# Or manually:
# 1. Compile
nvcc -arch=sm_89 -O1 -allow-unsupported-compiler -lineinfo ^
     -o build/ngp_validate.exe ^
     hashgrid_encode.cu mlp_forward.cu volume_render.cu ngp_validate.cu

# 2. Run validation + benchmarks
build\ngp_validate.exe

# 3. Dump SASS for inspection
nvcc -arch=sm_89 -O1 -allow-unsupported-compiler -cubin -o build/hashgrid_encode.cubin hashgrid_encode.cu
nvdisasm -g -sf build/hashgrid_encode.cubin > sass_output/hashgrid_encode.sass
```

## Validation Results (RTX 4070 Ti Super)

All three kernels pass bit-level or near-bit-level validation against reference CUDA:

| Kernel | Max Error | Mean Error | Speedup vs Reference |
|--------|-----------|------------|---------------------|
| Hash Grid Encoding | 0.00e+00 (exact) | 0.00e+00 | 0.69x |
| MLP Forward | 1.19e-07 | 7.15e-09 | **7.09x** |
| Volume Rendering | 2.98e-07 | 4.26e-08 | **1.81x** |

Notes:
- MLP 7x speedup comes from 8-wide ILP FMA chains + shared memory weight tiling
- Hash grid PTX is slightly slower at `-O1` because `asm volatile` prevents reordering;
  at `-O0` (where ptxas can't optimize), the PTX version dominates
- Volume rendering 1.8x from MUFU.EX2 fast-exp + lean compositing loop

## SASS Instruction Counts

Disassembled from cubins compiled with `nvcc -arch=sm_89 -O1`:

| Instruction | Hash Grid | MLP Forward | Volume Render |
|------------|-----------|-------------|--------------|
| FFMA       | 182       | 6249        | 14           |
| IMAD       | 151       | 20          | 23           |
| LOP3       | 208       | 38          | 0            |
| MUFU       | 0         | 14          | 4            |
| FMNMX      | 0         | 130         | 0            |
| LDG        | 118       | 386         | 7            |
| LDS        | 0         | 1553        | 0            |
| STG        | 14        | 2           | 6            |

## Architecture Notes

All kernels target SM 8.9 (Ada Lovelace, RTX 4070 Ti Super):
- 128 FP32 units per SM, 4 sub-partitions
- IADD3 (3-input add), LOP3 (3-input logic), IMAD (32-bit integer FMA)
- FFMA latency: ~4.5 cycles, throughput: 128 ops/clk/SM
- LDG latency: ~33 cycles (L1 hit), ~200 cycles (L2)
- Shared memory: 100 KB per SM, ~23 cycle latency
