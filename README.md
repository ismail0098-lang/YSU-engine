# YSU Engine

I reverse-engineered NVIDIA's Ada Lovelace GPU assembly (SASS), measured instruction latencies they don't publish, then used that to hand-write Instant-NGP's hot loops in inline PTX — **MLP inference 3.16x faster than what nvcc generates.** The whole thing sits on top of a 35,000-line C11 engine I built from scratch: path tracer, Vulkan GPU compute, NeRF inference, mesh editor, nuclear reactor sim. I'm 16.

## Highlights

**Reverse-engineered SASS on real hardware** — 9 probe kernels, 2 microbenchmarks, 3,107 instructions disassembled and analyzed on an RTX 4070 Ti Super (SM 8.9). Measured what NVIDIA doesn't document: FFMA latency is 4.54 cycles, MUFU.EX2 is 17.56, LDG pointer-chase is 92. Decoded the 64-bit instruction encoding format by diffing thousands of instruction words.

**Beat the compiler on Instant-NGP** — rewrote the three critical kernels (hash grid encoding, MLP forward, volume rendering) in SASS-level inline PTX. The MLP runs **3.16x** faster. Volume rendering **1.53x**. Hash grid went through three optimization iterations (0.69x regression → 1.03x parity → **1.11x** win) — documented every dead end and fix.

**35K lines of C/CUDA/GLSL, zero frameworks** — CPU path tracer with adaptive sampling and deterministic RNG, Vulkan compute with 28 render modes and hybrid mesh+NeRF, RBMK-1000 nuclear reactor thermal simulation with real IAEA material correlations, quantum orbital raymarcher, Blender-style mesh editor. All from scratch.

### Start here

| If you want to... | Read this |
|---|---|
| Understand the SASS measurements | [`src/sass_re/RESULTS.md`](src/sass_re/RESULTS.md) |
| Learn to read GPU assembly from scratch | [`LEARNING_GUIDE.md`](src/sass_re/instant_ngp/docs/LEARNING_GUIDE.md) |
| See the optimized kernels + SASS diffs | [`src/sass_re/instant_ngp/`](src/sass_re/instant_ngp/) |
| Reproduce the 3.16x benchmark | [`src/sass_re/instant_ngp/README.md`](src/sass_re/instant_ngp/README.md) |
| Browse the full engine | [`src/`](src/) — start with [`render/render.c`](src/render/render.c) |

---

## SASS Reverse Engineering

Built on RTX 4070 Ti Super (Ada Lovelace, SM 8.9). First-party measurements, not copied from docs.

| Instruction | Latency | Throughput (ops/clk/SM) |
|---|---|---|
| FFMA (fused multiply-add) | 4.54 cyc | 44.6 |
| IADD3 (3-input integer add) | 2.51 cyc | 68.2 |
| MUFU.EX2 (fast exp, SFU) | 17.56 cyc | 9.9 |
| MUFU.RCP (reciprocal, SFU) | 41.55 cyc | — |
| LDG (global memory chase) | 92.29 cyc | — |
| LDS (shared memory chase) | 28.03 cyc | — |
| SHFL.BFLY (warp shuffle) | 24.96 cyc | — |

**Toolkit:** 9 probe kernels (FP32, integer, MUFU, bitwise, memory, conversions, control flow, special regs, tensor cores) + latency/throughput microbenchmarks with 512-deep dependent chains. Multi-architecture pipeline with parameterized scripts — Pascal (SM 6.1) vs Ada (SM 8.9) comparison ready.

**Encoding analysis:** Reverse-engineered the 64-bit SASS instruction word by diffing same-opcode instructions with different operands. Mapped register fields for FADD, FFMA, IADD3, LOP3, MOV, LDG, STG. Opcode lives in the low 16 bits (e.g. FADD = 0x7221, IADD3 = 0x7210, LDG = 0x7981).

Full results: [`src/sass_re/RESULTS.md`](src/sass_re/RESULTS.md)

## Instant-NGP SASS Kernels

Three kernels rewritten in inline PTX to produce optimal SASS. Every instruction hand-chosen, every register allocation deliberate, verified by disassembly.

| Kernel | Speedup vs nvcc -O2 | Key technique |
|---|---|---|
| **MLP Forward** (27→64→64→4) | **3.16x** | 8-wide ILP FFMA chains, shared mem weight tiling, FMNMX ReLU, MUFU sigmoid |
| **Volume Rendering** | **1.53x** | MUFU.EX2 fast exp, predicated early exit, warp SHFL neighbor sharing |
| **Hash Grid Encoding** | **1.11x** | float2 vectorized LDG.E.64, SW pipelining across levels, LOP3 XOR |

The hash grid has a documented optimization journey: v1 used `asm volatile` everywhere and regressed to 0.69x (volatile barriers killed load/compute interleaving). v2 switched to non-volatile asm + pure C trilinear, reaching parity. v3 added float2 vectorized loads and software pipelining for the win.

### Why 3.16x over -O2 is real

The MLP kernel is compute-bound (6,249 FFMA instructions). The compiler generates sequential accumulator chains — it doesn't know the dot product has 8 independent lanes it can overlap. The hand-written version keeps 8 FFMA accumulators in flight simultaneously, saturating the FP32 pipeline. Shared memory weight tiling eliminates redundant global loads. FMNMX replaces a branch-based ReLU. These aren't tricks — they're what an engineer would do if they could see the assembly, which is exactly what this project enables.

The reference kernels are plain CUDA compiled with `nvcc -arch=sm_89 -O2`. No `-O0`, no strawman. You can read the reference code, read the PTX code, and diff the SASS yourself — everything is in [`src/sass_re/instant_ngp/`](src/sass_re/instant_ngp/).

### Reproduce it yourself

```powershell
cd src/sass_re/instant_ngp
powershell -ExecutionPolicy Bypass -File build_and_verify.ps1
```

This compiles both reference and PTX kernels, validates correctness (MLP max error: 1.19e-07), and prints wall-clock speedups. Requires CUDA 13.x and an SM 7.5+ GPU. Tested on RTX 4070 Ti Super — should show similar ratios on any Ada/Ampere/Turing card since the wins are architectural (ILP, memory access pattern), not clock-speed dependent.

### Learn from it

Three-tier docs — written so anyone can learn from this work:
- [Explained for Everyone](src/sass_re/instant_ngp/docs/EXPLAINED_FOR_EVERYONE.md) — no CS background needed
- [Learning Guide](src/sass_re/instant_ngp/docs/LEARNING_GUIDE.md) — teaches you to read SASS from scratch, with 5 real instructions decoded step by step
- [Technical Reference](src/sass_re/instant_ngp/docs/TECHNICAL_REFERENCE.md) — register counts, SASS diffs, architecture-specific details

---

## The Engine

Everything below is what the SASS work sits on top of — a complete rendering and physics engine in C11.

### Path Tracer
- Tile-based multithreaded renderer — persistent thread pool, 64-byte-aligned per-thread state, atomic job stealing
- Adaptive sampling via Welford online variance — each pixel converges independently, saves 40-70% of samples on smooth regions
- Deterministic per-pixel RNG (xorshift32 + Fibonacci hash + Murmur finalizer) — same image regardless of thread count
- Materials: Lambertian, Metal (fuzz), Dielectric (Snell + Schlick + TIR), Emissive (HDR)
- Beer-Lambert fog, Russian roulette, 4 debug viz modes

### BVH
- Arena-allocated contiguous DFS-order nodes, near-first child traversal with early-out
- ML policy pruning (experimental) — offline-trained prune decisions applied via binary search, with per-node visit telemetry

### NeRF on CPU
- AVX2/AVX-512 vectorized MLP inference (27→64→64→4) with runtime CPUID detection and scalar fallback
- 12-level hash grid encoding with software prefetch, 64³ occupancy grid for empty-space skipping
- Batched 8-ray variant, fp16 weights on disk with hand-written half-to-float expansion

### Vulkan GPU Compute
- Interactive raytracer with WASD + mouse-look camera, progressive accumulation
- GPU LBVH construction, hybrid mesh + NeRF rendering (28 modes), depth prepass at quarter resolution
- 6 GLSL compute shaders: raytracer, quantum wavefunction, quantum raymarch, nuclear density, thermal diffusion, tonemap

### Nuclear & Quantum Physics
- **RBMK-1000 reactor sim** — Chernobyl Unit 4 at 3200 MWt, 1661 fuel channels, 6 materials with IAEA correlations, 3D heat diffusion + 1D coolant flow with boiling transitions, Zircaloy oxidation with positive void coefficient feedback
- **Quantum orbital raymarcher** — two-pass Vulkan compute for hydrogen-like atoms up to Z≈30, Aufbau electron filling, Slater Z_eff, signed wavefunction phase coloring

### Mesh Editor
- Single-file immediate-mode 3D editor on raylib — Grab, Rotate, Scale, Extrude, Inset, Bevel with axis constraints
- Vertex/Edge/Face selection via Moller-Trumbore ray picking, OBJ import/export

### Denoiser
- Separable bilateral filter with cache-optimized vertical strips, Rec.709 luminance range kernel
- ONNX runtime and GPU shader paths for ML denoising

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

CMake auto-detects Vulkan, raylib, OpenMP, AVX2. Missing deps skip those targets — the core path tracer needs only pthreads.

## Run

```bash
# Quick render
YSU_W=320 YSU_H=180 YSU_SPP=4 ./build/bin/ysu

# Full quality
YSU_W=1920 YSU_H=1080 YSU_SPP=128 YSU_ADAPTIVE=1 YSU_NEURAL_DENOISE=1 ./build/bin/ysu
```

Windows:
```powershell
$env:YSU_W=1920; $env:YSU_H=1080; $env:YSU_SPP=128; .\build\bin\ysu.exe
```

## Configuration

Environment variables only — no config files, no arg parsing:

| Variable | Default | Description |
|---|---|---|
| `YSU_W` / `YSU_H` | 800 / 600 | Image dimensions |
| `YSU_SPP` | 64 | Samples per pixel |
| `YSU_DEPTH` | 10 | Max bounce depth |
| `YSU_THREADS` | auto | Thread count (0 = all cores) |
| `YSU_TILE` | 32 | Tile size for MT renderer |
| `YSU_ADAPTIVE` | 0 | Adaptive sampling (Welford variance) |
| `YSU_SPP_MIN` | 16 | Min SPP before adaptive early-stop |
| `YSU_REL_ERR` / `YSU_ABS_ERR` | 0.01 / 0.005 | Convergence thresholds |
| `YSU_NEURAL_DENOISE` | 0 | Enable denoiser |
| `YSU_FOG` | 0 | Beer-Lambert fog |

## Project Structure

```
src/
  core/        — vec2-4, ray, camera, sphere, triangle, image, material, color
  render/      — CPU renderer, BVH, scene loader, G-buffer, postprocess
  denoise/     — bilateral, neural, ONNX denoiser
  nerf/        — NeRF SIMD inference, hash grid, batch scheduler
  vulkan/      — Vulkan compute pipelines, LBVH, GPU BVH, OBJ loader
  physics/     — quantum volume, nuclear fission/fusion, reactor thermal
  editor/      — mesh editor, viewport, edit mode (requires raylib)
  sass_re/     — SASS reverse engineering: probes, microbench, instant-NGP kernels
  tools/       — CLI utilities
  third_party/ — stb_image_write.h
shaders/       — GLSL compute shaders + compiled .spv
docs/          — organized into sass/, nerf/, engine/, results/
scripts/       — build, test, and analysis scripts
```

## Requirements

- **Required:** C11 compiler, pthreads, CMake 3.16+
- **Optional:** Vulkan SDK, raylib (editor), OpenMP, ONNX Runtime
- **SASS toolkit:** CUDA 13.x (SM 7.5+) or CUDA 12.x (Pascal), MSVC or GCC, Python 3

## License

MIT

## Contact & Collaboration

I'm looking for:

- **A Blackwell GPU (RTX 5080 Ti / B200)** to extend the SASS measurements to SM 10.0. The multi-arch pipeline is already built — I just need the hardware. If you have one gathering dust or can lend cloud access, the comparison paper writes itself.
- **Code review from GPU engineers** — if you work on compiler backends, ISA design, or neural graphics and see something wrong (or right) in the SASS analysis, I want to hear it.
- **Collaboration on kernel optimization** — the methodology generalizes beyond Instant-NGP. Any memory-bound or compute-bound CUDA kernel can be profiled with this toolkit and hand-tuned the same way.

Reach me at: **umut7korkmaz@gmail.com** · [GitHub Issues](https://github.com/ismail0098-lang/YSU-engine/issues)

## Author

Umut Korkmaz — solo developer, 16 years old
