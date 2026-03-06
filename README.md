# YSU Engine

A from-scratch C11 path tracer with Vulkan GPU compute, real-time NeRF inference via hand-written AVX2/AVX-512 kernels, a Blender-style mesh editor, and nuclear reactor physics — all in ~40,000 lines of C.

YSU (Your Simulated Universe) is an MIT-licensed, from-scratch rendering and physics simulation engine built for solo developers and researchers who need a serious raytracer without corporate strings attached. Most high-quality rendering engines are proprietary, paywalled, or too coupled to specific pipelines to adapt. YSU is none of those — it's a single codebase covering CPU path tracing, GPU Vulkan compute, real-time NeRF inference, and nuclear/quantum physics visualization, free to use, modify, and build on for any scientific or creative purpose. Built by one person, designed to be understood and extended by one person.

---

## What's in here

### Path Tracer (CPU)
- **Tile-based multithreaded renderer** with a persistent pthread pool, 64-byte-aligned `WorkerLocal` structs to eliminate false sharing, and atomic job stealing in 8-tile chunks
- **Adaptive sampling** using Welford online variance — each pixel independently converges based on luminance standard error (`se ≤ max(abs_err, rel_err × |mean|)`), with configurable batch size and minimum SPP floor. Saves 40-70% of samples on smooth regions
- **Deterministic per-pixel RNG** — xorshift32 seeded with Fibonacci hashing constants (`0x9E3779B1`) and a Murmur-style finalizer, so renders are reproducible across any thread/tile configuration
- **Materials:** Lambertian, Metal (configurable fuzz), Dielectric (Snell + Schlick Fresnel + TIR), Emissive (HDR up to 10.0)
- **Extras:** Beer-Lambert fog with in-scattering, Russian roulette path termination, 4 debug viz modes (albedo, normals, depth, luminance heatmap)

### BVH Acceleration
- **Arena-allocated BVH** — contiguous DFS-order node pool (`2n+2` capacity), bump-allocated, zero per-node free cost, cache-friendly traversal
- **Near-first child ordering** with early-out: computes entry `t_min` for both children, visits closer first, skips far child when AABB test against tightened `closest_t` fails
- **ML policy pruning (experimental):** loads a CSV of `(node_id, prune)` pairs from offline analysis, applies via binary search — pruned subtrees are physically detached, skipping both AABB tests and child visits entirely. Per-node `visit_count`/`useful_count` telemetry dumps for traversal efficiency analysis

### NeRF — Instant Neural Radiance Fields on CPU
- **Runtime CPUID detection** for AVX2 and AVX-512F with scalar fallback — zero crashes on any x86 CPU
- **12-level multi-resolution hash grid encoding** with trilinear interpolation across 8 cube corners, software prefetch (`_mm_prefetch`) to hide random-access latency on the hash table
- **AVX2-vectorized MLP inference** (27→64→64→4) using 256-bit FMA, with a batched 8-ray variant for throughput. Weights stored as fp16 on disk, expanded to fp32 at load via hand-written half-to-float converter
- **64³ occupancy grid** for empty-space skipping (3× step in unoccupied voxels) and early ray termination at 99% alpha
- Front-to-back volume compositing with softplus density activation and sigmoid RGB

### Vulkan GPU Compute
- **Interactive Vulkan raytracer** — GLFW window with WASD + mouse-look FPS camera, progressive accumulation with temporal reset on movement
- **GPU LBVH construction** with OBJ loader (fan triangulation), triangle/BVH binary caching keyed on file size + mtime
- **Hybrid mesh + NeRF rendering** (28 render modes): CPU occupancy-grid depth prepass at ¼ resolution → depth hints SSBO → GPU narrow-band NeRF raymarching. Hash grid + MLP weights uploaded as GPU SSBOs
- **6 GLSL compute shaders** — raytracer core, quantum wavefunction, quantum raymarch, nuclear density, thermal diffusion, tonemap/denoise

### Nuclear & Quantum Physics Simulations

**RBMK-1000 Reactor Thermal Simulation** — models Chernobyl Unit 4 at 3200 MWt:
- 1661 fuel channels on 25 cm graphite lattice, 7 MPa boiling-water coolant, 64³–128³ grid
- Temperature-dependent properties for 6 materials (UO₂, Zircaloy-4, H₂O, graphite, B₄C, void) using IAEA thermal conductivity correlations and Fink 2000 specific heat
- 3D heat diffusion via explicit Euler with 6-neighbor Laplacian (CFL-limited) + GPU Jacobi solver path
- 1D coolant flow: enthalpy-based subcooled→boiling→superheated transitions, Zuber-Findlay drift-flux void fraction
- Baker-Just Zircaloy oxidation: `Zr + 2H₂O → ZrO₂ + 2H₂ + 6.5 MJ/kg`, positive void coefficient reactivity feedback (~2β)

**Quantum Volume Raymarcher** — hydrogen-like orbital visualization:
- Two-pass Vulkan compute: Pass 1 evaluates Σ|ψᵢ|² on a 3D SSBO grid, Pass 2 volume-raymarches with up to 1024 steps → RGBA32F output
- Full (n, l, m) quantum numbers with Aufbau-order electron filling for multi-electron atoms up to Z≈30, Slater's rules for Z_eff
- Signed wavefunction grid for phase coloring, Reinhard tone-mapping + log-compressed opacity, 7 color modes

### Mesh Editor
- **Single-file immediate-mode 3D editor** built on raylib — Blender-style controls: Edit/Viewport mode toggle, orbit cam (Alt+LMB), mouse-wheel zoom
- **7 transform tools:** Grab (G), Rotate (R), Scale (S) with axis constraint (X/Y/Z), Extrude (F), Inset (I), Face Bevel (B) — all with live mouse preview and Escape cancel
- Vertex/Edge/Face selection with Möller-Trumbore ray picking, half-edge topology, OBJ import/export
- Primitive generators: Cube, UV Sphere (20×20), Cylinder (20-seg)

### Denoiser
- **Separable bilateral filter** (horizontal + vertical pass) with cache-optimized 64-column vertical strips to fit in L2
- Rec.709 luminance-based range kernel for perceptually-weighted edge preservation
- ONNX runtime and GPU shader paths available for ML-based denoising

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

CMake auto-detects Vulkan, raylib, OpenMP, AVX2. Missing deps just skip those targets — the core CPU raytracer has zero external dependencies beyond pthreads.

## Run

```bash
# Quick render (low samples)
YSU_W=320 YSU_H=180 YSU_SPP=4 YSU_THREADS=0 ./build/bin/ysu

# Full quality with adaptive sampling + denoising
YSU_W=1920 YSU_H=1080 YSU_SPP=128 YSU_ADAPTIVE=1 YSU_NEURAL_DENOISE=1 ./build/bin/ysu
```

On Windows (PowerShell):
```powershell
$env:YSU_W=1920; $env:YSU_H=1080; $env:YSU_SPP=128; .\build\bin\ysu.exe
```

## Configuration

Everything is controlled via environment variables — no config files, no CLI arg parsing:

| Variable | Default | Description |
|---|---|---|
| `YSU_W` | 800 | Image width |
| `YSU_H` | 600 | Image height |
| `YSU_SPP` | 64 | Samples per pixel |
| `YSU_DEPTH` | 10 | Max bounce depth |
| `YSU_THREADS` | auto | Thread count (0 = all cores) |
| `YSU_TILE` | 32 | Tile size for MT renderer |
| `YSU_ADAPTIVE` | 0 | Adaptive sampling (Welford variance) |
| `YSU_SPP_MIN` | 16 | Minimum SPP before adaptive can early-stop |
| `YSU_SPP_BATCH` | 4 | Samples per adaptive batch |
| `YSU_REL_ERR` | 0.01 | Relative error threshold for convergence |
| `YSU_ABS_ERR` | 0.005 | Absolute error threshold for convergence |
| `YSU_NEURAL_DENOISE` | 0 | Enable denoiser |
| `YSU_FOG` | 0 | Enable Beer-Lambert fog |
| `YSU_DUMP_RGB` | 0 | Dump raw RGB buffer before postprocess |

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
  upscale/     — neural upscaling pipeline
  tools/       — CLI utilities (inspect_ppm, ysub_info, ysub_to_ppm)
  third_party/ — stb_image_write.h
shaders/       — GLSL compute shaders + compiled .spv
docs/          — documentation & design notes
scripts/       — build, test, and analysis scripts
models/        — pretrained NeRF model binaries
```

## Requirements

- **Required:** C11 compiler (GCC / Clang / MSVC), pthreads, CMake 3.16+
- **Optional:** Vulkan SDK (GPU raytracer + physics viz), raylib (mesh editor), OpenMP (NeRF batch), ONNX Runtime (ML denoiser)

## License

MIT License

## Author

Umut Korkmaz — 16-year-old solo developer. :)
