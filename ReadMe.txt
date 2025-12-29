YSU Engine — Your Simulated Universe

YSU Engine is an experimental, research-oriented rendering engine implemented entirely from scratch in C (C11).
The project explores the intersection of classical Monte Carlo ray tracing, low-level CPU optimization, and future-facing rendering paradigms such as wavefront execution and packet-based SIMD traversal.

YSU is not designed as a production engine; it is a research platform for studying performance, determinism, and algorithmic structure in physically-based rendering.

Core Goals

Investigate high-performance CPU rendering using modern instruction sets (AVX2)

Explore deterministic, reproducible Monte Carlo sampling

Study wavefront path tracing on CPU architectures

Provide a clean baseline for future research directions (NeRF, neural denoising, hybrid CPU/GPU pipelines)

Key Features
Rendering Architecture

Fully custom path tracer (no external rendering libraries)

Tile-based multi-threaded renderer

Deterministic per-pixel RNG seeding

Adaptive sampling (variance-based SPP control)

Geometry & Acceleration

Custom BVH implementation

Axis-Aligned Bounding Box (AABB) intersection in assembly

Triangle intersection:

Scalar C fallback

AVX2-accelerated SIMD kernels

Support for:

Spheres

Triangle meshes

Mixed primitive scenes

Low-Level Optimization

Hand-written x86-64 assembly for hot paths

AVX2 packet intersection experiments

Cache-aware traversal design

Strict separation of scalar vs SIMD code paths

Camera & Output

Perspective camera

360° equirectangular camera (VR-ready)

Floating-point image buffer

PPM output (debug-friendly, deterministic)

Wavefront Path Tracing (Experimental)

YSU includes an experimental wavefront renderer inspired by GPU execution models:

Ray generation

Intersection

Shading / scattering

Queue-based execution

This design enables:

Better instruction cache locality

SIMD-friendly batching

Future portability to GPU compute backends

⚠️ Wavefront mode is currently experimental and not enabled by default.

Deterministic RNG Design

YSU uses a deterministic per-pixel RNG seeding strategy:

Base seed

Pixel coordinates

Thread salt

This guarantees:

Reproducible renders

Stable benchmarking

No cross-thread correlation artifacts

This is critical for:

Research comparison

Algorithmic evaluation

Future neural training data generation

Build Requirements

C11-compatible compiler

AVX2-capable CPU

Tested with:

GCC (MinGW-w64 / MSYS2)

Platform:

Windows (primary)

Portable to Linux with minor adjustments

Example Build (PowerShell)
gcc -O3 -std=c11 -mavx2 -pthread ^
ysu_main.c vec3.c ray.c color.c material.c camera.c ^
sphere.c triangle.c primitives.c bvh.c render.c ^
sceneloader.c image.c denoise.c ^
ysu_360_engine_integration.c ^
experimental/ysu_packet.c experimental/ysu_wavefront.c ^
triangle_hit_asm.S triangle_hit_avx2.S aabb_hit_asm.S ^
-o ysuengine.exe

Project Structure (Simplified)
YSUengine/
├── core math & rays
├── primitives & materials
├── BVH & acceleration
├── renderer (MT + adaptive sampling)
├── assembly optimizations
├── experimental/
│   ├── SIMD packets
│   └── wavefront renderer
└── utilities (image, denoise, IO)

Research Directions (Planned)

SAH-based BVH construction

Packet-based BVH traversal

Neural denoising integration

NeRF / neural scene representations

Hybrid CPU/GPU execution

Progressive + temporal accumulation

Academic benchmarking suite

Intended Audience

Rendering researchers

Computer graphics students

Low-level performance engineers

Anyone interested in how renderers actually work under the hood

This project intentionally avoids abstraction layers to keep data flow and performance costs explicit.

Disclaimer

YSU Engine prioritizes clarity, control, and research value over ease of use.
The codebase assumes familiarity with:

Linear algebra

Monte Carlo integration

Multithreading

CPU microarchitecture

License

MIT License
Free to use, modify, and extend for research and educational purposes.