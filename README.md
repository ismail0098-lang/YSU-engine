# YSU Engine — Your Simulated Universe

**YSU Engine** is a high-performance experimental rendering engine built **entirely from scratch in C**, exploring the intersection of classical ray tracing, low-level optimization, and emerging neural rendering techniques.

The project focuses on **systems-level graphics engineering**: memory layout, parallelism, SIMD acceleration, and custom rendering pipelines — without relying on existing engines or frameworks.

---

##  Key Features

- **From-scratch ray tracing core** (no external rendering engines)
- **BVH acceleration structure** for efficient ray–geometry intersection
- **Multithreaded CPU renderer** with adaptive workload distribution
- **SIMD optimization (AVX2 + x86 assembly)** for hot paths
- **360° equirectangular rendering** (VR / panoramic pipelines)
- **Custom camera, material, and geometry system**
- **Denoising pipeline** for high-SPP renders
- **Experimental GPU compute path** (Vulkan-based)
- Modular architecture designed for **future neural / hybrid rendering**

---

##  What Makes This Project Different

YSU Engine is not a wrapper around existing libraries.  
It is an exploration of **how rendering engines actually work at a low level**.

The project involves:
- Designing math and geometry primitives manually
- Solving real performance bottlenecks (cache, threading, SIMD)
- Debugging architecture-level issues (ABI, linker, instruction sets)
- Iterating through multiple engine versions and design decisions

This makes YSU Engine closer to a **research + systems engineering project** than a typical graphics demo.

------- Project Structure (Simplified) -------

vec3.c / .h — vector math

ray.c — ray representation

triangle.c, sphere.c — geometry primitives

bvh.c — acceleration structure

render.c — core rendering loop

denoise.c — post-processing

ysu_360_engine_integration.c — panoramic pipeline

experimental/ — SIMD, wavefront, GPU experiments

YSU Engine is actively evolving.
Some modules are experimental and may change frequently as new architectures are tested.

------- The project prioritizes: -------

learning through implementation

correctness before abstraction

performance-driven design decisions

------- Roadmap -------

Improved wavefront / packet-based rendering

Expanded GPU compute backend

Neural denoising / NeRF-style experiments

Better scene description format

Cross-platform build support

📄 License

MIT License

---

## 🛠️ Build & Run (Windows / MSYS2)

### Standard build
```bash
gcc -O3 -std=c11 -pthread \
ysu_main.c vec3.c ray.c color.c material.c camera.c sphere.c triangle.c \
primitives.c bvh.c render.c sceneloader.c image.c denoise.c \
ysu_360_engine_integration.c \
-o ysuengine


## **AVX2 + Assembly Optimized Build**


gcc -O3 -std=c11 -mavx2 -pthread \
ysu_main.c vec3.c ray.c color.c material.c camera.c sphere.c triangle.c \
primitives.c bvh.c render.c sceneloader.c image.c denoise.c \
ysu_360_engine_integration.c \
triangle_hit_asm.S aabb_hit_asm.S \
-o ysuengine




