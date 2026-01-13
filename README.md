  # YSU Engine — Your Simulated Universe

**YSU Engine** is an experimental hybrid rendering engine developed from scratch in C, aimed at exploring core rendering algorithms, acceleration structures, and intelligent light simulation. The project emphasizes deep understanding over black-box usage — bypassing existing engines to experiment directly with rendering mathematics, data structures, and estimator designs.

This repository reflects ongoing research and engineering exploration rather than a finished product.

---

##  Motivation

Most existing renderers and digital content creation tools hide fundamental algorithms behind layers of abstraction. This limits the ability to experiment with:

- custom rendering pipelines  
- estimator behavior at low sample counts  
- acceleration structure semantics  
- data-driven denoising 

YSU Engine is built as a **platform for experimentation**, not merely as a demo or application.

---

##  Key Features

✔ **From-scratch ray tracing core** — defined by hand without third-party engines  
✔ **Bounding Volume Hierarchy (BVH)** for efficient ray intersection  
✔ **Modular architecture** designed for extension and research  
✔ **Custom geometry & material systems**  
✔ **Multithreaded CPU renderer** with adaptive workload distribution  
✔ **Post-processing denoising pipeline** (experimental)  
✔ **Experimental Vulkan compute backend** for GPU exploration  
✔ **360° equirectangular rendering support**

---

##  Project Structure

The source code is organized into modular subsystems to support experimentation,
rapid refactoring, and algorithm-level research.

| Module / Directory | Description |
|--------------------|-------------|
| `vec2.c / vec3.c / vec4.c` | Low-level vector and math utilities used throughout the engine |
| `mat3.c / mat4.c` | Matrix math for transformations and camera systems |
| `ray.c / ray.h` | Ray representation, generation, and utility operations |
| `camera.c / camera.h` | Camera models including perspective and panoramic projections |
| `triangle.c / sphere.c / mesh.c` | Geometry primitives and mesh handling |
| `material.c / material.h` | Material definitions and light interaction models |
| `scene.c / scene.h` | Scene graph, object management, and world representation |
| `bvh.c / bvh.h` | Bounding Volume Hierarchy construction and traversal |
| `lbvh.c / lbvh.h` | Linear BVH (LBVH) experiments and spatial ordering |
| `intersection.c` | Ray-geometry intersection routines |
| `sampling.c` | Random sampling utilities and Monte Carlo helpers |
| `integrator.c` | Light transport integrators (experimental) |
| `render.c / render.h` | Core rendering loop and scheduling |
| `threadpool.c` | Multithreaded task execution |
| `denoise.c / denoise.h` | Post-processing and experimental denoising algorithms |
| `image.c` | Image buffer management and output |
| `postprocess.c` | Tone mapping and image-space operations |
| `vulkan/` | Vulkan compute experiments and GPU acceleration paths |
| `experimental/` | Prototypes, SIMD tests, and algorithmic experiments |
| `scripts/` | Data analysis, benchmarking, and validation scripts |
| `assets/` | Test scenes and reference models |


---

##  Research Focus

This engine is not optimized for production. Instead, it supports:

- **custom estimator design**  
- **variance analysis at low sample rates**  
- **feature-aware denoising pipelines**  
- **algorithmic experimentation**

For example, current investigations include:
- BVH-informed grouping for denoising  
- adaptive sampling strategies  
- hybrid CPU/Vulkan render paths  
- neural / machine-learned denoiser prototypes

---
Note: This project is undergoing rapid changes. Some modules are experimental and may be replaced as research evolves.

Contribution & Extension

Although this repository is personal research, contributions (ideas, analysis tools, test scripts) are welcome — particularly for:

variance measurement tooling

estimator evaluation benchmarking

GPU pathway prototyping

feature-space denoising strategies

Please create an issue or discussion instead of direct PRs for research topics.

📜 License

This project is licensed under the MIT License.

📌 Why This Matters

YSU Engine is not a typical hobby renderer. It is:

A research & engineering exploration into how rendering subsystems actually work at the algorithmic level.

This makes it distinct from:

game engines

ready-made rendering frameworks

content creation tool add-ons

Instead, it speaks to systems-level graphics engineering and research readiness — a signal that matters in academic and advanced engineering discussions.

Author: Umut Korkmaz, a 15-year-old student :)

##  Usage & Development

Clone the repository and build:

```bash
git clone https://github.com/ismail0098-lang/YSU-engine.git
cd YSU-engine
# Add build steps here (CMake, make, etc)
