# YSU Engine — Your Simulated Universe (v0.4+)

A from-scratch experimental rendering engine written in C, exploring rendering as a physical, mathematical, and computational problem — not a black box.

YSU investigates the boundary between classical ray tracing, low-level CPU optimization, future GPU compute pipelines, and neural scene representations.

---

## Overview

YSU (Your Simulated Universe) is a long-term experimental rendering engine built entirely from scratch in C.

What began as a simple CPU ray tracer has evolved into a research-driven project focused on understanding how images emerge from simulation — by explicitly implementing every component rather than relying on existing engines or frameworks.

YSU is not designed for production use.
It is designed to **learn, experiment, break, rebuild, and understand**.

---

## Philosophy

YSU treats rendering as an experiment.

Instead of abstracting complexity away, YSU exposes it:
- Rays are explicit
- Geometry is explicit
- Acceleration structures are explicit
- Performance trade-offs are intentional and visible

Stability is secondary.
Clarity, control, and understanding come first.

---

## Current Features (v0.4+)

### Core Rendering
- Fully custom CPU ray tracer written entirely in C
- Explicit vector math, ray, camera, material, and primitive systems
- Triangle and mesh-based geometry support
- Modular rendering pipeline designed for experimentation and extension

### Performance & Acceleration
- Bounding Volume Hierarchy (BVH) acceleration structure (integrated module)
- Experimental multithreading infrastructure
- Low-level triangle intersection optimizations:
  - Hand-written x86 Assembly
  - AVX2 SIMD implementation
- Performance-oriented data layout and traversal design

### 360° & VR Rendering
- Full 360° equirectangular rendering pipeline
- Dual-camera stereo rendering path for VR workflows
- Dedicated 360° engine integration module

### Scene System
- Custom scene loader and scene description format
- Support for multiple objects and materials
- Scene-level experimentation hooks for future graph-based systems

### Tools & Experimental Systems
- Experimental denoising module
- Mesh editing and topology utilities
- OBJ export functionality
- Viewport and edit-mode experiments
- Early-stage HTML/JS scene editor prototype

> **Note:**  
> Many systems are intentionally experimental or incomplete.  
> YSU prioritizes research iteration speed over polish or stability.

---

## Repository Structure Highlights

Some key modules included in the repository:

- `bvh.*` — Bounding Volume Hierarchy acceleration
- `ysu_mt.*` — Multithreading infrastructure
- `denoise.*` — Denoising experiments
- `triangle_hit_asm.S` — Assembly triangle intersection
- `triangle_hit_avx2.S` — AVX2 SIMD triangle intersection
- `ysu_mesh_edit.*` — Mesh editing utilities
- `ysu_mesh_topology.*` — Topology experiments
- `ysu_obj_exporter.*` — OBJ export module
- `ysu_viewport.c` — Viewport experimentation
- `ysu_scene_editor.html` — HTML-based scene editor prototype

---

## Build & Run

YSU is built using standard C toolchains.  
Exact file lists may change as the engine evolves.

⭐ Author

ismail0098-lang
15-year-old developer exploring graphics, physics, and neural rendering through passion-driven experimentation.
YSU is a long-term project.
It will grow from a simple path tracer into a hybrid neural + Vulkan real-time engine — one commit at a time.
```txt
# Example scene.txt

camera 0 1 5   0 0 0    90
sphere 0 0 0   1    1.0 0.2 0.2   diffuse
sphere 2 0 0   0.5  0.2 0.2 1.0   metal

### Basic Build
```bash
gcc -O2 -std=c11 ysu_main.c vec3.c ray.c color.c material.c sphere.c triangle.c primitives.c render.c image.c sceneloader.c camera.c -o ysuengine
./ysuengine


