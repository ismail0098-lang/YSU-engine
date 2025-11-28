# YSU Engine – Your Simulated Universe  
A hybrid rendering engine exploring the frontier between classical ray tracing, neural scene representations, and real-time Vulkan compute.

---

##  Overview

**YSU (Your Simulated Universe)** is an experimental rendering engine built entirely from scratch in C.  
It began as a simple CPU path tracer and is evolving into a long-term research project that unifies:

-  Physically-based ray tracing  
-  Neural Radiance Fields (NeRF)  (in future)
-  Vulkan compute acceleration  (in the future)
-  360° & VR rendering  (exists right now)
-  Scene editing and experimental tools  

YSU aims to understand light, geometry, and simulation at the deepest possible level — not by using existing engines, but by building one from first principles.

---

##  Vision

YSU is designed as a *hybrid renderer*:  
classical geometry + neural radiance fields + real-time GPU pipelines.

The long-term goal is to create a real-time research platform where:

- Rays traverse both geometric and neural scenes  
- Vulkan shaders accelerate sampling & denoising  
- NeRF volumes and meshes coexist in the same world  
- Full 360° and VR scenes can be explored interactively  

YSU is not meant to be a clone of existing engines — it is a personal laboratory of curiosity, experimentation, and simulation.

---

##  Current Features (v0.3)

-  Fully custom C-based ray tracer  
-  Vector math, rays, spheres, camera, materials  
-  360° equirectangular rendering  
-  VR-ready dual-camera output  
-  Basic scene loader  
-  Clean, modular code for future expansion  

---

##  Upcoming Milestones

### **v0.4 — Scene System Upgrade**
- Scene graph  
- Multiple objects/materials  
- Adaptive sampling

### **v0.5 — CPU Acceleration**
- Multithreading  
- BVH acceleration structure

### **v0.6 — Interactive Editor**
- HTML/JS scene editor  
- Real-time parameter controls

### **v0.7 — Vulkan Backend (Experimental)**
- Compute shader ray generation  
- GPU traversal  
- GPU denoiser

### **v0.8 — NeRF Integration**
- Neural radiance field loader  
- Neural sampling  
- Hybrid neural + geometric rendering

### **v1.0 — Full Hybrid Renderer**
- Stable Vulkan backend  
- Geometry + NeRF pipeline  
- Interactive real-time viewport  
- VR support

---


⭐ Author

ismail0098-lang
15-year-old developer exploring graphics, physics, and neural rendering through passion-driven experimentation.



## 🛠 Build & Run

### Compile:
```bash
gcc render.c -o render
./render

360° Render:
gcc ysu_360_engine_integration.c -o ysu360
./ysu360

Open-source under the MIT License.

YSU is a long-term project.
It will grow from a simple path tracer into a hybrid neural + Vulkan real-time engine — one commit at a time.

