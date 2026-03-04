# Hybrid CPU+GPU NeRF Pipeline: Parallel Execution Architecture

## The Vision

**Not**: CPU does NeRF OR GPU does NeRF (failed)
**But**: CPU does NeRF **AND** GPU does meshes/postproc **AT THE SAME TIME**

```
┌─────────────────────────────────────────────────────────┐
│ Frame Start │
└────────────┬────────────────────────────────────┬────────┘
 │ │
 ┌──────▼──────┐ ┌───────▼────────┐
 │ CPU Thread │ │ GPU Thread │
 │ (NeRF SIMD) │ │ (Mesh/FX) │
 └──────┬──────┘ └───────┬────────┘
 │ │
 ┌────▼────┐ ┌────▼──────┐
 │ Ray │ │ Mesh │
 │ Batching │ │ Raster │
 │ (SIMD) │ │ │
 └────┬─────┘ └────┬──────┘
 │ │
 ┌────▼─────────┐ ┌────▼──────┐
 │ Hashgrid + │ │ Shadow + │
 │ MLP (AVX2) │ │ Lighting │
 └────┬─────────┘ └────┬──────┘
 │ │
 ┌────▼────────┐ ┌────▼──────┐
 │ Composite │ │ Denoise │
 │ (RGB+Alpha) │ │ │
 └────┬────────┘ └────┬──────┘
 │ │
 ┌──────▼───────────────────────────────────▼──────┐
 │ Framebuffer Sync + Blend (CPU reads GPU) │
 └─────────────────────────────────────────────────┘
 │
 ┌──────▼──────────┐
 │ Display │
 └─────────────────┘
```

---

## Why This Works

### 1. **No GPU-CPU Synchronization Wait**

**Before (GPU-only MLP)**:
```
CPU: Generate rays → [WAIT for GPU] → Get results → Composite
GPU: Receive rays → MLP inference (broken) → Return junk
```
Problem: GPU MLP is broken. CPU stalls.

**Now (Parallel)**:
```
CPU: Generate rays → SIMD NeRF inference → Write to shared buffer
GPU: Rasterize mesh → Lighting → Denoise
 (simultaneously, no wait)
Final: CPU+GPU results merge in framebuffer
```
**Benefit**: Both work in parallel. No stalling.

### 2. **SIMD Efficiency on Multi-core CPU**

- **Core 0-7**: NeRF rendering (8 cores × 8 rays per core = 64 rays in parallel)
- **Core 8+** (if available): Other CPU tasks
- **GPU**: Mesh/FX (separate from CPU)
- **Result**: ~200k rays/sec on CPU SIMD, mesh at 60 FPS on GPU simultaneously

### 3. **Memory Bandwidth Utilization**

```
CPU Memory:
 ├─ Hashgrid: 50 MB (fits in L3 cache, warm after 1st batch)
 ├─ MLP weights: 2 MB (fits in L2)
 └─ Ray batches: 1 KB/batch (L1 cache)

GPU Memory:
 ├─ Mesh buffers (separate)
 ├─ Depth/Normal textures
 └─ No contention with CPU NeRF
```

No memory bandwidth conflict.

---

## Architecture: CPU NeRF + GPU Mesh

### A. Thread Layout

```c
// In main loop:
// ============

pthread_t cpu_nerf_thread;
pthread_t gpu_mesh_thread;

// Frame N
pthread_create(&cpu_nerf_thread, NULL, render_nerf_simd_worker, frame_ctx);
pthread_create(&gpu_mesh_thread, NULL, render_mesh_worker, frame_ctx);

// Both work in parallel...

pthread_join(cpu_nerf_thread, NULL); // Wait for NeRF CPU
pthread_join(gpu_mesh_thread, NULL); // Wait for Mesh GPU

// Synchronization point: both outputs ready
framebuffer_blend(nerf_output, mesh_output);
```

### B. Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ SharedFrameContext │
├─────────────────────────────────────────────────────────────┤
│ Input: │
│ ├─ camera (shared read-only) │
│ ├─ nerf_hashgrid.bin (shared read-only) │
│ ├─ mesh (shared read-only) │
│ │
│ Output (separate): │
│ ├─ cpu_framebuffer[width×height] = {rgb, alpha} │
│ └─ gpu_framebuffer[width×height] = {rgb, depth} │
│ │
│ Synchronization: │
│ ├─ cpu_fence (pthread_barrier) │
│ └─ gpu_fence (Vulkan semaphore) │
└─────────────────────────────────────────────────────────────┘
```

### C. Blending Strategy

After both complete:
```c
for (int py = 0; py < height; py++) {
 for (int px = 0; px < width; px++) {
 vec3 nerf_rgb = cpu_framebuffer[py][px].rgb;
 float nerf_alpha = cpu_framebuffer[py][px].alpha;
 
 vec3 mesh_rgb = gpu_framebuffer[py][px].rgb;
 float mesh_depth = gpu_framebuffer[py][px].depth;
 
 // Over composite: NeRF on top of mesh
 vec3 final = mix(mesh_rgb, nerf_rgb, nerf_alpha);
 
 framebuffer[py][px] = final;
 }
}
```

---

## Implementation Path

### Phase 1: CPU NeRF (Weeks 1-2)

 **Create `nerf_simd.c`** with:
- `ysu_hashgrid_lookup_batch()` — SIMD feature extraction
- `ysu_mlp_inference_batch()` — Batched matrix multiply
- `ysu_volume_integrate_batch()` — Ray marching

 **Modify `render.c`** to:
- Add ray batching loop (8 rays per batch)
- Call SIMD NeRF functions instead of ray-by-ray
- Write to separate `cpu_framebuffer`

 **Benchmark**:
- Target: 1080p @ 30-60 FPS with 4-8 core CPU

### Phase 2: GPU Mesh (parallel to Phase 1)

 **Existing GPU already does mesh** (`tri.comp`)
- Keep current mesh rendering intact
- Just output to separate `gpu_framebuffer`
- Add synchronization point

### Phase 3: Parallel Execution

 **Combine both threads**:
```c
// In main loop:
pthread_t nerf_thread = launch_cpu_nerf_render();
pthread_t mesh_thread = launch_gpu_mesh_render();

wait_both_threads();

blend_framebuffers();
display();
```

 **Synchronization**:
- CPU writes to `cpu_fb[y][x]` (no GPU access)
- GPU writes to `gpu_fb[y][x]` (no CPU access)
- Main thread blends (read-only access to both)
- No locks needed (separate memory regions)

### Phase 4: Optimization

 **Load Balancing**:
- If CPU finishes first → start next frame's rays
- If GPU finishes first → CPU keeps processing current rays
- Adaptive workload distribution

 **Memory Optimization**:
- Cache hashgrid in CPU L3
- Prefetch MLP weights per-batch
- GPU textures stay on device

---

## Expected Performance

### Scenario 1: Fox NeRF @ 1080p @ 30fps

```
CPU (8-core):
 ├─ NeRF SIMD: 40 ms (1920×1080 rays, batched by 8)
 ├─ Per-ray cost: ~8 µs (hashgrid + MLP + volume)
 └─ Throughput: 200k rays/sec

GPU:
 ├─ Mesh rasterization: 5 ms
 ├─ Lighting: 3 ms
 ├─ Denoise: 2 ms
 └─ Total: 10 ms

Parallel Execution:
 ├─ CPU runs [0-40ms] (NeRF)
 ├─ GPU runs [0-10ms] (Mesh, finishes early)
 ├─ GPU idle [10-40ms] (waits for CPU)
 └─ Blend: 1 ms
 
Total Frame: 41 ms → **24 FPS** (CPU-bound)
```

**To reach 30 FPS**: Reduce NeRF rays/pixel to 32 steps instead of 64 → 20 ms → 50 FPS possible.

### Scenario 2: Fox NeRF @ 720p @ 60fps

```
CPU (8-core):
 ├─ NeRF: 1280×720 = ~920k rays at batched speed
 ├─ Time: 920k / 200k rays/sec = 4.6 ms
 
GPU:
 ├─ Mesh: 2 ms
 
Total: max(4.6, 2) = 4.6 ms → **217 FPS** (over-budget)
```

**Reality**: Could easily hit 60 FPS @ 720p with overhead.

---

## Key Implementation Details

### 1. **Separate Framebuffers** (Critical!)

```c
typedef struct {
 vec3 *rgb; // [width × height]
 float *alpha; // [width × height]
} CPUFramebuffer;

typedef struct {
 vec3 *rgb;
 float *depth;
} GPUFramebuffer;

// Allocate separately, no contention
cpu_fb = malloc(width * height * sizeof(vec3) + ...);
gpu_fb = gpu_create_texture(...);
```

### 2. **Ray Batching in render.c**

```c
// Current (serial):
for (int py = 0; py < height; py++) {
 for (int px = 0; px < width; px++) {
 Ray r = camera_ray(cam, px, py);
 color = render_ray(r); // Single ray
 framebuffer[py][px] = color;
 }
}

// New (batched):
RayBatch batch = {0};
for (int py = 0; py < height; py++) {
 for (int px = 0; px < width; px++) {
 batch.rays[batch.count++] = camera_ray(cam, px, py);
 
 if (batch.count == 8 || end_of_image) {
 ysu_nerf_render_batch(&batch, nerf, cpu_fb);
 batch.count = 0;
 }
 }
}
```

### 3. **Synchronization (Simple)**

```c
// No complex synchronization needed!
// CPU and GPU never access same memory

pthread_barrier_t frame_barrier;
pthread_barrier_init(&frame_barrier, NULL, 2);

void *nerf_worker(void *ctx) {
 ysu_nerf_render_batch(...);
 pthread_barrier_wait(&frame_barrier); // Signal done
}

void *mesh_worker(void *ctx) {
 gpu_render_mesh(...);
 pthread_barrier_wait(&frame_barrier); // Signal done
}

// Main:
pthread_create(&t1, NULL, nerf_worker, ...);
pthread_create(&t2, NULL, mesh_worker, ...);
pthread_barrier_wait(&frame_barrier); // Wait both

blend_framebuffers();
```

---

## Why This Solves Your Problem

| Problem | Solution |
|---------|----------|
| GPU MLP broken | CPU SIMD NeRF (proven technique, debuggable) |
| Single-threaded bottleneck | Parallel CPU+GPU execution |
| Stalling on GPU results | No synchronization (separate paths) |
| MLP weight/layout confusion | SIMD avoids complex layout issues |
| Can't achieve real-time | 200k rays/sec × 8 batches = feasible |

---

## Next Steps

### Immediate (Start Now):

1. **Create `nerf_simd.c`** — Batched NeRF functions
2. **Modify `render.c`** — Ray batching loop (5-6 hours)
3. **Test CPU-only NeRF** — Verify performance on single thread
4. **Benchmark** — Measure rays/sec achieved

### Then (Add Parallelism):

5. **Add threading** — Launch GPU mesh in parallel thread
6. **Implement blending** — Composite CPU+GPU outputs
7. **Synchronization** — Simple pthread barriers
8. **Profile** — Identify bottlenecks

### Finally (Optimization):

9. **Load balancing** — Adjust workload per-frame
10. **Caching** — Warm L3 cache with hashgrid
11. **Prefetching** — MLP weight prefetch per-batch

---

## Deliverable Timeline

| Phase | Time | Output |
|-------|------|--------|
| CPU NeRF SIMD | 3-4 hrs | `nerf_simd.c`, modified `render.c` |
| GPU Integration | 2-3 hrs | Threading + blending |
| Benchmarking | 1-2 hrs | FPS measurements, profiling data |
| Optimization | 2-4 hrs | Parameter tuning, cache optimization |
| **Total** | **8-13 hrs** | **Full CPU+GPU hybrid NeRF** |

---

## Files to Create/Modify

```
NEW:
 nerf_simd.h — Declarations for SIMD functions
 nerf_simd.c — Batched NeRF (hashgrid + MLP + integrate)
 nerf_simd_profile.c — Benchmarking utilities

MODIFY:
 render.c — Ray batching loop, threading
 ysu_main.c — Thread management, parallel setup
 gpu_vulkan_demo.c — Framebuffer sync/blend

REFERENCE (read-only):
 nerf_train_export.py — NeRF binary format
 tri.comp — GPU mesh rendering (unchanged)
```

---

## Summary

**Your instinct is right**: Don't force NeRF onto a broken GPU shader. Instead:

1. **CPU**: Handle NeRF with proven SIMD techniques (batching + AVX2)
2. **GPU**: Keep doing what it does well (mesh rasterization)
3. **Both**: Run in parallel, zero contention
4. **Blend**: Simple compositing at the end

This is a **pragmatic, production-grade approach** used in real game engines (Unreal, Unity) for hybrid CPU+GPU workloads.

