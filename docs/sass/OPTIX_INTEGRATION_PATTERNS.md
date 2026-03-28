# OptiX Integration Patterns

Reference for integrating NVIDIA OptiX ray tracing with the SASS-informed
kernel architecture. Patterns extracted from open_gororoba's OptiX boundary
layer (gororoba_optix crate) and tracer kernels (optix_tracer.cu).

---

## Architecture Principles

1. **Minimal FFI boundary**: The OptiX runtime should own only generic
   acceleration concerns (context creation, module compilation, pipeline
   linking). Domain-specific logic (BVH rebuild policy, SBT payload layout,
   particle tracing semantics) stays in simulation code.

2. **Dynamic loading**: Load `libnvoptix.so.1` / `nvoptix64_*.dll` at runtime
   via `dlopen`/`LoadLibrary`. Query function table for ABI compatibility
   (try ABI 91, 86, 78, 55 in order). This avoids hard compile-time
   dependency on a specific OptiX SDK version.

3. **RAII lifetime**: OptiX device context is tied to the CUDA context
   lifetime. Create on init, destroy on cleanup. No global state.

## SBT (Shader Binding Table) Pattern

From `optix_tracer.cu`:

```c
// SBT record with zero-copy device pointers to simulation data
typedef struct {
    // OptiX-required header (32 bytes, 16-byte aligned)
    char header[32];
    // Application payload: raw device pointers to LBM fields
    CUdeviceptr dist_in;     // Distribution buffer (read)
    CUdeviceptr rho;         // Density field
    CUdeviceptr velocity;    // Velocity field
    int nx, ny, nz;          // Grid dimensions
} LbmSbtRecord;
```

The SBT passes simulation data to OptiX programs without extra copies.
The raygen program reads `optixGetSbtDataPointer()` to access the fields.

## Payload Packing

OptiX payload registers are 32-bit unsigned integers. Pack application
data into 2-3 payload slots:

```c
// In closest-hit: pack brick index + hit distance
optixSetPayload_0(brick_index);
optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
```

## BVH for LBM Sparse Grids

The sparse brick map (8x8x8 bricks) maps naturally to OptiX AABBs:

```c
// Build AABB for each active brick
OptixAabb aabb;
aabb.minX = brick_x * 8;
aabb.minY = brick_y * 8;
aabb.minZ = brick_z * 8;
aabb.maxX = aabb.minX + 8;
aabb.maxY = aabb.minY + 8;
aabb.maxZ = aabb.minZ + 8;
```

BVH rebuild frequency: once per topology change (brick activation/deactivation),
not every LBM step. For static domains, build once at initialization.

## Performance Notes (from SASS RE)

- OptiX RT core operations are not visible in SASS dumps (hardware-accelerated
  BVH traversal runs on dedicated RT cores, not SM CUDA cores)
- The closest-hit program IS visible in SASS: trilinear interpolation inside
  `__closesthit__cell()` compiles to standard FFMA chains
- Payload packing/unpacking compiles to MOV + bitcast (minimal overhead)
- The main latency cost is BVH traversal itself (~100-500 cycles depending
  on tree depth), which overlaps with other SM work via warp scheduling

## Cross-References

- OptiX source patterns: gororoba_optix crate (external open_gororoba project)
- OptiX tracer kernel: lbm_3d_cuda crate (external open_gororoba project)
- Sparse brick map: `src/cuda_lbm/sparse/kernels_sparse_map.cu`
- SASS RE: `src/sass_re/SM89_LATENCY_THROUGHPUT_MEASUREMENTS.md` (for FFMA/LDG latencies in hit programs)
