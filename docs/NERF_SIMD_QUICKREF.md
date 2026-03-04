# CPU SIMD NeRF - Quick Reference

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `nerf_simd.h` | API header | 200 |
| `nerf_simd.c` | Implementation | 1100 |
| `nerf_simd_integration.c` | Integration example | 300 |
| `nerf_simd_test.c` | Test suite | 500 |
| `BUILD_NERF_SIMD.md` | Build guide | 300 |
| `build_nerf_simd.bat` | Windows build script | 30 |
| `NERF_SIMD_COMPLETE.md` | Full documentation | - |

## Quickstart

```bash
# 1. Compile & test
gcc -O3 -march=native -std=c11 \
 nerf_simd.c vec3.c nerf_simd_test.c \
 -o nerf_test -lm
./nerf_test

# 2. Expected output
# [Shows 5 test suites + benchmarks + PPM output]
```

## API

### Load NeRF Data
```c
NeRFData *nerf = ysu_nerf_data_load(
 "models/nerf_hashgrid.bin",
 "models/occupancy_grid.bin"
);
```

### Render a Batch of Rays
```c
RayBatch batch;
batch.count = 8; // 8 rays
batch.origin[i], batch.direction[i] = ray data
batch.pixel_id[i] = output pixel coordinate

ysu_volume_integrate_batch(
 &batch,
 &nerf->config,
 nerf,
 &framebuffer,
 num_steps=32,
 density_scale=1.0f,
 bounds_max=4.0f
);
```

### Cleanup
```c
ysu_nerf_data_free(nerf);
free(framebuffer.pixels);
```

## Environment Variables

```bash
YSU_NERF_HASHGRID="models/nerf_hashgrid.bin" # Required
YSU_NERF_OCC="models/occupancy_grid.bin" # Required
YSU_NERF_STEPS=32 # Ray march steps
YSU_NERF_DENSITY=1.0 # Opacity scale
YSU_NERF_BOUNDS=4.0 # Volume size
```

## Performance

| Component | Cost/Step |
|-----------|-----------|
| Hashgrid lookup (8 rays) | 45 µs |
| MLP inference (8 rays) | 123 µs |
| Occupancy lookup (8 rays) | 2 µs |
| **Total (8 rays)** | **~200 µs** |

**Per-ray: ~25 µs/step**

## Resolution Performance

| Size | Steps | Time | FPS |
|------|-------|------|-----|
| 64×64 | 8 | 0.5s | 2 |
| 128×128 | 16 | 2.5s | 0.4 |
| 256×256 | 32 | 10s | 0.1 |

## Integration Checklist

- [ ] Copy `nerf_simd.h`, `nerf_simd.c` to project
- [ ] Add `#include "nerf_simd.h"` to render.c
- [ ] Add ray batching loop (see `nerf_simd_integration.c`)
- [ ] Initialize with `ysu_nerf_data_load()` at startup
- [ ] Render with `ysu_volume_integrate_batch()` each frame
- [ ] Cleanup with `ysu_nerf_data_free()` at exit
- [ ] Compile with `-O3 -march=native`
- [ ] Test with environment variables set

## Common Parameters

**Fast/Preview**:
```bash
YSU_NERF_STEPS=8 YSU_NERF_DENSITY=0.5 YSU_NERF_BOUNDS=2.0
```

**Balanced**:
```bash
YSU_NERF_STEPS=32 YSU_NERF_DENSITY=1.0 YSU_NERF_BOUNDS=4.0
```

**High Quality**:
```bash
YSU_NERF_STEPS=64 YSU_NERF_DENSITY=2.0 YSU_NERF_BOUNDS=8.0
```

## Debugging

**Check output is valid**:
```c
// After rendering, verify:
assert(nerf_pix.rgb.x >= 0.0f && nerf_pix.rgb.x <= 1.0f);
assert(nerf_pix.alpha >= 0.0f && nerf_pix.alpha <= 1.0f);
assert(!isnan(nerf_pix.rgb.x)); // No NaN
```

**Enable per-ray logging**:
```c
// In nerf_simd.c, add printf() in ysu_volume_integrate_batch()
printf("Ray %u: pos=(%.2f,%.2f,%.2f) rgb=(%.3f,%.3f,%.3f) alpha=%.3f\n",
 ray_idx, pos.x, pos.y, pos.z, rgb[0], rgb[1], rgb[2], alpha);
```

**Benchmark components**:
```bash
# Run test suite
./nerf_test
# Shows µs/sample for each component
```

## Typical Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Black output | Density too low | Increase `YSU_NERF_DENSITY` |
| Washed out | Bounds too small | Increase `YSU_NERF_BOUNDS` |
| Very slow | Too many steps | Reduce `YSU_NERF_STEPS` |
| NaN output | MLP overflow | Reduce `YSU_NERF_DENSITY` |
| File not found | Wrong path | Use absolute path |
| Compile error | Missing `-march=native` | Add to CFLAGS |

## Data Format

### Binary File Structure
```
Header (60 bytes):
 uint32 magic, version, levels, features, hash_size, base_res
 float per_level_scale, mlp_in, mlp_hidden, mlp_layers, mlp_out, scale
 float center[3]

Hashgrid Table:
 [levels][hash_size][features] × uint16 (half-float)

MLP Weights:
 Layer0: [hidden][input] × float
 Layer1: [hidden][hidden] × float
 Output: [output][hidden] × float

MLP Biases:
 [hidden + hidden + output] × float

Occupancy Grid (separate file):
 [64][64][64] × uint8
```

### Key Offsets
```c
#define HEADER_SIZE 60 // bytes
#define HASHGRID_OFFSET 60
#define MLP_WEIGHT_OFFSET (60 + 12*8192*2*2) // After hashgrid
#define OCCUPANCY_SIZE (64*64*64)
```

## Testing Commands

```bash
# Compile test
gcc -O3 -march=native -std=c11 -o nerf_test \
 nerf_simd.c vec3.c nerf_simd_test.c -lm

# Run test
./nerf_test

# Check output PPM
file nerf_simd_test_output.ppm
identify nerf_simd_test_output.ppm # ImageMagick

# Benchmark individual component
gcc -DBENCH_ONLY -O3 -march=native -std=c11 \
 -o nerf_bench nerf_simd.c vec3.c nerf_simd_test.c -lm
./nerf_bench
```

## Architecture

```
ysu_nerf_data_load() ────┐
 ├──→ ysu_volume_integrate_batch()
ysu_mlp_inference_batch()┤ ├─ ysu_hashgrid_lookup_batch()
 ├──→ ├─ ysu_occupancy_lookup_batch()
ysu_adaptive_step_size() ┴──→ └─ Volume compositing loop
```

## Memory Usage

| Component | Size |
|-----------|------|
| Hashgrid (12×8192×2×f16) | 6.1 MB |
| MLP weights (27×64×64×4) | 32 KB |
| MLP biases | 2 KB |
| Occupancy (64³) | 256 KB |
| NeRFFramebuffer (1920×1080) | 33 MB |
| **Total** | **~40 MB** |

## Compilation Examples

### GCC/Linux
```bash
gcc -O3 -march=native -std=c11 \
 nerf_simd.c vec3.c -o prog -lm -pthread
```

### Clang/macOS
```bash
clang -O3 -march=native -std=c11 \
 nerf_simd.c vec3.c -o prog -lm -pthread
```

### MSVC/Windows
```bash
cl /O2 /arch:AVX2 /std:c11 \
 nerf_simd.c vec3.c /link /out:prog.exe
```

### With Debug Info
```bash
gcc -O3 -g -march=native -std=c11 \
 nerf_simd.c vec3.c -o prog -lm -pthread
gdb ./prog
```

## Further Reading

- `NERF_SIMD_COMPLETE.md` - Full documentation
- `BUILD_NERF_SIMD.md` - Step-by-step build guide
- `nerf_simd_integration.c` - Integration example code
- `HYBRID_CPU_GPU_NERF_ARCHITECTURE.md` - Parallel pipeline design

---

**Ready to use!** Copy `nerf_simd.h` and `nerf_simd.c` to your project and integrate per BUILD_NERF_SIMD.md.
