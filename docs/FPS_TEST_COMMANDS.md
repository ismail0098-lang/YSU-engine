# FPS Test Commands & Results

## Quick Test Commands

### Test 1: Default Render Scale (0.5)
```bash
# Expected: 80-120 FPS
YSU_GPU_W=1920 YSU_GPU_H=1080 YSU_GPU_FRAMES=16 \
YSU_GPU_RENDER_SCALE=0.5 YSU_GPU_TEMPORAL=1 \
YSU_GPU_NO_IO=1 ./gpu_demo.exe
```

### Test 2: Balanced Quality (0.75)
```bash
# Expected: 55-70 FPS
YSU_GPU_W=1920 YSU_GPU_H=1080 YSU_GPU_FRAMES=16 \
YSU_GPU_RENDER_SCALE=0.75 YSU_GPU_TEMPORAL=1 ./gpu_demo.exe
```

### Test 3: Full Quality (1.0)
```bash
# Expected: 39.5 FPS (Session 12 baseline)
YSU_GPU_W=1920 YSU_GPU_H=1080 YSU_GPU_FRAMES=16 \
YSU_GPU_RENDER_SCALE=1.0 YSU_GPU_TEMPORAL=1 ./gpu_demo.exe
```

### Test 4: Maximum Speed (0.25)
```bash
# Expected: 150+ FPS (low quality)
YSU_GPU_W=1920 YSU_GPU_H=1080 YSU_GPU_FRAMES=16 \
YSU_GPU_RENDER_SCALE=0.25 YSU_GPU_TEMPORAL=1 \
YSU_GPU_NO_IO=1 ./gpu_demo.exe
```

## Measurement Method

When running tests, measure total time and divide by frame count:

```bash
# Time the execution
time (YSU_GPU_W=1920 YSU_GPU_H=1080 YSU_GPU_FRAMES=16 ./gpu_demo.exe)

# Calculate FPS
# FPS = (FRAMES * 1000) / TOTAL_MS
# Example: 16 frames in 400ms = (16 * 1000) / 400 = 40 FPS
```

## Expected Results Table

| Test | Scale | Render Res | Frames | Total Time | Per-Frame | FPS |
|------|-------|-----------|--------|-----------|-----------|-----|
| 1 | 0.5 | 960×540 | 16 | ~110 ms | 6.9 ms | 145 |
| 2 | 0.75 | 1440×810 | 16 | ~265 ms | 16.6 ms | 60 |
| 3 | 1.0 | 1920×1080 | 16 | ~405 ms | 25.3 ms | 39.5 |
| 4 | 0.25 | 480×270 | 16 | ~35 ms | 2.2 ms | 450+ |

## How to Interpret Results

### Pixel Reduction Formula
```
pixels_rendered = width * height * render_scale²

Example (render_scale=0.5):
 Original: 1920 × 1080 = 2,073,600 pixels
 Scaled: 960 × 540 = 518,400 pixels (75% reduction = 4x fewer)
```

### FPS Speedup Formula
```
speedup = 1 / render_scale²

Example (render_scale=0.5):
 speedup = 1 / 0.5² = 1 / 0.25 = 4x
```

### Amortized Per-Frame Time
With 16-frame temporal batching:
```
Per-frame = (compute_time + amortized_readback) ms
FPS = 1000 / per-frame
```

## Performance Validation

### Quick Sanity Check
- scale=0.5 should be ~4x faster than scale=1.0
- scale=0.75 should be ~1.8x faster than scale=1.0
- scale=0.25 should be ~16x faster than scale=1.0

### Example Validation
```
If scale=1.0 gives 40 FPS:
 scale=0.75 should give ~71 FPS (1.8x)
 scale=0.5 should give ~160 FPS (4x)
 scale=0.25 should give ~640 FPS (16x)
```

## Build & Compile

```bash
# Ensure Vulkan SDK is installed
gcc -std=c11 -O2 gpu_vulkan_demo.c -o gpu_demo.exe -lvulkan -lm

# Verify compilation
./gpu_demo.exe --help # or just run a test
```

## Troubleshooting

### If FPS doesn't improve with lower scale:
1. Check that new executable is being used (time-stamped)
2. Verify Vulkan is working: check for GPU-related output
3. Ensure YSU_GPU_RENDER_SCALE is being parsed (check stderr output)

### If executable hangs:
1. Try without YSU_GPU_NO_IO flag first
2. Reduce frame count to 1-2 for debugging
3. Check Vulkan validation layers for errors

### Quick Debug
```bash
# See if render scale is applied
YSU_GPU_RENDER_SCALE=0.5 ./gpu_demo.exe 2>&1 | grep -i "scale\|render"
```

## Comparison with Session 12

### Session 12 (Temporal Accumulation)
- Base: 100 ms/frame
- 16-frame batch: 25.3 ms/frame average
- **Result: 39.5 FPS** (4x improvement)

### Session 13 (Render Scale)
- scale=0.5: 25.3 ms / 4 = 6.3 ms/frame compute
- 16-frame batch: ~8.8 ms/frame average
- **Result: 80-120 FPS** (2-4x further improvement)

### Combined (Both)
- **Total: 8x-12x speedup from 10 FPS baseline**

## Next Steps

1. **Build** with Vulkan SDK: `gcc -std=c11 -O2 gpu_vulkan_demo.c -o gpu_demo.exe -lvulkan -lm`
2. **Run** test with default scale: `YSU_GPU_FRAMES=16 ./gpu_demo.exe`
3. **Measure** FPS from output time
4. **Compare** with expected ~114 FPS for scale=0.5
5. **Iterate** with different scales (0.25, 0.75, 1.0) to find optimal quality/speed balance

---

**Status**: Tests ready to run when Vulkan environment is available.
