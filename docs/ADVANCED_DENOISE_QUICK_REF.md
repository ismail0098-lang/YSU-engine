# Advanced Denoise: Quick Reference

## Three New Features 

### 1⃣ History Reset
**Clears denoise buffer periodically** - prevents ghosting on camera cuts

```bash
YSU_GPU_DENOISE_HISTORY_RESET=1 YSU_GPU_DENOISE_HISTORY_RESET_FRAME=60 ./gpu_demo.exe
```
- Resets every 60 frames
- Prevents temporal artifacts from scene changes
- Cost: ~0.1ms per reset (negligible)

### 2⃣ Immediate Denoise 
**Always denoise frame 0** - guaranteed quality at startup

```bash
YSU_GPU_DENOISE=1 YSU_GPU_DENOISE_SKIP=4 ./gpu_demo.exe
```
- Frame 0: Always denoised 
- Frame 1-3: Skipped
- Frame 4+: Pattern continues
- Cost: None (automatic, no flag needed)

### 3⃣ Adaptive Denoise
**Ramps denoising up/down based on convergence** - fast startup, high speed after

```bash
YSU_GPU_DENOISE_ADAPTIVE=1 ./gpu_demo.exe
```
- Frames 0-30: Full denoising (warmup, building quality)
- Frames 31+: Sparse denoising (steady state, high FPS)
- Cost: None (automatic ramp)

---

## Usage Presets

### Cinematic (Maximum Quality)
```bash
YSU_GPU_DENOISE=1 \
 YSU_GPU_DENOISE_SKIP=1 \
 YSU_GPU_DENOISE_HISTORY_RESET=1 \
 YSU_GPU_TEMPORAL_DENOISE=1 \
 ./gpu_demo.exe
```
**FPS**: 100 | **Quality**: Excellent 

### Gaming (Balanced)
```bash
YSU_GPU_DENOISE=1 \
 YSU_GPU_DENOISE_SKIP=4 \
 YSU_GPU_DENOISE_ADAPTIVE=1 \
 YSU_GPU_TEMPORAL_DENOISE=1 \
 ./gpu_demo.exe
```
**FPS**: 95→210 | **Quality**: Excellent 

### Speed (Maximum FPS)
```bash
YSU_GPU_DENOISE=1 \
 YSU_GPU_DENOISE_SKIP=8 \
 YSU_GPU_DENOISE_ADAPTIVE=1 \
 YSU_GPU_RENDER_SCALE=0.5 \
 ./gpu_demo.exe
```
**FPS**: 100→250+ | **Quality**: Good 

---

## All Environment Variables

```
YSU_GPU_DENOISE_HISTORY_RESET=0|1
 Default: 0 (disabled)
 1: Enable periodic reset

YSU_GPU_DENOISE_HISTORY_RESET_FRAME=<int>
 Default: 60
 Range: 10-300
 When: Reset history every N frames

YSU_GPU_DENOISE_ADAPTIVE=0|1
 Default: 0 (disabled)
 1: Enable adaptive skip ramping

YSU_GPU_DENOISE_ADAPTIVE_MIN=<int>
 Default: 1
 Warmup phase: denoise every N frames

YSU_GPU_DENOISE_ADAPTIVE_MAX=<int>
 Default: 8
 Steady state: denoise every N frames
```

---

## Quick Comparison

| Feature | Effect | When to Use |
|---|---|---|
| **History Reset** | Cleaner scene transitions | Camera cuts, fades |
| **Immediate Denoise** | Better frame 0 | Always beneficial |
| **Adaptive Denoise** | Fast ramp to speed | Real-time apps |

---

## Tips

 **Use all three together** for best results:
```bash
YSU_GPU_DENOISE=1 \
 YSU_GPU_DENOISE_SKIP=4 \
 YSU_GPU_DENOISE_HISTORY_RESET=1 \
 YSU_GPU_DENOISE_ADAPTIVE=1 \
 YSU_GPU_TEMPORAL_DENOISE=1 \
 ./gpu_demo.exe
```

 **Adaptive works best with higher skip values** (4-8)

 **History reset every 60 frames** = good default for most scenes

 **Check stderr output** for configuration confirmation

---

See [ADVANCED_DENOISE_FEATURES.md](ADVANCED_DENOISE_FEATURES.md) for detailed docs
