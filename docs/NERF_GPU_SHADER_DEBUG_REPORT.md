# NeRF GPU Shader Debugging Report

**Date:** January 21, 2026 
**Duration:** 4 days 
**Status:** RESOLVED 

---

## Executive Summary

The GPU Vulkan shader for rendering NeRF (Neural Radiance Fields) models was producing corrupted output - appearing as blue/purple noise blobs instead of the expected yellow Lego bulldozer. After extensive debugging, three critical bugs were identified and fixed in the GLSL compute shader.

---

## Initial Symptoms

1. **Visual Corruption:** Output showed psychedelic blue/cyan/purple noise patterns
2. **Blobby Appearance:** No recognizable shape, just amorphous colored blobs
3. **Wrong Colors:** When shape became visible, colors were inverted (blue instead of yellow)

### Sample Progression of Visual Bugs:
- First attempt: Neon blue/cyan noise with vertical streaks
- After density tuning: Melted/foggy blob with scattered particles 
- After hash fix: Correct shape but BLUE instead of YELLOW
- Final fix: Correct yellow Lego bulldozer 

---

## Debugging Methodology

### Step 1: Verify the Training Pipeline
We first confirmed that the Python training script (`nerf_train_and_export.py`) was working correctly:

```bash
python nerf_train_and_export.py --data DataNeRF/data/nerf/lego --iters 20000
```

**Result:** Training completed successfully with loss converging to 0.002

### Step 2: Verify the Trained Model in Python
We rendered the trained model using pure Python to isolate whether the bug was in training or in the GPU shader:

```python
# Render test in Python
python -c "... full NeRF model loading and rendering ..."
```

**Result:** Python rendered a perfect yellow Lego bulldozer! This proved:
- Training pipeline is correct
- Binary export format is correct 
- GPU shader has bugs

### Step 3: Compare Python vs Shader Logic
We systematically compared every calculation:
- Hash function
- Memory layout indexing
- Trilinear interpolation
- MLP weight loading
- Activation functions (sigmoid, softplus)
- Color channel ordering

---

## Bugs Found and Fixed

### Bug #1: Hash Grid Memory Indexing (CRITICAL)

**Location:** `shaders/tri.comp` line ~530-545

**Problem:** The shader calculated hash table memory offsets incorrectly.

**Memory Layout in File:**
```
[Header: 60 bytes]
[Level 0: hashmap_size * features * 2 bytes]
[Level 1: hashmap_size * features * 2 bytes]
...
[Level N: hashmap_size * features * 2 bytes]
[MLP weights...]
```

**Wrong Code:**
```glsl
// WRONG: Multiplied in wrong order
float v000 = nerf_half(header_bytes + (get_hash(...) * entry_size + (l * hashmap_size * entry_size) + f) * 2u);
```

**Correct Code:**
```glsl
// CORRECT: level offset first, then hash, then feature
uint base_l = l * hashmap_size * features;
float v000 = nerf_half(header_bytes + (base_l + get_hash(...) * features + f) * 2u);
```

**Impact:** This caused the shader to read garbage data from wrong memory locations, producing random noise colors.

---

### Bug #2: Resolution Calculation Mismatch

**Location:** `shaders/tri.comp` line ~527

**Problem:** The shader used `(resolution - 1)` but Python uses `resolution` directly.

**Wrong Code:**
```glsl
vec3 grid = pn * (res - 1.0); // WRONG
```

**Correct Code:**
```glsl
vec3 grid = pn * res; // Matches Python exactly
```

**Impact:** This caused spatial misalignment - the shader was sampling slightly different positions than what was trained.

---

### Bug #3: RGB Channel Order Swap (BGR vs RGB)

**Location:** `shaders/tri.comp` line ~650

**Problem:** The MLP outputs RGB in order [R, G, B] but the shader was interpreting it as [B, G, R].

**Wrong Code:**
```glsl
return vec4(rgb, sigma); // rgb = vec3(outv[0], outv[1], outv[2])
```

**Correct Code:**
```glsl
return vec4(rgb.bgr, sigma); // Swap R and B channels
```

**Impact:** Yellow (high R, medium G, low B) appeared as blue (high B, medium G, low R).

---

## Files Modified

| File | Changes |
|------|---------|
| `shaders/tri.comp` | Fixed hash indexing, resolution calc, RGB order |
| `run_lego_gpu.bat` | Adjusted density and step parameters |

---

## Verification

After applying all fixes:

1. **Rebuild shaders:**
 ```powershell
 .\build_shaders.ps1
 ```

2. **Run viewer:**
 ```powershell
 .\run_lego_gpu.bat
 ```

3. **Result:** Yellow Lego bulldozer renders correctly in real-time! 

---

## Lessons Learned

1. **Always verify intermediate outputs:** Rendering in Python first isolated the bug to the shader layer.

2. **Memory layout documentation is critical:** The hash grid indexing bug was caused by misunderstanding the memory layout.

3. **Small calculation differences matter:** Even `res` vs `res-1` caused visible artifacts.

4. **Check channel ordering:** RGB/BGR swaps are common when crossing language/API boundaries.

5. **Density multipliers mask bugs:** High density values (100-500) made noise visible; the real fix was correct hash indexing, not density tuning.

---

## Final Configuration

**run_lego_gpu.bat settings that work:**
```batch
set YSU_GPU_SPP=4
set YSU_GPU_WINDOW=1
set YSU_GPU_TEMPORAL=1
set YSU_NERF_STEPS=128
set YSU_NERF_DENSITY=1.0
set YSU_NERF_SCALE=1.5
set YSU_NERF_SKIP_OCC=1
```

---

## Appendix: Key Code Snippets

### Correct Hash Grid Embedding (GLSL)
```glsl
for(uint l = 0u; l < levels; l++){
 float res = float(int(float(base_res) * pow(per_level_scale, float(l))));
 vec3 grid = pn * res;
 vec3 pos_f = floor(grid);
 ivec3 gi = ivec3(pos_f);
 vec3 w = grid - pos_f;

 for(uint f = 0u; f < features; f++){
 uint base_l = l * hashmap_size * features;
 float v000 = nerf_half(header_bytes + (base_l + get_hash(gi.x, gi.y, gi.z, hashmap_size) * features + f) * 2u);
 // ... 7 more corners for trilinear interpolation
 }
}
```

### Python Equivalent (Reference)
```python
for l in range(self.levels):
 res = int(self.base_res * (self.per_level_scale ** l))
 pos = x * res
 pos_floor = torch.floor(pos).long()
 # ... trilinear interpolation
 table = self.tables[l] # tables[level][hash_index][feature]
```

---

**Report generated after successful debugging session.** 
**The Lego bulldozer is now yellow as intended. **
