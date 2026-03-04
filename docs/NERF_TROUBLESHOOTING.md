# NeRF GPU Demo Troubleshooting Guide

## Problem: "Weird" / Incorrect Image Output

### Root Causes

1. **Missing environment variables** - NeRF buffers not loaded
2. **Uninitialized GPU buffers** - Shader reads garbage data
3. **Wrong render mode** - Not configured to use NeRF
4. **Incorrect scale/center** - NeRF volume positioned incorrectly
5. **MLP weights corrupted** - Model file has wrong format/endianness

---

## Quick Fix

Run this in PowerShell:

```powershell
$env:YSU_NERF_HASHGRID="models/nerf_hashgrid.bin"
$env:YSU_NERF_OCC="models/nerf_occ.bin"
$env:YSU_RENDER_MODE="2"
$env:YSU_NERF_STEPS="64"
$env:YSU_W="1280"
$env:YSU_H="720"

.\gpu_demo.exe
```

Or use the provided batch file:
```cmd
run_gpu_nerf.bat
```

---

## Diagnostic Steps

### Step 1: Verify Files Exist
```powershell
Test-Path models/nerf_hashgrid.bin
Test-Path models/nerf_occ.bin
```
Both should return `True`

### Step 2: Check File Sizes
```powershell
(Get-Item models/nerf_hashgrid.bin).Length
(Get-Item models/nerf_occ.bin).Length
```
HashGrid should be ~hundreds of KB, occupancy ~few KB

### Step 3: Run Debug Mode 19 (Buffer Validation)
```cmd
set YSU_NERF_HASHGRID=models/nerf_hashgrid.bin
set YSU_RENDER_MODE=19
gpu_demo.exe
```
**Expected:** Brownish-gray screen (RGB ≈ 0.3, 0.28, 0.28)
**If magenta/pink:** Buffer not loaded or wrong magic number

### Step 4: Run Debug Mode 20 (Push Constants)
```cmd
set YSU_RENDER_MODE=20
gpu_demo.exe
```
**Expected:** Color varies with center/scale values
**If black or solid color:** Parameters not set correctly

### Step 5: Run Debug Mode 17 (MLP Output)
```cmd
set YSU_RENDER_MODE=17
gpu_demo.exe
```
**Expected:** Should show colors from your trained NeRF model
**If black/white/noise:** MLP weights corrupted or wrong format

### Step 6: Run Debug Mode 18 (Full Integration)
```cmd
set YSU_RENDER_MODE=18
gpu_demo.exe
```
**Expected:** Volume-rendered NeRF output
**If weird:** Check nerfSteps, nerfBounds, nerfScale

### Step 7: Run Hybrid Mode 2
```cmd
set YSU_RENDER_MODE=2
set YSU_NERF_BLEND=0.35
gpu_demo.exe
```
**Expected:** Mesh with NeRF overlay (65% mesh, 35% NeRF)

---

## Common Issues & Fixes

### Issue: "Weird colors" or "Corrupted image"

**Cause:** NeRF buffers not loaded (reading uninitialized GPU memory)

**Fix:**
```cmd
set YSU_NERF_HASHGRID=models/nerf_hashgrid.bin
set YSU_NERF_OCC=models/nerf_occ.bin
```

---

### Issue: "Black screen"

**Causes:**
1. NeRF volume positioned outside camera view
2. Occupancy grid filtering everything out
3. nerfSteps too low

**Fixes:**
```cmd
REM Try skipping occupancy grid
set YSU_NERF_SKIP_OCC=1

REM Increase integration steps
set YSU_NERF_STEPS=128

REM Check center/scale (use values from model training)
set YSU_NERF_CENTER_X=0.0
set YSU_NERF_CENTER_Y=0.0
set YSU_NERF_CENTER_Z=0.0
set YSU_NERF_SCALE=1.0
```

---

### Issue: "Mesh visible but no NeRF"

**Cause:** Render mode is 0 (mesh only)

**Fix:**
```cmd
set YSU_RENDER_MODE=2
```

---

### Issue: "NeRF visible but very faint"

**Cause:** nerfBlend too low or nerfStrength too low

**Fix:**
```cmd
set YSU_NERF_BLEND=0.5
set YSU_NERF_STRENGTH=1.5
set YSU_NERF_DENSITY=2.0
```

---

### Issue: "Noisy / grainy NeRF output"

**Cause:** nerfSteps too low (undersampling)

**Fix:**
```cmd
set YSU_NERF_STEPS=128
```

---

### Issue: "NeRF cuts off / clipped"

**Cause:** nerfBounds too small

**Fix:**
```cmd
set YSU_NERF_BOUNDS=16.0
```

---

## Environment Variable Reference

### Required (for NeRF to work)
- `YSU_NERF_HASHGRID` - Path to hashgrid .bin file
- `YSU_NERF_OCC` - Path to occupancy .bin file
- `YSU_RENDER_MODE` - Render mode (2=hybrid, 3=nerf-only)

### Optional (NeRF parameters)
- `YSU_NERF_BLEND` - Mesh/NeRF blend (0.0-1.0, default 0.35)
- `YSU_NERF_STRENGTH` - Color multiplier (default 1.0)
- `YSU_NERF_DENSITY` - Volume density multiplier (default 1.0)
- `YSU_NERF_BOUNDS` - Max raymarch distance (default 8.0)
- `YSU_NERF_STEPS` - Integration steps (default 64, higher=better quality)
- `YSU_NERF_SKIP_OCC` - Skip occupancy culling (0/1, default 0)
- `YSU_NERF_CENTER_X/Y/Z` - Volume center (auto from .bin if not set)
- `YSU_NERF_SCALE` - Volume scale (auto from .bin if not set)

### Debug modes
- `YSU_RENDER_MODE=7` - BVH heatmap
- `YSU_RENDER_MODE=8` - BVH depth
- `YSU_RENDER_MODE=13` - Occupancy grid
- `YSU_RENDER_MODE=14` - Hashgrid lookup
- `YSU_RENDER_MODE=17` - MLP output (best for verifying model works)
- `YSU_RENDER_MODE=18` - Full NeRF integration
- `YSU_RENDER_MODE=19` - Buffer validation (magic number)
- `YSU_RENDER_MODE=20` - Push constant validation

---

## Shader Code Reference

The NeRF integration lives in `shaders/tri.comp`:

- **Lines 405-670**: NeRF buffer sampling (hashgrid + occupancy + MLP)
- **Lines 900-970**: Main render loop with hybrid blend
- **Lines 378-403**: Procedural NeRF proxy (fallback if no buffers)

Key functions:
- `nerf_hash_valid()` - Check if buffer loaded (line 420)
- `nerf_mlp_eval()` - Run MLP inference (line 546)
- `nerf_buffer_integrate()` - Volume raymarch (line 609)
- `hashgrid_embed()` - Spatial hash lookup (line 470)

---

## Training New Models

If you need to retrain or export a new NeRF model:

```bash
python nerf_train_export.py \
 --data data/fox \
 --out_hashgrid models/nerf_hashgrid.bin \
 --out_occ models/nerf_occ.bin \
 --epochs 100
```

The exporter creates binary files with this structure:
- **HashGrid header**: 60 bytes (magic, version, dimensions, MLP config)
- **Hash table**: L × H × F × 2 bytes (float16 features)
- **MLP weights**: (W×H + H) × 2 bytes per layer (float16)
- **Occupancy header**: 16 bytes
- **Occupancy grid**: (D³ / 4) bytes (4 samples per byte, uint8)

---

## Next Steps

1. **Run [run_gpu_nerf_debug.bat](run_gpu_nerf_debug.bat)** to diagnose issues
2. **Use mode 17** to verify MLP outputs correct colors
3. **Use mode 19** to verify buffer is loaded
4. **Use mode 2** for final hybrid rendering
5. **Adjust YSU_NERF_STEPS** (higher = better quality, slower)
6. **Adjust YSU_NERF_BLEND** (how much NeRF vs mesh)

If issues persist, check:
- Console output for "[NERF] hashgrid loaded: L=... F=... H=..."
- File sizes of .bin files (should match expected format)
- Endianness (if trained on different platform)
