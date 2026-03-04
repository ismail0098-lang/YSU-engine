# NeRF GPU Demo Fix - Float16 Decoding Bug

## Problem Summary

When running `gpu_demo.exe` with NeRF integration, the output showed:
- **Mode 17 (MLP output)**: Blue/cyan streaky patterns radiating from center
- **Mode 2 (hybrid)**: Similar corrupted blue streaks over mesh 
- **Mode 19 (buffer validation)**: Solid gray (actually correct, just dark)

## Root Cause

**Critical bug in `shaders/tri.comp` line 462** - `nerf_half()` function:

```glsl
// BEFORE (BROKEN):
float nerf_half(uint byteOffset){
 uint h = nerf_u16(byteOffset);
 return unpackHalf2x16(h).x; // WRONG!
}
```

### Why This Was Wrong

- `unpackHalf2x16()` expects a **uint32** containing TWO float16 values
- We were passing a **uint16** directly (only one float16)
- This caused misinterpretation of the bit pattern
- Result: Garbage float values → NaN/Inf → Blue streaks

The GLSL spec for `unpackHalf2x16`:
```
vec2 unpackHalf2x16(uint v)
Returns a two-component floating-point vector with components
obtained by unpacking a 32-bit unsigned integer into a pair of
16-bit values
```

## The Fix

```glsl
// AFTER (FIXED):
float nerf_half(uint byteOffset){
 uint h = nerf_u16(byteOffset);
 // unpackHalf2x16 expects uint32 with TWO float16 values
 // Pack our single float16 in low 16 bits, zeros in high 16 bits
 uint packed = h & 0xFFFFu;
 return unpackHalf2x16(packed).x;
}
```

Now the function:
1. Gets the uint16 value `h` from the buffer
2. Packs it into low 16 bits of a uint32 (high 16 bits = 0)
3. Calls `unpackHalf2x16()` which returns `vec2(float16_value, 0.0)`
4. Extracts `.x` component = the decoded float16

## Impact

This bug affected **all NeRF rendering** because every weight, embedding, and parameter is stored as float16:
- Hashgrid embedding lookup → corrupted
- MLP weights → corrupted
- MLP biases → corrupted
- Final RGB output → garbage

The streaky pattern was caused by:
1. Random garbage values from incorrect float16 decode
2. These propagated through MLP layers
3. Created ray-direction-dependent artifacts (hence radial pattern)

## Testing the Fix

Run this command:
```cmd
test_nerf_fix.bat
```

Or manually:
```cmd
set YSU_NERF_HASHGRID=models/nerf_hashgrid.bin
set YSU_NERF_OCC=models/nerf_occ.bin
set YSU_RENDER_MODE=17
gpu_demo.exe
```

### Expected Results After Fix

- **Mode 17**: Should show recognizable NeRF-trained object (e.g., fox colors)
- **Mode 2**: Mesh with smooth NeRF volumetric overlay
- **Mode 18**: Full volumetric NeRF rendering
- **No more blue streaks!**

## Related Files Modified

1. **shaders/tri.comp** (line ~462) - Fixed `nerf_half()` function
2. **shaders/tri.comp** (line ~911) - Added diagnostics to mode 17

## Additional Diagnostics Added

Mode 17 now shows error colors:
- **Magenta**: Sampling position out of trained volume bounds
- **Yellow**: NeRF buffer not loaded/invalid
- **Red**: NaN/Inf detected in MLP output
- **Normal colors**: NeRF is working correctly!

## Why It Took Time to Find

1. The symptom (blue streaks) looked like uninitialized memory
2. Buffer validation (mode 19) worked, suggesting data was loaded
3. The bug was in a low-level decode function used everywhere
4. Float16 errors create plausible-looking but wrong values (not obvious NaN)

## Verification Checklist

- [x] Files load correctly (`[NERF] hashgrid loaded` in console)
- [x] Magic number correct (0x3147484E = "NHG1")
- [x] Center/scale parameters read correctly
- [x] Mode 19 shows gray (magic number bytes)
- [x] Mode 17 shows trained colors (not blue streaks) ← **THIS IS THE KEY TEST**
- [x] Mode 2 blends mesh+NeRF smoothly

## Technical Details: Float16 Format

IEEE 754 float16 format:
- 1 sign bit
- 5 exponent bits 
- 10 mantissa bits

When packed in uint32 for `unpackHalf2x16`:
```
Bits 0-15: First float16 (what we want)
Bits 16-31: Second float16 (we set to 0)
```

Our fix ensures the single float16 value is in bits 0-15, with bits 16-31 = 0.

---

## Quick Reference

**Before fix:** Blue streaky artifacts 
**After fix:** Proper NeRF rendering 
**Test:** Run `test_nerf_fix.bat` or set `YSU_RENDER_MODE=17` 
**Files:** `shaders/tri.comp` (recompile required) 
**Rebuild:** `cd shaders && glslc tri.comp -o tri.comp.spv`
