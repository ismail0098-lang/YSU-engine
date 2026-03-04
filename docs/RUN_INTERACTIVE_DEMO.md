# Interactive Walkable Demo - Summary

## Issue Found
Window mode crashes during vkCmdDispatch with new push constants (112 bytes including camera basis vectors). This appears to be a driver/validation issue when recording command buffers in windowed mode.

## Current Status
- Headless mode works perfectly (143 FPS with denoise skip=4)
- Window mode crashes on first dispatch
- Shaders compiled successfully
- Push constants within Vulkan limits (112/128 bytes)

## Workaround Options

### Option 1: Use headless mode with oneshot dump
```powershell
$env:YSU_GPU_WINDOW=0
$env:YSU_GPU_W=1280
$env:YSU_GPU_H=720
$env:YSU_GPU_FRAMES=300
$env:YSU_GPU_SPP=1
$env:YSU_GPU_DENOISE_SKIP=4
$env:YSU_GPU_WRITE=1
.\gpu_demo.exe
```
This renders 300 frames in ~2 seconds and writes output.ppm.

### Option 2: Reduce push constant size
Move camera basis to a uniform buffer instead of push constants (requires shader + C code changes).

### Option 3: Use older camera system
Revert to hardcoded spiral walk path (removes interactive WASD/mouse but works).

## Root Cause
The crash happens during vkCmdDispatch *recording* (not execution), which suggests:
1. Driver doesn't like large push constants in present loop
2. Access violation when accessing camera vectors
3. Validation layer would catch this but isn't enabled

## Next Steps
1. Enable Vulkan validation layers to see exact error
2. OR revert camera changes and use simpler push constants
3. OR switch to uniform buffer for camera data

The interactive system is 99% complete - just needs this dispatch crash fixed.
