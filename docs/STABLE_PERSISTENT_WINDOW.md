# Stable Persistent Interactive Window 

## What Changed

**Window Mode Now Renders Continuously Until ESC**

Previous behavior:
- Window would render `YSU_GPU_FRAMES` samples per frame and then close
- Each environment variable change required restarting the program
- Not ideal for interactive exploration

New behavior:
- Window stays open indefinitely after starting
- Renders 1 frame per iteration for responsive input
- Press **ESC to quit** (or close window button)
- YSU_GPU_FRAMES is ignored in window mode
- WASD movement and mouse look work smoothly each frame

---

## Implementation

**gpu_vulkan_demo.c** (lines 610-611)
```c
// In window mode, always use 1 frame per iteration for responsive input
// (YSU_GPU_FRAMES is ignored in window mode; use ESC to quit)
if(window_enabled) frames = 1;
```

**Window Render Loop** (lines 2070+)
```c
while(window && !glfwWindowShouldClose(window)){
 // Each iteration:
 // - Input handling (WASD, mouse)
 // - Update camera UBO
 // - Render 1 frame
 // - Present to screen
 // - Increment frame_id
 // - Loop continues until ESC or window close
}
```

**Single Frame Dispatch** (lines 2156-2179)
```c
// Render 1 sample per frame in interactive window mode for responsive input
{
 PushConstants pc_push = {0};
 pc_push.frame = frame_id; // Advances each iteration
 // ... dispatch raytrace
}
frame_id += frames; // frames=1 in window mode, so frame_id++
```

---

## Usage

### Start Interactive Window
```powershell
$env:YSU_GPU_WINDOW=1
.\gpu_demo.exe
```

Window opens and continuously renders. Use controls:
- **WASD**: Move camera
- **Mouse**: Look around
- **ESC**: Quit application

### With Custom Resolution
```powershell
$env:YSU_GPU_WINDOW=1
$env:YSU_GPU_W=1920
$env:YSU_GPU_H=1080
.\gpu_demo.exe
```

### With Camera Speed Control
```powershell
$env:YSU_GPU_WINDOW=1
$env:YSU_CAM_SPEED=5.0 # Faster movement (default 3.0)
$env:YSU_CAM_MOUSE_SENS=0.002 # Faster mouse (default 0.0025)
.\gpu_demo.exe
```

---

## Performance

- **Window stays open**: Indefinite rendering until ESC
- **Responsive**: 1 frame per iteration for smooth input
- **FPS**: Varies by resolution (60-120+ FPS typical)
- **No frame batching**: YSU_GPU_FRAMES ignored in window mode

---

## Controls Reference

| Key | Action |
|-----|--------|
| `W` | Move forward |
| `S` | Move backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `SPACE` | Move up |
| `LEFT_SHIFT` | Move down |
| `ESC` | Quit window |
| `Mouse X` | Turn left/right |
| `Mouse Y` | Look up/down |

---

## Key Differences from Headless Mode

| Feature | Headless | Window |
|---------|----------|--------|
| **Persistence** | Fixed frames | Infinite until ESC |
| **Input** | None | WASD + Mouse |
| **Camera Control** | Scripted path | Interactive |
| **YSU_GPU_FRAMES** | Used (frame batching) | Ignored (always 1/iter) |
| **Output** | File writes | Screen display |
| **Use Case** | Rendering, testing | Interactive exploration |

---

## Technical Note

The key change was forcing `frames = 1` in window mode, which:
1. Simplifies the dispatch loop (no accumulation)
2. Makes input responsive (frame_id increments every iteration)
3. Allows continuous rendering (no frame count limit)
4. Keeps the while loop running until ESC/window close

This provides a proper interactive experience like a traditional FPS camera controller.
