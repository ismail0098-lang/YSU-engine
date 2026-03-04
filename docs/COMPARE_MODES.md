# Compare Modes (Mesh / Probe / Hybrid)

This project now supports three runtime render modes for comparison:

## Modes
- **0 = Mesh** (default)
 - Pure mesh ray tracing (current pipeline).
- **1 = Probe (placeholder)**
 - Procedural sky probe for baseline comparison. No mesh traversal.
- **2 = Hybrid (mesh + NeRF proxy)**
 - Mesh ray tracing blended with a lightweight volumetric proxy.

## Environment Variables
- `YSU_RENDER_MODE`:
 - `0` = mesh
 - `1` = probe placeholder
 - `2` = hybrid
- `YSU_NERF_PROXY` (optional): set to `1` to enable proxy even if render mode is 0
- `YSU_NERF_BLEND`: 0..1 blend factor (0 = mesh only, 1 = proxy only)
- `YSU_NERF_STRENGTH`: proxy intensity multiplier
- `YSU_NERF_DENSITY`: proxy density (controls opacity)
- `YSU_NERF_BOUNDS`: proxy max distance (volume extent)
- `YSU_NERF_STEPS`: ray-march steps (1..32)

## Examples

**Mesh only**
```powershell
$env:YSU_RENDER_MODE=0
$env:YSU_GPU_OBJ="real_scene_room.obj"
.\gpu_demo.exe
```

**Probe placeholder**
```powershell
$env:YSU_RENDER_MODE=1
.\gpu_demo.exe
```

**Hybrid (mesh + proxy)**
```powershell
$env:YSU_RENDER_MODE=2
$env:YSU_NERF_BLEND=0.35
$env:YSU_NERF_STRENGTH=1.0
$env:YSU_NERF_DENSITY=1.0
$env:YSU_NERF_BOUNDS=8.0
$env:YSU_NERF_STEPS=6
$env:YSU_GPU_OBJ="real_scene_room.obj"
.\gpu_demo.exe
```

## Notes
- Probe mode is a **placeholder** (procedural sky) for benchmarking pipeline cost without geometry.
- Hybrid mode is a **research scaffold** — replace the proxy with your custom NeRF backend later.
