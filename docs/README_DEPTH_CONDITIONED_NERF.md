# Depth-Conditioned Heterogeneous NeRF Rendering

A novel real-time NeRF rendering technique that uses CPU-computed depth hints from an occupancy grid to guide GPU ray marching, reducing sample counts by 4x for rays with known depth.

## Overview

Traditional NeRF rendering uses uniform sampling (e.g., 32 steps) along every ray. This work introduces **heterogeneous sampling** where:
- Rays with high-confidence depth hints use **8 samples** in a narrow band
- Rays without depth hints use the full **32 samples**

This achieves **10-40% speedup** with negligible quality loss.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Frame N │
├─────────────────────────────────────────────────────────────────┤
│ CPU: Occupancy Grid Prepass (1/4 resolution) │
│ ├── Load 64³ occupancy grid from training │
│ ├── Ray march 16 coarse steps per pixel block │
│ └── Output: depth hints buffer (vec4 per pixel) │
├─────────────────────────────────────────────────────────────────┤
│ GPU: Depth-Conditioned NeRF Integration │
│ ├── Read depth hint for pixel │
│ ├── If confidence > 0.5: sample [depth-δ, depth+δ] with 8 steps│
│ └── Else: sample full [t_near, t_far] with 32 steps │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Train a NeRF Model

```powershell
# Activate Python environment
.\.venv\Scripts\Activate.ps1

# Train on Lego scene (35k iterations for good quality)
python nerf_instant_ngp_fixed.py `
 --data nerf-synthetic/nerf_synthetic/lego `
 --iters 35000 `
 --hidden 64 `
 --levels 16 `
 --out_hashgrid models/lego.bin `
 --out_occ models/lego_occ.bin
```

### 2. Build the Renderer

```powershell
# Compile shaders
C:\VulkanSDK\1.4.335.0\Bin\glslangValidator.exe -V shaders/tri.comp -o shaders/tri.comp.spv

# Build executable
gcc -O3 -march=native -ffast-math -I./include -I./raylib/include -L./raylib/lib `
 -o gpu_vulkan_demo.exe `
 gpu_vulkan_demo.c nerf_scheduler.c nerf_batch.c neural_denoise.c `
 bilateral_denoise.c bvh.c vec3.c sphere.c ray.c `
 gpu_bvh_build.c gpu_bvh_lbv.c gpu_bvh_lbvh_builder.c `
 -lglfw3 -lraylib -lgdi32 -lwinmm -lopengl32 -lvulkan-1
```

### 3. Run

```powershell
# Mode 3: Baseline NeRF (32 uniform steps)
$env:YSU_GPU_WINDOW=1
$env:YSU_RENDER_MODE=3
$env:YSU_NERF_HASHGRID="models/lego.bin"
$env:YSU_NERF_OCC="models/lego_occ.bin"
.\gpu_vulkan_demo.exe

# Mode 26: Depth-Conditioned NeRF (8-32 adaptive steps)
$env:YSU_RENDER_MODE=26
.\gpu_vulkan_demo.exe
```

## Render Modes

| Mode | Description | Steps/Ray |
|------|-------------|-----------|
| 3 | Baseline NeRF | 32 uniform |
| 26 | Depth-conditioned NeRF | 8-32 adaptive |
| 27 | Debug visualization (depth hints) | N/A |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YSU_GPU_WINDOW` | 0 | Enable windowed mode (1) or headless (0) |
| `YSU_GPU_W` | 1920 | Window width |
| `YSU_GPU_H` | 1080 | Window height |
| `YSU_GPU_RENDER_SCALE` | 1.0 | Internal render scale (0.5 = half res) |
| `YSU_RENDER_MODE` | 3 | Render mode (see table above) |
| `YSU_NERF_HASHGRID` | - | Path to hashgrid .bin file |
| `YSU_NERF_OCC` | - | Path to occupancy .bin file |
| `YSU_NERF_STEPS` | 32 | Base step count |
| `YSU_NERF_CENTER_X/Y/Z` | 0.0 | Scene center offset |
| `YSU_NERF_SCALE` | 1.5 | Scene scale |

## File Formats

### Hashgrid Binary (.bin)
```
Header (64 bytes):
 - uint32 L: number of levels
 - uint32 F: features per level
 - uint32 H: hashmap size
 - uint32 base_resolution
 - uint32 num_layers
 - uint32 hidden_dim
 - float center[3]
 - float scale
 
Data:
 - float[L * H * F]: hashgrid features
 - float[...]: MLP weights
```

### Occupancy Binary (.bin)
```
Header (16 bytes):
 - uint32 dim: grid dimension (e.g., 64)
 - float threshold
 - uint32 reserved[2]
 
Data:
 - uint8[dim³]: occupancy values (0-255)
```

## Performance

Tested on RTX 3080, 1024x512 resolution:

| Scene | Mode 3 (baseline) | Mode 26 (depth-cond) | Speedup |
|-------|-------------------|----------------------|---------|
| Lego (5k iter) | 39.9 FPS | 44.5 FPS | 11% |
| Lego (35k iter) | ~40 FPS | ~52 FPS | ~30% |

*Note: Results with vsync disabled. Higher hit rates with better-trained models yield larger speedups.*

## Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move camera |
| Mouse | Look around |
| R | Reset camera |
| M | Cycle render mode |
| Esc | Quit |

## Citation

If you use this work, please cite:

```bibtex
@misc{ysu_depth_nerf_2026,
 title={Depth-Conditioned Heterogeneous Sampling for Real-Time NeRF},
 author={YSU Engine Contributors},
 year={2026},
 howpublished={\url{https://github.com/...}}
}
```

## License

MIT License - see LICENSE file.
