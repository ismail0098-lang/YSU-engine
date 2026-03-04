# Custom NeRF Export Format (Hash‑Grid MLP + Occupancy)

This spec defines a minimal, **inference‑only** binary format for your custom hash‑grid NeRF. It is designed for **fast GPU upload** and a simple, deterministic loader.

## 1) Files

**Required**
- models/nerf_hashgrid.bin
- models/occupancy_grid.bin

**Env Vars**
- `YSU_NERF_HASHGRID=models/nerf_hashgrid.bin`
- `YSU_NERF_OCC=models/occupancy_grid.bin`

## 2) Hash‑Grid Binary Layout (nerf_hashgrid.bin)

### Header (fixed 64 bytes)
All values are **little‑endian**.

```
struct NerfHashGridHeader {
 uint32_t magic; // 'NHG1' = 0x3147484E
 uint32_t version; // 1 or 2
 uint32_t levels; // L (e.g., 16)
 uint32_t features; // F (e.g., 2 or 4)
 uint32_t hashmap_size; // H per level (power of two)
 uint32_t base_resolution; // R0 (e.g., 16)
 float per_level_scale; // s (e.g., 1.3819)
 uint32_t mlp_in; // MLP input dim (L*F + 3 for dir)
 uint32_t mlp_hidden; // hidden width
 uint32_t mlp_layers; // number of hidden layers
 uint32_t mlp_out; // output dim (4: rgb + density)
 uint32_t flags; // v2: float bits of scene scale
 uint32_t reserved[3]; // v2: float bits of scene center (x,y,z)
};
```

### Payload (contiguous, no padding)
1. **Hash‑grid table** (float16 or float32; choose one and set a flag):
 - For each level `l` in `[0..L-1]`:
 - `hashmap_size` entries
 - each entry has `features` floats

2. **MLP weights** (float16 recommended)
 - Layer 0: `[mlp_in x mlp_hidden]` weights + `[mlp_hidden]` bias
 - Hidden layers: `[mlp_hidden x mlp_hidden]` weights + `[mlp_hidden]` bias
 - Output layer: `[mlp_hidden x mlp_out]` weights + `[mlp_out]` bias

3. **Activation**
 - ReLU for hidden
 - Density = ReLU
 - Color = Sigmoid

## 3) Occupancy Grid (occupancy_grid.bin)

### Header (16 bytes)
```
struct NerfOccHeader {
 uint32_t magic; // 'NOG1' = 0x31474F4E
 uint32_t dim; // grid dimension N (cube: N x N x N)
 float scale; // world units per grid cell
 float threshold; // occupancy threshold used during training
};
```

### Payload
- `N * N * N` bytes (uint8), row‑major, values 0 or 1

## 4) Coordinate Convention
- World space center at origin
- Hash grid assumes positions normalized to `[-1, 1]`
- Occupancy grid aligned to `[-1, 1]` cube

## 5) Export Checklist
- Ensure hash table + MLP are **inference‑only**
- Use **FP16** where possible to reduce bandwidth
- Keep `hashmap_size` power‑of‑two
- Ensure header `mlp_in` matches `levels*features + 3`

## 6) Minimal Loader Notes
- Map files, validate magic, read header
- Upload hash table + MLP to GPU buffers
- Upload occupancy grid to 3D texture or SSBO

---
If you want a template exporter script (PyTorch → this format), say the word and I will add it.
