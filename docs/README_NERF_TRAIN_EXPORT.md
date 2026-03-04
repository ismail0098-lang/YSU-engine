# NeRF Training + Export (Hash‑Grid MLP)

This script trains a small hash‑grid MLP NeRF on the **DataNeRF fox** dataset and exports:
- `models/nerf_hashgrid.bin`
- `models/occupancy_grid.bin`

## 1) Install dependencies
If virtual environment creation is stuck, install directly into your user site:
```powershell
py -3 -m pip install --user -r requirements.txt
```

If you prefer venv later (optional):
```powershell
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2) Train + export
```powershell
py -3 nerf_train_export.py --data DataNeRF/data/nerf/fox \
 --iters 2000 --batch_rays 1024 --n_samples 32 \
 --levels 12 --features 2 --hashmap_size 8192 \
 --base_res 16 --per_level_scale 1.5 \
 --hidden 64 --layers 2 \
 --occ_dim 64 --occ_threshold 1.0
```

## 3) Outputs
- `models/nerf_hashgrid.bin`
- `models/occupancy_grid.bin`

These follow [NERF_CUSTOM_FORMAT.md](NERF_CUSTOM_FORMAT.md).

## 4) Runtime load (gpu_demo)
```powershell
$env:YSU_NERF_HASHGRID="models/nerf_hashgrid.bin"
$env:YSU_NERF_OCC="models/occupancy_grid.bin"
```
You should see log lines like:
```
[NERF] hashgrid loaded: L=... F=... H=... base=... layers=... hidden=...
[NERF] occupancy loaded: dim=... threshold=...
```

## Notes
- This is a **minimal baseline** (slow but correct). It is not optimized.
- For faster training, increase batch size and use CUDA.
