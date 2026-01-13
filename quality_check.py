import math
import numpy as np
from PIL import Image

BASE = "output_base.png"
TEST = "output_policy.png"
DIFF_OUT = "diff.png"     # fark görseli üretmek istersen
DIFF_GAIN = 8.0           # farkı görünür yapmak için çarpan (8–20 arası deneyebilirsin)

def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

base = load_rgb(BASE)
test = load_rgb(TEST)

if base.shape != test.shape:
    raise SystemExit(f"Size mismatch: {BASE} {base.shape} vs {TEST} {test.shape}")

diff = test - base
mse = float(np.mean(diff * diff))

# PSNR: 1.0 max since we normalized to [0..1]
if mse == 0.0:
    psnr = float("inf")
else:
    psnr = 10.0 * math.log10(1.0 / mse)

print(f"MSE  : {mse:.10f}")
print(f"PSNR : {psnr:.4f} dB")

# Optional: write a visible diff image
# We shift by +0.5 to center, then scale by gain.
vis = np.clip(0.5 + diff * DIFF_GAIN, 0.0, 1.0)
vis8 = (vis * 255.0 + 0.5).astype(np.uint8)
Image.fromarray(vis8, mode="RGB").save(DIFF_OUT)
print(f"Wrote {DIFF_OUT} (visual diff, gain={DIFF_GAIN})")
