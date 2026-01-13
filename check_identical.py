#!/usr/bin/env python3
import numpy as np

def load_ppm(f):
    with open(f, 'rb') as fp:
        fp.readline()  # magic
        while True:
            line = fp.readline().decode().strip()
            if line and not line.startswith('#'): break
        w, h = map(int, line.split())
        fp.readline()  # maxval
        data = np.frombuffer(fp.read(), dtype=np.uint8).reshape((h, w, 3))
    return data

noisy = load_ppm('window_dump_3m_1spp_noisy.ppm')
denoised = load_ppm('window_dump_3m_1spp_denoised.ppm')

identical = np.array_equal(noisy, denoised)
diff = np.abs(noisy.astype(int) - denoised.astype(int)).max()
diff_pixels = np.sum(noisy != denoised)

print(f"Files identical: {identical}")
print(f"Max pixel diff: {diff}")
print(f"Pixels that differ: {diff_pixels} / {noisy.size}")
print(f"Noisy min/max: {noisy.min()}/{noisy.max()}")
print(f"Denoised min/max: {denoised.min()}/{denoised.max()}")

if noisy.min() == 0 and noisy.max() == 0:
    print("\n⚠ ERROR: Both images are BLACK (all zeros)")
