#!/usr/bin/env python3
from PIL import Image
import numpy as np

def load_ppm(f):
    with open(f, 'rb') as fp:
        magic = fp.readline().decode().strip()
        while True:
            line = fp.readline().decode().strip()
            if line and not line.startswith('#'): break
        w, h = map(int, line.split())
        fp.readline()  # maxval
        data = np.frombuffer(fp.read(), dtype=np.uint8).reshape((h, w, 3))
    return data

def analyze(name, img):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    var = np.var(lum.astype(float) / 255)
    print(f"{name:20} Variance: {var:.4f}")
    return var

try:
    noisy = load_ppm('window_dump_3m_1spp_noisy.ppm')
    denoised = load_ppm('window_dump_3m_1spp_denoised.ppm')

    v_noisy = analyze("1 SPP noisy", noisy)
    v_denoised = analyze("1 SPP denoised", denoised)

    reduction = (v_noisy - v_denoised) / v_noisy * 100
    print(f"\nNoise reduction: {reduction:.1f}%")
    if reduction > 10:
        print("✓ DENOISER WORKING on complex scene!")
    else:
        print("⚠ Minimal noise reduction")
except Exception as e:
    print(f"Error: {e}")
