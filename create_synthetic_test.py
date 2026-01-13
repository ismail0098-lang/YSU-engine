#!/usr/bin/env python3
"""Test denoiser with synthetic noise added to cube rendering"""
import numpy as np
import sys

def load_ppm(fname):
    with open(fname, 'rb') as f:
        f.readline()
        line = b''
        while not line.rstrip():
            line = f.readline()
            if not line.startswith(b'#'): break
        w, h = map(int, line.split())
        f.readline()
        return w, h, np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w, 3)).astype(float) / 255.0

def add_gaussian_noise(img, noise_level):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_level, img.shape)
    noisy = np.clip(img + noise, 0, 1)
    return noisy

def save_ppm(fname, w, h, img):
    """Save image as PPM"""
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    with open(fname, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img_uint8.tobytes())

def calc_noise(img):
    """Calculate noise std dev"""
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    return float(np.std(lum))

# Load the clean cube rendering
print("Loading clean cube rendering...")
w, h, clean = load_ppm('output_gpu.ppm')
print(f"Cube image: {w}x{h}")

# Add noise to create test case
print("\nCreating synthetic test case...")
np.random.seed(42)
noisy = add_gaussian_noise(clean, 0.05)  # Add 5% Gaussian noise

print(f"Clean noise level: {calc_noise(clean):.6f}")
print(f"Noisy (added noise) level: {calc_noise(noisy):.6f}")

# Save test files
save_ppm('test_clean.ppm', w, h, clean)
save_ppm('test_noisy.ppm', w, h, noisy)

print("\nSynthetic test files created:")
print("  test_clean.ppm - original clean cube")
print("  test_noisy.ppm - cube with added Gaussian noise")
print("\nNext steps:")
print("  1. Denoise test_noisy.ppm using the bilateral filter")
print("  2. Compare with test_clean.ppm")
