#!/usr/bin/env python3
"""Analyze denoiser effectiveness on 3M scene"""
import numpy as np
import math

def load_ppm_uint8(fname):
    """Load 8-bit PPM into numpy array"""
    try:
        with open(fname, 'rb') as f:
            magic = f.readline().decode().strip()
            while True:
                line = f.readline().decode().strip()
                if line and not line.startswith('#'): 
                    w, h = map(int, line.split())
                    break
            maxval = int(f.readline().decode().strip())
            
            # Read uint8 data
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w, 3))
            return w, h, data.astype(float) / 255.0
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return None, None, None

def analyze_noise(img):
    """Calculate luminance variance (noise metric)"""
    if img is None:
        return 0
    
    # Convert to luminance
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    return float(np.std(lum))

print("=== 3M Mesh Denoiser Effectiveness ===\n")

w, h, noisy = load_ppm_uint8('output_3m_noisy.ppm')
print(f"1 SPP Noisy:          Loaded {w}x{h}")
if noisy is not None:
    std_noisy = analyze_noise(noisy)
    print(f"  Noise std: {std_noisy:.6f}")

w, h, denoised = load_ppm_uint8('output_3m_denoised.ppm')
print(f"\n1 SPP Denoised:       Loaded {w}x{h}")
if denoised is not None:
    std_denoised = analyze_noise(denoised)
    print(f"  Noise std: {std_denoised:.6f}")

w, h, clean = load_ppm_uint8('output_3m_8spp.ppm')
print(f"\n8 SPP Clean:          Loaded {w}x{h}")
if clean is not None:
    std_clean = analyze_noise(clean)
    print(f"  Noise std: {std_clean:.6f}")

if noisy is not None and denoised is not None:
    reduction = (std_noisy - std_denoised) / std_noisy * 100 if std_noisy > 0 else 0
    print(f"\n=== EFFECTIVENESS ===")
    print(f"Noise reduction: {reduction:.1f}%")
    if reduction > 50:
        print(f"✓ HIGHLY EFFECTIVE ({reduction:.0f}% reduction)")
    elif reduction > 20:
        print(f"✓ EFFECTIVE ({reduction:.0f}% reduction)")
    elif reduction > 0:
        print(f"⚠ WORKING but modest ({reduction:.0f}% reduction)")
    else:
        print(f"✗ NOT WORKING (no reduction)")
    
    if clean is not None and std_clean > 0.001:
        # Compare quality
        ratio = std_denoised / std_clean
        print(f"\n1 SPP denoised vs 8 SPP clean:")
        print(f"  Noise ratio: {ratio:.2f}x")
        if ratio < 1.0:
            equivalent_spp = 8 / (ratio**2) if ratio > 0 else float('inf')
            print(f"  → Equivalent to {min(equivalent_spp, 100):.1f}x SPP!")
