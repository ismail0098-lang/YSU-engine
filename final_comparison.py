#!/usr/bin/env python3
"""Final denoiser comparison on 3M mesh"""
import numpy as np
import math

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

def calc_metrics(img):
    """Calculate image metrics"""
    h, w = img.shape[:2]
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    
    # Noise (std dev)
    noise = float(np.std(lum))
    
    # Local variance (high-pass energy)
    kx, ky = np.gradient(lum)
    edges = np.sqrt(kx**2 + ky**2)
    edge_energy = float(np.mean(edges))
    
    # Texture (unique colors)
    unique = len(np.unique(img.reshape(-1, 3), axis=0))
    
    return {
        'noise': noise,
        'edge_energy': edge_energy,
        'unique_colors': unique,
        'min': img.min(),
        'max': img.max(),
    }

print("=" * 70)
print("BILATERAL DENOISER COMPARISON ON 3M MESH")
print("=" * 70)

try:
    w, h, noisy = load_ppm('output_3m_noisy.ppm')
    w, h, denoised = load_ppm('output_3m_denoised.ppm')
    w, h, clean = load_ppm('output_3m_8spp.ppm')
    
    m_noisy = calc_metrics(noisy)
    m_denoised = calc_metrics(denoised)
    m_clean = calc_metrics(clean)
    
    print(f"\nImage size: {w}x{h}")
    print("\n1 SPP NOISY (baseline):")
    print(f"  Noise: {m_noisy['noise']:.6f}")
    print(f"  Edge energy: {m_noisy['edge_energy']:.6f}")
    print(f"  Unique colors: {m_noisy['unique_colors']}")
    
    print("\n1 SPP DENOISED (bilateral filter):")
    print(f"  Noise: {m_denoised['noise']:.6f}")
    print(f"  Edge energy: {m_denoised['edge_energy']:.6f}")
    print(f"  Unique colors: {m_denoised['unique_colors']}")
    
    print("\n8 SPP CLEAN (reference):")
    print(f"  Noise: {m_clean['noise']:.6f}")
    print(f"  Edge energy: {m_clean['edge_energy']:.6f}")
    print(f"  Unique colors: {m_clean['unique_colors']}")
    
    # Comparisons
    print("\n" + "=" * 70)
    print("DENOISER EFFECTIVENESS")
    print("=" * 70)
    
    noise_reduction = (m_noisy['noise'] - m_denoised['noise']) / m_noisy['noise'] * 100 if m_noisy['noise'] > 0.001 else 0
    edge_change = (m_denoised['edge_energy'] - m_noisy['edge_energy']) / m_noisy['edge_energy'] * 100 if m_noisy['edge_energy'] > 0.001 else 0
    
    print(f"\nNoise reduction: {noise_reduction:.2f}%")
    print(f"Edge preservation: {edge_change:.2f}% change")
    
    if noise_reduction > 10:
        print(f"✓ DENOISER EFFECTIVE (reduced noise by {noise_reduction:.1f}%)")
    elif noise_reduction > 0:
        print(f"⚠ DENOISER WORKING but modest effect ({noise_reduction:.1f}% reduction)")
    else:
        print(f"✗ NO NOISE REDUCTION (denoiser may not be needed or input is deterministic)")
    
    # Quality equivalence
    print("\n1 SPP DENOISED vs 8 SPP CLEAN:")
    print(f"  Noise ratio: {m_denoised['noise'] / m_clean['noise']:.2f}x")
    print(f"  Edge ratio: {m_denoised['edge_energy'] / m_clean['edge_energy']:.2f}x")
    
    # Pixel-level analysis
    diff = np.abs(noisy - denoised)
    changed = np.sum(np.any(diff > 0.001, axis=2))
    print(f"\nPixel-level changes:")
    print(f"  Pixels modified: {changed} / {w*h} ({100*changed/(w*h):.1f}%)")
    print(f"  Avg change: {np.mean(diff):.6f}")
    print(f"  Max change: {np.max(diff):.6f}")
    
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
