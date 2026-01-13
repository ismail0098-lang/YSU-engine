#!/usr/bin/env python3
"""Check output_gpu.ppm - large HDR file"""
import struct

def check_hdr_ppm(fname, samplerate=100):
    """Sample every Nth pixel to avoid huge output"""
    try:
        with open(fname, 'rb') as f:
            magic = f.readline().decode().strip()
            while True:
                line = f.readline().decode().strip()
                if line and not line.startswith('#'): 
                    w, h = map(int, line.split())
                    break
            maxval = int(f.readline().decode().strip())
            
            print(f"{fname}: {magic}")
            print(f"  {w}x{h}, maxval={maxval}")
            
            if maxval != 255:
                # HDR format: floats
                pixels = []
                for i in range(min(100, w*h)):  # Sample first 100 pixels
                    pix = struct.unpack('ffff', f.read(16))
                    pixels.append(pix)
                
                if pixels:
                    r_vals = [p[0] for p in pixels]
                    print(f"  First 10 R values: {r_vals[:10]}")
                    print(f"  R min/max: {min(r_vals):.6f}/{max(r_vals):.6f}")
                    non_zero = sum(1 for p in pixels if any(c > 0.001 for c in p[:3]))
                    print(f"  Non-zero pixels (first 100): {non_zero}")
    except Exception as e:
        print(f"{fname}: Error - {e}")

check_hdr_ppm('output_gpu.ppm')
