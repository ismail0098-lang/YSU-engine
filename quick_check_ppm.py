#!/usr/bin/env python3
"""Quick check of PPM file contents"""
import struct

def check_ppm(fname):
    try:
        with open(fname, 'rb') as f:
            magic = f.readline().decode().strip()
            print(f"{fname}: {magic}")
            
            # Read width and height
            while True:
                line = f.readline().decode().strip()
                if line and not line.startswith('#'): 
                    w, h = map(int, line.split())
                    break
            
            maxval_line = f.readline().decode().strip()
            maxval = int(maxval_line)
            
            print(f"  {w}x{h}, maxval={maxval}")
            
            # Read first pixel
            if maxval == 255:
                first_pixel = struct.unpack('BBB', f.read(3))
                print(f"  First pixel RGB: {first_pixel}")
                
                # Jump to last pixel
                f.seek(-3, 2)
                last_pixel = struct.unpack('BBB', f.read(3))
                print(f"  Last pixel RGB: {last_pixel}")
            else:
                print(f"  (HDR format, maxval={maxval})")
                
    except Exception as e:
        print(f"{fname}: Error - {e}")

check_ppm('window_dump.ppm')
check_ppm('window_dump_3m_1spp_noisy.ppm')
check_ppm('output_gpu.ppm')
