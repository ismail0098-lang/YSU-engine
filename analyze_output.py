#!/usr/bin/env python3
"""Quick image analyzer"""
import sys
import numpy as np

def analyze(fname, outfile):
    try:
        with open(fname, 'rb') as f:
            f.readline()  # magic
            line = b''
            while not line.rstrip():
                line = f.readline()
                if not line.startswith(b'#'): break
            w, h = map(int, line.split())
            f.readline()  # maxval
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w, 3))
        
        lum = 0.2126*data[:,:,0] + 0.7152*data[:,:,1] + 0.0722*data[:,:,2]
        noise = float(np.std(lum / 255.0))
        unique = len(np.unique(data.reshape(-1,3), axis=0))
        
        with open(outfile, 'w') as f:
            f.write(f"{fname}: {w}x{h}\n")
            f.write(f"  Noise: {noise:.6f}\n")
            f.write(f"  Colors: {unique}\n")
            f.write(f"  Min/Max: {data.min()}/{data.max()}\n")
    except Exception as e:
        with open(outfile, 'w') as f:
            f.write(f"Error: {e}\n")

if __name__ == '__main__':
    analyze(sys.argv[1], sys.argv[2])
