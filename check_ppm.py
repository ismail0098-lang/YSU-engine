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

img = load_ppm('output_gpu.ppm')
print(f"Image shape: {img.shape}")
print(f"Min/Max: {img.min()}/{img.max()}")
print(f"Non-zero pixels: {np.sum(img > 0)}")
