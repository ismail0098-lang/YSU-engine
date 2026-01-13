#!/usr/bin/env python3
"""Check bounds of OBJ file"""
import sys

def check_obj_bounds(path):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != 'v':
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
        
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        cz = (min_z + max_z) / 2
        sx = max_x - min_x
        sy = max_y - min_y
        sz = max_z - min_z
        
        print(f"Bounds:")
        print(f"  X: [{min_x:10.2f}, {max_x:10.2f}] (size={sx:10.2f}, center={cx:10.2f})")
        print(f"  Y: [{min_y:10.2f}, {max_y:10.2f}] (size={sy:10.2f}, center={cy:10.2f})")
        print(f"  Z: [{min_z:10.2f}, {max_z:10.2f}] (size={sz:10.2f}, center={cz:10.2f})")
        print(f"Diagonal: {(sx**2 + sy**2 + sz**2)**0.5:.2f}")
        print(f"Camera at origin (0,0,0) can see up to distance 10.0")
        dist_to_center = (cx**2 + cy**2 + cz**2)**0.5
        print(f"Distance to center: {dist_to_center:.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        check_obj_bounds(sys.argv[1])
    else:
        check_obj_bounds('TestSubjects/3M.obj')
