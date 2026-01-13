import random
OUT = "scene.txt"
NUM_SPHERES = random.randint(50, 400)

with open(OUT, "w", encoding="utf-8") as f:
    for _ in range(NUM_SPHERES):
        cx = random.uniform(-5, 5)
        cy = random.uniform(-2, 2)
        cz = random.uniform(-10, -1)
        r  = random.uniform(0.1, 1.0)
        cr = random.random()
        cg = random.random()
        cb = random.random()
        f.write(f"sphere {cx:.3f} {cy:.3f} {cz:.3f} {r:.3f} {cr:.3f} {cg:.3f} {cb:.3f}\n")

print("Wrote scene with", NUM_SPHERES, "spheres")
