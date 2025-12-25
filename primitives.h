// primitives.h
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "vec3.h"
#include "ray.h"

// Simple hit record used by all primitives
typedef struct {
    int hit;               // 0 = miss, 1 = hit
    float t;
    Vec3 point;
    Vec3 normal;
    int material_index;
    // texture coords / barycentric
    float u;
    float v;
    float b0, b1, b2;
} HitRecord;

// Triangle primitive (unique field names)
typedef struct {
    Vec3 p0, p1, p2;       // vertex positions
    Vec3 n0, n1, n2;       // optional vertex normals (may be vec3_zero)
    float u0, v0;
    float u1, v1;
    float u2, v2;
    int material_index;
} Triangle;

// Create triangle (called from render.c) — no per-vertex normals required
Triangle triangle_make(Vec3 p0, Vec3 p1, Vec3 p2,
                       float u0, float v0,
                       float u1, float v1,
                       float u2, float v2,
                       int material_index);

// Ray/triangle intersection
HitRecord hit_triangle(Triangle tri, Ray r, float t_min, float t_max);

// Other primitives prototypes (keep or adapt as you need)
HitRecord hit_plane   (/*Plane pl*/ void* pl, Ray r, float t_min, float t_max);
HitRecord hit_cylinder(/*Cylinder cy*/ void* cy, Ray r, float t_min, float t_max);
HitRecord hit_box     (/*Box b*/ void* b, Ray r, float t_min, float t_max);

#endif // PRIMITIVES_H
