#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"

typedef struct {
    Vec3 center;
    float radius;
    int material_index;   // Hangi materyale ait? (materials[] dizisinde index)
} Sphere;

// Ray–sphere intersection result
typedef struct {
    int hit;              // 1 = hit, 0 = no hit
    float t;              // distance along the ray
    Vec3 point;           // hit point
    Vec3 normal;          // surface normal
    int material_index;   // vurulan yüzeyin materyali
} HitRecord;

// Create sphere
Sphere sphere_create(Vec3 center, float radius, int material_index);

// Check intersection
HitRecord sphere_intersect(Sphere s, Ray r, float t_min, float t_max);

#endif
