#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

// Create a ray
Ray ray_create(Vec3 origin, Vec3 direction);

// Point along a ray at distance t (origin + t * direction)
Vec3 ray_at(Ray r, float t);

#endif
