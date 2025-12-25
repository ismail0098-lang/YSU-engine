#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

// Yeni isimlendirme
Ray ray_create(Vec3 origin, Vec3 direction);

// Ray üzerindeki nokta: origin + t * direction
Vec3 ray_at(Ray r, float t);

// Eski kodla uyum için alias (sphere.c, material.c vs. bunu çağırıyor)
Ray ray(Vec3 origin, Vec3 direction);

#endif
