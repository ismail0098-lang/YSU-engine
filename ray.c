#include "ray.h"

Ray ray_create(Vec3 origin, Vec3 direction)
{
    Ray r;
    r.origin = origin;
    r.direction = direction;
    return r;
}

Vec3 ray_at(Ray r, float t)
{
    return vec3_add(r.origin, vec3_scale(r.direction, t));
}

// Eski fonksiyon ismini destekle (ray(...) = ray_create(...))
Ray ray(Vec3 origin, Vec3 direction)
{
    return ray_create(origin, direction);
}
