#include "ray.h"

// Create ray
Ray ray_create(Vec3 origin, Vec3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction;
    return r;
}

// Compute point along ray
Vec3 ray_at(Ray r, float t) {
    return vec3_add(r.origin, vec3_scale(r.direction, t));
}
