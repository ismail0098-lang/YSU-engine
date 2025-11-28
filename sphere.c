#include <math.h>
#include "sphere.h"

Sphere sphere_create(Vec3 center, float radius, int material_index) {
    Sphere s;
    s.center = center;
    s.radius = radius;
    s.material_index = material_index;
    return s;
}

HitRecord sphere_intersect(Sphere s, Ray r, float t_min, float t_max) {
    HitRecord rec;
    rec.hit = 0;
    rec.material_index = -1;

    Vec3 oc = vec3_sub(r.origin, s.center);

    float a = vec3_dot(r.direction, r.direction);
    float b = 2.0f * vec3_dot(oc, r.direction);
    float c = vec3_dot(oc, oc) - s.radius * s.radius;

    float discriminant = b*b - 4*a*c;

    if (discriminant < 0.0f) {
        return rec; // no hit
    }

    float sqrtD = sqrtf(discriminant);

    // En yakın kökü bul (t_min–t_max aralığında)
    float root = (-b - sqrtD) / (2.0f * a);
    if (root < t_min || root > t_max) {
        root = (-b + sqrtD) / (2.0f * a);
        if (root < t_min || root > t_max) {
            return rec;
        }
    }

    // Hit!
    rec.hit = 1;
    rec.t = root;
    rec.point = ray_at(r, rec.t);
    rec.normal = vec3_normalize(vec3_sub(rec.point, s.center));
    rec.material_index = s.material_index;

    return rec;
}
