// sphere.c
#include <math.h>
#include "sphere.h"
#include "vec3.h"
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Sphere sphere_create(Vec3 center, float radius, int material_index)
{
    Sphere s;
    s.center = center;
    s.radius = radius;
    s.material_index = material_index;
    // Varsayılan albedo: beyaz (0–1 arası)
    s.albedo = (Color){ 1.0, 1.0, 1.0 };
    return s;
}

// Sphere için UV (spherical mapping)
static void sphere_get_uv(Vec3 p, float *out_u, float *out_v)
{
    // p = normalleştirilmiş nokta (center'dan çıkan)
    float theta = acosf(-p.y);
    float phi   = atan2f(-p.z, p.x) + (float)M_PI;

    float u = phi / (2.0f * (float)M_PI);
    float v = theta / (float)M_PI;

    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
}

HitRecord sphere_intersect(Sphere s, Ray r, float t_min, float t_max)
{
    HitRecord rec = (HitRecord){0};
    rec.hit = 0;

    Vec3 oc = vec3_sub(r.origin, s.center);
    float a      = vec3_length_squared(r.direction);
    float half_b = vec3_dot(oc, r.direction);
    float c      = vec3_length_squared(oc) - s.radius * s.radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return rec;
    }
    float sqrtd = sqrtf(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return rec;
        }
    }

    rec.t      = root;
    rec.point  = ray_at(r, rec.t);
    Vec3 outward_normal = vec3_scale(vec3_sub(rec.point, s.center), 1.0f / s.radius);
    rec.normal         = outward_normal;
    rec.material_index = s.material_index;
    rec.hit            = 1;

    // UV ve barycentric doldur
    sphere_get_uv(vec3_unit(outward_normal), &rec.u, &rec.v);
    rec.b0 = 1.0f;
    rec.b1 = 0.0f;
    rec.b2 = 0.0f;

    return rec;
}
