// triangle.c - triangle primitive + optional AVX2/ASM hit
#include "triangle.h"
#include "ray.h"
#include "vec3.h"

#include <math.h>
#include <stdint.h>

// ==================================================
// CONFIG
// ==================================================
#define YSU_TRI_IMPL_AVX2 1

// ==================================================
// ASM INTERFACES
// ==================================================
extern int ysu_hit_triangle_asm(
    const Triangle* tri,
    const Ray* r,
    float t_min,
    float t_max,
    float* out_t,
    float* out_u,
    float* out_v
);

// ASM sembol adı: triangle_hit_asm.S içinde ysu_hit_triangle_asm olarak export ediliyor.
// Bu projede AVX2 hit fonksiyonunu o sembole aliaslıyoruz.
#define ysu_hit_triangle_avx2 ysu_hit_triangle_asm

// ==================================================
// INTERNAL
// ==================================================
static inline HitRecord no_hit(void)
{
    HitRecord rec;
    rec.hit = 0;
    rec.t = 0.0f;
    rec.point = vec3(0.0f, 0.0f, 0.0f);
    rec.normal = vec3(0.0f, 0.0f, 0.0f);
    rec.material_index = -1;
    rec.u = 0.0f;
    rec.v = 0.0f;
    return rec;
}

static inline HitRecord make_hit(const Triangle* tri, const Ray* r, float t, float u, float v)
{
    HitRecord rec;
    rec.hit = 1;
    rec.t = t;
    rec.u = u;
    rec.v = v;
    rec.material_index = tri->material_index;

    rec.point = ray_at(*r, t);

    // geometric normal
    Vec3 e1 = vec3_sub(tri->p1, tri->p0);
    Vec3 e2 = vec3_sub(tri->p2, tri->p0);
    Vec3 n  = vec3_unit(vec3_cross(e1, e2));

    // face-forward
    if (vec3_dot(r->direction, n) > 0.0f) n = vec3_scale(n, -1.0f);
    rec.normal = n;

    return rec;
}

// Reference C implementation (Möller–Trumbore)
static int ysu_hit_triangle_c(const Triangle* tri, const Ray* r, float t_min, float t_max,
                             float* out_t, float* out_u, float* out_v)
{
    Vec3 e1 = vec3_sub(tri->p1, tri->p0);
    Vec3 e2 = vec3_sub(tri->p2, tri->p0);

    Vec3 pvec = vec3_cross(r->direction, e2);
    float det = vec3_dot(e1, pvec);

    // cull near-parallel
    if (fabsf(det) < 1e-8f) return 0;
    float inv_det = 1.0f / det;

    Vec3 tvec = vec3_sub(r->origin, tri->p0);
    float u = vec3_dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return 0;

    Vec3 qvec = vec3_cross(tvec, e1);
    float v = vec3_dot(r->direction, qvec) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) return 0;

    float t = vec3_dot(e2, qvec) * inv_det;
    if (t < t_min || t > t_max) return 0;

    if (out_t) *out_t = t;
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    return 1;
}

// ==================================================
// PUBLIC
// ==================================================
HitRecord triangle_hit(const Triangle* tri, const Ray* r, float t_min, float t_max)
{
    if (!tri || !r) return no_hit();

    float t = 0.0f, u = 0.0f, v = 0.0f;
    int ok = 0;

#if YSU_TRI_IMPL_AVX2
    ok = ysu_hit_triangle_avx2(tri, r, t_min, t_max, &t, &u, &v);
    // güvenlik: asm fail olursa C fallback
    if (!ok) ok = ysu_hit_triangle_c(tri, r, t_min, t_max, &t, &u, &v);
#else
    ok = ysu_hit_triangle_c(tri, r, t_min, t_max, &t, &u, &v);
#endif

    if (!ok) return no_hit();
    return make_hit(tri, r, t, u, v);
}
