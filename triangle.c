// triangle.c
#include "triangle.h"
#include "primitives.h"
#include "vec3.h"
#include "ray.h"

#include <math.h>
#include <stdio.h>

// ==================================================
// CONFIG
// ==================================================
#define TRI_EPS 1e-8f

// 0: SSE (ysu_hit_triangle_asm), 1: AVX2 (ysu_hit_triangle_avx2)
#define YSU_TRI_IMPL_AVX2 1

// 0: kapalı, 1: açık (C vs ASM doğrulama)
#define YSU_VALIDATE_TRI_HIT 0
#define YSU_VALIDATE_EPS 1e-5f

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

extern int ysu_hit_triangle_avx2(
    const Triangle* tri,
    const Ray* r,
    float t_min,
    float t_max,
    float* out_t,
    float* out_u,
    float* out_v
);

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

// ==================================================
// TRIANGLE CONSTRUCTOR
// ==================================================
Triangle triangle_make(
    Vec3 p0, Vec3 p1, Vec3 p2,
    float u0, float v0,
    float u1, float v1,
    float u2, float v2,
    int material_index)
{
    Triangle t;

    t.p0 = p0; t.p1 = p1; t.p2 = p2;

    t.u0 = u0; t.v0 = v0;
    t.u1 = u1; t.v1 = v1;
    t.u2 = u2; t.v2 = v2;

    t.material_index = material_index;
    return t;
}

// ==================================================
// C REFERENCE (Möller–Trumbore) for validation
// ==================================================
static int ysu_hit_triangle_c_ref(
    Triangle tri,
    Ray r,
    float t_min,
    float t_max,
    float* out_t,
    float* out_u,
    float* out_v
)
{
    Vec3 e1 = vec3_sub(tri.p1, tri.p0);
    Vec3 e2 = vec3_sub(tri.p2, tri.p0);

    Vec3 pvec = vec3_cross(r.direction, e2);
    float det = vec3_dot(e1, pvec);

    if (fabsf(det) < TRI_EPS)
        return 0;

    float invDet = 1.0f / det;

    Vec3 tvec = vec3_sub(r.origin, tri.p0);
    float u = vec3_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
        return 0;

    Vec3 qvec = vec3_cross(tvec, e1);
    float v = vec3_dot(r.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f)
        return 0;

    float t = vec3_dot(e2, qvec) * invDet;
    if (t < t_min || t > t_max)
        return 0;

    *out_t = t;
    *out_u = u;
    *out_v = v;
    return 1;
}

// ==================================================
// TRIANGLE INTERSECTION (SSE/AVX2) + optional validation
// ==================================================
HitRecord hit_triangle(Triangle tri, Ray r, float t_min, float t_max)
{
    HitRecord rec = no_hit();

    float t_asm = 0.0f, u_asm = 0.0f, v_asm = 0.0f;
    int hit_asm = 0;

#if YSU_TRI_IMPL_AVX2
    hit_asm = ysu_hit_triangle_avx2(&tri, &r, t_min, t_max, &t_asm, &u_asm, &v_asm);
#else
    hit_asm = ysu_hit_triangle_asm(&tri, &r, t_min, t_max, &t_asm, &u_asm, &v_asm);
#endif

#if YSU_VALIDATE_TRI_HIT
    float t_c = 0.0f, u_c = 0.0f, v_c = 0.0f;
    int hit_c = ysu_hit_triangle_c_ref(tri, r, t_min, t_max, &t_c, &u_c, &v_c);

    int mismatch = 0;
    if (hit_c != hit_asm) {
        mismatch = 1;
    } else if (hit_c) {
        if (fabsf(t_c - t_asm) > YSU_VALIDATE_EPS) mismatch = 1;
        if (fabsf(u_c - u_asm) > YSU_VALIDATE_EPS) mismatch = 1;
        if (fabsf(v_c - v_asm) > YSU_VALIDATE_EPS) mismatch = 1;
    }

    if (mismatch) {
        printf("[TRI MISMATCH] impl=%s\n", YSU_TRI_IMPL_AVX2 ? "AVX2" : "SSE");
        printf("  hit_c=%d hit_asm=%d\n", hit_c, hit_asm);
        printf("  t  c=%f asm=%f\n", t_c, t_asm);
        printf("  u  c=%f asm=%f\n", u_c, u_asm);
        printf("  v  c=%f asm=%f\n", v_c, v_asm);
    }
#endif

    if (!hit_asm)
        return rec;

    // ---- HIT RECORD ----
    rec.hit = 1;
    rec.t = t_asm;
    rec.point = ray_at(r, t_asm);
    rec.material_index = tri.material_index;

    // flat normal
    Vec3 e1 = vec3_sub(tri.p1, tri.p0);
    Vec3 e2 = vec3_sub(tri.p2, tri.p0);
    rec.normal = vec3_normalize(vec3_cross(e1, e2));

    // barycentric -> UV
    float w = 1.0f - u_asm - v_asm;
    rec.u = w * tri.u0 + u_asm * tri.u1 + v_asm * tri.u2;
    rec.v = w * tri.v0 + u_asm * tri.v1 + v_asm * tri.v2;

    return rec;
}
