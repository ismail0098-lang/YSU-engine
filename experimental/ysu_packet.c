// ysu_packet.c
#include "ysu_packet.h"
#include <string.h>

static inline float ysu_minf(float a, float b) { return a < b ? a : b; }

YSU_Ray8 ysu_pack_rays8(const Ray* rays8) {
    YSU_Ray8 out;
#ifdef __AVX2__
    float ox[8], oy[8], oz[8], dx[8], dy[8], dz[8];
    for (int i = 0; i < 8; ++i) {
        ox[i] = rays8[i].origin.x;   oy[i] = rays8[i].origin.y;   oz[i] = rays8[i].origin.z;
        dx[i] = rays8[i].direction.x;dy[i] = rays8[i].direction.y;dz[i] = rays8[i].direction.z;
    }
    out.ox = _mm256_loadu_ps(ox); out.oy = _mm256_loadu_ps(oy); out.oz = _mm256_loadu_ps(oz);
    out.dx = _mm256_loadu_ps(dx); out.dy = _mm256_loadu_ps(dy); out.dz = _mm256_loadu_ps(dz);
#else
    for (int i = 0; i < 8; ++i) {
        out.ox[i] = rays8[i].origin.x;   out.oy[i] = rays8[i].origin.y;   out.oz[i] = rays8[i].origin.z;
        out.dx[i] = rays8[i].direction.x;out.dy[i] = rays8[i].direction.y;out.dz[i] = rays8[i].direction.z;
    }
#endif
    return out;
}

YSU_Tri8 ysu_pack_tris8(const Vec3* p0_8, const Vec3* p1_8, const Vec3* p2_8) {
    YSU_Tri8 out;
#ifdef __AVX2__
    float v0x[8], v0y[8], v0z[8], e1x[8], e1y[8], e1z[8], e2x[8], e2y[8], e2z[8];
    for (int i = 0; i < 8; ++i) {
        v0x[i] = p0_8[i].x; v0y[i] = p0_8[i].y; v0z[i] = p0_8[i].z;
        e1x[i] = p1_8[i].x - p0_8[i].x; e1y[i] = p1_8[i].y - p0_8[i].y; e1z[i] = p1_8[i].z - p0_8[i].z;
        e2x[i] = p2_8[i].x - p0_8[i].x; e2y[i] = p2_8[i].y - p0_8[i].y; e2z[i] = p2_8[i].z - p0_8[i].z;
    }
    out.v0x = _mm256_loadu_ps(v0x); out.v0y = _mm256_loadu_ps(v0y); out.v0z = _mm256_loadu_ps(v0z);
    out.e1x = _mm256_loadu_ps(e1x); out.e1y = _mm256_loadu_ps(e1y); out.e1z = _mm256_loadu_ps(e1z);
    out.e2x = _mm256_loadu_ps(e2x); out.e2y = _mm256_loadu_ps(e2y); out.e2z = _mm256_loadu_ps(e2z);
#else
    for (int i = 0; i < 8; ++i) {
        out.v0x[i] = p0_8[i].x; out.v0y[i] = p0_8[i].y; out.v0z[i] = p0_8[i].z;
        out.e1x[i] = p1_8[i].x - p0_8[i].x; out.e1y[i] = p1_8[i].y - p0_8[i].y; out.e1z[i] = p1_8[i].z - p0_8[i].z;
        out.e2x[i] = p2_8[i].x - p0_8[i].x; out.e2y[i] = p2_8[i].y - p0_8[i].y; out.e2z[i] = p2_8[i].z - p0_8[i].z;
    }
#endif
    return out;
}

#ifdef __AVX2__

// Cross product (a x b)
static inline void cross3(__m256 ax, __m256 ay, __m256 az,
                          __m256 bx, __m256 by, __m256 bz,
                          __m256* rx, __m256* ry, __m256* rz)
{
    *rx = _mm256_sub_ps(_mm256_mul_ps(ay, bz), _mm256_mul_ps(az, by));
    *ry = _mm256_sub_ps(_mm256_mul_ps(az, bx), _mm256_mul_ps(ax, bz));
    *rz = _mm256_sub_ps(_mm256_mul_ps(ax, by), _mm256_mul_ps(ay, bx));
}

static inline __m256 dot3(__m256 ax, __m256 ay, __m256 az,
                          __m256 bx, __m256 by, __m256 bz)
{
    return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(ax, bx), _mm256_mul_ps(ay, by)),
                         _mm256_mul_ps(az, bz));
}

YSU_Hit8 ysu_intersect_ray8_tri1(const YSU_Ray8* r8,
                                Vec3 v0, Vec3 v1, Vec3 v2,
                                float t_min, float t_max)
{
    // Möller–Trumbore, vectorized over rays (8 lanes)
    const float eps = 1e-8f;
    const Vec3 e1s = (Vec3){ v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
    const Vec3 e2s = (Vec3){ v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };

    __m256 v0x = _mm256_set1_ps(v0.x), v0y = _mm256_set1_ps(v0.y), v0z = _mm256_set1_ps(v0.z);
    __m256 e1x = _mm256_set1_ps(e1s.x), e1y = _mm256_set1_ps(e1s.y), e1z = _mm256_set1_ps(e1s.z);
    __m256 e2x = _mm256_set1_ps(e2s.x), e2y = _mm256_set1_ps(e2s.y), e2z = _mm256_set1_ps(e2s.z);

    __m256 px, py, pz;
    cross3(r8->dx, r8->dy, r8->dz, e2x, e2y, e2z, &px, &py, &pz);
    __m256 det = dot3(e1x, e1y, e1z, px, py, pz);

    __m256 abs_det = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), det);
    __m256 det_ok  = _mm256_cmp_ps(abs_det, _mm256_set1_ps(eps), _CMP_GT_OQ);

    __m256 inv_det = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

    __m256 tx = _mm256_sub_ps(r8->ox, v0x);
    __m256 ty = _mm256_sub_ps(r8->oy, v0y);
    __m256 tz = _mm256_sub_ps(r8->oz, v0z);

    __m256 u = _mm256_mul_ps(dot3(tx, ty, tz, px, py, pz), inv_det);
    __m256 u_ok1 = _mm256_cmp_ps(u, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 u_ok2 = _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_LE_OQ);
    __m256 u_ok = _mm256_and_ps(u_ok1, u_ok2);

    __m256 qx, qy, qz;
    cross3(tx, ty, tz, e1x, e1y, e1z, &qx, &qy, &qz);

    __m256 v = _mm256_mul_ps(dot3(r8->dx, r8->dy, r8->dz, qx, qy, qz), inv_det);
    __m256 v_ok1 = _mm256_cmp_ps(v, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 v_ok2 = _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OQ);
    __m256 v_ok = _mm256_and_ps(v_ok1, v_ok2);

    __m256 t = _mm256_mul_ps(dot3(e2x, e2y, e2z, qx, qy, qz), inv_det);
    __m256 t_ok1 = _mm256_cmp_ps(t, _mm256_set1_ps(t_min), _CMP_GE_OQ);
    __m256 t_ok2 = _mm256_cmp_ps(t, _mm256_set1_ps(t_max), _CMP_LE_OQ);
    __m256 t_ok = _mm256_and_ps(t_ok1, t_ok2);

    __m256 mask = _mm256_and_ps(_mm256_and_ps(det_ok, u_ok), _mm256_and_ps(v_ok, t_ok));

    // Convert mask to bitmask
    int m = _mm256_movemask_ps(mask);

    YSU_Hit8 out;
    out.hit_mask = (uint8_t)m;

    // Store t
    _mm256_storeu_ps(out.t, t);
    return out;
}

YSU_Hit1 ysu_intersect_ray1_tri8(const Ray* r,
                                const YSU_Tri8* t8,
                                float t_min, float t_max)
{
    // Vectorize over triangles (8 lanes), scalar ray broadcast.
    const float eps = 1e-8f;

    __m256 ox = _mm256_set1_ps(r->origin.x);
    __m256 oy = _mm256_set1_ps(r->origin.y);
    __m256 oz = _mm256_set1_ps(r->origin.z);
    __m256 dx = _mm256_set1_ps(r->direction.x);
    __m256 dy = _mm256_set1_ps(r->direction.y);
    __m256 dz = _mm256_set1_ps(r->direction.z);

    __m256 px, py, pz;
    cross3(dx, dy, dz, t8->e2x, t8->e2y, t8->e2z, &px, &py, &pz);

    __m256 det = dot3(t8->e1x, t8->e1y, t8->e1z, px, py, pz);
    __m256 abs_det = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), det);
    __m256 det_ok  = _mm256_cmp_ps(abs_det, _mm256_set1_ps(eps), _CMP_GT_OQ);

    __m256 inv_det = _mm256_div_ps(_mm256_set1_ps(1.0f), det);

    __m256 tx = _mm256_sub_ps(ox, t8->v0x);
    __m256 ty = _mm256_sub_ps(oy, t8->v0y);
    __m256 tz = _mm256_sub_ps(oz, t8->v0z);

    __m256 u = _mm256_mul_ps(dot3(tx, ty, tz, px, py, pz), inv_det);
    __m256 u_ok = _mm256_and_ps(_mm256_cmp_ps(u, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                                _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 qx, qy, qz;
    cross3(tx, ty, tz, t8->e1x, t8->e1y, t8->e1z, &qx, &qy, &qz);

    __m256 v = _mm256_mul_ps(dot3(dx, dy, dz, qx, qy, qz), inv_det);
    __m256 v_ok = _mm256_and_ps(_mm256_cmp_ps(v, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                                _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OQ));

    __m256 t = _mm256_mul_ps(dot3(t8->e2x, t8->e2y, t8->e2z, qx, qy, qz), inv_det);
    __m256 t_ok = _mm256_and_ps(_mm256_cmp_ps(t, _mm256_set1_ps(t_min), _CMP_GE_OQ),
                                _mm256_cmp_ps(t, _mm256_set1_ps(t_max), _CMP_LE_OQ));

    __m256 mask = _mm256_and_ps(_mm256_and_ps(det_ok, u_ok), _mm256_and_ps(v_ok, t_ok));

    // Select minimum t among hits
    // Set t = +INF where not hit
    __m256 inf = _mm256_set1_ps(1e30f);
    __m256 t_sel = _mm256_blendv_ps(inf, t, mask);

    float t_arr[8];
    _mm256_storeu_ps(t_arr, t_sel);

    YSU_Hit1 out = {0, 0.0f, -1};
    float best = t_max;
    for (int i = 0; i < 8; ++i) {
        if (t_arr[i] < best) {
            best = t_arr[i];
            out.hit = 1;
            out.t = t_arr[i];
            out.tri_index = i;
        }
    }
    return out;
}

#else // no AVX2

YSU_Hit8 ysu_intersect_ray8_tri1(const YSU_Ray8* r8,
                                Vec3 v0, Vec3 v1, Vec3 v2,
                                float t_min, float t_max)
{
    // Scalar fallback: 8 times Möller–Trumbore
    const float eps = 1e-8f;
    Vec3 e1 = (Vec3){ v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
    Vec3 e2 = (Vec3){ v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };

    YSU_Hit8 out;
    out.hit_mask = 0;

    for (int i = 0; i < 8; ++i) {
        float ox = r8->ox[i], oy = r8->oy[i], oz = r8->oz[i];
        float dx = r8->dx[i], dy = r8->dy[i], dz = r8->dz[i];

        // pvec = cross(dir, e2)
        float px = dy*e2.z - dz*e2.y;
        float py = dz*e2.x - dx*e2.z;
        float pz = dx*e2.y - dy*e2.x;

        float det = e1.x*px + e1.y*py + e1.z*pz;
        if (det > -eps && det < eps) { out.t[i] = 0.0f; continue; }
        float inv_det = 1.0f / det;

        float tx = ox - v0.x, ty = oy - v0.y, tz = oz - v0.z;
        float u = (tx*px + ty*py + tz*pz) * inv_det;
        if (u < 0.0f || u > 1.0f) { out.t[i] = 0.0f; continue; }

        // qvec = cross(tvec, e1)
        float qx = ty*e1.z - tz*e1.y;
        float qy = tz*e1.x - tx*e1.z;
        float qz = tx*e1.y - ty*e1.x;

        float v = (dx*qx + dy*qy + dz*qz) * inv_det;
        if (v < 0.0f || (u + v) > 1.0f) { out.t[i] = 0.0f; continue; }

        float t = (e2.x*qx + e2.y*qy + e2.z*qz) * inv_det;
        out.t[i] = t;
        if (t >= t_min && t <= t_max) out.hit_mask |= (1u << i);
    }
    return out;
}

YSU_Hit1 ysu_intersect_ray1_tri8(const Ray* r,
                                const YSU_Tri8* t8,
                                float t_min, float t_max)
{
    // Scalar fallback: test 8 triangles
    YSU_Hit1 out = {0, 0.0f, -1};
    float best = t_max;

    // NOTE: t8 stores v0 and edges; reconstruct v1/v2 as v0+e1, v0+e2
    for (int i = 0; i < 8; ++i) {
        Vec3 v0 = { t8->v0x[i], t8->v0y[i], t8->v0z[i] };
        Vec3 v1 = { t8->v0x[i] + t8->e1x[i], t8->v0y[i] + t8->e1y[i], t8->v0z[i] + t8->e1z[i] };
        Vec3 v2 = { t8->v0x[i] + t8->e2x[i], t8->v0y[i] + t8->e2y[i], t8->v0z[i] + t8->e2z[i] };

        // Reuse the ray8->tri1 scalar path by building a 1-lane packet in lane 0
        Ray rr[8];
        for (int k = 0; k < 8; ++k) rr[k] = *r;
        YSU_Ray8 r8 = ysu_pack_rays8(rr);
        YSU_Hit8 h8 = ysu_intersect_ray8_tri1(&r8, v0, v1, v2, t_min, best);
        if (h8.hit_mask & 1u) {
            float t = h8.t[0];
            if (t < best) { best = t; out.hit = 1; out.t = t; out.tri_index = i; }
        }
    }
    return out;
}
#endif
