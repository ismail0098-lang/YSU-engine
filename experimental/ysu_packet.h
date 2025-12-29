// ysu_packet.h
// Experimental: AVX2 ray/triangle packet intersections (CPU)
// This module is intentionally self-contained and does NOT modify core engine code.

#ifndef YSU_PACKET_H
#define YSU_PACKET_H

#include <stdint.h>

#ifdef __AVX2__
  #include <immintrin.h>
#endif

#include "../ray.h"
#include "../vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// 8-wide ray packet (Structure of Arrays)
typedef struct {
#ifdef __AVX2__
    __m256 ox, oy, oz;   // origins
    __m256 dx, dy, dz;   // directions
#else
    float ox[8], oy[8], oz[8];
    float dx[8], dy[8], dz[8];
#endif
} YSU_Ray8;

// 8-wide triangle packet (Structure of Arrays)
typedef struct {
#ifdef __AVX2__
    __m256 v0x, v0y, v0z;
    __m256 e1x, e1y, e1z;   // v1 - v0
    __m256 e2x, e2y, e2z;   // v2 - v0
#else
    float v0x[8], v0y[8], v0z[8];
    float e1x[8], e1y[8], e1z[8];
    float e2x[8], e2y[8], e2z[8];
#endif
} YSU_Tri8;

typedef struct {
    uint8_t hit_mask;   // bit i = ray i hit (for ray8->tri1)
    float   t[8];       // t per lane (undefined if not hit)
} YSU_Hit8;

// Utilities to pack/unpack
YSU_Ray8 ysu_pack_rays8(const Ray* rays8);
YSU_Tri8 ysu_pack_tris8(const Vec3* p0_8, const Vec3* p1_8, const Vec3* p2_8);

// Intersections
// (A) 8 rays vs 1 triangle: returns which rays hit and their t's
YSU_Hit8 ysu_intersect_ray8_tri1(const YSU_Ray8* r8,
                                Vec3 v0, Vec3 v1, Vec3 v2,
                                float t_min, float t_max);

// (B) 1 ray vs 8 triangles: returns best hit among the 8 tris (or hit=0)
typedef struct {
    int hit;
    float t;
    int tri_index; // 0..7 inside the packet
} YSU_Hit1;

YSU_Hit1 ysu_intersect_ray1_tri8(const Ray* r,
                                const YSU_Tri8* t8,
                                float t_min, float t_max);

#ifdef __cplusplus
}
#endif

#endif // YSU_PACKET_H
