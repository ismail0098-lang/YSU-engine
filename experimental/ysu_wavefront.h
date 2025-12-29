// ysu_wavefront.h
// Experimental: Wavefront path tracing skeleton (CPU)
// Goal: Separate generation/intersection/shading into queues for better batching and cache behavior.
// This is a serious base that you can wire into your existing integrator step by step.

#ifndef YSU_WAVEFRONT_H
#define YSU_WAVEFRONT_H

#include <stdint.h>
#include "../ray.h"
#include "../vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Ray ray;
    Vec3 throughput;   // path throughput (RGB)
    uint32_t pixel;    // pixel index (x + y*w)
    uint32_t depth;    // bounce depth
    uint32_t rng;      // per-path RNG state (example: xorshift/pcg seed)
} YSU_Path;

typedef struct {
    int hit;
    float t;
    Vec3 p;
    Vec3 n;
    int material_id;
} YSU_SurfHit;

typedef struct {
    YSU_Path* items;
    uint32_t count;
    uint32_t capacity;
} YSU_PathQueue;

void ysu_queue_init(YSU_PathQueue* q, uint32_t capacity);
void ysu_queue_free(YSU_PathQueue* q);
void ysu_queue_clear(YSU_PathQueue* q);
int  ysu_queue_push(YSU_PathQueue* q, const YSU_Path* p);

// Main wavefront pipeline (engine adapter will be needed):
// - generate primary rays into q_active
// - intersect them (BVH) into hits array
// - shade/scatter -> produce q_next
// This file provides the queueing + pipeline skeleton, not your materials/BVH bindings.
typedef struct {
    uint32_t width, height;
    uint32_t spp;
    uint32_t max_depth;
    uint32_t base_seed;
} YSU_WavefrontSettings;

typedef struct {
    YSU_PathQueue q_active;
    YSU_PathQueue q_next;
    YSU_SurfHit*  hits;     // size = q_active.capacity
} YSU_WavefrontState;

int  ysu_wavefront_init(YSU_WavefrontState* st, uint32_t max_paths);
void ysu_wavefront_free(YSU_WavefrontState* st);

// Entry point (adapter expected):
// The adapter should provide two callbacks:
//  (1) intersect_cb: fills hits for each active path
//  (2) shade_cb: consumes hits and produces next paths / accumulates to framebuffer
typedef void (*YSU_IntersectCB)(const YSU_Path* paths, uint32_t n, YSU_SurfHit* out_hits, void* user);
typedef void (*YSU_ShadeCB)(const YSU_Path* paths, const YSU_SurfHit* hits, uint32_t n,
                            YSU_PathQueue* q_next, void* user);

void ysu_wavefront_render(const YSU_WavefrontSettings* s,
                          YSU_WavefrontState* st,
                          YSU_IntersectCB intersect_cb,
                          YSU_ShadeCB shade_cb,
                          void* user);

#ifdef __cplusplus
}
#endif

#endif // YSU_WAVEFRONT_H
