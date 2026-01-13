#ifndef RENDER_H
#define RENDER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"

/**
 * Debug view modes (env: YSU_DEBUG)
 *  - "albedo"
 *  - "normal"
 *  - "depth"
 *  - "luma"
 */
typedef enum {
    DEBUG_NONE = 0,
    DEBUG_ALBEDO,
    DEBUG_NORMAL,
    DEBUG_DEPTH,
    DEBUG_LUMINANCE
} DebugView;

/**
 * Single-thread render (simple loop). If you want MT, use render_scene_mt().
 */
void render_scene_st(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth);

/**
 * Multi-thread render using internal pthread threadpool + tiled jobs.
 */
void render_scene_mt(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth,
                     int thread_count,
                     int tile_size);

/**
 * Convenience wrapper: chooses ST/MT internally (currently calls MT auto).
 */
void render_scene(Vec3 *pixels,
                  int image_width,
                  int image_height,
                  Camera cam,
                  int samples_per_pixel,
                  int max_depth);

/**
 * Integrator entry used by renderer.
 */
Vec3 ray_color_internal(Ray r, int depth);

/**
 * Portable RNG helper for integrator code.
 * Uses a simple xorshift32 on the given state.
 */
float ysu_rng_next01(uint32_t *state);

/**
 * Russian roulette helper: returns 1 to continue, 0 to terminate.
 * p_survive is clamped to [0,1].
 */
int   ysu_russian_roulette(uint32_t *state, float p_survive);

#ifdef __cplusplus
}
#endif

#endif // RENDER_H
