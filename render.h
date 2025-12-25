#ifndef RENDER_H
#define RENDER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "vec3.h"
#include "ray.h"
#include "camera.h"

void render_scene(Vec3 *pixels,
                  int image_width,
                  int image_height,
                  Camera cam,
                  int samples_per_pixel,
                  int max_depth);

void render_scene_mt(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth,
                     int thread_count,
                     int tile_size);

void render_scene_st(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth);

Vec3 ray_color_internal(Ray r, int depth);

#ifdef __cplusplus
}
#endif

#endif
