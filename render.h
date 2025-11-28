// render.h
#ifndef RENDER_H
#define RENDER_H

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"

// Sabitler (istersen buradan da değiştirebilirsin)
#define IMAGE_WIDTH_DEFAULT 400
#define ASPECT_RATIO_DEFAULT (16.0f / 9.0f)
#define SAMPLES_PER_PIXEL_DEFAULT 50
#define MAX_DEPTH_DEFAULT 10

// Sahne kurulumunu dışarıdan da çağırabilelim
void setup_scene(void);

// Ana render fonksiyonu: piksel buffer'ı doldurur
// pixels = image_width * image_height elemanlı Vec3 dizisi
void render_scene(Vec3 *pixels,
                  int image_width,
                  int image_height,
                  Camera cam,
                  int samples_per_pixel,
                  int max_depth);

// Trace a single ray (used for 360° rendering)
Vec3 ray_color_internal(Ray r, int depth);

#endif
