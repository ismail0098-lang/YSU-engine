// ysu_main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "vec3.h"
#include "camera.h"
#include "render.h"
#include "image.h"

// Neural stage-1
#include "neural_denoise.h"
#include "gbuffer_dump.h"

// Header yok -> manuel prototype
void ysu_render_360(const Camera *cam, const char *out_ppm);

static int env_int(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return atoi(s);
}

static void print_cfg(int w, int h, int spp, int depth, int threads, int tile) {
    printf("[main] CFG: W=%d H=%d SPP=%d DEPTH=%d THREADS=%d TILE=%d\n",
           w, h, spp, depth, threads, tile);
}

int main(void)
{
    printf("[main] START\n");

    // -------------------------
    // Config (env ile kontrol)
    // -------------------------
    int image_width       = env_int("YSU_W", 800);
    int image_height      = env_int("YSU_H", 450);
    int samples_per_pixel = env_int("YSU_SPP", 64);
    int max_depth         = env_int("YSU_DEPTH", 8);

    int thread_count      = env_int("YSU_THREADS", 0); // 0 => auto
    int tile_size         = env_int("YSU_TILE", 32);

    if (image_width < 1) image_width = 1;
    if (image_height < 1) image_height = 1;
    if (samples_per_pixel < 1) samples_per_pixel = 1;
    if (max_depth < 1) max_depth = 1;
    if (tile_size < 4) tile_size = 4;

    print_cfg(image_width, image_height, samples_per_pixel, max_depth, thread_count, tile_size);

    // -------------------------
    // Allocate framebuffer
    // -------------------------
    Vec3 *pixels = (Vec3*)calloc((size_t)image_width * (size_t)image_height, sizeof(Vec3));
    if (!pixels) {
        printf("[main] ERROR: could not allocate pixels (%dx%d)\n", image_width, image_height);
        return 1;
    }

    // -------------------------
    // Camera setup
    // -------------------------
    float aspect_ratio   = (float)image_width / (float)image_height;
    float viewport_h     = 2.0f;
    float focal_length   = 1.0f;

    Camera cam = camera_create(aspect_ratio, viewport_h, focal_length);

    // -------------------------
    // Render
    // -------------------------
    printf("[main] calling render...\n");

    if (thread_count > 0) {
        render_scene_mt(pixels,
                        image_width,
                        image_height,
                        cam,
                        samples_per_pixel,
                        max_depth,
                        thread_count,
                        tile_size);
    } else {
        render_scene_st(pixels,
                        image_width,
                        image_height,
                        cam,
                        samples_per_pixel,
                        max_depth);
    }

    // -------------------------
    // Neural postprocess (toggle: YSU_NEURAL_DENOISE=1)
    // -------------------------
    ysu_neural_denoise_maybe(pixels, image_width, image_height);

    // Optional dump (toggle: YSU_DUMP_RGB=1)
    if (getenv("YSU_DUMP_RGB")) {
        if (ysu_dump_rgb32("output_color.ysub", pixels, image_width, image_height)) {
            printf("[main] dumped output_color.ysub\n");
        } else {
            printf("[main] WARN: dump failed\n");
        }
    }

    // -------------------------
    // Write image
    // -------------------------
    image_write_ppm("output.ppm", image_width, image_height, pixels);
    printf("[main] wrote output.ppm\n");

    free(pixels);

    // -------------------------
    // 360 render
    // -------------------------
    printf("[main] calling ysu_render_360...\n");
    ysu_render_360(&cam, "output_360.ppm");
    printf("[main] wrote output_360.ppm\n");

    printf("[main] END\n");
    return 0;
}
