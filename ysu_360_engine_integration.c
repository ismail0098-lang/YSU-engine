// YSU Engine - 360 Equirectangular Rendering (Engine-Integrated Version)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "render.h"
#include "image.h"

#define YSU_360_WIDTH      2048
#define YSU_360_HEIGHT     1024
#define YSU_360_MAX_DEPTH  50

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// render.c içinde tanımlı:
extern Vec3 ray_color_internal(Ray ray, int max_depth);

// Piksel (x,y) → 360° equirectangular yön vektörü
static Vec3 ysu_360_pixel_to_dir(int x, int y)
{
    double u = (double)x / (double)YSU_360_WIDTH;
    double v = (double)y / (double)YSU_360_HEIGHT;

    double theta = u * 2.0 * M_PI;      // 0 → 2π (360°)
    double phi   = (v - 0.5) * M_PI;    // -π/2 → +π/2 (-90° → +90°)

    double dx = cos(phi) * cos(theta);
    double dy = sin(phi);
    double dz = cos(phi) * sin(theta);

    return vec3((float)dx, (float)dy, (float)dz);
}

// 360° panorama render
// Normal render'dan bağımsız çalışıyor, doğrudan kamera origin'ini kullanıyor.
void ysu_render_360(const Camera *cam, const char *out_ppm)
{
    Vec3 *pixels = (Vec3*)malloc(sizeof(Vec3) * YSU_360_WIDTH * YSU_360_HEIGHT);
    if (!pixels) {
        fprintf(stderr, "YSU 360: pixel buffer allocate edilemedi.\n");
        return;
    }

    Vec3 origin = cam->origin;

    for (int y = 0; y < YSU_360_HEIGHT; ++y) {
        printf("YSU 360 scanline %d / %d\r", y + 1, YSU_360_HEIGHT);
        fflush(stdout);

        for (int x = 0; x < YSU_360_WIDTH; ++x) {
            Vec3 dir = ysu_360_pixel_to_dir(x, y);
            dir = vec3_normalize(dir);

            Ray r = ray_create(origin, dir);
            Vec3 col = ray_color_internal(r, YSU_360_MAX_DEPTH);

            int idx = (YSU_360_HEIGHT - 1 - y) * YSU_360_WIDTH + x;
            pixels[idx] = col;
        }
    }

    printf("\nYSU 360: çıktıyı yazıyor: %s\n", out_ppm);
    image_write_ppm(out_ppm, YSU_360_WIDTH, YSU_360_HEIGHT, pixels);

    free(pixels);
    printf("YSU 360: tamam.\n");
}
