// ysu_main.c
#include <stdio.h>
#include <stdlib.h>

#include "vec3.h"
#include "camera.h"
#include "render.h"
#include "image.h"

// Header yok -> manuel prototype
void ysu_render_360(const Camera *cam, const char *out_ppm);

int main(void)
{
    printf("[main] START\n");

    // -------------------------
    // Normal render ayarları
    // -------------------------
    const int image_width  = 400;
    const int image_height = 225;
    const int spp          = 10;
    const int depth        = 5;

    printf("[main] image: %dx%d spp=%d depth=%d\n", image_width, image_height, spp, depth);
    printf("[main] calling render_scene...\n");

    Vec3 *pixels = (Vec3*)malloc(sizeof(Vec3) * image_width * image_height);
    if (!pixels) {
        printf("[main] ERROR: malloc failed (pixels)\n");
        return 1;
    }

    float aspect_ratio = (float)image_width / (float)image_height;

    // Senin camera.h'ına göre doğru imza:
    // Camera camera_create(float aspect_ratio, float viewport_height, float focal_length);
    Camera cam = camera_create(aspect_ratio, 2.0f, 1.0f);

    render_scene(pixels, image_width, image_height, cam, spp, depth);

    printf("[main] render_scene finished\n");
    image_write_ppm("output.ppm", image_width, image_height, pixels);
    printf("[main] wrote output.ppm\n");

    free(pixels);

    // -------------------------
    // 360 render (ZIP içindeki gerçek fonksiyon)
    // -------------------------
    printf("[main] calling ysu_render_360...\n");
    ysu_render_360(&cam, "output_360.ppm");
    printf("[main] wrote output_360.ppm\n");

    printf("[main] END\n");
    return 0;
}
