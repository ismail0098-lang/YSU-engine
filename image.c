// image.c - PPM writer (P6 binary)

#include <stdio.h>
#include <math.h>

#include "image.h"
#include "vec3.h"

static float clamp01f(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static unsigned char to_u8_gamma22(float x) {
    x = clamp01f(x);
    // Gamma correction for display
    x = powf(x, 1.0f / 2.2f);

    int v = (int)(x * 255.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)v;
}

void image_write_ppm(const char *filename, int width, int height, Vec3 *pixels)
{
    if (!filename || !pixels || width <= 0 || height <= 0) {
        printf("image_write_ppm: invalid args\n");
        return;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("PPM write: cannot open %s\n", filename);
        return;
    }

    // P6 = binary PPM
    fprintf(f, "P6\n%d %d\n255\n", width, height);

    // Write RGB bytes
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;

            unsigned char rgb[3];
            rgb[0] = to_u8_gamma22(pixels[i].x);
            rgb[1] = to_u8_gamma22(pixels[i].y);
            rgb[2] = to_u8_gamma22(pixels[i].z);

            fwrite(rgb, 1, 3, f);
        }
    }

    fclose(f);
}
