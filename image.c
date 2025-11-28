#include <stdio.h>
#include "image.h"

// Clamp helper
static int clamp_color(float x) {
    if (x < 0.0f) return 0;
    if (x > 0.999f) return 255;
    return (int)(x * 255.999f);
}

void image_write_ppm(const char *filename, int width, int height, Vec3 *pixels) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Could not open file %s\n", filename);
        return;
    }

    // PPM header
    fprintf(f, "P3\n%d %d\n255\n", width, height);

    // Write pixels row by row
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            Vec3 c = pixels[j * width + i];

            int r = clamp_color(c.x);
            int g = clamp_color(c.y);
            int b = clamp_color(c.z);

            fprintf(f, "%d %d %d\n", r, g, b);
        }
    }

    fclose(f);
}
