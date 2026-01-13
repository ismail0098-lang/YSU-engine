// denoise.c  — edge-aware denoiser (basit bilateral tarzı)

#include <stdlib.h>
#include <math.h>
#include "denoise.h"

#define IDX(x, y, w) ((y) * (w) + (x))

static float vec3_color_dist2(Vec3 a, Vec3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

void denoise_box(Vec3 *pixels, int width, int height, int radius)
{
    if (!pixels || width <= 0 || height <= 0 || radius <= 0) {
        return;
    }

    int total = width * height;
    Vec3 *temp = (Vec3 *)malloc((size_t)total * sizeof(Vec3));
    if (!temp) {
        return;
    }

    // radius ne kadar büyükse uzamsal sigma da o kadar büyük
    float sigma_space = (float)radius * 0.75f;
    float sigma_color = 0.15f;   // renk farkı hassasiyeti (0–1 aralığı için)
    float two_sigma_space2 = 2.0f * sigma_space * sigma_space;
    float two_sigma_color2 = 2.0f * sigma_color * sigma_color;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            Vec3 center = pixels[IDX(x, y, width)];

            float sum_w = 0.0f;
            float acc_x = 0.0f;
            float acc_y = 0.0f;
            float acc_z = 0.0f;

            int x0 = x - radius;
            int x1 = x + radius;
            int y0 = y - radius;
            int y1 = y + radius;

            if (x0 < 0)       x0 = 0;
            if (y0 < 0)       y0 = 0;
            if (x1 >= width)  x1 = width - 1;
            if (y1 >= height) y1 = height - 1;

            for (int j = y0; j <= y1; ++j) {
                for (int i = x0; i <= x1; ++i) {
                    Vec3 c = pixels[IDX(i, j, width)];

                    int dx = i - x;
                    int dy = j - y;
                    float dist2 = (float)(dx*dx + dy*dy);

                    float w_space = expf(-dist2 / two_sigma_space2);

                    float color2 = vec3_color_dist2(center, c);
                    float w_color = expf(-color2 / two_sigma_color2);

                    float w = w_space * w_color;

                    acc_x += c.x * w;
                    acc_y += c.y * w;
                    acc_z += c.z * w;
                    sum_w += w;
                }
            }

            if (sum_w > 0.0f) {
                float inv = 1.0f / sum_w;
                Vec3 out = {
                    acc_x * inv,
                    acc_y * inv,
                    acc_z * inv
                };
                temp[IDX(x, y, width)] = out;
            } else {
                temp[IDX(x, y, width)] = center;
            }
        }
    }

    for (int i = 0; i < total; ++i) {
        pixels[i] = temp[i];
    }

    free(temp);
}
#include "vec3.h"

void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height) {
    (void)pixels;
    (void)width;
    (void)height;
}
