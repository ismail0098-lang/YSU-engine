// image.c - PPM writer (P6 binary)
// Optional PostFX: Bloom + Tonemap via env toggles (YSU_POSTFX / YSU_BLOOM)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "vec3.h"
#include "postprocess.h"

// PNG writer (header-only)
// Put stb_image_write.h next to this file (project root or same folder).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float clamp01f(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static unsigned char to_u8_gamma22(float x) {
    // gamma 2.2 approximation (same spirit as your ZIP)
    x = clamp01f(x);
    x = powf(x, 1.0f / 2.2f);
    int v = (int)(x * 255.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)v;
}

static int ysu_env_int(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return atoi(s);
}

static float ysu_env_float(const char *name, float defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return (float)atof(s);
}

static void image_write_ppm_u8(const char *filename, int width, int height, const unsigned char *rgb_u8) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb_u8, 1, (size_t)width * (size_t)height * 3, f);
    fclose(f);
}

// ------------------------------------------------------------
// NEW: HDR(Vec3) -> 8-bit RGB buffer (malloc; free caller)
// Mirrors the exact logic of image_write_ppm() in your ZIP.
// ------------------------------------------------------------
unsigned char* image_rgb_from_hdr(const Vec3 *pixels, int width, int height) {
    if (!pixels || width <= 0 || height <= 0) return NULL;

    // Toggle: enable postfx if YSU_POSTFX=1 or YSU_BLOOM=1
    int postfx = ysu_env_int("YSU_POSTFX", 0) || ysu_env_int("YSU_BLOOM", 0);

    size_t n = (size_t)width * (size_t)height;

    // Fast path: gamma 2.2 only (assumes pixels already in 0..1)
    if (!postfx) {
        unsigned char *ldr = (unsigned char*)malloc(n * 3);
        if (!ldr) return NULL;

        for (size_t i = 0; i < n; ++i) {
            ldr[i*3+0] = to_u8_gamma22(pixels[i].x);
            ldr[i*3+1] = to_u8_gamma22(pixels[i].y);
            ldr[i*3+2] = to_u8_gamma22(pixels[i].z);
        }
        return ldr;
    }

    // PostFX path: treat pixels as linear HDR (can be >1), apply exposure+bloom+ACES+gamma.
    float *hdr = (float*)malloc(n * 4 * sizeof(float));
    unsigned char *ldr = (unsigned char*)malloc(n * 3);
    if (!hdr || !ldr) {
        free(hdr);
        free(ldr);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        hdr[i*4+0] = pixels[i].x;
        hdr[i*4+1] = pixels[i].y;
        hdr[i*4+2] = pixels[i].z;
        hdr[i*4+3] = 1.0f;
    }

    PostFX fx;
    fx.exposure         = ysu_env_float("YSU_EXPOSURE",        1.0f);
    printf("[image] POSTFX=%s BLOOM=%s EXPOSURE=%s (fx.exposure=%.3f)\n",
       getenv("YSU_POSTFX"), getenv("YSU_BLOOM"), getenv("YSU_EXPOSURE"), fx.exposure);
       
    fx.bloom_threshold  = ysu_env_float("YSU_BLOOM_THRESHOLD", 1.2f);
    fx.bloom_knee       = ysu_env_float("YSU_BLOOM_KNEE",      0.6f);
    fx.bloom_intensity  = ysu_env_float("YSU_BLOOM_INTENSITY", 0.15f);
    fx.bloom_iterations = ysu_env_int  ("YSU_BLOOM_ITERS",     2);

    ysu_apply_bloom_tonemap_u8(hdr, width, height, ldr, &fx);

    free(hdr);
    return ldr;
}

// ------------------------------------------------------------
// NEW: PNG writer (8-bit RGB)
// ------------------------------------------------------------
void image_write_png(const char *filename, int width, int height, const unsigned char *rgb_u8) {
    if (!filename || !rgb_u8 || width <= 0 || height <= 0) return;
    // stride = width * 3
    stbi_write_png(filename, width, height, 3, rgb_u8, width * 3);
}

// ------------------------------------------------------------
// Existing: PPM writer (kept). Now implemented via image_rgb_from_hdr().
// ------------------------------------------------------------
void image_write_ppm(const char *filename, int width, int height, Vec3 *pixels) {
    if (!pixels || width <= 0 || height <= 0) return;

    unsigned char *ldr = image_rgb_from_hdr(pixels, width, height);
    if (!ldr) return;

    image_write_ppm_u8(filename, width, height, ldr);
    free(ldr);
}
