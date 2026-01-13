// neural_denoise.c - Stage-1 neural render integration (now with real bilateral filtering)

#include "neural_denoise.h"

/*
 * CHECKPOINT: See .github/CHECKPOINTS.md for denoiser integration notes.
 * - `YSU_NEURAL_DENOISE` toggles the neural denoiser (now bilateral by default).
 * - Bilateral filter: edge-aware, perceptually high-quality.
 * - Check `onnx_denoise.c` if ONNX runtime integration is required.
 * - `shaders/` contains SPV assets used by other denoiser paths.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "denoise.h"
#include "bilateral_denoise.h"

static int ysu_env_int(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return atoi(s);
}

static float ysu_env_float(const char *name, float defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    char buf[128];
    size_t n = strlen(s);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, s, n);
    buf[n] = 0;
    for (size_t i = 0; i < n; i++) if (buf[i] == ',') buf[i] = '.';
    return (float)atof(buf);
}

static void ysu_denoise_impl(Vec3 *pixels, int width, int height) {
    // Use bilateral filter (edge-aware, real denoising)
    // This is much better than box filter for raytraced images
    
    // Get parameters from environment (same as bilateral_denoise.c)
    float sigma_s = ysu_env_float("YSU_BILATERAL_SIGMA_S", 1.5f);   // spatial (pixels)
    float sigma_r = ysu_env_float("YSU_BILATERAL_SIGMA_R", 0.1f);   // range (luminance)
    int radius = ysu_env_int("YSU_BILATERAL_RADIUS", 3);            // filter radius
    
    if (sigma_s < 0.1f) sigma_s = 0.1f;
    if (sigma_r < 0.01f) sigma_r = 0.01f;
    if (radius < 1) radius = 1;
    if (radius > 20) radius = 20;
    
    bilateral_denoise(pixels, width, height, sigma_s, sigma_r, radius);
}

void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height)
{
    if (!pixels || width <= 0 || height <= 0) return;
    int enabled = ysu_env_int("YSU_NEURAL_DENOISE", 0) ? 1 : 0;
    if (!enabled) return;
    
    fprintf(stderr, "[DENOISE] YSU_NEURAL_DENOISE enabled, using bilateral filter\n");
    ysu_denoise_impl(pixels, width, height);
}
