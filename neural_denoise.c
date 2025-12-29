// neural_denoise.c - Stage-1 neural render integration (stub, ONNX-ready)

#include "neural_denoise.h"

#include <stdlib.h>

#include "denoise.h"

static int ysu_env_int(const char *name, int defv) {
    const char *s = getenv(name);
    if (!s || !s[0]) return defv;
    return atoi(s);
}

static void ysu_denoise_impl(Vec3 *pixels, int width, int height) {
    // Placeholder: fast & stable.
    int radius = ysu_env_int("YSU_NEURAL_RADIUS", 2);
    if (radius < 1) radius = 1;
    if (radius > 6) radius = 6;
    denoise_box(pixels, width, height, radius);
}

void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height)
{
    if (!pixels || width <= 0 || height <= 0) return;
    int enabled = ysu_env_int("YSU_NEURAL_DENOISE", 0) ? 1 : 0;
    if (!enabled) return;
    ysu_denoise_impl(pixels, width, height);
}
