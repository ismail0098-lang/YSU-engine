// ysu_360_engine_integration.c
// 360 Equirect Render - Multi-thread tile + Adaptive SPP (variance-based)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#if __STDC_VERSION__ >= 201112L
  #include <stdatomic.h>
#else
  #error "C11 gerekiyor (stdatomic). GCC'de -std=c11 kullan."
#endif

#include <pthread.h>

#if defined(_WIN32)
  #include <windows.h>
#endif

#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===================== 360 Output Settings =====================
#define YSU_360_WIDTH      4096
#define YSU_360_HEIGHT     2048

// Adaptive sampling defaults (override via env if you want)
#define YSU_360_SPP_MIN_DEFAULT    16
#define YSU_360_SPP_MAX_DEFAULT    256
#define YSU_360_SPP_BATCH_DEFAULT  16

// Error thresholds (tune)
#define YSU_360_REL_ERR_DEFAULT  0.03f   // relative error target
#define YSU_360_ABS_ERR_DEFAULT  0.002f  // absolute error target (helps dark areas)

#define YSU_360_MAX_DEPTH  25
#define YSU_360_TILE_DEFAULT  64

// Engine function (from render.c or elsewhere)
extern Vec3 ray_color_internal(Ray ray, int max_depth);

// ===================== RNG (xorshift32) =====================
typedef struct { uint32_t state; } YSU_Rng;

static inline uint32_t ysu_rng_u32(YSU_Rng *r) {
    uint32_t x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    return x;
}

static inline float ysu_rng_f01(YSU_Rng *r) {
    return (ysu_rng_u32(r) >> 8) * (1.0f / 16777216.0f); // [0,1)
}

static int ysu_suggest_threads(void) {
    const char *env = getenv("YSU_THREADS");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }
#if defined(_WIN32)
    SYSTEM_INFO sys;
    GetSystemInfo(&sys);
    int n = (int)sys.dwNumberOfProcessors;
    return (n > 0) ? n : 8;
#else
    return 8;
#endif
}

static int ysu_env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    int x = atoi(v);
    return (x > 0) ? x : fallback;
}

static float ysu_env_float(const char *name, float fallback) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    float x = (float)atof(v);
    return (x > 0.0f) ? x : fallback;
}

// ===================== 360 mapping =====================
static inline Vec3 ysu_360_pixel_to_dir(float fx, float fy)
{
    // fx in [0,W), fy in [0,H)
    double u = (double)fx / (double)YSU_360_WIDTH;   // [0,1)
    double v = (double)fy / (double)YSU_360_HEIGHT;  // [0,1)

    double theta = u * 2.0 * M_PI;       // 0..2pi
    double phi   = (v - 0.5) * M_PI;     // -pi/2..+pi/2

    double cphi = cos(phi);
    double sphi = sin(phi);
    double cth  = cos(theta);
    double sth  = sin(theta);

    double dx = cphi * cth;
    double dy = sphi;
    double dz = cphi * sth;

    return vec3((float)dx, (float)dy, (float)dz);
}

static inline float ysu_luminance(Vec3 c) {
    // simple Rec.709-ish weights
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// ===================== Adaptive SPP per pixel =====================
// Welford running mean/variance on luminance for stop decision.
// We still average RGB normally.
static inline Vec3 ysu_render_pixel_adaptive(Vec3 origin, int x, int y, YSU_Rng *rng,
                                            int spp_min, int spp_max, int spp_batch,
                                            float rel_err, float abs_err)
{
    Vec3 sum = vec3(0.0f, 0.0f, 0.0f);

    // Welford stats for luminance
    int n = 0;
    float mean = 0.0f;
    float m2 = 0.0f;

    int target = spp_min;
    if (target > spp_max) target = spp_max;

    while (n < spp_max) {
        int todo = spp_batch;
        if (n < target) {
            int need = target - n;
            if (need < todo) todo = need;
        } else {
            // we already met target; still do in batches
        }
        if (n + todo > spp_max) todo = spp_max - n;

        for (int i = 0; i < todo; ++i) {
            float jx = ysu_rng_f01(rng);
            float jy = ysu_rng_f01(rng);

            Vec3 dir = ysu_360_pixel_to_dir((float)x + jx, (float)y + jy);
            dir = vec3_normalize(dir);

            Ray r = ray_create(origin, dir);
            Vec3 col = ray_color_internal(r, YSU_360_MAX_DEPTH);

            sum = vec3_add(sum, col);

            float lum = ysu_luminance(col);
            n++;
            float delta = lum - mean;
            mean += delta / (float)n;
            float delta2 = lum - mean;
            m2 += delta * delta2;
        }

        // stop check only after at least spp_min and variance available
        if (n >= spp_min && n >= 2) {
            float var = m2 / (float)(n - 1);                 // sample variance
            float se  = sqrtf(var / (float)n);               // standard error of mean

            float thresh = abs_err + rel_err * fabsf(mean);  // abs + relative
            if (se <= thresh) {
                break; // good enough
            }
        }

        // if we haven't reached minimum yet, ensure we do
        if (n < spp_min) target = spp_min;
    }

    Vec3 out = vec3_scale(sum, 1.0f / (float)n);

    // Gamma 2.0 (clamp to avoid sqrt of negative if your tracer can go negative)
    if (out.x < 0.0f) out.x = 0.0f;
    if (out.y < 0.0f) out.y = 0.0f;
    if (out.z < 0.0f) out.z = 0.0f;

    out.x = sqrtf(out.x);
    out.y = sqrtf(out.y);
    out.z = sqrtf(out.z);

    return out;
}

// ===================== MT tile system =====================
typedef struct {
    Vec3 *pixels;
    Vec3 origin;

    int tile;
    int tiles_x;
    int tiles_y;

    atomic_int *next_job;
    int thread_id;
    uint32_t seed_base;

    // adaptive params
    int spp_min, spp_max, spp_batch;
    float rel_err, abs_err;
} YSU360_Worker;

static void ysu360_render_tile(const YSU360_Worker *w, int x0, int y0, int x1, int y1)
{
    YSU_Rng rng;
    rng.state = w->seed_base
              ^ (uint32_t)(w->thread_id * 0x9E3779B9u)
              ^ (uint32_t)(x0 * 73856093u)
              ^ (uint32_t)(y0 * 19349663u);
    if (rng.state == 0) rng.state = 1;

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            Vec3 out = ysu_render_pixel_adaptive(
                w->origin, x, y, &rng,
                w->spp_min, w->spp_max, w->spp_batch,
                w->rel_err, w->abs_err
            );
            w->pixels[y * YSU_360_WIDTH + x] = out;
        }
    }
}

static void *ysu360_worker_main(void *arg)
{
    YSU360_Worker *w = (YSU360_Worker*)arg;
    int total = w->tiles_x * w->tiles_y;

    for (;;) {
        int job = atomic_fetch_add(w->next_job, 1);
        if (job >= total) break;

        int tx = job % w->tiles_x;
        int ty = job / w->tiles_x;

        int x0 = tx * w->tile;
        int y0 = ty * w->tile;
        int x1 = x0 + w->tile;
        int y1 = y0 + w->tile;

        if (x1 > YSU_360_WIDTH)  x1 = YSU_360_WIDTH;
        if (y1 > YSU_360_HEIGHT) y1 = YSU_360_HEIGHT;

        ysu360_render_tile(w, x0, y0, x1, y1);
    }

    return NULL;
}

// ===================== Public API =====================
void ysu_render_360(const Camera *cam, const char *out_ppm)
{
    int threads   = ysu_suggest_threads();
    int tile      = ysu_env_int("YSU_360_TILE", YSU_360_TILE_DEFAULT);

    int spp_min   = ysu_env_int("YSU_360_SPP_MIN",   YSU_360_SPP_MIN_DEFAULT);
    int spp_max   = ysu_env_int("YSU_360_SPP_MAX",   YSU_360_SPP_MAX_DEFAULT);
    int spp_batch = ysu_env_int("YSU_360_SPP_BATCH", YSU_360_SPP_BATCH_DEFAULT);

    float rel_err = ysu_env_float("YSU_360_REL_ERR", YSU_360_REL_ERR_DEFAULT);
    float abs_err = ysu_env_float("YSU_360_ABS_ERR", YSU_360_ABS_ERR_DEFAULT);

    if (tile < 16) tile = 16;
    if (spp_batch < 1) spp_batch = 1;
    if (spp_min < 1) spp_min = 1;
    if (spp_max < spp_min) spp_max = spp_min;

    printf("YSU 360 Adaptive MT: %dx%d depth=%d threads=%d tile=%d\n",
           YSU_360_WIDTH, YSU_360_HEIGHT, YSU_360_MAX_DEPTH, threads, tile);
    printf("SPP: min=%d max=%d batch=%d  err: rel=%.4f abs=%.4f\n",
           spp_min, spp_max, spp_batch, rel_err, abs_err);

    Vec3 *pixels = (Vec3*)malloc(sizeof(Vec3) * (size_t)YSU_360_WIDTH * (size_t)YSU_360_HEIGHT);
    if (!pixels) {
        fprintf(stderr, "YSU 360: pixel buffer allocate edilemedi.\n");
        return;
    }

    int tiles_x = (YSU_360_WIDTH  + tile - 1) / tile;
    int tiles_y = (YSU_360_HEIGHT + tile - 1) / tile;

    atomic_int next_job;
    atomic_init(&next_job, 0);

    pthread_t *tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
    YSU360_Worker *ctx = (YSU360_Worker*)malloc(sizeof(YSU360_Worker) * (size_t)threads);

    uint32_t seed_base = ((uint32_t)time(NULL) ^ 0xC0FFEE11u);
    if (seed_base == 0) seed_base = 1;

    for (int i = 0; i < threads; ++i) {
        ctx[i].pixels   = pixels;
        ctx[i].origin   = cam->origin;
        ctx[i].tile     = tile;
        ctx[i].tiles_x  = tiles_x;
        ctx[i].tiles_y  = tiles_y;
        ctx[i].next_job = &next_job;
        ctx[i].thread_id = i;
        ctx[i].seed_base = seed_base;

        ctx[i].spp_min   = spp_min;
        ctx[i].spp_max   = spp_max;
        ctx[i].spp_batch = spp_batch;
        ctx[i].rel_err   = rel_err;
        ctx[i].abs_err   = abs_err;

        pthread_create(&tids[i], NULL, ysu360_worker_main, &ctx[i]);
    }

    for (int i = 0; i < threads; ++i) {
        pthread_join(tids[i], NULL);
    }

    free(ctx);
    free(tids);

    printf("YSU 360: çıktı yazılıyor: %s\n", out_ppm);
    image_write_ppm(out_ppm, YSU_360_WIDTH, YSU_360_HEIGHT, pixels);

    free(pixels);
    printf("YSU 360: tamam.\n");
}
