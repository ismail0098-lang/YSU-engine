// render.c (FULL) - pthread threadpool + chunked jobs + tile renderer + RNG + stub
#include "render.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>

#if __STDC_VERSION__ >= 201112L
  #include <stdatomic.h>
#else
  #error "C11 gerekiyor (stdatomic). GCC'de -std=c11 kullan."
#endif

#include <pthread.h>

#if defined(_WIN32)
  #include <windows.h>
#endif

#include "material.h"
#include "sphere.h"
#include "triangle.h"
#include "bvh.h"
#include "primitives.h"
#include "vec3.h"
#include "ray.h"
#include "camera.h"
Vec3 ray_color_internal(Ray r, int depth);
// ================================================================
// Adaptive sampling config + stats (env kontrollü)
// ================================================================

static int   g_adapt_enabled = 0;
static int   g_adapt_spp_min = 16;
static int   g_adapt_spp_batch = 4;
static float g_adapt_rel_err = 0.02f;
static float g_adapt_abs_err = 0.001f;

// stats
static _Atomic uint64_t g_adapt_total_samples = 0;
static _Atomic uint64_t g_adapt_early_pixels  = 0;
static _Atomic uint64_t g_adapt_max_pixels    = 0;

static int ysu_env_int(const char* name, int defv) {
    const char* s = getenv(name);
    if (!s || !s[0]) return defv;
    int v = atoi(s);
    return v;
}

static float ysu_env_float(const char* name, float defv) {
    const char* s = getenv(name);
    if (!s || !s[0]) return defv;
    float v = (float)atof(s);
    return v;
}

static void ysu_adapt_load_config(void) {
    g_adapt_enabled  = ysu_env_int("YSU_ADAPTIVE", 0) ? 1 : 0;
    g_adapt_spp_min  = ysu_env_int("YSU_SPP_MIN", 16);
    g_adapt_spp_batch= ysu_env_int("YSU_SPP_BATCH", 4);
    g_adapt_rel_err  = ysu_env_float("YSU_REL_ERR", 0.02f);
    g_adapt_abs_err  = ysu_env_float("YSU_ABS_ERR", 0.001f);

    if (g_adapt_spp_min < 8) g_adapt_spp_min = 8;
    if (g_adapt_spp_batch < 1) g_adapt_spp_batch = 1;
    if (g_adapt_rel_err < 0.0f) g_adapt_rel_err = 0.0f;
    if (g_adapt_abs_err < 0.0f) g_adapt_abs_err = 0.0f;
}

static inline float ysu_luminance(Vec3 c) {
    // Rec.709-ish
    return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
}

// ------------------------- RNG (xorshift32) -------------------------
typedef struct { uint32_t state; } YSU_Rng;

static inline uint32_t ysu_rng_u32(YSU_Rng *r) {
    uint32_t x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    return x;
}


static inline uint32_t ysu_hash_u32(uint32_t x) {
    // A small avalanching hash (public domain style). Good enough for seed mixing.
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x != 0u) ? x : 1u;
}

// Seed a per-pixel RNG deterministically from a base seed + pixel coords (+ optional salt).
static inline uint32_t ysu_seed_pixel(uint32_t base, uint32_t px, uint32_t py, uint32_t salt) {
    // Mix coords with different odd constants then avalanche.
    uint32_t x = base;
    x ^= px * 0x9E3779B1u;
    x ^= py * 0x85EBCA77u;
    x ^= salt * 0xC2B2AE3Du;
    x = ysu_hash_u32(x);
    // xorshift32 must never be 0
    return (x == 0u) ? 1u : x;
}


static inline float ysu_rng_f01(YSU_Rng *r) {
    // [0,1)
    return (ysu_rng_u32(r) >> 8) * (1.0f / 16777216.0f);
}


/* Public RNG helpers (declared in render.h) */
float ysu_rng_next01(uint32_t *state) {
    if (!state) return 0.0f;
    YSU_Rng r;
    r.state = (*state != 0u) ? *state : 1u;
    float u = ysu_rng_f01(&r);
    *state = r.state;
    return u;
}

int ysu_russian_roulette(uint32_t *state, float p_survive) {
    if (p_survive <= 0.0f) return 0;
    if (p_survive >= 1.0f) return 1;
    float u = ysu_rng_next01(state);
    return (u < p_survive) ? 1 : 0;
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

// ------------------------- ray_color_internal STUB -------------------------
#define YSU_STUB_RAY_COLOR_INTERNAL 1

#if YSU_STUB_RAY_COLOR_INTERNAL
static Vec3 ysu_sky(Ray r) {
    Vec3 u = vec3_unit(r.direction);
    float t = 0.5f * (u.y + 1.0f);
    Vec3 a = vec3(1.0f, 1.0f, 1.0f);
    Vec3 b = vec3(0.5f, 0.7f, 1.0f);
    return vec3_add(vec3_scale(a, 1.0f - t), vec3_scale(b, t));
}
Vec3 ray_color_internal(Ray r, int depth) {
    (void)depth;
    return ysu_sky(r);
}
#endif

// ------------------------- Single-thread render -------------------------
void render_scene_st(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth)
{
    if (!pixels || image_width <= 0 || image_height <= 0) return;
    if (samples_per_pixel < 1) samples_per_pixel = 1;
    if (max_depth < 1) max_depth = 1;

    ysu_adapt_load_config();
    atomic_store(&g_adapt_total_samples, 0);
    atomic_store(&g_adapt_early_pixels, 0);
    atomic_store(&g_adapt_max_pixels, 0);

    YSU_Rng rng;
    rng.state = ((uint32_t)time(NULL) ^ 0xA511E9B3u);
    if (rng.state == 0) rng.state = 1;

    float inv_wm1 = (image_width  > 1) ? (1.0f / (float)(image_width - 1)) : 0.0f;
    float inv_hm1 = (image_height > 1) ? (1.0f / (float)(image_height - 1)) : 0.0f;

    for (int j = 0; j < image_height; ++j) {
        Vec3* row = pixels + (image_height - 1 - j) * image_width;

        for (int i = 0; i < image_width; ++i) {
            float accx = 0.0f, accy = 0.0f, accz = 0.0f;

            int spp_used = 0;
            int early_stop = 0;

            // Welford for luminance
            float mean = 0.0f, m2 = 0.0f;

            int spp_max = samples_per_pixel;
            int spp_min = (g_adapt_spp_min < spp_max) ? g_adapt_spp_min : spp_max;

            for (int s = 0; s < spp_max; ++s) {
                float u = ((float)i + ysu_rng_f01(&rng)) * inv_wm1;
                float v = ((float)j + ysu_rng_f01(&rng)) * inv_hm1;

                Ray r = camera_get_ray(cam, u, v);
                Vec3 c = ray_color_internal(r, max_depth);

                accx += c.x; accy += c.y; accz += c.z;
                spp_used++;

                if (g_adapt_enabled) {
                    float lum = ysu_luminance(c);
                    float n = (float)spp_used;
                    float delta = lum - mean;
                    mean += delta / n;
                    float delta2 = lum - mean;
                    m2 += delta * delta2;

                    if (spp_used >= spp_min && (spp_used % g_adapt_spp_batch) == 0) {
                        float var = (spp_used > 1) ? (m2 / (float)(spp_used - 1)) : 0.0f;
                        float se  = sqrtf(fmaxf(var, 0.0f) / (float)spp_used);

                        float tol = fmaxf(g_adapt_abs_err, g_adapt_rel_err * fabsf(mean));
                        if (se <= tol) { early_stop = 1; break; }
                    }
                }
            }

            float inv_spp = 1.0f / (float)spp_used;
            row[i] = vec3(accx * inv_spp, accy * inv_spp, accz * inv_spp);

            if (g_adapt_enabled) {
                atomic_fetch_add(&g_adapt_total_samples, (uint64_t)spp_used);
                if (early_stop) atomic_fetch_add(&g_adapt_early_pixels, 1);
                if (!early_stop) atomic_fetch_add(&g_adapt_max_pixels, 1);
            }
        }
    }

    if (g_adapt_enabled) {
        uint64_t total_samples = atomic_load(&g_adapt_total_samples);
        uint64_t early_pixels  = atomic_load(&g_adapt_early_pixels);
        uint64_t max_pixels    = atomic_load(&g_adapt_max_pixels);
        uint64_t px = (uint64_t)image_width * (uint64_t)image_height;
        double avg_spp = (px > 0) ? ((double)total_samples / (double)px) : 0.0;

        printf("\n[ADAPT] total_samples=%" PRIu64 "  avg_spp=%.2f  early_pixels=%" PRIu64 "  max_pixels=%" PRIu64 "\n",
               total_samples, avg_spp, early_pixels, max_pixels);
    }
}

// =====================================================================
// THREAD POOL (persistent)
// =====================================================================

// Atomic job counter + “job chunk” ile contention azaltıyoruz
#define JOB_CHUNK 8

typedef struct {
    // render target
    Vec3 *pixels;
    Camera cam;
    int width, height;
    int spp, depth;
    int tile_size;

    int tiles_x;
    int tiles_y;
    atomic_int next_job;

    uint32_t seed_base;

    // sync
    pthread_mutex_t mtx;
    pthread_cond_t  cv_start;
    pthread_cond_t  cv_done;

    int work_id;        // new render starts => ++work_id
    int active_workers; // how many workers should participate
    int done_workers;   // workers finished this work_id
    int shutdown;

    int pool_threads;   // created thread count
    pthread_t *threads;
    struct WorkerLocal *locals;
} RenderPool;

static RenderPool g_pool = {0};

typedef struct WorkerLocal {
    // NOTE: Keep per-thread data in its own cache line to avoid false sharing.
#if __STDC_VERSION__ >= 201112L
    _Alignas(64)
#endif
    int tid;

    // Persistent per-thread RNG state (non-zero). Mixed per-tile for decorrelation.
    uint32_t rng_state;

    // Conservative cache-line padding for typical 64B lines (int32 + u32 = 8B).
    // This is intentionally simple/portable (no zero-length arrays).
    uint8_t _pad[56];
} WorkerLocal;

static void render_tile_chunk(RenderPool *p, WorkerLocal *wl, int job) {
    int tid = wl->tid;
    int tx = job % p->tiles_x;
    int ty = job / p->tiles_x;

    int x0 = tx * p->tile_size;
    int y0 = ty * p->tile_size;
    int x1 = x0 + p->tile_size;
    int y1 = y0 + p->tile_size;
    if (x1 > p->width)  x1 = p->width;
    if (y1 > p->height) y1 = p->height;

    // precompute for speed
    float inv_wm1 = (p->width  > 1) ? (1.0f / (float)(p->width - 1)) : 0.0f;
    float inv_hm1 = (p->height > 1) ? (1.0f / (float)(p->height - 1)) : 0.0f;

    int spp_max = p->spp;
    int spp_min = (g_adapt_spp_min < spp_max) ? g_adapt_spp_min : spp_max;

    
// RNG: derive a deterministic base for this tile, then derive per-pixel seeds.
// This avoids correlation across pixels/threads and scales better with many threads.
uint32_t tile_base = ysu_hash_u32(wl->rng_state
                               ^ p->seed_base
                               ^ (uint32_t)(job * 0xA511E9B3u));
if (tile_base == 0u) tile_base = 1u;
YSU_Rng rng;
for (int j = y0; j < y1; ++j) {
        Vec3* row = p->pixels + (p->height - 1 - j) * p->width;

        for (int i = x0; i < x1; ++i) {
            rng.state = ysu_seed_pixel(tile_base, (uint32_t)i, (uint32_t)j, (uint32_t)tid);
            float accx = 0.0f, accy = 0.0f, accz = 0.0f;

            int spp_used = 0;
            int early_stop = 0;

            float mean = 0.0f, m2 = 0.0f;

            for (int s = 0; s < spp_max; ++s) {
                float u = ((float)i + ysu_rng_f01(&rng)) * inv_wm1;
                float v = ((float)j + ysu_rng_f01(&rng)) * inv_hm1;

                Ray r = camera_get_ray(p->cam, u, v);
                Vec3 c = ray_color_internal(r, p->depth);

                accx += c.x; accy += c.y; accz += c.z;
                spp_used++;

                if (g_adapt_enabled) {
                    float lum = ysu_luminance(c);
                    float n = (float)spp_used;
                    float delta = lum - mean;
                    mean += delta / n;
                    float delta2 = lum - mean;
                    m2 += delta * delta2;

                    if (spp_used >= spp_min && (spp_used % g_adapt_spp_batch) == 0) {
                        float var = (spp_used > 1) ? (m2 / (float)(spp_used - 1)) : 0.0f;
                        float se  = sqrtf(fmaxf(var, 0.0f) / (float)spp_used);

                        float tol = fmaxf(g_adapt_abs_err, g_adapt_rel_err * fabsf(mean));
                        if (se <= tol) { early_stop = 1; break; }
                    }
                }
            }

            float inv_spp = 1.0f / (float)spp_used;
            row[i] = vec3(accx * inv_spp, accy * inv_spp, accz * inv_spp);

            if (g_adapt_enabled) {
                atomic_fetch_add(&g_adapt_total_samples, (uint64_t)spp_used);
                if (early_stop) atomic_fetch_add(&g_adapt_early_pixels, 1);
                if (!early_stop) atomic_fetch_add(&g_adapt_max_pixels, 1);
            }
        }
    }
    // Advance persistent per-thread RNG state for next tiles.
    wl->rng_state = ysu_hash_u32(wl->rng_state ^ rng.state ^ (uint32_t)job);

}

static void *pool_worker(void *arg) {
    WorkerLocal *wl = (WorkerLocal*)arg;
    int tid = wl->tid;

    int last_work = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mtx);
        while (!g_pool.shutdown && g_pool.work_id == last_work) {
            pthread_cond_wait(&g_pool.cv_start, &g_pool.mtx);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mtx);
            return NULL;
        }
        last_work = g_pool.work_id;

        // Bu work’te aktif değilse “done” sayılır
        int active = (tid < g_pool.active_workers);
        pthread_mutex_unlock(&g_pool.mtx);

        if (active) {
            int total = g_pool.tiles_x * g_pool.tiles_y;
            for (;;) {
                int base = atomic_fetch_add(&g_pool.next_job, JOB_CHUNK);
                if (base >= total) break;
                int end = base + JOB_CHUNK;
                if (end > total) end = total;
                for (int job = base; job < end; ++job) {
                    render_tile_chunk(&g_pool, wl, job);
                }
            }
        }

        pthread_mutex_lock(&g_pool.mtx);
        g_pool.done_workers++;
        if (g_pool.done_workers >= g_pool.active_workers) {
            pthread_cond_signal(&g_pool.cv_done);
        }
        pthread_mutex_unlock(&g_pool.mtx);
    }
}

static void pool_shutdown(void) {
    if (!g_pool.threads) return;
    pthread_mutex_lock(&g_pool.mtx);
    g_pool.shutdown = 1;
    pthread_cond_broadcast(&g_pool.cv_start);
    pthread_mutex_unlock(&g_pool.mtx);

    for (int i = 0; i < g_pool.pool_threads; ++i) {
        pthread_join(g_pool.threads[i], NULL);
    }
    free(g_pool.threads);
    g_pool.threads = NULL;

    free(g_pool.locals);
    g_pool.locals = NULL;

    pthread_mutex_destroy(&g_pool.mtx);
    pthread_cond_destroy(&g_pool.cv_start);
    pthread_cond_destroy(&g_pool.cv_done);
}

static void pool_init_if_needed(int create_threads) {
    if (g_pool.threads) return;

    g_pool.pool_threads = (create_threads > 0) ? create_threads : ysu_suggest_threads();
    if (g_pool.pool_threads < 1) g_pool.pool_threads = 1;

    pthread_mutex_init(&g_pool.mtx, NULL);
    pthread_cond_init(&g_pool.cv_start, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);

    g_pool.work_id = 0;
    g_pool.shutdown = 0;

    g_pool.threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)g_pool.pool_threads);
    // worker local ids (kept alive for the lifetime of the pool)
    g_pool.locals = (WorkerLocal*)malloc(sizeof(WorkerLocal) * (size_t)g_pool.pool_threads);
    if (!g_pool.locals) {
        free(g_pool.threads);
        g_pool.threads = NULL;
        // if allocation fails, fall back to single-thread later
        g_pool.pool_threads = 0;
        return;
    }
    for (int i = 0; i < g_pool.pool_threads; ++i) {
        g_pool.locals[i].tid = i;
        g_pool.locals[i].rng_state = ysu_hash_u32((uint32_t)time(NULL)
                                   ^ (uint32_t)(i * 0x9E3779B9u)
                                   ^ (uint32_t)(uintptr_t)(&g_pool.locals[i]));
        pthread_create(&g_pool.threads[i], NULL, pool_worker, &g_pool.locals[i]);
    }

    atexit(pool_shutdown);
}

// =====================================================================
// Public MT render using threadpool
// =====================================================================
void render_scene_mt(Vec3 *pixels,
                     int image_width,
                     int image_height,
                     Camera cam,
                     int samples_per_pixel,
                     int max_depth,
                     int thread_count,
                     int tile_size)
{
    if (!pixels || image_width <= 0 || image_height <= 0) return;
    if (samples_per_pixel < 1) samples_per_pixel = 1;
    if (max_depth < 1) max_depth = 1;

    ysu_adapt_load_config();
    atomic_store(&g_adapt_total_samples, 0);
    atomic_store(&g_adapt_early_pixels, 0);
    atomic_store(&g_adapt_max_pixels, 0);

    if (thread_count <= 0) thread_count = ysu_suggest_threads();
    if (thread_count < 1) thread_count = 1;

    // Default tile daha büyük: atomic daha az, cache daha iyi
    if (tile_size <= 0) tile_size = 64;
    if (tile_size < 16) tile_size = 16;
    if (thread_count >= 8 && tile_size < 32) tile_size = 32;

    // pool en az thread_count kadar thread açsın (ilk render’da)
    pool_init_if_needed(thread_count);

    int tiles_x = (image_width  + tile_size - 1) / tile_size;
    int tiles_y = (image_height + tile_size - 1) / tile_size;

    pthread_mutex_lock(&g_pool.mtx);

    g_pool.pixels = pixels;
    g_pool.cam = cam;
    g_pool.width = image_width;
    g_pool.height = image_height;
    g_pool.spp = samples_per_pixel;
    g_pool.depth = max_depth;
    g_pool.tile_size = tile_size;

    g_pool.tiles_x = tiles_x;
    g_pool.tiles_y = tiles_y;

    atomic_store(&g_pool.next_job, 0);

    g_pool.seed_base = ((uint32_t)time(NULL) ^ 0xD1B54A35u);
    if (g_pool.seed_base == 0) g_pool.seed_base = 1;

    // aktif worker sayısını sınırla (fazla thread => atomic contention + cache thrash)
int total_jobs = g_pool.tiles_x * g_pool.tiles_y;
if (total_jobs < 1) total_jobs = 1;

if (thread_count > g_pool.pool_threads) thread_count = g_pool.pool_threads;
if (thread_count > total_jobs)         thread_count = total_jobs;
if (thread_count < 1)                  thread_count = 1;

g_pool.active_workers = thread_count;
    g_pool.done_workers = 0;

    g_pool.work_id++;
    pthread_cond_broadcast(&g_pool.cv_start);

    // work bitene kadar bekle
    while (g_pool.done_workers < g_pool.active_workers) {
        pthread_cond_wait(&g_pool.cv_done, &g_pool.mtx);
    }

    pthread_mutex_unlock(&g_pool.mtx);

    if (g_adapt_enabled) {
        uint64_t total_samples = atomic_load(&g_adapt_total_samples);
        uint64_t early_pixels  = atomic_load(&g_adapt_early_pixels);
        uint64_t max_pixels    = atomic_load(&g_adapt_max_pixels);
        uint64_t px = (uint64_t)image_width * (uint64_t)image_height;
        double avg_spp = (px > 0) ? ((double)total_samples / (double)px) : 0.0;

        printf("\n[ADAPT] total_samples=%" PRIu64 "  avg_spp=%.2f  early_pixels=%" PRIu64 "  max_pixels=%" PRIu64 "\n",
               total_samples, avg_spp, early_pixels, max_pixels);
    }
}

void render_scene(Vec3 *pixels,
                  int image_width,
                  int image_height,
                  Camera cam,
                  int samples_per_pixel,
                  int max_depth)
{
    render_scene_mt(pixels, image_width, image_height, cam, samples_per_pixel, max_depth, 0, 64);
}