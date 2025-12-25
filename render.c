// render.c (FULL) - pthread threadpool + chunked jobs + tile renderer + RNG + stub
#include "render.h"

#include <stdlib.h>
#include <stdint.h>
#include <time.h>

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

static inline float ysu_rng_f01(YSU_Rng *r) {
    return (ysu_rng_u32(r) >> 8) * (1.0f / 16777216.0f);
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

    YSU_Rng rng;
    rng.state = ((uint32_t)time(NULL) ^ 0xA511E9B3u);
    if (rng.state == 0) rng.state = 1;

    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            Vec3 acc = vec3(0.0f, 0.0f, 0.0f);

            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = ((float)i + ysu_rng_f01(&rng)) / (float)(image_width - 1);
                float v = ((float)j + ysu_rng_f01(&rng)) / (float)(image_height - 1);

                Ray r = camera_get_ray(cam, u, v);
                acc = vec3_add(acc, ray_color_internal(r, max_depth));
            }

            acc = vec3_scale(acc, 1.0f / (float)samples_per_pixel);
            pixels[(image_height - 1 - j) * image_width + i] = acc;
        }
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
} RenderPool;

static RenderPool g_pool = {0};

typedef struct {
    int tid;
} WorkerLocal;

static void render_tile_chunk(RenderPool *p, int tid, int job) {
    int tx = job % p->tiles_x;
    int ty = job / p->tiles_x;

    int x0 = tx * p->tile_size;
    int y0 = ty * p->tile_size;
    int x1 = x0 + p->tile_size;
    int y1 = y0 + p->tile_size;
    if (x1 > p->width)  x1 = p->width;
    if (y1 > p->height) y1 = p->height;

    // thread-local RNG seed
    YSU_Rng rng;
    rng.state = p->seed_base
              ^ (uint32_t)(tid * 0x9E3779B9u)
              ^ (uint32_t)(x0 * 73856093u)
              ^ (uint32_t)(y0 * 19349663u);
    if (rng.state == 0) rng.state = 1;

    for (int j = y0; j < y1; ++j) {
        for (int i = x0; i < x1; ++i) {
            Vec3 acc = vec3(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < p->spp; ++s) {
                float u = ((float)i + ysu_rng_f01(&rng)) / (float)(p->width - 1);
                float v = ((float)j + ysu_rng_f01(&rng)) / (float)(p->height - 1);
                Ray r = camera_get_ray(p->cam, u, v);
                acc = vec3_add(acc, ray_color_internal(r, p->depth));
            }
            acc = vec3_scale(acc, 1.0f / (float)p->spp);
            p->pixels[(p->height - 1 - j) * p->width + i] = acc;
        }
    }
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
                    render_tile_chunk(&g_pool, tid, job);
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

    // worker local ids
    WorkerLocal *locals = (WorkerLocal*)malloc(sizeof(WorkerLocal) * (size_t)g_pool.pool_threads);
    for (int i = 0; i < g_pool.pool_threads; ++i) {
        locals[i].tid = i;
        pthread_create(&g_pool.threads[i], NULL, pool_worker, &locals[i]);
    }

    // locals'ı free edemeyiz çünkü worker arg pointer’ı kullanıyor.
    // Basit çözüm: locals'ı intentionally leak (küçük ve tek sefer).
    // İstersen locals'ı static array yaparız.
    (void)locals;

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

    if (thread_count <= 0) thread_count = ysu_suggest_threads();
    if (thread_count < 1) thread_count = 1;

    // Default tile daha büyük: atomic daha az, cache daha iyi
    if (tile_size <= 0) tile_size = 64;
    if (tile_size < 16) tile_size = 16;

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

    // aktif worker sayısını sınırla
    if (thread_count > g_pool.pool_threads) thread_count = g_pool.pool_threads;
    g_pool.active_workers = thread_count;
    g_pool.done_workers = 0;

    g_pool.work_id++;
    pthread_cond_broadcast(&g_pool.cv_start);

    // work bitene kadar bekle
    while (g_pool.done_workers < g_pool.active_workers) {
        pthread_cond_wait(&g_pool.cv_done, &g_pool.mtx);
    }

    pthread_mutex_unlock(&g_pool.mtx);
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
