/*
 * nerf_simd_integration.c
 * 
 * Integration example showing how to call nerf_simd functions from render.c
 * This file demonstrates the ray batching loop and framebuffer management.
 * 
 * Copy-paste relevant sections into render.c or use as reference.
 */

#include "nerf_simd.h"
#include "camera.h"
#include "ray.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "image.h"

/* ===== Global NeRF Data (load once, reuse per-frame) ===== */

static NeRFData *g_nerf_data = NULL;
static NeRFFramebuffer g_nerf_framebuffer = {0};

/* ===== Initialization (call once at startup) ===== */

void ysu_nerf_init(const char *hashgrid_path, const char *occ_path, uint32_t width, uint32_t height) {
    printf("\n[NeRF] Initializing CPU SIMD NeRF renderer...\n");
    
    /* Load NeRF data from binary */
    g_nerf_data = ysu_nerf_data_load(hashgrid_path, occ_path);
    if (!g_nerf_data) {
        fprintf(stderr, "[NeRF] ERROR: Failed to load NeRF data\n");
        return;
    }
    
    /* Allocate framebuffer */
    g_nerf_framebuffer.width = width;
    g_nerf_framebuffer.height = height;
    /* Cast to size_t first to prevent uint32_t overflow at large dimensions. */
    g_nerf_framebuffer.pixels = (NeRFPixel*)malloc((size_t)width * (size_t)height * sizeof(NeRFPixel));
    
    if (!g_nerf_framebuffer.pixels) {
        fprintf(stderr, "[NeRF] ERROR: Failed to allocate framebuffer\n");
        return;
    }
    
    printf("[NeRF] Initialized: %ux%u framebuffer\n", width, height);
    printf("[NeRF] Config: %u levels, %u hash entries, %u->%u->%u MLP\n",
           g_nerf_data->config.num_levels,
           g_nerf_data->config.hashmap_size,
           g_nerf_data->config.mlp_in_dim,
           g_nerf_data->config.mlp_hidden_dim,
           g_nerf_data->config.mlp_out_dim);
}

void ysu_nerf_shutdown(void) {
    if (g_nerf_data) {
        ysu_nerf_data_free(g_nerf_data);
        g_nerf_data = NULL;
    }
    if (g_nerf_framebuffer.pixels) {
        free(g_nerf_framebuffer.pixels);
        g_nerf_framebuffer.pixels = NULL;
    }
}

/* ===== Main Rendering Function (call once per frame) ===== */

void ysu_render_nerf_frame(
    const Camera *camera,
    uint32_t width,
    uint32_t height,
    uint32_t num_steps,
    float density_scale,
    float bounds_max
) {
    if (!g_nerf_data || !g_nerf_framebuffer.pixels) {
        fprintf(stderr, "[NeRF] ERROR: Not initialized. Call ysu_nerf_init() first\n");
        return;
    }
    
    /* Clear framebuffer — use size_t loop counter to avoid uint32_t overflow. */
    size_t fb_pixel_count = (size_t)width * (size_t)height;
    for (size_t i = 0; i < fb_pixel_count; i++) {
        g_nerf_framebuffer.pixels[i].rgb.x = 0.0f;
        g_nerf_framebuffer.pixels[i].rgb.y = 0.0f;
        g_nerf_framebuffer.pixels[i].rgb.z = 0.0f;
        g_nerf_framebuffer.pixels[i].alpha = 0.0f;
    }
    
    /* Ray batching loop */
    RayBatch batch = {0};
    batch.count = 0;
    
    clock_t start_time = clock();
    uint32_t ray_count = 0;
    
    for (uint32_t py = 0; py < height; py++) {
        for (uint32_t px = 0; px < width; px++) {
            /* Generate camera ray (normalized coords) */
            float u = ((float)px + 0.5f) / (float)width;
            float v = ((float)py + 0.5f) / (float)height;
            Ray ray = camera_get_ray(*camera, u, v);
            
            /* Add to batch */
            batch.origin[batch.count] = ray.origin;
            batch.direction[batch.count] = ray.direction;
            batch.tmin[batch.count] = 0.0f;
            batch.tmax[batch.count] = 1e9f;
            batch.pixel_id[batch.count] = py * width + px;
            batch.active[batch.count] = 1;
            batch.count++;
            ray_count++;
            
            /* Process full batch or end of image */
            if (batch.count == SIMD_BATCH_SIZE || (py == height - 1 && px == width - 1)) {
                /* Pad inactive lanes */
                for (uint32_t i = batch.count; i < SIMD_BATCH_SIZE; i++) {
                    batch.active[i] = 0;
                }
                
                /* Render batch */
                ysu_volume_integrate_batch(
                    &batch,
                    &g_nerf_data->config,
                    g_nerf_data,
                    &g_nerf_framebuffer,
                    num_steps,
                    density_scale,
                    bounds_max
                );
                
                batch.count = 0;
            }
        }
        
        /* Progress indicator */
        if ((py + 1) % 100 == 0) {
            printf("[NeRF] Row %u / %u\n", py + 1, height);
        }
    }
    
    clock_t end_time = clock();
    double elapsed_ms = (start_time != (clock_t)-1 && end_time != (clock_t)-1)
        ? ((double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0)
        : 0.0;
    
    printf("[NeRF] Rendered %u rays in %.2f ms (%.1f rays/ms)\n",
           ray_count, elapsed_ms, ray_count / elapsed_ms);

    /* Debug PNG dump: gated behind YSU_NERF_DEBUG_PNG (matches render_nerf_cpu pattern). */
    if (getenv("YSU_NERF_DEBUG_PNG")) {
        const char *exp_s = getenv("YSU_NERF_EXPOSURE");
        float exposure = exp_s ? atof(exp_s) : 1.0f;
        printf("[NeRF] Writing debug PNGs (exposure=%.2f)...\n", exposure);
        uint32_t w = g_nerf_framebuffer.width;
        uint32_t h = g_nerf_framebuffer.height;

        /* Allocate 8-bit RGB buffer */
        unsigned char *rgb8 = (unsigned char*)malloc((size_t)w * (size_t)h * 3);
        if (rgb8) {
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    uint32_t idx = y * w + x;
                    NeRFPixel pix = g_nerf_framebuffer.pixels[idx];
                    // Reinhard tonemap: L' = L / (1+L)
                    float r = pix.rgb.x * exposure;
                    float g = pix.rgb.y * exposure;
                    float b = pix.rgb.z * exposure;
                    r = r / (1.0f + r);
                    g = g / (1.0f + g);
                    b = b / (1.0f + b);
                    // Gamma correction (2.2)
                    r = powf(fmaxf(0.0f, r), 1.0f / 2.2f);
                    g = powf(fmaxf(0.0f, g), 1.0f / 2.2f);
                    b = powf(fmaxf(0.0f, b), 1.0f / 2.2f);
                    rgb8[(idx * 3) + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f);
                    rgb8[(idx * 3) + 1] = (unsigned char)(fminf(g, 1.0f) * 255.0f);
                    rgb8[(idx * 3) + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f);
                }
            }

            image_write_png("nerf_debug_scaled.png", w, h, rgb8);
            printf("[NeRF] Wrote nerf_debug_scaled.png (exposure=%.2f)\n", exposure);
            free(rgb8);
        } else {
            fprintf(stderr, "[NeRF] failed to allocate rgb8 for debug dump\n");
        }

        /* Also write alpha as grayscale PNG for quick inspection */
        unsigned char *a8 = (unsigned char*)malloc((size_t)w * (size_t)h * 3);
        if (a8) {
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    uint32_t idx = y * w + x;
                    float a = fmaxf(0.0f, fminf(1.0f, g_nerf_framebuffer.pixels[idx].alpha));
                    unsigned char v = (unsigned char)(a * 255.0f);
                    a8[(idx * 3) + 0] = v;
                    a8[(idx * 3) + 1] = v;
                    a8[(idx * 3) + 2] = v;
                }
            }
            image_write_png("nerf_alpha.png", w, h, a8);
            printf("[NeRF] Wrote nerf_alpha.png\n");
            free(a8);
        }
    } /* end YSU_NERF_DEBUG_PNG */
}

/* ===== Framebuffer Export ===== */

void ysu_nerf_framebuffer_to_ppm(const char *filename) {
    if (!g_nerf_framebuffer.pixels) return;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    uint32_t w = g_nerf_framebuffer.width;
    uint32_t h = g_nerf_framebuffer.height;
    
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t idx = y * w + x;
            NeRFPixel pix = g_nerf_framebuffer.pixels[idx];
            
            uint8_t r = (uint8_t)(fmaxf(0.0f, fminf(1.0f, pix.rgb.x)) * 255.0f);
            uint8_t g = (uint8_t)(fmaxf(0.0f, fminf(1.0f, pix.rgb.y)) * 255.0f);
            uint8_t b = (uint8_t)(fmaxf(0.0f, fminf(1.0f, pix.rgb.z)) * 255.0f);
            
            fputc(r, f);
            fputc(g, f);
            fputc(b, f);
        }
    }
    
    fclose(f);
    printf("[NeRF] Wrote %s\n", filename);
}

void ysu_nerf_framebuffer_to_alpha_ppm(const char *filename) {
    if (!g_nerf_framebuffer.pixels) return;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    uint32_t w = g_nerf_framebuffer.width;
    uint32_t h = g_nerf_framebuffer.height;
    
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t idx = y * w + x;
            NeRFPixel pix = g_nerf_framebuffer.pixels[idx];
            
            /* Alpha channel as grayscale */
            uint8_t a = (uint8_t)(fmaxf(0.0f, fminf(1.0f, pix.alpha)) * 255.0f);
            
            fputc(a, f);
            fputc(a, f);
            fputc(a, f);
        }
    }
    
    fclose(f);
    printf("[NeRF] Wrote alpha map %s\n", filename);
}

/* ===== Environment Variable Helpers ===== */

uint32_t ysu_nerf_get_steps(void) {
    const char *env = getenv("YSU_NERF_STEPS");
    return env ? atoi(env) : 32;
}

float ysu_nerf_get_density(void) {
    const char *env = getenv("YSU_NERF_DENSITY");
    return env ? atof(env) : 1.0f;
}

float ysu_nerf_get_bounds(void) {
    const char *env = getenv("YSU_NERF_BOUNDS");
    return env ? atof(env) : 4.0f;
}

/* Expose internal framebuffer for external copying/debugging */
NeRFFramebuffer* ysu_nerf_get_framebuffer(void) {
    return &g_nerf_framebuffer;
}

/* ===== Integration into render.c ===== */

/*
 * In render.c, add:
 * 
 * #include "nerf_simd.h"
 * 
 * In the main render function:
 * 
 * void render_scene_nerf_simd(const Camera *cam, Framebuffer *fb) {
 *     uint32_t steps = ysu_nerf_get_steps();
 *     float density = ysu_nerf_get_density();
 *     float bounds = ysu_nerf_get_bounds();
 *     
 *     ysu_render_nerf_frame(cam, fb->width, fb->height, steps, density, bounds);
 *     
 *     // Copy NeRF framebuffer to output framebuffer
 *     for (uint32_t y = 0; y < fb->height; y++) {
 *         for (uint32_t x = 0; x < fb->width; x++) {
 *             uint32_t idx = y * fb->width + x;
 *             NeRFPixel nerf_pix = g_nerf_framebuffer.pixels[idx];
 *             fb->pixels[idx].r = (uint8_t)(nerf_pix.rgb.x * 255.0f);
 *             fb->pixels[idx].g = (uint8_t)(nerf_pix.rgb.y * 255.0f);
 *             fb->pixels[idx].b = (uint8_t)(nerf_pix.rgb.z * 255.0f);
 *         }
 *     }
 * }
 * 
 * At startup (in main):
 * 
 *     ysu_nerf_init("models/nerf_hashgrid.bin", "models/occupancy_grid.bin", width, height);
 * 
 * At shutdown (in main cleanup):
 * 
 *     ysu_nerf_shutdown();
 */
