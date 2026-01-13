// bilateral_denoise.c - Edge-aware bilateral filtering for raytraced images
// Separable bilateral filter: spatial kernel (distance) + range kernel (color similarity)

#include "bilateral_denoise.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// ============================================================================
// Bilateral Filter: Edge-aware denoising via spatial + range kernels
// ============================================================================

// Gaussian spatial kernel: exp(-d^2 / (2 * sigma_s^2))
static inline float gauss_spatial(float dist_sq, float sigma_s_sq) {
    return expf(-dist_sq / (2.0f * sigma_s_sq));
}

// Gaussian range kernel: exp(-diff^2 / (2 * sigma_r^2))
// Penalizes large color differences, preserves edges
static inline float gauss_range(float color_diff_sq, float sigma_r_sq) {
    return expf(-color_diff_sq / (2.0f * sigma_r_sq));
}

// Luminance (for range kernel, more perceptually accurate)
static inline float luminance(const Vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// ============================================================================
// 1D Separable Bilateral Pass (horizontal or vertical)
// ============================================================================

typedef struct {
    float sigma_s;      // spatial std dev (pixels)
    float sigma_r;      // range std dev (luminance units)
    int radius;         // filter radius (pixels)
} BilateralParams;

static void bilateral_filter_1d(
    const Vec3 *input,
    Vec3 *output,
    int width, int height,
    int horizontal,          // 1 = horizontal pass, 0 = vertical pass
    const BilateralParams *p)
{
    float sigma_s_sq = p->sigma_s * p->sigma_s;
    float sigma_r_sq = p->sigma_r * p->sigma_r;

    if (horizontal) {
        // Horizontal pass: filter rows
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int center_idx = y * width + x;
                Vec3 center_col = input[center_idx];
                float center_lum = luminance(center_col);

                Vec3 sum = {0, 0, 0};
                float weight_sum = 0.0f;

                for (int dx = -p->radius; dx <= p->radius; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;

                    int neighbor_idx = y * width + nx;
                    Vec3 neighbor_col = input[neighbor_idx];
                    float neighbor_lum = luminance(neighbor_col);

                    float dist_sq = (float)(dx * dx);
                    float lum_diff_sq = (center_lum - neighbor_lum) * (center_lum - neighbor_lum);

                    float w_spatial = gauss_spatial(dist_sq, sigma_s_sq);
                    float w_range = gauss_range(lum_diff_sq, sigma_r_sq);
                    float weight = w_spatial * w_range;

                    sum.x += neighbor_col.x * weight;
                    sum.y += neighbor_col.y * weight;
                    sum.z += neighbor_col.z * weight;
                    weight_sum += weight;
                }

                if (weight_sum > 1e-6f) {
                    output[center_idx].x = sum.x / weight_sum;
                    output[center_idx].y = sum.y / weight_sum;
                    output[center_idx].z = sum.z / weight_sum;
                } else {
                    output[center_idx] = center_col;
                }
            }
        }
    } else {
        // Vertical pass: filter columns
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int center_idx = y * width + x;
                Vec3 center_col = input[center_idx];
                float center_lum = luminance(center_col);

                Vec3 sum = {0, 0, 0};
                float weight_sum = 0.0f;

                for (int dy = -p->radius; dy <= p->radius; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    int neighbor_idx = ny * width + x;
                    Vec3 neighbor_col = input[neighbor_idx];
                    float neighbor_lum = luminance(neighbor_col);

                    float dist_sq = (float)(dy * dy);
                    float lum_diff_sq = (center_lum - neighbor_lum) * (center_lum - neighbor_lum);

                    float w_spatial = gauss_spatial(dist_sq, sigma_s_sq);
                    float w_range = gauss_range(lum_diff_sq, sigma_r_sq);
                    float weight = w_spatial * w_range;

                    sum.x += neighbor_col.x * weight;
                    sum.y += neighbor_col.y * weight;
                    sum.z += neighbor_col.z * weight;
                    weight_sum += weight;
                }

                if (weight_sum > 1e-6f) {
                    output[center_idx].x = sum.x / weight_sum;
                    output[center_idx].y = sum.y / weight_sum;
                    output[center_idx].z = sum.z / weight_sum;
                } else {
                    output[center_idx] = center_col;
                }
            }
        }
    }
}

// ============================================================================
// Main API: Separable Bilateral Filter
// ============================================================================

void bilateral_denoise(Vec3 *pixels, int width, int height,
                       float sigma_s, float sigma_r, int radius)
{
    if (!pixels || width <= 0 || height <= 0 || radius < 1) return;

    // Allocate temporary buffer for intermediate pass
    Vec3 *temp = (Vec3*)malloc((size_t)width * (size_t)height * sizeof(Vec3));
    if (!temp) {
        fprintf(stderr, "[DENOISE] malloc failed for temp buffer\n");
        return;
    }

    BilateralParams p;
    p.sigma_s = sigma_s;
    p.sigma_r = sigma_r;
    p.radius = radius;

    // Horizontal pass: input -> temp
    bilateral_filter_1d(pixels, temp, width, height, 1, &p);

    // Vertical pass: temp -> output (pixels)
    bilateral_filter_1d(temp, pixels, width, height, 0, &p);

    free(temp);

    fprintf(stderr, "[DENOISE] bilateral complete: sigma_s=%.2f sigma_r=%.4f radius=%d\n",
            sigma_s, sigma_r, radius);
}

// ============================================================================
// Environment-based configuration
// ============================================================================

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

void bilateral_denoise_maybe(Vec3 *pixels, int width, int height)
{
    if (!pixels || width <= 0 || height <= 0) return;

    int enabled = ysu_env_int("YSU_BILATERAL_DENOISE", 0) ? 1 : 0;
    if (!enabled) return;

    // Configuration via environment variables
    float sigma_s = ysu_env_float("YSU_BILATERAL_SIGMA_S", 1.5f);   // spatial (pixels)
    float sigma_r = ysu_env_float("YSU_BILATERAL_SIGMA_R", 0.1f);   // range (luminance)
    int radius = ysu_env_int("YSU_BILATERAL_RADIUS", 3);            // filter radius

    if (sigma_s < 0.1f) sigma_s = 0.1f;
    if (sigma_r < 0.01f) sigma_r = 0.01f;
    if (radius < 1) radius = 1;
    if (radius > 20) radius = 20;

    fprintf(stderr, "[DENOISE] bilateral enabled: sigma_s=%.2f sigma_r=%.4f radius=%d\n",
            sigma_s, sigma_r, radius);

    bilateral_denoise(pixels, width, height, sigma_s, sigma_r, radius);
}
