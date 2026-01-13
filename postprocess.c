#include "postprocess.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static inline float clampf(float x, float a, float b) {
    return x < a ? a : (x > b ? b : x);
}
static inline float maxf(float a, float b) { return a > b ? a : b; }

static inline float luminance(float r, float g, float b) {
    // Rec.709
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

// Soft threshold (bloom) with "knee" (smooth ramp)
static inline float soft_threshold(float x, float threshold, float knee) {
    if (knee <= 0.0f) return x > threshold ? (x - threshold) : 0.0f;

    float t0 = threshold - knee;
    float t1 = threshold + knee;

    if (x <= t0) return 0.0f;
    if (x >= t1) return x - threshold;

    float s = (x - t0) / (t1 - t0);      // 0..1
    float smooth = s*s*(3.0f - 2.0f*s);  // smoothstep
    return smooth * (t1 - threshold);
}

static inline void aces_tonemap(float* r, float* g, float* b) {
    // ACES fitted
    const float a = 2.51f;
    const float bb = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    float R = *r, G = *g, B = *b;

    R = (R*(a*R + bb)) / (R*(c*R + d) + e);
    G = (G*(a*G + bb)) / (G*(c*G + d) + e);
    B = (B*(a*B + bb)) / (B*(c*B + d) + e);

    *r = clampf(R, 0.0f, 1.0f);
    *g = clampf(G, 0.0f, 1.0f);
    *b = clampf(B, 0.0f, 1.0f);
}

static inline float linear_to_srgb_gamma22(float x) {
    // Fast gamma approximation (ok for this engine stage)
    return powf(clampf(x, 0.0f, 1.0f), 1.0f/2.2f);
}

// 5-tap separable kernel: [1 4 6 4 1] / 16
static void blur_h(const float* src, float* dst, int w, int h) {
    const float k0 = 1.0f/16.0f, k1 = 4.0f/16.0f, k2 = 6.0f/16.0f;
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int xm2 = x-2; if (xm2 < 0) xm2 = 0;
            int xm1 = x-1; if (xm1 < 0) xm1 = 0;
            int xp1 = x+1; if (xp1 >= w) xp1 = w-1;
            int xp2 = x+2; if (xp2 >= w) xp2 = w-1;

            const float* p0 = &src[(y*w + xm2)*4];
            const float* p1 = &src[(y*w + xm1)*4];
            const float* p2 = &src[(y*w + x  )*4];
            const float* p3 = &src[(y*w + xp1)*4];
            const float* p4 = &src[(y*w + xp2)*4];

            float* o = &dst[(y*w + x)*4];
            for (int c=0; c<3; c++) {
                o[c] = p0[c]*k0 + p1[c]*k1 + p2[c]*k2 + p3[c]*k1 + p4[c]*k0;
            }
            o[3] = 1.0f;
        }
    }
}

static void blur_v(const float* src, float* dst, int w, int h) {
    const float k0 = 1.0f/16.0f, k1 = 4.0f/16.0f, k2 = 6.0f/16.0f;
    for (int y=0; y<h; y++) {
        int ym2 = y-2; if (ym2 < 0) ym2 = 0;
        int ym1 = y-1; if (ym1 < 0) ym1 = 0;
        int yp1 = y+1; if (yp1 >= h) yp1 = h-1;
        int yp2 = y+2; if (yp2 >= h) yp2 = h-1;

        for (int x=0; x<w; x++) {
            const float* p0 = &src[(ym2*w + x)*4];
            const float* p1 = &src[(ym1*w + x)*4];
            const float* p2 = &src[(y  *w + x)*4];
            const float* p3 = &src[(yp1*w + x)*4];
            const float* p4 = &src[(yp2*w + x)*4];

            float* o = &dst[(y*w + x)*4];
            for (int c=0; c<3; c++) {
                o[c] = p0[c]*k0 + p1[c]*k1 + p2[c]*k2 + p3[c]*k1 + p4[c]*k0;
            }
            o[3] = 1.0f;
        }
    }
}

void ysu_apply_bloom_tonemap_u8(
    const float* hdr_rgba, int w, int h,
    unsigned char* out_rgb_u8,
    const PostFX* fx_in
){
    if (!hdr_rgba || !out_rgb_u8 || w <= 0 || h <= 0) return;

    PostFX fx;
    fx.exposure = 1.0f;
    fx.bloom_threshold = 1.2f;
    fx.bloom_knee = 0.6f;
    fx.bloom_intensity = 0.15f;
    fx.bloom_iterations = 2;
    if (fx_in) fx = *fx_in;

    size_t n = (size_t)w * (size_t)h;
    float* bright = (float*)malloc(n * 4 * sizeof(float));
    float* ping   = (float*)malloc(n * 4 * sizeof(float));
    float* pong   = (float*)malloc(n * 4 * sizeof(float));
    if (!bright || !ping || !pong) {
        free(bright); free(ping); free(pong);
        return;
    }

    // 1) bright-pass in linear HDR (after exposure)
    for (size_t i=0; i<n; i++) {
        float r = hdr_rgba[i*4+0] * fx.exposure;
        float g = hdr_rgba[i*4+1] * fx.exposure;
        float b = hdr_rgba[i*4+2] * fx.exposure;

        float l = luminance(r,g,b);
        float t = soft_threshold(l, fx.bloom_threshold, fx.bloom_knee);

        float scale = (l > 1e-6f) ? (t / l) : 0.0f;

        bright[i*4+0] = r * scale;
        bright[i*4+1] = g * scale;
        bright[i*4+2] = b * scale;
        bright[i*4+3] = 1.0f;
    }

    // 2) blur
    memcpy(ping, bright, n*4*sizeof(float));
    int iters = fx.bloom_iterations;
    if (iters < 1) iters = 1;
    if (iters > 8) iters = 8;

    for (int k=0; k<iters; k++) {
        blur_h(ping, pong, w, h);
        blur_v(pong, ping, w, h);
    }
    // ping = blurred bloom

    // 3) combine + tonemap + gamma
    for (size_t i=0; i<n; i++) {
        float r = hdr_rgba[i*4+0] * fx.exposure + ping[i*4+0] * fx.bloom_intensity;
        float g = hdr_rgba[i*4+1] * fx.exposure + ping[i*4+1] * fx.bloom_intensity;
        float b = hdr_rgba[i*4+2] * fx.exposure + ping[i*4+2] * fx.bloom_intensity;

        r = maxf(r, 0.0f); g = maxf(g, 0.0f); b = maxf(b, 0.0f);

        aces_tonemap(&r, &g, &b);

        r = linear_to_srgb_gamma22(r);
        g = linear_to_srgb_gamma22(g);
        b = linear_to_srgb_gamma22(b);

        out_rgb_u8[i*3+0] = (unsigned char)(clampf(r,0,1) * 255.0f + 0.5f);
        out_rgb_u8[i*3+1] = (unsigned char)(clampf(g,0,1) * 255.0f + 0.5f);
        out_rgb_u8[i*3+2] = (unsigned char)(clampf(b,0,1) * 255.0f + 0.5f);
    }

    free(bright);
    free(ping);
    free(pong);
}
