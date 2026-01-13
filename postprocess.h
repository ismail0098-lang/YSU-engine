#pragma once
#include <stddef.h>

typedef struct {
    float exposure;         // 1.0 = default
    float bloom_threshold;  // 0.8 - 2.0 typical
    float bloom_knee;       // 0.2 - 1.0 soft threshold width
    float bloom_intensity;  // 0.05 - 0.5 typical
    int   bloom_iterations; // 1 - 4 typical (caps internally)
} PostFX;

// hdr_rgba: linear HDR, size = w*h*4 floats
// out_rgb_u8: size = w*h*3 bytes (8-bit, display-ready)
void ysu_apply_bloom_tonemap_u8(
    const float* hdr_rgba, int w, int h,
    unsigned char* out_rgb_u8,
    const PostFX* fx
);
