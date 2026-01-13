// bilateral_denoise.h - Edge-aware bilateral filtering header

#pragma once

#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bilateral filter: combines spatial (distance) and range (color) kernels
// Preserves edges while smoothing noise
//
// Parameters:
//   pixels    - input/output pixel array (modified in-place)
//   width     - image width
//   height    - image height
//   sigma_s   - spatial std dev (pixels). Higher = larger filter radius. Typical: 1.0-2.0
//   sigma_r   - range std dev (luminance units 0..1). Higher = preserve more detail. Typical: 0.05-0.2
//   radius    - filter support radius (pixels). Typical: 2-5
//
// Typical usage for 4 SPP -> looks like 32-64 SPP:
//   bilateral_denoise(pixels, w, h, 1.5f, 0.1f, 3);
//
void bilateral_denoise(Vec3 *pixels, int width, int height,
                       float sigma_s, float sigma_r, int radius);

// Environment-controlled version
// Reads: YSU_BILATERAL_DENOISE, YSU_BILATERAL_SIGMA_S, YSU_BILATERAL_SIGMA_R, YSU_BILATERAL_RADIUS
void bilateral_denoise_maybe(Vec3 *pixels, int width, int height);

#ifdef __cplusplus
}
#endif
