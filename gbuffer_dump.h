// gbuffer_dump.h - simple float32 buffer dumps for ML/neural denoising pipelines
#pragma once

#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes interleaved float32 RGB buffer (Vec3 per pixel) to a .ysub file.
// Returns 1 on success, 0 on failure.
int ysu_dump_rgb32(const char *path, const Vec3 *rgb, int width, int height);

// Writes float32 single-channel buffer to a .ysub file.
// Returns 1 on success, 0 on failure.
int ysu_dump_f32(const char *path, const float *buf, int width, int height);

#ifdef __cplusplus
}
#endif
