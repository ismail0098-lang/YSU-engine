// neural_denoise.h - Stage-1 neural render integration (stub, ONNX-ready)
#pragma once

#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// If YSU_NEURAL_DENOISE=1, runs the postprocess denoiser.
// Today: safe placeholder (edge-aware-ish box).
// Later: swap impl with ONNX Runtime / TensorRT.
void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height);

#ifdef __cplusplus
}
#endif
