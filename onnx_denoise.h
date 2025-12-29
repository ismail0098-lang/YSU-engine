#pragma once
#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// YSU_NEURAL_DENOISE=1 ise ONNX denoise çalıştırır.
// YSU_ONNX_MODEL="path/to/model.onnx"
// Beklenen IO: float32 NCHW [1,3,H,W] -> [1,3,H,W] (0..1 aralığı)
void ysu_neural_denoise_maybe(Vec3 *pixels, int width, int height);

#ifdef __cplusplus
}
#endif
