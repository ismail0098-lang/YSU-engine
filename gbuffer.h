#pragma once
#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Vec3 *normal;   // RGB float32 (xyz)
    Vec3 *albedo;   // RGB float32
    float *depth;   // float32 (t veya mesafe)
    int width, height;
} YSU_GBuffer;

// render.c içinden erişilecek global target
void ysu_gbuffer_set_targets(YSU_GBuffer gb);

#ifdef __cplusplus
}
#endif
