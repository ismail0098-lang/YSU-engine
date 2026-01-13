// gpu_bvh.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// std430 uyumlu, vec4 hizalı
typedef struct {
    float bmin[4];   // xyz + pad
    float bmax[4];   // xyz + pad
    int32_t left;    // leaf ise -1
    int32_t right;   // leaf ise -1
    int32_t triOffset;
    int32_t triCount;
} GPUBVHNode;

#ifdef __cplusplus
}
#endif
