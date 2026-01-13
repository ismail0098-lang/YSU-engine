#ifndef GPU_OBJ_LOADER_H
#define GPU_OBJ_LOADER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float v0[4];  // xyz + pad
    float v1[4];
    float v2[4];
} GPUTriangle;

// Loads triangles from a Wavefront OBJ (supports v and f lines; triangulates polygons).
// Returns 1 on success, 0 on failure.
// On success, allocates *out_tris (free with free()) and sets *out_count.
int gpu_load_obj_triangles(const char* path, GPUTriangle** out_tris, size_t* out_count);

// Creates a simple fallback cube (12 triangles), centered at (0,0,-3) with size ~2.
// Allocates *out_tris (free with free()) and sets *out_count = 12.
void gpu_make_fallback_cube(GPUTriangle** out_tris, size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif
