// gpu_bvh_lbv.h
#pragma once
#include <stdint.h>
#include <stdbool.h>

// IMPORTANT: Use the project's canonical GPUBVHNode definition
#include "gpu_bvh.h"

#ifdef __cplusplus
extern "C" {
#endif

// tri_vec4 layout: each triangle = 3 * vec4 (xyz + pad), so float count = tri_count * 12
// Output:
//  - out_nodes: malloc'd array of GPUBVHNode, length out_node_count
//  - out_indices: malloc'd int32 list of triangle IDs (sorted by morton), length out_index_count (= tri_count)
bool gpu_build_bvh_from_tri_vec4_lbv(
    const float* tri_vec4,
    uint32_t tri_count,
    GPUBVHNode** out_nodes,
    uint32_t* out_node_count,
    int32_t** out_indices,
    uint32_t* out_index_count
);

#ifdef __cplusplus
}
#endif
