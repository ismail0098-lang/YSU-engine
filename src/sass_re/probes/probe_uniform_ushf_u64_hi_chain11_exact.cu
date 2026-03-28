/*
 * SASS RE Probe: chain-11 uniform U64.HI follow-up
 *
 * Goal: mirror the library-mined immediate-11 uniform shift chain more
 * closely by feeding multiple dependent 64-bit shifts and reusing both the
 * high half and the rebased byte offset.
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint64_t probe_uniform_chain11_table[4] = {
    0x0000000100000001ull,
    0x0000000200000003ull,
    0x0000000400000007ull,
    0x000000080000000full,
};

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_u64_hi_chain11_exact(float *out,
                                        const float *base,
                                        uint32_t sel) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t x0 = probe_uniform_chain11_table[(sel >> 0) & 0x3u];
    uint64_t x1 = probe_uniform_chain11_table[(sel >> 2) & 0x3u];

    uint64_t y0 = (x0 << 11) + x1;
    uint64_t y1 = (y0 << 11) + x0;
    uint64_t y2 = (y1 << 11) + x1;

    uint32_t hi0 = (uint32_t)(y0 >> 32);
    uint32_t hi1 = (uint32_t)(y1 >> 32);
    uint32_t hi2 = (uint32_t)(y2 >> 32);

    uint64_t byte_off = y2 & 0xfffull;
    const char *base_bytes = (const char *)base + byte_off;
    const float *ptr = (const float *)(base_bytes + ((size_t)(threadIdx.x & 3) << 2));

    float v = ptr[0];
    v += (float)((hi0 ^ hi1) & 0xffu) * 0.0009765625f;
    v += (float)(hi2 & 0xffu) * 0.001953125f;
    out[tid] = v;
}
