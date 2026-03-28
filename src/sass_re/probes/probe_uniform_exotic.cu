/*
 * SASS RE Probe: uniform-path exotic variants
 *
 * Targets library-mined uniform-path spellings that are still missing from the
 * direct local sm_89 corpus:
 *   - ULDC.U8 / ULDC.S8
 *   - USHF.L.U64.HI
 *   - ULEA.HI.X.SX32
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint8_t probe_uniform_u8_table[8] =
    {3u, 7u, 11u, 19u, 23u, 29u, 31u, 37u};
__device__ __constant__ int8_t probe_uniform_s8_table[8] =
    {-3, -7, 11, -19, 23, -29, 31, -37};

extern "C" __global__ void __launch_bounds__(128)
probe_uldc_u8_s8(uint32_t *out, int idx_u, int idx_s, uint32_t bias) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t u = (uint32_t)probe_uniform_u8_table[idx_u & 7];
    int32_t s = (int32_t)probe_uniform_s8_table[idx_s & 7];

    out[i] = ((u + bias) << 8) ^ (uint32_t)(s + i);
}

extern "C" __global__ void __launch_bounds__(128)
probe_ushf_l_u64_hi(uint32_t *out_hi, uint64_t base, uint32_t shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t uniform_val = base + (uint64_t)(shift & 31u);
    uint64_t shifted = uniform_val << (shift & 31u);
    out_hi[i] = (uint32_t)(shifted >> 32);
}

extern "C" __global__ void __launch_bounds__(128)
probe_ulea_hi_x_sx32(float *out, const float *base, int signed_offset_words) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Keep the offset signed and warp-uniform so the compiler has a chance to
    // route the high-half address carry through the uniform SX32 path.
    const float *ptr = base + signed_offset_words;
    out[i] = ptr[i] * 2.0f;
}
