/*
 * SASS RE Probe: subword constant-memory LDC variants
 *
 * Goal: directly confirm local LDC.U8 and LDC.S8 spellings, complementing the
 * already confirmed uniform-path ULDC.U8 and ULDC.S8 forms.
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint8_t probe_ldc_u8_table[8] =
    {1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u};
__device__ __constant__ int8_t probe_ldc_s8_table[8] =
    {-1, -3, 5, -7, 9, -11, 13, -15};

extern "C" __global__ void __launch_bounds__(128)
probe_ldc_u8(uint32_t *out, int idx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t v = (uint32_t)probe_ldc_u8_table[(idx + i) & 7];
    out[i] = v + 0x10u;
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldc_s8(int32_t *out, int idx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t v = (int32_t)probe_ldc_s8_table[(idx + i) & 7];
    out[i] = v - 3;
}
