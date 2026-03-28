/*
 * SASS RE Probe: stricter warp-uniform address and shift forms
 *
 * Goal: keep the interesting computation warp-uniform all the way to the load
 * or final scalar result, so ptxas has the best chance to choose fully uniform
 * address and shift spellings such as ULEA.HI.X.SX32 and USHF.L.U64.HI.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_load_broadcast(float *out, const float *base, int signed_offset_words) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    const float *ptr = base + signed_offset_words;
    float v = ptr[0];
    out[i] = v;
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_shift64_broadcast(uint32_t *out, uint64_t base, uint32_t shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t s = shift & 31u;
    uint64_t shifted = (base + (uint64_t)s) << s;
    out[i] = (uint32_t)(shifted >> 32);
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_shift64_lane0(uint32_t *out, uint64_t base, uint32_t shift) {
    uint32_t s = shift & 31u;
    uint64_t shifted = (base + (uint64_t)s) << s;

    if ((threadIdx.x | blockIdx.x) == 0) {
        out[0] = (uint32_t)(shifted >> 32);
    }
}
