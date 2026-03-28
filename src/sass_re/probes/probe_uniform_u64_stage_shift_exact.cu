/*
 * SASS RE Probe: exact uniform u64 stage-shift follow-up
 *
 * Goal: keep the entire carry/shift chain in the uniform domain while still
 * consuming the high 32 bits in an address-like sink. This is the last clean
 * attempt at the remaining USHF.L.U64.HI gap.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_u64_stage_shift_exact(float *out,
                                    const float *base,
                                    uint64_t seed,
                                    int shift_sel,
                                    int stage_mask,
                                    int rounds) {
    int lane = threadIdx.x;

    uint64_t u = seed ^ ((uint64_t)(stage_mask & 0xff) << 32);
    uint64_t mix = ((uint64_t)(stage_mask & 0x40) << 21)
                 | ((uint64_t)(stage_mask & 0x80) << 13)
                 | 0x0000001000000001ull;

    #pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        int s = ((shift_sel + i) & 0x1f) + 1;
        uint64_t left = u << 11;
        uint64_t right = u >> s;
        u = left ^ right ^ mix ^ ((uint64_t)(i + 1) << 32);
        if ((stage_mask & 0x40) != 0) {
            u += 0x0000000100000001ull;
        }
        if ((stage_mask & 0x80) != 0) {
            u ^= 0x0000000200000020ull;
        }
        asm volatile("" : "+l"(u));
    }

    uint32_t lo = (uint32_t)u;
    uint32_t hi = (uint32_t)(u >> 32);
    uint32_t idx = (lo ^ hi ^ (uint32_t)(stage_mask << 1)) & 31u;
    float val = base[idx] + (float)((hi >> 1) & 0xffu);
    out[lane] = val;
}
