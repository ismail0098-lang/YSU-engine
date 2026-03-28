/*
 * SASS RE Probe: exact uniform U64 carry-chain follow-up
 *
 * Goal: keep a warp-uniform 64-bit byte offset alive through multiple dependent
 * left shifts, carry-producing adds, address-generation uses, and a parallel
 * integer export of the high half. This is a stricter follow-up for the
 * remaining USHF.L.U64.HI gap.
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint64_t probe_uniform_carry_table[4] = {
    0x0000000100000001ull,
    0x0000000200000003ull,
    0x0000000400000007ull,
    0x000000080000000full,
};

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_u64_carry_chain_exact(float *out,
                                    const float *base,
                                    uint32_t shift_a,
                                    uint32_t shift_b,
                                    uint32_t select_mask) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t x0 = probe_uniform_carry_table[(select_mask >> 0) & 0x3u];
    uint64_t x1 = probe_uniform_carry_table[(select_mask >> 2) & 0x3u];
    uint64_t x2 = probe_uniform_carry_table[(select_mask >> 4) & 0x3u];

    uint32_t sa = shift_a & 7u;
    uint32_t sb = shift_b & 7u;
    uint64_t y0 = (x0 << sa) + (x1 << sb);
    uint64_t y1 = (y0 << ((select_mask >> 6) & 0x7u)) + x2;
    uint64_t y2 = (y1 << 5) + (x0 << 3);

    uint32_t hi0 = (uint32_t)(y0 >> 32);
    uint32_t hi1 = (uint32_t)(y1 >> 32);
    uint32_t hi2 = (uint32_t)(y2 >> 32);

    uint64_t bytes = y2 & 0xfffull;
    const char *block_ptr = (const char *)base + bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 3) << 2));

    float v = ptr[0];
    v += (float)((hi0 ^ hi1) & 0xffu) * 0.001953125f;
    v += (float)(hi2 & 0xffu) * 0.0009765625f;
    out[i] = v;
}
