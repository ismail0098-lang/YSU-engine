/*
 * SASS RE Probe: byte-carrier predicate bank exact follow-up
 *
 * Goal: keep three byte-like predicate banks alive in parallel carrier
 * registers, then merge and consume them later. This is another attempt to
 * elicit P2R.B1/B2/B3 on the direct local path.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_predicate_bank_bytecarriers_exact(uint32_t *out,
                                        const int *in,
                                        const uint32_t *seed) {
    int lane = threadIdx.x;

    bool p0 = in[lane + 0 * 32] > 0;
    bool p1 = in[lane + 1 * 32] < 0;
    bool p2 = in[lane + 2 * 32] != 0;
    bool p3 = in[lane + 3 * 32] >= 1;

    bool p4 = in[lane + 4 * 32] > -3;
    bool p5 = in[lane + 5 * 32] <= 7;
    bool p6 = in[lane + 6 * 32] == -1;
    bool p7 = in[lane + 7 * 32] != 11;

    bool p8 = in[lane + 8 * 32] > 2;
    bool p9 = in[lane + 9 * 32] < 5;
    bool pA = in[lane + 10 * 32] != 13;
    bool pB = in[lane + 11 * 32] == 0;

    uint32_t c0 = seed[lane] ^ 0x01020304u;
    uint32_t c1 = seed[lane] ^ 0x10203040u;
    uint32_t c2 = seed[lane] ^ 0x89abcdefu;

    uint32_t b0 = (uint32_t)p0 | ((uint32_t)p1 << 1) | ((uint32_t)p2 << 2) | ((uint32_t)p3 << 3);
    uint32_t b1 = (uint32_t)p4 | ((uint32_t)p5 << 1) | ((uint32_t)p6 << 2) | ((uint32_t)p7 << 3);
    uint32_t b2 = (uint32_t)p8 | ((uint32_t)p9 << 1) | ((uint32_t)pA << 2) | ((uint32_t)pB << 3);

    c0 = (c0 & ~0x0000000fu) | b0;
    c1 = (c1 & ~0x00000f00u) | (b1 << 8);
    c2 = (c2 & ~0x000f0000u) | (b2 << 16);
    asm volatile("" : "+r"(c0), "+r"(c1), "+r"(c2));

    uint32_t merged = c0 ^ c1 ^ c2;
    bool q0 = (merged & 0x00000001u) != 0u;
    bool q1 = (merged & 0x00000200u) != 0u;
    bool q2 = (merged & 0x00040000u) != 0u;
    bool q3 = ((merged >> 24) & 0x1u) != 0u;

    if (q0 && q1) merged ^= 0x11110000u;
    if (q2 || q3) merged += 0x00002222u;
    if (q0 && q2 && !q3) merged ^= 0x00f000f0u;
    if (q1 || (q2 && q3)) merged += 0x13579bdfu;

    out[lane] = merged;
}
