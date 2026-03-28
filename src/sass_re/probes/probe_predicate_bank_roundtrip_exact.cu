/*
 * SASS RE Probe: exact predicate-bank roundtrip follow-up
 *
 * Goal: mirror the cuDNN-local neighborhood more closely:
 *   P2R.B1/B2/B3 on the same carrier register, followed later by a byte-wise
 *   predicate reload and reuse after helper boundaries and divergence.
 *
 * R2P is already directly observed elsewhere in the corpus. This probe keeps
 * it in the loop while focusing on the still-missing byte-bank spellings.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __noinline__ uint32_t
probe_roundtrip_mix(uint32_t carrier, int x0, int x1, int x2, int x3) {
    bool q0 = ((carrier >> 8) & 0x1u) != 0u;
    bool q1 = ((carrier >> 9) & 0x1u) != 0u;
    bool q2 = ((carrier >> 16) & 0x1u) != 0u;
    bool q3 = ((carrier >> 17) & 0x1u) != 0u;
    bool q4 = ((carrier >> 24) & 0x1u) != 0u;
    bool q5 = ((carrier >> 25) & 0x1u) != 0u;

    uint32_t acc = carrier ^ 0x13579bdfu;
    if (q0 && q2) acc += (uint32_t)(x0 ^ x2);
    if (q1 || q3) acc ^= (uint32_t)(x1 + x3);
    if (q4 && !q5) acc += 0x00f000f0u;
    if ((q0 && q1) || (q2 && q4)) acc ^= 0x0f0f0000u;
    asm volatile("" : "+r"(acc));
    return acc;
}

extern "C" __global__ void __launch_bounds__(32)
probe_predicate_bank_roundtrip_exact(uint32_t *out,
                                     const int *in,
                                     const uint32_t *seed_in,
                                     int iters) {
    int lane = threadIdx.x;

    bool p0 = (in[lane + 0 * 32] > 0);
    bool p1 = (in[lane + 1 * 32] < 0);
    bool p2 = (in[lane + 2 * 32] != 0);
    bool p3 = (in[lane + 3 * 32] >= 3);
    bool p4 = (in[lane + 4 * 32] <= 8);
    bool p5 = (in[lane + 5 * 32] == -1);
    bool p6 = (in[lane + 6 * 32] != 11);
    bool p7 = (in[lane + 7 * 32] > -7);
    bool p8 = (in[lane + 8 * 32] < 5);
    bool p9 = (in[lane + 9 * 32] >= -2);
    bool pA = (in[lane + 10 * 32] != 13);
    bool pB = (in[lane + 11 * 32] == 0);

    uint32_t carrier = seed_in[lane] ^ 0x2468ace0u;

    uint32_t bank1 = (uint32_t)p0
                   | ((uint32_t)p1 << 1)
                   | ((uint32_t)p2 << 2)
                   | ((uint32_t)p3 << 3)
                   | ((uint32_t)p4 << 4)
                   | ((uint32_t)p5 << 5)
                   | ((uint32_t)p6 << 6);
    carrier = (carrier & ~0x00007f00u) | (bank1 << 8);
    asm volatile("" : "+r"(carrier));

    uint32_t bank2 = (uint32_t)p4
                   | ((uint32_t)p5 << 1)
                   | ((uint32_t)p6 << 2)
                   | ((uint32_t)p7 << 3);
    carrier = (carrier & ~0x000f0000u) | (bank2 << 16);
    asm volatile("" : "+r"(carrier));

    uint32_t bank3 = (uint32_t)p8
                   | ((uint32_t)p9 << 1)
                   | ((uint32_t)pA << 2)
                   | ((uint32_t)pB << 3);
    carrier = (carrier & ~0x0f000000u) | (bank3 << 24);
    asm volatile("" : "+r"(carrier));

    uint32_t acc = carrier ^ 0x5a5aa5a5u;

    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        bool take_left = ((lane + iter) & 1) == 0;
        if (take_left) {
            acc += (uint32_t)(in[lane + ((iter + 0) & 7) * 32] * 9 + iter);
            carrier ^= 0x00010001u + (uint32_t)(iter << 8);
        } else {
            acc ^= (uint32_t)(in[lane + ((iter + 4) & 7) * 32] * 17 - iter);
            carrier += 0x01000100u ^ (uint32_t)(iter << 4);
        }

        asm volatile("" : "+r"(carrier), "+r"(acc));
        uint32_t helper = probe_roundtrip_mix(
            carrier,
            in[lane + 0 * 32], in[lane + 1 * 32],
            in[lane + 2 * 32], in[lane + 3 * 32]);

        bool r1 = ((carrier >> 8) & 0x3u) != 0u;
        bool r2 = ((carrier >> 16) & 0x3u) != 0u;
        bool r3 = ((carrier >> 24) & 0x3u) != 0u;

        if (r1 && r2) acc += helper ^ 0x11110000u;
        if (r2 || r3) acc ^= helper + 0x00002222u;
        if (r1 && r3) acc += (helper >> 3);
        if (r3 && !r1) acc ^= (helper << 1);
    }

    out[lane] = acc ^ carrier;
}
