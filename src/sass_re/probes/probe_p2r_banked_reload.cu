/*
 * SASS RE Probe: banked predicate save / restore lifecycle
 *
 * cuDNN-mined P2R.B1/B2/B3 sites look like long-lived packed predicate masks:
 * fill one byte bank, keep the carrier live, fill later banks, then decode the
 * packed banks back into predicates or mask logic later. This probe mirrors
 * that lifecycle more closely than the earlier one-shot pack attempts.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_bank_lifecycle_0f(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;

    bool p0 = (in[i + 0 * 32] > 0);
    bool p1 = (in[i + 1 * 32] < 0);
    bool p2 = (in[i + 2 * 32] != 0);
    bool p3 = (in[i + 3 * 32] >= 7);

    uint32_t carrier_a = seed_in[i];
    uint32_t bank0 = (uint32_t)p0
                   | ((uint32_t)p1 << 1)
                   | ((uint32_t)p2 << 2)
                   | ((uint32_t)p3 << 3);
    carrier_a = (carrier_a & ~0x0000000fu) | bank0;
    asm volatile("" : "+r"(carrier_a));

    bool p4 = (in[i + 4 * 32] > -2);
    bool p5 = (in[i + 5 * 32] <= 3);
    bool p6 = (in[i + 6 * 32] == 9);
    bool p7 = (in[i + 7 * 32] != -5);

    uint32_t bank1 = (uint32_t)p4
                   | ((uint32_t)p5 << 1)
                   | ((uint32_t)p6 << 2)
                   | ((uint32_t)p7 << 3);
    carrier_a = (carrier_a & ~0x00000f00u) | (bank1 << 8);
    asm volatile("" : "+r"(carrier_a));

    uint32_t carrier_b = carrier_a ^ 0x00f000f0u;
    bool q0 = (in[i + 8 * 32] > 1);
    bool q1 = (in[i + 9 * 32] < 4);
    bool q2 = (in[i + 10 * 32] != 2);
    bool q3 = (in[i + 11 * 32] >= 6);

    uint32_t bank2 = (uint32_t)q0
                   | ((uint32_t)q1 << 1)
                   | ((uint32_t)q2 << 2)
                   | ((uint32_t)q3 << 3);
    carrier_b = (carrier_b & ~0x000f0000u) | (bank2 << 16);
    asm volatile("" : "+r"(carrier_b));

    bool q4 = (in[i + 12 * 32] > -3);
    bool q5 = (in[i + 13 * 32] <= 5);
    bool q6 = (in[i + 14 * 32] == 11);
    bool q7 = (in[i + 15 * 32] != -7);

    uint32_t bank3 = (uint32_t)q4
                   | ((uint32_t)q5 << 1)
                   | ((uint32_t)q6 << 2)
                   | ((uint32_t)q7 << 3);
    carrier_b = (carrier_b & ~0x0f000000u) | (bank3 << 24);
    asm volatile("" : "+r"(carrier_b));

    bool r0 = ((carrier_a >> 0) & 0x1u) != 0u;
    bool r1 = ((carrier_a >> 9) & 0x1u) != 0u;
    bool r2 = ((carrier_b >> 18) & 0x1u) != 0u;
    bool r3 = ((carrier_b >> 27) & 0x1u) != 0u;
    bool r4 = ((carrier_b >> 24) & 0x8u) != 0u;

    uint32_t result = carrier_a ^ carrier_b;
    if (r0 && r1) result ^= 0x11110000u;
    if (r2 && !r3) result ^= 0x00002222u;
    if (r4) result ^= 0x00f000f0u;
    if (r0 && r2 && r4) result += 0x13579bdfu;

    out[i] = result;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_bank_lifecycle_7f(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;

    bool p0 = (in[i + 0 * 32] > 0);
    bool p1 = (in[i + 1 * 32] < 0);
    bool p2 = (in[i + 2 * 32] != 0);
    bool p3 = (in[i + 3 * 32] >= 3);
    bool p4 = (in[i + 4 * 32] <= 8);
    bool p5 = (in[i + 5 * 32] == -1);
    bool p6 = (in[i + 6 * 32] != 11);

    uint32_t carrier = seed_in[i];
    uint32_t bank7 = (uint32_t)p0
                   | ((uint32_t)p1 << 1)
                   | ((uint32_t)p2 << 2)
                   | ((uint32_t)p3 << 3)
                   | ((uint32_t)p4 << 4)
                   | ((uint32_t)p5 << 5)
                   | ((uint32_t)p6 << 6);
    carrier = (carrier & ~0x00007f00u) | (bank7 << 8);
    asm volatile("" : "+r"(carrier));

    uint32_t folded = ((carrier >> 8) & 0x7fu) ^ ((carrier >> 10) & 0x3fu);
    bool s0 = (folded & 0x01u) != 0u;
    bool s1 = (folded & 0x02u) != 0u;
    bool s2 = (folded & 0x04u) != 0u;
    bool s3 = (folded & 0x08u) != 0u;

    uint32_t result = carrier ^ 0x2468ace0u;
    if (s0) result += 0x10u;
    if (s1) result ^= 0x2000u;
    if (s2 && s3) result ^= 0x55005500u;
    if (s0 && s2) result += 0x12340000u;

    out[i] = result;
}
