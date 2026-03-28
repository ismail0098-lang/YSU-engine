/*
 * SASS RE Probe: same-carrier byteview predicate packing
 *
 * Goal: mirror the cuDNN-local P2R neighborhood more literally than the older
 * probes by repeatedly updating byte lanes of the same live carrier register
 * with a 7-bit predicate mask. The library-mined shape we care about is:
 *   P2R Rn, PR, Rn, 0x7f
 *   P2R.B1 Rn, PR, Rn, 0x7f
 * We already reproduce the base P2R neighborhood; this probe tries to make
 * the compiler see the byte-lane updates as byte writes to the same carrier.
 */

#include <cuda_runtime.h>
#include <stdint.h>

union probe_p2r_byteview_u32 {
    uint32_t w;
    uchar4 b;
};

static __device__ __forceinline__ uint32_t
probe_p2r_mask7(bool p0, bool p1, bool p2, bool p3, bool p4, bool p5, bool p6) {
    return (uint32_t)p0
         | ((uint32_t)p1 << 1)
         | ((uint32_t)p2 << 2)
         | ((uint32_t)p3 << 3)
         | ((uint32_t)p4 << 4)
         | ((uint32_t)p5 << 5)
         | ((uint32_t)p6 << 6);
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_byteview_carrier_exact(uint32_t *out,
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

    bool q0 = (in[lane + 7 * 32] > -7);
    bool q1 = (in[lane + 8 * 32] < 5);
    bool q2 = (in[lane + 9 * 32] >= -2);
    bool q3 = (in[lane + 10 * 32] != 13);
    bool q4 = (in[lane + 11 * 32] == 0);
    bool q5 = (in[lane + 12 * 32] > 1);
    bool q6 = (in[lane + 13 * 32] < 9);

    bool r0 = (in[lane + 14 * 32] >= -3);
    bool r1 = (in[lane + 15 * 32] != 4);
    bool r2 = (in[lane + 16 * 32] > -8);
    bool r3 = (in[lane + 17 * 32] < 12);
    bool r4 = (in[lane + 18 * 32] == 2);
    bool r5 = (in[lane + 19 * 32] != -6);
    bool r6 = (in[lane + 20 * 32] >= 7);

    probe_p2r_byteview_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x5a5aa5a5u;

    carrier.b.y = (unsigned char)probe_p2r_mask7(p0, p1, p2, p3, p4, p5, p6);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.z = (unsigned char)probe_p2r_mask7(q0, q1, q2, q3, q4, q5, q6);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.w = (unsigned char)probe_p2r_mask7(r0, r1, r2, r3, r4, r5, r6);
    asm volatile("" : "+r"(carrier.w));

    uint32_t acc = carrier.w ^ 0x13579bdfu;

    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        bool s0 = ((carrier.b.y >> (iter & 6)) & 1u) != 0u;
        bool s1 = ((carrier.b.z >> ((iter + 1) & 6)) & 1u) != 0u;
        bool s2 = ((carrier.b.w >> ((iter + 2) & 6)) & 1u) != 0u;

        if (s0) acc += (uint32_t)(in[lane + ((iter + 0) & 7) * 32] * 3 + iter);
        if (s1) acc ^= (uint32_t)(in[lane + ((iter + 2) & 7) * 32] * 5 - iter);
        if (s2) acc += (uint32_t)(in[lane + ((iter + 4) & 7) * 32] * 7 + iter);

        carrier.b.y ^= (unsigned char)((iter << 1) & 0x7f);
        carrier.b.z ^= (unsigned char)((iter << 2) & 0x7f);
        carrier.b.w ^= (unsigned char)((iter << 3) & 0x7f);
        asm volatile("" : "+r"(carrier.w), "+r"(acc) :: "memory");
    }

    out[lane] = acc ^ carrier.w;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_byteview_b1_topbit_exact(uint32_t *out,
                                   const int *in,
                                   const uint32_t *seed_in) {
    int lane = threadIdx.x;
    bool p0 = (in[lane + 0 * 32] > 0);
    bool p1 = (in[lane + 1 * 32] < 0);
    bool p2 = (in[lane + 2 * 32] != 0);
    bool p3 = (in[lane + 3 * 32] >= 3);
    bool p4 = (in[lane + 4 * 32] <= 8);
    bool p5 = (in[lane + 5 * 32] == -1);
    bool p6 = (in[lane + 6 * 32] != 11);

    probe_p2r_byteview_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x2468ace0u;
    carrier.b.y = 0x80u;
    asm volatile("" : "+r"(carrier.w));
    carrier.b.y = (unsigned char)(carrier.b.y | (unsigned char)probe_p2r_mask7(p0, p1, p2, p3, p4, p5, p6));
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x13579bdfu;
}
