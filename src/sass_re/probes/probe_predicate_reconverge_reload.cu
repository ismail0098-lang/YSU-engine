/*
 * SASS RE Probe: predicate reconvergence + reload lifecycle
 *
 * Goal: preserve a packed predicate carrier across divergence, reconvergence,
 * and a noinline helper boundary. The missing P2R.B1/B2/B3 and R2P spellings
 * look like banked predicate save / restore operations, so this probe makes
 * the compiler carry many predicate bits through the exact kind of lifecycle
 * where a packed reload would be profitable.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __noinline__ uint32_t
probe_predicate_reload_helper(uint32_t carrier,
                              int x0, int x1, int x2, int x3,
                              int x4, int x5, int x6, int x7) {
    bool q0 = ((carrier >> 0) & 1u) != 0u;
    bool q1 = ((carrier >> 1) & 1u) != 0u;
    bool q2 = ((carrier >> 8) & 1u) != 0u;
    bool q3 = ((carrier >> 9) & 1u) != 0u;
    bool q4 = ((carrier >> 16) & 1u) != 0u;
    bool q5 = ((carrier >> 17) & 1u) != 0u;
    bool q6 = ((carrier >> 24) & 1u) != 0u;
    bool q7 = ((carrier >> 25) & 1u) != 0u;

    uint32_t acc = carrier ^ 0x5a5a3c3cu;
    if (q0 && q4) acc += (uint32_t)(x0 ^ x4);
    if (q1 || q5) acc ^= (uint32_t)(x1 + x5);
    if (q2 && !q6) acc += (uint32_t)(x2 - x6);
    if (q3 || q7) acc ^= (uint32_t)(x3 * 3 + x7);
    if ((q0 && q1) || (q2 && q3)) acc += 0x00010001u;
    if ((q4 && q5) || (q6 && q7)) acc ^= 0x00ff00ffu;
    asm volatile("" : "+r"(acc));
    return acc;
}

extern "C" __global__ void __launch_bounds__(32)
probe_predicate_reconverge_reload(uint32_t *out,
                                  const int *in,
                                  const uint32_t *seed_in,
                                  int iters) {
    int lane = threadIdx.x;

    bool p0 = (in[lane + 0 * 32] > 0);
    bool p1 = (in[lane + 1 * 32] < 0);
    bool p2 = (in[lane + 2 * 32] != 0);
    bool p3 = (in[lane + 3 * 32] >= 7);
    bool p4 = (in[lane + 4 * 32] > -2);
    bool p5 = (in[lane + 5 * 32] <= 3);
    bool p6 = (in[lane + 6 * 32] == 9);
    bool p7 = (in[lane + 7 * 32] != -5);
    bool p8 = (in[lane + 8 * 32] > 1);
    bool p9 = (in[lane + 9 * 32] < 4);
    bool pA = (in[lane + 10 * 32] != 2);
    bool pB = (in[lane + 11 * 32] >= 6);

    uint32_t carrier = seed_in[lane];
    carrier ^= ((uint32_t)p0 << 0) | ((uint32_t)p1 << 1) | ((uint32_t)p2 << 2) | ((uint32_t)p3 << 3);
    carrier ^= (((uint32_t)p4 << 0) | ((uint32_t)p5 << 1) | ((uint32_t)p6 << 2) | ((uint32_t)p7 << 3)) << 8;
    carrier ^= (((uint32_t)p8 << 0) | ((uint32_t)p9 << 1) | ((uint32_t)pA << 2) | ((uint32_t)pB << 3)) << 16;
    asm volatile("" : "+r"(carrier));

    uint32_t acc = carrier ^ 0x9e3779b9u;

    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        bool left = ((lane + iter) & 1) == 0;
        if (left) {
            acc += (uint32_t)(in[lane + (iter & 7) * 32] * 13 + iter);
            carrier ^= 0x00010001u + (uint32_t)iter;
        } else {
            acc ^= (uint32_t)(in[lane + ((iter + 4) & 7) * 32] * 17 - iter);
            carrier += 0x01000100u ^ (uint32_t)(iter << 4);
        }

        asm volatile("" : "+r"(carrier), "+r"(acc));

        uint32_t helper = probe_predicate_reload_helper(
            carrier,
            in[lane + 0 * 32], in[lane + 1 * 32], in[lane + 2 * 32], in[lane + 3 * 32],
            in[lane + 4 * 32], in[lane + 5 * 32], in[lane + 6 * 32], in[lane + 7 * 32]);

        bool r0 = ((carrier >> 0) & 1u) != 0u;
        bool r1 = ((carrier >> 8) & 1u) != 0u;
        bool r2 = ((carrier >> 16) & 1u) != 0u;
        bool r3 = ((carrier >> 24) & 1u) != 0u;

        if (r0 && r2) acc += helper ^ 0x13579bdfu;
        if (r1 || r3) acc ^= helper + 0x2468ace0u;
        if (r0 && r1 && r2) acc += (helper >> 3);
        if (r3 && !r0) acc ^= (helper << 1);
    }

    out[lane] = acc ^ carrier;
}
