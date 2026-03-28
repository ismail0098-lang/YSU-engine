/*
 * SASS RE Probe: compiler-driven predicate bank spill exact follow-up
 *
 * Goal: stop manually packing predicate bits and instead keep a large set of
 * compiler-managed predicates live across helper calls, divergence, and
 * reconvergence. This is aimed at the stubborn P2R.B1/B2/B3 frontier.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __noinline__ uint32_t
probe_predicate_mix_helper(uint32_t acc,
                           bool p0, bool p1, bool p2, bool p3,
                           bool p4, bool p5, bool p6, bool p7,
                           bool p8, bool p9, bool pA, bool pB) {
    if (p0 && p4) acc ^= 0x0000000fu;
    if (p1 || p5) acc += 0x000000f0u;
    if (p2 && !p6) acc ^= 0x00000f00u;
    if (p3 || p7) acc += 0x0000f000u;
    if (p8 && pA) acc ^= 0x000f0000u;
    if (p9 || pB) acc += 0x00f00000u;
    asm volatile("" : "+r"(acc));
    return acc;
}

extern "C" __global__ void __launch_bounds__(64)
probe_predicate_bank_compiler_spill_exact(uint32_t *out,
                                          const int *in,
                                          const uint32_t *seed,
                                          int iters) {
    int lane = threadIdx.x & 31;
    uint32_t acc = seed[lane] ^ 0x6b8b4567u;

    bool p0 = in[lane + 0 * 32] > 0;
    bool p1 = in[lane + 1 * 32] < 0;
    bool p2 = in[lane + 2 * 32] != 0;
    bool p3 = in[lane + 3 * 32] >= 3;
    bool p4 = in[lane + 4 * 32] <= 8;
    bool p5 = in[lane + 5 * 32] == -1;
    bool p6 = in[lane + 6 * 32] != 11;
    bool p7 = in[lane + 7 * 32] > -7;
    bool p8 = in[lane + 8 * 32] < 5;
    bool p9 = in[lane + 9 * 32] >= -2;
    bool pA = in[lane + 10 * 32] != 13;
    bool pB = in[lane + 11 * 32] == 0;

    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        bool q0 = p0 ^ (((iter + lane) & 1) != 0);
        bool q1 = p1 ^ (((iter + lane) & 2) != 0);
        bool q2 = p2 ^ (((iter + lane) & 4) != 0);
        bool q3 = p3 ^ (((iter + lane) & 8) != 0);
        bool q4 = p4 ^ (((iter + lane) & 1) == 0);
        bool q5 = p5 ^ (((iter + lane) & 2) == 0);
        bool q6 = p6 ^ (((iter + lane) & 4) == 0);
        bool q7 = p7 ^ (((iter + lane) & 8) == 0);
        bool q8 = p8 ^ ((iter & 1) != 0);
        bool q9 = p9 ^ ((iter & 2) != 0);
        bool qA = pA ^ ((iter & 4) != 0);
        bool qB = pB ^ ((iter & 8) != 0);

        if ((q0 && q3) || (q8 && qB)) {
            acc += (uint32_t)(in[lane + ((iter + 0) & 7) * 32] * 3 + iter);
        } else {
            acc ^= (uint32_t)(in[lane + ((iter + 4) & 7) * 32] * 5 - iter);
        }

        acc = probe_predicate_mix_helper(acc, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, qA, qB);

        if (q0 && q4 && q8) acc ^= 0x11111111u;
        if (q1 || q5 || q9) acc += 0x01010101u;
        if (q2 && !q6 && qA) acc ^= 0x22220000u;
        if (q3 || q7 || qB) acc += 0x00003333u;
        if ((q0 && q1) || (q8 && q9)) acc ^= (acc << 1);
        if ((q2 && q3) || (qA && qB)) acc += (acc >> 3);

        asm volatile("" : "+r"(acc) :: "memory");
    }

    out[threadIdx.x + blockIdx.x * blockDim.x] = acc;
}
