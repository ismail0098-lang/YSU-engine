/*
 * SASS RE Probe: PLOP3-fed P2R carrier follow-up
 *
 * Goal: attack the final P2R.B* frontier through predicate-source kind
 * rather than more carrier-shape perturbations. The nearby combo-family O3
 * kernels emit real LOP3.LUT P* / PLOP3.LUT chains, so this probe borrows
 * that style and feeds the resulting predicate bank into a live carrier.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ uint32_t
probe_plop3_mask4(bool p0, bool p1, bool p2, bool p3) {
    return (uint32_t)p0
         | ((uint32_t)p1 << 1)
         | ((uint32_t)p2 << 2)
         | ((uint32_t)p3 << 3);
}

static __device__ __forceinline__ uint32_t
probe_plop3_mask7(bool p0, bool p1, bool p2, bool p3, bool p4, bool p5, bool p6) {
    return (uint32_t)p0
         | ((uint32_t)p1 << 1)
         | ((uint32_t)p2 << 2)
         | ((uint32_t)p3 << 3)
         | ((uint32_t)p4 << 4)
         | ((uint32_t)p5 << 5)
         | ((uint32_t)p6 << 6);
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_plop3_tripack_exact(uint32_t *out,
                              int stage_mask,
                              int tail_mask,
                              int limit_mask,
                              const uint32_t *seed_in) {
    int lane = threadIdx.x;

    unsigned stage_bit = (static_cast<unsigned>(stage_mask) >> 6) & 1u;
    unsigned tail_bit = (static_cast<unsigned>(tail_mask) >> 7) & 1u;
    unsigned limit_bit = static_cast<unsigned>(limit_mask) & 1u;

    bool gate40 = (stage_mask & 0x40) != 0;
    bool gate80 = (stage_mask & 0x80) != 0;
    bool tail0 = (tail_mask & 0x40) != 0;
    bool tail1 = (tail_mask & 0x80) != 0;
    bool limit0 = (limit_mask & 0x1) != 0;
    bool stage_eq_1 = (stage_bit != 1u);
    bool tail_eq_1 = (tail_bit != 1u);
    bool limit_eq_1 = (limit_bit != 1u);

    bool p0 = (gate40 && !limit0) || (tail0 && gate80) || stage_eq_1;
    bool p1 = (gate80 && limit0) || (tail1 && gate40) || tail_eq_1;
    bool p2 = (gate40 && tail1) ^ (gate80 && tail0);
    bool p3 = (limit_eq_1 && !tail_eq_1) || (p0 && !p1);

    bool q0 = (p0 && p1) || p2;
    bool q1 = (p1 && !p3) || p0;
    bool q2 = (p2 ^ p3) || p1;
    bool q3 = (p3 && !p0) || p2;

    bool r0 = (q0 && !q1) || q2;
    bool r1 = (q1 && q3) || !q0;
    bool r2 = (q2 && !q3) || q1;
    bool r3 = (q3 || q0) && !q2;

    uint32_t carrier = seed_in[lane] ^ 0xe7e7e7e7u;
    uint32_t x = probe_plop3_mask4(p0, p1, p2, p3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | x;
    asm volatile("" : "+r"(carrier));

    uint32_t y = (probe_plop3_mask4(q0, q1, q2, q3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (y << 8);
    asm volatile("" : "+r"(carrier));

    uint32_t z = (probe_plop3_mask4(r0, r1, r2, r3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xff00ffffu) | (z << 16);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0fe7e7e7u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_plop3_samecarrier_r7style_exact(uint32_t *out,
                                          const uint32_t *seed_a,
                                          const uint32_t *seed_b) {
    int lane = threadIdx.x;

    uint32_t a0 = seed_a[lane];
    uint32_t a1 = seed_a[lane + 32];
    uint32_t a2 = seed_a[lane + 64];
    uint32_t a3 = seed_a[lane + 96];
    uint32_t b0 = seed_b[lane];
    uint32_t b1 = seed_b[lane + 32];
    uint32_t b2 = seed_b[lane + 64];

    bool p0 = ((a0 >> 3) & 1u) != 0;
    bool p1 = ((a1 >> 5) & 1u) != 0;
    bool p2 = ((a2 ^ b0) & 0x40u) != 0;
    bool p3 = ((a3 + b1) & 0x80u) != 0;
    bool p4 = (a0 < a2) != (b0 < b2);
    bool p5 = ((a1 | b1) & 0x20u) != 0;
    bool p6 = ((a3 ^ b2) & 0x10u) != 0;

    bool q0 = (p0 && !p1) || p2;
    bool q1 = (p1 && p3) || !p4;
    bool q2 = (p2 ^ p5) || p6;
    bool q3 = (p3 && !p0) || p4;
    bool q4 = (p4 && p6) || !p2;
    bool q5 = (p5 || p1) && !p3;
    bool q6 = (p6 && !p4) || p0;

    uint32_t pack = probe_plop3_mask7(q0, q1, q2, q3, q4, q5, q6) & 0x7fu;
    uint32_t carrier = seed_b[lane + 96] ^ 0x31415900u;
    carrier = (carrier & 0xffffff00u) | pack;
    asm volatile("" : "+r"(carrier));

    bool r0 = (q0 && q1) || q2;
    bool r1 = (q1 && !q3) || q4;
    bool r2 = (q2 ^ q5) || q6;
    bool r3 = (q3 && !q0) || q1;

    uint32_t hi = (probe_plop3_mask4(r0, r1, r2, r3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (hi << 8);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x31415900u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_plop3_selpack_r7style_exact(uint32_t *out,
                                      const uint32_t *seed_a,
                                      const uint32_t *seed_b) {
    int lane = threadIdx.x;

    uint32_t a0 = seed_a[lane];
    uint32_t a1 = seed_a[lane + 32];
    uint32_t a2 = seed_a[lane + 64];
    uint32_t a3 = seed_a[lane + 96];
    uint32_t b0 = seed_b[lane];
    uint32_t b1 = seed_b[lane + 32];
    uint32_t b2 = seed_b[lane + 64];
    uint32_t carrier = seed_b[lane + 96] ^ 0x31415900u;

    bool p0 = ((a0 >> 3) & 1u) != 0;
    bool p1 = ((a1 >> 5) & 1u) != 0;
    bool p2 = ((a2 ^ b0) & 0x40u) != 0;
    bool p3 = ((a3 + b1) & 0x80u) != 0;
    bool p4 = (a0 < a2) != (b0 < b2);
    bool p5 = ((a1 | b1) & 0x20u) != 0;
    bool p6 = ((a3 ^ b2) & 0x10u) != 0;

    bool q0 = (p0 && !p1) || p2;
    bool q1 = (p1 && p3) || !p4;
    bool q2 = (p2 ^ p5) || p6;
    bool q3 = (p3 && !p0) || p4;
    bool q4 = (p4 && p6) || !p2;
    bool q5 = (p5 || p1) && !p3;
    bool q6 = (p6 && !p4) || p0;

    uint32_t x0 = q0 ? 0x01u : 0u;
    uint32_t x1 = q1 ? 0x02u : 0u;
    uint32_t x2 = q2 ? 0x04u : 0u;
    uint32_t x3 = q3 ? 0x08u : 0u;
    uint32_t x4 = q4 ? 0x10u : 0u;
    uint32_t x5 = q5 ? 0x20u : 0u;
    uint32_t x6 = q6 ? 0x40u : 0u;
    uint32_t pack = x0 + x1 + x2 + x3 + x4 + x5 + x6;
    carrier = (carrier & 0xffffff00u) | pack;
    asm volatile("" : "+r"(carrier));

    bool r0 = (q0 && q1) || q2;
    bool r1 = (q1 && !q3) || q4;
    bool r2 = (q2 ^ q5) || q6;
    bool r3 = (q3 && !q0) || q1;

    uint32_t y0 = r0 ? 0x01u : 0u;
    uint32_t y1 = r1 ? 0x02u : 0u;
    uint32_t y2 = r2 ? 0x04u : 0u;
    uint32_t y3 = r3 ? 0x08u : 0u;
    uint32_t high = (y0 + y1 + y2 + y3) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (high << 8);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x31415900u;
}
