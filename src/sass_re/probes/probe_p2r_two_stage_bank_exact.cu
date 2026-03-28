/*
 * SASS RE Probe: two-stage predicate-bank carrier update
 *
 * Goal: mimic the library-mined neighborhood more closely:
 *   1. build a 7-bit predicate bank into one live carrier
 *   2. seed a second live carrier with byte1 = 0x80
 *   3. merge the first carrier into the second
 *   4. build a fresh 7-predicate bank and rewrite byte1 of that same carrier
 *
 * The target raw spellings are the stubborn byte-qualified forms:
 *   P2R.B1 / P2R.B2 / P2R.B3
 */

#include <cuda_runtime.h>
#include <stdint.h>

union probe_p2r_twostage_u32 {
    uint32_t w;
    uchar4 b;
};

static __device__ __forceinline__ uint32_t
probe_mask7(bool p0, bool p1, bool p2, bool p3, bool p4, bool p5, bool p6) {
    return (uint32_t)p0
         | ((uint32_t)p1 << 1)
         | ((uint32_t)p2 << 2)
         | ((uint32_t)p3 << 3)
         | ((uint32_t)p4 << 4)
         | ((uint32_t)p5 << 5)
         | ((uint32_t)p6 << 6);
}

static __device__ __forceinline__ uint32_t
probe_mask4(bool p0, bool p1, bool p2, bool p3) {
    return (uint32_t)p0
         | ((uint32_t)p1 << 1)
         | ((uint32_t)p2 << 2)
         | ((uint32_t)p3 << 3);
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_two_stage_bank_exact(uint32_t *out,
                               const int *in,
                               const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] > 0);
    bool a1 = (in[lane + 1 * 32] < 0);
    bool a2 = (in[lane + 2 * 32] != 0);
    bool a3 = (in[lane + 3 * 32] >= 3);
    bool a4 = (in[lane + 4 * 32] <= 8);
    bool a5 = (in[lane + 5 * 32] == -1);
    bool a6 = (in[lane + 6 * 32] != 11);

    probe_p2r_twostage_u32 carrier_a;
    carrier_a.w = seed_in[lane] ^ 0x07fffffeu;
    carrier_a.b.x = (unsigned char)probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier_a.w));

    int v0 = in[lane + 7 * 32] + 0x4;
    int v1 = in[lane + 8 * 32] + 0x8;
    int v2 = in[lane + 9 * 32] + 0xC;
    int lo0 = (v0 * 0x79) >> 8;
    int lo1 = (v1 * 0x79) >> 8;
    bool b3 = (v0 >= 0x5e);
    bool b4 = (v1 >= 0x5e);
    bool b5 = (v2 >= 0x5e);

    probe_p2r_twostage_u32 carrier_b;
    carrier_b.w = seed_in[lane] ^ 0x2468ace0u;
    carrier_b.b.y = (unsigned char)(b4 ? 0x00u : 0x80u);
    carrier_b.b.z = (unsigned char)(b3 ? 0x00u : 0x80u);
    asm volatile("" : "+r"(carrier_b.w));

    carrier_b.w = (carrier_b.w & ~0x000000ffu) | (carrier_a.w & 0x000000ffu);
    asm volatile("" : "+r"(carrier_b.w));

    bool c0 = (lo1 < 0x58) && (!b4);
    bool c1 = (lo1 < 0x58) && (!b5);
    bool c2 = (lo1 < 0x58) && (!b3);
    bool c3 = (lo1 < 0x58) && (!c0);
    bool c4 = (lo0 < 0x58) && (!c1);
    bool c5 = (lo0 < 0x58) && (!c2);
    bool c6 = (lo0 < 0x58) && (!c3);

    carrier_b.b.y = (unsigned char)probe_mask7(c0, c1, c2, c3, c4, c5, c6);
    asm volatile("" : "+r"(carrier_b.w));

    out[lane] = carrier_b.w ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_two_stage_same_carrier_b1_exact(uint32_t *out,
                                          const int *in,
                                          const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] > 0);
    bool a1 = (in[lane + 1 * 32] < 0);
    bool a2 = (in[lane + 2 * 32] != 0);
    bool a3 = (in[lane + 3 * 32] >= 3);
    bool a4 = (in[lane + 4 * 32] <= 8);
    bool a5 = (in[lane + 5 * 32] == -1);
    bool a6 = (in[lane + 6 * 32] != 11);

    bool b0 = (in[lane + 7 * 32] > -7);
    bool b1 = (in[lane + 8 * 32] < 5);
    bool b2 = (in[lane + 9 * 32] >= -2);
    bool b3 = (in[lane + 10 * 32] != 13);
    bool b4 = (in[lane + 11 * 32] == 0);
    bool b5 = (in[lane + 12 * 32] > 1);
    bool b6 = (in[lane + 13 * 32] < 9);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x07fffffeu;
    carrier.b.x = (unsigned char)probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.y = 0x80u;
    asm volatile("" : "+r"(carrier.w));
    carrier.b.y = (unsigned char)(carrier.b.y | (unsigned char)probe_mask7(b0, b1, b2, b3, b4, b5, b6));
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x2468ace0u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_byte1_cudnn_shape_exact(uint32_t *out,
                                  const int *in,
                                  const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x56);
    bool a1 = (in[lane + 1 * 32] < 0x56);
    bool a2 = (in[lane + 2 * 32] < 0x56);
    bool a3 = (in[lane + 3 * 32] < 0x56);
    bool a4 = (in[lane + 4 * 32] < 0x56);
    bool a5 = (in[lane + 5 * 32] < 0x56);
    bool a6 = (in[lane + 6 * 32] < 0x56);

    probe_p2r_twostage_u32 bank0;
    bank0.w = seed_in[lane] ^ 0x31415926u;
    bank0.b.x = (unsigned char)probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(bank0.w));

    int v0 = in[lane + 7 * 32];
    int v1 = in[lane + 8 * 32];
    int v2 = in[lane + 9 * 32];
    int v3 = in[lane + 10 * 32];
    int v4 = in[lane + 11 * 32];
    int v5 = in[lane + 12 * 32];
    int v6 = in[lane + 13 * 32];

    bool b0 = (v0 >= 0x56) && (v0 < 0x80);
    bool b1 = (v1 >= 0x56) && (v1 < 0x80);
    bool b2 = (v2 >= 0x56) && (v2 < 0x80);
    bool b3 = (v3 >= 0x56) && (v3 < 0x80);
    bool b4 = (v4 >= 0x56) && (v4 < 0x80);
    bool b5 = (v5 >= 0x56) && (v5 < 0x80);
    bool b6 = (v6 >= 0x56) && (v6 < 0x80);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x2468ace0u;
    carrier.b.x = bank0.b.x;
    carrier.b.y = 0x80u;
    asm volatile("" : "+r"(carrier.w));

    unsigned char bank1 = (unsigned char)probe_mask7(b0, b1, b2, b3, b4, b5, b6);
    carrier.b.y = (unsigned char)((carrier.b.y & 0x80u) | bank1);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ bank0.w ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_byte1_halfword_exact(uint32_t *out,
                               const int *in,
                               const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x56);
    bool a1 = (in[lane + 1 * 32] < 0x56);
    bool a2 = (in[lane + 2 * 32] < 0x56);
    bool a3 = (in[lane + 3 * 32] < 0x56);
    bool a4 = (in[lane + 4 * 32] < 0x56);
    bool a5 = (in[lane + 5 * 32] < 0x56);
    bool a6 = (in[lane + 6 * 32] < 0x56);

    bool b0 = (in[lane + 7 * 32] >= 0x56) && (in[lane + 7 * 32] < 0x80);
    bool b1 = (in[lane + 8 * 32] >= 0x56) && (in[lane + 8 * 32] < 0x80);
    bool b2 = (in[lane + 9 * 32] >= 0x56) && (in[lane + 9 * 32] < 0x80);
    bool b3 = (in[lane + 10 * 32] >= 0x56) && (in[lane + 10 * 32] < 0x80);
    bool b4 = (in[lane + 11 * 32] >= 0x56) && (in[lane + 11 * 32] < 0x80);
    bool b5 = (in[lane + 12 * 32] >= 0x56) && (in[lane + 12 * 32] < 0x80);
    bool b6 = (in[lane + 13 * 32] >= 0x56) && (in[lane + 13 * 32] < 0x80);

    uint32_t carrier = seed_in[lane] ^ 0x55aa3300u;
    uint32_t bank0 = probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    carrier = (carrier & 0xffffff00u) | bank0;
    asm volatile("" : "+r"(carrier));

    uint32_t bank1 = probe_mask7(b0, b1, b2, b3, b4, b5, b6) | 0x80u;
    uint32_t lo16 = (carrier & 0x000000ffu) | (bank1 << 8);
    carrier = (carrier & 0xffff0000u) | lo16;
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x5a5a3c3cu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_literal_cudnn_exact(uint32_t *out,
                                 const int *in,
                                 const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r8 = in[lane + 3 * 32];
    int r9 = in[lane + 4 * 32];
    int r13 = in[lane + 5 * 32];

    bool a0 = (r6 < 0x58);
    bool a1 = (r2 < 0x58);
    bool a2 = (r2 < 0x58);
    bool a3 = (r2 < 0x58);
    bool a4 = (r4 < 0x58);
    bool a5 = (r4 < 0x58);
    bool a6 = (r4 < 0x58);

    uint32_t carrier_a = seed_in[lane] ^ 0x00ff00ffu;
    carrier_a = (carrier_a & 0xffffff00u) | probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier_a));

    bool p1 = (r2 < 0x58);
    uint32_t carrier_b = seed_in[lane] ^ 0x2468ace0u;
    carrier_b = (carrier_b & 0xffffff00u) | (p1 ? 0x00u : 0x80u);
    asm volatile("" : "+r"(carrier_b));
    carrier_b = (carrier_b & 0xffffff00u) | (carrier_a & 0xffu);
    asm volatile("" : "+r"(carrier_b));

    bool b0 = (r6 < 0x58);
    bool b1 = (r6 < 0x58);
    bool b2 = (r6 < 0x58);
    bool b3 = (r6 < 0x58);
    bool b4 = (r4 < 0x58);
    bool b5 = (r4 < 0x58);
    bool b6 = (r4 < 0x58);

    carrier_b = (carrier_b & 0xffff00ffu) | ((uint32_t)probe_mask7(b0, b1, b2, b3, b4, b5, b6) << 8);
    asm volatile("" : "+r"(carrier_b));

    out[lane] = carrier_b ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_literal_cudnn_exact(uint32_t *out,
                                 const int *in,
                                 const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool a4 = (in[lane + 4 * 32] < 0x58);
    bool a5 = (in[lane + 5 * 32] < 0x58);
    bool a6 = (in[lane + 6 * 32] < 0x58);

    bool b0 = (in[lane + 7 * 32] >= 0x58);
    bool b1 = (in[lane + 8 * 32] >= 0x58);
    bool b2 = (in[lane + 9 * 32] >= 0x58);
    bool b3 = (in[lane + 10 * 32] >= 0x58);
    bool b4 = (in[lane + 11 * 32] >= 0x58);
    bool b5 = (in[lane + 12 * 32] >= 0x58);
    bool b6 = (in[lane + 13 * 32] >= 0x58);

    uint32_t carrier = seed_in[lane] ^ 0x89abcdefu;
    carrier = (carrier & 0xffffff00u) | probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier));
    carrier = (carrier & 0xff00ffffu) | ((uint32_t)0x80u << 16);
    asm volatile("" : "+r"(carrier));
    carrier = (carrier & 0xff00ffffu) |
              ((uint32_t)(probe_mask7(b0, b1, b2, b3, b4, b5, b6) | 0x80u) << 16);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x2468ace0u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_literal_cudnn_exact(uint32_t *out,
                                 const int *in,
                                 const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool a4 = (in[lane + 4 * 32] < 0x58);
    bool a5 = (in[lane + 5 * 32] < 0x58);
    bool a6 = (in[lane + 6 * 32] < 0x58);

    bool b0 = (in[lane + 7 * 32] >= 0x58);
    bool b1 = (in[lane + 8 * 32] >= 0x58);
    bool b2 = (in[lane + 9 * 32] >= 0x58);
    bool b3 = (in[lane + 10 * 32] >= 0x58);
    bool b4 = (in[lane + 11 * 32] >= 0x58);
    bool b5 = (in[lane + 12 * 32] >= 0x58);
    bool b6 = (in[lane + 13 * 32] >= 0x58);

    uint32_t carrier = seed_in[lane] ^ 0x76543210u;
    carrier = (carrier & 0xffffff00u) | probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier));
    carrier = (carrier & 0x00ffffffu) | ((uint32_t)0x80u << 24);
    asm volatile("" : "+r"(carrier));
    carrier = (carrier & 0x00ffffffu) |
              ((uint32_t)(probe_mask7(b0, b1, b2, b3, b4, b5, b6) | 0x80u) << 24);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x5a5a3c3cu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_split_seed_exact(uint32_t *out,
                              const int *in,
                              const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r12 = in[lane + 3 * 32];
    int r15 = in[lane + 4 * 32];

    bool gate0 = (r2 >= 0x5e);
    bool gate1 = (r4 >= 0x5e);

    uint32_t carrier7 = seed_in[lane] ^ 0x2468ace0u;
    uint32_t carrier6 = seed_in[lane] ^ 0x13579bdfu;

    carrier7 = (carrier7 & 0xffffff00u) | (gate1 ? 0x00u : 0x80u);
    asm volatile("" : "+r"(carrier7));
    carrier6 = (carrier6 & 0xffff00ffu) | ((uint32_t)(gate0 ? 0x0000u : 0x8000u));
    asm volatile("" : "+r"(carrier6));

    bool p0 = (r15 < 0x58) && (!gate1);
    bool p1 = (r15 < 0x58) && (!gate1);
    bool p2 = (r15 < 0x58) && (!gate1);
    bool p3 = (r15 < 0x58) && (!((r15 < 0x58) && (!gate0)));
    bool p4 = (r12 < 0x58) && (!gate1);
    bool p5 = (r12 < 0x58) && (!gate1);
    bool p6 = (r12 < 0x58) && (!gate1);

    uint32_t low_bank = probe_mask7(p0, p1, p2, p3, p4, p5, p6);
    carrier7 = (carrier7 & 0xffffff00u) | low_bank;
    asm volatile("" : "+r"(carrier7));

    uint32_t merged = (carrier7 & ~0x00008000u) | (carrier6 & 0x00008000u);
    asm volatile("" : "+r"(merged));

    out[lane] = merged ^ 0x3c3c5a5au;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_secondbank_halfword_exact(uint32_t *out,
                                       const int *in,
                                       const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool a4 = (in[lane + 4 * 32] < 0x58);
    bool a5 = (in[lane + 5 * 32] < 0x58);
    bool a6 = (in[lane + 6 * 32] < 0x58);

    uint32_t carrier = seed_in[lane] ^ 0x6c6c9292u;
    carrier = (carrier & 0xffffff00u) | probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier));
    carrier = (carrier & 0xffff00ffu) | ((uint32_t)0x80u << 8);
    asm volatile("" : "+r"(carrier));

    int h0 = in[lane + 7 * 32] << 8;
    int h1 = in[lane + 8 * 32] << 8;
    int h2 = in[lane + 9 * 32] << 8;
    int h3 = in[lane + 10 * 32] << 8;
    int h4 = in[lane + 11 * 32] << 8;
    int h5 = in[lane + 12 * 32] << 8;
    int h6 = in[lane + 13 * 32] << 8;

    bool b0 = (h0 >= 0x5800);
    bool b1 = (h1 >= 0x5800);
    bool b2 = (h2 >= 0x5800);
    bool b3 = (h3 >= 0x5800);
    bool b4 = (h4 >= 0x5800);
    bool b5 = (h5 >= 0x5800);
    bool b6 = (h6 >= 0x5800);

    carrier = (carrier & 0xffff00ffu) |
              ((uint32_t)(probe_mask7(b0, b1, b2, b3, b4, b5, b6) | 0x80u) << 8);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x1d1d4b4bu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_split_seed_exact(uint32_t *out,
                              const int *in,
                              const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r12 = in[lane + 3 * 32];
    int r15 = in[lane + 4 * 32];

    bool gate0 = (r2 >= 0x5e);
    bool gate1 = (r4 >= 0x5e);

    uint32_t carrier7 = seed_in[lane] ^ 0x2468ace0u;
    uint32_t carrier5 = seed_in[lane] ^ 0x5a5a3c3cu;

    carrier7 = (carrier7 & 0xffffff00u) | (gate1 ? 0x00u : 0x80u);
    asm volatile("" : "+r"(carrier7));
    carrier5 = (carrier5 & 0x00ffffffu) | ((uint32_t)(gate0 ? 0x00000000u : 0x800000u));
    asm volatile("" : "+r"(carrier5));

    bool p0 = (r15 < 0x58) && (!gate1);
    bool p1 = (r15 < 0x58) && (!gate1);
    bool p2 = (r15 < 0x58) && (!gate1);
    bool p3 = (r15 < 0x58) && (!((r15 < 0x58) && (!gate0)));
    bool p4 = (r12 < 0x58) && (!gate1);
    bool p5 = (r12 < 0x58) && (!gate1);
    bool p6 = (r12 < 0x58) && (!gate1);

    uint32_t low_bank = probe_mask7(p0, p1, p2, p3, p4, p5, p6);
    carrier7 = (carrier7 & 0xffffff00u) | low_bank;
    asm volatile("" : "+r"(carrier7));

    uint32_t merged = (carrier7 & ~0x00800000u) | (carrier5 & 0x00800000u);
    asm volatile("" : "+r"(merged));

    out[lane] = merged ^ 0x7a7a1c1cu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_split_seed_exact(uint32_t *out,
                              const int *in,
                              const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r12 = in[lane + 3 * 32];
    int r15 = in[lane + 4 * 32];

    bool gate0 = (r2 >= 0x5e);
    bool gate1 = (r4 >= 0x5e);

    uint32_t carrier7 = seed_in[lane] ^ 0x2468ace0u;
    uint32_t carrier4 = seed_in[lane] ^ 0x13579bdfu;

    carrier7 = (carrier7 & 0xffffff00u) | (gate1 ? 0x00u : 0x80u);
    asm volatile("" : "+r"(carrier7));
    carrier4 = (carrier4 & 0x00ffffffu) | ((uint32_t)(gate0 ? 0x00000000u : 0x80000000u));
    asm volatile("" : "+r"(carrier4));

    bool p0 = (r15 < 0x58) && (!gate1);
    bool p1 = (r15 < 0x58) && (!gate1);
    bool p2 = (r15 < 0x58) && (!gate1);
    bool p3 = (r15 < 0x58) && (!((r15 < 0x58) && (!gate0)));
    bool p4 = (r12 < 0x58) && (!gate1);
    bool p5 = (r12 < 0x58) && (!gate1);
    bool p6 = (r12 < 0x58) && (!gate1);

    uint32_t low_bank = probe_mask7(p0, p1, p2, p3, p4, p5, p6);
    carrier7 = (carrier7 & 0xffffff00u) | low_bank;
    asm volatile("" : "+r"(carrier7));

    uint32_t merged = (carrier7 & ~0x80000000u) | (carrier4 & 0x80000000u);
    asm volatile("" : "+r"(merged));

    out[lane] = merged ^ 0x2c2c4e4eu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_samecarrier_late4_exact(uint32_t *out,
                                     const int *in,
                                     const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r8 = in[lane + 3 * 32];
    int r10 = in[lane + 4 * 32];
    int r12 = in[lane + 5 * 32];
    int r13 = in[lane + 6 * 32];

    bool p0 = (r6 < 0x58);
    bool p1 = (r2 < 0x58);
    bool p2 = (r2 < 0x58);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x2468ace0u;
    carrier.b.x = (unsigned char)probe_mask7(p0, p1, p2, false, false, false, false);
    asm volatile("" : "+r"(carrier.w));

    uint32_t blend = ((uint32_t)r8 << 8) ^ ((uint32_t)r10 & 0x0000ff00u);
    carrier.w = (carrier.w ^ blend) & 0xffff7fffu;
    asm volatile("" : "+r"(carrier.w));

    bool p3 = (r2 < 0x58);
    bool p4 = (r4 < 0x58);
    bool p5 = (r4 < 0x58);
    bool p6 = (r4 < 0x58);

    carrier.b.y = (unsigned char)probe_mask7(p0, p1, p2, p3, p4, p5, p6);
    asm volatile("" : "+r"(carrier.w));

    carrier.w ^= ((uint32_t)(r12 < 0x58) << 24) ^ ((uint32_t)(r13 < 0x58) << 16);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_samecarrier_r7style_exact(uint32_t *out,
                                       const int *in,
                                       const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r2 = in[lane + 0 * 32];
    int r4 = in[lane + 1 * 32];
    int r6 = in[lane + 2 * 32];
    int r10 = in[lane + 3 * 32];
    int r12 = in[lane + 4 * 32];
    int r13 = in[lane + 5 * 32];

    bool a0 = (r10 >= 0);
    bool a1 = (r12 != 0);
    bool a2 = (r13 > 1);
    bool a3 = (r6 < 0x58);
    bool a4 = (r2 < 0x58);
    bool a5 = (r2 < 0x58);
    bool a6 = (r2 < 0x58);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x31415926u;
    carrier.b.x = (unsigned char)probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier.w));

    carrier.w = (carrier.w | ((uint32_t)r10 & 0x0000ff00u)) ^ 0x00001000u;
    asm volatile("" : "+r"(carrier.w));

    bool b3 = (r6 < 0x58);
    bool b4 = (r4 < 0x58);
    bool b5 = (r4 < 0x58);
    bool b6 = (r4 < 0x58);

    carrier.b.y = (unsigned char)probe_mask7(a0, a1, a2, b3, b4, b5, b6);
    asm volatile("" : "+r"(carrier.w));

    carrier.w = (carrier.w & 0xffff00ffu) | (((uint32_t)carrier.b.y) << 8);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x5a5a3c3cu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_dualpack_transition_exact(uint32_t *out,
                                       const int *in,
                                       const uint32_t *seed_in) {
    int lane = threadIdx.x;

    int r0 = in[lane + 0 * 32];
    int r6 = in[lane + 1 * 32];
    int r7 = in[lane + 2 * 32];
    int r8 = in[lane + 3 * 32];
    int r9 = in[lane + 4 * 32];
    int r10 = in[lane + 5 * 32];
    int r12 = in[lane + 6 * 32];
    int r13 = in[lane + 7 * 32];
    int r14 = in[lane + 8 * 32];

    bool a0 = (r10 > 0);
    bool a1 = (r6 != 0);
    bool a2 = (r7 > 2);
    bool a3 = (r8 >= 9);
    bool a4 = (r9 != -1);
    bool a5 = (r12 > 0x55);
    bool a6 = (r13 < 0x56);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x2468ace0u;
    carrier.b.x = (unsigned char)probe_mask7(a0, a1, a2, a3, a4, a5, a6);
    asm volatile("" : "+r"(carrier.w));

    carrier.w = (carrier.w & 0xffffff00u) ^ (((uint32_t)r14 << 8) & 0x0000ff00u);
    asm volatile("" : "+r"(carrier.w));

    bool b0 = (r6 < 0x58);
    bool b1 = (r0 < 0x58);
    bool b2 = (r0 < 0x58);
    bool b3 = (r0 < 0x58);
    bool b4 = (r7 < 0x58);
    bool b5 = (r7 < 0x58);
    bool b6 = (r7 < 0x58);

    carrier.b.y = (unsigned char)probe_mask7(b0, b1, b2, b3, b4, b5, b6);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_nibble_exact(uint32_t *out,
                          const int *in,
                          const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool p0 = (in[lane + 0 * 32] < 0x58);
    bool p1 = (in[lane + 1 * 32] < 0x58);
    bool p2 = (in[lane + 2 * 32] < 0x58);
    bool p3 = (in[lane + 3 * 32] < 0x58);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x4a4a0000u;
    carrier.b.y = (unsigned char)probe_mask4(p0, p1, p2, p3);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0f0f4a4au;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_nibble_exact(uint32_t *out,
                          const int *in,
                          const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool p0 = (in[lane + 4 * 32] >= 0x58);
    bool p1 = (in[lane + 5 * 32] >= 0x58);
    bool p2 = (in[lane + 6 * 32] >= 0x58);
    bool p3 = (in[lane + 7 * 32] >= 0x58);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0xaeae0000u;
    carrier.b.z = (unsigned char)probe_mask4(p0, p1, p2, p3);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0f0faeaeu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_nibble_exact(uint32_t *out,
                          const int *in,
                          const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool p0 = (in[lane + 8 * 32] != 0);
    bool p1 = (in[lane + 9 * 32] != 0);
    bool p2 = (in[lane + 10 * 32] != 0);
    bool p3 = (in[lane + 11 * 32] != 0);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0xc3c30000u;
    carrier.b.w = (unsigned char)probe_mask4(p0, p1, p2, p3);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0f0fc3c3u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_nibble_transition_exact(uint32_t *out,
                                     const int *in,
                                     const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool b0 = (in[lane + 4 * 32] >= 0x58);
    bool b1 = (in[lane + 5 * 32] >= 0x58);
    bool b2 = (in[lane + 6 * 32] >= 0x58);
    bool b3 = (in[lane + 7 * 32] >= 0x58);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0x4a4a4a4au;
    carrier.b.x = (unsigned char)probe_mask4(a0, a1, a2, a3);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.y = (unsigned char)(probe_mask4(b0, b1, b2, b3) | 0x80u);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0f4a4a4au;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_nibble_transition_exact(uint32_t *out,
                                     const int *in,
                                     const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 4 * 32] >= 0x58);
    bool a1 = (in[lane + 5 * 32] >= 0x58);
    bool a2 = (in[lane + 6 * 32] >= 0x58);
    bool a3 = (in[lane + 7 * 32] >= 0x58);
    bool b0 = (in[lane + 8 * 32] != 0);
    bool b1 = (in[lane + 9 * 32] != 0);
    bool b2 = (in[lane + 10 * 32] != 0);
    bool b3 = (in[lane + 11 * 32] != 0);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0xaeaeaeaeu;
    carrier.b.x = (unsigned char)probe_mask4(a0, a1, a2, a3);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.z = (unsigned char)(probe_mask4(b0, b1, b2, b3) | 0x80u);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0faeaeaeu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_nibble_transition_exact(uint32_t *out,
                                     const int *in,
                                     const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 8 * 32] != 0);
    bool a1 = (in[lane + 9 * 32] != 0);
    bool a2 = (in[lane + 10 * 32] != 0);
    bool a3 = (in[lane + 11 * 32] != 0);
    bool b0 = (in[lane + 12 * 32] > 0);
    bool b1 = (in[lane + 13 * 32] > 0);
    bool b2 = (in[lane + 14 * 32] > 0);
    bool b3 = (in[lane + 15 * 32] > 0);

    probe_p2r_twostage_u32 carrier;
    carrier.w = seed_in[lane] ^ 0xc3c3c3c3u;
    carrier.b.x = (unsigned char)probe_mask4(a0, a1, a2, a3);
    asm volatile("" : "+r"(carrier.w));
    carrier.b.w = (unsigned char)(probe_mask4(b0, b1, b2, b3) | 0x80u);
    asm volatile("" : "+r"(carrier.w));

    out[lane] = carrier.w ^ 0x0fc3c3c3u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_regmask_transition_exact(uint32_t *out,
                                      const int *in,
                                      const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool b0 = (in[lane + 4 * 32] >= 0x58);
    bool b1 = (in[lane + 5 * 32] >= 0x58);
    bool b2 = (in[lane + 6 * 32] >= 0x58);
    bool b3 = (in[lane + 7 * 32] >= 0x58);

    uint32_t carrier = seed_in[lane] ^ 0x4a4a4a4au;
    uint32_t lo = probe_mask4(a0, a1, a2, a3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | lo;
    asm volatile("" : "+r"(carrier));

    uint32_t mid = (probe_mask4(b0, b1, b2, b3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (mid << 8);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0f4a4a4au;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_regmask_transition_exact(uint32_t *out,
                                      const int *in,
                                      const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 4 * 32] >= 0x58);
    bool a1 = (in[lane + 5 * 32] >= 0x58);
    bool a2 = (in[lane + 6 * 32] >= 0x58);
    bool a3 = (in[lane + 7 * 32] >= 0x58);
    bool b0 = (in[lane + 8 * 32] != 0);
    bool b1 = (in[lane + 9 * 32] != 0);
    bool b2 = (in[lane + 10 * 32] != 0);
    bool b3 = (in[lane + 11 * 32] != 0);

    uint32_t carrier = seed_in[lane] ^ 0xaeaeaeaeu;
    uint32_t lo = probe_mask4(a0, a1, a2, a3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | lo;
    asm volatile("" : "+r"(carrier));

    uint32_t hi = (probe_mask4(b0, b1, b2, b3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xff00ffffu) | (hi << 16);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0faeaeaeu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_regmask_transition_exact(uint32_t *out,
                                      const int *in,
                                      const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 8 * 32] != 0);
    bool a1 = (in[lane + 9 * 32] != 0);
    bool a2 = (in[lane + 10 * 32] != 0);
    bool a3 = (in[lane + 11 * 32] != 0);
    bool b0 = (in[lane + 12 * 32] > 0);
    bool b1 = (in[lane + 13 * 32] > 0);
    bool b2 = (in[lane + 14 * 32] > 0);
    bool b3 = (in[lane + 15 * 32] > 0);

    uint32_t carrier = seed_in[lane] ^ 0xc3c3c3c3u;
    uint32_t lo = probe_mask4(a0, a1, a2, a3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | lo;
    asm volatile("" : "+r"(carrier));

    uint32_t top = (probe_mask4(b0, b1, b2, b3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0x00ffffffu) | (top << 24);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0fc3c3c3u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_tripack_prefix_exact(uint32_t *out,
                                  const int *in,
                                  const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool b0 = (in[lane + 4 * 32] >= 0x58);
    bool b1 = (in[lane + 5 * 32] >= 0x58);
    bool b2 = (in[lane + 6 * 32] >= 0x58);
    bool b3 = (in[lane + 7 * 32] >= 0x58);
    bool c0 = (in[lane + 8 * 32] != 0);
    bool c1 = (in[lane + 9 * 32] != 0);
    bool c2 = (in[lane + 10 * 32] != 0);
    bool c3 = (in[lane + 11 * 32] != 0);

    uint32_t carrier = seed_in[lane] ^ 0xe7e7e7e7u;
    uint32_t x = probe_mask4(a0, a1, a2, a3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | x;
    asm volatile("" : "+r"(carrier));

    uint32_t y = (probe_mask4(b0, b1, b2, b3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (y << 8);
    asm volatile("" : "+r"(carrier));

    uint32_t z = (probe_mask4(c0, c1, c2, c3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xff00ffffu) | (z << 16);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0fe7e7e7u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_tripack_prefix_exact(uint32_t *out,
                                  const int *in,
                                  const uint32_t *seed_in) {
    int lane = threadIdx.x;

    bool a0 = (in[lane + 0 * 32] < 0x58);
    bool a1 = (in[lane + 1 * 32] < 0x58);
    bool a2 = (in[lane + 2 * 32] < 0x58);
    bool a3 = (in[lane + 3 * 32] < 0x58);
    bool b0 = (in[lane + 4 * 32] >= 0x58);
    bool b1 = (in[lane + 5 * 32] >= 0x58);
    bool b2 = (in[lane + 6 * 32] >= 0x58);
    bool b3 = (in[lane + 7 * 32] >= 0x58);
    bool c0 = (in[lane + 8 * 32] != 0);
    bool c1 = (in[lane + 9 * 32] != 0);
    bool c2 = (in[lane + 10 * 32] != 0);
    bool c3 = (in[lane + 11 * 32] != 0);
    bool d0 = (in[lane + 12 * 32] > 0);
    bool d1 = (in[lane + 13 * 32] > 0);
    bool d2 = (in[lane + 14 * 32] > 0);
    bool d3 = (in[lane + 15 * 32] > 0);

    uint32_t carrier = seed_in[lane] ^ 0xe7e7e7e7u;
    uint32_t x = probe_mask4(a0, a1, a2, a3) & 0x0fu;
    carrier = (carrier & 0xffffff00u) | x;
    asm volatile("" : "+r"(carrier));

    uint32_t y = (probe_mask4(b0, b1, b2, b3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xffff00ffu) | (y << 8);
    asm volatile("" : "+r"(carrier));

    uint32_t z = (probe_mask4(c0, c1, c2, c3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0xff00ffffu) | (z << 16);
    asm volatile("" : "+r"(carrier));

    uint32_t w = (probe_mask4(d0, d1, d2, d3) & 0x0fu) | 0x80u;
    carrier = (carrier & 0x00ffffffu) | (w << 24);
    asm volatile("" : "+r"(carrier));

    out[lane] = carrier ^ 0x0fe7e7e7u;
}
