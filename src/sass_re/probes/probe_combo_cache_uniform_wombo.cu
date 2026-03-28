/*
 * SASS RE Probe: cache/uniform "wombo combo" neighborhoods
 *
 * Goal: force the strongest next-step chain motifs surfaced by chain mining:
 *   1. LDGSTS.E.BYPASS.LTC128B.128 + LDGDEPBAR + DEPBAR.LE + UISETP.*
 *   2. ULOP3.LUT + UIADD3 + ULDC(.64/.U8/.S8) + USHF.L.U32
 */

#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint64_t combo_const_u64[4] = {
    0x0123456789abcdefull,
    0xfedcba9876543210ull,
    0x00000000000000ffull,
    0x0000000000001f00ull,
};

__constant__ unsigned char combo_const_u8[16] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
};

static __device__ __forceinline__ unsigned combo_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_ldgsts_depbar_uisept(float *out,
                                 const unsigned char *src,
                                 int mode,
                                 int limit,
                                 int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    unsigned saddr = combo_smem_addr(smem);
    int lane = threadIdx.x;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    int u0 = mode + bias;
    int u1 = limit - bias;
    int gate_a = (u0 >= u1);
    int gate_b = (u0 != 0);
    int gate_c = (u1 > 7);

    unsigned word = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word) : "r"(saddr));

    float x = static_cast<float>((word & 0xffu) + lane);
    if (gate_a) {
        x += static_cast<float>(combo_const_u8[(lane + mode) & 15]);
    }
    if (gate_b) {
        x *= 1.0001f;
    }
    if (!gate_c) {
        x -= 0.25f;
    }

    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_ulop3_uiadd3_uldc(uint32_t *out,
                              int mode,
                              int bias,
                              int shift_seed) {
    int lane = threadIdx.x;
    uint64_t u64 = combo_const_u64[(mode + blockIdx.x) & 3];
    unsigned char u8 = combo_const_u8[(bias + blockIdx.x) & 15];

    uint32_t lo = static_cast<uint32_t>(u64);
    uint32_t hi = static_cast<uint32_t>(u64 >> 32);

    int warp_base = blockIdx.x * blockDim.x;
    int uniform_offset = warp_base + mode + bias;
    int shift = (shift_seed + bias) & 31;

    uint32_t mixed = (lo ^ hi) + static_cast<uint32_t>(uniform_offset);
    mixed ^= static_cast<uint32_t>(u8) << (shift & 7);
    mixed = (mixed << shift) | (mixed >> ((32 - shift) & 31));

    uint32_t final = mixed + static_cast<uint32_t>(lane);
    out[lane] = final;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_ldgsts_uniform_masks(float *out,
                                 const unsigned char *src,
                                 uint64_t byte_stride,
                                 int stage_mask,
                                 int tail_mask) {
    __shared__ __align__(16) unsigned char smem[2][32];
    int lane = threadIdx.x;
    const unsigned base0 = combo_smem_addr(smem[0]);
    const unsigned base1 = combo_smem_addr(smem[1]);
    uint64_t byte_off = 0;
    float acc = 0.0f;

    #pragma unroll 1
    for (int t = 0; t < 2; ++t) {
        int stage = t & 1;
        int next_stage = stage ^ 1;
        const unsigned cur_dst = stage ? base1 : base0;
        const unsigned nxt_dst = next_stage ? base1 : base0;
        const unsigned char *gptr = src + byte_off;

        bool gate40 = (stage_mask & 0x40) != 0;
        bool gate80 = (stage_mask & 0x80) != 0;
        bool tail0 = (tail_mask & 0x40) != 0;
        bool tail1 = (tail_mask & 0x80) != 0;
        bool loop0 = ((t + stage_mask) & 1) == 0;
        bool loop1 = ((t + tail_mask) & 2) != 0;
        bool do_prefetch = (t + 1) < 2;
        bool gate_a = (gate40 && !loop0) || (tail0 && loop1);
        bool gate_b = (gate80 && loop0) || (tail1 && !loop1);

        if (!tail0 || !tail1) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                         :: "r"(cur_dst), "l"(gptr));
            if (do_prefetch) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                             :: "r"(nxt_dst), "l"(gptr + byte_stride));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
        }

        if (gate_a || gate_b) {
            unsigned word = 0;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word) : "r"(cur_dst));
            acc += static_cast<float>(word & 0xffu);
        }
        if (gate40 && tail0) acc += static_cast<float>(smem[stage][0]);
        if (gate80 && tail1) acc += static_cast<float>(smem[next_stage][1]);

        byte_off += ((uint64_t)(gate40 ? 2 : 1)) * byte_stride;
        asm volatile("" : "+l"(byte_off));
    }

    out[lane] = acc + static_cast<float>(lane);
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_ldgsts_uisetp_exact(float *out,
                                const unsigned char *src,
                                int stage_mask,
                                int tail_mask,
                                int limit_mask) {
    __shared__ __align__(16) unsigned char smem[2][32];
    int lane = threadIdx.x;
    const unsigned base0 = combo_smem_addr(smem[0]);
    const unsigned base1 = combo_smem_addr(smem[1]);
    const unsigned char *gptr = src;

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

    if (stage_eq_1 || !tail0 || !tail1) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                     :: "r"(base0), "l"(gptr));
        if (tail_eq_1 || limit_eq_1) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                         :: "r"(base1), "l"(gptr + 16));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }

    float acc = 0.0f;
    if ((gate40 && !limit0) || (tail0 && gate80) || stage_eq_1) {
        unsigned word = 0;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word) : "r"(base0));
        acc += static_cast<float>(word & 0xffu);
    }
    if ((gate80 && limit0) || (tail1 && gate40) || tail_eq_1) {
        unsigned word = 0;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word) : "r"(base1));
        acc += static_cast<float>(word & 0xffu);
    }

    out[lane] = acc + static_cast<float>(lane);
}
