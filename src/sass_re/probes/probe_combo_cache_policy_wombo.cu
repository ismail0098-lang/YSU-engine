/*
 * SASS RE Probe: cache-policy + async-depbar combo
 *
 * Goal: encourage coexistence of:
 *   - LDGSTS.E.BYPASS.LTC128B.128 / LDGDEPBAR / DEPBAR.LE
 *   - LDG.E.*.STRONG.GPU under -Xptxas -dlcm=cg
 */

#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint32_t combo_policy_masks[8] = {
    0x00000001u,
    0x00000003u,
    0x00000007u,
    0x0000000fu,
    0x0000001fu,
    0x0000003fu,
    0x0000007fu,
    0x000000ffu,
};

static __device__ __forceinline__ unsigned combo_policy_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_cache_policy_wombo(float *out,
                               const unsigned char *src_u8,
                               const uint16_t *src_u16,
                               const uint32_t *src_u32,
                               int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    unsigned saddr = combo_policy_smem_addr(smem);
    int lane = threadIdx.x;

    const unsigned char *gptr = src_u8 + lane;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr), "l"(gptr));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned u32 = src_u32[lane];
    unsigned short u16 = src_u16[lane];
    unsigned char u8 = src_u8[lane];

    unsigned smem_word = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(smem_word) : "r"(saddr));

    float x = static_cast<float>((u32 & 0xffu) + u16 + u8 + (smem_word & 0xffu));
    if (bias & 1) x += 1.0f;
    if (bias & 2) x *= 1.0001f;
    if (bias & 4) x -= 0.5f;
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_cache_policy_uniform_masks(float *out,
                                       const unsigned char *src_u8,
                                       const uint16_t *src_u16,
                                       const uint32_t *src_u32,
                                       int mode,
                                       int bias,
                                       int limit) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_policy_smem_addr(smem[0]);
    const unsigned base1 = combo_policy_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    const uint32_t mask_a = combo_policy_masks[(mode + blockIdx.x) & 7];
    const uint32_t mask_b = combo_policy_masks[(bias + blockIdx.x) & 7];
    const uint32_t mask_c = combo_policy_masks[(limit + blockIdx.x) & 7];
    const uint32_t mix = mask_a ^ (mask_b << 1) ^ (mask_c << 2);

    const bool gate0 = ((mix & 0x40u) != 0u) || ((unsigned)limit > (unsigned)mode);
    const bool gate1 = ((mix & 0x80u) != 0u) || ((unsigned)bias >= (unsigned)limit);
    const bool gate2 = (((unsigned)(mode + bias) & 7u) <= ((unsigned)limit & 7u));
    const bool gate3 = (((unsigned)(mask_a + mask_b) & 0xffu) != ((unsigned)mask_c & 0xffu));

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base0), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base1), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const uint32_t u32 = src_u32[lane];
    const uint16_t u16 = src_u16[lane];
    const unsigned char u8 = src_u8[lane];

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base1));

    uint32_t acc = (u32 & mask_a) + (uint32_t)u16 + (uint32_t)u8 + (sm0 & mask_b);
    if (gate0) acc ^= (sm1 & mask_c);
    if (gate1) acc += mask_b;
    if (gate2) acc ^= (mask_a >> 1);
    if (gate3) acc += (uint32_t)lane;

    float x = (float)(acc & 0xffffu);
    if (gate0 && gate2) x *= 1.0001f;
    if (gate1 || gate3) x += 0.25f;
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_cache_policy_uniform_uisept(float *out,
                                        const unsigned char *src_u8,
                                        const uint16_t *src_u16,
                                        const uint32_t *src_u32,
                                        uint64_t seed,
                                        int stage_mask,
                                        int rounds) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_policy_smem_addr(smem[0]);
    const unsigned base1 = combo_policy_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    uint64_t u = seed ^ ((uint64_t)(stage_mask & 0xff) << 32);
    const uint64_t mix = ((uint64_t)(stage_mask & 0x40) << 21)
                       | ((uint64_t)(stage_mask & 0x80) << 13)
                       | 0x0000001000000001ull;

    #pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        uint64_t left = u << 11;
        uint64_t right = u >> (((i + stage_mask) & 0x1f) + 1);
        u = left ^ right ^ mix ^ ((uint64_t)(i + 1) << 32);
        asm volatile("" : "+l"(u));
    }

    uint32_t lo = (uint32_t)u;
    uint32_t hi = (uint32_t)(u >> 32);
    uint32_t idx_mask = combo_policy_masks[stage_mask & 7];
    uint32_t idx = (lo ^ hi ^ idx_mask) & 31u;
    bool u_gate0 = (hi != 0u);
    bool u_gate1 = ((lo & 1u) != 0u);
    bool u_gate2 = (hi >= (uint32_t)(idx_mask & 0xffu));

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base0), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base1), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const uint32_t u32 = src_u32[idx];
    const uint16_t u16 = src_u16[idx];
    const unsigned char u8 = src_u8[idx];

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base1));

    uint32_t acc = (u32 & idx_mask) + (uint32_t)u16 + (uint32_t)u8 + (sm0 & 0xffu);
    if (u_gate0) acc ^= (sm1 & 0xffu);
    if (u_gate1) acc += hi;
    if (u_gate2) acc ^= lo;

    float x = (float)(acc & 0xffffu);
    if (u_gate0 && u_gate2) x *= 1.0001f;
    if (u_gate1) x += 0.5f;
    out[lane] = x + (float)lane;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_cache_policy_uniform_async_uisept_strict(float *out,
                                                     const unsigned char *src_u8,
                                                     const uint16_t *src_u16,
                                                     const uint32_t *src_u32,
                                                     uint64_t seed,
                                                     int shift_sel,
                                                     int stage_mask,
                                                     int rounds) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_policy_smem_addr(smem[0]);
    const unsigned base1 = combo_policy_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    uint64_t u = seed ^ ((uint64_t)(stage_mask & 0xff) << 32);
    uint64_t mix = ((uint64_t)(stage_mask & 0x40) << 21)
                 | ((uint64_t)(stage_mask & 0x80) << 13)
                 | 0x0000001000000001ull;

    #pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        int s = ((shift_sel + i) & 0x1f) + 1;
        uint64_t left = u << 11;
        uint64_t right = u >> s;
        u = left ^ right ^ mix ^ ((uint64_t)(i + 1) << 32);
        if ((stage_mask & 0x40) != 0) {
            u += 0x0000000100000001ull;
        }
        if ((stage_mask & 0x80) != 0) {
            u ^= 0x0000000200000020ull;
        }
        asm volatile("" : "+l"(u));
    }

    uint32_t lo = (uint32_t)u;
    uint32_t hi = (uint32_t)(u >> 32);
    uint32_t idx_mask = combo_policy_masks[stage_mask & 7];
    uint32_t idx = (lo ^ hi ^ idx_mask) & 31u;

    bool stage_nonzero = (hi != 0u);
    bool stage_lo_bit = ((lo & 1u) != 0u);
    bool stage_hi_ge = (hi >= idx_mask);
    bool stage_mix_ne = ((hi ^ lo) != 0u);

    const unsigned char *gptr0 = src_u8 + idx + lane;
    const unsigned char *gptr1 = src_u8 + idx + lane + 16;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base0), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base1), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const uint32_t u32 = src_u32[idx];
    const uint16_t u16 = src_u16[idx];
    const unsigned char u8 = src_u8[idx];

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base1));

    uint32_t acc = (u32 & idx_mask) + (uint32_t)u16 + (uint32_t)u8 + (sm0 & 0xffu);
    if (stage_nonzero) acc ^= (sm1 & 0xffu);
    if (stage_lo_bit) acc += hi;
    if (stage_hi_ge) acc ^= lo;
    if (stage_mix_ne) acc += idx_mask;

    float x = (float)(acc & 0xffffu);
    if (stage_nonzero && stage_hi_ge) x *= 1.0001f;
    if (stage_lo_bit || stage_mix_ne) x += 0.25f;
    out[lane] = x + (float)lane;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_split_uniform_gate_async(float *out,
                                     const unsigned char *src_u8,
                                     const uint16_t *src_u16,
                                     const uint32_t *src_u32,
                                     uint64_t seed,
                                     int shift_sel,
                                     int stage_mask,
                                     int rounds) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_policy_smem_addr(smem[0]);
    const unsigned base1 = combo_policy_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    uint64_t u = seed ^ ((uint64_t)(stage_mask & 0xff) << 32);
    uint64_t mix = ((uint64_t)(stage_mask & 0x40) << 21)
                 | ((uint64_t)(stage_mask & 0x80) << 13)
                 | 0x0000001000000001ull;

    #pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        int s = ((shift_sel + i) & 0x1f) + 1;
        uint64_t left = u << 11;
        uint64_t right = u >> s;
        u = left ^ right ^ mix ^ ((uint64_t)(i + 1) << 32);
        if ((stage_mask & 0x40) != 0) {
            u += 0x0000000100000001ull;
        }
        if ((stage_mask & 0x80) != 0) {
            u ^= 0x0000000200000020ull;
        }
        asm volatile("" : "+l"(u));
    }

    uint32_t lo = (uint32_t)u;
    uint32_t hi = (uint32_t)(u >> 32);
    uint32_t idx_mask = combo_policy_masks[stage_mask & 7];
    uint32_t idx = (lo ^ hi ^ idx_mask) & 31u;

    bool u_gate0 = (hi != 0u);
    bool u_gate1 = ((lo & 1u) != 0u);
    bool u_gate2 = (hi >= idx_mask);
    bool u_gate3 = ((hi ^ lo) != 0u);

    const unsigned char *gptr0 = src_u8 + idx + lane;
    const unsigned char *gptr1 = src_u8 + idx + lane + 16;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base0), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base1), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const uint32_t u32 = src_u32[idx];
    const uint16_t u16 = src_u16[idx];
    const unsigned char u8 = src_u8[idx];

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base1));

    uint32_t acc = (u32 & idx_mask) + (uint32_t)u16 + (uint32_t)u8 + (sm0 & 0xffu);
    if (u_gate0) acc ^= (sm1 & 0xffu);
    if (u_gate1) acc += hi;
    if (u_gate2) acc ^= lo;
    if (u_gate3) acc += idx_mask;

    float x = (float)(acc & 0xffffu);
    if (u_gate0 && u_gate2) x *= 1.0001f;
    if (u_gate1 || u_gate3) x += 0.25f;
    out[lane] = x + (float)lane;
}
