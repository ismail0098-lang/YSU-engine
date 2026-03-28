/*
 * SASS RE Probe: pure-uniform gate + async/cache "wombo combo"
 *
 * Goal: keep the compare/predicate half in the uniform domain for as long as
 * possible, then use only the resulting uniform gates to modulate a proven
 * async/cache backbone:
 *   - ULDC(.64/.U8) + UIADD3 + ULOP3.LUT + USHF + USEL/UISETP? family
 *   - LDG(.STRONG.GPU) + LDGSTS(.ZFILL) + LDGDEPBAR + DEPBAR.LE
 */

#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint64_t combo_uniform_gate_u64[4] = {
    0x0123456789abcdefull,
    0xfedcba9876543210ull,
    0x00ff00ff00ff00ffull,
    0x1f001f001f001f00ull,
};

__constant__ uint32_t combo_uniform_gate_u32[8] = {
    0x00000001u,
    0x00000003u,
    0x00000007u,
    0x0000000fu,
    0x0000001fu,
    0x0000003fu,
    0x0000007fu,
    0x000000ffu,
};

__constant__ unsigned char combo_uniform_gate_u8[16] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
};

static __device__ __forceinline__ unsigned combo_uniform_gate_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_uniform_gate_async_wombo(float *out,
                                     const unsigned char *src_u8,
                                     const uint16_t *src_u16,
                                     const uint32_t *src_u32,
                                     uint64_t seed,
                                     int mode,
                                     int bias,
                                     int limit) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_uniform_gate_smem_addr(smem[0]);
    const unsigned base1 = combo_uniform_gate_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    const uint64_t c0 = combo_uniform_gate_u64[(mode + blockIdx.x) & 3];
    const uint64_t c1 = combo_uniform_gate_u64[(bias + blockIdx.x) & 3];
    const uint32_t m0 = combo_uniform_gate_u32[(mode + bias) & 7];
    const uint32_t m1 = combo_uniform_gate_u32[(limit + bias) & 7];
    const unsigned char k0 = combo_uniform_gate_u8[(mode + limit) & 15];
    const unsigned char k1 = combo_uniform_gate_u8[(bias + limit) & 15];

    uint64_t u = seed ^ c0 ^ (c1 << 1);
    u ^= ((uint64_t)(m0 ^ m1) << 32);
    u += ((uint64_t)k0 << 24) | ((uint64_t)k1 << 8);

    const unsigned shift0 = (unsigned)((mode + bias) & 31);
    const unsigned shift1 = (unsigned)((limit + mode + 1) & 31);
    const uint64_t rot0 = (u << shift0) | (u >> ((32u - shift0) & 31u));
    const uint64_t rot1 = (u << shift1) | (u >> ((32u - shift1) & 31u));
    const uint32_t lo = (uint32_t)(rot0 ^ rot1);
    const uint32_t hi = (uint32_t)((rot0 >> 32) ^ (rot1 >> 32));
    const uint32_t mix = (lo & m0) ^ (hi & m1) ^ ((uint32_t)k0 << 8) ^ (uint32_t)k1;

    const bool u_gate0 = (hi >= (uint32_t)(limit & 0xff));
    const bool u_gate1 = ((mix & 0x40u) != 0u);
    const bool u_gate2 = (((mix ^ hi) & 0xffu) != (uint32_t)(mode & 0xff));
    const bool u_gate3 = ((lo + m0) >= (hi ^ m1));

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

    uint32_t acc = (u32 & m0) + (uint32_t)u16 + (uint32_t)u8 + (sm0 & 0xffu);
    if (u_gate0) acc ^= (sm1 & 0xffu);
    if (u_gate1) acc += hi;
    if (u_gate2) acc ^= lo;
    if (u_gate3) acc += (uint32_t)lane;

    float x = (float)(acc & 0xffffu);
    if (u_gate0 && u_gate2) x *= 1.0001f;
    if (u_gate1 || u_gate3) x += 0.25f;
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_uniform_gate_async_select_wombo(float *out,
                                            const unsigned char *src_u8,
                                            const uint16_t *src_u16,
                                            const uint32_t *src_u32,
                                            uint64_t seed,
                                            int mode,
                                            int bias,
                                            int limit) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const unsigned base0 = combo_uniform_gate_smem_addr(smem[0]);
    const unsigned base1 = combo_uniform_gate_smem_addr(smem[1]);
    const int lane = threadIdx.x;

    const uint64_t c0 = combo_uniform_gate_u64[(mode + blockIdx.x) & 3];
    const uint64_t c1 = combo_uniform_gate_u64[(bias + blockIdx.x) & 3];
    const uint32_t m0 = combo_uniform_gate_u32[(mode + limit) & 7];
    const uint32_t m1 = combo_uniform_gate_u32[(bias + limit) & 7];

    uint64_t u = seed + c0 + (c1 << 3);
    const unsigned sh = (unsigned)((mode + limit + 1) & 31);
    u ^= (u << sh) | (u >> ((32u - sh) & 31u));
    const uint32_t lo = (uint32_t)u;
    const uint32_t hi = (uint32_t)(u >> 32);
    const uint32_t sel_a = (lo & m0) ^ hi;
    const uint32_t sel_b = (hi & m1) ^ lo;

    const bool use_a = (sel_a >= sel_b);
    const bool use_b = ((sel_a ^ sel_b) & 0x80u) != 0u;
    const bool use_c = ((sel_a + m0) != (sel_b + m1));

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

    uint32_t picked = use_a ? sel_a : sel_b;
    picked = use_b ? (picked ^ (uint32_t)u16) : (picked + (uint32_t)u8);
    picked = use_c ? (picked + (sm0 & 0xffu)) : (picked ^ (sm1 & 0xffu));
    picked += (u32 & m0);

    float x = (float)(picked & 0xffffu);
    if (use_a && use_c) x *= 1.0001f;
    if (use_b) x += 0.5f;
    out[lane] = x + (float)lane;
}
