/*
 * SASS RE Probe: runtime-safe fused branch for the top explorer-ranked combo:
 *
 *   uniform helpers + block-red + warp helpers + dense 64-bit SYS atomics
 *   + direct SYS load/store + async/cache backbone
 *
 * Goal: preserve the proven store-side SYS-safe branch while injecting the
 * direct local uniform helper front-end:
 *   ULDC.64 + UIADD3 + ULOP3.LUT + USHF
 */

#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint64_t combo_uniform_blockred_sys_store_u64[4] = {
    0x0123456789abcdefull,
    0xfedcba9876543210ull,
    0x00ff00ff00ff00ffull,
    0x1f001f001f001f00ull,
};

__constant__ uint32_t combo_uniform_blockred_sys_store_u32[8] = {
    0x00000001u,
    0x00000003u,
    0x00000007u,
    0x0000000fu,
    0x0000001fu,
    0x0000003fu,
    0x0000007fu,
    0x000000ffu,
};

__constant__ unsigned char combo_uniform_blockred_sys_store_u8[16] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
};

static __device__ __forceinline__ unsigned
combo_uniform_blockred_sys_store_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_uniform_blockred_warp_atomg64sys_ops_store_profile_safe(
    unsigned long long *out,
    unsigned long long *add_dst,
    unsigned long long *min_dst,
    unsigned long long *max_dst,
    unsigned long long *and_dst,
    unsigned long long *or_dst,
    unsigned long long *xor_dst,
    volatile unsigned long long *sys_store_dst,
    volatile const unsigned long long *sys_load_src,
    const unsigned char *src_u8,
    const unsigned long long *src_u64,
    unsigned long long seed,
    int mode,
    int bias,
    int limit) {
    __shared__ __align__(16) unsigned char smem[4096];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base =
        combo_uniform_blockred_sys_store_smem_addr(smem) + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 2048u + lane * 16u;

    const uint64_t c0 = combo_uniform_blockred_sys_store_u64[(mode + blockIdx.x) & 3];
    const uint64_t c1 = combo_uniform_blockred_sys_store_u64[(bias + blockIdx.x) & 3];
    const uint32_t m0 = combo_uniform_blockred_sys_store_u32[(mode + bias) & 7];
    const uint32_t m1 = combo_uniform_blockred_sys_store_u32[(limit + bias) & 7];
    const unsigned char k0 = combo_uniform_blockred_sys_store_u8[(mode + limit) & 15];
    const unsigned char k1 = combo_uniform_blockred_sys_store_u8[(bias + limit) & 15];

    uint64_t u = seed ^ c0 ^ (c1 << 1);
    u ^= ((uint64_t)(m0 ^ m1) << 32);
    u += ((uint64_t)k0 << 24) | ((uint64_t)k1 << 8);

    const unsigned shift0 = (unsigned)((mode + bias) & 31);
    const unsigned shift1 = (unsigned)((limit + mode + 1) & 31);
    const uint64_t rot0 = (u << shift0) | (u >> ((32u - shift0) & 31u));
    const uint64_t rot1 = (u << shift1) | (u >> ((32u - shift1) & 31u));
    const uint32_t lo = (uint32_t)(rot0 ^ rot1);
    const uint32_t hi = (uint32_t)((rot0 >> 32) ^ (rot1 >> 32));
    const uint32_t mix =
        (lo & m0) ^ (hi & m1) ^ ((uint32_t)k0 << 8) ^ (uint32_t)k1;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base + 0u), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base + 2048u), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    unsigned sm2 = 0;
    unsigned sm3 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base + 0u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base + 8u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm2) : "r"(base + 2048u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm3) : "r"(base + 2056u));

    unsigned long long acc =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 16) ^
        ((unsigned long long)(sm2 & 0xffu) << 24) ^
        ((unsigned long long)(sm3 & 0xffu) << 32) ^
        seed ^ (unsigned long long)lane;
    acc ^= ((unsigned long long)lo << 1) ^ ((unsigned long long)hi << 9);
    acc += ((unsigned long long)(mix & 0xffu) << 17);
    acc ^= ((acc << 7) | (acc >> 57));
    acc += 0x9e3779b97f4a7c15ull ^ (unsigned long long)(lane * 0x101u);
    acc ^= ((acc << 13) | (acc >> 51));
    acc += ((unsigned long long)sm0 << 1) ^ ((unsigned long long)sm2 << 9);

    const int pred_odd = (int)((acc & 1ull) != 0ull);
    const int pred_small = (int)(((acc ^ hi) & 0xffull) < (unsigned long long)(limit & 0xff));
    const int pred_nonzero = (int)(((mix ^ lo) & 0xffu) != 0u);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    const int lane_val = (int)(acc & 0xffull);
    const unsigned ballot = __ballot_sync(0xffffffffu, (lane_val & 1) != 0);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const unsigned vote_any = __any_sync(0xffffffffu, lane_val > 31);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int warp_min = __reduce_min_sync(0xffffffffu, lane_val);
    const int warp_max = __reduce_max_sync(0xffffffffu, lane_val);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int leader = __ffs((int)group) - 1;

    acc ^=
        (unsigned long long)block_popc ^
        ((unsigned long long)block_and << 8) ^
        ((unsigned long long)block_or << 16) ^
        (unsigned long long)(ballot & 0xffu) ^
        ((unsigned long long)(vote_all & 1) << 24) ^
        ((unsigned long long)(vote_any & 1u) << 28) ^
        ((unsigned long long)(leader & 0xff) << 32) ^
        ((unsigned long long)(warp_min & 0xff) << 40) ^
        ((unsigned long long)(warp_max & 0xff) << 48) ^
        ((unsigned long long)(warp_sum & 0xff) << 56);

    if ((lane & 7u) == 0u) {
        const unsigned long long observed0 = sys_load_src[(lane >> 3) & 1u];
        sys_store_dst[(lane >> 3) & 1u] =
            observed0 ^ acc ^ 0x0101010101010101ull ^ (unsigned long long)mix;

        const unsigned long long old_add = atomicAdd_system(add_dst, acc | 1ull);
        const unsigned long long old_min =
            atomicMin_system(min_dst, acc ^ 0x1111111111111111ull);
        const unsigned long long old_max =
            atomicMax_system(max_dst, acc ^ 0x2222222222222222ull);
        const unsigned long long old_and =
            atomicAnd_system(and_dst, acc | 0xff00ff00ff00ff00ull);
        const unsigned long long old_or =
            atomicOr_system(or_dst, acc | 0x00ff00ff00ff00ffull);
        const unsigned long long old_xor =
            atomicXor_system(xor_dst, acc ^ 0x3333333333333333ull);

        acc ^= observed0 ^ old_add ^ old_min ^ old_max ^ old_and ^ old_or ^ old_xor;
        acc += ((observed0 & 0xffull) << 5) ^ ((old_xor & 0xffull) << 13);

        const unsigned long long observed1 = sys_load_src[((lane >> 3) & 1u) ^ 1u];
        sys_store_dst[((lane >> 3) & 1u) ^ 1u] =
            observed1 ^ acc ^ 0x0202020202020202ull ^ (unsigned long long)hi;

        acc ^= atomicAdd_system(add_dst, acc | 3ull);
        acc ^= atomicMin_system(min_dst, acc ^ 0x4444444444444444ull);
        acc ^= atomicMax_system(max_dst, acc ^ 0x5555555555555555ull);
        acc ^= atomicAnd_system(and_dst, acc | 0xf0f0f0f0f0f0f0f0ull);
        acc ^= atomicOr_system(or_dst, acc | 0x0f0f0f0f0f0f0f0full);
        acc ^= atomicXor_system(xor_dst, acc ^ 0x6666666666666666ull);
        __threadfence_system();
        acc ^= observed1 ^ (unsigned long long)lo;
    }

    out[lane] = acc;
}
