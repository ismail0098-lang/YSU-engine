/*
 * SASS RE Probe: runtime-safe surrogate for the dense 64-bit SYS store-side family
 *
 * Goal: preserve the proven deep 64-bit SYS-safe body while reintroducing
 * direct 64-bit system load/store reuse in a runtime-safe form so the
 * store-side symbolic frontier can be profiled rather than only disassembled.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned
combo_blockred_warp_atomg64sys_store_profile_safe_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_blockred_warp_atomg64sys_ops_store_profile_safe(
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
    unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[4096];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base =
        combo_blockred_warp_atomg64sys_store_profile_safe_smem_addr(smem) +
        lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 2048u + lane * 16u;

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
        bias ^ (unsigned long long)lane;
    acc ^= ((acc << 7) | (acc >> 57));
    acc += 0x9e3779b97f4a7c15ull ^ (unsigned long long)(lane * 0x101u);
    acc ^= ((acc << 13) | (acc >> 51));
    acc += ((unsigned long long)sm0 << 1) ^ ((unsigned long long)sm2 << 9);

    const int pred_odd = (int)((acc & 1ull) != 0ull);
    const int pred_small = (int)((acc & 0xffull) < 192ull);
    const int pred_nonzero = (int)((acc & 0xffull) != 0ull);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    const int lane_val = (int)(acc & 0xffull);
    const unsigned ballot = __ballot_sync(0xffffffffu, (lane_val & 1) != 0);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
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
        ((unsigned long long)(leader & 0xff) << 32) ^
        ((unsigned long long)(warp_min & 0xff) << 40) ^
        ((unsigned long long)(warp_max & 0xff) << 48) ^
        ((unsigned long long)(warp_sum & 0xff) << 56);

    if ((lane & 7u) == 0u) {
        const unsigned long long observed0 = sys_load_src[(lane >> 3) & 1u];
        sys_store_dst[(lane >> 3) & 1u] = observed0 ^ acc ^ 0x0101010101010101ull;

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
            observed1 ^ acc ^ 0x0202020202020202ull;

        acc ^= atomicAdd_system(add_dst, acc | 3ull);
        acc ^= atomicMin_system(min_dst, acc ^ 0x4444444444444444ull);
        acc ^= atomicMax_system(max_dst, acc ^ 0x5555555555555555ull);
        acc ^= atomicAnd_system(and_dst, acc | 0xf0f0f0f0f0f0f0f0ull);
        acc ^= atomicOr_system(or_dst, acc | 0x0f0f0f0f0f0f0f0full);
        acc ^= atomicXor_system(xor_dst, acc ^ 0x6666666666666666ull);
        __threadfence_system();
        acc ^= observed1;
    }

    out[lane] = acc;
}
