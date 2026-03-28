/*
 * SASS RE Probe: narrow WARPSYNC chase on top of the fused local family
 *
 * Goal: try the least-disruptive syncwarp shapes that could select WARPSYNC in
 * optimized output while preserving as much of the fused family as possible.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_warpsync_fused_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_warpsync_fullmask_fused(unsigned long long *out,
                                    unsigned long long *add_dst,
                                    unsigned long long *min_dst,
                                    unsigned long long *max_dst,
                                    const unsigned char *src_u8,
                                    const unsigned long long *src_u64,
                                    unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[256];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr =
        combo_warpsync_fused_smem_addr(smem) + ((lane & 15u) << 4);

    if ((lane & 1u) == 0u) {
        smem[(lane & 15u) << 4] = (unsigned char)(lane ^ (unsigned)bias);
    } else {
        smem[((lane & 15u) << 4) + 1u] = (unsigned char)(lane + (unsigned)bias);
    }
    __syncwarp(0xffffffffu);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 128), "l"(src_u8 + lane + 32));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 128));

    const unsigned long long v64 =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane;

    const int pred_odd = (int)((v64 & 1ull) != 0ull);
    const int pred_small = (int)((v64 & 0xffull) < 192ull);
    const int pred_nonzero = (int)((v64 & 0xffull) != 0ull);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    const int lane_val = (int)(v64 & 0xffull);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int leader = __ffs((int)group) - 1;

    unsigned long long acc =
        v64 ^
        (unsigned long long)block_popc ^
        ((unsigned long long)block_and << 8) ^
        ((unsigned long long)block_or << 16) ^
        ((unsigned long long)(vote_all & 1) << 24) ^
        ((unsigned long long)(leader & 0xff) << 32) ^
        ((unsigned long long)(warp_sum & 0xff) << 40);

    if ((lane & 7u) == 0u) {
        acc ^= atomicAdd_system(add_dst, acc | 1ull);
        acc ^= atomicMin_system(min_dst, acc ^ 0x1111111111111111ull);
        acc ^= atomicMax_system(max_dst, acc ^ 0x2222222222222222ull);
        __threadfence_system();
    }

    out[lane] = acc;
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_warpsync_ballotmask_fused(unsigned long long *out,
                                      unsigned long long *add_dst,
                                      unsigned long long *min_dst,
                                      unsigned long long *max_dst,
                                      const unsigned char *src_u8,
                                      const unsigned long long *src_u64,
                                      unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[256];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr =
        combo_warpsync_fused_smem_addr(smem) + ((lane & 15u) << 4);

    const unsigned pre_mask = __ballot_sync(0xffffffffu, ((lane + (unsigned)bias) & 1u) != 0u);
    if ((lane & 1u) == 0u) {
        smem[(lane & 15u) << 4] = (unsigned char)(lane ^ (unsigned)bias);
    } else {
        smem[((lane & 15u) << 4) + 1u] = (unsigned char)(lane + (unsigned)bias);
    }
    __syncwarp(pre_mask | 1u);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 128), "l"(src_u8 + lane + 32));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 128));

    const unsigned long long v64 =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane;

    const int pred_odd = (int)((v64 & 1ull) != 0ull);
    const int pred_small = (int)((v64 & 0xffull) < 192ull);
    const int pred_nonzero = (int)((v64 & 0xffull) != 0ull);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    const int lane_val = (int)(v64 & 0xffull);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int leader = __ffs((int)group) - 1;

    unsigned long long acc =
        v64 ^
        (unsigned long long)block_popc ^
        ((unsigned long long)block_and << 8) ^
        ((unsigned long long)block_or << 16) ^
        ((unsigned long long)(vote_all & 1) << 24) ^
        ((unsigned long long)(leader & 0xff) << 32) ^
        ((unsigned long long)(warp_sum & 0xff) << 40);

    if ((lane & 7u) == 0u) {
        acc ^= atomicAdd_system(add_dst, acc | 1ull);
        acc ^= atomicMin_system(min_dst, acc ^ 0x1111111111111111ull);
        acc ^= atomicMax_system(max_dst, acc ^ 0x2222222222222222ull);
        __threadfence_system();
    }

    out[lane] = acc;
}
