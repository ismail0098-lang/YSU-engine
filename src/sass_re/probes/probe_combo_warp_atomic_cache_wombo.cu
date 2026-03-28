/*
 * SASS RE Probe: warp + atomic + async/cache "wombo combo"
 *
 * Goal: combine:
 *   - warp vote/match/redux neighborhood
 *   - RED/atomic strong-GPU forms
 *   - LDG(.STRONG.GPU) + LDGSTS + LDGDEPBAR + DEPBAR
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_warp_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_warp_atomic_cache_wombo(float *out,
                                    float *accum,
                                    int *atomic_dst,
                                    const unsigned char *src_u8,
                                    const uint16_t *src_u16,
                                    int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_warp_smem_addr(smem);
    const int lane = threadIdx.x;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 32), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const unsigned char u8 = src_u8[lane];
    const uint16_t u16 = src_u16[lane];
    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 32));

    unsigned v = (unsigned)u8 + (unsigned)u16 + (sm0 & 0xffu) + (sm1 & 0xffu) + (unsigned)bias;
    unsigned mask = __match_any_sync(0xffffffffu, v & 3u);
    unsigned warp_sum = __reduce_add_sync(0xffffffffu, (int)(v & 0xffu));
    int leader = __ffs((int)mask) - 1;
    float x = (float)((v & 0xffu) + warp_sum + leader + lane);

    if ((bias & 1) != 0) x *= 1.0001f;
    if ((bias & 2) != 0) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)(mask & 0xffu));
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_warpsync_atomic_cache_wombo(float *out,
                                        float *accum,
                                        int *atomic_dst,
                                        const unsigned char *src_u8,
                                        const uint16_t *src_u16,
                                        int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_warp_smem_addr(smem);
    const int lane = threadIdx.x;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 32), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const unsigned char u8 = src_u8[lane];
    const uint16_t u16 = src_u16[lane];
    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 32));

    unsigned mask = __ballot_sync(0xffffffffu, ((u8 + bias + lane) & 1u) != 0u);
    __syncwarp(mask | 1u);
    unsigned group = __match_any_sync(0xffffffffu, (u8 + u16) & 3u);
    unsigned sum = __reduce_add_sync(0xffffffffu, (int)((u8 + u16 + sm0 + sm1) & 0xffu));
    int lead = __ffs((int)group) - 1;

    float x = (float)(((u8 + u16 + (sm0 & 0xffu) + (sm1 & 0xffu)) & 0xffu) + sum + lead + lane);
    if ((bias & 1) != 0) x *= 1.0001f;
    if ((bias & 2) != 0) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)(mask & 0xffu));
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_voteall_minmax_atomic_cache_wombo(float *out,
                                              float *accum,
                                              int *atomic_dst,
                                              const unsigned char *src_u8,
                                              const uint16_t *src_u16,
                                              int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_warp_smem_addr(smem);
    const int lane = threadIdx.x;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 32), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const unsigned char u8 = src_u8[lane];
    const uint16_t u16 = src_u16[lane];
    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 32));

    const unsigned packed = (unsigned)u8 + (unsigned)u16 +
                            (sm0 & 0xffu) + (sm1 & 0xffu) + (unsigned)bias;
    const int lane_val = (int)(packed & 0xffu);
    const unsigned ballot = __ballot_sync(0xffffffffu, (lane_val & 1) != 0);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int warp_min = __reduce_min_sync(0xffffffffu, lane_val);
    const int warp_max = __reduce_max_sync(0xffffffffu, lane_val);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int pop = __popc(ballot);
    const int leader = __ffs((int)group) - 1;

    float x = (float)(warp_sum + warp_min + warp_max + pop + leader + vote_all + lane);
    if ((bias & 1) != 0) x *= 1.0001f;
    if ((bias & 2) != 0) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)((ballot & 0xffu) + (group & 0xffu)));
    out[lane] = x;
}
