/*
 * SASS RE Probe: runtime-safe warp + atomic + async/cache profile surrogate
 * with deeper dependent dataflow.
 *
 * Goal: preserve the same emitted family as the profile-safe anchor while
 * increasing dependency depth enough to test whether long-scoreboard pressure
 * scales the way the first `ncu` pass suggests.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_profile_depth_safe_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_warp_atomic_cache_profile_depth_safe(float *out,
                                                 float *accum,
                                                 int *atomic_dst,
                                                 const unsigned char *src_u8,
                                                 const uint16_t *src_u16,
                                                 int bias) {
    __shared__ __align__(16) unsigned char smem[2048];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base = combo_profile_depth_safe_smem_addr(smem) + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 512u + lane * 16u;
    const unsigned char *gptr2 = src_u8 + 1024u + lane * 16u;
    const unsigned char *gptr3 = src_u8 + 1536u + lane * 16u;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(base + 0u), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(base + 512u), "l"(gptr1));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(base + 1024u), "l"(gptr2));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(base + 1536u), "l"(gptr3));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    unsigned sm2 = 0;
    unsigned sm3 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base + 0u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base + 512u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm2) : "r"(base + 1024u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm3) : "r"(base + 1536u));

    const unsigned char u8 = src_u8[lane * 16u];
    const uint16_t u16 = src_u16[lane];

    unsigned chain = (unsigned)u8 + (unsigned)u16 + (unsigned)bias + lane;
    chain ^= (sm0 & 0xffu) + ((sm1 & 0xffu) << 1);
    chain = chain * 33u + (sm2 & 0xffu);
    chain ^= (chain >> 7) + (sm3 & 0xffu);
    chain = chain * 17u + ((sm0 >> 8) & 0xffu);
    chain ^= (chain >> 11) + ((sm1 >> 8) & 0xffu);

    const int lane_val = (int)(chain & 0xffu);
    const unsigned ballot = __ballot_sync(0xffffffffu, (lane_val & 1) != 0);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const int vote_any = __any_sync(0xffffffffu, lane_val > 96);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int warp_min = __reduce_min_sync(0xffffffffu, lane_val);
    const int warp_max = __reduce_max_sync(0xffffffffu, lane_val);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int pop = __popc(ballot);
    const int leader = __ffs((int)group) - 1;

    float x = (float)(warp_sum + warp_min + warp_max + pop + leader + vote_all + vote_any + (int)lane);
    x += (float)((chain >> 8) & 0xffu) * 0.125f;
    if ((bias & 1) != 0) x *= 1.0001f;
    if ((bias & 2) != 0) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)((ballot & 0xffu) + (group & 0xffu)));
    out[lane] = x;
}
