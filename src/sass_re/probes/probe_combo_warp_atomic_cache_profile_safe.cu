/*
 * SASS RE Probe: runtime-safe warp + atomic + async/cache profile surrogate
 *
 * Goal: preserve the key emitted family:
 *   - LDGSTS(.ZFILL) + LDGDEPBAR + DEPBAR
 *   - MATCH / VOTE / REDUX / POPC
 *   - RED.E.ADD(.F32) / RED.E.ADD
 * while making the cp.async path execution-safe by keeping every per-lane
 * global source aligned to 16 bytes.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_profile_safe_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_warp_atomic_cache_profile_safe(float *out,
                                           float *accum,
                                           int *atomic_dst,
                                           const unsigned char *src_u8,
                                           const uint16_t *src_u16,
                                           int bias) {
    __shared__ __align__(16) unsigned char smem[1024];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr0 = combo_profile_safe_smem_addr(smem) + lane * 16u;
    const unsigned saddr1 = combo_profile_safe_smem_addr(smem) + 512u + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 512u + lane * 16u;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr0), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr1), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr1));

    const unsigned char u8 = src_u8[lane * 16u];
    const uint16_t u16 = src_u16[lane];
    const unsigned packed =
        (unsigned)u8 +
        (unsigned)u16 +
        (sm0 & 0xffu) +
        (sm1 & 0xffu) +
        (unsigned)bias +
        lane;

    const int lane_val = (int)(packed & 0xffu);
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
    if ((bias & 1) != 0) x *= 1.0001f;
    if ((bias & 2) != 0) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)((ballot & 0xffu) + (group & 0xffu)));
    out[lane] = x;
}
