/*
 * SASS RE Probe: divergence/reconvergence + async/cache "wombo combo"
 *
 * Goal: create genuine lane-divergent work and explicit warp resynchronization
 * before the proven async/cache backbone, then see whether the emitted family
 * gains reconvergence helpers such as WARPSYNC/BSSY/BSYNC.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_div_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

static __device__ __noinline__ int combo_divergent_stage(int lane, int v, int bias) {
    if ((lane & 1) == 0) {
        v = v * 3 + bias;
        if ((lane & 4) != 0) {
            v ^= 0x55;
        } else {
            v += 17;
        }
    } else {
        v = v * 5 - bias;
        if ((lane & 8) != 0) {
            v ^= 0xaa;
        } else {
            v -= 23;
        }
    }

    if ((lane & 3) == 1) {
        v += (lane << 2);
    } else if ((lane & 3) == 2) {
        v ^= (lane << 1);
    }

    return v;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_divergence_cache_wombo(float *out,
                                   int *accum,
                                   const unsigned char *src_u8,
                                   const uint16_t *src_u16,
                                   int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_div_smem_addr(smem);
    const int lane = threadIdx.x;

    unsigned ballot = __ballot_sync(0xffffffffu, ((lane + bias) & 1) != 0);
    int v = combo_divergent_stage(lane, lane + bias, bias);
    __syncwarp(ballot | 1u);

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

    unsigned group = __match_any_sync(0xffffffffu, (unsigned)(v & 3));
    int warp_sum = __reduce_add_sync(0xffffffffu, (v + (int)u8 + (int)(u16 & 0xff)));
    int leader = __ffs((int)group) - 1;

    float x = (float)(v + warp_sum + leader + (int)(sm0 & 0xffu) + (int)(sm1 & 0xffu));
    atomicAdd(accum, (int)(group & 0xffu));
    out[lane] = x;
}
