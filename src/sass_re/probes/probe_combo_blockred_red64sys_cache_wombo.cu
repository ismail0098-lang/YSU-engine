/*
 * SASS RE Probe: block-red + 64-bit system-RED + async/cache "wombo combo"
 *
 * Goal: extend the async/cache scope-mix family so one emitted kernel carries:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 *   - BAR.RED.POPC/AND/OR.DEFER_BLOCKING
 *   - B2R.RESULT
 *   - RED.E.ADD.64.STRONG.SYS
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_blockred_red64sys_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_blockred_red64sys_cache_wombo(unsigned long long *out,
                                          unsigned long long *sys_add64_dst,
                                          const unsigned char *src_u8,
                                          const unsigned long long *src_u64,
                                          unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[256];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr =
        combo_blockred_red64sys_smem_addr(smem) + ((lane & 15u) << 4);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 128), "l"(src_u8 + lane + 32));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const unsigned long long v64 = src_u64[lane];
    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 128));

    const unsigned long long mixed =
        v64 +
        (unsigned long long)(sm0 & 0xffu) +
        ((unsigned long long)(sm1 & 0xffu) << 8) +
        bias + (unsigned long long)lane;

    const int pred_odd = (int)((mixed & 1ull) != 0ull);
    const int pred_small = (int)((mixed & 0xffull) < 224ull);
    const int pred_nonzero = (int)((mixed & 0xffull) != 0ull);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    if ((lane & 3u) == 0u) {
        atomicAdd_system(sys_add64_dst,
                         (mixed | 1ull) +
                         (unsigned long long)block_popc +
                         ((unsigned long long)block_and << 8) +
                         ((unsigned long long)block_or << 16));
    }

    out[lane] =
        mixed +
        (unsigned long long)block_popc +
        (unsigned long long)block_and +
        (unsigned long long)block_or;
}
