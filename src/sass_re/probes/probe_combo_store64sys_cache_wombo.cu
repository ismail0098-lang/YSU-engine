/*
 * SASS RE Probe: 64-bit system load/store + async/cache "wombo combo"
 *
 * Goal: test whether the same 64-bit async/cache backbone can carry the
 * system-visible load/store control neighborhood:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 *   - LDG.E.STRONG.SYS
 *   - STG.E.STRONG.SYS or STG.E.64 plus MEMBAR/ERRBAR/CCTL
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_store64sys_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_store64sys_cache_wombo(unsigned long long *out,
                                   volatile unsigned long long *sys_dst,
                                   volatile const unsigned long long *sys_src,
                                   const unsigned char *src_u8,
                                   const unsigned long long *src_u64,
                                   unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr = combo_store64sys_smem_addr(smem);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 32), "l"(src_u8 + lane + 16));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 32));

    const unsigned long long v64 =
        src_u64[lane] +
        (unsigned long long)(sm0 & 0xffu) +
        ((unsigned long long)(sm1 & 0xffu) << 8) +
        bias + (unsigned long long)lane;

    unsigned long long observed = 0ull;
    if ((lane & 3u) == 0u) {
        observed = sys_src[0];
        sys_dst[0] = observed ^ v64 ^ 0x0101010101010101ull;
        __threadfence_system();
        observed ^= sys_src[0];
    }

    out[lane] = v64 ^ observed;
}
