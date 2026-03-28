/*
 * SASS RE Probe: 64-bit system-RED + async/cache "wombo combo"
 *
 * Goal: extend the current scope-mix family into the 64-bit reduction space:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL) + LDGDEPBAR + DEPBAR.LE
 *   - RED.E.ADD.64.STRONG.SYS
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_red64sys_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_red64sys_cache_wombo(unsigned long long *out,
                                 unsigned long long *sys_add64_dst,
                                 const unsigned char *src_u8,
                                 const unsigned long long *src_u64,
                                 unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_red64sys_smem_addr(smem);
    const unsigned lane = (unsigned)threadIdx.x;

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

    if ((lane & 3u) == 0u) {
        atomicAdd_system(sys_add64_dst, v64 | 1ull);
    }

    out[lane] = v64;
}
