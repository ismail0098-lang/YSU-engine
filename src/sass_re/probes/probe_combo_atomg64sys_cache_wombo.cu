/*
 * SASS RE Probe: 64-bit system-scope ATOMG + async/cache "wombo combo"
 *
 * Goal: test whether return-valued 64-bit system atomics can join the same
 * local family already carrying:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 * while adding:
 *   - ATOMG.E.CAS.64.STRONG.SYS or adjacent 64-bit system CAS lowering
 *   - ATOMG.E.EXCH.64.STRONG.SYS or adjacent 64-bit system EXCH lowering
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_atomg64sys_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomg64sys_cache_wombo(unsigned long long *out,
                                   unsigned long long *sys_cas_dst,
                                   unsigned long long *sys_exch_dst,
                                   const unsigned char *src_u8,
                                   const unsigned long long *src_u64,
                                   unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr = combo_atomg64sys_smem_addr(smem);

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
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane;

    unsigned long long old_cas = 0ull;
    unsigned long long old_exch = 0ull;

    if ((lane & 3u) == 0u) {
        old_cas = atomicCAS_system(sys_cas_dst, v64 ^ 0x55ull, v64 | 1ull);
        old_exch = atomicExch_system(sys_exch_dst, (v64 << 1) | 1ull);
        __threadfence_system();
    }

    out[lane] = v64 ^ old_cas ^ old_exch;
}
