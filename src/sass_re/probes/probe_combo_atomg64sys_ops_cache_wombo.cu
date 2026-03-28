/*
 * SASS RE Probe: 64-bit system-scope ATOMG op matrix + async/cache backbone
 *
 * Goal: determine which return-valued 64-bit system atomics can survive inside
 * the established 64-bit async/cache family:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 * while trying to land:
 *   - ATOMG.E.ADD.64.STRONG.SYS
 *   - ATOMG.E.MIN.64.STRONG.SYS
 *   - ATOMG.E.MAX.64.STRONG.SYS
 *   - ATOMG.E.AND.64.STRONG.SYS
 *   - ATOMG.E.OR.64.STRONG.SYS
 *   - ATOMG.E.XOR.64.STRONG.SYS
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_atomg64sys_ops_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomg64sys_ops_cache_wombo(unsigned long long *out,
                                       unsigned long long *add_dst,
                                       unsigned long long *min_dst,
                                       unsigned long long *max_dst,
                                       unsigned long long *and_dst,
                                       unsigned long long *or_dst,
                                       unsigned long long *xor_dst,
                                       const unsigned char *src_u8,
                                       const unsigned long long *src_u64,
                                       unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr = combo_atomg64sys_ops_smem_addr(smem);

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

    unsigned long long acc = v64;
    if ((lane & 7u) == 0u) {
        acc ^= atomicAdd_system(add_dst, (v64 | 1ull));
        acc ^= atomicMin_system(min_dst, v64 ^ 0x1111111111111111ull);
        acc ^= atomicMax_system(max_dst, v64 ^ 0x2222222222222222ull);
        acc ^= atomicAnd_system(and_dst, v64 | 0xff00ff00ff00ff00ull);
        acc ^= atomicOr_system(or_dst, v64 | 0x00ff00ff00ff00ffull);
        acc ^= atomicXor_system(xor_dst, v64 ^ 0x3333333333333333ull);
        __threadfence_system();
    }

    out[lane] = acc;
}
