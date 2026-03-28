/*
 * SASS RE Probe: mixed ATOMG non-add ops + async/cache "wombo combo"
 *
 * Goal: keep the proven async/cache backbone alive while forcing return-valued
 * global atomic ops so the compiler has a chance to emit the wider `ATOMG`
 * family in the same kernel:
 *   - ATOMG.E.MIN/MAX/AND/OR/XOR/EXCH/CAS.STRONG.GPU
 *   - LDG(.STRONG.GPU) + LDGSTS(.ZFILL) + LDGDEPBAR + DEPBAR.LE
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_atomg_nonadd_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomg_nonadd_cache_wombo(int *out,
                                     int *min_dst,
                                     int *max_dst,
                                     int *and_dst,
                                     int *or_dst,
                                     int *xor_dst,
                                     int *exch_dst,
                                     int *cas_dst,
                                     const unsigned char *src_u8,
                                     const uint16_t *src_u16,
                                     int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_atomg_nonadd_smem_addr(smem);
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

    int v = (int)((unsigned)u8 + (unsigned)u16 +
                  (sm0 & 0xffu) + (sm1 & 0xffu) + (unsigned)bias + (unsigned)lane);
    int m = (v ^ 0x55aa) | 1;

    int old_min = atomicMin(min_dst, v);
    int old_max = atomicMax(max_dst, v ^ lane);
    int old_and = atomicAnd(and_dst, m);
    int old_or = atomicOr(or_dst, m << 1);
    int old_xor = atomicXor(xor_dst, m << 2);
    int old_exch = atomicExch(exch_dst, v + 17);
    int old_cas = atomicCAS(cas_dst, old_exch, v ^ 0x33);

    out[lane] = old_min + old_max + old_and + old_or + old_xor + old_exch + old_cas + v;
}
