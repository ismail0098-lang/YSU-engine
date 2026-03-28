/*
 * SASS RE Probe: atomic/reduction + async/cache "wombo combo"
 *
 * Goal: force coexistence of:
 *   - ATOMG/RED strong-GPU forms
 *   - LDG(.STRONG.GPU) subword loads
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_atomic_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomic_cache_wombo(float *out,
                               float *accum,
                               int *atomic_dst,
                               const unsigned char *src_u8,
                               const uint16_t *src_u16,
                               int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_atomic_smem_addr(smem);
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

    float x = (float)(u8 + u16 + (sm0 & 0xffu) + (sm1 & 0xffu) + bias + lane);
    if (bias & 1) x *= 1.0001f;
    if (bias & 2) x += 0.25f;

    atomicAdd(accum, x);
    atomicAdd(atomic_dst, (int)(u8 + (u16 & 0xffu)));
    out[lane] = x;
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomic_minmax_cache_wombo(int *out,
                                      int *min_dst,
                                      int *max_dst,
                                      int *sum_dst,
                                      const unsigned char *src_u8,
                                      const uint16_t *src_u16,
                                      int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_atomic_smem_addr(smem);
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
    if ((bias & 1) != 0) v ^= 0x55;
    if ((bias & 2) != 0) v += 17;

    int old_sum = atomicAdd(sum_dst, v);
    atomicMin(min_dst, v);
    atomicMax(max_dst, v ^ lane);
    out[lane] = old_sum + v;
}
