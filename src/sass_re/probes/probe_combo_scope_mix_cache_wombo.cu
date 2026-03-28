/*
 * SASS RE Probe: scope-mix + async/cache "wombo combo"
 *
 * Goal: keep the proven async/cache backbone while attempting to pull system-
 * scope atomics into the same emitted family.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_scope_mix_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_scope_mix_cache_wombo(int *out,
                                  int *sys_exch_dst,
                                  int *sys_cas_dst,
                                  float *sys_add_dst,
                                  const unsigned char *src_u8,
                                  const uint16_t *src_u16,
                                  int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    const unsigned saddr = combo_scope_mix_smem_addr(smem);
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

    int old_exch = atomicExch_system(sys_exch_dst, v ^ 0x55);
    int old_cas = atomicCAS_system(sys_cas_dst, old_exch, v ^ 0x33);
    float old_add = atomicAdd_system(sys_add_dst, (float)(v & 0xff));

    out[lane] = v + old_exch + old_cas + (int)old_add;
}
