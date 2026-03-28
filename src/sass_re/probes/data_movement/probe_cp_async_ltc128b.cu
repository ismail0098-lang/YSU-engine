/*
 * SASS RE Probe: cp.async with L2::128B prefetch hint
 *
 * Directly tests whether Ada emits the cuDNN-mined LDGSTS.*.LTC128B.128
 * spellings when inline PTX forces the .L2::128B prefetch-size qualifier.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned smem_addr_ltc(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_ca_ltc128b_16(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr_ltc(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(dst));
    ((unsigned int *)out)[0] = r0;
    ((unsigned int *)out)[1] = r1;
    ((unsigned int *)out)[2] = r2;
    ((unsigned int *)out)[3] = r3;
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_ca_ltc128b_16_8(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr_ltc(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, 8;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(dst));
    ((unsigned int *)out)[0] = r0;
    ((unsigned int *)out)[1] = r1;
    ((unsigned int *)out)[2] = r2;
    ((unsigned int *)out)[3] = r3;
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_cg_ltc128b_16(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr_ltc(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(dst));
    ((unsigned int *)out)[0] = r0;
    ((unsigned int *)out)[1] = r1;
    ((unsigned int *)out)[2] = r2;
    ((unsigned int *)out)[3] = r3;
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_cg_ltc128b_16_8(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr_ltc(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(dst));
    ((unsigned int *)out)[0] = r0;
    ((unsigned int *)out)[1] = r1;
    ((unsigned int *)out)[2] = r2;
    ((unsigned int *)out)[3] = r3;
}
