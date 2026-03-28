/*
 * SASS RE Probe: explicit cp.async zero-fill and ignore-src lowering
 *
 * Confirms the LDGSTS zero-fill spellings by using inline PTX with compile-time
 * immediate cp-size/src-size operands and statically aligned shared memory.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_zfill_4_2(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr(smem);
    unsigned int r0 = 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 2;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(dst));
    ((unsigned int *)out)[0] = r0;
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_zfill_8_4(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr(smem);
    unsigned int r0 = 0, r1 = 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 4;" :: "r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(dst));
    ((unsigned int *)out)[0] = r0;
    ((unsigned int *)out)[1] = r1;
}

extern "C" __global__ void __launch_bounds__(32)
probe_cp_async_zfill_16_8(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 8;" :: "r"(dst), "l"(src));
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
probe_cp_async_ignore_src(unsigned char *out, const unsigned char *src, int ignore_src) {
    __shared__ __align__(16) unsigned char smem[32];
    unsigned dst = smem_addr(smem);
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.s32 p, %2, 0;\n\t"
        "cp.async.cg.shared.global [%0], [%1], 16, p;\n\t"
        "}\n"
        :
        : "r"(dst), "l"(src), "r"(ignore_src));
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
probe_cp_async_misaligned(unsigned char *out, const unsigned char *src) {
    __shared__ __align__(16) unsigned char smem_raw[48];
    unsigned char *smem = smem_raw + 1;
    unsigned dst = smem_addr(smem);
    const unsigned char *misaligned = src + 1;
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 8;" :: "r"(dst), "l"(misaligned));
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
