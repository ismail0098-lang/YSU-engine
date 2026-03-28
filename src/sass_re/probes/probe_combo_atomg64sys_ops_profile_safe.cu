/*
 * SASS RE Probe: runtime-safe surrogate for the narrow 64-bit SYS ATOMG matrix
 *
 * Goal: preserve the narrower async/cache + 64-bit system ATOMG op-matrix
 * family without the extra block-red / warp neighborhood so it can be
 * compared directly against the larger SYS-safe branches.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned
combo_atomg64sys_ops_profile_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomg64sys_ops_profile_safe(
    unsigned long long *out,
    unsigned long long *add_dst,
    unsigned long long *min_dst,
    unsigned long long *max_dst,
    unsigned long long *and_dst,
    unsigned long long *or_dst,
    unsigned long long *xor_dst,
    const unsigned char *src_u8,
    const unsigned long long *src_u64,
    unsigned long long bias) {
    __shared__ __align__(16) unsigned char smem[1024];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base =
        combo_atomg64sys_ops_profile_smem_addr(smem) + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 512u + lane * 16u;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base + 0u), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base + 512u), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base + 0u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base + 512u));

    const unsigned long long v64 =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane;

    unsigned long long acc = v64;
    if ((lane & 7u) == 0u) {
        const unsigned long long old_add = atomicAdd_system(add_dst, (v64 | 1ull));
        const unsigned long long old_min =
            atomicMin_system(min_dst, v64 ^ 0x1111111111111111ull);
        const unsigned long long old_max =
            atomicMax_system(max_dst, v64 ^ 0x2222222222222222ull);
        const unsigned long long old_and =
            atomicAnd_system(and_dst, v64 | 0xff00ff00ff00ff00ull);
        const unsigned long long old_or =
            atomicOr_system(or_dst, v64 | 0x00ff00ff00ff00ffull);
        const unsigned long long old_xor =
            atomicXor_system(xor_dst, v64 ^ 0x3333333333333333ull);

        acc ^= old_add ^ old_min ^ old_max ^ old_and ^ old_or ^ old_xor;
        acc += ((old_add & 0xffull) << 7) ^ ((old_xor & 0xffull) << 15);
        __threadfence_system();
    }

    out[lane] = acc;
}
