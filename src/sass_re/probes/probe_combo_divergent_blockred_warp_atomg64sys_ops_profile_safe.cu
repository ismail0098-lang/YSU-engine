/*
 * SASS RE Probe: runtime-safe surrogate for the divergent fused 64-bit SYS family
 *
 * Goal: preserve the reconvergence-shaped branch that adds BSSY/BSYNC to the
 * strongest fused async/cache + block-red + warp + 64-bit SYS atomic family,
 * while keeping the cp.async staging fully aligned and runtime-safe.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned
combo_div_blockred_warp_atomg64sys_profile_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

static __device__ __noinline__ int
combo_div_blockred_warp_profile_stage(int lane, int v, int bias) {
    if ((lane & 1) == 0) {
        v = v * 3 + bias;
        if ((lane & 4) != 0) {
            v ^= 0x55;
        } else {
            v += 17;
        }
    } else {
        v = v * 5 - bias;
        if ((lane & 8) != 0) {
            v ^= 0xaa;
        } else {
            v -= 23;
        }
    }

    if ((lane & 3) == 1) {
        v += (lane << 2);
    } else if ((lane & 3) == 2) {
        v ^= (lane << 1);
    }

    return v;
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_divergent_blockred_warp_atomg64sys_ops_profile_safe(
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
    __shared__ __align__(16) unsigned char smem[4096];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base =
        combo_div_blockred_warp_atomg64sys_profile_smem_addr(smem) + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 2048u + lane * 16u;

    unsigned ballot_pre = __ballot_sync(0xffffffffu, ((lane + (unsigned)bias) & 1u) != 0u);
    int dv = combo_div_blockred_warp_profile_stage((int)lane, (int)(lane + (unsigned)bias), (int)bias);
    __syncwarp(ballot_pre | 1u);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base + 0u), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base + 2048u), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base + 0u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base + 2048u));

    const unsigned long long v64 =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane ^
        (unsigned long long)(dv & 0xff);

    const int pred_odd = (int)((v64 & 1ull) != 0ull);
    const int pred_small = (int)((v64 & 0xffull) < 192ull);
    const int pred_nonzero = (int)((v64 & 0xffull) != 0ull);

    const int block_popc = __syncthreads_count(pred_odd);
    const int block_and = __syncthreads_and(pred_small);
    const int block_or = __syncthreads_or(pred_nonzero);

    const int lane_val = (int)(v64 & 0xffull);
    const unsigned ballot = __ballot_sync(0xffffffffu, (lane_val & 1) != 0);
    const int vote_all = __all_sync(0xffffffffu, lane_val < 224);
    const unsigned group = __match_any_sync(0xffffffffu, lane_val & 7);
    const int warp_min = __reduce_min_sync(0xffffffffu, lane_val);
    const int warp_max = __reduce_max_sync(0xffffffffu, lane_val);
    const int warp_sum = __reduce_add_sync(0xffffffffu, lane_val);
    const int leader = __ffs((int)group) - 1;

    unsigned long long acc =
        v64 ^
        (unsigned long long)block_popc ^
        ((unsigned long long)block_and << 8) ^
        ((unsigned long long)block_or << 16) ^
        (unsigned long long)(ballot & 0xffu) ^
        ((unsigned long long)(vote_all & 1) << 24) ^
        ((unsigned long long)(leader & 0xff) << 32) ^
        ((unsigned long long)(warp_min & 0xff) << 40) ^
        ((unsigned long long)(warp_max & 0xff) << 48) ^
        ((unsigned long long)(warp_sum & 0xff) << 56);

    if ((lane & 7u) == 0u) {
        const unsigned long long old_add = atomicAdd_system(add_dst, acc | 1ull);
        const unsigned long long old_min =
            atomicMin_system(min_dst, acc ^ 0x1111111111111111ull);
        const unsigned long long old_max =
            atomicMax_system(max_dst, acc ^ 0x2222222222222222ull);
        const unsigned long long old_and =
            atomicAnd_system(and_dst, acc | 0xff00ff00ff00ff00ull);
        const unsigned long long old_or =
            atomicOr_system(or_dst, acc | 0x00ff00ff00ff00ffull);
        const unsigned long long old_xor =
            atomicXor_system(xor_dst, acc ^ 0x3333333333333333ull);
        acc ^= old_add ^ old_min ^ old_max ^ old_and ^ old_or ^ old_xor;
        __threadfence_system();
    }

    out[lane] = acc;
}
