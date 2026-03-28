/*
 * SASS RE Probe: runtime-safe surrogate for the fused 64-bit SYS family
 *
 * Goal: preserve as much as possible of the strongest symbolic family:
 *   - LDG.E.64(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 *   - BAR.RED.* + B2R.RESULT
 *   - MATCH / REDUX / VOTE
 *   - dense ATOMG.E.*.64.STRONG.SYS block
 * while fixing the execution hazard by forcing all cp.async source addresses
 * to be 16-byte aligned.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_blockred_warp_atomg64sys_profile_safe_smem_addr(
    unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_blockred_warp_atomg64sys_ops_profile_safe(
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
    __shared__ __align__(16) unsigned char smem[2048];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned base = combo_blockred_warp_atomg64sys_profile_safe_smem_addr(smem) + lane * 16u;
    const unsigned char *gptr0 = src_u8 + lane * 16u;
    const unsigned char *gptr1 = src_u8 + 2048u + lane * 16u;

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(base + 0u), "l"(gptr0));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(base + 1024u), "l"(gptr1));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(base + 0u));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(base + 1024u));

    const unsigned long long v64 =
        src_u64[lane] ^
        ((unsigned long long)(sm0 & 0xffu) << 8) ^
        ((unsigned long long)(sm1 & 0xffu) << 24) ^
        bias ^ (unsigned long long)lane;

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
        acc ^= atomicAdd_system(add_dst, acc | 1ull);
        acc ^= atomicMin_system(min_dst, acc ^ 0x1111111111111111ull);
        acc ^= atomicMax_system(max_dst, acc ^ 0x2222222222222222ull);
        acc ^= atomicAnd_system(and_dst, acc | 0xff00ff00ff00ff00ull);
        acc ^= atomicOr_system(or_dst, acc | 0x00ff00ff00ff00ffull);
        acc ^= atomicXor_system(xor_dst, acc ^ 0x3333333333333333ull);
        __threadfence_system();
    }

    out[lane] = acc;
}
