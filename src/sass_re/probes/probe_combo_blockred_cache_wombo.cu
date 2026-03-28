/*
 * SASS RE Probe: block-reduction + async/cache "wombo combo"
 *
 * Goal: keep the proven async/cache backbone:
 *   - LDG(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 *
 * while trying to pull block-level barrier reductions into the same emitted
 * kernel:
 *   - BAR.RED.POPC
 *   - BAR.RED.AND
 *   - BAR.RED.OR
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_blockred_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_blockred_cache_wombo(int *out,
                                 int *atomic_dst,
                                 const unsigned char *src_u8,
                                 const uint16_t *src_u16,
                                 int bias) {
    __shared__ __align__(16) unsigned char smem[256];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr = combo_blockred_smem_addr(smem) + ((lane & 15u) << 4);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                 :: "r"(saddr), "l"(src_u8 + lane));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                 :: "r"(saddr + 128), "l"(src_u8 + lane + 32));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

    const unsigned char u8 = src_u8[lane];
    const uint16_t u16 = src_u16[lane];
    unsigned sm0 = 0;
    unsigned sm1 = 0;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(saddr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(saddr + 128));

    const int v = (int)((unsigned)u8 + (unsigned)u16 +
                        (sm0 & 0xffu) + (sm1 & 0xffu) + (unsigned)bias + lane);
    const int pred_odd = (v & 1) != 0;
    const int pred_small = v < 224;
    const int pred_nonzero = (v & 0xff) != 0;

    int block_popc = __syncthreads_count(pred_odd);
    int block_and = __syncthreads_and(pred_small);
    int block_or = __syncthreads_or(pred_nonzero);

    if ((lane & 31u) == 0u) {
        atomicAdd(atomic_dst, block_popc + block_and + block_or);
    }

    out[lane] = v + block_popc + block_and + block_or;
}
