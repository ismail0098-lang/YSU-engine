/*
 * SASS RE Probe: block-red + system-scope RED + async/cache "wombo combo"
 *
 * Goal: preserve the established async/cache backbone:
 *   - LDG(.STRONG.GPU)
 *   - LDGSTS.E.BYPASS.LTC128B.128(.ZFILL)
 *   - LDGDEPBAR + DEPBAR.LE
 *
 * while testing whether unused system-scope atomics can coexist with block-
 * level barrier reductions in the same emitted kernel:
 *   - RED.E.MIN/MAX.S32.STRONG.SYS
 *   - RED.E.ADD(.F32).STRONG.SYS
 *   - BAR.RED.POPC/AND/OR.DEFER_BLOCKING
 *   - B2R.RESULT
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_redsys_blockred_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(128)
probe_combo_redsys_blockred_cache_wombo(int *out,
                                        int *sys_min_dst,
                                        int *sys_max_dst,
                                        int *sys_add_dst,
                                        float *sys_fadd_dst,
                                        const unsigned char *src_u8,
                                        const uint16_t *src_u16,
                                        int bias) {
    __shared__ __align__(16) unsigned char smem[256];
    const unsigned lane = (unsigned)threadIdx.x;
    const unsigned saddr = combo_redsys_blockred_smem_addr(smem) + ((lane & 15u) << 4);

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

    if ((lane & 3u) == 0u) {
        atomicMin_system(sys_min_dst, v);
        atomicMax_system(sys_max_dst, v ^ 0x33);
        atomicAdd_system(sys_add_dst, (v & 0xff) + block_popc);
        atomicAdd_system(sys_fadd_dst, (float)((v & 0x7f) + block_or));
    }

    out[lane] = v + block_popc + block_and + block_or;
}
