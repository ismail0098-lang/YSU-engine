/*
 * SASS RE Probe: exact stage-mask uniform control + system-RED + async/cache
 *
 * Goal: copy the strongest known local UISETP-producing source shape from the
 * uniform HMMA toggle probe, but replace the HMMA body with the already-proven
 * async/cache + system-RED family.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned combo_stage_redsys_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_combo_uniform_stage_redsys_wombo(int *out,
                                       int *sys_min_dst,
                                       int *sys_max_dst,
                                       int *sys_add_dst,
                                       float *sys_fadd_dst,
                                       const unsigned char *src_u8,
                                       const uint16_t *src_u16,
                                       uint64_t byte_stride,
                                       int tiles,
                                       int stage_mask,
                                       int tail_mask,
                                       int bias) {
    __shared__ __align__(16) unsigned char smem[2][32];
    const int lane = threadIdx.x;
    const unsigned base0 = combo_stage_redsys_smem_addr(smem[0]);
    const unsigned base1 = combo_stage_redsys_smem_addr(smem[1]);
    uint64_t byte_off = 0;
    int acc = bias + lane;

    #pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
        const int stage = t & 1;
        const int next_stage = stage ^ 1;
        const unsigned cur_dst = stage ? base1 : base0;
        const unsigned nxt_dst = next_stage ? base1 : base0;
        const unsigned char *gptr = src_u8 + byte_off;

        const bool gate40 = (stage_mask & 0x40) != 0;
        const bool gate80 = (stage_mask & 0x80) != 0;
        const bool tail0 = (tail_mask & 0x40) != 0;
        const bool tail1 = (tail_mask & 0x80) != 0;
        const bool loop0 = ((t + stage_mask) & 1) == 0;
        const bool loop1 = ((t + tail_mask) & 2) != 0;
        const bool do_prefetch = (t + 1) < tiles;
        const bool gate_a = (gate40 && !loop0) || (tail0 && loop1);
        const bool gate_b = (gate80 && loop0) || (tail1 && !loop1);

        if (!tail0 || !tail1) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                         :: "r"(cur_dst), "l"(gptr + lane));
            if (do_prefetch) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                             :: "r"(nxt_dst), "l"(gptr + byte_stride + lane));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
        }

        const uint16_t u16 = src_u16[(byte_off / 2 + (uint64_t)lane) & 31ull];
        unsigned sm0 = 0;
        unsigned sm1 = 0;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm0) : "r"(cur_dst));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sm1) : "r"(nxt_dst));

        int v = acc + (int)(u16 & 0xffu) + (int)(sm0 & 0xffu);
        if (gate_a) v ^= (int)(sm1 & 0xffu);
        if (gate_b) v += (int)((sm1 >> 8) & 0xffu);

        if (gate_a || gate40) atomicMin_system(sys_min_dst, v);
        if (gate_b || gate80) atomicMax_system(sys_max_dst, v ^ 0x33);
        if (gate40 && tail0) atomicAdd_system(sys_add_dst, (v & 0xff) + t);
        if (gate80 || tail1) atomicAdd_system(sys_fadd_dst, (float)(v & 0x7f));

        acc = v + (gate_a ? 1 : 0) + (gate_b ? 2 : 0);
        byte_off += ((uint64_t)(gate40 ? 2 : 1)) * byte_stride;
        asm volatile("" : "+l"(byte_off));
    }

    out[lane] = acc;
}
