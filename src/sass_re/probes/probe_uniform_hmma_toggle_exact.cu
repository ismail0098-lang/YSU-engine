/*
 * SASS RE Probe: exact uniform HMMA toggle follow-up
 *
 * Goal: stay close to the cuDNN-mined UPLOP3.LUT neighborhood:
 *   uniform control from scalar args, explicit 0x40 / 0x80 stage bits,
 *   repeated HMMA blocks predicated by a long-lived uniform gate, and
 *   surrounding ULDC / USEL / UISETP / cp.async scaffolding.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned probe_toggle_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_hmma_toggle_exact(float *out,
                                const float *a_in,
                                const float *b_in,
                                const unsigned char *src,
                                int tiles,
                                uint64_t byte_stride,
                                int stage_mask,
                                int tail_mask) {
    __shared__ __align__(16) unsigned char smem[2][32];
    int lane = threadIdx.x;

    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    unsigned b0 = 0, b1 = 0;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a0) : "f"(a_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a1) : "f"(a_in[lane + 1 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a2) : "f"(a_in[lane + 2 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a3) : "f"(a_in[lane + 3 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b0) : "f"(b_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b1) : "f"(b_in[lane + 1 * 32]));

    const unsigned base0 = probe_toggle_smem_addr(smem[0]);
    const unsigned base1 = probe_toggle_smem_addr(smem[1]);
    uint64_t byte_off = 0;

    #pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
        int stage = t & 1;
        int next_stage = stage ^ 1;
        const unsigned cur_dst = stage ? base1 : base0;
        const unsigned nxt_dst = next_stage ? base1 : base0;
        const unsigned char *gptr = src + byte_off;

        bool gate40 = (stage_mask & 0x40) != 0;
        bool gate80 = (stage_mask & 0x80) != 0;
        bool tail0 = (tail_mask & 0x40) != 0;
        bool tail1 = (tail_mask & 0x80) != 0;
        bool loop0 = ((t + stage_mask) & 1) == 0;
        bool loop1 = ((t + tail_mask) & 2) != 0;
        bool do_prefetch = (t + 1) < tiles;
        bool gate_a = (gate40 && !loop0) || (tail0 && loop1);
        bool gate_b = (gate80 && loop0) || (tail1 && !loop1);

        if (!tail0 || !tail1) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                         :: "r"(cur_dst), "l"(gptr));
            if (do_prefetch) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                             :: "r"(nxt_dst), "l"(gptr + byte_stride));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
        }

        if (gate_a || gate_b) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }

        if (gate40) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }

        if (gate80) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }

        if (gate40 && tail0) d0 += (float)smem[stage][0];
        if (gate80 && tail1) d1 += (float)smem[next_stage][1];
        if (gate_a && !gate_b) d2 += (float)smem[stage][2];
        if (gate_b && !gate_a) d3 += (float)smem[next_stage][3];

        byte_off += ((uint64_t)(gate40 ? 2 : 1)) * byte_stride;
        asm volatile("" : "+l"(byte_off));
    }

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1;
    out[lane + 2 * 32] = d2;
    out[lane + 3 * 32] = d3;
}
