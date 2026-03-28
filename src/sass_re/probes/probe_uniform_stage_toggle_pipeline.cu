/*
 * SASS RE Probe: software-pipelined uniform stage toggles
 *
 * cuDNN-mined UPLOP3.LUT sites look like uniform stage toggles wrapped around
 * HMMA strips, cp.async traffic, and uniform pointer rebasing. This probe
 * builds a small double-buffered mainloop with warp-uniform stage bits, shared
 * pointer rebasing, and alternating HMMA work to see whether local sm_89 will
 * surface UPLOP3.LUT instead of only PLOP3/UIADD3 scaffolding.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned stage_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_stage_toggle_pipeline(float *out,
                                    const float *a_in,
                                    const float *b_in,
                                    const unsigned char *src,
                                    int tiles,
                                    int stride) {
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

    const unsigned base0 = stage_smem_addr(smem[0]);
    const unsigned base1 = stage_smem_addr(smem[1]);

    #pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
        int stage = t & 1;
        int next_stage = stage ^ 1;
        int wrap = ((t & 3) == 3);
        const unsigned char *gptr = src + t * stride;
        unsigned dst = stage ? base1 : base0;
        unsigned next_dst = next_stage ? base1 : base0;

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                     :: "r"(dst), "l"(gptr));
        if (t + 1 < tiles) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                         :: "r"(next_dst), "l"(gptr + stride));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

        if (stage == 0) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        } else {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }

        if (wrap) {
            d0 += static_cast<float>(smem[stage][0]);
            d1 += static_cast<float>(smem[next_stage][1]);
        } else {
            d2 += static_cast<float>(smem[stage][2]);
            d3 += static_cast<float>(smem[next_stage][3]);
        }
    }

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1;
    out[lane + 2 * 32] = d2;
    out[lane + 3 * 32] = d3;
}
