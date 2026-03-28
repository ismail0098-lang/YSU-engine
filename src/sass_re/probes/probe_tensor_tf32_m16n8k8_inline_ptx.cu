/*
 * SASS RE Probe: direct mma.sync.m16n8k8 TF32 inline PTX
 *
 * cuDNN library mining first surfaced HMMA.1688.F32.TF32 near Ada. This probe
 * bypasses WMMA wrappers and directly forces the m16n8k8 TF32 PTX form so the
 * spelling can be confirmed on local sm_89.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_mma_tf32_m16n8k8_once(float *out,
                            const float *a_in,
                            const float *b_in,
                            const float *c_in) {
    int lane = threadIdx.x;
    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    unsigned b0 = 0, b1 = 0;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float c0 = c_in[lane + 0 * 32];
    float c1 = c_in[lane + 1 * 32];
    float c2 = c_in[lane + 2 * 32];
    float c3 = c_in[lane + 3 * 32];

    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a0) : "f"(a_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a1) : "f"(a_in[lane + 1 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a2) : "f"(a_in[lane + 2 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a3) : "f"(a_in[lane + 3 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b0) : "f"(b_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b1) : "f"(b_in[lane + 1 * 32]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1;
    out[lane + 2 * 32] = d2;
    out[lane + 3 * 32] = d3;
}

extern "C" __global__ void __launch_bounds__(32)
probe_mma_tf32_m16n8k8_chain(float *out,
                             const float *a_in,
                             const float *b_in,
                             const float *c_in) {
    int lane = threadIdx.x;
    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    unsigned b0 = 0, b1 = 0;
    float d0 = c_in[lane + 0 * 32];
    float d1 = c_in[lane + 1 * 32];
    float d2 = c_in[lane + 2 * 32];
    float d3 = c_in[lane + 3 * 32];

    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a0) : "f"(a_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a1) : "f"(a_in[lane + 1 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a2) : "f"(a_in[lane + 2 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(a3) : "f"(a_in[lane + 3 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b0) : "f"(b_in[lane + 0 * 32]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(b1) : "f"(b_in[lane + 1 * 32]));

    #pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1));
    }

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1;
    out[lane + 2 * 32] = d2;
    out[lane + 3 * 32] = d3;
}
