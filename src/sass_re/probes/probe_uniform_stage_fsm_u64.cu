/*
 * SASS RE Probe: uniform stage FSM with loop-carried U64 rebasing
 *
 * Goal: make ptxas compute warp-uniform next-stage state once, reuse it across
 * cp.async scheduling, HMMA gating, and a loop-carried 64-bit byte pointer
 * bump. This is the closest direct local source shape to the remaining
 * UPLOP3.LUT / USHF.L.U64.HI cluster seen in cuDNN-mined cubins.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned probe_fsm_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_stage_fsm_u64(float *out,
                            const float *a_in,
                            const float *b_in,
                            const unsigned char *src,
                            int tiles,
                            uint64_t byte_stride,
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

    const unsigned base0 = probe_fsm_smem_addr(smem[0]);
    const unsigned base1 = probe_fsm_smem_addr(smem[1]);

    uint64_t byte_off = 0;

    #pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
        int stage = t & 1;
        int tail = (t + 1) >= tiles;
        int prefetch_ok = (t + 1) < tiles;
        int fsm = 0;

        asm volatile("lop3.b32 %0, %1, %2, %3, 0xe8;"
                     : "=r"(fsm)
                     : "r"(stage), "r"(tail), "r"(prefetch_ok));
        fsm &= 1;

        unsigned dst = stage ? base1 : base0;
        unsigned next_dst = fsm ? base1 : base0;
        const unsigned char *gptr = src + byte_off;

        if (!tail) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                         :: "r"(dst), "l"(gptr));
            if (prefetch_ok) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                             :: "r"(next_dst), "l"(gptr + byte_stride));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
        }

        if ((tail_mask ^ fsm) & 1) {
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

        d0 += (float)smem[stage][0];
        d1 += (float)smem[fsm][1];
        d2 += (float)smem[stage][2];
        d3 += (float)smem[fsm][3];

        byte_off += ((uint64_t)(fsm + 1)) * byte_stride;
        asm volatile("" : "+l"(byte_off));
    }

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1;
    out[lane + 2 * 32] = d2;
    out[lane + 3 * 32] = d3;
}
