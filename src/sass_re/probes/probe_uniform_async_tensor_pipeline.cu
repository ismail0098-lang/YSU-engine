/*
 * SASS RE Probe: warp-uniform async-copy plus tensor-style control
 *
 * cuDNN-mined UPLOP3.LUT clusters around warp-uniform predicate state near
 * HMMA and LDGSTS.LTC128B.128. This probe combines warp-uniform conditions,
 * cp.async.L2::128B traffic, and tensor-style math in a single kernel so the
 * compiler has a stronger reason to preserve uniform predicate fan-in.
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

static __device__ __forceinline__ unsigned uniform_async_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_async_tensor_gate(float *out,
                                const float *a_in,
                                const float *b_in,
                                const unsigned char *src,
                                int mode,
                                int bias) {
    __shared__ __align__(16) unsigned char smem[64];
    unsigned saddr = uniform_async_smem_addr(smem);

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(saddr), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");

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

    int run_a = ((mode & 1) != 0);
    int run_b = ((mode & 2) != 0);
    int run_c = ((bias & 4) != 0);

    if (run_a) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    if (!run_b) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    if ((run_a && run_c) || (!run_a && run_b)) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    out[lane + 0 * 32] = d0;
    out[lane + 1 * 32] = d1 + static_cast<float>(smem[0]);
    out[lane + 2 * 32] = d2 + static_cast<float>(smem[1]);
    out[lane + 3 * 32] = d3;
}
