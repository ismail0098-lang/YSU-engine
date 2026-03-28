/*
 * SASS RE Probe: exact uniform truth-table follow-up
 *
 * Goal: preserve a warp-uniform predicate truth table around the same
 * neighborhood as the cuDNN-mined UPLOP3.LUT forms: 0x40 / 0x80 control bits,
 * uniform selects, and tensor/cp.async-adjacent arithmetic.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned probe_truth_smem_addr(unsigned char *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_uniform_uplop3_truth_table_exact(float *out,
                                       const float *in0,
                                       const float *in1,
                                       const unsigned char *src,
                                       int iters,
                                       int stage_mask,
                                       uint64_t stride_bytes) {
    __shared__ __align__(16) unsigned char smem[32];
    int lane = threadIdx.x;
    unsigned tf0 = 0, tf1 = 0;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(tf0) : "f"(in0[lane]));
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(tf1) : "f"(in1[lane]));

    unsigned dst = probe_truth_smem_addr(smem);
    uint64_t byte_off = 0;

    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        bool up40 = (stage_mask & 0x40) != 0;
        bool up80 = (stage_mask & 0x80) != 0;
        bool sel0 = ((iter + stage_mask) & 1) == 0;
        bool sel1 = ((iter + stage_mask) & 2) != 0;
        bool sel2 = ((iter + stage_mask) & 4) != 0;
        bool gate0 = (up40 && !sel0) || (up80 && sel1);
        bool gate1 = (up80 && !sel1) || (up40 && sel2);
        bool gate2 = gate0 ^ gate1;

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 8;"
                     :: "r"(dst), "l"(src + byte_off));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");

        if (gate0 || gate1) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %4, %4, %4}, "
                "{%5, %5}, "
                "{%0, %1, %2, %3};"
                : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
                : "r"(tf0), "r"(tf1));
        }

        if (gate0) acc0 += (float)smem[(lane + 0) & 15];
        if (gate1) acc1 += (float)smem[(lane + 1) & 15];
        if (gate2) acc2 += (float)smem[(lane + 2) & 15];
        if (up40 && up80) acc3 += (float)smem[(lane + 3) & 15];

        byte_off += stride_bytes;
        asm volatile("" : "+l"(byte_off));
    }

    out[lane + 0 * 32] = acc0;
    out[lane + 1 * 32] = acc1;
    out[lane + 2 * 32] = acc2;
    out[lane + 3 * 32] = acc3;
}
