/*
 * SASS RE Probe: Deep Tensor Core Pipeline Characterization
 * Isolates: TC pipeline depth, TC+ALU overlap, sustained TC throughput
 *
 * Questions this probe answers:
 *   1. How many independent HMMA instructions are needed to saturate the TC pipe?
 *   2. Can tensor core and FP32 ALU run simultaneously?
 *   3. What is the sustained HMMA throughput at different ILP depths?
 *   4. Does LDSM->HMMA->STS pipeline overlap with ALU work?
 *
 * Ada Lovelace TC architecture:
 *   Each SM has 4 tensor core units (one per sub-partition).
 *   Each TC unit can execute one HMMA per cycle.
 *   HMMA.16816 (FP16): 256 FMA ops per instruction.
 *   HMMA.1684 (TF32): 128 FMA ops per instruction.
 *
 * Expected pipeline depth: 4-8 independent HMMA needed for saturation
 * (matches the 4 sub-partitions per SM).
 */

#include <mma.h>
using namespace nvcuda;

// ── Pipeline depth: 1 to 8 independent HMMA chains ──

// Depth 1: single chain (minimum ILP)
__global__ void __launch_bounds__(32)
probe_tc_depth_1(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f);
    for (int i = 0; i < 64; i++)
        wmma::mma_sync(fC0, fA, fB, fC0);
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC0.num_elements; j++) fD.x[j] = __float2half(fC0.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

// Depth 2
__global__ void __launch_bounds__(32)
probe_tc_depth_2(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0, fC1;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f); wmma::fill_fragment(fC1, 0.0f);
    for (int i = 0; i < 64; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
    }
    for (int j = 0; j < fC0.num_elements; j++) fC0.x[j] += fC1.x[j];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC0.num_elements; j++) fD.x[j] = __float2half(fC0.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

// Depth 4
__global__ void __launch_bounds__(32)
probe_tc_depth_4(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0, fC1, fC2, fC3;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f); wmma::fill_fragment(fC1, 0.0f);
    wmma::fill_fragment(fC2, 0.0f); wmma::fill_fragment(fC3, 0.0f);
    for (int i = 0; i < 64; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
        wmma::mma_sync(fC2, fA, fB, fC2);
        wmma::mma_sync(fC3, fA, fB, fC3);
    }
    for (int j = 0; j < fC0.num_elements; j++)
        fC0.x[j] += fC1.x[j] + fC2.x[j] + fC3.x[j];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC0.num_elements; j++) fD.x[j] = __float2half(fC0.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

// Depth 8
__global__ void __launch_bounds__(32)
probe_tc_depth_8(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0, fC1, fC2, fC3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC4, fC5, fC6, fC7;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f); wmma::fill_fragment(fC1, 0.0f);
    wmma::fill_fragment(fC2, 0.0f); wmma::fill_fragment(fC3, 0.0f);
    wmma::fill_fragment(fC4, 0.0f); wmma::fill_fragment(fC5, 0.0f);
    wmma::fill_fragment(fC6, 0.0f); wmma::fill_fragment(fC7, 0.0f);
    for (int i = 0; i < 64; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
        wmma::mma_sync(fC2, fA, fB, fC2);
        wmma::mma_sync(fC3, fA, fB, fC3);
        wmma::mma_sync(fC4, fA, fB, fC4);
        wmma::mma_sync(fC5, fA, fB, fC5);
        wmma::mma_sync(fC6, fA, fB, fC6);
        wmma::mma_sync(fC7, fA, fB, fC7);
    }
    for (int j = 0; j < fC0.num_elements; j++)
        fC0.x[j] += fC1.x[j]+fC2.x[j]+fC3.x[j]+fC4.x[j]+fC5.x[j]+fC6.x[j]+fC7.x[j];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC0.num_elements; j++) fD.x[j] = __float2half(fC0.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

// ── TC + ALU overlap: interleave HMMA with FFMA ──

__global__ void __launch_bounds__(32)
probe_tc_alu_overlap(half *d_D, float *d_alu_out,
                     const half *d_A, const half *d_B, const float *d_C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);

    int tid = threadIdx.x;
    float alu_acc = d_C[tid];

    // Interleave: TC work + FP32 ALU work
    for (int i = 0; i < 64; i++) {
        wmma::mma_sync(fC, fA, fB, fC);   // TC pipe
        alu_acc = fmaf(alu_acc, 0.999f, 0.001f);  // FP32 pipe (should overlap)
        alu_acc = fmaf(alu_acc, 0.999f, 0.001f);
        alu_acc = fmaf(alu_acc, 0.999f, 0.001f);
        alu_acc = fmaf(alu_acc, 0.999f, 0.001f);
    }

    d_alu_out[tid] = alu_acc;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC.num_elements; j++) fD.x[j] = __float2half(fC.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

// ── TC-only throughput measurement ──

__global__ void __launch_bounds__(32)
probe_tc_throughput_fp16(half *d_D, const half *d_A, const half *d_B,
                         volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);

    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    // 256 back-to-back HMMA (dependent chain = pipeline depth measurement)
    #pragma unroll 1
    for (int i = 0; i < 256; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC.num_elements; j++) fD.x[j] = __float2half(fC.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);

    if (threadIdx.x == 0) {
        timing[0] = t1 - t0;
        timing[1] = 256;
    }
}
