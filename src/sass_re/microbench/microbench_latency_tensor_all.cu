/*
 * SASS RE: Tensor Core Latency for ALL Precision Formats
 *
 * Measures dependent-chain HMMA/IMMA latency for every tensor core
 * precision available on Ada Lovelace SM 8.9:
 *
 *   FP16  -> FP32 accum:  HMMA.16816.F32        (256 FMA per instruction)
 *   FP16  -> FP16 accum:  HMMA.16816.F16        (256 FMA, lower precision accum)
 *   BF16  -> FP32 accum:  HMMA.16816.F32.BF16   (256 FMA)
 *   TF32  -> FP32 accum:  HMMA.1684.F32.TF32    (128 FMA, K=4 per instruction)
 *   INT8  -> INT32 accum: IMMA.16816.S8.S8       (256 INT8 multiply-accumulate)
 *   UINT8 -> INT32 accum: IMMA.16816.U8.U8       (256 UINT8 multiply-accumulate)
 *   INT4  -> INT32 accum: IMMA.8832.S4.S4        (256 INT4 multiply-accumulate)
 *   UINT4 -> INT32 accum: IMMA.8832.U4.U4        (256 UINT4 multiply-accumulate)
 *
 * Method: 256-deep dependent chain of WMMA mma_sync calls.
 * Each call depends on the previous accumulator, forcing serialization.
 * Reported: cycles per WMMA call (which may decompose to 1-2 HMMA instructions).
 *
 * Build: nvcc -arch=sm_89 -O1 -o lat_tc_all microbench_latency_tensor_all.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define TC_CHAIN 256

/* ── FP16 -> FP32 accumulator (HMMA.16816.F32) ──────── */
__global__ void __launch_bounds__(32)
k_tc_fp16_f32(half *d_D, const half *d_A, const half *d_B,
              volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    // Store to prevent DCE
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC.num_elements; j++) fD.x[j] = __float2half(fC.x[j]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── FP16 -> FP16 accumulator (HMMA.16816.F16) ──────── */
__global__ void __launch_bounds__(32)
k_tc_fp16_f16(half *d_D, const half *d_A, const half *d_B,
              volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, __float2half(0.0f));
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── BF16 -> FP32 accumulator (HMMA.16816.F32.BF16) ── */
__global__ void __launch_bounds__(32)
k_tc_bf16_f32(float *d_D, const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
              volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── TF32 -> FP32 accumulator (HMMA.1684.F32.TF32) ── */
__global__ void __launch_bounds__(32)
k_tc_tf32_f32(float *d_D, const float *d_A, const float *d_B,
              volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── INT8 -> INT32 accumulator (IMMA.16816.S8.S8) ──── */
__global__ void __launch_bounds__(32)
k_tc_int8_i32(int *d_D, const signed char *d_A, const signed char *d_B,
              volatile long long *timing) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── INT4 -> INT32 accumulator (IMMA.8832.S4.S4) ──── */
__global__ void __launch_bounds__(32)
k_tc_int4_i32(int *d_D, const void *d_A, const void *d_B,
              volatile long long *timing) {
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> fC;
    wmma::load_matrix_sync(fA, d_A, 32);
    wmma::load_matrix_sync(fB, d_B, 32);
    wmma::fill_fragment(fC, 0);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 8, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ── UINT4 -> INT32 accumulator (IMMA.8832.U4.U4) ── */
__global__ void __launch_bounds__(32)
k_tc_uint4_i32(int *d_D, const void *d_A, const void *d_B,
               volatile long long *timing) {
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::u4, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::u4, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> fC;
    wmma::load_matrix_sync(fA, d_A, 32);
    wmma::load_matrix_sync(fB, d_B, 32);
    wmma::fill_fragment(fC, 0);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < TC_CHAIN; i++)
        wmma::mma_sync(fC, fA, fB, fC);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    wmma::store_matrix_sync(d_D, fC, 8, wmma::mem_row_major);
    if (threadIdx.x == 0) { timing[0] = t1 - t0; timing[1] = TC_CHAIN; }
}

/* ================================================================ */

static double measure_tc(void *kernel, void *d_D, const void *d_A,
                         const void *d_B, long long *d_t, long long *h) {
    // Generic launcher for all TC kernels (same signature pattern)
    // Each kernel is launched, warmed up, then measured 10x
    typedef void (*tc_kernel_t)(void*, const void*, const void*, volatile long long*);
    tc_kernel_t k = (tc_kernel_t)kernel;
    k<<<1,32>>>(d_D, d_A, d_B, d_t); cudaDeviceSynchronize();
    double tot = 0;
    for (int r = 0; r < 10; r++) {
        k<<<1,32>>>(d_D, d_A, d_B, d_t); cudaDeviceSynchronize();
        cudaMemcpy(h, d_t, 16, cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / 10;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Tensor Core Latency: ALL Precisions ===\n");
    printf("SM %d.%d | %s | 256-deep dependent chains\n\n", prop.major, prop.minor, prop.name);

    // Allocate buffers (oversized for all formats)
    void *d_A, *d_B, *d_D;
    long long *d_t, h[4];
    CHECK(cudaMalloc(&d_A, 4096)); CHECK(cudaMemset(d_A, 0, 4096));
    CHECK(cudaMalloc(&d_B, 4096)); CHECK(cudaMemset(d_B, 0, 4096));
    CHECK(cudaMalloc(&d_D, 4096));
    CHECK(cudaMalloc(&d_t, 32));

    printf("%-32s %10s %10s %12s\n", "Format", "cy/WMMA", "SASS instr", "FMA/WMMA");
    printf("%-32s %10s %10s %12s\n",
           "--------------------------------", "----------", "----------", "------------");

    double cy;
    cy = measure_tc((void*)k_tc_fp16_f32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "FP16 -> FP32 (HMMA.16816)", cy, "1 HMMA", 256);

    cy = measure_tc((void*)k_tc_fp16_f16, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "FP16 -> FP16 (HMMA.16816.F16)", cy, "1 HMMA", 256);

    cy = measure_tc((void*)k_tc_bf16_f32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "BF16 -> FP32 (HMMA.16816.BF16)", cy, "1 HMMA", 256);

    cy = measure_tc((void*)k_tc_tf32_f32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "TF32 -> FP32 (HMMA.1684x2)", cy, "2 HMMA", 256);

    cy = measure_tc((void*)k_tc_int8_i32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "INT8 -> INT32 (IMMA.16816.S8)", cy, "1 IMMA", 256);

    cy = measure_tc((void*)k_tc_int4_i32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "INT4 -> INT32 (IMMA.8832.S4)", cy, "1 IMMA", 256);

    cy = measure_tc((void*)k_tc_uint4_i32, d_D, d_A, d_B, d_t, h);
    printf("%-32s %10.2f %10s %12d\n", "UINT4 -> INT32 (IMMA.8832.U4)", cy, "1 IMMA", 256);

    printf("\nNotes:\n");
    printf("  TF32 WMMA 16x16x8 decomposes to 2x HMMA.1684 (K=4 each).\n");
    printf("  INT4/UINT4 WMMA 8x8x32 uses different shape than 16x16x16.\n");
    printf("  All dependent chains: accumulator feeds back as input.\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_t);
    return 0;
}
