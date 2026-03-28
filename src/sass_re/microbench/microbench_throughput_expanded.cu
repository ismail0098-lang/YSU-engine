/*
 * SASS RE: Expanded Throughput Microbenchmark
 *
 * Measures sustained throughput (ops/clock/SM) for instructions
 * not covered by the original microbench_throughput.cu.
 *
 * Method: 8 independent accumulator streams, 1024 threads/block,
 * 512 iterations per accumulator. Same anti-optimization as latency v4.
 *
 * Build: nvcc -arch=sm_89 -O1 -o tput_exp microbench_throughput_expanded.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define ITERS 512

// HADD2 throughput: 8 independent half2 add streams
__global__ void __launch_bounds__(1024)
k_tput_hadd2(volatile unsigned *sink) {
    __half2 a0 = __float2half2_rn(1.0f);
    __half2 a1 = __float2half2_rn(1.0f);
    __half2 a2 = __float2half2_rn(1.0f);
    __half2 a3 = __float2half2_rn(1.0f);
    __half2 a4 = __float2half2_rn(1.0f);
    __half2 a5 = __float2half2_rn(1.0f);
    __half2 a6 = __float2half2_rn(1.0f);
    __half2 a7 = __float2half2_rn(1.0f);
    __half2 one = __float2half2_rn(0.001f);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 = __hadd2(a0, one); a1 = __hadd2(a1, one);
        a2 = __hadd2(a2, one); a3 = __hadd2(a3, one);
        a4 = __hadd2(a4, one); a5 = __hadd2(a5, one);
        a6 = __hadd2(a6, one); a7 = __hadd2(a7, one);
    }
    unsigned r; memcpy(&r, &a0, 4);
    sink[threadIdx.x] = r;
}

// IDP.4A throughput: 8 independent dp4a streams
__global__ void __launch_bounds__(1024)
k_tput_dp4a(volatile int *sink) {
    int a = 0x01020304, b = 0x05060708;
    int c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        c0 = __dp4a(a, b, c0); c1 = __dp4a(a, b, c1);
        c2 = __dp4a(a, b, c2); c3 = __dp4a(a, b, c3);
        c4 = __dp4a(a, b, c4); c5 = __dp4a(a, b, c5);
        c6 = __dp4a(a, b, c6); c7 = __dp4a(a, b, c7);
    }
    sink[threadIdx.x] = c0+c1+c2+c3+c4+c5+c6+c7;
}

// HFMA2.BF16 throughput
__global__ void __launch_bounds__(1024)
k_tput_hfma2_bf16(volatile unsigned *sink) {
    __nv_bfloat162 a0 = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 a1 = a0, a2 = a0, a3 = a0;
    __nv_bfloat162 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __nv_bfloat162 scale = __float2bfloat162_rn(0.999f);
    __nv_bfloat162 bias = __float2bfloat162_rn(0.001f);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 = __hfma2(scale, a0, bias); a1 = __hfma2(scale, a1, bias);
        a2 = __hfma2(scale, a2, bias); a3 = __hfma2(scale, a3, bias);
        a4 = __hfma2(scale, a4, bias); a5 = __hfma2(scale, a5, bias);
        a6 = __hfma2(scale, a6, bias); a7 = __hfma2(scale, a7, bias);
    }
    unsigned r; memcpy(&r, &a0, 4);
    sink[threadIdx.x] = r;
}

// DADD throughput: 8 independent FP64 add streams
__global__ void __launch_bounds__(1024)
k_tput_dadd(volatile double *sink) {
    double a0=1.0, a1=1.0, a2=1.0, a3=1.0;
    double a4=1.0, a5=1.0, a6=1.0, a7=1.0;
    double one = 0.001;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 += one; a1 += one; a2 += one; a3 += one;
        a4 += one; a5 += one; a6 += one; a7 += one;
    }
    sink[threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// DFMA throughput
__global__ void __launch_bounds__(1024)
k_tput_dfma(volatile double *sink) {
    double a0=1.0, a1=1.0, a2=1.0, a3=1.0;
    double a4=1.0, a5=1.0, a6=1.0, a7=1.0;
    double scale = 0.999, bias = 0.001;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 = fma(scale, a0, bias); a1 = fma(scale, a1, bias);
        a2 = fma(scale, a2, bias); a3 = fma(scale, a3, bias);
        a4 = fma(scale, a4, bias); a5 = fma(scale, a5, bias);
        a6 = fma(scale, a6, bias); a7 = fma(scale, a7, bias);
    }
    sink[threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int n_sms = prop.multiProcessorCount;
    int clk;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);

    printf("=== Expanded Throughput Benchmark ===\n");
    printf("SM %d.%d | %s | %d SMs | %d MHz\n\n",
           prop.major, prop.minor, prop.name, n_sms, clk/1000);

    // Allocate sinks
    unsigned *d_u; cudaMalloc(&d_u, 1024*4);
    int *d_i; cudaMalloc(&d_i, 1024*4);
    double *d_d; cudaMalloc(&d_d, 1024*8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("%-20s %14s %14s\n", "Instruction", "ops/clk/SM", "peak theor.");
    printf("%-20s %14s %14s\n", "--------------------", "--------------", "--------------");

    float ms; double total_ops, clocks, ops_per_clk_sm;
    int blocks = n_sms;

#define BENCH(name, kernel, sink, ops_per, peak) \
    kernel<<<blocks, 1024>>>(sink); cudaDeviceSynchronize(); \
    cudaEventRecord(start); \
    kernel<<<blocks, 1024>>>(sink); \
    cudaEventRecord(stop); cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms, start, stop); \
    total_ops = (double)blocks * 1024.0 * ITERS * 8.0 * ops_per; \
    clocks = (double)ms * 1e-3 * (double)clk * 1000.0; \
    ops_per_clk_sm = total_ops / clocks / n_sms; \
    printf("%-20s %14.1f %14s\n", name, ops_per_clk_sm, peak);

    BENCH("HADD2 (2xFP16)",  k_tput_hadd2,      d_u, 2, "256");
    BENCH("IDP.4A (4xINT8)", k_tput_dp4a,        d_i, 4, "256");
    BENCH("HFMA2.BF16",      k_tput_hfma2_bf16,  d_u, 2, "256");
    BENCH("DADD",             k_tput_dadd,         d_d, 1, "4");
    BENCH("DFMA",             k_tput_dfma,         d_d, 1, "4");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_u); cudaFree(d_i); cudaFree(d_d);
    return 0;
}
