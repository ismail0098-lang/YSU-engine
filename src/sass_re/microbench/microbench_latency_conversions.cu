/*
 * SASS RE: Conversion Instruction Latency Microbenchmark
 *
 * Measures latency for format conversion instructions that appear in
 * the storage-compute split hot path of LBM precision-tier kernels.
 *
 * Conversions measured:
 *   F2FP.E4M3 (FP32->FP8 encode)    F2FP.E5M2 (FP32->FP8 E5M2)
 *   F2FP.F16.E4M3 (FP8->FP16 decode) PRMT (BF16->FP32 via byte permute)
 *   F2FP.BF16 (FP32->BF16)           F2F.F64.F32 (FP32->FP64 widen)
 *   I2FP.F32.S32 (INT->float)        F2I (float->INT)
 *   LDC (constant memory load)
 *
 * Build: nvcc -arch=sm_89 -O1 -o lat_conv microbench_latency_conversions.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define N 512

__constant__ float CONST_TABLE[256];

// FP32 -> FP16 conversion chain
__global__ void __launch_bounds__(32)
k_f32_to_f16(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        __half h = __float2half(x);
        x = __half2float(h);  // Round-trip creates dependency
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// FP32 -> BF16 conversion chain (via PRMT path)
__global__ void __launch_bounds__(32)
k_f32_to_bf16(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        __nv_bfloat16 bf = __float2bfloat16(x);
        x = __bfloat162float(bf);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// FP32 -> FP8 E4M3 round-trip
__global__ void __launch_bounds__(32)
k_f32_to_fp8_e4m3(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
        __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
        __half h; __builtin_memcpy(&h, &hraw, sizeof(h));
        x = __half2float(h);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// FP32 -> FP64 widen chain
__global__ void __launch_bounds__(32)
k_f32_to_f64(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        double d = (double)x;
        x = (float)d;  // Narrow back for chain
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// INT32 -> FP32 conversion chain
__global__ void __launch_bounds__(32)
k_i32_to_f32(volatile int *vals, volatile long long *out) {
    int x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        float f = (float)x;
        x = (int)f;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// Constant memory load chain
__global__ void __launch_bounds__(32)
k_ldc_chain(volatile float *sink, volatile long long *out) {
    float acc = 0.0f;
    int idx = 0;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        float val = CONST_TABLE[idx % 256];
        acc += val;
        idx = (int)(acc * 7.0f) & 0xFF;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = acc;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

typedef void (*kfp_t)(volatile float*, volatile long long*);
typedef void (*kip_t)(volatile int*, volatile long long*);

static double measure_fp(kfp_t k, float *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0;
    for (int r = 0; r < 20; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 16, cudaMemcpyDeviceToHost);
        tot += (double)h[0]/(double)h[1];
    }
    return tot / 20;
}
static double measure_ip(kip_t k, int *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0;
    for (int r = 0; r < 20; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 16, cudaMemcpyDeviceToHost);
        tot += (double)h[0]/(double)h[1];
    }
    return tot / 20;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Conversion Latency Benchmark ===\n");
    printf("SM %d.%d | %s\n\n", prop.major, prop.minor, prop.name);

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 32));
    float hf[4] = {1.0f, 0.5f, 0.001f, 0.0f};
    float *d_fv; CHECK(cudaMalloc(&d_fv, 16));
    CHECK(cudaMemcpy(d_fv, hf, 16, cudaMemcpyHostToDevice));
    int hi[4] = {42, 7, 3, 0};
    int *d_iv; CHECK(cudaMalloc(&d_iv, 16));
    CHECK(cudaMemcpy(d_iv, hi, 16, cudaMemcpyHostToDevice));

    // Init constant memory
    float hc[256];
    for (int i = 0; i < 256; i++) hc[i] = (float)i * 0.01f;
    cudaMemcpyToSymbol(CONST_TABLE, hc, sizeof(hc));

    printf("%-24s %14s\n", "Conversion", "Latency (cyc)");
    printf("%-24s %14s\n", "------------------------", "--------------");

    printf("%-24s %14.2f\n", "FP32<->FP16 round-trip", measure_fp(k_f32_to_f16, d_fv, d_out, h_out));
    printf("%-24s %14.2f\n", "FP32<->BF16 round-trip", measure_fp(k_f32_to_bf16, d_fv, d_out, h_out));
    printf("%-24s %14.2f\n", "FP32<->FP8_E4M3 r/t",   measure_fp(k_f32_to_fp8_e4m3, d_fv, d_out, h_out));
    printf("%-24s %14.2f\n", "FP32<->FP64 round-trip", measure_fp(k_f32_to_f64, d_fv, d_out, h_out));
    printf("%-24s %14.2f\n", "INT32<->FP32 round-trip",measure_ip(k_i32_to_f32, d_iv, d_out, h_out));
    printf("%-24s %14.2f\n", "LDC chain (const mem)",  measure_fp(k_ldc_chain, d_fv, d_out, h_out));

    cudaFree(d_out); cudaFree(d_fv); cudaFree(d_iv);
    return 0;
}
