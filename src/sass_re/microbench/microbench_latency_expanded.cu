/*
 * SASS RE: Expanded Instruction Latency Microbenchmark
 *
 * Measures dependent-chain latency for instructions discovered in the
 * expanded probes (waves 2-3) that are NOT covered by microbench_latency.cu.
 *
 * Same methodology as v4: clock64 timing, volatile stores to keep chains
 * live, 512-deep chains, 20 averaged measurements, 1-warp launch.
 *
 * New instructions measured:
 *   DADD    -- FP64 add
 *   DFMA    -- FP64 fused multiply-add
 *   HADD2   -- FP16 packed half2 add
 *   HFMA2   -- FP16 packed half2 FMA
 *   HFMA2.BF16 -- BF16 packed bfloat162 FMA
 *   IDP.4A  -- INT8 dot product accumulate (dp4a)
 *   MUFU.RCP64H -- FP64 reciprocal approximation
 *   NANOSLEEP -- warp yield with timer (PTX nanosleep)
 *
 * Build: nvcc -arch=sm_89 -O1 -o latency_expanded.exe microbench_latency_expanded.cu
 */

#include <stdio.h>
#include <stdlib.h>
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

#define N 512

/* ── DADD: FP64 add ─────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_dadd(volatile double *vals, volatile long long *out) {
    double x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("add.f64 %0, %0, %1;" : "+d"(x) : "d"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── DFMA: FP64 fused multiply-add ──────────────────── */
__global__ void __launch_bounds__(32)
k_dfma(volatile double *vals, volatile long long *out) {
    double x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x) : "d"(y), "d"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── HADD2: FP16 packed half2 add ───────────────────── */
__global__ void __launch_bounds__(32)
k_hadd2(volatile unsigned *vals, volatile long long *out) {
    unsigned raw = vals[0];
    __half2 x; memcpy(&x, &raw, sizeof(x));
    __half2 y = __float2half2_rn(0.001f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        x = __hadd2(x, y);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned out_raw; memcpy(&out_raw, &x, sizeof(out_raw));
    vals[2] = out_raw;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── HFMA2: FP16 packed half2 FMA ───────────────────── */
__global__ void __launch_bounds__(32)
k_hfma2(volatile unsigned *vals, volatile long long *out) {
    unsigned raw = vals[0];
    __half2 x; memcpy(&x, &raw, sizeof(x));
    __half2 scale = __float2half2_rn(0.999f);
    __half2 bias = __float2half2_rn(0.001f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        x = __hfma2(scale, x, bias);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned out_raw; memcpy(&out_raw, &x, sizeof(out_raw));
    vals[2] = out_raw;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── HFMA2.BF16: BF16 packed bfloat162 FMA ──────────── */
__global__ void __launch_bounds__(32)
k_hfma2_bf16(volatile unsigned *vals, volatile long long *out) {
    __nv_bfloat162 x = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 scale = __float2bfloat162_rn(0.999f);
    __nv_bfloat162 bias = __float2bfloat162_rn(0.001f);
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        x = __hfma2(scale, x, bias);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned out_raw; memcpy(&out_raw, &x, sizeof(out_raw));
    vals[2] = out_raw;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── IDP.4A: INT8 dot product accumulate ────────────── */
__global__ void __launch_bounds__(32)
k_dp4a(volatile int *vals, volatile long long *out) {
    int a = vals[0];  // 4 packed INT8
    int b = vals[1];  // 4 packed INT8
    int acc = 0;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        acc = __dp4a(a, b, acc);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = acc;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.RCP64H: FP64 reciprocal approximation ────── */
__global__ void __launch_bounds__(32)
k_mufu_rcp64(volatile double *vals, volatile long long *out) {
    double x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("rcp.approx.ftz.f64 %0, %0;" : "+d"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── NANOSLEEP: warp-level yield with timer ─────────── */
__global__ void __launch_bounds__(32)
k_nanosleep(volatile int *sink, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    // 512 nanosleep calls at minimum duration (0 ns = yield only)
    #pragma unroll
    for (int i = 0; i < N; i++)
        __nanosleep(0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── DMUL: FP64 multiply ───────────────────────────── */
__global__ void __launch_bounds__(32)
k_dmul(volatile double *vals, volatile long long *out) {
    double x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("mul.f64 %0, %0, %1;" : "+d"(x) : "d"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ================================================================
 *  Host: measure and print all new instruction latencies
 * ================================================================ */

struct Res { const char *name; double cyc; };

typedef void (*kdp_t)(volatile double*,   volatile long long*);
typedef void (*kup_t)(volatile unsigned*, volatile long long*);
typedef void (*kip_t)(volatile int*,      volatile long long*);

static double measure_dp(kdp_t k, double *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 20;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 2*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}
static double measure_up(kup_t k, unsigned *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 20;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 2*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}
static double measure_ip(kip_t k, int *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 20;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 2*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);

    printf("==============================================\n");
    printf("  SASS Expanded Latency Benchmark\n");
    printf("  SM %d.%d  |  %s\n", prop.major, prop.minor, prop.name);
    printf("  Clock %d MHz  |  %d SMs  |  Chain %d\n",
           clkKHz/1000, prop.multiProcessorCount, N);
    printf("==============================================\n\n");

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4*sizeof(long long)));

    // FP64 params
    double hd[4] = {1.0, 0.999, 0.001, 0.0};
    double *d_dv; CHECK(cudaMalloc(&d_dv, sizeof(hd)));
    CHECK(cudaMemcpy(d_dv, hd, sizeof(hd), cudaMemcpyHostToDevice));

    // Half2/BF16 params (as unsigned)
    unsigned hu[4] = {0x3C003C00u, 0x3800u, 0u, 0u};  // half2(1.0, 1.0)
    unsigned *d_uv; CHECK(cudaMalloc(&d_uv, sizeof(hu)));
    CHECK(cudaMemcpy(d_uv, hu, sizeof(hu), cudaMemcpyHostToDevice));

    // INT params
    int hi[4] = {0x01020304, 0x05060708, 0, 0};
    int *d_iv; CHECK(cudaMalloc(&d_iv, sizeof(hi)));
    CHECK(cudaMemcpy(d_iv, hi, sizeof(hi), cudaMemcpyHostToDevice));

    printf("Running ...\n\n");

    Res r[20]; int n = 0;

    r[n++] = {"DADD",        measure_dp(k_dadd,      d_dv, d_out, h_out)};
    r[n++] = {"DFMA",        measure_dp(k_dfma,      d_dv, d_out, h_out)};
    r[n++] = {"DMUL",        measure_dp(k_dmul,      d_dv, d_out, h_out)};
    r[n++] = {"MUFU.RCP64H", measure_dp(k_mufu_rcp64,d_dv, d_out, h_out)};
    r[n++] = {"HADD2",       measure_up(k_hadd2,     d_uv, d_out, h_out)};
    r[n++] = {"HFMA2",       measure_up(k_hfma2,     d_uv, d_out, h_out)};
    r[n++] = {"HFMA2.BF16",  measure_up(k_hfma2_bf16,d_uv, d_out, h_out)};
    r[n++] = {"IDP.4A",      measure_ip(k_dp4a,      d_iv, d_out, h_out)};

    // NANOSLEEP
    {
        k_nanosleep<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
        double tot = 0;
        for (int rr = 0; rr < 20; rr++) {
            k_nanosleep<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 2*sizeof(long long), cudaMemcpyDeviceToHost);
            tot += (double)h_out[0]/(double)h_out[1];
        }
        r[n++] = {"NANOSLEEP(0)", tot/20};
    }

    printf("%-20s %14s\n", "Instruction", "Latency (cyc)");
    printf("%-20s %14s\n", "--------------------", "--------------");
    for (int i = 0; i < n; i++)
        printf("%-20s %14.2f\n", r[i].name, r[i].cyc);

    printf("\n--- Context ---\n");
    printf("FP64 gaming SKU: 64:1 ratio vs FP32. Expect DADD/DFMA >> FADD/FFMA.\n");
    printf("HADD2/HFMA2: packed half2, expect ~same as FADD (dual-issue on FP16 pipe).\n");
    printf("IDP.4A: INT8 dot product, expect ~4-5 cyc (INT32 datapath).\n");
    printf("NANOSLEEP(0): yield-only, expect scheduler overhead (~few cyc).\n");

    cudaFree(d_out); cudaFree(d_dv); cudaFree(d_uv); cudaFree(d_iv);
    return 0;
}
