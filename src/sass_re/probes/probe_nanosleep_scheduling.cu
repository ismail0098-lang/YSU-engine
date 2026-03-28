/*
 * SASS RE Probe: NANOSLEEP and Warp Scheduling Behavior
 * Isolates: NANOSLEEP instruction, warp yield, scheduling fairness
 *
 * From latency measurement: NANOSLEEP(0) = 2685 cycles.
 * This probe characterizes the instruction at different timer values
 * and explores warp scheduling interactions.
 *
 * NANOSLEEP(ns) translates to PTX `nanosleep.u32 %r;` which generates
 * SASS instruction (TBD -- this probe will reveal the SASS mnemonic).
 *
 * The massive 2685-cycle overhead at ns=0 suggests NANOSLEEP forces a
 * full warp deschedule + reschedule even with zero delay. This is useful
 * for reducing power consumption in spin-wait loops but catastrophic for
 * latency-sensitive code.
 *
 * Also probes warp-level float reduction via SHFL (since REDUX.FADD
 * does not exist on Ada -- confirmed integer-only).
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 512

/* ── NANOSLEEP at various timer values ──────────────── */
extern "C" __global__ void __launch_bounds__(32)
k_nanosleep_0(volatile int *sink, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        __nanosleep(0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

extern "C" __global__ void __launch_bounds__(32)
k_nanosleep_100(volatile int *sink, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        __nanosleep(100);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

extern "C" __global__ void __launch_bounds__(32)
k_nanosleep_1000(volatile int *sink, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 64; i++)  // Fewer iterations for longer sleep
        __nanosleep(1000);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 64; }
}

/* ── Warp-level FLOAT reduction (SHFL, since REDUX.FADD doesn't exist) ── */
extern "C" __global__ void __launch_bounds__(32)
k_float_reduce_shfl(volatile float *sink, volatile long long *out) {
    float x = (float)threadIdx.x + sink[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    // 512 float reductions via SHFL butterfly (5 stages each)
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, 16);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 8);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 4);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 2);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        sink[1] = x;
        // Report as cycles per reduction (each reduction = 5 SHFL + 5 FADD)
        out[0] = t1 - t0; out[1] = N;
    }
}

/* ── Warp-level INT reduction: REDUX.SUM vs SHFL comparison ── */
extern "C" __global__ void __launch_bounds__(32)
k_int_reduce_redux(volatile int *sink, volatile long long *out) {
    int x = threadIdx.x + sink[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        x = __reduce_add_sync(0xFFFFFFFF, x);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        sink[1] = x;
        out[0] = t1 - t0; out[1] = N;
    }
}

extern "C" __global__ void __launch_bounds__(32)
k_int_reduce_shfl(volatile int *sink, volatile long long *out) {
    int x = threadIdx.x + sink[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, 16);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 8);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 4);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 2);
        x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        sink[1] = x;
        out[0] = t1 - t0; out[1] = N;
    }
}

/* ================================================================ */

static double measure(void (*k)(volatile int*, volatile long long*),
                      int *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 10;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 2*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}

static double measure_fp(void (*k)(volatile float*, volatile long long*),
                         float *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 10;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 2*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}

#ifndef SASS_RE_EMBEDDED_RUNNER
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== NANOSLEEP & Warp Scheduling Probe ===\n");
    printf("SM %d.%d | %s\n\n", prop.major, prop.minor, prop.name);

    long long *d_out, h_out[4];
    cudaMalloc(&d_out, 4*sizeof(long long));
    int hi[4] = {1, 0, 0, 0};
    int *d_iv; cudaMalloc(&d_iv, sizeof(hi));
    cudaMemcpy(d_iv, hi, sizeof(hi), cudaMemcpyHostToDevice);
    float hf[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float *d_fv; cudaMalloc(&d_fv, sizeof(hf));
    cudaMemcpy(d_fv, hf, sizeof(hf), cudaMemcpyHostToDevice);

    printf("%-24s %14s\n", "Test", "Cycles/iter");
    printf("%-24s %14s\n", "------------------------", "--------------");

    printf("%-24s %14.2f\n", "NANOSLEEP(0ns)",
           measure(k_nanosleep_0, d_iv, d_out, h_out));
    printf("%-24s %14.2f\n", "NANOSLEEP(100ns)",
           measure(k_nanosleep_100, d_iv, d_out, h_out));
    printf("%-24s %14.2f\n", "NANOSLEEP(1000ns)",
           measure(k_nanosleep_1000, d_iv, d_out, h_out));
    printf("%-24s %14.2f\n", "Float reduce (5xSHFL)",
           measure_fp(k_float_reduce_shfl, d_fv, d_out, h_out));
    printf("%-24s %14.2f\n", "Int reduce (REDUX.SUM)",
           measure(k_int_reduce_redux, d_iv, d_out, h_out));
    printf("%-24s %14.2f\n", "Int reduce (5xSHFL)",
           measure(k_int_reduce_shfl, d_iv, d_out, h_out));

    printf("\n--- Analysis ---\n");
    printf("NANOSLEEP(0): pure warp deschedule+reschedule overhead\n");
    printf("Float reduce: no REDUX.FADD exists; must use 5xSHFL+5xFADD\n");
    printf("REDUX.SUM vs SHFL: single instruction vs 5-stage tree\n");

    cudaFree(d_out); cudaFree(d_iv); cudaFree(d_fv);
    return 0;
}
#endif
