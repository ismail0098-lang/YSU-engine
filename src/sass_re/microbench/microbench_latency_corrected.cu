/*
 * SASS RE: Corrected Latency Chains
 *
 * Fixes measurement artifacts identified by ncu cross-validation:
 *   1. IABS: use negate-then-abs to prevent idempotent folding
 *   2. POPC: use asm volatile with explicit dependency to prevent extra XOR
 *   3. FLO: same fix as POPC
 *   4. DMNMX: isolate just the comparison without loop-counter arithmetic
 *
 * Build: nvcc -arch=sm_89 -O1 -o lat_corrected microbench_latency_corrected.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define N 512

/* ── IABS corrected: negate then abs (not idempotent) ──── */
__global__ void __launch_bounds__(32)
k_iabs_corrected(volatile int *vals, volatile long long *out) {
    int x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        // neg(x) then abs(neg(x)) creates a real dependency chain
        // abs(neg(x)) = abs(x), but neg(abs(x)) = -|x| then abs(-|x|) = |x|
        // We need: x = -x; x = abs(x); which alternates sign
        asm volatile("neg.s32 %0, %0;" : "+r"(x));
        asm volatile("abs.s32 %0, %0;" : "+r"(x));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N * 2; } // 2 ops per iter
}

/* ── POPC corrected: explicit asm volatile chain ───────── */
__global__ void __launch_bounds__(32)
k_popc_corrected(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        // Use popc result directly as input to next popc via asm volatile
        // This prevents compiler from inserting extra XOR
        unsigned cnt;
        asm volatile("popc.b32 %0, %1;" : "=r"(cnt) : "r"(x));
        // Feed count back as next input (popc(small_number) oscillates: 1->1->1...)
        // Add a constant to keep it interesting
        asm volatile("or.b32 %0, %1, 0xff;" : "=r"(x) : "r"(cnt));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N * 2; } // POPC + OR per iter
}

/* ── FLO corrected: explicit asm volatile chain ────────── */
__global__ void __launch_bounds__(32)
k_flo_corrected(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned lz;
        asm volatile("bfind.u32 %0, %1;" : "=r"(lz) : "r"(x));
        // Feed back with OR to maintain non-trivial value
        asm volatile("or.b32 %0, %1, 0xff00;" : "=r"(x) : "r"(lz));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N * 2; } // FLO + OR per iter
}

/* ── DMNMX corrected: pure comparison chain via asm ────── */
__global__ void __launch_bounds__(32)
k_dmnmx_corrected(volatile double *vals, volatile long long *out) {
    double x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        // min then max alternating: x = min(x,y); x = max(x,y+0.001)
        // This ensures DMNMX is the bottleneck, not loop overhead
        asm volatile("min.f64 %0, %0, %1;" : "+d"(x) : "d"(y));
        y = y + 0.001;
        asm volatile("max.f64 %0, %0, %1;" : "+d"(x) : "d"(y));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N * 2; } // 2 DMNMX per iter
}

/* ── BREV corrected: pure chain via asm volatile ───────── */
__global__ void __launch_bounds__(32)
k_brev_corrected(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("brev.b32 %0, %0;" : "+r"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── FP32 FTZ vs IEEE latency comparison ───────────────── */
__global__ void __launch_bounds__(32)
k_ffma_ieee(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

__global__ void __launch_bounds__(32)
k_ffma_ftz(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ================================================================ */

typedef void (*kup_t)(volatile unsigned*, volatile long long*);
typedef void (*kfp_t)(volatile float*, volatile long long*);
typedef void (*kip_t)(volatile int*, volatile long long*);
typedef void (*kdp_t)(volatile double*, volatile long long*);

static double measure_up(kup_t k, unsigned *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}
static double measure_fp(kfp_t k, float *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}
static double measure_ip(kip_t k, int *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}
static double measure_dp(kdp_t k, double *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Corrected Latency Measurements ===\n");
    printf("SM %d.%d | %s\n\n", prop.major, prop.minor, prop.name);

    long long *d_out, h[4]; CHECK(cudaMalloc(&d_out, 32));
    unsigned hu[4]={0xDEADBEEFu,0xCAFEu,0,0};
    unsigned *d_u; CHECK(cudaMalloc(&d_u,16)); CHECK(cudaMemcpy(d_u,hu,16,cudaMemcpyHostToDevice));
    int hi2[4]={42,7,3,0};
    int *d_i; CHECK(cudaMalloc(&d_i,16)); CHECK(cudaMemcpy(d_i,hi2,16,cudaMemcpyHostToDevice));
    float hf[4]={1.0f,0.999f,0.001f,0.0f};
    float *d_f; CHECK(cudaMalloc(&d_f,16)); CHECK(cudaMemcpy(d_f,hf,16,cudaMemcpyHostToDevice));
    double hd[4]={1.5,0.7,0.0,0.0};
    double *d_d; CHECK(cudaMalloc(&d_d,32)); CHECK(cudaMemcpy(d_d,hd,32,cudaMemcpyHostToDevice));

    printf("%-28s %12s %12s\n", "Instruction", "Corrected cy", "Previous cy");
    printf("%-28s %12s %12s\n", "----------------------------", "------------", "------------");

    printf("%-28s %12.2f %12s\n", "IABS (neg+abs chain)", measure_ip(k_iabs_corrected, d_i, d_out, h), "0.53 (artifact)");
    printf("%-28s %12.2f %12s\n", "POPC (popc+or chain)", measure_up(k_popc_corrected, d_u, d_out, h), "23.52 (2x insts)");
    printf("%-28s %12.2f %12s\n", "FLO (flo+or chain)", measure_up(k_flo_corrected, d_u, d_out, h), "23.52 (2x insts)");
    printf("%-28s %12.2f %12s\n", "DMNMX (min+max chain)", measure_dp(k_dmnmx_corrected, d_d, d_out, h), "114.63 (9x insts)");
    printf("%-28s %12.2f %12s\n", "BREV (pure asm chain)", measure_up(k_brev_corrected, d_u, d_out, h), "17.49");
    printf("%-28s %12.2f %12s\n", "FFMA IEEE (baseline)", measure_fp(k_ffma_ieee, d_f, d_out, h), "4.53");
    printf("%-28s %12.2f %12s\n", "FFMA FTZ (flush-to-zero)", measure_fp(k_ffma_ftz, d_f, d_out, h), "N/A");

    printf("\nNote: IABS/POPC/FLO report cy per op pair (2 ops per iteration).\n");
    printf("Divide by 2 for single-instruction latency if ops are independent.\n");

    cudaFree(d_out); cudaFree(d_u); cudaFree(d_i); cudaFree(d_f); cudaFree(d_d);
    return 0;
}
