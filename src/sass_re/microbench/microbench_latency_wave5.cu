/*
 * SASS RE: Wave 5 Instruction Latency Microbenchmark
 *
 * Measures dependent-chain latency for instructions discovered in
 * wave 5 probes. Same v4 methodology: clock64 timing, volatile stores,
 * 512-deep chains, 20 averaged measurements, 1-warp launch.
 *
 * New instructions measured:
 *   BREV     -- bit reversal
 *   POPC     -- population count
 *   FLO      -- find leading one (count leading zeros)
 *   BFI      -- bit field insert
 *   BFE      -- bit field extract
 *   FSEL     -- float conditional select (predicated)
 *   IABS     -- integer absolute value
 *   DMNMX    -- FP64 min/max
 *   YIELD    -- explicit warp yield
 *   MEMBAR.GPU -- GPU-scope memory fence
 *   MEMBAR.SYS -- system-scope memory fence
 *   LDC      -- constant memory load (broadcast)
 *
 * Build: nvcc -arch=sm_89 -O1 -o latency_wave5 microbench_latency_wave5.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define N 512

/* ── BREV: bit reversal ─────────────────────────────── */
__global__ void __launch_bounds__(32)
k_brev(volatile unsigned *vals, volatile long long *out) {
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

/* ── POPC: population count ─────────────────────────── */
__global__ void __launch_bounds__(32)
k_popc(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned cnt;
        asm volatile("popc.b32 %0, %1;" : "=r"(cnt) : "r"(x));
        x = x ^ cnt;  // Feed back to create dependency
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── FLO: find leading one ──────────────────────────── */
__global__ void __launch_bounds__(32)
k_flo(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned lz;
        asm volatile("bfind.u32 %0, %1;" : "=r"(lz) : "r"(x));
        x = x ^ lz;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── BFE: bit field extract ─────────────────────────── */
__global__ void __launch_bounds__(32)
k_bfe(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("bfe.u32 %0, %0, 4, 8;" : "+r"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── BFI: bit field insert ──────────────────────────── */
__global__ void __launch_bounds__(32)
k_bfi(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("bfi.b32 %0, %1, %0, 8, 8;" : "+r"(x) : "r"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── FSEL: float conditional select ─────────────────── */
__global__ void __launch_bounds__(32)
k_fsel(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        x = (x > 0.5f) ? y : x;  // FSEL or FMNMX
        y = y + 0.001f;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── IABS: integer absolute value ───────────────────── */
__global__ void __launch_bounds__(32)
k_iabs(volatile int *vals, volatile long long *out) {
    int x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("abs.s32 %0, %0;" : "+r"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── DMNMX: FP64 min/max ───────────────────────────── */
__global__ void __launch_bounds__(32)
k_dmnmx(volatile double *vals, volatile long long *out) {
    double x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        double mn = fmin(x, y);
        x = mn + 0.001;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── YIELD: explicit warp yield ─────────────────────── */
__global__ void __launch_bounds__(32)
k_yield(volatile int *sink, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    // PTX yield instruction
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
        asm volatile("nanosleep.u32 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 64; }
}

/* ── MEMBAR.GPU: GPU-scope fence ────────────────────── */
__global__ void __launch_bounds__(32)
k_membar_gpu(volatile int *vals, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("membar.gl;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MEMBAR.SYS: system-scope fence ─────────────────── */
__global__ void __launch_bounds__(32)
k_membar_sys(volatile int *vals, volatile long long *out) {
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("membar.sys;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[0] = (int)(t1 & 0xFF);
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ================================================================ */

struct Res { const char *name; double cyc; };

typedef void (*kup_t)(volatile unsigned*, volatile long long*);
typedef void (*kfp_t)(volatile float*, volatile long long*);
typedef void (*kip_t)(volatile int*, volatile long long*);
typedef void (*kdp_t)(volatile double*, volatile long long*);

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
static double measure_fp(kfp_t k, float *d_v, long long *d_o, long long *h) {
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

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);

    printf("==============================================\n");
    printf("  SASS Wave 5 Latency Benchmark\n");
    printf("  SM %d.%d  |  %s\n", prop.major, prop.minor, prop.name);
    printf("  Clock %d MHz  |  %d SMs  |  Chain %d\n",
           clkKHz/1000, prop.multiProcessorCount, N);
    printf("==============================================\n\n");

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4*sizeof(long long)));

    unsigned hu[4] = {0xDEADBEEFu, 0xCAFEBABEu, 0x12345678u, 0u};
    unsigned *d_uv; CHECK(cudaMalloc(&d_uv, sizeof(hu)));
    CHECK(cudaMemcpy(d_uv, hu, sizeof(hu), cudaMemcpyHostToDevice));

    float hf[4] = {0.7f, 0.3f, 0.001f, 0.0f};
    float *d_fv; CHECK(cudaMalloc(&d_fv, sizeof(hf)));
    CHECK(cudaMemcpy(d_fv, hf, sizeof(hf), cudaMemcpyHostToDevice));

    int hi[4] = {-42, 7, 3, 0};
    int *d_iv; CHECK(cudaMalloc(&d_iv, sizeof(hi)));
    CHECK(cudaMemcpy(d_iv, hi, sizeof(hi), cudaMemcpyHostToDevice));

    double hd[4] = {1.5, 0.7, 0.0, 0.0};
    double *d_dv; CHECK(cudaMalloc(&d_dv, sizeof(hd)));
    CHECK(cudaMemcpy(d_dv, hd, sizeof(hd), cudaMemcpyHostToDevice));

    printf("Running ...\n\n");
    Res r[20]; int n = 0;

    r[n++] = {"BREV",        measure_up(k_brev,  d_uv, d_out, h_out)};
    r[n++] = {"POPC",        measure_up(k_popc,  d_uv, d_out, h_out)};
    r[n++] = {"FLO(bfind)",  measure_up(k_flo,   d_uv, d_out, h_out)};
    r[n++] = {"BFE",         measure_up(k_bfe,   d_uv, d_out, h_out)};
    r[n++] = {"BFI",         measure_up(k_bfi,   d_uv, d_out, h_out)};
    r[n++] = {"FSEL",        measure_fp(k_fsel,  d_fv, d_out, h_out)};
    r[n++] = {"IABS",        measure_ip(k_iabs,  d_iv, d_out, h_out)};
    r[n++] = {"DMNMX",       measure_dp(k_dmnmx, d_dv, d_out, h_out)};
    r[n++] = {"MEMBAR.GPU",  measure_ip(k_membar_gpu, d_iv, d_out, h_out)};
    r[n++] = {"MEMBAR.SYS",  measure_ip(k_membar_sys, d_iv, d_out, h_out)};

    // YIELD (shorter chain)
    {
        k_yield<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
        double tot = 0;
        for (int rr = 0; rr < 20; rr++) {
            k_yield<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 2*sizeof(long long), cudaMemcpyDeviceToHost);
            tot += (double)h_out[0]/(double)h_out[1];
        }
        r[n++] = {"YIELD(nanosleep0)", tot/20};
    }

    printf("%-20s %14s %14s\n", "Instruction", "Latency (cyc)", "Pipeline");
    printf("%-20s %14s %14s\n", "--------------------", "--------------", "--------------");
    for (int i = 0; i < n; i++) {
        const char *pipe;
        if (r[i].cyc < 3.0)       pipe = "INT (fast)";
        else if (r[i].cyc < 6.0)  pipe = "INT/FP32";
        else if (r[i].cyc < 15.0) pipe = "Multi-cycle";
        else if (r[i].cyc < 50.0) pipe = "SFU/FP64";
        else if (r[i].cyc < 200.0) pipe = "Memory/Fence";
        else                        pipe = "Scheduler";
        printf("%-20s %14.2f %14s\n", r[i].name, r[i].cyc, pipe);
    }

    cudaFree(d_out); cudaFree(d_uv); cudaFree(d_fv); cudaFree(d_iv); cudaFree(d_dv);
    return 0;
}
