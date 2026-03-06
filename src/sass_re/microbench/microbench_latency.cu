/*
 * SASS RE: Instruction Latency Microbenchmark v4
 *
 * KEY FIX: Initial values are loaded from global memory AND the chain
 * result is STORED BACK to global memory.  This makes both the chain
 * inputs and outputs truly live so ptxas cannot constant-fold or DCE
 * the chain.  Previous versions were entirely eliminated by ptxas.
 *
 * Uses clock64 for timing.  Launches 1 warp (32 threads).
 *
 * Build: nvcc -arch=sm_89 -O1 -allow-unsupported-compiler \
 *             -o latency_bench.exe microbench_latency.cu
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

#define N 512   /* chain length */

/* ────────────────────────────────────────────────────────────
 * Each kernel:
 *  - loads x (and helpers) from vals[]
 *  - times a dependent chain of N instructions
 *  - stores t1-t0 to out[0], N to out[1]
 *  - ***stores x back to vals[3]*** ← keeps chain live
 * ──────────────────────────────────────────────────────────── */

/* helper: reinterpret float as int for store */
__device__ __forceinline__ int f2i(float f) {
    int r; asm("mov.b32 %0, %1;" : "=r"(r) : "f"(f)); return r;
}

/* ── FADD ───────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_fadd(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;   /* keep chain live */
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── FMUL ───────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_fmul(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("mul.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── FFMA ───────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_ffma(volatile float *vals, volatile long long *out) {
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

/* ── IADD3 ──────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_iadd3(volatile int *vals, volatile long long *out) {
    int x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("add.s32 %0, %0, %1;" : "+r"(x) : "r"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── IMAD ───────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_imad(volatile int *vals, volatile long long *out) {
    int x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.RCP ──────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_mufu_rcp(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.RSQ ──────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_mufu_rsq(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.SIN ──────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_mufu_sin(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("sin.approx.f32 %0, %0;" : "+f"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.EX2 ──────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_mufu_ex2(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("ex2.approx.f32 %0, %0;" : "+f"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── MUFU.LG2 ──────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_mufu_lg2(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("lg2.approx.f32 %0, %0;" : "+f"(x));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── LOP3 ──────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_lop3(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0xCA;" : "+r"(x) : "r"(y), "r"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── SHF ───────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_shf(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("shf.r.wrap.b32 %0, %0, %1, 4;" : "+r"(x) : "r"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── PRMT ──────────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_prmt(volatile unsigned *vals, volatile long long *out) {
    unsigned x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(x) : "r"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── F2I + I2F ─────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_cvt(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
        int ix;
        asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(x));
        asm volatile("cvt.rn.f32.s32  %0, %1;" : "=f"(x)  : "r"(ix));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── SHFL.BFLY ─────────────────────────────────────── */
__global__ void __launch_bounds__(32)
k_shfl(volatile int *sink, volatile long long *out) {
    int x = threadIdx.x + sink[0];  /* runtime-opaque */
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
#if __CUDA_ARCH__ >= 700
        x = __shfl_xor_sync(0xFFFFFFFF, x, 1);
#else
        x = __shfl_xor(x, 1);
#endif
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    sink[3] = x;   /* keep live */
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

/* ── LDG pointer chase ─────────────────────────────── */
__global__ void __launch_bounds__(32)
k_ldg(volatile long long *out, const int *chase) {
    int idx = 0;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int j = 0; j < N; j++)
        idx = chase[idx];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; out[2] = idx; }
}

/* ── LDS pointer chase ─────────────────────────────── */
__global__ void __launch_bounds__(32)
k_lds(volatile long long *out) {
    __shared__ int s[256];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) s[i] = (i + 1) & 255;
    __syncthreads();

    int idx = tid;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int j = 0; j < N; j++)
        idx = s[idx];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; out[2] = idx; }
}

/* ================================================================
 *  Host
 * ================================================================ */

struct Res { const char *name; double cyc; };

typedef void (*kfp_t)(volatile float*,    volatile long long*);
typedef void (*kip_t)(volatile int*,      volatile long long*);
typedef void (*kup_t)(volatile unsigned*, volatile long long*);

static double measure_fp(kfp_t k, float *d_v, long long *d_o, long long *h) {
    k<<<1,32>>>(d_v, d_o); cudaDeviceSynchronize();          /* warmup */
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

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);

    printf("==============================================\n");
    printf("  SASS Instruction Latency Benchmark v4\n");
    printf("  SM %d.%d  |  %s\n", prop.major, prop.minor, prop.name);
    printf("  Clock %d MHz  |  %d SMs  |  Chain %d\n",
           clkKHz/1000, prop.multiProcessorCount, N);
    printf("==============================================\n\n");

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4*sizeof(long long)));

    /* float params: vals[0]=1.0, [1]=0.5, [2]=0.001 */
    float hf[4] = {1.0f, 0.5f, 0.001f, 0.0f};
    float *d_fv; CHECK(cudaMalloc(&d_fv, sizeof(hf)));
    CHECK(cudaMemcpy(d_fv, hf, sizeof(hf), cudaMemcpyHostToDevice));

    int hi[4] = {1, 1, 1, 0};
    int *d_iv; CHECK(cudaMalloc(&d_iv, sizeof(hi)));
    CHECK(cudaMemcpy(d_iv, hi, sizeof(hi), cudaMemcpyHostToDevice));

    unsigned hu[4] = {0xDEADBEEFu, 0xCAFEBABEu, 0x12345678u, 0u};
    unsigned *d_uv; CHECK(cudaMalloc(&d_uv, sizeof(hu)));
    CHECK(cudaMemcpy(d_uv, hu, sizeof(hu), cudaMemcpyHostToDevice));

    printf("Running ...\n\n");

    Res r[20]; int n = 0;

    r[n++] = {"FADD",     measure_fp(k_fadd,     d_fv, d_out, h_out)};
    r[n++] = {"FMUL",     measure_fp(k_fmul,     d_fv, d_out, h_out)};
    r[n++] = {"FFMA",     measure_fp(k_ffma,     d_fv, d_out, h_out)};
    r[n++] = {"IADD3",    measure_ip(k_iadd3,    d_iv, d_out, h_out)};
    r[n++] = {"IMAD",     measure_ip(k_imad,     d_iv, d_out, h_out)};
    r[n++] = {"MUFU.RCP", measure_fp(k_mufu_rcp, d_fv, d_out, h_out)};
    r[n++] = {"MUFU.RSQ", measure_fp(k_mufu_rsq, d_fv, d_out, h_out)};
    r[n++] = {"MUFU.SIN", measure_fp(k_mufu_sin, d_fv, d_out, h_out)};
    r[n++] = {"MUFU.EX2", measure_fp(k_mufu_ex2, d_fv, d_out, h_out)};
    r[n++] = {"MUFU.LG2", measure_fp(k_mufu_lg2, d_fv, d_out, h_out)};
    r[n++] = {"LOP3",     measure_up(k_lop3,     d_uv, d_out, h_out)};
    r[n++] = {"SHF",      measure_up(k_shf,      d_uv, d_out, h_out)};
    r[n++] = {"PRMT",     measure_up(k_prmt,     d_uv, d_out, h_out)};
    r[n++] = {"F2I+I2F",  measure_fp(k_cvt,      d_fv, d_out, h_out)};

    /* SHFL */
    {
        k_shfl<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
        double tot = 0;
        for (int rr = 0; rr < 20; rr++) {
            k_shfl<<<1,32>>>(d_iv, d_out); cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 2*sizeof(long long), cudaMemcpyDeviceToHost);
            tot += (double)h_out[0]/(double)h_out[1];
        }
        r[n++] = {"SHFL.BFLY", tot/20};
    }

    /* LDG chase */
    {
        int *d_chase, hc[1024];
        for (int i = 0; i < 1024; i++) hc[i] = (i+1)%1024;
        CHECK(cudaMalloc(&d_chase, sizeof(hc)));
        CHECK(cudaMemcpy(d_chase, hc, sizeof(hc), cudaMemcpyHostToDevice));
        k_ldg<<<1,32>>>(d_out, d_chase); cudaDeviceSynchronize();
        double tot = 0;
        for (int rr = 0; rr < 20; rr++) {
            k_ldg<<<1,32>>>(d_out, d_chase); cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 2*sizeof(long long), cudaMemcpyDeviceToHost);
            tot += (double)h_out[0]/(double)h_out[1];
        }
        r[n++] = {"LDG chase", tot/20};
        cudaFree(d_chase);
    }

    /* LDS chase */
    {
        k_lds<<<1,32>>>(d_out); cudaDeviceSynchronize();
        double tot = 0;
        for (int rr = 0; rr < 20; rr++) {
            k_lds<<<1,32>>>(d_out); cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 2*sizeof(long long), cudaMemcpyDeviceToHost);
            tot += (double)h_out[0]/(double)h_out[1];
        }
        r[n++] = {"LDS chase", tot/20};
    }

    /* ── print ── */
    printf("%-20s %14s\n", "Instruction", "Latency (cyc)");
    printf("%-20s %14s\n", "--------------------", "--------------");
    for (int i = 0; i < n; i++)
        printf("%-20s %14.2f\n", r[i].name, r[i].cyc);

    if (prop.major >= 8) {
        printf("\n--- Expected (Ada / SM 8.x) ---\n");
        printf("FP32 arith  : ~4 cyc    INT32 arith  : ~4-5 cyc\n");
        printf("MUFU (SFU)  : ~4-8 cyc  Bitwise      : ~1-4 cyc\n");
        printf("SHFL        : ~2-4 cyc  F2I/I2F      : ~4-6 cyc\n");
        printf("LDS (L0)    : ~20-28    LDG (L1 hit) : ~33-36\n");
    } else {
        printf("\n--- Expected (Pascal / SM 6.x) ---\n");
        printf("FP32 arith  : ~6 cyc    INT32 arith  : ~6 cyc\n");
        printf("MUFU (SFU)  : ~8 cyc    Bitwise      : ~6 cyc\n");
        printf("SHFL        : ~2-5 cyc  F2I/I2F      : ~6-8 cyc\n");
        printf("LDS (L0)    : ~22-28    LDG (L1 hit) : ~80-200\n");
    }

    cudaFree(d_out); cudaFree(d_fv); cudaFree(d_iv); cudaFree(d_uv);
    return 0;
}
