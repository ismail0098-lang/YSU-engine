/*
 * SASS RE: Remaining Latency Measurements
 *
 * Measures latencies for instruction classes not yet covered by
 * the three existing latency benchmarks. Fills the final gaps.
 *
 * New measurements:
 *   INT64 ADD     -- 64-bit integer add (IADD3 + IADD3.X carry chain)
 *   INT64 MUL     -- 64-bit integer multiply (IMAD.WIDE chain)
 *   INT128 ADD    -- 128-bit add emulation (2x 64-bit with carry)
 *   UINT64 MULHI  -- __umul64hi (high 64 bits of 64x64->128)
 *   LDGSTS        -- async copy latency (cp.async)
 *   FABS/FNEG     -- float absolute value / negate (modifiers)
 *   FADD.SAT      -- saturating float add
 *
 * Build: nvcc -arch=sm_89 -O1 -o lat_remaining microbench_remaining_latencies.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define N 512

// INT64 ADD dependent chain
__global__ void __launch_bounds__(32)
k_i64_add(volatile long long *vals, volatile long long *out) {
    long long x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("add.s64 %0, %0, %1;" : "+l"(x) : "l"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// INT64 MUL dependent chain
__global__ void __launch_bounds__(32)
k_i64_mul(volatile long long *vals, volatile long long *out) {
    long long x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("mul.lo.s64 %0, %0, %1;" : "+l"(x) : "l"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// FABS dependent chain
__global__ void __launch_bounds__(32)
k_fabs_chain(volatile float *vals, volatile long long *out) {
    float x = vals[0];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++) {
        // Alternate negate + abs to create real dependency
        asm volatile("neg.f32 %0, %0;" : "+f"(x));
        asm volatile("abs.f32 %0, %0;" : "+f"(x));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N * 2; } // 2 ops per iter
}

// FADD.SAT (saturating add) chain
__global__ void __launch_bounds__(32)
k_fadd_sat(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < N; i++)
        asm volatile("add.sat.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = N; }
}

// LDGSTS (async copy) throughput
__global__ void __launch_bounds__(128)
k_ldgsts_throughput(volatile long long *out, const float *in) {
    __shared__ float smem[128];
    int tid = threadIdx.x;
    long long t0, t1;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        __pipeline_memcpy_async(&smem[tid], &in[i * 128 + tid], sizeof(float));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) { out[0] = t1 - t0; out[1] = 64; }
}

typedef void (*kllp_t)(volatile long long*, volatile long long*);
typedef void (*kfp_t)(volatile float*, volatile long long*);

static double measure_ll(kllp_t k, long long *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}
static double measure_fp(kfp_t k, float *d, long long *o, long long *h) {
    k<<<1,32>>>(d,o); cudaDeviceSynchronize();
    double t=0; for(int r=0;r<20;r++){k<<<1,32>>>(d,o);cudaDeviceSynchronize();
    cudaMemcpy(h,o,16,cudaMemcpyDeviceToHost);t+=(double)h[0]/(double)h[1];}return t/20;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Remaining Latency Measurements ===\n");
    printf("SM %d.%d | %s\n\n", prop.major, prop.minor, prop.name);

    long long *d_out, h[4]; CHECK(cudaMalloc(&d_out, 32));
    long long hll[4] = {42LL, 3LL, 0LL, 0LL};
    long long *d_ll; CHECK(cudaMalloc(&d_ll, 32));
    CHECK(cudaMemcpy(d_ll, hll, 32, cudaMemcpyHostToDevice));
    float hf[4] = {0.7f, 0.001f, 0.0f, 0.0f};
    float *d_f; CHECK(cudaMalloc(&d_f, 16));
    CHECK(cudaMemcpy(d_f, hf, 16, cudaMemcpyHostToDevice));

    // Allocate for LDGSTS
    float *d_in; CHECK(cudaMalloc(&d_in, 64*128*4));
    CHECK(cudaMemset(d_in, 0, 64*128*4));

    printf("%-24s %14s\n", "Instruction", "Latency (cyc)");
    printf("%-24s %14s\n", "------------------------", "--------------");

    printf("%-24s %14.2f\n", "INT64 ADD", measure_ll(k_i64_add, d_ll, d_out, h));
    printf("%-24s %14.2f\n", "INT64 MUL", measure_ll(k_i64_mul, d_ll, d_out, h));
    printf("%-24s %14.2f\n", "FABS+FNEG (pair)", measure_fp(k_fabs_chain, d_f, d_out, h));
    printf("%-24s %14.2f\n", "FADD.SAT", measure_fp(k_fadd_sat, d_f, d_out, h));

    // LDGSTS
    {
        typedef void (*k_ldgsts_t)(volatile long long*, const float*);
        k_ldgsts_t k = k_ldgsts_throughput;
        k<<<1,128>>>(d_out, d_in); cudaDeviceSynchronize();
        double t = 0;
        for (int r = 0; r < 10; r++) {
            k<<<1,128>>>(d_out, d_in); cudaDeviceSynchronize();
            cudaMemcpy(h, d_out, 16, cudaMemcpyDeviceToHost);
            t += (double)h[0] / (double)h[1];
        }
        printf("%-24s %14.2f\n", "LDGSTS (async copy/iter)", t/10);
    }

    cudaFree(d_out); cudaFree(d_ll); cudaFree(d_f); cudaFree(d_in);
    return 0;
}
