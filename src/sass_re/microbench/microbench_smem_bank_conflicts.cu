/*
 * SASS RE: Shared Memory Bank Conflict Characterization
 *
 * Measures LDS latency as a function of access stride to characterize
 * bank conflict penalties on Ada Lovelace SM 8.9.
 *
 * Ada shared memory: 32 banks, 4-byte bank width, 48-100 KB per SM.
 * Bank index = (byte_address / 4) % 32
 *
 * Expected behavior:
 *   Stride 1:  No conflicts (each thread hits a different bank)
 *   Stride 2:  2-way conflict (2 threads share each bank) -> 2x latency
 *   Stride 4:  4-way conflict -> 4x latency
 *   Stride 8:  8-way conflict -> 8x latency
 *   Stride 16: 16-way conflict -> 16x latency
 *   Stride 32: 32-way conflict (all threads hit bank 0) -> 32x latency
 *   Broadcast: All threads read same address -> NO conflict (hardware broadcast)
 *
 * Build: nvcc -arch=sm_89 -O1 -o smem_bench microbench_smem_bank_conflicts.cu
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

#define N 4096

/* Generic stride-access kernel.
 * Each thread reads smem[threadIdx.x * stride % ARRAY_SIZE] N times. */
template<int STRIDE>
__global__ void __launch_bounds__(32)
k_smem_stride(volatile long long *out) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;

    // Initialize shared memory
    for (int i = tid; i < 1024; i += 32)
        smem[i] = (float)i;
    __syncthreads();

    float acc = 0.0f;
    int idx = (tid * STRIDE) % 1024;

    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        acc += smem[idx];
        idx = ((idx + STRIDE) % 1024);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Prevent DCE
    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = N;
        out[2] = (long long)__float_as_int(acc);
    }
}

/* Broadcast: all threads read same address (should be conflict-free) */
__global__ void __launch_bounds__(32)
k_smem_broadcast(volatile long long *out) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    if (tid < 32) smem[tid] = (float)tid;
    __syncthreads();

    float acc = 0.0f;

    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        acc += smem[0];  // All threads read address 0 (broadcast)
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = N;
        out[2] = (long long)__float_as_int(acc);
    }
}

/* XOR-swizzle pattern: smem[tid ^ delta] (used in FFT, warp shuffle emulation) */
template<int DELTA>
__global__ void __launch_bounds__(32)
k_smem_xor(volatile long long *out) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();

    float acc = 0.0f;

    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        acc += smem[tid ^ DELTA];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = N;
        out[2] = (long long)__float_as_int(acc);
    }
}

static double measure(void (*k)(volatile long long*), long long *d_o, long long *h) {
    k<<<1,32>>>(d_o); cudaDeviceSynchronize();
    double tot = 0; const int R = 10;
    for (int r = 0; r < R; r++) {
        k<<<1,32>>>(d_o); cudaDeviceSynchronize();
        cudaMemcpy(h, d_o, 3*sizeof(long long), cudaMemcpyDeviceToHost);
        tot += (double)h[0] / (double)h[1];
    }
    return tot / R;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Shared Memory Bank Conflict Characterization ===\n");
    printf("SM %d.%d | %s | Shared: %zu KB | 32 banks\n\n",
           prop.major, prop.minor, prop.name, prop.sharedMemPerBlock / 1024);

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4 * sizeof(long long)));

    printf("%-24s %12s %12s\n", "Pattern", "Cycles/load", "Conflict");
    printf("%-24s %12s %12s\n", "------------------------", "------------", "------------");

    double base;

    base = measure(k_smem_stride<1>, d_out, h_out);
    printf("%-24s %12.2f %12s\n", "Stride 1 (conflict-free)", base, "none");

    double s2 = measure(k_smem_stride<2>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Stride 2 (2-way)", s2, s2/base);

    double s4 = measure(k_smem_stride<4>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Stride 4 (4-way)", s4, s4/base);

    double s8 = measure(k_smem_stride<8>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Stride 8 (8-way)", s8, s8/base);

    double s16 = measure(k_smem_stride<16>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Stride 16 (16-way)", s16, s16/base);

    double s32 = measure(k_smem_stride<32>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Stride 32 (32-way)", s32, s32/base);

    double bc = measure(k_smem_broadcast, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "Broadcast (all same)", bc, bc/base);

    printf("\n--- XOR swizzle patterns ---\n");
    double x1 = measure(k_smem_xor<1>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "XOR 1 (neighbor swap)", x1, x1/base);

    double x16 = measure(k_smem_xor<16>, d_out, h_out);
    printf("%-24s %12.2f %12.1fx\n", "XOR 16 (half-warp swap)", x16, x16/base);

    printf("\n--- Expected ---\n");
    printf("Stride 1 (base): ~28 cy (LDS latency from SASS RE)\n");
    printf("N-way conflict: ~N * base_latency (serialized bank access)\n");
    printf("Broadcast: ~base_latency (hardware multicast, no conflict)\n");

    cudaFree(d_out);
    return 0;
}
