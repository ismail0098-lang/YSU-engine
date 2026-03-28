/*
 * SASS RE Probe: Global Memory Coalescing Patterns
 * Isolates: Bandwidth achieved at different access strides and patterns
 *
 * Measures effective global memory bandwidth for:
 *   Stride 1 (coalesced): 128-byte transaction per warp
 *   Stride 2: 2 transactions per warp
 *   Stride 4: 4 transactions
 *   Stride 32: 32 transactions (worst case: 1 useful byte per 128-byte line)
 *   Random: scattered access (cache-line divergence)
 *
 * Ada Lovelace SM 8.9: 504 GB/s peak DRAM BW, 128-byte cache lines.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define ELEMS (1024*1024)
#define ITERS 100

template<int STRIDE>
__global__ void __launch_bounds__(256)
k_stride_read(float *out, const float *in, int n) {
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * STRIDE;
    float acc = 0.0f;
    #pragma unroll 1
    for (int iter = 0; iter < ITERS; iter++) {
        int idx = (base + iter * 256 * STRIDE) % n;
        acc += in[idx];
    }
    out[threadIdx.x + blockIdx.x * blockDim.x] = acc;
}

// Random access via LCG permutation
__global__ void __launch_bounds__(256)
k_random_read(float *out, const float *in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0.0f;
    unsigned rng = (unsigned)idx * 1664525u + 1013904223u;
    #pragma unroll 1
    for (int iter = 0; iter < ITERS; iter++) {
        rng = rng * 1664525u + 1013904223u;
        acc += in[rng % n];
    }
    out[idx] = acc;
}

// Sequential write (coalesced stores)
__global__ void __launch_bounds__(256)
k_coalesced_write(float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    out[i] = (float)i;
}

// Strided write (non-coalesced stores)
__global__ void __launch_bounds__(256)
k_strided_write(float *out, int n, int stride) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (i * stride) % n;
    out[idx] = (float)i;
}
