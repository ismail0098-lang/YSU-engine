/*
 * SASS RE Probe: Precise L1/L2/DRAM Boundary Measurement
 * Isolates: Pointer-chase at fine-grained working set sizes
 *
 * Improves on microbench_cache_topology.cu by using a proper random
 * permutation (Fisher-Yates shuffle, not stride-coprime) to avoid
 * cache set aliasing artifacts.
 *
 * Tests at 1 KB granularity around the expected boundaries:
 *   L1:   ~128 KB per SM (test 16-256 KB in 16 KB steps)
 *   L2:   ~48 MB shared  (test 32-64 MB in 4 MB steps)
 *   DRAM: > 48 MB
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

#define CHASE_LEN 8192

extern "C" __global__ void __launch_bounds__(1)
k_chase_single(const int *arr, volatile long long *out, int n) {
    int idx = 0;
    // Warmup
    for (int j = 0; j < CHASE_LEN; j++) idx = arr[idx];
    idx = 0;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int j = 0; j < CHASE_LEN; j++) idx = arr[idx];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[1] = CHASE_LEN;
    out[2] = idx;
}

// Fisher-Yates shuffle to create a random hamiltonian cycle
static void build_random_cycle(int *perm, int n, unsigned seed) {
    // Initialize identity
    for (int i = 0; i < n; i++) perm[i] = i;
    // Fisher-Yates to create random permutation
    for (int i = n - 1; i > 0; i--) {
        seed = seed * 1664525u + 1013904223u;
        int j = seed % (i + 1);
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    // Convert permutation to linked list (cycle)
    int *cycle = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n - 1; i++) cycle[perm[i]] = perm[i + 1];
    cycle[perm[n-1]] = perm[0];
    for (int i = 0; i < n; i++) perm[i] = cycle[i];
    free(cycle);
}

#ifndef SASS_RE_EMBEDDED_RUNNER
int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== L1/L2/DRAM Boundary Measurement (random cycle) ===\n");
    printf("SM %d.%d | %s | L2: %d KB\n\n", prop.major, prop.minor,
           prop.name, prop.l2CacheSize / 1024);

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4 * sizeof(long long)));

    // Fine-grained sizes around expected boundaries
    int sizes_kb[] = {
        // L1 region (128 KB/SM)
        8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256,
        // L1->L2 transition
        512, 1024, 2048, 4096,
        // L2 region (48 MB = 49152 KB)
        8192, 16384, 24576, 32768, 40960, 45056, 47104, 48128,
        // L2 boundary (exact)
        49152,
        // DRAM
        50176, 51200, 53248, 57344, 65536, 81920, 131072
    };
    int n_sizes = sizeof(sizes_kb) / sizeof(sizes_kb[0]);

    printf("%-12s %12s %12s\n", "Working Set", "Latency(cy)", "Regime");
    printf("%-12s %12s %12s\n", "------------", "------------", "------------");

    for (int si = 0; si < n_sizes; si++) {
        int ws_kb = sizes_kb[si];
        int ws_bytes = ws_kb * 1024;
        int n_elements = ws_bytes / (int)sizeof(int);

        int *h_arr = (int*)malloc(ws_bytes);
        if (!h_arr) continue;
        build_random_cycle(h_arr, n_elements, 42 + si);

        int *d_arr;
        CHECK(cudaMalloc(&d_arr, ws_bytes));
        CHECK(cudaMemcpy(d_arr, h_arr, ws_bytes, cudaMemcpyHostToDevice));

        // Single-thread launch to avoid L2 set pollution from multi-SM
        double total_cyc = 0.0;
        const int RUNS = 3;
        for (int r = 0; r < RUNS; r++) {
            k_chase_single<<<1, 1>>>(d_arr, d_out, n_elements);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, 3 * sizeof(long long), cudaMemcpyDeviceToHost));
            total_cyc += (double)h_out[0] / (double)h_out[1];
        }
        double avg_cyc = total_cyc / RUNS;

        const char *regime;
        if (avg_cyc < 40)       regime = "L1";
        else if (avg_cyc < 100) regime = "L1/L2";
        else if (avg_cyc < 300) regime = "L2";
        else                    regime = "DRAM";

        printf("%8d KB %12.2f %12s\n", ws_kb, avg_cyc, regime);

        cudaFree(d_arr);
        free(h_arr);
    }

    cudaFree(d_out);
    return 0;
}
#endif
