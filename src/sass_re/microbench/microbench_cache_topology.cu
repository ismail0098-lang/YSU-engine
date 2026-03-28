/*
 * SASS RE: Cache Topology Characterization Microbenchmark
 *
 * Measures memory latency as a function of working set size to map the
 * L1/L2/DRAM boundary on Ada Lovelace (SM 8.9).
 *
 * Expected topology (RTX 4070 Ti, AD104):
 *   L1 data cache: 128 KB per SM, ~33 cycle latency
 *   L2 cache: 48 MB shared across all SMs, ~200 cycle latency
 *   GDDR6X DRAM: 12 GB, ~400+ cycle latency, 504 GB/s peak BW
 *   Shared memory: 48-100 KB per SM, ~28 cycle latency (32 banks)
 *   Cache line: 128 bytes (32 x 4-byte sectors)
 *   L2 sector: 32 bytes
 *
 * Method: Pointer-chase with controlled working set sizes.
 *   1. Allocate array of size W bytes
 *   2. Initialize a permutation that visits all elements (stride-coprime)
 *   3. Chase N pointers and measure total clock64 cycles
 *   4. Report cycles/chase as a function of W
 *
 * When W fits in L1, latency is ~33 cycles.
 * When W exceeds L1 but fits L2, latency is ~200 cycles.
 * When W exceeds L2, latency is ~400+ cycles (DRAM).
 *
 * Build: nvcc -arch=sm_89 -O1 -o cache_topo microbench_cache_topology.cu
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

#define CHASE_LEN 4096

/* Pointer chase kernel: follows a permutation through memory.
 * Each element is an index pointing to the next element.
 * The permutation is constructed so that consecutive accesses
 * are far apart in memory (defeats HW prefetcher). */
__global__ void __launch_bounds__(32)
k_chase(const int *arr, volatile long long *out, int chase_len) {
    int idx = 0;
    long long t0, t1;

    // Warmup: run the chase once to fill caches
    for (int j = 0; j < chase_len; j++)
        idx = arr[idx];

    // Reset to known start
    idx = 0;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int j = 0; j < chase_len; j++)
        idx = arr[idx];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = chase_len;
        out[2] = idx;  // prevent DCE
    }
}

/* Build a permutation that visits all N elements with maximum spatial distance.
 * Uses stride = coprime with N that is close to N/3 (maximizes average distance). */
static void build_permutation(int *perm, int n) {
    // Find a stride coprime with n, close to n/3
    int stride = n / 3;
    while (stride > 1) {
        // Check coprime via GCD
        int a = stride, b = n;
        while (b) { int t = b; b = a % b; a = t; }
        if (a == 1) break;
        stride--;
    }
    if (stride <= 1) stride = 1;

    // Build permutation: perm[i] = (i + stride) % n
    for (int i = 0; i < n; i++)
        perm[i] = (i + stride) % n;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);

    printf("==============================================\n");
    printf("  Cache Topology Characterization\n");
    printf("  SM %d.%d  |  %s\n", prop.major, prop.minor, prop.name);
    printf("  L2 cache: %d KB  |  Chase length: %d\n",
           prop.l2CacheSize / 1024, CHASE_LEN);
    printf("  Clock %d MHz  |  %d SMs\n", clkKHz/1000, prop.multiProcessorCount);
    printf("==============================================\n\n");

    long long *d_out, h_out[4];
    CHECK(cudaMalloc(&d_out, 4 * sizeof(long long)));

    // Working set sizes: 4 KB to 128 MB (powers of 2 + intermediate)
    int sizes_kb[] = {
        4, 8, 16, 32, 64, 96, 128,         // L1 range (128 KB/SM)
        256, 512, 1024, 2048, 4096,         // L1->L2 transition
        8192, 16384, 24576, 32768, 49152,   // L2 range (48 MB)
        65536, 98304, 131072                 // L2->DRAM transition
    };
    int n_sizes = sizeof(sizes_kb) / sizeof(sizes_kb[0]);

    printf("%-12s %12s %12s %12s\n",
           "Working Set", "Elements", "Latency(cy)", "Regime");
    printf("%-12s %12s %12s %12s\n",
           "------------", "------------", "------------", "------------");

    for (int si = 0; si < n_sizes; si++) {
        int ws_kb = sizes_kb[si];
        int ws_bytes = ws_kb * 1024;
        int n_elements = ws_bytes / (int)sizeof(int);

        // Build permutation on host
        int *h_arr = (int*)malloc(ws_bytes);
        if (!h_arr) continue;
        build_permutation(h_arr, n_elements);

        // Copy to device
        int *d_arr;
        CHECK(cudaMalloc(&d_arr, ws_bytes));
        CHECK(cudaMemcpy(d_arr, h_arr, ws_bytes, cudaMemcpyHostToDevice));

        // Run chase (averaged over 5 runs)
        double total_cyc = 0.0;
        const int RUNS = 5;
        for (int r = 0; r < RUNS; r++) {
            k_chase<<<1, 32>>>(d_arr, d_out, CHASE_LEN);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, 3 * sizeof(long long), cudaMemcpyDeviceToHost));
            total_cyc += (double)h_out[0] / (double)h_out[1];
        }
        double avg_cyc = total_cyc / RUNS;

        // Classify regime
        const char *regime;
        if (avg_cyc < 50)       regime = "L1";
        else if (avg_cyc < 150) regime = "L1->L2";
        else if (avg_cyc < 350) regime = "L2";
        else                    regime = "DRAM";

        printf("%8d KB %12d %12.2f %12s\n",
               ws_kb, n_elements, avg_cyc, regime);

        cudaFree(d_arr);
        free(h_arr);
    }

    printf("\n--- Expected boundaries ---\n");
    printf("L1:   128 KB/SM, ~33 cy latency\n");
    printf("L2:   %d KB shared, ~200 cy latency\n", prop.l2CacheSize / 1024);
    printf("DRAM: 12 GB, ~400+ cy latency\n");

    cudaFree(d_out);
    return 0;
}
