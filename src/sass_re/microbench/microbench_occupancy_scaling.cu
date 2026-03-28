/*
 * SASS RE: Occupancy Scaling -- Latency Hiding Effectiveness
 *
 * Measures how LDG latency and FFMA throughput change as a function of
 * active warps per SM. This reveals the warp scheduler's ability to hide
 * memory latency with compute from other warps.
 *
 * Method: Launch kernels with __launch_bounds__ controlling max warps.
 * Each kernel does a mixed memory+compute workload:
 *   1. Load 19 floats from global memory (simulates D3Q19 distribution read)
 *   2. Compute a 57-FFMA BGK collision
 *   3. Store 19 floats back
 *
 * By varying the number of blocks launched (1-16 per SM), we control
 * the number of active warps and measure total throughput.
 *
 * Build: nvcc -arch=sm_89 -O1 -o occupancy_bench microbench_occupancy_scaling.cu
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

/* Simplified D3Q19 BGK step: 19 loads + collision + 19 stores per cell */
__global__ void __launch_bounds__(128, 1)  /* 1 block/SM max */
k_lbm_occupancy_1(float *out, const float *in, int n_cells, int steps) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    for (int s = 0; s < steps; s++) {
        float f[19];
        #pragma unroll
        for (int d = 0; d < 19; d++)
            f[d] = in[d * n_cells + cell];

        float rho = 0.0f;
        #pragma unroll
        for (int d = 0; d < 19; d++) rho += f[d];

        float inv_tau = 1.0f / 0.6f;
        #pragma unroll
        for (int d = 0; d < 19; d++) {
            float feq = rho / 19.0f;
            f[d] -= (f[d] - feq) * inv_tau;
        }

        #pragma unroll
        for (int d = 0; d < 19; d++)
            out[d * n_cells + cell] = f[d];
    }
}

__global__ void __launch_bounds__(128, 4)  /* 4 blocks/SM max */
k_lbm_occupancy_4(float *out, const float *in, int n_cells, int steps) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    for (int s = 0; s < steps; s++) {
        float f[19];
        #pragma unroll
        for (int d = 0; d < 19; d++)
            f[d] = in[d * n_cells + cell];

        float rho = 0.0f;
        #pragma unroll
        for (int d = 0; d < 19; d++) rho += f[d];

        float inv_tau = 1.0f / 0.6f;
        #pragma unroll
        for (int d = 0; d < 19; d++) {
            float feq = rho / 19.0f;
            f[d] -= (f[d] - feq) * inv_tau;
        }

        #pragma unroll
        for (int d = 0; d < 19; d++)
            out[d * n_cells + cell] = f[d];
    }
}

__global__ void __launch_bounds__(128, 8)  /* 8 blocks/SM max */
k_lbm_occupancy_8(float *out, const float *in, int n_cells, int steps) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    for (int s = 0; s < steps; s++) {
        float f[19];
        #pragma unroll
        for (int d = 0; d < 19; d++)
            f[d] = in[d * n_cells + cell];

        float rho = 0.0f;
        #pragma unroll
        for (int d = 0; d < 19; d++) rho += f[d];

        float inv_tau = 1.0f / 0.6f;
        #pragma unroll
        for (int d = 0; d < 19; d++) {
            float feq = rho / 19.0f;
            f[d] -= (f[d] - feq) * inv_tau;
        }

        #pragma unroll
        for (int d = 0; d < 19; d++)
            out[d * n_cells + cell] = f[d];
    }
}

static double run_kernel(void (*k)(float*, const float*, int, int),
                         float *d_out, float *d_in, int n_cells,
                         int blocks, int steps) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    k<<<blocks, 128>>>(d_out, d_in, n_cells, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    k<<<blocks, 128>>>(d_out, d_in, n_cells, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    // MLUPS = n_cells * steps / (ms * 1e3)
    return (double)n_cells * steps / (ms * 1e3);
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int n_sms = prop.multiProcessorCount;

    printf("=== Occupancy Scaling: Latency Hiding Effectiveness ===\n");
    printf("SM %d.%d | %s | %d SMs\n\n", prop.major, prop.minor, prop.name, n_sms);

    int n_cells = 128 * 128 * 128;  // 2M cells (128^3 grid)
    int steps = 10;

    float *d_in, *d_out;
    size_t dist_bytes = (size_t)19 * n_cells * sizeof(float);
    CHECK(cudaMalloc(&d_in, dist_bytes));
    CHECK(cudaMalloc(&d_out, dist_bytes));
    CHECK(cudaMemset(d_in, 0, dist_bytes));

    printf("Grid: 128^3 (%d cells) | %d steps per measurement\n", n_cells, steps);
    printf("Kernel: FP32 BGK, 19 loads + 57 FMA + 19 stores per cell\n\n");

    printf("%-28s %10s %12s %12s\n",
           "Configuration", "Blocks", "MLUPS", "Warps/SM");
    printf("%-28s %10s %12s %12s\n",
           "----------------------------", "----------", "------------", "------------");

    // 1 block/SM
    int blocks_1 = n_sms;
    double mlups_1 = run_kernel(k_lbm_occupancy_1, d_out, d_in, n_cells, blocks_1, steps);
    printf("%-28s %10d %12.1f %12d\n", "__launch_bounds__(128, 1)", blocks_1, mlups_1, 4);

    // 4 blocks/SM
    int blocks_4 = n_sms * 4;
    double mlups_4 = run_kernel(k_lbm_occupancy_4, d_out, d_in, n_cells, blocks_4, steps);
    printf("%-28s %10d %12.1f %12d\n", "__launch_bounds__(128, 4)", blocks_4, mlups_4, 16);

    // 8 blocks/SM (high occupancy)
    int blocks_8 = n_sms * 8;
    double mlups_8 = run_kernel(k_lbm_occupancy_8, d_out, d_in, n_cells, blocks_8, steps);
    printf("%-28s %10d %12.1f %12d\n", "__launch_bounds__(128, 8)", blocks_8, mlups_8, 32);

    // Over-subscription
    int blocks_max = (n_cells + 127) / 128;
    double mlups_max = run_kernel(k_lbm_occupancy_4, d_out, d_in, n_cells, blocks_max, steps);
    printf("%-28s %10d %12.1f %12s\n", "Full grid (oversubscribed)", blocks_max, mlups_max, "max");

    printf("\n--- Analysis ---\n");
    printf("If 4 warps/SM is enough to hide LDG latency (~92cy / 4.5cy FFMA = 20 warps needed),\n");
    printf("then __launch_bounds__(128,1) should be SLOWER than (128,4).\n");
    printf("If memory bandwidth is the bottleneck (128^3 = GDDR6X-bound),\n");
    printf("all configs should converge to the same MLUPS (BW-limited).\n");

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
