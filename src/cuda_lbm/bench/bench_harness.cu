// Core timing and metrics computation for LBM benchmarks.
// Uses CUDA events for sub-ms timing precision.
// Annotates with NVTX ranges for nsys integration.

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include "bench_baselines.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LBM_HAVE_NVTX
#include <nvToolsExt.h>
#define NVTX_PUSH(msg) nvtxRangePushA(msg)
#define NVTX_POP()     nvtxRangePop()
#else
#define NVTX_PUSH(msg) ((void)0)
#define NVTX_POP()     ((void)0)
#endif

// Forward declarations from host_wrappers.cu
extern int launch_lbm_step(LbmKernelVariant variant, const void* grid,
                           void* bufs, const void* f_in, void* f_out,
                           int parity, cudaStream_t stream);
extern int launch_lbm_init(LbmKernelVariant variant, const void* grid,
                           void* bufs, float rho, float ux, float uy, float uz,
                           cudaStream_t stream);

// Allocate LBM buffers for a given variant and grid size.
// Returns 0 on success, nonzero if VRAM insufficient.
int bench_alloc_buffers(
    LbmKernelVariant variant,
    int nx, int ny, int nz,
    void** f_a, void** f_b,
    void** f_c, void** f_d,
    float** rho, float** u,
    float** tau, float** force,
    size_t* total_vram
) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    size_t n_cells = (size_t)nx * ny * nz;

    // Check available VRAM
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t dist_bytes = lbm_dist_vram_bytes(variant, nx, ny, nz);
    size_t aux_bytes = n_cells * (1 + 3 + 1 + 3) * sizeof(float); // rho, u[3], tau, force[3]
    size_t needed = dist_bytes + aux_bytes;

    if (needed > free_mem * 0.9) {
        fprintf(stderr, "  [SKIP] %s at %dx%dx%d: needs %.0f MB, only %.0f MB free\n",
                info->name, nx, ny, nz,
                (double)needed / (1024.0 * 1024.0),
                (double)free_mem / (1024.0 * 1024.0));
        return -1;
    }

    *total_vram = needed;
    *f_c = NULL;
    *f_d = NULL;

    // Distribution buffers
    if (variant == LBM_DD_SOA) {
        size_t dd_buf = 19 * n_cells * sizeof(double);
        cudaMalloc(f_a, dd_buf);  // f_hi_a (ping hi)
        cudaMalloc(f_b, dd_buf);  // f_lo_a (ping lo)
        cudaMalloc(f_c, dd_buf);  // f_hi_b (pong hi)
        cudaMalloc(f_d, dd_buf);  // f_lo_b (pong lo)
    } else if (info->is_aa) {
        // Single buffer for A-A
        size_t buf_size = dist_bytes; // Already accounts for single buffer
        cudaMalloc(f_a, buf_size);
        *f_b = NULL;
    } else {
        size_t per_buf = dist_bytes / 2;
        cudaMalloc(f_a, per_buf);
        cudaMalloc(f_b, per_buf);
    }

    // Auxiliary arrays
    cudaMalloc(rho,   n_cells * sizeof(float));
    cudaMalloc(u,     n_cells * 3 * sizeof(float));
    cudaMalloc(tau,   n_cells * sizeof(float));
    cudaMalloc(force, n_cells * 3 * sizeof(float));

    // Initialize tau to 0.6 and force to 0
    {
        float* h_tau = (float*)malloc(n_cells * sizeof(float));
        for (size_t i = 0; i < n_cells; i++) h_tau[i] = 0.6f;
        cudaMemcpy(*tau, h_tau, n_cells * sizeof(float), cudaMemcpyHostToDevice);
        free(h_tau);
    }
    cudaMemset(*force, 0, n_cells * 3 * sizeof(float));

    return 0;
}

// Free all LBM buffers.
void bench_free_buffers(
    void* f_a, void* f_b, void* f_c, void* f_d,
    float* rho, float* u, float* tau, float* force
) {
    if (f_a)   cudaFree(f_a);
    if (f_b)   cudaFree(f_b);
    if (f_c)   cudaFree(f_c);
    if (f_d)   cudaFree(f_d);
    if (rho)   cudaFree(rho);
    if (u)     cudaFree(u);
    if (tau)   cudaFree(tau);
    if (force) cudaFree(force);
}

// Run a single benchmark: warmup + timed steps.
// Returns BenchResult.
BenchResult bench_run_variant(
    LbmKernelVariant variant,
    int nx, int ny, int nz,
    int warmup_steps,
    int timing_steps,
    const GpuSpecs* gpu
) {
    BenchResult result;
    memset(&result, 0, sizeof(result));
    result.variant = variant;
    result.nx = nx; result.ny = ny; result.nz = nz;
    result.warmup_steps = warmup_steps;
    result.timing_steps = timing_steps;

    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    size_t n_cells = (size_t)nx * ny * nz;

    // Allocate
    void *f_a = NULL, *f_b = NULL, *f_c = NULL, *f_d = NULL;
    float *rho = NULL, *u = NULL, *tau_d = NULL, *force_d = NULL;
    size_t vram = 0;

    if (bench_alloc_buffers(variant, nx, ny, nz,
                            &f_a, &f_b, &f_c, &f_d,
                            &rho, &u, &tau_d, &force_d, &vram) != 0) {
        result.mlups = -1.0f;
        return result;
    }
    result.vram_bytes = vram;

    // Build grid and buffer structs for host_wrappers
    typedef struct { int nx, ny, nz, n_cells; } LBMGrid;
    typedef struct { void *f_a, *f_b, *f_c, *f_d; float *rho, *u, *tau, *force; } LBMBuffers;

    LBMGrid grid = {nx, ny, nz, (int)n_cells};
    LBMBuffers bufs = {f_a, f_b, f_c, f_d, rho, u, tau_d, force_d};

    // Initialize
    launch_lbm_init(variant, (const void*)&grid, (void*)&bufs,
                    1.0f, 0.0f, 0.0f, 0.0f, 0);
    cudaDeviceSynchronize();

    // Warmup
    NVTX_PUSH("warmup");
    for (int s = 0; s < warmup_steps; s++) {
        if (info->is_aa) {
            launch_lbm_step(variant, (const void*)&grid, (void*)&bufs,
                           NULL, NULL, s % 2, 0);
        } else {
            void* in  = (s % 2 == 0) ? f_a : f_b;
            void* out = (s % 2 == 0) ? f_b : f_a;
            launch_lbm_step(variant, (const void*)&grid, (void*)&bufs,
                           in, out, 0, 0);
        }
    }
    NVTX_POP();

    // Hard sync after warmup -- flush driver queue
    cudaDeviceSynchronize();

    // Timed region
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    NVTX_PUSH(info->name);
    cudaEventRecord(start, 0);

    for (int s = 0; s < timing_steps; s++) {
        int total_s = warmup_steps + s;
        if (info->is_aa) {
            launch_lbm_step(variant, (const void*)&grid, (void*)&bufs,
                           NULL, NULL, total_s % 2, 0);
        } else {
            void* in  = (total_s % 2 == 0) ? f_a : f_b;
            void* out = (total_s % 2 == 0) ? f_b : f_a;
            launch_lbm_step(variant, (const void*)&grid, (void*)&bufs,
                           in, out, 0, 0);
        }
    }

    cudaEventRecord(stop, 0);
    NVTX_POP();
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute metrics
    result.elapsed_ms = elapsed_ms;
    double elapsed_s = elapsed_ms / 1000.0;
    double total_cells = (double)n_cells * timing_steps;
    result.mlups = (float)(total_cells / elapsed_s / 1.0e6);

    size_t bytes_per_cell = lbm_bytes_per_cell_per_step(variant);
    double total_bytes = (double)bytes_per_cell * total_cells;
    result.bw_gbs = (float)(total_bytes / elapsed_s / 1.0e9);
    result.bw_pct = result.bw_gbs / gpu->peak_bw_gbs * 100.0f;

    // Bandwidth regime
    result.bw_regime = classify_bw_regime(variant, nx, ny, nz, gpu->l2_bytes);

    // Cleanup
    bench_free_buffers(f_a, f_b, f_c, f_d, rho, u, tau_d, force_d);

    return result;
}

// Print a single result row in table format.
void bench_print_result_table(const BenchResult* r) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[r->variant];
    printf("%-28s  %4dx%4dx%4d  %8.0f  %7.1f  %5.1f%%  %6zu  %s\n",
           info->name,
           r->nx, r->ny, r->nz,
           r->mlups,
           r->bw_gbs,
           r->bw_pct,
           r->vram_bytes / (1024 * 1024),
           bw_regime_str(r->bw_regime));
}

// Print table header.
void bench_print_header(void) {
    printf("%-28s  %14s  %8s  %7s  %6s  %6s  %s\n",
           "Kernel", "Grid", "MLUPS", "BW(GB/s)", "BW%", "VRAM(MB)", "Regime");
    printf("%-28s  %14s  %8s  %7s  %6s  %6s  %s\n",
           "------", "----", "-----", "-------", "---", "-------", "------");
}

// Print result as CSV row.
void bench_print_result_csv(const BenchResult* r) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[r->variant];
    printf("%s,%d,%d,%d,%.0f,%.1f,%.1f,%zu,%s\n",
           info->name, r->nx, r->ny, r->nz,
           r->mlups, r->bw_gbs, r->bw_pct,
           r->vram_bytes / (1024 * 1024),
           bw_regime_str(r->bw_regime));
}
