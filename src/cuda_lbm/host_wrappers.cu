// Host-side dispatch wrappers for all D3Q19 LBM kernel variants.
// Provides unified launch interface regardless of precision tier or layout.

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// Grid/buffer structs
// ============================================================================

typedef struct {
    int nx, ny, nz;
    int n_cells;
} LBMGrid;

typedef struct {
    void* f_a;       // Ping buffer (or single buffer for A-A)
    void* f_b;       // Pong buffer (NULL for A-A)
    // DD kernels use 4 buffers: f_a = f_hi_a, f_b = f_lo_a, f_c = f_hi_b, f_d = f_lo_b
    void* f_c;       // DD pong hi (NULL for non-DD)
    void* f_d;       // DD pong lo (NULL for non-DD)
    float* rho;
    float* u;        // [3 * n_cells] SoA
    float* tau;
    float* force;    // [3 * n_cells] SoA
} LBMBuffers;

static inline LBMGrid lbm_grid_make(int nx, int ny, int nz) {
    LBMGrid g;
    g.nx = nx;
    g.ny = ny;
    g.nz = nz;
    g.n_cells = nx * ny * nz;
    return g;
}

// ============================================================================
// Kernel forward declarations
// ============================================================================

// Standard ping-pong signature:
//   (const T* f_in, T* f_out, float* rho, float* u, const float* tau,
//    const float* force, int nx, int ny, int nz)
//
// A-A signature:
//   (T* f, float* rho, float* u, const float* tau, const float* force,
//    int nx, int ny, int nz, int parity)
//
// DD signature:
//   (const double* f_hi_in, const double* f_lo_in, double* f_hi_out, double* f_lo_out,
//    float* rho, float* u, const float* force, const float* tau, int nx, int ny, int nz)
//
// Tiled kernels use 3D grid. All others use 1D grid.
// All kernel functions are declared extern "C" in their .cu files.

// ============================================================================
// Launch helpers
// ============================================================================

// Standard 1D grid for non-tiled, non-coarsened kernels.
static inline dim3 lbm_grid_1d(int n_cells, int threads_per_block) {
    return dim3((n_cells + threads_per_block - 1) / threads_per_block);
}

// Coarsened grid: ceil(n_cells / cells_per_thread / threads_per_block).
static inline dim3 lbm_grid_coarsened(int n_cells, int cells_per_thread, int threads_per_block) {
    int effective = (n_cells + cells_per_thread - 1) / cells_per_thread;
    return dim3((effective + threads_per_block - 1) / threads_per_block);
}

// Tiled 3D grid for tiled kernels (8x8x4 tiles).
static inline dim3 lbm_grid_tiled(int nx, int ny, int nz) {
    return dim3((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
}

// ============================================================================
// launch_lbm_step -- dispatch a single LBM step
// ============================================================================

// For ping-pong: f_in = buffers.f_a (or f_b), f_out = buffers.f_b (or f_a).
// For A-A: f = buffers.f_a, parity determines direction swap.
// Caller is responsible for alternating buffers between steps.

typedef void (*lbm_step_fn)(const void*, void*, float*, float*, const float*, const float*, int, int, int);
typedef void (*lbm_aa_fn)(void*, float*, float*, const float*, const float*, int, int, int, int);

int launch_lbm_step(
    LbmKernelVariant variant,
    const LBMGrid* grid,
    LBMBuffers* bufs,
    const void* f_in,
    void* f_out,
    int parity,  // Only used for A-A kernels
    cudaStream_t stream
) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    int n = grid->n_cells;
    int tpb = info->threads_per_block;
    dim3 block(tpb);
    dim3 grd;

    // Compute grid dimensions
    int is_tiled = (variant == LBM_FP32_SOA_TILED || variant == LBM_FP32_SOA_MRT_TILED);
    if (is_tiled) {
        grd = lbm_grid_tiled(grid->nx, grid->ny, grid->nz);
    } else if (info->cells_per_thread > 1) {
        grd = lbm_grid_coarsened(n, info->cells_per_thread, tpb);
    } else {
        grd = lbm_grid_1d(n, tpb);
    }

    // DD has a unique 4-buffer signature
    if (variant == LBM_DD_SOA) {
        // DD: (f_hi_in, f_lo_in, f_hi_out, f_lo_out, rho, u, force, tau, nx, ny, nz)
        void* args[] = {
            (void*)&bufs->f_a, (void*)&bufs->f_b,
            (void*)&bufs->f_c, (void*)&bufs->f_d,
            (void*)&bufs->rho, (void*)&bufs->u,
            (void*)&bufs->force, (void*)&bufs->tau,
            (void*)&grid->nx, (void*)&grid->ny, (void*)&grid->nz
        };
        // Note: cudaLaunchKernel requires the function pointer. Since all
        // kernels are extern "C", we must use the symbol directly. For DD:
        extern void lbm_step_fused_dd_kernel(
            const double*, const double*, double*, double*,
            float*, float*, const float*, const float*, int, int, int);
        return cudaLaunchKernel((const void*)lbm_step_fused_dd_kernel,
                                grd, block, args, 0, stream);
    }

    // A-A kernels: single-buffer with parity
    if (info->is_aa) {
        void* args[] = {
            (void*)&bufs->f_a,
            (void*)&bufs->rho, (void*)&bufs->u,
            (void*)&bufs->tau, (void*)&bufs->force,
            (void*)&grid->nx, (void*)&grid->ny, (void*)&grid->nz,
            (void*)&parity
        };

        if (variant == LBM_FP32_SOA_AA) {
            extern void lbm_step_soa_aa(float*, float*, float*, const float*, const float*, int, int, int, int);
            return cudaLaunchKernel((const void*)lbm_step_soa_aa, grd, block, args, 0, stream);
        }
        if (variant == LBM_FP32_SOA_MRT_AA) {
            extern void lbm_step_soa_mrt_aa(float*, float*, float*, const float*, const float*, int, int, int, int);
            return cudaLaunchKernel((const void*)lbm_step_soa_mrt_aa, grd, block, args, 0, stream);
        }
        return cudaErrorInvalidValue;
    }

    // Standard ping-pong: (f_in, f_out, rho, u, tau, force, nx, ny, nz)
    void* args[] = {
        (void*)&f_in, (void*)&f_out,
        (void*)&bufs->rho, (void*)&bufs->u,
        (void*)&bufs->tau, (void*)&bufs->force,
        (void*)&grid->nx, (void*)&grid->ny, (void*)&grid->nz
    };

    // Macro to reduce boilerplate for standard-signature kernels.
    // Each kernel is declared extern and launched via cudaLaunchKernel.
    #define DISPATCH_STEP(VARIANT, FUNC_NAME, ELEM_T)                         \
        case VARIANT: {                                                       \
            extern void FUNC_NAME(const ELEM_T*, ELEM_T*, float*, float*,     \
                                  const float*, const float*, int, int, int); \
            return cudaLaunchKernel((const void*)FUNC_NAME,                   \
                                   grd, block, args, 0, stream);             \
        }

    switch (variant) {
    // FP32 SoA
    DISPATCH_STEP(LBM_FP32_SOA_FUSED,         lbm_step_soa_fused,              float)
    DISPATCH_STEP(LBM_FP32_SOA_MRT_FUSED,     lbm_step_soa_mrt_fused,          float)
    DISPATCH_STEP(LBM_FP32_SOA_PULL,          lbm_step_soa_pull,               float)
    DISPATCH_STEP(LBM_FP32_SOA_MRT_PULL,      lbm_step_soa_mrt_pull,           float)
    DISPATCH_STEP(LBM_FP32_SOA_TILED,         lbm_step_soa_tiled,              float)
    DISPATCH_STEP(LBM_FP32_SOA_MRT_TILED,     lbm_step_soa_mrt_tiled,          float)
    DISPATCH_STEP(LBM_FP32_SOA_COARSENED,     lbm_step_soa_coarsened,          float)
    DISPATCH_STEP(LBM_FP32_SOA_MRT_COARSENED, lbm_step_soa_mrt_coarsened,      float)
    DISPATCH_STEP(LBM_FP32_SOA_COARSENED_F4,  lbm_step_soa_coarsened_float4,   float)
    DISPATCH_STEP(LBM_FP32_SOA_CS,            lbm_step_fp32_soa_cs_kernel,     float)

    // FP16
    DISPATCH_STEP(LBM_FP16_AOS,              lbm_step_fused_fp16_kernel,       void)
    DISPATCH_STEP(LBM_FP16_SOA,              lbm_step_fp16_soa_kernel,         void)
    DISPATCH_STEP(LBM_FP16_SOA_HALF2,        lbm_step_fp16_soa_half2_kernel,   void)

    // BF16
    DISPATCH_STEP(LBM_BF16_AOS,              lbm_step_fused_bf16_kernel,       void)
    DISPATCH_STEP(LBM_BF16_SOA,              lbm_step_bf16_soa_kernel,         void)

    // FP8
    DISPATCH_STEP(LBM_FP8_E4M3_AOS,          lbm_step_fused_fp8_kernel,        void)
    DISPATCH_STEP(LBM_FP8_E4M3_SOA,          lbm_step_fp8_soa_kernel,          void)
    DISPATCH_STEP(LBM_FP8_E5M2_AOS,          lbm_step_fused_fp8e5m2_kernel,    void)
    DISPATCH_STEP(LBM_FP8_E5M2_SOA,          lbm_step_fp8_e5m2_soa_kernel,     void)

    // INT8
    DISPATCH_STEP(LBM_INT8_AOS,              lbm_step_fused_int8_kernel,       void)
    DISPATCH_STEP(LBM_INT8_SOA,              lbm_step_int8_soa_kernel,         void)

    // INT16
    DISPATCH_STEP(LBM_INT16_AOS,             lbm_step_int16_kernel,            void)
    DISPATCH_STEP(LBM_INT16_SOA,             lbm_step_int16_soa_kernel,        void)

    // FP64
    DISPATCH_STEP(LBM_FP64_AOS,              lbm_step_fused_fp64_kernel,       void)
    DISPATCH_STEP(LBM_FP64_SOA,              lbm_step_fp64_soa_kernel,         void)

    // BW ceiling
    DISPATCH_STEP(LBM_INT4_SOA,              lbm_step_fused_int4_kernel,       void)
    DISPATCH_STEP(LBM_FP4_SOA,               lbm_step_fp4_kernel,              void)

    default:
        return cudaErrorInvalidValue;
    }
    #undef DISPATCH_STEP
}

// ============================================================================
// launch_lbm_init -- initialize distributions to equilibrium
// ============================================================================

int launch_lbm_init(
    LbmKernelVariant variant,
    const LBMGrid* grid,
    LBMBuffers* bufs,
    float rho_init,
    float ux_init, float uy_init, float uz_init,
    cudaStream_t stream
) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    int n = grid->n_cells;
    int tpb = 128;  // Init kernels always use 1 thread per cell, 128 tpb
    dim3 block(tpb);
    dim3 grd((n + tpb - 1) / tpb);

    // DD init has unique signature
    if (variant == LBM_DD_SOA) {
        extern void initialize_uniform_dd_kernel(
            double*, double*, float*, float*, float, float, float, float, int, int, int);
        void* args[] = {
            (void*)&bufs->f_a, (void*)&bufs->f_b,
            (void*)&bufs->rho, (void*)&bufs->u,
            &rho_init, &ux_init, &uy_init, &uz_init,
            (void*)&grid->nx, (void*)&grid->ny, (void*)&grid->nz
        };
        return cudaLaunchKernel((const void*)initialize_uniform_dd_kernel,
                                grd, block, args, 0, stream);
    }

    // Standard init: (f, rho, u, rho_init, ux, uy, uz, nx, ny, nz)
    void* args[] = {
        (void*)&bufs->f_a,
        (void*)&bufs->rho, (void*)&bufs->u,
        &rho_init, &ux_init, &uy_init, &uz_init,
        (void*)&grid->nx, (void*)&grid->ny, (void*)&grid->nz
    };

    #define DISPATCH_INIT(VARIANT, FUNC_NAME, ELEM_T)                        \
        case VARIANT: {                                                      \
            extern void FUNC_NAME(ELEM_T*, float*, float*,                   \
                                  float, float, float, float,                \
                                  int, int, int);                            \
            return cudaLaunchKernel((const void*)FUNC_NAME,                  \
                                   grd, block, args, 0, stream);            \
        }

    switch (variant) {
    DISPATCH_INIT(LBM_FP32_SOA_FUSED,         initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_MRT_FUSED,     initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_PULL,          initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_MRT_PULL,      initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_TILED,         initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_MRT_TILED,     initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_COARSENED,     initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_MRT_COARSENED, initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_COARSENED_F4,  initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_AA,            initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_MRT_AA,        initialize_uniform_soa_kernel,         float)
    DISPATCH_INIT(LBM_FP32_SOA_CS,            initialize_uniform_fp32_soa_cs_kernel, float)

    DISPATCH_INIT(LBM_FP16_AOS,              initialize_uniform_fp16_kernel,         void)
    DISPATCH_INIT(LBM_FP16_SOA,              initialize_uniform_fp16_soa_kernel,     void)
    DISPATCH_INIT(LBM_FP16_SOA_HALF2,        initialize_uniform_fp16_soa_half2_kernel, void)

    DISPATCH_INIT(LBM_BF16_AOS,              initialize_uniform_bf16_kernel,         void)
    DISPATCH_INIT(LBM_BF16_SOA,              initialize_uniform_bf16_soa_kernel,     void)

    DISPATCH_INIT(LBM_FP8_E4M3_AOS,          initialize_uniform_fp8_kernel,          void)
    DISPATCH_INIT(LBM_FP8_E4M3_SOA,          initialize_uniform_fp8_soa_kernel,      void)
    DISPATCH_INIT(LBM_FP8_E5M2_AOS,          initialize_uniform_fp8e5m2_kernel,      void)
    DISPATCH_INIT(LBM_FP8_E5M2_SOA,          initialize_uniform_fp8_e5m2_soa_kernel, void)

    DISPATCH_INIT(LBM_INT8_AOS,              initialize_uniform_int8_kernel,         void)
    DISPATCH_INIT(LBM_INT8_SOA,              initialize_uniform_int8_soa_kernel,     void)

    DISPATCH_INIT(LBM_INT16_AOS,             initialize_uniform_int16_kernel,        void)
    DISPATCH_INIT(LBM_INT16_SOA,             initialize_uniform_int16_soa_kernel,    void)

    DISPATCH_INIT(LBM_FP64_AOS,              initialize_uniform_fp64_kernel,         void)
    DISPATCH_INIT(LBM_FP64_SOA,              initialize_uniform_fp64_soa_kernel,     void)

    DISPATCH_INIT(LBM_INT4_SOA,              initialize_uniform_int4_kernel,         void)
    DISPATCH_INIT(LBM_FP4_SOA,               initialize_uniform_fp4_kernel,          void)

    default:
        return cudaErrorInvalidValue;
    }
    #undef DISPATCH_INIT
}

// ============================================================================
// Occupancy queries
// ============================================================================

typedef struct {
    LbmKernelVariant variant;
    int max_active_blocks;
    int max_warps;
    float occupancy_pct;
} OccupancyInfo;

// Query occupancy for a single variant on the current device.
int query_occupancy(LbmKernelVariant variant, int device_id, OccupancyInfo* out) {
    (void)device_id;
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    out->variant = variant;
    out->max_active_blocks = 0;
    out->max_warps = 0;
    out->occupancy_pct = 0.0f;

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor requires the function
    // pointer. For a generic approach, we use the same switch dispatch.
    // This is verbose but necessary since we cannot store device function
    // pointers in a table at compile time.

    int block_size = info->threads_per_block;
    int num_blocks = 0;

    // Use a simplified approach: query via cudaOccupancyMaxActiveBlocksPerMultiprocessor
    // for the most common kernels.
    // For now, return a reasonable estimate based on register count heuristics.
    // The exact values require linking against the specific kernel symbols.

    // Estimate: 48 max warps per SM on Ada
    int warps_per_block = (block_size + 31) / 32;

    // Conservative estimate assuming 64 regs/thread for standard kernels,
    // 128 for MRT, 192 for DD
    int est_regs;
    if (variant == LBM_DD_SOA) est_regs = 192;
    else if (info->is_mrt) est_regs = 128;
    else est_regs = 64;

    int max_warps_by_regs = 65536 / (est_regs * 32);
    int max_blocks_by_regs = max_warps_by_regs / warps_per_block;
    int max_blocks_by_slots = 24; // Ada: 24 blocks/SM limit
    num_blocks = max_blocks_by_regs < max_blocks_by_slots ? max_blocks_by_regs : max_blocks_by_slots;
    if (num_blocks < 1) num_blocks = 1;

    out->max_active_blocks = num_blocks;
    out->max_warps = num_blocks * warps_per_block;
    if (out->max_warps > 48) out->max_warps = 48;
    out->occupancy_pct = (float)out->max_warps / 48.0f * 100.0f;

    return 0;
}

// Print occupancy table for all kernel variants.
void report_all_occupancy(int device_id) {
    printf("%-30s  %6s  %6s  %8s\n", "Kernel", "Blocks", "Warps", "Occ%%");
    printf("%-30s  %6s  %6s  %8s\n", "------", "------", "-----", "----");

    for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
        OccupancyInfo occ;
        query_occupancy((LbmKernelVariant)v, device_id, &occ);
        printf("%-30s  %6d  %6d  %7.1f%%\n",
               LBM_KERNEL_INFO[v].name,
               occ.max_active_blocks,
               occ.max_warps,
               occ.occupancy_pct);
    }
}
