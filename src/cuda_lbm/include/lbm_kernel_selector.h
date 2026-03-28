#ifndef LBM_KERNEL_SELECTOR_H
#define LBM_KERNEL_SELECTOR_H

// Optimal kernel auto-selector for Ada Lovelace D3Q19 LBM.
//
// Encodes three architectural truths proven by ncu/nsys profiling on
// RTX 4070 Ti Super (SM 8.9, 504 GB/s peak DRAM BW, 48 MB L2):
//
//   1. A-A streaming is mandatory at grid <= 256^3 (C-1390: 79.7% peak BW)
//   2. MRT collision is free for all tiers below FP16_H2 (C-1391: 722 FMA/cell
//      fills memory-stall pipeline bubbles -- latency hiding)
//   3. Low-precision storage yields up to 4x BW reduction (C-1385: INT8 SoA
//      Pareto-optimal for throughput vs VRAM)
//
// Ported from open_gororoba kernel_selector.rs (C-1392 formal decision table).

#include "lbm_kernels.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Precision requirement -- caller's intent, not a specific kernel
// ============================================================================

typedef enum {
    // Maximum throughput, acceptable quantization (~1.6% relative error).
    // Selects: INT8 SoA MRT A-A (~5643 MLUPS, 76 MB at 128^3)
    LBM_PREC_MAX_THROUGHPUT,

    // Maximum float throughput with ~5% relative error.
    // Selects: FP8 e4m3 SoA MRT A-A (~5408 MLUPS, 76 MB)
    LBM_PREC_MAX_FLOAT_THROUGHPUT,

    // Balanced accuracy for moderate-Re flows (10-bit mantissa).
    // Selects: FP16 SoA Half2 MRT (~3802 MLUPS, 152 MB)
    LBM_PREC_BALANCED,

    // Full FP32 precision for high-accuracy requirements.
    // Selects: FP32 SoA MRT A-A (~2062 MLUPS, 152 MB)
    LBM_PREC_HIGH_ACCURACY,

    // FP64 reference precision for validation.
    // Selects: FP64 SoA (~406 MLUPS, 608 MB)
    LBM_PREC_VALIDATION,
} LbmPrecisionRequirement;

// ============================================================================
// Selection result
// ============================================================================

typedef struct {
    LbmKernelVariant variant;       // Enum value for dispatch
    const LbmKernelInfo* info;      // Pointer into LBM_KERNEL_INFO table
    size_t vram_dist_bytes;         // Distribution buffer VRAM (bytes)
    size_t vram_total_bytes;        // Dist + macroscopic overhead estimate
    double estimated_mlups_128;     // Expected MLUPS at 128^3 (benchmark data)
} LbmKernelSelection;

// Select the optimal kernel for the given grid size and precision requirement.
//
// The selection encodes the C-1392 decision table from profiling on Ada
// Lovelace. MRT is always preferred (free via latency hiding on Ada). A-A
// streaming is preferred where available (halves VRAM, 79.7% peak BW).
static inline LbmKernelSelection lbm_select_optimal_kernel(
    int nx, int ny, int nz,
    LbmPrecisionRequirement precision
) {
    LbmKernelSelection sel;
    sel.vram_dist_bytes = 0;
    sel.vram_total_bytes = 0;
    sel.estimated_mlups_128 = 0.0;

    switch (precision) {
    case LBM_PREC_MAX_THROUGHPUT:
        // INT8 SoA: Pareto-optimal (highest MLUPS at lowest VRAM)
        // No MRT+A-A variant yet in kernel matrix; fall back to INT8 SoA
        sel.variant = LBM_INT8_SOA;
        sel.estimated_mlups_128 = 5643.0;
        break;

    case LBM_PREC_MAX_FLOAT_THROUGHPUT:
        // FP8 e4m3 SoA: highest float throughput
        sel.variant = LBM_FP8_E4M3_SOA;
        sel.estimated_mlups_128 = 5408.0;
        break;

    case LBM_PREC_BALANCED:
        // FP16 SoA Half2: 10-bit mantissa, +9.8% vs plain FP16 SoA
        sel.variant = LBM_FP16_SOA_HALF2;
        sel.estimated_mlups_128 = 3802.0;
        break;

    case LBM_PREC_HIGH_ACCURACY:
        // FP32 MRT A-A: full precision with VRAM halving
        sel.variant = LBM_FP32_SOA_MRT_AA;
        sel.estimated_mlups_128 = 2062.0;
        break;

    case LBM_PREC_VALIDATION:
        // FP64 SoA: reference precision (compute-bound on Ada)
        sel.variant = LBM_FP64_SOA;
        sel.estimated_mlups_128 = 406.0;
        break;

    default:
        // Safe fallback: FP32 MRT A-A
        sel.variant = LBM_FP32_SOA_MRT_AA;
        sel.estimated_mlups_128 = 2062.0;
        break;
    }

    sel.info = &LBM_KERNEL_INFO[sel.variant];
    sel.vram_dist_bytes = lbm_dist_vram_bytes(sel.variant, nx, ny, nz);
    // Macroscopic fields overhead: rho(f32) + u[3](f32) + tau(f32) + force[3](f32)
    // = 8 floats * n_cells = 32 * n_cells bytes
    size_t n_cells = (size_t)nx * ny * nz;
    size_t macro_bytes = 32 * n_cells;
    sel.vram_total_bytes = sel.vram_dist_bytes + macro_bytes;

    return sel;
}

// Check if the selected kernel fits in the available VRAM.
static inline int lbm_selection_fits_vram(
    const LbmKernelSelection* sel,
    size_t vram_bytes
) {
    return sel->vram_total_bytes < vram_bytes;
}

// Compute estimated bandwidth utilization percentage at 128^3.
// Uses the known RTX 4070 Ti Super peak of 504 GB/s.
static inline double lbm_estimated_bw_pct(const LbmKernelSelection* sel) {
    size_t bytes_per_step = lbm_bytes_per_cell_per_step(sel->variant);
    // BW = MLUPS * 1e6 * bytes_per_step / 1e9 (GB/s)
    double bw_gbs = sel->estimated_mlups_128 * 1.0e6 * (double)bytes_per_step / 1.0e9;
    return bw_gbs / 504.0 * 100.0;
}

// ============================================================================
// Grid-aware tier recommendation (from SM89 Architecture Reference)
// ============================================================================

// Returns the regime classification for the given grid and element size.
// This informs whether L2 caching or DRAM streaming dominates performance.
//
// Based on docs/sass/SM89_ARCHITECTURE_REFERENCE.md bandwidth regime table:
//   32^3:  L2-resident (effective BW >> DRAM peak)
//   64^3:  L2-transitional (mixed hit rate)
//   128^3: GDDR6X-bound (all tiers spill to DRAM)
//   256^3: GDDR6X-bound (pure DRAM streaming)
typedef enum {
    LBM_REGIME_L2_RESIDENT,         // Working set fits in L2 (48 MB Ada)
    LBM_REGIME_L2_TRANSITIONAL,     // Partially fits; mixed hit rate
    LBM_REGIME_GDDR6X_BOUND,        // Exceeds L2; DRAM streaming
} LbmBandwidthRegime;

static inline LbmBandwidthRegime lbm_classify_regime(
    int nx, int ny, int nz,
    int bytes_per_dist
) {
    size_t n_cells = (size_t)nx * ny * nz;
    // Working set: 19 dists * n_cells * bytes_per_dist (single buffer)
    size_t working_set = 19 * n_cells * (size_t)bytes_per_dist;
    size_t l2_bytes = 48 * 1024 * 1024; // 48 MB Ada L2

    if (working_set <= l2_bytes / 2) {
        return LBM_REGIME_L2_RESIDENT;
    } else if (working_set <= l2_bytes * 2) {
        return LBM_REGIME_L2_TRANSITIONAL;
    } else {
        return LBM_REGIME_GDDR6X_BOUND;
    }
}

#ifdef __cplusplus
}
#endif

#endif // LBM_KERNEL_SELECTOR_H
