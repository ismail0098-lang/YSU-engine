#ifndef LBM_MANAGED_MEMORY_H
#define LBM_MANAGED_MEMORY_H

// CUDA Unified Memory configuration for out-of-core LBM at 1024^3+.
//
// When the working set exceeds GPU VRAM, CUDA Unified Memory transparently
// pages inactive bricks between VRAM and system RAM via PCIe. Explicit
// prefetch hints via cuMemPrefetchAsync bring the active Z-slab to GPU
// before each LBM step.
//
// Architecture (from open_gororoba managed_memory.rs):
//   At 1024^3 with 30% sparsity + ephemeral macros + null force:
//     INT8 distributions: 6.1 GB (permanent, GPU-resident)
//     tau field: 1.3 GB (permanent, GPU-resident)
//     FP16 velocity: 2.6 GB (transient, allocated on-demand)
//
// Performance model (PCIe 4.0 x16, ~25 GB/s bidirectional):
//   Active slab migration per step: ~200 MB (one Z-layer of bricks)
//   Migration time: ~8 ms (vs ~35 ms step time at 512^3)
//   Overhead: ~23% of step time (acceptable for out-of-core)

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

typedef struct {
    int nx, ny, nz;             // Grid dimensions
    int brick_size;             // Brick edge length for prefetch granularity (default: 8)
    int prefetch_ahead;         // Z-slabs to prefetch ahead of current step (default: 2)
    double sparsity;            // Fraction of empty cells (0.0 = dense, 1.0 = empty)
    int bytes_per_dist;         // Storage bytes per distribution (1 for INT8, 2 for FP16, etc.)
} LbmManagedMemoryConfig;

static inline LbmManagedMemoryConfig lbm_managed_memory_default(void) {
    LbmManagedMemoryConfig cfg;
    cfg.nx = 1024;
    cfg.ny = 1024;
    cfg.nz = 1024;
    cfg.brick_size = 8;
    cfg.prefetch_ahead = 2;
    cfg.sparsity = 0.70;       // 30% occupied
    cfg.bytes_per_dist = 1;    // INT8 default
    return cfg;
}

// ============================================================================
// VRAM and PCIe overhead estimation
// ============================================================================

// Estimate GPU-resident VRAM in bytes for distribution + tau fields.
// Accounts for sparsity (only active cells consume VRAM).
static inline size_t lbm_managed_gpu_resident_bytes(const LbmManagedMemoryConfig* cfg) {
    size_t n_cells = (size_t)cfg->nx * cfg->ny * cfg->nz;
    double occupancy = 1.0 - cfg->sparsity;
    size_t active_cells = (size_t)((double)n_cells * occupancy);
    // Distributions (19 * bytes_per_dist, single buffer A-A) + tau (4 bytes)
    return active_cells * (19 * (size_t)cfg->bytes_per_dist + 4);
}

// Estimate Z-slab prefetch size in bytes.
// One Z-slab = nx * ny * brick_size cells (accounting for sparsity).
static inline size_t lbm_managed_slab_prefetch_bytes(const LbmManagedMemoryConfig* cfg) {
    size_t slab_cells = (size_t)cfg->nx * cfg->ny * cfg->brick_size;
    double occupancy = 1.0 - cfg->sparsity;
    size_t active_slab = (size_t)((double)slab_cells * occupancy);
    return active_slab * 19 * (size_t)cfg->bytes_per_dist;
}

// Estimate PCIe migration overhead per LBM step in milliseconds.
// Assumes PCIe 4.0 x16 at 25 GB/s effective bidirectional throughput.
static inline double lbm_managed_migration_ms(const LbmManagedMemoryConfig* cfg) {
    size_t slab_bytes = lbm_managed_slab_prefetch_bytes(cfg) * (size_t)cfg->prefetch_ahead;
    // bytes / (25 * 1024^3 bytes/sec) * 1000 ms/sec
    return (double)slab_bytes / (25.0 * 1024.0 * 1024.0 * 1024.0) * 1000.0;
}

// Check if the working set fits in VRAM without unified memory paging.
// If this returns true, managed memory is unnecessary overhead.
static inline int lbm_managed_fits_in_vram(
    const LbmManagedMemoryConfig* cfg,
    size_t vram_bytes
) {
    return lbm_managed_gpu_resident_bytes(cfg) < vram_bytes;
}

#ifdef __cplusplus
}
#endif

#endif // LBM_MANAGED_MEMORY_H
