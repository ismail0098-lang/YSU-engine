#ifndef LBM_METRICS_H
#define LBM_METRICS_H

#include "lbm_kernels.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GPU hardware specs
// ============================================================================

typedef struct {
    char name[256];
    int sm_major;
    int sm_minor;
    int n_sms;
    int cuda_cores;         // Estimated: n_sms * 128
    size_t vram_bytes;
    float peak_bw_gbs;      // Peak DRAM bandwidth (GB/s)
    float peak_fp32_tflops;  // Peak FP32 TFLOPS (dual-issue)
    float l2_bytes;          // L2 cache size in bytes
    float clock_ghz;         // Boost clock (GHz)
} GpuSpecs;

// Known SKU table for Ada Lovelace dies.
// Returns 1 if the GPU matched a known SKU; 0 if specs were computed from
// cudaDeviceProp (fallback for unknown GPUs).
#ifdef __CUDACC__
#include <cuda_runtime.h>

static inline int gpu_specs_from_device(GpuSpecs* out, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Copy name
    int i;
    for (i = 0; i < 255 && prop.name[i]; i++) out->name[i] = prop.name[i];
    out->name[i] = '\0';

    out->sm_major = prop.major;
    out->sm_minor = prop.minor;
    out->n_sms = prop.multiProcessorCount;
    out->cuda_cores = out->n_sms * 128; // Ada: 128 FP32 cores/SM
    out->vram_bytes = prop.totalGlobalMem;
    out->l2_bytes = (float)prop.l2CacheSize;
    out->clock_ghz = prop.clockRate / 1.0e6f;

    // Compute peak BW: memoryClockRate (kHz) * memoryBusWidth (bits) * 2 (DDR) / 8
    out->peak_bw_gbs = (prop.memoryClockRate / 1.0e6f) * (prop.memoryBusWidth / 8.0f) * 2.0f;
    // Peak FP32 TFLOPS: n_sms * 128 * 2 (FMA) * clock_ghz / 1000
    out->peak_fp32_tflops = (float)out->cuda_cores * 2.0f * out->clock_ghz / 1000.0f;

    // Known SKU overrides for accurate peak BW
    // (memoryClockRate may not reflect effective data rate for GDDR6X)
    if (out->n_sms == 128 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4090 (AD102)
        out->peak_bw_gbs = 1008.0f;
        out->l2_bytes = 72.0f * 1024 * 1024;
        return 1;
    }
    if (out->n_sms == 80 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4080 Super (AD103)
        out->peak_bw_gbs = 736.0f;
        out->l2_bytes = 64.0f * 1024 * 1024;
        return 1;
    }
    if (out->n_sms == 66 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4070 Ti Super (AD103)
        out->peak_bw_gbs = 504.0f;
        out->l2_bytes = 48.0f * 1024 * 1024;
        return 1;
    }
    if (out->n_sms == 60 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4070 Ti (AD104)
        out->peak_bw_gbs = 504.0f;
        out->l2_bytes = 48.0f * 1024 * 1024;
        return 1;
    }
    if (out->n_sms == 46 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4070 (AD104)
        out->peak_bw_gbs = 504.0f;
        out->l2_bytes = 36.0f * 1024 * 1024;
        return 1;
    }
    if (out->n_sms == 34 && out->sm_major == 8 && out->sm_minor == 9) {
        // RTX 4060 Ti (AD106)
        out->peak_bw_gbs = 288.0f;
        out->l2_bytes = 32.0f * 1024 * 1024;
        return 1;
    }

    return 0; // Unknown SKU -- specs computed from cudaDeviceProp
}
#endif // __CUDACC__

// ============================================================================
// Bandwidth regime classification
// ============================================================================

typedef enum {
    BW_REGIME_L2_RESIDENT,     // Working set fits in L2 (inflated MLUPS)
    BW_REGIME_L2_TRANSITIONAL, // Partial L2 fit
    BW_REGIME_GDDR6X_BOUND     // Working set >> L2 (true DRAM-limited)
} BwRegime;

static inline BwRegime classify_bw_regime(LbmKernelVariant v, int nx, int ny, int nz, float l2_bytes) {
    size_t working_set = lbm_dist_vram_bytes(v, nx, ny, nz);
    // Add auxiliary arrays
    size_t n_cells = (size_t)nx * ny * nz;
    working_set += n_cells * 8 * 4; // rho, u[3], tau, force[3]

    if ((float)working_set <= l2_bytes * 0.6f) return BW_REGIME_L2_RESIDENT;
    if ((float)working_set <= l2_bytes * 1.5f) return BW_REGIME_L2_TRANSITIONAL;
    return BW_REGIME_GDDR6X_BOUND;
}

static inline const char* bw_regime_str(BwRegime r) {
    switch (r) {
    case BW_REGIME_L2_RESIDENT:     return "L2-resident";
    case BW_REGIME_L2_TRANSITIONAL: return "L2-transitional";
    case BW_REGIME_GDDR6X_BOUND:    return "GDDR6X-bound";
    }
    return "unknown";
}

// ============================================================================
// Benchmark result
// ============================================================================

typedef struct {
    LbmKernelVariant variant;
    int nx, ny, nz;
    int warmup_steps;
    int timing_steps;
    float elapsed_ms;
    float mlups;            // Mega Lattice Updates Per Second
    float bw_gbs;           // Achieved bandwidth (GB/s)
    float bw_pct;           // BW utilization (% of peak)
    size_t vram_bytes;       // Total VRAM allocated
    BwRegime bw_regime;
    float mass_error;        // |sum(rho) - N| after timing steps (0 if not checked)
    float momentum_error;    // |sum(rho*u)| after timing steps (0 if not checked)
} BenchResult;

// ============================================================================
// Validation result
// ============================================================================

typedef struct {
    LbmKernelVariant variant;
    int nx, ny, nz;
    int steps;
    float initial_mass;
    float final_mass;
    float mass_drift;        // |final - initial| / initial
    float momentum_drift;    // |sum(rho*u)| after steps
    int density_stable;      // 1 if no NaN/Inf and all rho in [0.5, 2.0]
    int passed;              // 1 if all checks passed
    float tolerance;         // Precision-specific tolerance used
} ValidationResult;

#ifdef __cplusplus
}
#endif

#endif // LBM_METRICS_H
