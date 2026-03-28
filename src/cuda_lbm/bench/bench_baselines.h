#ifndef BENCH_BASELINES_H
#define BENCH_BASELINES_H

// Regression baselines from README.md measured tables.
// RTX 4070 Ti Super (AD103, 66 SMs, 504 GB/s peak), 128^3 grid.
// All values are MLUPS at 128^3 (GDDR6X-bound regime).
// Tolerance: 10% for GDDR6X-bound (stable), 20% for L2-transitional.

#include "lbm_kernels.h"

typedef struct {
    LbmKernelVariant variant;
    float mlups_128;    // Baseline MLUPS at 128^3 (0 = no baseline)
    float tolerance;    // Regression tolerance (fraction, e.g., 0.10 = 10%)
} BaselineEntry;

static const BaselineEntry BASELINES[] = {
    // Physics-valid SoA kernels (128^3, GDDR6X-bound)
    {LBM_INT8_SOA,              5643.0f, 0.10f},
    {LBM_FP8_E4M3_SOA,         5408.0f, 0.10f},
    {LBM_FP8_E5M2_SOA,         5280.0f, 0.10f},
    {LBM_FP16_SOA_HALF2,       3802.0f, 0.10f},
    {LBM_INT16_SOA,            3569.0f, 0.10f},
    {LBM_FP16_SOA,             3463.0f, 0.10f},
    {LBM_BF16_SOA,             3204.0f, 0.10f},
    {LBM_FP32_SOA_CS,          2027.0f, 0.10f},
    {LBM_FP32_SOA_FUSED,       1984.0f, 0.10f},  // ~standard baseline
    {LBM_FP64_SOA,              406.0f, 0.15f},  // Compute-bound, wider tolerance

    // Physics-valid AoS kernels (128^3)
    {LBM_FP8_E4M3_AOS,         3202.0f, 0.10f},
    {LBM_FP8_E5M2_AOS,         3149.0f, 0.10f},
    {LBM_INT8_AOS,             2541.0f, 0.10f},
    {LBM_FP16_AOS,             1912.0f, 0.10f},
    {LBM_INT16_AOS,            1904.0f, 0.10f},
    {LBM_BF16_AOS,             1278.0f, 0.10f},
    {LBM_FP64_AOS,              461.0f, 0.15f},

    // Non-physics BW ceiling
    {LBM_INT4_SOA,             6169.0f, 0.10f},
    {LBM_FP4_SOA,              4727.0f, 0.10f},

    // DD at 64^3 (only benchmark size due to VRAM limits)
    {LBM_DD_SOA,                 58.0f, 0.20f},
};

#define BASELINE_COUNT (sizeof(BASELINES) / sizeof(BASELINES[0]))

// Lookup baseline for a given variant. Returns NULL if no baseline exists.
static inline const BaselineEntry* find_baseline(LbmKernelVariant v) {
    for (int i = 0; i < (int)BASELINE_COUNT; i++) {
        if (BASELINES[i].variant == v) return &BASELINES[i];
    }
    return 0;
}

#endif // BENCH_BASELINES_H
