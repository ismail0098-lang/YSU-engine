// Performance regression tests.
// Ensures MLUPS is within tolerance of baselines for each kernel at 128^3.

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include "bench_baselines.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Forward declaration from bench_harness.cu
extern BenchResult bench_run_variant(LbmKernelVariant variant,
    int nx, int ny, int nz, int warmup, int timing, const GpuSpecs* gpu);

// Run performance regression tests.
// Returns number of regressions.
int test_regression_all(void) {
    printf("=== Performance Regression Tests ===\n");

    int device = 0;
    GpuSpecs gpu;
    gpu_specs_from_device(&gpu, device);

    int regressions = 0;
    int tested = 0;

    for (int i = 0; i < (int)BASELINE_COUNT; i++) {
        const BaselineEntry* b = &BASELINES[i];
        if (b->mlups_128 <= 0) continue;

        // DD is tested at 64^3 due to VRAM limits
        int gs = (b->variant == LBM_DD_SOA) ? 64 : 128;

        BenchResult r = bench_run_variant(b->variant, gs, gs, gs, 5, 30, &gpu);
        if (r.mlups < 0) continue; // Skipped (VRAM, SM)

        tested++;

        float expected = b->mlups_128;
        float threshold = expected * (1.0f - b->tolerance);
        int regressed = (r.mlups < threshold);

        printf("  %-28s  measured=%7.0f  expected=%7.0f  tol=%.0f%%  [%s]\n",
               LBM_KERNEL_INFO[b->variant].name,
               r.mlups, expected, b->tolerance * 100,
               regressed ? "REGRESSION" : "OK");

        if (regressed) regressions++;
    }

    printf("Regression: %d tested, %d passed, %d regressions\n",
           tested, tested - regressions, regressions);
    return regressions;
}
