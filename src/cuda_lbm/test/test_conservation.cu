// Per-kernel physics conservation tests.
// Same logic as bench_validate.cu but with hard assertions.

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include <stdio.h>

// Forward declaration from bench_validate.cu
extern ValidationResult validate_kernel(LbmKernelVariant variant);
extern void validate_print_result(const ValidationResult* vr);

// Run conservation tests for all physics-valid kernels.
// Returns number of failures.
int test_conservation_all(void) {
    printf("=== Conservation Tests (32^3, 100 steps, quiescent) ===\n");
    int failures = 0;
    int tested = 0;

    for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
        const LbmKernelInfo* info = &LBM_KERNEL_INFO[v];
        if (!info->physics_valid) continue;

        ValidationResult vr = validate_kernel((LbmKernelVariant)v);
        validate_print_result(&vr);
        tested++;

        if (!vr.passed) {
            fprintf(stderr, "  ASSERTION FAILED: %s conservation check\n", info->name);
            failures++;
        }
    }

    printf("Conservation: %d tested, %d passed, %d failed\n",
           tested, tested - failures, failures);
    return failures;
}
