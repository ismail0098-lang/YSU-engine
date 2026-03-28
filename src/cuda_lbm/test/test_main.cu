// Minimal test runner for LBM kernel test suite.
// No external test framework dependency.

#include <cuda_runtime.h>
#include <stdio.h>

// Forward declarations from test files
extern int test_conservation_all(void);
extern int test_regression_all(void);
extern int test_mrt_stability(void);

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // Initialize CUDA
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== LBM Test Suite ===\n");
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int total_failures = 0;

    // 1. Conservation tests
    int cons_fail = test_conservation_all();
    total_failures += cons_fail;
    printf("\n");

    // 2. MRT stability test
    int mrt_fail = test_mrt_stability();
    total_failures += mrt_fail;
    printf("\n");

    // 3. Performance regression tests (optional -- slow)
    // Only run if explicitly requested or if --regression flag is passed
    int do_regression = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--regression") == 0) do_regression = 1;
    }

    if (do_regression) {
        int reg_fail = test_regression_all();
        total_failures += reg_fail;
        printf("\n");
    } else {
        printf("=== Performance Regression Tests (skipped -- pass --regression to enable) ===\n\n");
    }

    // Summary
    printf("================================\n");
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", total_failures);
    }
    printf("================================\n");

    return total_failures > 0 ? 1 : 0;
}
