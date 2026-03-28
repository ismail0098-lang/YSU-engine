// CLI entry point for LBM benchmark, validation, and regression testing.
//
// Usage:
//   lbm_bench --all --grid 128 --output table
//   lbm_bench --kernel int8_soa --grid 64,128,256
//   lbm_bench --validate
//   lbm_bench --regression
//   lbm_bench --occupancy

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include "bench_baselines.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations from bench_harness.cu
extern BenchResult bench_run_variant(LbmKernelVariant variant,
    int nx, int ny, int nz, int warmup, int timing, const GpuSpecs* gpu);
extern void bench_print_header(void);
extern void bench_print_result_table(const BenchResult* r);
extern void bench_print_result_csv(const BenchResult* r);

// Forward declarations from bench_validate.cu
extern int validate_all(void);

// Forward declarations from host_wrappers.cu
extern void report_all_occupancy(int device_id);

// Parse comma-separated grid sizes (e.g., "64,128,256").
static int parse_grid_sizes(const char* str, int* sizes, int max_sizes) {
    int count = 0;
    const char* p = str;
    while (*p && count < max_sizes) {
        sizes[count++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return count;
}

// Find kernel variant by name substring (case-insensitive).
static int find_variant_by_name(const char* name) {
    for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
        const char* kname = LBM_KERNEL_INFO[v].name;
        // Simple substring match (lowercase comparison)
        const char* p = kname;
        const char* q = name;
        int match = 0;
        while (*p) {
            const char* pp = p;
            const char* qq = q;
            while (*pp && *qq) {
                char a = (*pp >= 'A' && *pp <= 'Z') ? *pp + 32 : *pp;
                char b = (*qq >= 'A' && *qq <= 'Z') ? *qq + 32 : *qq;
                if (a != b) break;
                pp++; qq++;
            }
            if (*qq == '\0') { match = 1; break; }
            p++;
        }
        if (match) return v;
    }
    return -1;
}

static void print_usage(void) {
    printf("Usage: lbm_bench [options]\n\n");
    printf("Options:\n");
    printf("  --all                Run all kernel variants\n");
    printf("  --kernel <name>      Run a specific kernel by name\n");
    printf("  --grid <sizes>       Comma-separated grid sizes (default: 128)\n");
    printf("  --warmup <N>         Warmup steps (default: 5)\n");
    printf("  --steps <N>          Timing steps (default: 30)\n");
    printf("  --output csv|table   Output format (default: table)\n");
    printf("  --validate           Run physics conservation checks\n");
    printf("  --regression         Compare against baselines, exit nonzero on regression\n");
    printf("  --occupancy          Print occupancy table and exit\n");
    printf("  --list               List all kernel variants\n");
    printf("  --help               Show this help\n");
}

int main(int argc, char** argv) {
    // Defaults
    int do_all = 0;
    int do_validate = 0;
    int do_regression = 0;
    int do_occupancy = 0;
    int do_list = 0;
    int warmup = 5;
    int timing_steps = 30;
    int grid_sizes[16] = {128};
    int n_grids = 1;
    int specific_variant = -1;
    int csv_output = 0;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            do_all = 1;
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            specific_variant = find_variant_by_name(argv[++i]);
            if (specific_variant < 0) {
                fprintf(stderr, "Unknown kernel: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            n_grids = parse_grid_sizes(argv[++i], grid_sizes, 16);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            timing_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            i++;
            csv_output = (strcmp(argv[i], "csv") == 0);
        } else if (strcmp(argv[i], "--validate") == 0) {
            do_validate = 1;
        } else if (strcmp(argv[i], "--regression") == 0) {
            do_regression = 1;
        } else if (strcmp(argv[i], "--occupancy") == 0) {
            do_occupancy = 1;
        } else if (strcmp(argv[i], "--list") == 0) {
            do_list = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    // Initialize CUDA
    int device = 0;
    cudaSetDevice(device);

    GpuSpecs gpu;
    gpu_specs_from_device(&gpu, device);

    printf("GPU: %s (%d SMs, %.0f GB/s peak, %.1f TFLOPS FP32)\n",
           gpu.name, gpu.n_sms, gpu.peak_bw_gbs, gpu.peak_fp32_tflops);
    printf("L2: %.0f MB, VRAM: %.0f MB\n\n",
           gpu.l2_bytes / (1024 * 1024),
           (float)gpu.vram_bytes / (1024 * 1024));

    // --list
    if (do_list) {
        printf("%-4s  %-28s  %4s  %4s  %6s  %4s\n",
               "ID", "Name", "B/d", "SM", "Layout", "Phys");
        for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
            const LbmKernelInfo* k = &LBM_KERNEL_INFO[v];
            printf("%-4d  %-28s  %4d  %4d  %6s  %4s\n",
                   v, k->name, k->bytes_per_dist, k->min_sm,
                   k->is_soa ? "SoA" : "AoS",
                   k->physics_valid ? "yes" : "no");
        }
        return 0;
    }

    // --occupancy
    if (do_occupancy) {
        report_all_occupancy(device);
        return 0;
    }

    // --validate
    if (do_validate) {
        return validate_all() > 0 ? 1 : 0;
    }

    // --regression
    if (do_regression) {
        printf("=== Performance Regression Check (128^3) ===\n");
        int regressions = 0;
        for (int i = 0; i < (int)BASELINE_COUNT; i++) {
            const BaselineEntry* b = &BASELINES[i];
            if (b->mlups_128 <= 0) continue;

            // DD is tested at 64^3
            int gs = (b->variant == LBM_DD_SOA) ? 64 : 128;

            BenchResult r = bench_run_variant(b->variant, gs, gs, gs,
                                             warmup, timing_steps, &gpu);
            if (r.mlups < 0) continue; // Skipped

            float expected = b->mlups_128;
            float threshold = expected * (1.0f - b->tolerance);
            int regressed = (r.mlups < threshold);

            printf("  %-28s  measured=%.0f  expected=%.0f  threshold=%.0f  [%s]\n",
                   LBM_KERNEL_INFO[b->variant].name,
                   r.mlups, expected, threshold,
                   regressed ? "REGRESSION" : "OK");

            if (regressed) regressions++;
        }
        printf("\nRegression check: %d failures\n", regressions);
        return regressions > 0 ? 1 : 0;
    }

    // Benchmark mode
    if (!do_all && specific_variant < 0) {
        print_usage();
        return 1;
    }

    if (!csv_output) bench_print_header();
    else printf("kernel,nx,ny,nz,mlups,bw_gbs,bw_pct,vram_mb,regime\n");

    for (int g = 0; g < n_grids; g++) {
        int gs = grid_sizes[g];
        for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
            if (!do_all && v != specific_variant) continue;

            BenchResult r = bench_run_variant((LbmKernelVariant)v,
                                             gs, gs, gs,
                                             warmup, timing_steps, &gpu);
            if (r.mlups < 0) continue; // Skipped

            if (csv_output) bench_print_result_csv(&r);
            else bench_print_result_table(&r);
        }
    }

    return 0;
}
