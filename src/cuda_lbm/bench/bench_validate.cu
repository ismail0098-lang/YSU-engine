// Physics correctness validation for LBM kernels.
// Runs quiescent simulations and checks conservation laws.

#include "lbm_kernels.h"
#include "lbm_metrics.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations from host_wrappers.cu
extern int launch_lbm_step(LbmKernelVariant variant, const void* grid,
                           void* bufs, const void* f_in, void* f_out,
                           int parity, cudaStream_t stream);
extern int launch_lbm_init(LbmKernelVariant variant, const void* grid,
                           void* bufs, float rho, float ux, float uy, float uz,
                           cudaStream_t stream);

// Precision-specific tolerances for conservation checks.
static float get_tolerance(LbmKernelVariant v) {
    switch (v) {
    case LBM_FP8_E4M3_AOS: case LBM_FP8_E4M3_SOA:
        return 0.15f;   // FP8 e4m3: ~12.5% relative error
    case LBM_FP8_E5M2_AOS: case LBM_FP8_E5M2_SOA:
        return 0.15f;   // FP8 e5m2: ~25% relative error
    case LBM_INT8_AOS: case LBM_INT8_SOA:
        return 1e-2f;
    case LBM_FP16_AOS: case LBM_FP16_SOA: case LBM_FP16_SOA_HALF2:
    case LBM_BF16_AOS: case LBM_BF16_SOA:
    case LBM_INT16_AOS: case LBM_INT16_SOA:
        return 1e-3f;
    case LBM_FP64_AOS: case LBM_FP64_SOA: case LBM_DD_SOA:
        return 1e-10f;
    default:
        return 1e-6f;   // FP32 default
    }
}

// Run validation for a single kernel variant.
// Uses 32^3 grid, 100 steps, quiescent flow (u=0, rho=1, tau=0.6).
ValidationResult validate_kernel(LbmKernelVariant variant) {
    ValidationResult vr;
    memset(&vr, 0, sizeof(vr));
    vr.variant = variant;
    vr.nx = 32; vr.ny = 32; vr.nz = 32;
    vr.steps = 100;
    vr.tolerance = get_tolerance(variant);

    const LbmKernelInfo* info = &LBM_KERNEL_INFO[variant];
    int nx = 32, ny = 32, nz = 32;
    size_t n_cells = (size_t)nx * ny * nz;

    // Skip non-physics kernels
    if (!info->physics_valid) {
        vr.passed = 1; // Skip = pass
        return vr;
    }

    // Check SM compatibility
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm = prop.major * 10 + prop.minor;
    if (sm < info->min_sm) {
        fprintf(stderr, "  [SKIP] %s: requires SM %d, have SM %d\n",
                info->name, info->min_sm, sm);
        vr.passed = 1; // Skip = pass
        return vr;
    }

    // DD is capped at 64^3 in benchmarks but we use 32^3 here
    // Allocate buffers
    typedef struct { int nx, ny, nz, n_cells; } LBMGrid;
    typedef struct { void *f_a, *f_b, *f_c, *f_d; float *rho, *u, *tau, *force; } LBMBuffers;

    void *f_a = NULL, *f_b = NULL, *f_c = NULL, *f_d = NULL;
    float *rho = NULL, *u = NULL, *tau_d = NULL, *force_d = NULL;

    size_t dirs = 19;
    if (variant == LBM_DD_SOA) {
        size_t dd_buf = dirs * n_cells * sizeof(double);
        cudaMalloc(&f_a, dd_buf);
        cudaMalloc(&f_b, dd_buf);
        cudaMalloc(&f_c, dd_buf);
        cudaMalloc(&f_d, dd_buf);
    } else if (info->is_aa) {
        size_t buf;
        if (info->is_soa) buf = dirs * n_cells * info->bytes_per_dist;
        else buf = 20 * n_cells * info->bytes_per_dist;
        cudaMalloc(&f_a, buf);
    } else {
        size_t per_buf;
        if (info->is_soa) per_buf = dirs * n_cells * info->bytes_per_dist;
        else per_buf = 20 * n_cells * info->bytes_per_dist;
        // Nibble types: halve the buffer
        if (variant == LBM_INT4_SOA || variant == LBM_FP4_SOA) {
            per_buf = (dirs * n_cells + 1) / 2;
        }
        cudaMalloc(&f_a, per_buf);
        cudaMalloc(&f_b, per_buf);
    }

    cudaMalloc(&rho, n_cells * sizeof(float));
    cudaMalloc(&u, n_cells * 3 * sizeof(float));
    cudaMalloc(&tau_d, n_cells * sizeof(float));
    cudaMalloc(&force_d, n_cells * 3 * sizeof(float));

    // Set tau = 0.6, force = 0
    {
        float* h_tau = (float*)malloc(n_cells * sizeof(float));
        for (size_t i = 0; i < n_cells; i++) h_tau[i] = 0.6f;
        cudaMemcpy(tau_d, h_tau, n_cells * sizeof(float), cudaMemcpyHostToDevice);
        free(h_tau);
    }
    cudaMemset(force_d, 0, n_cells * 3 * sizeof(float));

    LBMGrid grid = {nx, ny, nz, (int)n_cells};
    LBMBuffers bufs = {f_a, f_b, f_c, f_d, rho, u, tau_d, force_d};

    // Initialize to equilibrium (rho=1, u=0)
    launch_lbm_init(variant, (const void*)&grid, (void*)&bufs,
                    1.0f, 0.0f, 0.0f, 0.0f, 0);
    cudaDeviceSynchronize();

    // Measure initial mass
    float* h_rho = (float*)malloc(n_cells * sizeof(float));
    cudaMemcpy(h_rho, rho, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    double init_mass = 0.0;
    for (size_t i = 0; i < n_cells; i++) init_mass += h_rho[i];
    vr.initial_mass = (float)init_mass;

    // Run steps
    for (int s = 0; s < vr.steps; s++) {
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
    cudaDeviceSynchronize();

    // Measure final mass and check density
    cudaMemcpy(h_rho, rho, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    double final_mass = 0.0;
    int density_ok = 1;
    for (size_t i = 0; i < n_cells; i++) {
        float r = h_rho[i];
        final_mass += r;
        if (r != r || r < 0.5f || r > 2.0f) { // NaN or out of range
            density_ok = 0;
        }
    }
    vr.final_mass = (float)final_mass;
    vr.mass_drift = (float)(fabs(final_mass - init_mass) / fabs(init_mass));
    vr.density_stable = density_ok;

    // Measure momentum
    float* h_u = (float*)malloc(n_cells * 3 * sizeof(float));
    cudaMemcpy(h_u, u, n_cells * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    double mom_x = 0, mom_y = 0, mom_z = 0;
    for (size_t i = 0; i < n_cells; i++) {
        mom_x += h_rho[i] * h_u[i];
        mom_y += h_rho[i] * h_u[n_cells + i];
        mom_z += h_rho[i] * h_u[2 * n_cells + i];
    }
    vr.momentum_drift = (float)sqrt(mom_x * mom_x + mom_y * mom_y + mom_z * mom_z);

    // Pass/fail
    vr.passed = (vr.mass_drift < vr.tolerance) &&
                (vr.momentum_drift < vr.tolerance) &&
                density_ok;

    // Cleanup
    free(h_rho);
    free(h_u);
    if (f_a) cudaFree(f_a);
    if (f_b) cudaFree(f_b);
    if (f_c) cudaFree(f_c);
    if (f_d) cudaFree(f_d);
    cudaFree(rho);
    cudaFree(u);
    cudaFree(tau_d);
    cudaFree(force_d);

    return vr;
}

// Print validation result.
void validate_print_result(const ValidationResult* vr) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[vr->variant];
    printf("  %-28s  mass_drift=%.2e  mom_drift=%.2e  density=%s  [%s]\n",
           info->name,
           vr->mass_drift,
           vr->momentum_drift,
           vr->density_stable ? "OK" : "FAIL",
           vr->passed ? "PASS" : "FAIL");
}

// Run validation for all physics-valid kernels.
// Returns number of failures.
int validate_all(void) {
    printf("=== Physics Validation (32^3, 100 steps, quiescent) ===\n");
    int failures = 0;
    for (int v = 0; v < LBM_VARIANT_COUNT; v++) {
        const LbmKernelInfo* info = &LBM_KERNEL_INFO[v];
        if (!info->physics_valid) continue;

        ValidationResult vr = validate_kernel((LbmKernelVariant)v);
        validate_print_result(&vr);
        if (!vr.passed) failures++;
    }
    printf("Validation: %d failures\n", failures);
    return failures;
}
