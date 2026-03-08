/*
 * Instant-NGP SASS Kernel Validation Harness
 *
 * Runs both the hand-written PTX kernels and reference CUDA implementations,
 * compares outputs, and reports maximum error.
 *
 * Build:
 *   nvcc -arch=sm_89 -O1 -allow-unsupported-compiler \
 *        -o ngp_validate ngp_validate.cu \
 *        hashgrid_encode.cu mlp_forward.cu volume_render.cu
 *
 * Run:
 *   ./ngp_validate
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ═══ Configuration ═══ */
#define NGP_NUM_LEVELS      12
#define NGP_FEATURES_PER    2
#define NGP_HASHMAP_SIZE    131072
#define NGP_TOTAL_FEATURES  (NGP_NUM_LEVELS * NGP_FEATURES_PER)

#define MLP_IN   27
#define MLP_OUT  4
#define MLP_HIDDEN 64

#define W0_SIZE  (MLP_HIDDEN * MLP_IN)
#define B0_SIZE  (MLP_HIDDEN)
#define W1_SIZE  (MLP_HIDDEN * MLP_HIDDEN)
#define B1_SIZE  (MLP_HIDDEN)
#define W2_SIZE  (MLP_OUT * MLP_HIDDEN)
#define B2_SIZE  (MLP_OUT)
#define TOTAL_WEIGHTS (W0_SIZE + B0_SIZE + W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE)

/* Test sizes */
#define TEST_N_POINTS     4096
#define TEST_N_RAYS       1024
#define TEST_N_STEPS      64
#define VOL_BLOCK_SIZE    128

/* ═══ Kernel declarations ═══ */

/* Hash grid */
extern "C" __global__ void hashgrid_encode_ptx(
    const float*, const float*, float*, int);
extern "C" __global__ void hashgrid_encode_ref(
    const float*, const float*, float*, int);

/* MLP */
extern "C" __global__ void mlp_forward_ptx(
    const float*, const float*, float*, int);
extern "C" __global__ void mlp_forward_ref(
    const float*, const float*, float*, int);

/* Volume rendering */
extern "C" __global__ void volume_render_ptx(
    const float*, const float*, float*, int, int);
extern "C" __global__ void volume_render_ref(
    const float*, const float*, float*, int, int);


/* ═══ Helpers ═══ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

/* Simple LCG for reproducibility */
static unsigned rng_state = 42;
static float det_rand(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(rng_state >> 16) / 65536.0f;
}

static float max_abs_error(const float *a, const float *b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float mean_abs_error(const float *a, const float *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += fabs(a[i] - b[i]);
    return (float)(sum / n);
}


/* ════════════════════════════════════════════════════════════════
 * Test 1: Hash Grid Encoding
 * ════════════════════════════════════════════════════════════════ */

static int test_hashgrid(void) {
    printf("═══════════════════════════════════════\n");
    printf("  Test 1: Hash Grid Encoding\n");
    printf("═══════════════════════════════════════\n");

    int N = TEST_N_POINTS;
    size_t pos_bytes   = (size_t)N * 3 * sizeof(float);
    size_t feat_bytes  = (size_t)N * NGP_TOTAL_FEATURES * sizeof(float);
    size_t table_bytes = (size_t)NGP_NUM_LEVELS * NGP_HASHMAP_SIZE * NGP_FEATURES_PER * sizeof(float);

    /* Host allocations */
    float *h_positions  = (float*)malloc(pos_bytes);
    float *h_hash_table = (float*)malloc(table_bytes);
    float *h_feat_ptx   = (float*)malloc(feat_bytes);
    float *h_feat_ref   = (float*)malloc(feat_bytes);

    /* Generate test data */
    printf("  Generating %d random positions + hash table (%zu MB)...\n",
           N, table_bytes / (1024*1024));
    rng_state = 42;
    for (int i = 0; i < N * 3; i++)
        h_positions[i] = det_rand() * 0.9f + 0.05f; /* [0.05, 0.95] */
    for (size_t i = 0; i < (size_t)NGP_NUM_LEVELS * NGP_HASHMAP_SIZE * NGP_FEATURES_PER; i++)
        h_hash_table[i] = (det_rand() - 0.5f) * 2.0f; /* [-1, 1] */

    /* Device allocations */
    float *d_pos, *d_table, *d_feat_ptx, *d_feat_ref;
    CUDA_CHECK(cudaMalloc(&d_pos,      pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_table,    table_bytes));
    CUDA_CHECK(cudaMalloc(&d_feat_ptx, feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_feat_ref, feat_bytes));

    CUDA_CHECK(cudaMemcpy(d_pos,   h_positions,  pos_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_hash_table, table_bytes, cudaMemcpyHostToDevice));

    /* Run both kernels */
    int grid = (N + 255) / 256;
    printf("  Running PTX kernel (%d threads)...\n", N);
    hashgrid_encode_ptx<<<grid, 256>>>(d_pos, d_table, d_feat_ptx, N);
    printf("  Running reference kernel...\n");
    hashgrid_encode_ref<<<grid, 256>>>(d_pos, d_table, d_feat_ref, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Compare */
    CUDA_CHECK(cudaMemcpy(h_feat_ptx, d_feat_ptx, feat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_feat_ref, d_feat_ref, feat_bytes, cudaMemcpyDeviceToHost));

    int total_feats = N * NGP_TOTAL_FEATURES;
    float max_err  = max_abs_error(h_feat_ptx, h_feat_ref, total_feats);
    float mean_err = mean_abs_error(h_feat_ptx, h_feat_ref, total_feats);

    printf("  Results: max_err = %.2e, mean_err = %.2e\n", max_err, mean_err);

    int pass = (max_err < 1e-4f);
    printf("  %s (threshold: 1e-4)\n\n", pass ? "PASS" : "FAIL");

    /* Benchmark */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int warmup = 10, iters = 100;
    for (int i = 0; i < warmup; i++)
        hashgrid_encode_ptx<<<grid, 256>>>(d_pos, d_table, d_feat_ptx, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        hashgrid_encode_ptx<<<grid, 256>>>(d_pos, d_table, d_feat_ptx, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ptx_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ptx_ms, start, stop));
    ptx_ms /= iters;

    for (int i = 0; i < warmup; i++)
        hashgrid_encode_ref<<<grid, 256>>>(d_pos, d_table, d_feat_ref, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        hashgrid_encode_ref<<<grid, 256>>>(d_pos, d_table, d_feat_ref, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ref_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ref_ms, start, stop));
    ref_ms /= iters;

    printf("  Benchmark (%d iters):\n", iters);
    printf("    PTX kernel: %.3f ms\n", ptx_ms);
    printf("    Reference:  %.3f ms\n", ref_ms);
    printf("    Speedup:    %.2fx\n\n", ref_ms / ptx_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_pos); cudaFree(d_table);
    cudaFree(d_feat_ptx); cudaFree(d_feat_ref);
    free(h_positions); free(h_hash_table);
    free(h_feat_ptx); free(h_feat_ref);

    return pass;
}


/* ════════════════════════════════════════════════════════════════
 * Test 2: MLP Forward
 * ════════════════════════════════════════════════════════════════ */

static int test_mlp(void) {
    printf("═══════════════════════════════════════\n");
    printf("  Test 2: MLP Forward Pass\n");
    printf("═══════════════════════════════════════\n");

    int N = TEST_N_POINTS;
    size_t in_bytes     = (size_t)N * MLP_IN * sizeof(float);
    size_t out_bytes    = (size_t)N * MLP_OUT * sizeof(float);
    size_t weight_bytes = (size_t)TOTAL_WEIGHTS * sizeof(float);

    float *h_input   = (float*)malloc(in_bytes);
    float *h_weights = (float*)malloc(weight_bytes);
    float *h_out_ptx = (float*)malloc(out_bytes);
    float *h_out_ref = (float*)malloc(out_bytes);

    /* Generate test data — small weights for numerical stability */
    rng_state = 123;
    for (int i = 0; i < N * MLP_IN; i++)
        h_input[i] = (det_rand() - 0.5f) * 2.0f;
    /* Xavier-like initialization: scale by 1/sqrt(fan_in) */
    for (int i = 0; i < TOTAL_WEIGHTS; i++)
        h_weights[i] = (det_rand() - 0.5f) * 0.2f;

    float *d_input, *d_weights, *d_out_ptx, *d_out_ref;
    CUDA_CHECK(cudaMalloc(&d_input,   in_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_ptx, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_ref, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_input,   h_input,   in_bytes,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_bytes, cudaMemcpyHostToDevice));

    int grid = (N + 127) / 128;
    printf("  Running PTX kernel (%d samples)...\n", N);
    mlp_forward_ptx<<<grid, 128>>>(d_input, d_weights, d_out_ptx, N);
    printf("  Running reference kernel...\n");
    mlp_forward_ref<<<grid, 128>>>(d_input, d_weights, d_out_ref, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_ptx, d_out_ptx, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_ref, d_out_ref, out_bytes, cudaMemcpyDeviceToHost));

    int total = N * MLP_OUT;
    float max_err  = max_abs_error(h_out_ptx, h_out_ref, total);
    float mean_err = mean_abs_error(h_out_ptx, h_out_ref, total);

    /* Higher tolerance for MLP due to sigmoid approximation (MUFU.EX2 vs expf) */
    printf("  Results: max_err = %.2e, mean_err = %.2e\n", max_err, mean_err);
    int pass = (max_err < 5e-3f);
    printf("  %s (threshold: 5e-3, includes MUFU.EX2 approx error)\n\n", pass ? "PASS" : "FAIL");

    /* Benchmark */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int warmup = 10, iters = 100;
    for (int i = 0; i < warmup; i++)
        mlp_forward_ptx<<<grid, 128>>>(d_input, d_weights, d_out_ptx, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        mlp_forward_ptx<<<grid, 128>>>(d_input, d_weights, d_out_ptx, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ptx_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ptx_ms, start, stop));
    ptx_ms /= iters;

    for (int i = 0; i < warmup; i++)
        mlp_forward_ref<<<grid, 128>>>(d_input, d_weights, d_out_ref, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        mlp_forward_ref<<<grid, 128>>>(d_input, d_weights, d_out_ref, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ref_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ref_ms, start, stop));
    ref_ms /= iters;

    printf("  Benchmark (%d iters):\n", iters);
    printf("    PTX kernel: %.3f ms\n", ptx_ms);
    printf("    Reference:  %.3f ms\n", ref_ms);
    printf("    Speedup:    %.2fx\n\n", ref_ms / ptx_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_input); cudaFree(d_weights);
    cudaFree(d_out_ptx); cudaFree(d_out_ref);
    free(h_input); free(h_weights);
    free(h_out_ptx); free(h_out_ref);

    return pass;
}


/* ════════════════════════════════════════════════════════════════
 * Test 3: Volume Rendering
 * ════════════════════════════════════════════════════════════════ */

static int test_volume_render(void) {
    printf("═══════════════════════════════════════\n");
    printf("  Test 3: Volume Rendering\n");
    printf("═══════════════════════════════════════\n");

    int N     = TEST_N_RAYS;
    int steps = TEST_N_STEPS;
    size_t rgba_bytes  = (size_t)N * steps * 4 * sizeof(float);
    size_t delta_bytes = (size_t)N * steps * sizeof(float);
    size_t out_bytes   = (size_t)N * 4 * sizeof(float);

    float *h_rgba   = (float*)malloc(rgba_bytes);
    float *h_deltas = (float*)malloc(delta_bytes);
    float *h_out_ptx = (float*)malloc(out_bytes);
    float *h_out_ref = (float*)malloc(out_bytes);

    /* Generate synthetic ray data */
    rng_state = 777;
    for (int r = 0; r < N; r++) {
        for (int s = 0; s < steps; s++) {
            h_rgba[(r * steps + s) * 4 + 0] = det_rand();       /* R */
            h_rgba[(r * steps + s) * 4 + 1] = det_rand();       /* G */
            h_rgba[(r * steps + s) * 4 + 2] = det_rand();       /* B */
            h_rgba[(r * steps + s) * 4 + 3] = det_rand() * 5.0f; /* sigma */
            h_deltas[r * steps + s] = 0.01f + det_rand() * 0.05f; /* dt */
        }
    }

    float *d_rgba, *d_deltas, *d_out_ptx, *d_out_ref;
    CUDA_CHECK(cudaMalloc(&d_rgba,    rgba_bytes));
    CUDA_CHECK(cudaMalloc(&d_deltas,  delta_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_ptx, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_ref, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_rgba,   h_rgba,   rgba_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_deltas, h_deltas, delta_bytes, cudaMemcpyHostToDevice));

    int grid = (N + VOL_BLOCK_SIZE - 1) / VOL_BLOCK_SIZE;
    printf("  Running PTX kernel (%d rays × %d steps)...\n", N, steps);
    volume_render_ptx<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ptx, N, steps);
    printf("  Running reference kernel...\n");
    volume_render_ref<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ref, N, steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_ptx, d_out_ptx, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_ref, d_out_ref, out_bytes, cudaMemcpyDeviceToHost));

    int total = N * 4;
    float max_err  = max_abs_error(h_out_ptx, h_out_ref, total);
    float mean_err = mean_abs_error(h_out_ptx, h_out_ref, total);

    /* Higher tolerance due to MUFU.EX2 approximation in exp() */
    printf("  Results: max_err = %.2e, mean_err = %.2e\n", max_err, mean_err);
    int pass = (max_err < 1e-2f);
    printf("  %s (threshold: 1e-2, includes MUFU.EX2 error accumulation)\n\n",
           pass ? "PASS" : "FAIL");

    /* Benchmark */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int warmup = 10, iters = 100;
    for (int i = 0; i < warmup; i++)
        volume_render_ptx<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ptx, N, steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        volume_render_ptx<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ptx, N, steps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ptx_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ptx_ms, start, stop));
    ptx_ms /= iters;

    for (int i = 0; i < warmup; i++)
        volume_render_ref<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ref, N, steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        volume_render_ref<<<grid, VOL_BLOCK_SIZE>>>(d_rgba, d_deltas, d_out_ref, N, steps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ref_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ref_ms, start, stop));
    ref_ms /= iters;

    printf("  Benchmark (%d iters):\n", iters);
    printf("    PTX kernel: %.3f ms\n", ptx_ms);
    printf("    Reference:  %.3f ms\n", ref_ms);
    printf("    Speedup:    %.2fx\n\n", ref_ms / ptx_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_rgba); cudaFree(d_deltas);
    cudaFree(d_out_ptx); cudaFree(d_out_ref);
    free(h_rgba); free(h_deltas);
    free(h_out_ptx); free(h_out_ref);

    return pass;
}


/* ════════════════════════════════════════════════════════════════
 * Main
 * ════════════════════════════════════════════════════════════════ */

int main(void) {
    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clockMHz = 0;
    cudaDeviceGetAttribute(&clockMHz, cudaDevAttrClockRate, 0);
    clockMHz /= 1000;
    printf("\n");
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  Instant-NGP SASS Kernel Validation Suite        ║\n");
    printf("╠═══════════════════════════════════════════════════╣\n");
    printf("║  GPU: %-43s ║\n", prop.name);
    printf("║  SM:  %d.%d   SMs: %-3d   Clock: %d MHz          ║\n",
           prop.major, prop.minor, prop.multiProcessorCount, clockMHz);
    printf("║  Shared mem/block: %zu KB                        ║\n",
           prop.sharedMemPerBlock / 1024);
    printf("╚═══════════════════════════════════════════════════╝\n\n");

    int pass_hash    = test_hashgrid();
    int pass_mlp     = test_mlp();
    int pass_volrend = test_volume_render();

    printf("═══════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("═══════════════════════════════════════\n");
    printf("  Hash Grid Encoding:  %s\n", pass_hash    ? "PASS" : "FAIL");
    printf("  MLP Forward:         %s\n", pass_mlp     ? "PASS" : "FAIL");
    printf("  Volume Rendering:    %s\n", pass_volrend ? "PASS" : "FAIL");
    printf("═══════════════════════════════════════\n");

    int all_pass = pass_hash && pass_mlp && pass_volrend;
    printf("  Overall: %s\n\n", all_pass ? "ALL PASSED" : "SOME FAILED");

    return all_pass ? 0 : 1;
}
