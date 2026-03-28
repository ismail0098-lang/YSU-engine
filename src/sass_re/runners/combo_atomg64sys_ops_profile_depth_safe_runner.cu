#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void __launch_bounds__(32)
probe_combo_atomg64sys_ops_profile_depth_safe(
    unsigned long long *out,
    unsigned long long *add_dst,
    unsigned long long *min_dst,
    unsigned long long *max_dst,
    unsigned long long *and_dst,
    unsigned long long *or_dst,
    unsigned long long *xor_dst,
    const unsigned char *src_u8,
    const unsigned long long *src_u64,
    unsigned long long bias);

static void ck(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 400;
    const int threads = 32;
    const int count_u8 = 1024;
    const int count_u64 = 32;

    unsigned char *h_u8 = (unsigned char*)malloc((size_t)count_u8);
    unsigned long long *h_u64 =
        (unsigned long long*)malloc((size_t)count_u64 * sizeof(unsigned long long));
    unsigned long long *h_out =
        (unsigned long long*)malloc((size_t)count_u64 * sizeof(unsigned long long));
    if (!h_u8 || !h_u64 || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }

    for (int i = 0; i < count_u8; ++i) h_u8[i] = (unsigned char)((i * 19 + 7) & 0xff);
    for (int i = 0; i < count_u64; ++i) {
        h_u64[i] = 0x1234567800000000ull ^ (unsigned long long)(i * 0x9e3779b1u);
    }
    memset(h_out, 0, (size_t)count_u64 * sizeof(unsigned long long));

    unsigned char *d_u8 = nullptr;
    unsigned long long *d_u64 = nullptr;
    unsigned long long *d_out = nullptr;
    unsigned long long *d_add = nullptr;
    unsigned long long *d_min = nullptr;
    unsigned long long *d_max = nullptr;
    unsigned long long *d_and = nullptr;
    unsigned long long *d_or = nullptr;
    unsigned long long *d_xor = nullptr;

    ck(cudaMalloc(&d_u8, (size_t)count_u8), "cudaMalloc d_u8");
    ck(cudaMalloc(&d_u64, (size_t)count_u64 * sizeof(unsigned long long)), "cudaMalloc d_u64");
    ck(cudaMalloc(&d_out, (size_t)count_u64 * sizeof(unsigned long long)), "cudaMalloc d_out");
    ck(cudaMallocManaged(&d_add, sizeof(unsigned long long)), "cudaMallocManaged d_add");
    ck(cudaMallocManaged(&d_min, sizeof(unsigned long long)), "cudaMallocManaged d_min");
    ck(cudaMallocManaged(&d_max, sizeof(unsigned long long)), "cudaMallocManaged d_max");
    ck(cudaMallocManaged(&d_and, sizeof(unsigned long long)), "cudaMallocManaged d_and");
    ck(cudaMallocManaged(&d_or, sizeof(unsigned long long)), "cudaMallocManaged d_or");
    ck(cudaMallocManaged(&d_xor, sizeof(unsigned long long)), "cudaMallocManaged d_xor");

    ck(cudaMemcpy(d_u8, h_u8, (size_t)count_u8, cudaMemcpyHostToDevice), "copy d_u8");
    ck(cudaMemcpy(d_u64, h_u64, (size_t)count_u64 * sizeof(unsigned long long), cudaMemcpyHostToDevice), "copy d_u64");

    *d_add = 1ull;
    *d_min = ~0ull;
    *d_max = 0ull;
    *d_and = ~0ull;
    *d_or = 0ull;
    *d_xor = 0ull;
    ck(cudaDeviceSynchronize(), "managed init sync");

    for (int i = 0; i < 20; ++i) {
        probe_combo_atomg64sys_ops_profile_depth_safe<<<1, threads>>>(
            d_out, d_add, d_min, d_max, d_and, d_or, d_xor,
            d_u8, d_u64, 0x1234abcd55aa7711ull);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start), "event start");
    ck(cudaEventCreate(&stop), "event stop");
    ck(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) {
        probe_combo_atomg64sys_ops_profile_depth_safe<<<1, threads>>>(
            d_out, d_add, d_min, d_max, d_and, d_or, d_xor,
            d_u8, d_u64, 0x1234abcd55aa7711ull + (unsigned long long)i);
    }
    ck(cudaEventRecord(stop), "record stop");
    ck(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ck(cudaMemcpy(h_out, d_out, (size_t)count_u64 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy out");

    unsigned long long checksum = 0ull;
    for (int i = 0; i < count_u64; ++i) checksum ^= h_out[i];

    printf("combo_atomg64sys_ops_profile_depth_safe_ms=%.6f\n", ms);
    printf("combo_atomg64sys_ops_profile_depth_safe_iters=%d\n", iters);
    printf("combo_atomg64sys_ops_profile_depth_safe_checksum=0x%016llx\n",
           (unsigned long long)checksum);

    cudaFree(d_u8);
    cudaFree(d_u64);
    cudaFree(d_out);
    cudaFree(d_add);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_and);
    cudaFree(d_or);
    cudaFree(d_xor);
    free(h_u8);
    free(h_u64);
    free(h_out);
    return 0;
}
