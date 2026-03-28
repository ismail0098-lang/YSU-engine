#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void __launch_bounds__(32)
probe_combo_warp_atomic_cache_profile_safe(float *out,
                                           float *accum,
                                           int *atomic_dst,
                                           const unsigned char *src_u8,
                                           const uint16_t *src_u16,
                                           int bias);

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
    const int count_u16 = 32;
    const int count_out = 32;

    unsigned char *h_u8 = (unsigned char*)malloc((size_t)count_u8);
    uint16_t *h_u16 = (uint16_t*)malloc((size_t)count_u16 * sizeof(uint16_t));
    float *h_out = (float*)malloc((size_t)count_out * sizeof(float));
    if (!h_u8 || !h_u16 || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }

    for (int i = 0; i < count_u8; ++i) h_u8[i] = (unsigned char)((i * 13 + 5) & 0xff);
    for (int i = 0; i < count_u16; ++i) h_u16[i] = (uint16_t)((i * 257 + 11) & 0xffff);
    memset(h_out, 0, (size_t)count_out * sizeof(float));

    unsigned char *d_u8 = nullptr;
    uint16_t *d_u16 = nullptr;
    float *d_out = nullptr;
    float *d_accum = nullptr;
    int *d_atomic = nullptr;
    ck(cudaMalloc(&d_u8, (size_t)count_u8), "cudaMalloc d_u8");
    ck(cudaMalloc(&d_u16, (size_t)count_u16 * sizeof(uint16_t)), "cudaMalloc d_u16");
    ck(cudaMalloc(&d_out, (size_t)count_out * sizeof(float)), "cudaMalloc d_out");
    ck(cudaMallocManaged(&d_accum, sizeof(float)), "cudaMallocManaged d_accum");
    ck(cudaMallocManaged(&d_atomic, sizeof(int)), "cudaMallocManaged d_atomic");

    ck(cudaMemcpy(d_u8, h_u8, (size_t)count_u8, cudaMemcpyHostToDevice), "copy d_u8");
    ck(cudaMemcpy(d_u16, h_u16, (size_t)count_u16 * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy d_u16");
    *d_accum = 0.0f;
    *d_atomic = 0;
    ck(cudaDeviceSynchronize(), "managed init sync");

    for (int i = 0; i < 20; ++i) {
        probe_combo_warp_atomic_cache_profile_safe<<<1, threads>>>(
            d_out, d_accum, d_atomic, d_u8, d_u16, 3);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start), "event start");
    ck(cudaEventCreate(&stop), "event stop");
    ck(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) {
        probe_combo_warp_atomic_cache_profile_safe<<<1, threads>>>(
            d_out, d_accum, d_atomic, d_u8, d_u16, 3 + (i & 3));
    }
    ck(cudaEventRecord(stop), "record stop");
    ck(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ck(cudaMemcpy(h_out, d_out, (size_t)count_out * sizeof(float), cudaMemcpyDeviceToHost), "copy out");

    double checksum = 0.0;
    for (int i = 0; i < count_out; ++i) checksum += (double)h_out[i];

    printf("combo_warp_atomic_cache_profile_safe_ms=%.6f\n", ms);
    printf("combo_warp_atomic_cache_profile_safe_iters=%d\n", iters);
    printf("combo_warp_atomic_cache_profile_safe_checksum=%.6f\n", checksum);
    printf("combo_warp_atomic_cache_profile_safe_accum=%.6f\n", (double)(*d_accum));
    printf("combo_warp_atomic_cache_profile_safe_atomic=%d\n", *d_atomic);

    cudaFree(d_u8);
    cudaFree(d_u16);
    cudaFree(d_out);
    cudaFree(d_accum);
    cudaFree(d_atomic);
    free(h_u8);
    free(h_u16);
    free(h_out);
    return 0;
}
