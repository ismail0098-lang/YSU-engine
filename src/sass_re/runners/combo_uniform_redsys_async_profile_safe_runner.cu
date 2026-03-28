#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void __launch_bounds__(32)
probe_combo_uniform_redsys_async_profile_safe(
    int *out,
    int *sys_min_dst,
    int *sys_max_dst,
    int *sys_add_dst,
    float *sys_fadd_dst,
    const unsigned char *src_u8,
    const uint16_t *src_u16,
    const uint32_t *src_u32,
    uint64_t seed,
    int mode,
    int bias,
    int limit);

static void ck(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 300;
    const int threads = 32;
    const int count_u8 = 1024;
    const int count_u16 = 64;
    const int count_u32 = 32;

    unsigned char *h_u8 = (unsigned char*)malloc((size_t)count_u8);
    uint16_t *h_u16 = (uint16_t*)malloc((size_t)count_u16 * sizeof(uint16_t));
    uint32_t *h_u32 = (uint32_t*)malloc((size_t)count_u32 * sizeof(uint32_t));
    int *h_out = (int*)malloc((size_t)threads * sizeof(int));
    if (!h_u8 || !h_u16 || !h_u32 || !h_out) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }

    for (int i = 0; i < count_u8; ++i) h_u8[i] = (unsigned char)((i * 17 + 9) & 0xff);
    for (int i = 0; i < count_u16; ++i) h_u16[i] = (uint16_t)((i * 257 + 13) & 0xffff);
    for (int i = 0; i < count_u32; ++i) h_u32[i] = 0x10203040u ^ (uint32_t)(i * 0x9e37u);
    memset(h_out, 0, (size_t)threads * sizeof(int));

    unsigned char *d_u8 = nullptr;
    uint16_t *d_u16 = nullptr;
    uint32_t *d_u32 = nullptr;
    int *d_out = nullptr;
    int *d_min = nullptr;
    int *d_max = nullptr;
    int *d_add = nullptr;
    float *d_fadd = nullptr;

    ck(cudaMalloc(&d_u8, (size_t)count_u8), "cudaMalloc d_u8");
    ck(cudaMalloc(&d_u16, (size_t)count_u16 * sizeof(uint16_t)), "cudaMalloc d_u16");
    ck(cudaMalloc(&d_u32, (size_t)count_u32 * sizeof(uint32_t)), "cudaMalloc d_u32");
    ck(cudaMalloc(&d_out, (size_t)threads * sizeof(int)), "cudaMalloc d_out");
    ck(cudaMallocManaged(&d_min, sizeof(int)), "cudaMallocManaged d_min");
    ck(cudaMallocManaged(&d_max, sizeof(int)), "cudaMallocManaged d_max");
    ck(cudaMallocManaged(&d_add, sizeof(int)), "cudaMallocManaged d_add");
    ck(cudaMallocManaged(&d_fadd, sizeof(float)), "cudaMallocManaged d_fadd");

    ck(cudaMemcpy(d_u8, h_u8, (size_t)count_u8, cudaMemcpyHostToDevice), "copy d_u8");
    ck(cudaMemcpy(d_u16, h_u16, (size_t)count_u16 * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy d_u16");
    ck(cudaMemcpy(d_u32, h_u32, (size_t)count_u32 * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy d_u32");

    *d_min = 0x7fffffff;
    *d_max = -0x7fffffff;
    *d_add = 0;
    *d_fadd = 0.0f;
    ck(cudaDeviceSynchronize(), "managed init sync");

    for (int i = 0; i < 20; ++i) {
        probe_combo_uniform_redsys_async_profile_safe<<<1, threads>>>(
            d_out, d_min, d_max, d_add, d_fadd, d_u8, d_u16, d_u32,
            0x0123456789abcdefull, 3, 5, 127);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start), "event start");
    ck(cudaEventCreate(&stop), "event stop");
    ck(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) {
        probe_combo_uniform_redsys_async_profile_safe<<<1, threads>>>(
            d_out, d_min, d_max, d_add, d_fadd, d_u8, d_u16, d_u32,
            0x0123456789abcdefull + (uint64_t)i, 3 + (i & 1), 5, 127);
    }
    ck(cudaEventRecord(stop), "record stop");
    ck(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ck(cudaMemcpy(h_out, d_out, (size_t)threads * sizeof(int), cudaMemcpyDeviceToHost), "copy out");

    unsigned long long checksum = 0ull;
    for (int i = 0; i < threads; ++i) {
        checksum ^= (unsigned long long)(unsigned int)h_out[i];
    }
    checksum ^= (unsigned long long)(unsigned int)(*d_min);
    checksum ^= (unsigned long long)(unsigned int)(*d_max);
    checksum ^= (unsigned long long)(unsigned int)(*d_add);

    printf("combo_uniform_redsys_async_profile_safe_ms=%.6f\n", ms);
    printf("combo_uniform_redsys_async_profile_safe_iters=%d\n", iters);
    printf("combo_uniform_redsys_async_profile_safe_checksum=0x%016llx\n", checksum);

    cudaFree(d_u8);
    cudaFree(d_u16);
    cudaFree(d_u32);
    cudaFree(d_out);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_add);
    cudaFree(d_fadd);
    free(h_u8);
    free(h_u16);
    free(h_u32);
    free(h_out);
    return 0;
}
