#include <cuda_runtime.h>
#include <stdio.h>

#define SASS_RE_EMBEDDED_RUNNER 1
#include "../probes/probe_tiling_hierarchical.cu"

static int check_cuda(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%d) at %s:%d\n",
                cudaGetErrorString(err), (int)err, file, line);
        return 1;
    }
    return 0;
}

#define CHECK_CUDA(expr) do { if (check_cuda((expr), __FILE__, __LINE__)) return 1; } while (0)

int main(void) {
    const int M = 128;
    const int N = 128;
    const int K = 8;
    const int conv_n = 256;
    const int ksize = 7;
    const int reduce_n = 512;
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_kernel = nullptr;
    float *d_reduce_in = nullptr;
    float *d_reduce_out = nullptr;
    float h_A[M * K];
    float h_B[K * N];
    float h_in[conv_n];
    float h_kernel[ksize];
    float h_reduce_in[reduce_n];

    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)((i & 31) - 15) * 0.03125f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)((i & 15) - 7) * 0.0625f;
    for (int i = 0; i < conv_n; ++i)
        h_in[i] = (float)(i & 63) * 0.015625f;
    for (int i = 0; i < ksize; ++i)
        h_kernel[i] = 0.125f * (float)(i + 1);
    for (int i = 0; i < reduce_n; ++i)
        h_reduce_in[i] = 1.0f + (float)(i & 7) * 0.25f;

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(h_in)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(h_in)));
    CHECK_CUDA(cudaMalloc(&d_kernel, sizeof(h_kernel)));
    CHECK_CUDA(cudaMalloc(&d_reduce_in, sizeof(h_reduce_in)));
    CHECK_CUDA(cudaMalloc(&d_reduce_out, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_reduce_in, h_reduce_in, sizeof(h_reduce_in), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(h_in)));
    CHECK_CUDA(cudaMemset(d_reduce_out, 0, sizeof(float)));

    probe_tile_hierarchical_gemm<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(d_C, d_A, d_B, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_tile_2level_conv<<<dim3(1, 1, 1), dim3(256, 1, 1), (256 + ksize - 1) * sizeof(float)>>>(d_out, d_in, d_kernel, conv_n, ksize);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_tile_recursive_half<<<dim3(1, 1, 1), dim3(512, 1, 1)>>>(d_reduce_out, d_reduce_in, reduce_n);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("tiling_hierarchical_runner_ok=1\n");

    cudaFree(d_reduce_out);
    cudaFree(d_reduce_in);
    cudaFree(d_kernel);
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    return 0;
}
