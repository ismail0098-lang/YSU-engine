#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define SASS_RE_EMBEDDED_RUNNER 1
#include "../probes/probe_cooperative_launch.cu"

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
    cudaDeviceProp prop{};
    const int n = 256;
    const int iterations = 2;
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_data = nullptr;
    float h[n];

    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    for (int i = 0; i < n; ++i)
        h[i] = 1.0f + (float)(i & 31) * 0.0625f;

    CHECK_CUDA(cudaMalloc(&d_in, sizeof(h)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(h)));
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h)));
    CHECK_CUDA(cudaMemcpy(d_in, h, sizeof(h), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_data, h, sizeof(h), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(h)));

    probe_block_sync<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_cg_thread_block<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_cg_warp_partition<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_cg_subwarp_partition<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    probe_cg_coalesced<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (prop.cooperativeLaunch) {
        void *args[] = { &d_data, (void*)&n, (void*)&iterations };
        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)probe_cg_grid_sync, dim3(1, 1, 1), dim3(128, 1, 1), args, 0, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    printf("cooperative_launch_supported=%d\n", prop.cooperativeLaunch ? 1 : 0);
    printf("cooperative_launch_runner_ok=1\n");

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_in);
    return 0;
}
