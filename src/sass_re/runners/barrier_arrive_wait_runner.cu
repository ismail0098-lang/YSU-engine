#include <cuda_runtime.h>
#include <stdio.h>

#define SASS_RE_EMBEDDED_RUNNER 1
#include "../probes/probe_barrier_arrive_wait.cu"

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
    const int n = 256;
    const int stages = 4;
    float *d_in_pc = nullptr;
    float *d_out_pc = nullptr;
    float *d_in_pipe = nullptr;
    float *d_out_pipe = nullptr;
    float h_in_pc[n];
    float h_in_pipe[128 * stages];

    for (int i = 0; i < n; ++i)
        h_in_pc[i] = 0.25f + (float)(i & 31) * 0.03125f;
    for (int i = 0; i < 128 * stages; ++i)
        h_in_pipe[i] = 1.0f + (float)(i & 63) * 0.015625f;

    CHECK_CUDA(cudaMalloc(&d_in_pc, sizeof(h_in_pc)));
    CHECK_CUDA(cudaMalloc(&d_out_pc, sizeof(h_in_pc)));
    CHECK_CUDA(cudaMalloc(&d_in_pipe, sizeof(h_in_pipe)));
    CHECK_CUDA(cudaMalloc(&d_out_pipe, sizeof(h_in_pipe)));

    CHECK_CUDA(cudaMemcpy(d_in_pc, h_in_pc, sizeof(h_in_pc), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_pipe, h_in_pipe, sizeof(h_in_pipe), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out_pc, 0, sizeof(h_in_pc)));
    CHECK_CUDA(cudaMemset(d_out_pipe, 0, sizeof(h_in_pipe)));

    probe_arrive_wait_producer_consumer<<<1, 256>>>(d_out_pc, d_in_pc, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    probe_arrive_wait_pipeline<<<1, 256>>>(d_out_pipe, d_in_pipe, stages);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("barrier_arrive_wait_runner_ok=1\n");

    cudaFree(d_out_pipe);
    cudaFree(d_in_pipe);
    cudaFree(d_out_pc);
    cudaFree(d_in_pc);
    return 0;
}
