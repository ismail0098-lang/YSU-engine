#include <cuda_runtime.h>
#include <stdio.h>

#define SASS_RE_EMBEDDED_RUNNER 1
#include "../probes/barrier_sync2/probe_depbar_explicit.cu"

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
    float *d_in = nullptr;
    float *d_out = nullptr;
    float h[4 * n];

    for (int i = 0; i < 4 * n; ++i)
        h[i] = 1.0f + (float)(i & 15) * 0.125f;

    CHECK_CUDA(cudaMalloc(&d_in, sizeof(h)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * n));
    CHECK_CUDA(cudaMemcpy(d_in, h, sizeof(h), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float) * n));

    depbar_explicit<<<2, 128>>>(d_out, d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("depbar_explicit_runner_ok=1\n");

    cudaFree(d_out);
    cudaFree(d_in);
    return 0;
}
