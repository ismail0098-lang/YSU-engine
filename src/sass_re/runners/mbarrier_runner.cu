#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "../probes/mbarrier/probe_mbarrier_core.cu"

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
    uint64_t *d_tokens = nullptr;
    uint32_t *d_pending = nullptr;
    uint64_t host_tokens[2] = {0, 0};
    uint32_t host_pending[2] = {0, 0};

    CHECK_CUDA(cudaMalloc(&d_tokens, sizeof(host_tokens)));
    CHECK_CUDA(cudaMalloc(&d_pending, sizeof(host_pending)));
    CHECK_CUDA(cudaMemset(d_tokens, 0, sizeof(host_tokens)));
    CHECK_CUDA(cudaMemset(d_pending, 0, sizeof(host_pending)));

    uint64_t *d_tokens_init = d_tokens;
    uint32_t *d_pending_init = d_pending;
    void *args0[] = { &d_tokens_init, &d_pending_init };
    CHECK_CUDA(cudaLaunchKernel((const void*)probe_mbarrier_init_arrive_wait,
                                dim3(1, 1, 1), dim3(128, 1, 1), args0, 0, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    uint64_t *d_tokens_drop = d_tokens + 1;
    uint32_t *d_pending_drop = d_pending + 1;
    void *args1[] = { &d_tokens_drop, &d_pending_drop };
    CHECK_CUDA(cudaLaunchKernel((const void*)probe_mbarrier_arrive_drop,
                                dim3(1, 1, 1), dim3(128, 1, 1), args1, 0, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_tokens, d_tokens, sizeof(host_tokens), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_pending, d_pending, sizeof(host_pending), cudaMemcpyDeviceToHost));

    printf("mbarrier_init_token=%llu pending=%u\n",
           (unsigned long long)host_tokens[0], (unsigned)host_pending[0]);
    printf("mbarrier_drop_token=%llu pending=%u\n",
           (unsigned long long)host_tokens[1], (unsigned)host_pending[1]);
    printf("mbarrier_try_wait_skipped=1\n");

    cudaFree(d_pending);
    cudaFree(d_tokens);
    return 0;
}
