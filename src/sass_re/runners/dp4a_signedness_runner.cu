#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../probes/probe_int8_dp4a.cu"

#define CHECK_CUDA(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%d) at %s:%d\n", \
                cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static uint32_t pack4(unsigned char a0, unsigned char a1, unsigned char a2, unsigned char a3) {
    return (uint32_t)a0 |
           ((uint32_t)a1 << 8) |
           ((uint32_t)a2 << 16) |
           ((uint32_t)a3 << 24);
}

static int run_case(const char *label,
                    const void *kernel,
                    int expected) {
    const uint32_t h_a = pack4(0x80u, 0x7Fu, 0xFFu, 0x01u);
    const uint32_t h_b = pack4(0x02u, 0x80u, 0x01u, 0xFFu);
    uint32_t *d_a = NULL;
    uint32_t *d_b = NULL;
    int *d_out = NULL;
    int h_out = 0;

    CHECK_CUDA(cudaMalloc(&d_a, sizeof(h_a)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(h_b)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(h_out)));
    CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(h_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(h_b), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(h_out)));

    void *args[] = { &d_out, &d_a, &d_b };
    CHECK_CUDA(cudaLaunchKernel(kernel, dim3(1, 1, 1), dim3(1, 1, 1), args, 0, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    printf("%s: got=%d expected=%d\n", label, h_out, expected);

    cudaFree(d_out);
    cudaFree(d_b);
    cudaFree(d_a);
    return h_out == expected ? 0 : 2;
}

int main(void) {
    int rc = 0;
    rc |= run_case("U8.U8", (const void *)probe_dp4a_u8_u8, 17022);
    rc |= run_case("U8.S8", (const void *)probe_dp4a_u8_s8, -15746);
    rc |= run_case("S8.U8", (const void *)probe_dp4a_s8_u8, 16254);
    rc |= run_case("S8.S8", (const void *)probe_dp4a_s8_s8, -16514);
    return rc;
}
