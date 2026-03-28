#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../probes/data_movement/probe_cp_async_zfill.cu"

#define CHECK_CUDA(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%d) at %s:%d\n", \
                cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static int check_prefix(const char *label,
                        const unsigned char *got,
                        const unsigned char *expected,
                        size_t bytes) {
    int ok = memcmp(got, expected, bytes) == 0;
    printf("%s:", label);
    for (size_t i = 0; i < bytes; ++i) {
        printf(" %02x", (unsigned)got[i]);
    }
    printf("%s\n", ok ? "  OK" : "  MISMATCH");
    return ok ? 0 : 2;
}

int main(int argc, char **argv) {
    int profile_safe = 0;
    if (argc > 1 && strcmp(argv[1], "--profile-safe") == 0) {
        profile_safe = 1;
    }

    unsigned char h_src[32];
    unsigned char h_out[32];
    unsigned char expect_4_2[4] = { 0x11u, 0x22u, 0x00u, 0x00u };
    unsigned char expect_8_4[8] = { 0x11u, 0x22u, 0x33u, 0x44u, 0x00u, 0x00u, 0x00u, 0x00u };
    unsigned char expect_16_8[16] = {
        0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u,
        0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u
    };
    unsigned char expect_ignore[16] = {
        0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u,
        0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u
    };
    for (int i = 0; i < 32; ++i) {
        h_src[i] = (unsigned char)(0x11u * (unsigned)(i + 1));
        h_out[i] = 0xCDu;
    }

    unsigned char *d_src = NULL;
    unsigned char *d_out = NULL;
    CHECK_CUDA(cudaMalloc(&d_src, sizeof(h_src)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(h_out)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, sizeof(h_src), cudaMemcpyHostToDevice));

    int rc = 0;

    CHECK_CUDA(cudaMemset(d_out, 0xCD, sizeof(h_out)));
    probe_cp_async_zfill_4_2<<<1, 1>>>(d_out, d_src);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    rc |= check_prefix("zf_4_2", h_out, expect_4_2, sizeof(expect_4_2));

    CHECK_CUDA(cudaMemset(d_out, 0xCD, sizeof(h_out)));
    probe_cp_async_zfill_8_4<<<1, 1>>>(d_out, d_src);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    rc |= check_prefix("zf_8_4", h_out, expect_8_4, sizeof(expect_8_4));

    CHECK_CUDA(cudaMemset(d_out, 0xCD, sizeof(h_out)));
    probe_cp_async_zfill_16_8<<<1, 1>>>(d_out, d_src);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    rc |= check_prefix("zf_16_8", h_out, expect_16_8, sizeof(expect_16_8));

    CHECK_CUDA(cudaMemset(d_out, 0xCD, sizeof(h_out)));
    probe_cp_async_ignore_src<<<1, 1>>>(d_out, d_src, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    rc |= check_prefix("ignore_src_pred", h_out, expect_ignore, sizeof(expect_ignore));

    if (!profile_safe) {
        CHECK_CUDA(cudaMemset(d_out, 0xCD, sizeof(h_out)));
        probe_cp_async_misaligned<<<1, 1>>>(d_out, d_src);
        {
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err == cudaErrorMisalignedAddress) {
                printf("misaligned: runtime reported cudaErrorMisalignedAddress as expected\n");
                cudaGetLastError();
            } else if (sync_err == cudaSuccess) {
                CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
                printf("misaligned:");
                for (int i = 0; i < 16; ++i) {
                    printf(" %02x", (unsigned)h_out[i]);
                }
                printf("\n");
            } else {
                fprintf(stderr, "unexpected misaligned sync error: %s\n", cudaGetErrorString(sync_err));
                rc |= 2;
            }
        }
    } else {
        printf("misaligned: skipped in --profile-safe mode\n");
    }

    cudaFree(d_out);
    cudaFree(d_src);
    return rc;
}
