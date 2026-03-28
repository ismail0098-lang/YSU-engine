/*
 * SASS RE: INT + Bitops probe runner for ncu profiling.
 * Launches key kernels that produced new SASS mnemonics.
 */
#include <stdio.h>
#include <cuda_runtime.h>

// Include probes that produced new mnemonics
#include "probe_int8_tile_dp4a_stencil.cu"
#include "probe_int16_tile_stencil.cu"
#include "probe_int8_tile_pack4.cu"
#include "probe_int8_tile_coarsened.cu"
#include "probe_int64_tile_clz64.cu"
#include "probe_int64_tile_popc64.cu"
#include "probe_int32_tile_bitonic_sort.cu"
#include "probe_int32_tile_compact.cu"
#include "probe_brev_tile_fft.cu"
#include "probe_popc_tile_hamming.cu"
#include "probe_iabs_tile_l1norm.cu"
#include "probe_clz_tile_log2.cu"

int main() {
    printf("=== INT+Bitops Probe Runner ===\n");
    int N = 1024*1024;
    void *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, N*8); cudaMalloc(&d_b, N*8);
    cudaMalloc(&d_c, N*8); cudaMalloc(&d_d, N*8);
    cudaMemset(d_a, 1, N*8); cudaMemset(d_b, 2, N*8);
    cudaMemset(d_c, 0, N*8);

    int blocks = (N+127)/128;
    dim3 g2d((1024+15)/16, (1024+15)/16);

    // INT8 stencil (LDS.S8)
    int8_dp4a_stencil<<<g2d, 128>>>((signed char*)d_c, (const signed char*)d_a, 1024, 1024);

    // INT16 stencil (LDS.S16)
    int16_stencil<<<g2d, 256>>>((short*)d_c, (const short*)d_a, 1024, 1024);

    // INT8 pack4 (UPRMT)
    int8_pack4<<<blocks, 128>>>((int*)d_c, (const int*)d_a, N/4);

    // INT8 coarsened
    int8_coarsened<<<blocks, 128>>>((int*)d_c, (const int*)d_a, N/4);

    // INT64 CLZ (F2I.U64.TRUNC path)
    int64_clz<<<blocks, 128>>>((int*)d_c, (const unsigned long long*)d_a, N/2);

    // INT64 POPC
    int64_popc<<<blocks, 128>>>((int*)d_c, (const unsigned long long*)d_a, N/2);

    // INT32 bitonic sort
    int32_bitonic_warp<<<N/32, 32>>>((int*)d_a, N);

    // INT32 compact
    int cnt = 0; int *d_cnt; cudaMalloc(&d_cnt, 4); cudaMemset(d_cnt, 0, 4);
    int32_compact<<<blocks, 128>>>((int*)d_c, d_cnt, (const int*)d_a, 50, N);

    // BREV FFT permute
    brev_fft_permute<<<(1024+255)/256, 256>>>((float*)d_c, (const float*)d_a, 10);

    // POPC Hamming distance
    popc_hamming<<<blocks, 128>>>((int*)d_c, (const unsigned*)d_a, (const unsigned*)d_b, N);

    // IABS L1 norm
    cudaMemset(d_c, 0, 4);
    iabs_l1norm<<<blocks, 128>>>((int*)d_c, (const int*)d_a, N);

    // CLZ log2
    clz_log2<<<blocks, 128>>>((int*)d_c, (const unsigned*)d_a, N);

    cudaDeviceSynchronize();
    printf("All kernels launched OK\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_cnt);
    return 0;
}
