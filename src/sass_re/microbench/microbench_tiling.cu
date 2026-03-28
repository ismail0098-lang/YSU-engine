/*
 * SASS RE: Tiling Pattern Benchmark
 * Measures execution time for all tiling probe kernels.
 * Build: nvcc -arch=sm_89 -O3 ... -Iprobes -o tiling_bench microbench_tiling.cu
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "probe_tiling_2d_stencil.cu"
#include "probe_tiling_register_blocking.cu"
#include "probe_tiling_smem_patterns.cu"

int main() {
    printf("=== Tiling Probe Benchmark ===\n\n");
    int N = 1024;
    int n_elem = N * N;
    float *d_in, *d_out, *d_A, *d_B;
    cudaMalloc(&d_in, n_elem*4*sizeof(float));
    cudaMalloc(&d_out, n_elem*4*sizeof(float));
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMemset(d_in, 0, n_elem*4*sizeof(float));
    cudaMemset(d_A, 0, N*N*sizeof(float));
    cudaMemset(d_B, 0, N*N*sizeof(float));
    cudaEvent_t t0, t1; float ms;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    printf("%-40s %10s\n", "Kernel", "Time(ms)");
    printf("%-40s %10s\n", "----------------------------------------","----------");
    dim3 b2d(256), g2d((N+15)/16,(N+15)/16);
    dim3 g3d((64+7)/8,(64+7)/8,(64+3)/4);

#define RUN(label, launch) \
    launch; cudaDeviceSynchronize(); cudaEventRecord(t0); \
    for(int _r=0;_r<10;_r++){launch;} \
    cudaEventRecord(t1); cudaEventSynchronize(t1); \
    cudaEventElapsedTime(&ms,t0,t1); \
    printf("%-40s %10.3f\n", label, ms/10.0f);

    RUN("5pt stencil 16x16 r=1",         (probe_tile_5pt_r1<<<g2d,b2d>>>(d_out,d_in,N,N)))
    RUN("9pt stencil 16x16 r=1",         (probe_tile_9pt_r1<<<g2d,b2d>>>(d_out,d_in,N,N)))
    RUN("25pt stencil 16x16 r=2",        (probe_tile_25pt_r2<<<g2d,b2d>>>(d_out,d_in,N,N)))
    RUN("3D 8x8x4 tile 7pt 64^3",       (probe_tile_3d_8x8x4<<<g3d,256>>>(d_out,d_in,64,64,64)))
    RUN("RegTile 1x1 baseline",          (probe_regtile_1x1<<<(n_elem+127)/128,128>>>(d_out,d_in,n_elem)))
    RUN("RegTile 2x1 float2",            (probe_regtile_2x1<<<(n_elem/2+127)/128,128>>>(d_out,d_in,n_elem)))
    RUN("RegTile 4x1 float4",            (probe_regtile_4x1<<<(n_elem/4+127)/128,128>>>(d_out,d_in,n_elem)))
    dim3 gb2(16,16), gg2((128+15)/16,(128+15)/16);
    RUN("RegTile 2x2 GEMM 256^3",       (probe_regtile_2x2<<<gg2,gb2>>>(d_out,d_A,d_B,256,256,256)))
    dim3 gb4(8,8), gg4((64+7)/8,(64+7)/8);
    RUN("RegTile 4x4 GEMM 256^3",       (probe_regtile_4x4<<<gg4,gb4>>>(d_out,d_A,d_B,256,256,256)))
    RUN("LBM 2-cell coarsened 1M",      (probe_regtile_lbm_2cell<<<(n_elem/2+127)/128,128>>>(d_out,d_in,n_elem)))
    dim3 sg((N+31)/32,(N+31)/32);
    RUN("SMEM row-major (conflicts)",    (probe_smem_row_major<<<sg,256>>>(d_out,d_in,N)))
    RUN("SMEM padded +1 (no conflicts)", (probe_smem_padded<<<sg,256>>>(d_out,d_in,N)))
    RUN("SMEM XOR-swizzled",            (probe_smem_xor_swizzle<<<sg,256>>>(d_out,d_in,N)))
    RUN("SMEM transpose padded",        (probe_smem_transpose<<<sg,256>>>(d_out,d_in,N,N)))

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_A); cudaFree(d_B);
    return 0;
}
