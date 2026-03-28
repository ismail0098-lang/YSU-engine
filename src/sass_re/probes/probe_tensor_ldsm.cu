/*
 * SASS RE Probe: LDSM (Load Shared Memory Matrix) for Tensor Cores
 * Isolates: LDSM instruction and HMMA with FP16 accumulator (not FP32)
 *
 * LDSM is a specialized instruction that loads shared memory data directly
 * into the register layout expected by HMMA (tensor core MMA). It replaces
 * the manual load+swizzle pattern that wmma::load_matrix_sync generates.
 *
 * On Ada Lovelace, LDSM loads a 16x16 matrix tile from shared memory
 * using a single instruction per thread, with automatic lane-to-bank
 * mapping that avoids bank conflicts.
 *
 * Also probes HMMA with FP16 accumulator (instead of FP32) to measure
 * if there's a throughput difference.
 *
 * Key SASS:
 *   LDSM.16.M88.2  -- load shared memory matrix (2 registers per thread)
 *   LDSM.16.M88.4  -- load shared memory matrix (4 registers per thread)
 *   LDSM.16.MT88.2 -- transposed variant
 *   HMMA.16816.F16  -- tensor core MMA with FP16 accumulator
 */

#include <mma.h>
using namespace nvcuda;

// Standard WMMA with FP16 accumulator (not FP32)
// Generates HMMA.16816.F16 instead of HMMA.16816.F32
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_fp16_accum(half *d_D, const half *d_A, const half *d_B, const half *d_C) {
    // FP16 accumulator (output is half, not float)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_C;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_D;

    wmma::load_matrix_sync(frag_A, d_A, 16);
    wmma::load_matrix_sync(frag_B, d_B, 16);
    wmma::load_matrix_sync(frag_C, d_C, 16, wmma::mem_row_major);

    wmma::mma_sync(frag_D, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_D, 16, wmma::mem_row_major);
}

// WMMA with shared memory staging (triggers LDSM)
// Load A and B from shared memory instead of global memory
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_via_smem(half *d_D, const half *d_A, const half *d_B) {
    __shared__ half smem_A[16 * 16];
    __shared__ half smem_B[16 * 16];

    int tid = threadIdx.x;

    // Cooperative load: each of 32 threads loads 8 elements
    for (int j = tid; j < 256; j += 32) {
        smem_A[j] = d_A[j];
        smem_B[j] = d_B[j];
    }
    __syncthreads();

    // Load from shared memory -> should generate LDSM
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;

    wmma::load_matrix_sync(frag_A, smem_A, 16);  // LDSM from shared
    wmma::load_matrix_sync(frag_B, smem_B, 16);  // LDSM from shared
    wmma::fill_fragment(frag_C, 0.0f);

    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    // Store result
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_D;
    // Convert FP32 accum to FP16 for store
    for (int i = 0; i < frag_C.num_elements; i++)
        frag_D.x[i] = __float2half(frag_C.x[i]);
    wmma::store_matrix_sync(d_D, frag_D, 16, wmma::mem_row_major);
}

// Back-to-back HMMA with smem reload (simulates tiled GEMM inner loop)
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_tiled_loop(half *d_D, const half *d_A, const half *d_B,
                      int K_tiles) {
    __shared__ half smem_A[16 * 16];
    __shared__ half smem_B[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;
    wmma::fill_fragment(frag_C, 0.0f);

    int tid = threadIdx.x;

    for (int k = 0; k < K_tiles; k++) {
        // Load tile from global to shared
        for (int j = tid; j < 256; j += 32) {
            smem_A[j] = d_A[k * 256 + j];
            smem_B[j] = d_B[k * 256 + j];
        }
        __syncthreads();

        // LDSM + HMMA
        wmma::load_matrix_sync(frag_A, smem_A, 16);
        wmma::load_matrix_sync(frag_B, smem_B, 16);
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

        __syncthreads();
    }

    // Store accumulated result
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_D;
    for (int i = 0; i < frag_C.num_elements; i++)
        frag_D.x[i] = __float2half(frag_C.x[i]);
    wmma::store_matrix_sync(d_D, frag_D, 16, wmma::mem_row_major);
}
