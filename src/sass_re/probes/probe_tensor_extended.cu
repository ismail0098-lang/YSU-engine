/*
 * SASS RE Probe: Extended Tensor Core Coverage
 * Isolates: TF32 (HMMA.16168), BF16, INT8 (IMMA.16816), INT4 (HMMA.8832.S4)
 *
 * The existing probe_tensor.cu covers only FP16 HMMA via WMMA.
 * This probe extends coverage to all Ada Lovelace tensor core precisions.
 *
 * Ada SM 8.9 tensor core measured throughput (from kernels_tensor_core.cu):
 *   TF32  16x16x8:  22,880 GFLOPS
 *   FP16  16x16x16: 45,901 GFLOPS
 *   BF16  16x16x16: 45,954 GFLOPS
 *   INT8  16x16x16: 166,189 TOPS
 *   INT4  8x8x32:   189,103 TOPS
 *
 * Key SASS instructions:
 *   HMMA.16816.F32    -- FP16 MMA
 *   HMMA.16168.F32    -- TF32 MMA (different shape: 16x16x8)
 *   HMMA.16816.F32.BF -- BF16 MMA
 *   IMMA.16816        -- INT8 MMA -> INT32 accumulator
 *   IMMA.8832.S4      -- INT4 (signed 4-bit) MMA -> INT32 accumulator
 */

#include <mma.h>
using namespace nvcuda;

// TF32 tensor core: 16x16x8 (different shape from FP16's 16x16x16)
// TF32 uses 19-bit format: 1 sign + 8 exponent + 10 mantissa
// Same range as FP32 but reduced mantissa (10 vs 23 bits)
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_tf32(float *d_D, const float *d_A, const float *d_B, const float *d_C) {
    // TF32: M=16, N=16, K=8 (note K=8, not 16)
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 16);
    wmma::load_matrix_sync(frag_B, d_B, 16);
    wmma::load_matrix_sync(frag_C, d_C, 16, wmma::mem_row_major);

    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 16, wmma::mem_row_major);
}

// BF16 tensor core: 16x16x16 (same shape as FP16)
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_bf16(float *d_D, const __nv_bfloat16 *d_A,
                const __nv_bfloat16 *d_B, const float *d_C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 16);
    wmma::load_matrix_sync(frag_B, d_B, 16);
    wmma::load_matrix_sync(frag_C, d_C, 16, wmma::mem_row_major);

    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 16, wmma::mem_row_major);
}

// INT8 tensor core (IMMA): 16x16x16 -> INT32 accumulator
extern "C" __global__ void __launch_bounds__(32)
probe_imma_int8(int *d_D, const signed char *d_A,
                const signed char *d_B, const int *d_C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 16);
    wmma::load_matrix_sync(frag_B, d_B, 16);
    wmma::load_matrix_sync(frag_C, d_C, 16, wmma::mem_row_major);

    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 16, wmma::mem_row_major);
}

// INT4 tensor core (S4 experimental precision): 8x8x32 -> INT32
// Uses experimental API: nvcuda::wmma::experimental::precision::s4
__global__ void __launch_bounds__(32)
probe_imma_int4(int *d_D, const void *d_A, const void *d_B, const int *d_C) {
    using namespace nvcuda::wmma::experimental;
    // INT4 uses sub-byte packing: each byte holds 2 INT4 values
    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 32);
    wmma::load_matrix_sync(frag_B, d_B, 32);
    wmma::load_matrix_sync(frag_C, d_C, 8, wmma::mem_row_major);

    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 8, wmma::mem_row_major);
}

// TF32 chained: multiple MMA ops to measure throughput
extern "C" __global__ void __launch_bounds__(32)
probe_hmma_tf32_chain(float *d_D, const float *d_A,
                      const float *d_B, const float *d_C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 16);
    wmma::load_matrix_sync(frag_B, d_B, 16);
    wmma::fill_fragment(frag_C, 0.0f);

    // 8 back-to-back TF32 MMA ops
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 16, wmma::mem_row_major);
}
