/*
 * SASS RE Probe: tensor-core work under warp-uniform predicates
 *
 * cuDNN library mining surfaced UPLOP3.LUT near HMMA blocks. This probe
 * creates warp-uniform control around tensor-core work so ptxas can materialize
 * uniform predicates and, ideally, uniform predicate logic instead of ordinary
 * predicate logic only.
 */

#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void __launch_bounds__(32)
probe_tensor_uniform_gate(half *d_out,
                          const half *d_a,
                          const half *d_b,
                          int mode,
                          int extra) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_a, 16);
    wmma::load_matrix_sync(fB, d_b, 16);
    wmma::fill_fragment(fC, 0.0f);

    int run_main = ((mode & 1) != 0);
    int run_alt = ((mode & 2) != 0);
    int run_bias = ((extra & 4) != 0);

    #pragma unroll 1
    for (int i = 0; i < 16; ++i) {
        if (run_main && !run_alt) {
            wmma::mma_sync(fC, fA, fB, fC);
        } else if (!run_main && run_alt) {
            wmma::mma_sync(fC, fA, fB, fC);
            wmma::mma_sync(fC, fA, fB, fC);
        } else if (run_main && run_alt) {
            wmma::mma_sync(fC, fA, fB, fC);
            if (run_bias) {
                wmma::mma_sync(fC, fA, fB, fC);
            }
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC.num_elements; ++j) {
        fD.x[j] = __float2half(fC.x[j]);
    }
    wmma::store_matrix_sync(d_out, fD, 16, wmma::mem_row_major);
}

extern "C" __global__ void __launch_bounds__(32)
probe_tensor_uniform_deadguard(half *d_out,
                               const half *d_a,
                               const half *d_b,
                               int mode) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_a, 16);
    wmma::load_matrix_sync(fB, d_b, 16);
    wmma::fill_fragment(fC, 0.0f);

    int never = (mode == -1234567);
    int always = (mode != -1234567);

    if (always) {
        wmma::mma_sync(fC, fA, fB, fC);
    }
    if (never) {
        wmma::mma_sync(fC, fA, fB, fC);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int j = 0; j < fC.num_elements; ++j) {
        fD.x[j] = __float2half(fC.x[j]);
    }
    wmma::store_matrix_sync(d_out, fD, 16, wmma::mem_row_major);
}
