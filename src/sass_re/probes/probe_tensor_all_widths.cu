/*
 * SASS RE Probe: Tensor Core at Every Possible Width
 * Tests which tensor formats exist at each precision width on Ada SM 8.9.
 *
 * TENSOR CORE FORMAT MATRIX (Ada Lovelace SM 8.9):
 *
 * Width   Format      Shape        SASS instruction           Exists?
 * -----   ------      -----        ----------------           -------
 * 4-bit   INT4 (S4)   8x8x32       IMMA.8832.S4.S4           YES [P]
 * 4-bit   UINT4 (U4)  8x8x32       IMMA.8832.U4.U4           YES [P]
 * 4-bit   FP4 E2M1    --           --                         NO (Blackwell only)
 * 8-bit   INT8 (S8)   16x16x16     IMMA.16816.S8.S8          YES [P]
 * 8-bit   FP8 E4M3    --           --                         NO TC (CVT only)
 * 8-bit   FP8 E5M2    --           --                         NO TC (CVT only)
 * 16-bit  FP16        16x16x16     HMMA.16816.F32 / .F16     YES [P]
 * 16-bit  BF16        16x16x16     HMMA.16816.F32.BF16       YES [P]
 * 16-bit  TF16        --           --                         DOES NOT EXIST
 * 19-bit  TF32        16x16x8(4)   HMMA.1684.F32.TF32        YES [P]
 * 32-bit  FP32        --           --                         NO TC on Ada
 * 64-bit  FP64        --           --                         NO TC on Ada gaming
 *
 * NOTE: Ada gaming SKUs have NO FP64 tensor cores.
 * Data center GPUs (A100, H100) have DMMA for FP64 tensor ops.
 *
 * x2 and x4 tensor throughput:
 *   x2 = 2 independent HMMA chains -> tests ILP within TC pipeline
 *   x4 = 4 independent HMMA chains -> tests maximum TC saturation
 *
 * This probe includes all confirmed formats plus throughput chains.
 */

#include <mma.h>
using namespace nvcuda;

/* ── FP16 TC: x1, x2, x4 independent chains ──────── */

__global__ void __launch_bounds__(32)
probe_tc_fp16_x1(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);

    // x1: single chain of 16 HMMA
    for (int i = 0; i < 16; i++)
        wmma::mma_sync(fC, fA, fB, fC);

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int i = 0; i < fC.num_elements; i++)
        fD.x[i] = __float2half(fC.x[i]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

__global__ void __launch_bounds__(32)
probe_tc_fp16_x2(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0, fC1;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f);
    wmma::fill_fragment(fC1, 0.0f);

    // x2: two independent HMMA chains interleaved
    for (int i = 0; i < 16; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
    }

    // Store sum to prevent DCE
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int i = 0; i < fC0.num_elements; i++)
        fD.x[i] = __float2half(fC0.x[i] + fC1.x[i]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

__global__ void __launch_bounds__(32)
probe_tc_fp16_x4(half *d_D, const half *d_A, const half *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC0, fC1, fC2, fC3;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f);
    wmma::fill_fragment(fC1, 0.0f);
    wmma::fill_fragment(fC2, 0.0f);
    wmma::fill_fragment(fC3, 0.0f);

    // x4: four independent HMMA chains
    for (int i = 0; i < 16; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
        wmma::mma_sync(fC2, fA, fB, fC2);
        wmma::mma_sync(fC3, fA, fB, fC3);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fD;
    for (int i = 0; i < fC0.num_elements; i++)
        fD.x[i] = __float2half(fC0.x[i] + fC1.x[i] + fC2.x[i] + fC3.x[i]);
    wmma::store_matrix_sync(d_D, fD, 16, wmma::mem_row_major);
}

/* ── TF32 TC: x1, x2, x4 ────────────────────────── */

__global__ void __launch_bounds__(32)
probe_tc_tf32_x1(float *d_D, const float *d_A, const float *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);

    for (int i = 0; i < 16; i++)
        wmma::mma_sync(fC, fA, fB, fC);

    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
}

__global__ void __launch_bounds__(32)
probe_tc_tf32_x4(float *d_D, const float *d_A, const float *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> fC0, fC1, fC2, fC3;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0.0f);
    wmma::fill_fragment(fC1, 0.0f);
    wmma::fill_fragment(fC2, 0.0f);
    wmma::fill_fragment(fC3, 0.0f);

    for (int i = 0; i < 16; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
        wmma::mma_sync(fC2, fA, fB, fC2);
        wmma::mma_sync(fC3, fA, fB, fC3);
    }

    // Sum to prevent DCE
    for (int i = 0; i < fC0.num_elements; i++)
        fC0.x[i] += fC1.x[i] + fC2.x[i] + fC3.x[i];
    wmma::store_matrix_sync(d_D, fC0, 16, wmma::mem_row_major);
}

/* ── INT8 TC: x1, x4 ────────────────────────────── */

__global__ void __launch_bounds__(32)
probe_tc_int8_x1(int *d_D, const signed char *d_A,
                  const signed char *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0);

    for (int i = 0; i < 16; i++)
        wmma::mma_sync(fC, fA, fB, fC);

    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
}

__global__ void __launch_bounds__(32)
probe_tc_int8_x4(int *d_D, const signed char *d_A,
                  const signed char *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fC0, fC1, fC2, fC3;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC0, 0);
    wmma::fill_fragment(fC1, 0);
    wmma::fill_fragment(fC2, 0);
    wmma::fill_fragment(fC3, 0);

    for (int i = 0; i < 16; i++) {
        wmma::mma_sync(fC0, fA, fB, fC0);
        wmma::mma_sync(fC1, fA, fB, fC1);
        wmma::mma_sync(fC2, fA, fB, fC2);
        wmma::mma_sync(fC3, fA, fB, fC3);
    }

    for (int i = 0; i < fC0.num_elements; i++)
        fC0.x[i] += fC1.x[i] + fC2.x[i] + fC3.x[i];
    wmma::store_matrix_sync(d_D, fC0, 16, wmma::mem_row_major);
}

/* ── BF16 TC: x1, x4 ────────────────────────────── */

__global__ void __launch_bounds__(32)
probe_tc_bf16_x1(float *d_D, const __nv_bfloat16 *d_A,
                  const __nv_bfloat16 *d_B) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::load_matrix_sync(fA, d_A, 16);
    wmma::load_matrix_sync(fB, d_B, 16);
    wmma::fill_fragment(fC, 0.0f);

    for (int i = 0; i < 16; i++)
        wmma::mma_sync(fC, fA, fB, fC);

    wmma::store_matrix_sync(d_D, fC, 16, wmma::mem_row_major);
}

/*
 * TENSOR CORE FORMATS THAT DO NOT EXIST ON ADA:
 *
 * TF16:  No such format. FP16 and BF16 are the only 16-bit TC formats.
 *        "TF" only exists at 32-bit width (TF32 = 19-bit truncated FP32).
 *
 * TF64:  No tensor core FP64 on Ada gaming SKUs.
 *        Data center GPUs (A100: DMMA, H100: DMMA) support FP64 TC.
 *
 * TF128: Does not exist. Double-double is NOT a tensor core format.
 *
 * TF256: Does not exist and is not practical.
 *
 * FP8 TC (E4M3/E5M2): NOT available as tensor core input on Ada.
 *        FP8 tensor cores exist only on Hopper (SM 9.0) and later.
 *        On Ada, FP8 is a storage format with F2FP conversion to FP16.
 *
 * FP4 TC (E2M1): NOT available as tensor core input on Ada.
 *        FP4 tensor cores exist only on Blackwell (SM 10.0).
 *        On Ada, only INT4 (S4/U4) has tensor core support.
 */
