/*
 * SASS RE Probe: Comprehensive 4-bit Format Investigation
 * Isolates: FP4 E2M1, INT4 signed/unsigned, and investigates BF4/TF4
 *
 * 4-bit format landscape on Ada Lovelace SM 8.9:
 *
 *   FORMAT     | EXISTS ON ADA? | HARDWARE SUPPORT | NOTES
 *   -----------|----------------|------------------|------
 *   INT4 (s4)  | YES            | IMMA.8832.S4     | Tensor core MMA (8x8x32 shape)
 *   INT4 (u4)  | YES            | IMMA.8832.U4     | Unsigned variant
 *   FP4 E2M1   | EMULATED       | Nibble pack+LUT  | Blackwell SM10.0 native; Ada emulates
 *   BF4        | NO             | N/A              | Does not exist in any NVIDIA ISA
 *   TF4        | NO             | N/A              | Does not exist in any NVIDIA ISA
 *
 * INT4 is the only 4-bit format with native tensor core support on Ada.
 * FP4 E2M1 is defined as a Blackwell (SM 10.0) format and is emulated
 * on Ada via lookup table decode + manual nibble packing.
 *
 * BF4 (brain float 4-bit) and TF4 (tensor float 4-bit) do NOT exist
 * in any published NVIDIA ISA. There is no 4-bit floating point format
 * with a reduced-mantissa BF16-style layout. The E2M1 format (FP4) is
 * the only sub-byte float format defined by NVIDIA.
 *
 * This probe characterizes:
 *   1. INT4 tensor core throughput (IMMA.8832.S4)
 *   2. FP4 E2M1 emulated decode throughput (LUT lookup)
 *   3. Nibble pack/unpack instruction sequences and cycle counts
 *   4. Quantization error analysis for each format
 */

#include <mma.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace nvcuda;

/* ============================================================
 * FP4 E2M1 Format Reference
 *
 * 4 bits: [sign:1][exponent:2][mantissa:1]
 *
 * Encoding table (unsigned nibble -> float value):
 *   0x0: +0.0    0x8: -0.0
 *   0x1: +0.5    0x9: -0.5
 *   0x2: +1.0    0xA: -1.0
 *   0x3: +1.5    0xB: -1.5
 *   0x4: +2.0    0xC: -2.0
 *   0x5: +3.0    0xD: -3.0
 *   0x6: +4.0    0xE: -4.0
 *   0x7: +6.0    0xF: -6.0
 *
 * Dynamic range: [0, 6.0] (positive), [-6.0, 0] (negative)
 * Only 8 unique magnitudes. Massive quantization error.
 * ============================================================ */

__constant__ float FP4_E2M1_DECODE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* ============================================================
 * INT4 Format Reference
 *
 * 4 bits: signed two's complement [-8, 7]
 *   0x0: 0     0x8: -8
 *   0x1: 1     0x9: -7
 *   ...        ...
 *   0x7: 7     0xF: -1
 *
 * For LBM: DIST_SCALE=14, so range maps to f_i in [-0.571, 0.5]
 * Edge weights (1/36 * 14 = 0.39 -> 0): physics broken at INT4.
 * ============================================================ */

/* ── FP4 decode throughput (LUT-based, emulated on Ada) ─── */
extern "C" __global__ void __launch_bounds__(128)
probe_fp4_decode_throughput(float *out, const unsigned char *packed,
                            int n_bytes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_bytes) return;

    unsigned char byte = packed[i];

    // Lower nibble -> FP4 decode (1 LDS from constant memory)
    float lo = FP4_E2M1_DECODE[byte & 0xF];
    // Upper nibble -> FP4 decode
    float hi = FP4_E2M1_DECODE[(byte >> 4) & 0xF];

    out[i * 2 + 0] = lo;
    out[i * 2 + 1] = hi;
}

/* ── FP4 encode throughput (min-distance quantization) ──── */
__device__ __forceinline__
unsigned char float_to_fp4_e2m1(float val) {
    // Clamp to representable range
    float aval = fabsf(val);
    unsigned char nibble;

    // Find nearest FP4 value (binary search over 8 magnitudes)
    if (aval < 0.25f)       nibble = 0;  // 0.0
    else if (aval < 0.75f)  nibble = 1;  // 0.5
    else if (aval < 1.25f)  nibble = 2;  // 1.0
    else if (aval < 1.75f)  nibble = 3;  // 1.5
    else if (aval < 2.5f)   nibble = 4;  // 2.0
    else if (aval < 3.5f)   nibble = 5;  // 3.0
    else if (aval < 5.0f)   nibble = 6;  // 4.0
    else                     nibble = 7;  // 6.0

    // Set sign bit
    if (val < 0.0f) nibble |= 0x8;

    return nibble;
}

extern "C" __global__ void __launch_bounds__(128)
probe_fp4_encode_throughput(unsigned char *packed, const float *in,
                            int n_pairs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_pairs) return;

    float val_lo = in[i * 2 + 0];
    float val_hi = in[i * 2 + 1];

    unsigned char lo = float_to_fp4_e2m1(val_lo);
    unsigned char hi = float_to_fp4_e2m1(val_hi);

    packed[i] = (hi << 4) | lo;
}

/* ── INT4 decode throughput (shift+sign-extend) ─────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_int4_decode_throughput(float *out, const unsigned char *packed,
                             int n_bytes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_bytes) return;

    unsigned char byte = packed[i];

    // Lower nibble: extract + sign-extend
    int lo_raw = byte & 0xF;
    int lo_signed = (lo_raw >= 8) ? (lo_raw - 16) : lo_raw;

    // Upper nibble
    int hi_raw = (byte >> 4) & 0xF;
    int hi_signed = (hi_raw >= 8) ? (hi_raw - 16) : hi_raw;

    // Scale to float (DIST_SCALE = 14 for D3Q19 INT4)
    const float INV_SCALE = 1.0f / 14.0f;
    out[i * 2 + 0] = (float)lo_signed * INV_SCALE;
    out[i * 2 + 1] = (float)hi_signed * INV_SCALE;
}

/* ── INT4 encode throughput (saturating quantize) ───────── */
extern "C" __global__ void __launch_bounds__(128)
probe_int4_encode_throughput(unsigned char *packed, const float *in,
                             int n_pairs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_pairs) return;

    const float DIST_SCALE = 14.0f;

    float val_lo = in[i * 2 + 0];
    float val_hi = in[i * 2 + 1];

    // Quantize + clamp to [-8, 7]
    int q_lo = (int)roundf(val_lo * DIST_SCALE);
    int q_hi = (int)roundf(val_hi * DIST_SCALE);
    q_lo = max(-8, min(7, q_lo));
    q_hi = max(-8, min(7, q_hi));

    // Pack two nibbles
    packed[i] = (unsigned char)(((q_hi & 0xF) << 4) | (q_lo & 0xF));
}

/* ── INT4 tensor core (IMMA.8832.S4) ───────────────────── */
__global__ void __launch_bounds__(32)
probe_int4_tensor_core(int *d_D, const void *d_A, const void *d_B,
                       const int *d_C) {
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> frag_B;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> frag_C;

    wmma::load_matrix_sync(frag_A, d_A, 32);
    wmma::load_matrix_sync(frag_B, d_B, 32);
    wmma::fill_fragment(frag_C, 0);

    // IMMA.8832.S4.S4: 8x8x32 INT4 tensor core multiply
    wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

    wmma::store_matrix_sync(d_D, frag_C, 8, wmma::mem_row_major);
}

/* ── D3Q19 full collision with INT4 (bandwidth ceiling test) ── */
extern "C" __global__ void __launch_bounds__(128)
probe_int4_d3q19_collision(unsigned char *dist, float *rho_out,
                           int n_cells) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cell_a = tid * 2;
    int cell_b = tid * 2 + 1;
    if (cell_b >= n_cells) return;

    const float INV_SCALE = 1.0f / 14.0f;
    const float DIST_SCALE = 14.0f;

    // Decode 19 nibble pairs -> 19 float pairs
    float fa[19], fb[19];
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        int byte_idx = d * (n_cells / 2) + tid;
        unsigned char byte = dist[byte_idx];

        int raw_a = byte & 0xF;
        int raw_b = (byte >> 4) & 0xF;
        int sa = (raw_a >= 8) ? (raw_a - 16) : raw_a;
        int sb = (raw_b >= 8) ? (raw_b - 16) : raw_b;

        fa[d] = (float)sa * INV_SCALE;
        fb[d] = (float)sb * INV_SCALE;
    }

    // Compute density (sum of distributions)
    float rho_a = 0.0f, rho_b = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        rho_a += fa[d];
        rho_b += fb[d];
    }

    // Simplified BGK collision
    float inv_tau = 1.0f / 0.6f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float feq_a = rho_a / 19.0f;
        float feq_b = rho_b / 19.0f;
        fa[d] -= (fa[d] - feq_a) * inv_tau;
        fb[d] -= (fb[d] - feq_b) * inv_tau;
    }

    // Re-encode to INT4
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        int qa = (int)roundf(fa[d] * DIST_SCALE);
        int qb = (int)roundf(fb[d] * DIST_SCALE);
        qa = max(-8, min(7, qa));
        qb = max(-8, min(7, qb));

        int byte_idx = d * (n_cells / 2) + tid;
        dist[byte_idx] = (unsigned char)(((qb & 0xF) << 4) | (qa & 0xF));
    }

    rho_out[cell_a] = rho_a;
    rho_out[cell_b] = rho_b;
}

/* ── Quantization error analysis ────────────────────────── */
extern "C" __global__ void __launch_bounds__(32)
probe_4bit_quantization_error(float *errors, int format) {
    // Test values spanning the distribution range
    const float test_vals[32] = {
        0.0f, 0.01f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f,
        0.333f, 0.35f, 0.4f, 0.45f, 0.5f, 0.6f, 0.7f, 0.8f,
        -0.01f, -0.05f, -0.1f, -0.15f, -0.2f, -0.25f, -0.3f, -0.333f,
        1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -6.0f
    };

    int i = threadIdx.x;
    float val = test_vals[i];
    float reconstructed;

    if (format == 0) {
        // INT4: quantize then reconstruct
        int q = (int)roundf(val * 14.0f);
        q = max(-8, min(7, q));
        reconstructed = (float)q / 14.0f;
    } else {
        // FP4 E2M1: quantize then reconstruct
        unsigned char nibble = float_to_fp4_e2m1(val);
        reconstructed = FP4_E2M1_DECODE[nibble];
    }

    errors[i] = fabsf(val - reconstructed);
}
