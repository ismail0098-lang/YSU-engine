/*
 * SASS RE Probe: BF16 (Brain Floating Point) Arithmetic
 * Isolates: BF16 conversions, __nv_bfloat16 scalar and bfloat162 vector ops
 *
 * BF16 has same dynamic range as FP32 (8-bit exponent) but only 7-bit
 * mantissa (~0.8% relative error). On Ada Lovelace:
 *   - BF16 scalar load latency appears higher than FP16 scalar load
 *     (kernels_bf16_soa.cu measured 7.5% below FP16 SoA despite same element size)
 *   - BF16 tensor core throughput matches FP16 (45.9 TFLOPS)
 *   - Native bfloat162 arithmetic available on SM 8.0+
 *
 * Key SASS instructions to look for:
 *   F2FP.BF16  -- float32 to bfloat16 conversion
 *   FP2F.BF16  -- bfloat16 to float32 conversion
 *   HADD2.BF16 -- bfloat162 packed addition (if supported)
 *   HMUL2.BF16 -- bfloat162 packed multiply
 *   HFMA2.BF16 -- bfloat162 packed fused multiply-add
 */

#include <cuda_bf16.h>

// BF16 scalar conversions: FP32 -> BF16 -> FP32 round-trip
extern "C" __global__ void __launch_bounds__(32)
probe_bf16_convert(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // FP32 -> BF16 (truncation of mantissa from 23 to 7 bits)
    __nv_bfloat16 bf = __float2bfloat16(val);

    // BF16 -> FP32 (zero-extend mantissa)
    float result = __bfloat162float(bf);

    out[i] = result;
}

// BF16 scalar arithmetic chain (measures if BF16 FMA exists on scalar path)
extern "C" __global__ void __launch_bounds__(32)
probe_bf16_scalar_chain(float *out, const float *in) {
    int i = threadIdx.x;
    __nv_bfloat16 acc = __float2bfloat16(in[i]);
    __nv_bfloat16 one = __float2bfloat16(1.0f);
    __nv_bfloat16 scale = __float2bfloat16(0.99f);

    // 32-deep dependent BF16 scalar FMA chain
    // On Ada, BF16 scalar may decompose to: BF16->FP32 + FFMA + FP32->BF16
    #pragma unroll 1
    for (int j = 0; j < 32; j++) {
        acc = __hfma(scale, acc, one);
    }
    out[i] = __bfloat162float(acc);
}

// BF16x2 packed vector arithmetic (bfloat162)
extern "C" __global__ void __launch_bounds__(32)
probe_bf16x2_arith(__nv_bfloat162 *out, const __nv_bfloat162 *a,
                   const __nv_bfloat162 *b) {
    int i = threadIdx.x;
    __nv_bfloat162 va = a[i];
    __nv_bfloat162 vb = b[i];

    // Packed BF16x2 add
    __nv_bfloat162 sum = __hadd2(va, vb);

    // Packed BF16x2 multiply
    __nv_bfloat162 prod = __hmul2(va, vb);

    // Packed BF16x2 FMA
    __nv_bfloat162 fma_result = __hfma2(va, vb, sum);

    out[i] = __hadd2(sum, __hadd2(prod, fma_result));
}

// BF16x2 dependent chain for latency measurement
extern "C" __global__ void __launch_bounds__(32)
probe_bf16x2_chain(__nv_bfloat162 *out, const __nv_bfloat162 *a) {
    int i = threadIdx.x;
    __nv_bfloat162 acc = a[i];
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);

    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc = __hadd2(acc, one);
    }
    out[i] = acc;
}

// 19-direction BF16 decode pattern (from kernels_bf16.cu LBM pattern)
// Measures the cost of 19 BF16->FP32 conversions per cell
extern "C" __global__ void __launch_bounds__(128)
probe_bf16_d3q19_decode(float *out, const __nv_bfloat16 *dist, int n_cells) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    float rho = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        __nv_bfloat16 bf = dist[d * n_cells + cell];
        rho += __bfloat162float(bf);
    }
    out[cell] = rho;
}
