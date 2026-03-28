/*
 * SASS RE Probe: FP16 Half2 Native Arithmetic
 * Isolates: HADD2, HMUL2, HFMA2, H2F, F2H (half2 packed vector ops)
 *
 * Ada Lovelace SM 8.9 executes half2 ops at 2x the scalar FP16 rate:
 *   HADD2/HMUL2: 2 FP16 ops per instruction, same throughput as FADD
 *   HFMA2: 2 FP16 FMAs per instruction
 *
 * These are the core ops in kernels_fp16_soa_half2.cu where 2 cells/thread
 * use __half2 accumulators for velocity moment computation (+9.8% ILP).
 *
 * Key SASS instructions to look for:
 *   HADD2  -- packed half2 add
 *   HMUL2  -- packed half2 multiply
 *   HFMA2  -- packed half2 fused multiply-add
 *   F2FP   -- float to half2 conversion
 *   I2FP   -- integer to half2 conversion
 */

#include <cuda_fp16.h>

// Half2 arithmetic: add, mul, fma in isolation
extern "C" __global__ void __launch_bounds__(32)
probe_half2_arith(__half2 *out, const __half2 *a, const __half2 *b) {
    int i = threadIdx.x;

    __half2 va = a[i];
    __half2 vb = b[i];

    // HADD2: packed half2 addition
    __half2 sum = __hadd2(va, vb);

    // HMUL2: packed half2 multiply
    __half2 prod = __hmul2(va, vb);

    // HFMA2: packed half2 fused multiply-add
    __half2 fma_result = __hfma2(va, vb, sum);

    // HSUB2: packed half2 subtraction
    __half2 diff = __hsub2(va, vb);

    // Chain to prevent dead-code elimination
    out[i] = __hadd2(__hadd2(sum, prod), __hadd2(fma_result, diff));
}

// Half2 dependent chain for latency measurement
extern "C" __global__ void __launch_bounds__(32)
probe_half2_chain(__half2 *out, const __half2 *a) {
    int i = threadIdx.x;
    __half2 acc = a[i];
    __half2 one = __float2half2_rn(1.0f);

    // 64-deep dependent HADD2 chain
    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc = __hadd2(acc, one);
    }
    out[i] = acc;
}

// Half2 FMA dependent chain
extern "C" __global__ void __launch_bounds__(32)
probe_half2_fma_chain(__half2 *out, const __half2 *a) {
    int i = threadIdx.x;
    __half2 acc = a[i];
    __half2 scale = __float2half2_rn(0.99f);
    __half2 bias = __float2half2_rn(0.01f);

    // 64-deep dependent HFMA2 chain
    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc = __hfma2(scale, acc, bias);
    }
    out[i] = acc;
}

// Conversions: float <-> half2
extern "C" __global__ void __launch_bounds__(32)
probe_half2_conversions(float *fout, __half2 *hout,
                        const float *fin, const __half2 *hin) {
    int i = threadIdx.x;

    // F2H: float -> half2 (pack two floats into one half2)
    __half2 packed = __floats2half2_rn(fin[i * 2], fin[i * 2 + 1]);

    // H2F: half2 -> two floats (unpack)
    float lo = __low2float(hin[i]);
    float hi = __high2float(hin[i]);

    // float -> half2 broadcast (same value in both lanes)
    __half2 broadcast = __float2half2_rn(fin[i]);

    hout[i] = __hadd2(packed, broadcast);
    fout[i] = lo + hi;
}

// Half2 comparison and selection (HSETP2, HMNMX2)
extern "C" __global__ void __launch_bounds__(32)
probe_half2_compare(__half2 *out, const __half2 *a, const __half2 *b) {
    int i = threadIdx.x;
    __half2 va = a[i];
    __half2 vb = b[i];

    // Max of two half2 vectors (element-wise)
    __half2 mx = __hmax2(va, vb);

    // Min of two half2 vectors
    __half2 mn = __hmin2(va, vb);

    // Absolute value
    __half2 ab = __habs2(va);

    // Negate
    __half2 neg = __hneg2(va);

    out[i] = __hadd2(__hadd2(mx, mn), __hadd2(ab, neg));
}
