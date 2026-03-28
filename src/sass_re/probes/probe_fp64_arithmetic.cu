/*
 * SASS RE Probe: FP64 Double-Precision Arithmetic
 * Isolates: DADD, DFMA, DMUL, DSETP, D2F, F2D (FP64 datapath)
 *
 * Ada Lovelace SM 8.9 gaming SKU has a 64:1 FP64:FP32 throughput ratio:
 *   FP32: 128 ops/clock/SM = ~40 TFLOPS
 *   FP64: 2 ops/clock/SM = ~0.6 TFLOPS
 *
 * FP64 kernels are therefore compute-bound, not memory-bound (even at 128^3).
 * From LBM results: FP64 AoS = 461 MLUPS, FP64 SoA = 406 MLUPS.
 * SoA is SLOWER because improved memory coalescing doesn't help when the
 * bottleneck is the FP64 ALU, not memory bandwidth.
 *
 * Key SASS instructions:
 *   DADD    -- double-precision add
 *   DFMA    -- double-precision fused multiply-add
 *   DMUL    -- double-precision multiply
 *   DSETP   -- double-precision set predicate (comparison)
 *   MUFU.RCP64H -- double-precision reciprocal (SFU, high word)
 *   F2F.F64.F32 -- FP32 -> FP64 widening conversion
 *   F2F.F32.F64 -- FP64 -> FP32 narrowing conversion
 */

// FP64 arithmetic: DADD, DMUL, DFMA in isolation
extern "C" __global__ void __launch_bounds__(32)
probe_fp64_arith(double *out, const double *a, const double *b) {
    int i = threadIdx.x;
    double va = a[i];
    double vb = b[i];

    // DADD
    double sum = va + vb;

    // DMUL
    double prod = va * vb;

    // DFMA
    double fma_result = fma(va, vb, sum);

    // DSETP (comparison)
    double sel = (va > vb) ? sum : prod;

    out[i] = fma_result + sel;
}

// FP64 dependent chain for latency measurement
extern "C" __global__ void __launch_bounds__(32)
probe_fp64_dfma_chain(double *out, const double *a) {
    int i = threadIdx.x;
    double acc = a[i];
    double scale = 0.999;
    double bias = 0.001;

    // 64-deep dependent DFMA chain
    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc = fma(scale, acc, bias);
    }
    out[i] = acc;
}

// FP64 throughput: 8 independent accumulator streams
extern "C" __global__ void __launch_bounds__(128)
probe_fp64_throughput(double *out, const double *a, const double *b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    double va = a[i];
    double vb = b[i];

    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    double a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;

    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        a0 = fma(va, vb, a0);
        a1 = fma(va, vb, a1);
        a2 = fma(va, vb, a2);
        a3 = fma(va, vb, a3);
        a4 = fma(va, vb, a4);
        a5 = fma(va, vb, a5);
        a6 = fma(va, vb, a6);
        a7 = fma(va, vb, a7);
    }

    out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// FP64 <-> FP32 conversions (widening and narrowing)
extern "C" __global__ void __launch_bounds__(32)
probe_fp64_conversions(double *dout, float *fout,
                       const double *din, const float *fin) {
    int i = threadIdx.x;

    // F2F.F64.F32: FP32 -> FP64 widening
    double widened = (double)fin[i];

    // F2F.F32.F64: FP64 -> FP32 narrowing
    float narrowed = (float)din[i];

    // Double-precision reciprocal (MUFU.RCP64H)
    double recip = 1.0 / din[i];

    dout[i] = widened + recip;
    fout[i] = narrowed;
}

// Mixed-precision: FP64 accumulation with FP32 inputs (DD-lite pattern)
// This is the pattern used in validation kernels where inputs are FP32
// but accumulation needs higher precision.
extern "C" __global__ void __launch_bounds__(32)
probe_fp64_mixed_accumulate(double *out, const float *in, int n_elements) {
    int i = threadIdx.x;
    double acc = 0.0;

    // Widen FP32 inputs to FP64, accumulate in double precision
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float f32_val = in[d * 32 + i];
        acc += (double)f32_val;  // F2F.F64.F32 + DADD
    }

    out[i] = acc;
}
