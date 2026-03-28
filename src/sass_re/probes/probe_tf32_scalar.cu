/*
 * SASS RE Probe: TF32 (TensorFloat-32) Scalar and Non-Tensor Paths
 * Isolates: Whether TF32 exists outside tensor cores
 *
 * TF32 format: 1 sign + 8 exponent + 10 mantissa = 19 bits
 *   - Same range as FP32 (8-bit exponent)
 *   - Reduced precision (10-bit mantissa vs FP32's 23-bit)
 *   - Stored in 32-bit container (upper 19 bits used, lower 13 zeroed)
 *
 * On Ada Lovelace:
 *   - TF32 is ONLY available via tensor cores (HMMA.1684.F32.TF32)
 *   - There are NO scalar TF32 ALU instructions (no FADD.TF32, etc.)
 *   - There is NO TF32 storage format (no LDG.TF32)
 *   - TF32 "truncation" is implicit: tensor cores read FP32 values and
 *     internally truncate to 19-bit precision before the MMA operation
 *
 * This probe verifies these claims by:
 *   1. Attempting TF32-precision scalar math (should compile to FP32 + truncate)
 *   2. Measuring the truncation path cost
 *   3. Comparing FP32 vs TF32-truncated accuracy
 *
 * Key finding to verify: TF32 is NOT a first-class type on Ada.
 * It exists ONLY as a tensor core internal precision mode.
 *
 * BF32 also does NOT exist -- BF16 is the only "brain float" format.
 * There is no 32-bit brain float (that would just be FP32 with
 * the same exponent, making it identical to FP32).
 *
 * TF16 does NOT exist. The only 16-bit tensor format is FP16/BF16.
 * TF64 does NOT exist. There are no reduced-mantissa FP64 tensor formats.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Manually truncate FP32 to TF32 precision (zero lower 13 mantissa bits)
__device__ __forceinline__
float truncate_to_tf32(float val) {
    unsigned int bits = __float_as_uint(val);
    bits &= 0xFFFFE000u;  // Zero lower 13 bits of mantissa
    return __uint_as_float(bits);
}

// FP32 scalar compute at TF32 precision (manual truncation)
// This is the FASTEST software TF32 path: truncate then FP32 FMA
extern "C" __global__ void __launch_bounds__(128)
probe_tf32_scalar_emulated(float *out, const float *a, const float *b,
                           const float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // Truncate inputs to TF32 precision
    float ta = truncate_to_tf32(a[i]);
    float tb = truncate_to_tf32(b[i]);

    // Standard FP32 FMA on truncated values
    // This matches what tensor cores do internally
    float result = fmaf(ta, tb, c[i]);

    out[i] = result;
}

// Compare FP32 vs TF32 accuracy
extern "C" __global__ void __launch_bounds__(32)
probe_tf32_accuracy(float *out_fp32, float *out_tf32, float *out_error,
                    const float *a, const float *b, const float *c) {
    int i = threadIdx.x;

    float va = a[i], vb = b[i], vc = c[i];

    // Full FP32 FMA
    float fp32_result = fmaf(va, vb, vc);

    // TF32-emulated FMA (truncate inputs)
    float ta = truncate_to_tf32(va);
    float tb = truncate_to_tf32(vb);
    float tf32_result = fmaf(ta, tb, vc);

    out_fp32[i] = fp32_result;
    out_tf32[i] = tf32_result;
    out_error[i] = fabsf(fp32_result - tf32_result);
}

// TF32 truncation cost: chain of truncate+FMA vs pure FMA
extern "C" __global__ void __launch_bounds__(32)
probe_tf32_truncation_latency(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;

    // Pure FP32 FMA chain (baseline)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 512; i++)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 512; }
}

extern "C" __global__ void __launch_bounds__(32)
probe_tf32_truncated_fma_latency(volatile float *vals, volatile long long *out) {
    float x = vals[0], y = vals[1], z = vals[2];
    long long t0, t1;

    // Truncate+FMA chain (TF32 emulation overhead)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 512; i++) {
        // Truncate x to TF32 precision (AND with mask)
        unsigned int bits;
        asm volatile("mov.b32 %0, %1;" : "=r"(bits) : "f"(x));
        bits &= 0xFFFFE000u;
        asm volatile("mov.b32 %0, %1;" : "=f"(x) : "r"(bits));
        // Then FMA
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 512; }
}

/*
 * NON-EXISTENT FORMATS DOCUMENTATION:
 *
 * TF16: Does not exist. There is no reduced-mantissa FP16 tensor format.
 *       FP16 (1+5+10) and BF16 (1+8+7) cover the 16-bit space.
 *
 * TF64: Does not exist. There is no tensor-optimized FP64 format.
 *       FP64 tensor cores don't exist on Ada gaming SKUs.
 *       On A100/H100 data center GPUs, FP64 tensor cores use full IEEE FP64.
 *
 * BF32: Does not exist. BF16's purpose is to match FP32's dynamic range
 *       in half the bits. A 32-bit brain float would be identical to FP32
 *       (same 8-bit exponent, same 23-bit mantissa = IEEE FP32).
 *
 * BF4:  Does not exist. The minimum BF format is BF16.
 *       A 4-bit BF would have 1 sign + 3 exponent + 0 mantissa = only 8 values.
 *       This is strictly worse than FP4 E2M1 (which has 16 values).
 *
 * BF8:  Does not exist as a named format on Ada.
 *       FP8 E5M2 (1+5+2) is the closest: 5-bit exponent matches BF16's
 *       range scaling philosophy (maximize range, minimize mantissa).
 *       Some frameworks call E5M2 "BF8" but NVIDIA officially names it FP8 E5M2.
 *
 * TF8:  Does not exist. No tensor-optimized 8-bit float format.
 *       FP8 E4M3 and E5M2 are the only 8-bit float formats on Ada.
 *
 * INT128/UINT128: No native hardware. Emulated via 2x64-bit (IADD3 + IADD3.X
 *       carry chain). See probe_int128_uint128_emulated.cu.
 *
 * INT256/FP256: No native hardware. Would require 4x64-bit emulation.
 *       No practical use case in GPU computing at this width.
 */
