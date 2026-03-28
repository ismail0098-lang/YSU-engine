/*
 * SASS RE Probe: Mixed-Precision Compute Patterns
 * Isolates: Storage-compute split (FP16/INT8/FP8 storage, FP32 arithmetic)
 *
 * The core pattern in all LBM precision-tier kernels: distributions are
 * stored in compressed format but promoted to FP32 before any arithmetic.
 * This probe isolates the decode -> compute -> encode pipeline to measure
 * the conversion overhead relative to the computation itself.
 *
 * Patterns probed:
 *   1. FP16 storage + FP32 compute (most common production pattern)
 *   2. INT8 storage + FP32 compute (Pareto-optimal LBM tier)
 *   3. FP8 E4M3 storage + FP32 compute (highest float throughput)
 *   4. BF16 storage + FP32 compute
 *   5. INT16 storage + FP32 compute (moderate-Re)
 *
 * Each pattern does: load N values in compressed format, promote to FP32,
 * do a reduction + simple collision, demote back to compressed, store.
 *
 * Key SASS patterns to observe:
 *   FP16:  LDG.E.U16 -> F2FP.F32.F16 -> FFMA chain -> F2FP.F16.F32 -> STG.E.U16
 *   INT8:  LDG.E.U8  -> I2F.F32.S8   -> FFMA chain -> F2I.S8.F32   -> STG.E.U8
 *   FP8:   LDG.E.U8  -> F2FP.F16.E4M3 -> HADD2.F32 -> F2FP.E4M3.F32 -> STG.E.U8
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// Pattern 1: FP16 storage, FP32 compute, FP16 store-back
extern "C" __global__ void __launch_bounds__(128)
probe_mixed_fp16_fp32(__half *dist, int n_cells) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    // Decode 19 FP16 -> FP32
    float f[19];
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        f[d] = __half2float(dist[d * n_cells + cell]);
    }

    // Compute: density + simplified BGK collision
    float rho = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) rho += f[d];
    float inv_rho = 1.0f / rho;
    float tau_inv = 1.0f / 0.6f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float feq = rho / 19.0f;  // simplified equilibrium
        f[d] -= (f[d] - feq) * tau_inv;
    }

    // Encode FP32 -> FP16 and store
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        dist[d * n_cells + cell] = __float2half(f[d]);
    }
}

// Pattern 2: INT8 storage, FP32 compute, INT8 store-back
extern "C" __global__ void __launch_bounds__(128)
probe_mixed_int8_fp32(signed char *dist, int n_cells) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    const float DIST_SCALE = 64.0f;
    const float INV_SCALE = 1.0f / 64.0f;

    // Decode 19 INT8 -> FP32
    float f[19];
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        f[d] = (float)dist[d * n_cells + cell] * INV_SCALE;
    }

    // Compute: density + simplified BGK
    float rho = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) rho += f[d];
    float tau_inv = 1.0f / 0.6f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float feq = rho / 19.0f;
        f[d] -= (f[d] - feq) * tau_inv;
    }

    // Encode FP32 -> INT8 and store (saturating)
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float scaled = f[d] * DIST_SCALE;
        int clamped = (int)fmaxf(-128.0f, fminf(127.0f, scaled));
        dist[d * n_cells + cell] = (signed char)clamped;
    }
}

// Pattern 3: FP8 E4M3 storage, FP32 compute, FP8 store-back
extern "C" __global__ void __launch_bounds__(128)
probe_mixed_fp8_fp32(__nv_fp8_storage_t *dist, int n_cells) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    // Decode 19 FP8 -> FP16 -> FP32
    float f[19];
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        __nv_fp8_storage_t fp8 = dist[d * n_cells + cell];
        __half_raw hraw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
        __half h;
        __builtin_memcpy(&h, &hraw, sizeof(h));
        f[d] = __half2float(h);
    }

    // Compute
    float rho = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) rho += f[d];
    float tau_inv = 1.0f / 0.6f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float feq = rho / 19.0f;
        f[d] -= (f[d] - feq) * tau_inv;
    }

    // Encode FP32 -> FP8 E4M3
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(f[d], __NV_SATFINITE, __NV_E4M3);
        dist[d * n_cells + cell] = fp8;
    }
}

// Pattern 4: Interleaved FP16 load + FP32 compute (no bulk decode)
// Tests whether interleaving decode with compute improves pipeline overlap
extern "C" __global__ void __launch_bounds__(128)
probe_mixed_interleaved(__half *dist, float *out, int n_cells) {
    int cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell >= n_cells) return;

    float rho = 0.0f;
    // Interleave: load one FP16, convert, accumulate immediately
    // (no bulk decode phase -- tests if this aids pipeline scheduling)
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float val = __half2float(dist[d * n_cells + cell]);
        rho += val;
    }

    out[cell] = rho;
}
