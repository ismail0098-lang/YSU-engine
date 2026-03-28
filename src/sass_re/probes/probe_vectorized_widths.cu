/*
 * SASS RE Probe: Vectorized Load/Store Widths Across All Precisions
 * Isolates: LDG.E.32/64/128, STG.E.32/64/128, LDS.32/64/128
 *           for every precision type and its x2/x4 vector variants
 *
 * Ada Lovelace SM 8.9 memory transaction widths:
 *   32-bit:  LDG.E     (float, int, half2, bf162)
 *   64-bit:  LDG.E.64  (float2, double, int2, short4)
 *   128-bit: LDG.E.128 (float4, double2, int4, char16)
 *
 * The memory subsystem delivers 128-byte cache lines. Wider loads reduce
 * instruction count per byte transferred, improving ILP and reducing
 * scheduling pressure. The optimal width depends on register pressure:
 *   128-bit loads use 4 registers but halve the instruction count vs 32-bit.
 *
 * Key SASS instructions to observe per width:
 *   LDG.E       -- 32-bit global load
 *   LDG.E.64    -- 64-bit global load (2 registers)
 *   LDG.E.128   -- 128-bit global load (4 registers)
 *   LDG.E.U8    -- 8-bit unsigned (FP8, INT8)
 *   LDG.E.U16   -- 16-bit unsigned (FP16, BF16, INT16)
 *   STG.E.U8    -- 8-bit store
 *   STG.E.U16   -- 16-bit store
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

/* ── FP32 x1/x2/x4 ─────────────────────────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f32x1(float *out, const float *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E + STG.E (32-bit)
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f32x2(float2 *out, const float2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64 + STG.E.64
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f32x4(float4 *out, const float4 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.128 + STG.E.128
}

/* ── FP64 x1/x2 ────────────────────────────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f64x1(double *out, const double *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64 (double = 64-bit naturally)
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f64x2(double2 *out, const double2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.128 (double2 = 128-bit)
}

/* ── FP16 x1/x2/x4 (half, half2, 2xhalf2) ─────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f16x1(__half *out, const __half *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.U16 + STG.E.U16
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f16x2(__half2 *out, const __half2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E (32-bit: half2 = 2x16 = 32 bits)
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_f16x4(float2 *out, const float2 *in) {
    // 4x FP16 = 64 bits = float2 container
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64
}

/* ── BF16 x1/x2 ────────────────────────────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_bf16x1(__nv_bfloat16 *out, const __nv_bfloat16 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.U16 + STG.E.U16
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_bf16x2(__nv_bfloat162 *out, const __nv_bfloat162 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned raw;
    memcpy(&raw, &in[i], sizeof(raw));
    memcpy(&out[i], &raw, sizeof(raw));
    // LDG.E (32-bit: bf162 = 2x16 = 32 bits)
}

/* ── INT8 x1/x4 (char, char4=uchar4) ───────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i8x1(signed char *out, const signed char *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.U8 + STG.E.U8
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i8x4(uchar4 *out, const uchar4 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E (32-bit: 4x8 = 32 bits)
}

/* ── INT16 x1/x2/x4 (short, short2, short4) ────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i16x1(short *out, const short *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.U16 + STG.E.U16
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i16x2(short2 *out, const short2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E (32-bit: 2x16 = 32 bits)
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i16x4(short4 *out, const short4 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64 (64-bit: 4x16 = 64 bits)
}

/* ── INT32 x1/x2/x4 ────────────────────────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i32x1(int *out, const int *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E (32-bit)
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i32x2(int2 *out, const int2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i32x4(int4 *out, const int4 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.128
}

/* ── INT64 x1/x2 ───────────────────────────────────── */
extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i64x1(long long *out, const long long *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.64
}

extern "C" __global__ void __launch_bounds__(128)
probe_ldg_i64x2(longlong2 *out, const longlong2 *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i];  // LDG.E.128
}
