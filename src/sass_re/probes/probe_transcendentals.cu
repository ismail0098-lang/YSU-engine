/*
 * SASS RE Probe: Transcendental and Special Function Decomposition
 * Isolates: How math.h functions decompose to MUFU + FFMA sequences
 *
 * CUDA provides two paths for transcendentals:
 *   1. Fast-math (--use_fast_math): single MUFU instruction
 *      sinf() -> MUFU.SIN, cosf() -> MUFU.COS, expf() -> MUFU.EX2
 *      Accuracy: ~2^-21 relative error (suitable for graphics/physics)
 *
 *   2. IEEE-compliant (default): multi-instruction sequence
 *      sinf() -> range reduction + MUFU.SIN + polynomial correction
 *      expf() -> MUFU.EX2 + FFMA correction chain
 *      Accuracy: 1-2 ULP (required for numerical analysis)
 *
 * From SASS RE latency measurements:
 *   MUFU.SIN: 23.50 cy, MUFU.EX2: 17.55 cy, MUFU.RCP: 41.53 cy
 *   FFMA: 4.53 cy (correction chain cost)
 *
 * This probe compiles WITHOUT --use_fast_math to show the full
 * IEEE decomposition, then key functions are also shown with __fmaf_rn()
 * to demonstrate the minimal MUFU path.
 *
 * Key SASS patterns:
 *   IEEE sinf:  ~12 instructions (range reduce + MUFU.SIN + 2 FFMA corrections)
 *   IEEE expf:  ~8 instructions (scale + MUFU.EX2 + FFMA correction)
 *   IEEE logf:  ~10 instructions (MUFU.LG2 + FFMA polynomial refinement)
 *   Fast sinf:  1 instruction (MUFU.SIN)
 *   sincosf:    2 instructions (MUFU.SIN + MUFU.COS, potentially fused)
 *
 * FP64 transcendentals use libdevice function CALLS (not MUFU):
 *   sin() -> CALL.REL to __nv_sin()
 *   exp() -> CALL.REL to __nv_exp()
 *   Each libdevice function is a ~50-200 instruction polynomial approx.
 */

#include <math.h>

/* ── FP32 IEEE transcendentals (default, no fast-math) ─── */

extern "C" __global__ void __launch_bounds__(32)
probe_sinf_ieee(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = sinf(in[i]);
}

extern "C" __global__ void __launch_bounds__(32)
probe_cosf_ieee(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = cosf(in[i]);
}

extern "C" __global__ void __launch_bounds__(32)
probe_expf_ieee(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = expf(in[i]);
}

extern "C" __global__ void __launch_bounds__(32)
probe_logf_ieee(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = logf(in[i]);
}

extern "C" __global__ void __launch_bounds__(32)
probe_sqrtf_ieee(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = sqrtf(in[i]);
}

extern "C" __global__ void __launch_bounds__(32)
probe_powf_ieee(float *out, const float *base, const float *exp_arr) {
    int i = threadIdx.x;
    out[i] = powf(base[i], exp_arr[i]);
}

/* ── FP32 fast-math intrinsics (single MUFU instruction) ── */

extern "C" __global__ void __launch_bounds__(32)
probe_sinf_fast(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = __sinf(in[i]);  // Direct MUFU.SIN
}

extern "C" __global__ void __launch_bounds__(32)
probe_cosf_fast(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = __cosf(in[i]);  // Direct MUFU.COS
}

extern "C" __global__ void __launch_bounds__(32)
probe_expf_fast(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = __expf(in[i]);  // MUFU.EX2 + multiply by log2(e)
}

extern "C" __global__ void __launch_bounds__(32)
probe_logf_fast(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = __logf(in[i]);  // MUFU.LG2 + multiply by ln(2)
}

/* ── sincosf: paired sin+cos (potentially fused on hardware) ── */
extern "C" __global__ void __launch_bounds__(32)
probe_sincosf(float *out_sin, float *out_cos, const float *in) {
    int i = threadIdx.x;
    float s, c;
    sincosf(in[i], &s, &c);
    out_sin[i] = s;
    out_cos[i] = c;
}

/* ── FP64 transcendentals (libdevice CALL sequences) ────── */

extern "C" __global__ void __launch_bounds__(32)
probe_sin_f64(double *out, const double *in) {
    int i = threadIdx.x;
    out[i] = sin(in[i]);  // CALL.REL to __nv_sin
}

extern "C" __global__ void __launch_bounds__(32)
probe_exp_f64(double *out, const double *in) {
    int i = threadIdx.x;
    out[i] = exp(in[i]);  // CALL.REL to __nv_exp
}

extern "C" __global__ void __launch_bounds__(32)
probe_log_f64(double *out, const double *in) {
    int i = threadIdx.x;
    out[i] = log(in[i]);  // CALL.REL to __nv_log
}

extern "C" __global__ void __launch_bounds__(32)
probe_sqrt_f64(double *out, const double *in) {
    int i = threadIdx.x;
    out[i] = sqrt(in[i]);  // May use MUFU.DSQRT or CALL.REL
}

/* ── Special functions ──────────────────────────────────── */

extern "C" __global__ void __launch_bounds__(32)
probe_rsqrtf(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = rsqrtf(in[i]);  // MUFU.RSQ (fast) or IEEE path
}

extern "C" __global__ void __launch_bounds__(32)
probe_erfcf(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = erfcf(in[i]);  // Complementary error function (libdevice)
}

extern "C" __global__ void __launch_bounds__(32)
probe_cbrtf(float *out, const float *in) {
    int i = threadIdx.x;
    out[i] = cbrtf(in[i]);  // Cube root (RCP + EX2/LG2 sequence)
}
