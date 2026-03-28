/*
 * SASS RE Probe: Fixed-Point Arithmetic (Q-Format)
 * Isolates: Q8.8, Q16.16, Q1.15 integer-based fixed-point operations
 *
 * Fixed-point uses integer ALU with implicit scaling:
 *   Q8.8:  16 bits (8 integer + 8 fractional), range [-128, 127.996]
 *   Q16.16: 32 bits (16 int + 16 frac), range [-32768, 32767.99998]
 *   Q1.15: 16 bits (1 sign + 15 frac), range [-1.0, 0.99997], DSP standard
 *
 * Ada SM 8.9: NO native fixed-point ALU.
 * All fixed-point ops map to integer instructions:
 *   FXP ADD: IADD3 (same as integer add, free!)
 *   FXP MUL: IMAD + SHF (multiply then shift right by frac_bits)
 *   FXP FMA: IMAD + IADD3 + SHF
 *
 * Key insight: fixed-point add has IDENTICAL latency to integer add (2.52 cy).
 * Fixed-point multiply adds a right-shift: IMAD (4.53) + SHF (4.52) = ~9 cy.
 */

// Q16.16: 32-bit fixed-point (16 integer + 16 fractional bits)
typedef int q16_16_t;

__device__ __forceinline__ q16_16_t float_to_q16_16(float val) {
    return (q16_16_t)(val * 65536.0f);
}
__device__ __forceinline__ float q16_16_to_float(q16_16_t val) {
    return (float)val / 65536.0f;
}
__device__ __forceinline__ q16_16_t q16_16_add(q16_16_t a, q16_16_t b) {
    return a + b;  // IADD3: same as integer add!
}
__device__ __forceinline__ q16_16_t q16_16_mul(q16_16_t a, q16_16_t b) {
    // (a * b) >> 16: need 64-bit intermediate to avoid overflow
    long long product = (long long)a * (long long)b;
    return (q16_16_t)(product >> 16);
}

// Q8.8: 16-bit fixed-point (8 integer + 8 fractional bits)
typedef short q8_8_t;

__device__ __forceinline__ q8_8_t float_to_q8_8(float val) {
    return (q8_8_t)(val * 256.0f);
}
__device__ __forceinline__ float q8_8_to_float(q8_8_t val) {
    return (float)val / 256.0f;
}
__device__ __forceinline__ q8_8_t q8_8_mul(q8_8_t a, q8_8_t b) {
    return (q8_8_t)(((int)a * (int)b) >> 8);
}

// Q1.15: 16-bit signed fractional (DSP format, range [-1, 1))
typedef short q1_15_t;

__device__ __forceinline__ q1_15_t float_to_q1_15(float val) {
    return (q1_15_t)(val * 32768.0f);
}
__device__ __forceinline__ q1_15_t q1_15_mul(q1_15_t a, q1_15_t b) {
    // Saturating multiply: ((int)a * (int)b) >> 15, clamped
    int product = ((int)a * (int)b) >> 15;
    if (product > 32767) product = 32767;
    if (product < -32768) product = -32768;
    return (q1_15_t)product;
}

// Q16.16 add chain (should be same latency as IADD3: 2.52 cy)
extern "C" __global__ void __launch_bounds__(32)
probe_fxp_q16_16_add(volatile int *vals, volatile long long *out) {
    q16_16_t x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 512; i++)
        x = q16_16_add(x, y);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 512; }
}

// Q16.16 multiply chain (IMAD + shift: ~9 cy expected)
extern "C" __global__ void __launch_bounds__(32)
probe_fxp_q16_16_mul(volatile int *vals, volatile long long *out) {
    q16_16_t x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 512; i++)
        x = q16_16_mul(x, y);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 512; }
}

// Q8.8 multiply chain
extern "C" __global__ void __launch_bounds__(32)
probe_fxp_q8_8_mul(volatile short *vals, volatile long long *out) {
    q8_8_t x = vals[0], y = vals[1];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll
    for (int i = 0; i < 512; i++)
        x = q8_8_mul(x, y);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 512; }
}

// D3Q19 LBM equilibrium in Q16.16 fixed-point (hot path comparison)
extern "C" __global__ void __launch_bounds__(128)
probe_fxp_lbm_equilibrium(int *feq_out, const int *rho_q,
                           const int *ux_q, const int *uy_q,
                           int n_cells) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_cells) return;
    q16_16_t rho = rho_q[i];
    q16_16_t ux = ux_q[i], uy = uy_q[i];
    // Simplified 2D equilibrium in fixed-point
    q16_16_t usq = q16_16_mul(ux, ux) + q16_16_mul(uy, uy);
    q16_16_t w0 = float_to_q16_16(4.0f/9.0f);
    q16_16_t base = float_to_q16_16(1.0f) - q16_16_mul(float_to_q16_16(1.5f), usq);
    feq_out[i] = q16_16_mul(w0, q16_16_mul(rho, base));
}
