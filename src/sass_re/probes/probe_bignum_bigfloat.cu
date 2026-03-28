/*
 * SASS RE Probe: Arbitrary Precision Arithmetic (Bignum / BigFloat)
 * Isolates: Multi-word carry chains, cascaded DD/QD, scaling behavior
 *
 * Bignum: arbitrary-precision integers using arrays of 32/64-bit "limbs".
 * BigFloat: arbitrary-precision floats using DD (double-double) cascade.
 *
 * This probe measures how cost scales with precision:
 *   128-bit:  2-limb carry chain (IADD3 + IADD3.X)
 *   256-bit:  4-limb carry chain
 *   512-bit:  8-limb carry chain
 *   1024-bit: 16-limb carry chain
 *
 * Expected scaling: linear in number of limbs for add, quadratic for multiply.
 *
 * BigFloat via DD cascade:
 *   double-double (DD): 2 doubles, ~106-bit mantissa
 *   quad-double (QD): 4 doubles, ~212-bit mantissa
 *   Scaling: DD arithmetic cost ~4-6x FP64, QD ~16-24x FP64
 */

// Knuth 2-sum: error-free addition
__device__ __forceinline__
void two_sum(double a, double b, double &s, double &e) {
    s = a + b;
    double v = s - a;
    e = (a - (s - v)) + (b - v);
}

// DD add: (a_hi, a_lo) + (b_hi, b_lo)
__device__ __forceinline__
void dd_add(double ah, double al, double bh, double bl,
            double &sh, double &sl) {
    double s, e;
    two_sum(ah, bh, s, e);
    e += al + bl;
    two_sum(s, e, sh, sl);
}

// ── Bignum ADD at various widths (latency chains) ──

extern "C" __global__ void __launch_bounds__(32)
probe_bignum_128_add(volatile unsigned long long *vals, volatile long long *out) {
    unsigned long long x0 = vals[0], x1 = vals[1];
    unsigned long long one = 1;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        unsigned long long s0 = x0 + one;
        unsigned long long c = (s0 < x0) ? 1ULL : 0ULL;
        x1 += c;
        x0 = s0;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = x0; vals[3] = x1;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

extern "C" __global__ void __launch_bounds__(32)
probe_bignum_256_add(volatile unsigned long long *vals, volatile long long *out) {
    unsigned long long x[4];
    for (int k = 0; k < 4; k++) x[k] = vals[k];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        unsigned long long s = x[0] + 1;
        unsigned long long c = (s < x[0]) ? 1ULL : 0ULL;
        x[0] = s;
        for (int k = 1; k < 4; k++) {
            s = x[k] + c;
            c = (s < x[k]) ? 1ULL : 0ULL;
            x[k] = s;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    for (int k = 0; k < 4; k++) vals[k + 4] = x[k];
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

extern "C" __global__ void __launch_bounds__(32)
probe_bignum_512_add(volatile unsigned long long *vals, volatile long long *out) {
    unsigned long long x[8];
    for (int k = 0; k < 8; k++) x[k] = vals[k];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        unsigned long long s = x[0] + 1;
        unsigned long long c = (s < x[0]) ? 1ULL : 0ULL;
        x[0] = s;
        for (int k = 1; k < 8; k++) {
            s = x[k] + c;
            c = (s < x[k]) ? 1ULL : 0ULL;
            x[k] = s;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    for (int k = 0; k < 8; k++) vals[k + 8] = x[k];
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

extern "C" __global__ void __launch_bounds__(32)
probe_bignum_1024_add(volatile unsigned long long *vals, volatile long long *out) {
    unsigned long long x[16];
    for (int k = 0; k < 16; k++) x[k] = vals[k];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        unsigned long long s = x[0] + 1;
        unsigned long long c = (s < x[0]) ? 1ULL : 0ULL;
        x[0] = s;
        for (int k = 1; k < 16; k++) {
            s = x[k] + c;
            c = (s < x[k]) ? 1ULL : 0ULL;
            x[k] = s;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    for (int k = 0; k < 16; k++) vals[k + 16] = x[k];
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

// ── BigFloat: DD and QD add chains ──

extern "C" __global__ void __launch_bounds__(32)
probe_bigfloat_dd_add(volatile double *vals, volatile long long *out) {
    double ah = vals[0], al = vals[1];
    double bh = 0.001, bl = 1e-18;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++)
        dd_add(ah, al, bh, bl, ah, al);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    vals[2] = ah; vals[3] = al;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

extern "C" __global__ void __launch_bounds__(32)
probe_bigfloat_qd_add(volatile double *vals, volatile long long *out) {
    double a[4], b[4] = {0.001, 1e-18, 1e-34, 1e-50};
    for (int k = 0; k < 4; k++) a[k] = vals[k];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        // Simplified QD add: cascade of two_sum
        double s0, e0; two_sum(a[0], b[0], s0, e0);
        double s1, e1; two_sum(a[1] + e0, b[1], s1, e1);
        double s2, e2; two_sum(a[2] + e1, b[2], s2, e2);
        a[3] = a[3] + b[3] + e2;
        a[0] = s0; a[1] = s1; a[2] = s2;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    for (int k = 0; k < 4; k++) vals[k + 4] = a[k];
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}
