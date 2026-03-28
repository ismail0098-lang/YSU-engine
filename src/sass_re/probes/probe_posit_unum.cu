/*
 * SASS RE Probe: Posit (Type III Unum) Arithmetic Emulation
 * Isolates: Software posit decode/encode/add/mul via FP32 promotion
 *
 * Posit<n,es> format (Gustafson, 2017):
 *   n bits total, es exponent seed bits
 *   Fields: [sign][regime][exponent][fraction]
 *   Regime: variable-length run of same bit (terminated by opposite)
 *     Determines the "super-exponent" (scaling by useed^regime_value)
 *   Exponent: es bits of standard binary exponent
 *   Fraction: remaining bits (variable length!)
 *
 *   Posit<8,0>:  8 bits, no exponent seed, useed=2
 *   Posit<16,1>: 16 bits, 1 exponent bit, useed=4
 *   Posit<32,2>: 32 bits, 2 exponent bits, useed=16 (standard)
 *
 * Ada SM 8.9: NO native posit hardware on ANY GPU architecture.
 * All posit ops must be decoded to FP32/FP64, computed, then re-encoded.
 * Decode cost: ~20-40 SASS instructions (regime extraction + shift chain)
 * Encode cost: ~30-50 SASS instructions (regime construction + packing)
 *
 * Posits are interesting for comparison: same bit width as FP, but
 * variable-precision (tapered) -- high precision near 1.0, low at extremes.
 */

// Posit<8,0> decode: 8 bits -> float
__device__ __forceinline__ float posit8_decode(unsigned char p) {
    if (p == 0) return 0.0f;
    if (p == 0x80) return 1.0f / 0.0f;  // NaR (Not a Real)

    int sign = (p >> 7) & 1;
    unsigned char abs_p = sign ? (~p + 1) : p;  // Two's complement for negative

    // Regime: count leading bits after sign
    int regime_sign = (abs_p >> 6) & 1;
    int regime_len = 0;
    for (int b = 6; b >= 0; b--) {
        if (((abs_p >> b) & 1) == regime_sign) regime_len++;
        else break;
    }
    int k = regime_sign ? (regime_len - 1) : (-regime_len);

    // Fraction: remaining bits after regime + terminator
    int frac_bits = 7 - regime_len - 1;  // -1 for terminator
    if (frac_bits < 0) frac_bits = 0;
    unsigned char frac_mask = (1 << frac_bits) - 1;
    unsigned char frac = abs_p & frac_mask;

    // Value = useed^k * (1 + frac/2^frac_bits)
    // For Posit<8,0>: useed = 2, so value = 2^k * (1 + frac/2^frac_bits)
    float value = exp2f((float)k) * (1.0f + (float)frac / (float)(1 << frac_bits));

    return sign ? -value : value;
}

// Posit<8,0> encode: float -> 8 bits
__device__ __forceinline__ unsigned char posit8_encode(float val) {
    if (val == 0.0f) return 0;
    if (!isfinite(val)) return 0x80;  // NaR

    int sign = (val < 0.0f);
    float abs_val = fabsf(val);

    // Find regime k: k = floor(log2(abs_val))
    int k = (int)floorf(log2f(abs_val));

    // Fraction: abs_val / 2^k - 1
    float frac_f = abs_val / exp2f((float)k) - 1.0f;

    // Construct regime bits
    unsigned char result = 0;
    int bit_pos = 6;
    if (k >= 0) {
        // Regime: (k+1) ones followed by a zero
        for (int i = 0; i <= k && bit_pos >= 0; i++) {
            result |= (1 << bit_pos);
            bit_pos--;
        }
        // Terminator zero (already zero)
        bit_pos--;
    } else {
        // Regime: |k| zeros followed by a one
        for (int i = 0; i < -k - 1 && bit_pos >= 0; i++) {
            bit_pos--;  // Leave as zero
        }
        if (bit_pos >= 0) {
            result |= (1 << bit_pos);  // Terminator one
            bit_pos--;
        }
    }

    // Fraction bits
    int frac_bits = bit_pos + 1;
    if (frac_bits > 0) {
        int frac_int = (int)roundf(frac_f * (float)(1 << frac_bits));
        frac_int = min(frac_int, (1 << frac_bits) - 1);
        result |= (unsigned char)frac_int;
    }

    // Apply sign (two's complement)
    if (sign) result = ~result + 1;

    return result;
}

// Posit<8,0> decode chain for latency measurement
extern "C" __global__ void __launch_bounds__(32)
probe_posit8_decode_chain(float *out, const unsigned char *in) {
    int i = threadIdx.x;
    unsigned char p = in[i];
    float acc = 0.0f;

    #pragma unroll 1
    for (int j = 0; j < 512; j++) {
        acc += posit8_decode(p);
        p = (unsigned char)((int)acc & 0xFF);  // Feed back
    }
    out[i] = acc;
}

// Posit<8,0> round-trip: encode + decode
extern "C" __global__ void __launch_bounds__(128)
probe_posit8_roundtrip(float *out, const float *in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    unsigned char p = posit8_encode(in[i]);
    out[i] = posit8_decode(p);
}

// Posit<8,0> add via FP32 promotion (the only viable path on GPU)
extern "C" __global__ void __launch_bounds__(128)
probe_posit8_add(unsigned char *out, const unsigned char *a,
                 const unsigned char *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float fa = posit8_decode(a[i]);
    float fb = posit8_decode(b[i]);
    out[i] = posit8_encode(fa + fb);
}
