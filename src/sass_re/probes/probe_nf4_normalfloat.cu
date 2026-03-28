/*
 * SASS RE Probe: NF4 (NormalFloat4) -- QLoRA Quantization Format
 * Isolates: LUT-based NF4 encode/decode, throughput comparison vs FP4/INT4
 *
 * NormalFloat4 (Dettmers et al., 2023): 4-bit quantization with levels
 * placed at the quantiles of a standard normal distribution N(0,1).
 * 16 values (asymmetric around 0):
 *   {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.09105, 0.0,
 *    0.07959, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
 *
 * Used in QLoRA for 4-bit LLM quantization. Better SNR than uniform INT4
 * because levels match the distribution of neural network weights.
 *
 * Ada SM 8.9: NO native NF4 hardware. Emulated via 16-entry LUT (same
 * as FP4 E2M1, but different LUT values). Decode: 1 LDS from constant mem.
 * Encode: binary search over 16 values or pre-computed bin boundaries.
 *
 * NO tensor core support for NF4 on any NVIDIA GPU.
 */

// NF4 quantile-based lookup table (from QLoRA paper)
__constant__ float NF4_DECODE[16] = {
    -1.0f,     -0.6961928f, -0.5250730f, -0.3949338f,
    -0.2844391f, -0.1848489f, -0.09105003f, 0.0f,
     0.07958923f, 0.1609302f,  0.2461123f,  0.3379253f,
     0.4407197f,  0.5626170f,  0.7229568f,  1.0f
};

// Bin boundaries for fast NF4 encode (midpoints between consecutive values)
__constant__ float NF4_BOUNDARIES[15] = {
    -0.8480964f, -0.6106329f, -0.4600034f, -0.3396864f,
    -0.2346440f, -0.1379745f, -0.04552512f,
     0.03979461f, 0.1202597f,  0.2035213f,  0.2920188f,
     0.3893225f,  0.5016683f,  0.6427869f,  0.8614784f
};

// NF4 decode: nibble -> float via LUT
extern "C" __global__ void __launch_bounds__(128)
probe_nf4_decode(float *out, const unsigned char *packed, int n_bytes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_bytes) return;
    unsigned char byte = packed[i];
    out[i * 2 + 0] = NF4_DECODE[byte & 0xF];
    out[i * 2 + 1] = NF4_DECODE[(byte >> 4) & 0xF];
}

// NF4 encode: float -> nibble via binary search on boundaries
__device__ __forceinline__ unsigned char nf4_encode(float val) {
    // Clamp to [-1, 1]
    val = fmaxf(-1.0f, fminf(1.0f, val));
    // Binary search over 15 boundaries
    unsigned char idx = 0;
    #pragma unroll
    for (int b = 0; b < 15; b++) {
        if (val > NF4_BOUNDARIES[b]) idx = b + 1;
    }
    return idx;
}

extern "C" __global__ void __launch_bounds__(128)
probe_nf4_encode(unsigned char *packed, const float *in, int n_pairs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_pairs) return;
    unsigned char lo = nf4_encode(in[i * 2 + 0]);
    unsigned char hi = nf4_encode(in[i * 2 + 1]);
    packed[i] = (hi << 4) | lo;
}

// NF4 round-trip throughput: encode + decode
extern "C" __global__ void __launch_bounds__(128)
probe_nf4_roundtrip(float *out, const float *in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    // Encode to NF4
    unsigned char nf4 = nf4_encode(in[i]);
    // Decode from NF4
    out[i] = NF4_DECODE[nf4];
}

// Quantization error comparison: NF4 vs INT4 vs FP4 E2M1
extern "C" __global__ void __launch_bounds__(32)
probe_nf4_error(float *err_nf4, float *err_int4, float *err_fp4,
                const float *test_values, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    float val = test_values[i];

    // NF4 error
    unsigned char nf4 = nf4_encode(val);
    err_nf4[i] = fabsf(val - NF4_DECODE[nf4]);

    // INT4 error (DIST_SCALE=14)
    int q_int4 = (int)roundf(val * 14.0f);
    q_int4 = max(-8, min(7, q_int4));
    err_int4[i] = fabsf(val - (float)q_int4 / 14.0f);

    // FP4 E2M1 error (from probe_4bit_formats.cu pattern)
    float aval = fabsf(val);
    unsigned char fp4;
    if (aval < 0.25f)       fp4 = 0;
    else if (aval < 0.75f)  fp4 = 1;
    else if (aval < 1.25f)  fp4 = 2;
    else if (aval < 1.75f)  fp4 = 3;
    else if (aval < 2.5f)   fp4 = 4;
    else if (aval < 3.5f)   fp4 = 5;
    else if (aval < 5.0f)   fp4 = 6;
    else                     fp4 = 7;
    if (val < 0.0f) fp4 |= 0x8;
    const float fp4_lut[16] = {0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};
    err_fp4[i] = fabsf(val - fp4_lut[fp4]);
}
