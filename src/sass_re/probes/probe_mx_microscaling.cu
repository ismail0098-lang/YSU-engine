/*
 * SASS RE Probe: MX (Microscaling) Formats -- MXFP8, MXFP4, MXINT8
 * Isolates: Block-scaled quantization with shared exponent per 32 elements
 *
 * OCP Microscaling (Open Compute Project standard):
 *   Instead of each element having its own exponent, a block of 32 elements
 *   shares a single E8M0 scale factor (8-bit exponent, no mantissa).
 *   Each element stores only mantissa + sign in the sub-byte format.
 *
 *   MXFP8 (E4M3): 32 elements * 1 byte + 1 byte shared scale = 33 bytes/block
 *     vs per-element FP8: 32 bytes/block (3% overhead for shared exponent)
 *   MXFP4 (E2M1): 32 elements * 0.5 bytes + 1 byte = 17 bytes/block
 *     vs per-element FP4: 16 bytes/block (6% overhead)
 *   MXINT8: 32 elements * 1 byte + 1 byte = 33 bytes/block
 *     Integer elements scaled by shared FP32 factor
 *
 * Ada SM 8.9: NO native MX hardware. Emulated via:
 *   1. Load shared scale factor (1 LDG per 32 elements)
 *   2. Load element values (1 LDG per element)
 *   3. Multiply element by scale factor (1 FMUL per element)
 *
 * Blackwell SM 10.0 has native MX tensor core support for MXFP8/MXFP4.
 */

// E8M0 shared scale factor: 8-bit unsigned exponent, bias=127
// Value = 2^(e - 127), where e is the stored byte value
__device__ __forceinline__ float e8m0_to_float(unsigned char e8m0) {
    // Construct FP32 from exponent: set exponent field, zero mantissa
    unsigned int bits = ((unsigned int)e8m0) << 23;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// ── MXFP8 (E4M3 elements + E8M0 shared scale per 32 elements) ──

extern "C" __global__ void __launch_bounds__(128)
probe_mxfp8_decode(float *out,
                   const unsigned char *elements,  // [n_blocks * 32] E4M3 values
                   const unsigned char *scales,     // [n_blocks] E8M0 scale factors
                   int n_blocks) {
    int block_idx = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int elem_in_block = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    if (block_idx >= n_blocks) return;

    // Load shared scale (broadcast: all 32 threads read same address)
    float scale = e8m0_to_float(scales[block_idx]);

    // Load and decode element
    unsigned char e4m3 = elements[block_idx * 32 + elem_in_block];

    // Decode E4M3 -> FP16 -> FP32 (same as FP8 probe)
    // Simplified: use the element as a 1+4+3 float fraction
    int sign = (e4m3 >> 7) & 1;
    int exp = (e4m3 >> 3) & 0xF;
    int man = e4m3 & 0x7;
    float val;
    if (exp == 0) {
        val = (float)man / 8.0f * (1.0f / 8.0f);  // Denormal
    } else if (exp == 15) {
        val = (man == 0) ? 448.0f : 0.0f;  // Max value or NaN
    } else {
        val = (1.0f + (float)man / 8.0f) * exp2f((float)(exp - 7));
    }
    if (sign) val = -val;

    // Apply shared scale
    out[block_idx * 32 + elem_in_block] = val * scale;
}

// ── MXFP4 (E2M1 elements + E8M0 shared scale per 32 elements) ──

__constant__ float MXFP4_ELEM_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void __launch_bounds__(128)
probe_mxfp4_decode(float *out,
                   const unsigned char *packed_elements, // [n_blocks * 16] nibble-packed
                   const unsigned char *scales,           // [n_blocks] E8M0
                   int n_blocks) {
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int block_idx = global_idx / 32;
    int elem_in_block = global_idx % 32;
    if (block_idx >= n_blocks) return;

    float scale = e8m0_to_float(scales[block_idx]);

    // Each byte holds 2 nibbles
    int byte_idx = block_idx * 16 + elem_in_block / 2;
    unsigned char byte = packed_elements[byte_idx];
    unsigned char nibble = (elem_in_block & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);

    float val = MXFP4_ELEM_LUT[nibble] * scale;
    out[global_idx] = val;
}

// ── MXINT8 (INT8 elements + FP32 shared scale per 32 elements) ──

extern "C" __global__ void __launch_bounds__(128)
probe_mxint8_decode(float *out,
                    const signed char *elements,  // [n_blocks * 32] INT8 values
                    const float *scales,           // [n_blocks] FP32 scale factors
                    int n_blocks) {
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int block_idx = global_idx / 32;
    int elem_in_block = global_idx % 32;
    if (block_idx >= n_blocks) return;

    // Load shared scale (broadcast within warp if block_size=32=warp_size)
    float scale = scales[block_idx];

    // Dequantize: float_value = int8_element * scale
    float val = (float)elements[global_idx] * scale;
    out[global_idx] = val;
}

// ── MXFP8 encode (FP32 -> block-scaled E4M3) ──

extern "C" __global__ void __launch_bounds__(128)
probe_mxfp8_encode(unsigned char *out_elements,
                   unsigned char *out_scales,
                   const float *in,
                   int n_blocks) {
    // Each warp processes one block of 32 elements
    int block_idx = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int elem_in_block = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    if (block_idx >= n_blocks) return;

    float val = in[block_idx * 32 + elem_in_block];

    // Step 1: find max absolute value in block (warp-level reduction)
    float abs_val = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, offset));

    // Step 2: compute shared scale (E8M0 = 2^floor(log2(max_abs)))
    float scale;
    unsigned char e8m0;
    if (abs_val > 0.0f) {
        int exp;
        frexpf(abs_val, &exp);
        e8m0 = (unsigned char)min(255, max(0, exp + 127));
        scale = e8m0_to_float(e8m0);
    } else {
        e8m0 = 0;
        scale = 1.0f;
    }

    // Lane 0 writes the scale
    if (elem_in_block == 0)
        out_scales[block_idx] = e8m0;

    // Step 3: quantize element to E4M3 in scaled space
    float scaled = val / scale;
    // Clamp to E4M3 range and quantize (simplified)
    scaled = fmaxf(-448.0f, fminf(448.0f, scaled));
    // Simplified encode: just store clamped value as byte
    out_elements[block_idx * 32 + elem_in_block] = (unsigned char)(int)roundf(scaled);
}
