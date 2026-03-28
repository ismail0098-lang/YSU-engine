/*
 * SASS RE Probe: Sub-Byte Integer Packing (INT4/FP4 Nibble Operations)
 * Isolates: SHF (funnel shift), BFI (bit field insert), BFE (bit field extract)
 *           for nibble-granularity pack/unpack patterns
 *
 * INT4 and FP4 formats store 2 values per byte. The pack/unpack operations
 * use combinations of shifts, masks, and bit field instructions.
 *
 * From kernels_int4.cu:
 *   Extract upper nibble: (byte >> 4) & 0xF
 *   Extract lower nibble: byte & 0xF
 *   Sign extend INT4: (raw >= 8) ? (raw - 16) : raw
 *   Pack two nibbles: (upper << 4) | lower
 *
 * From kernels_fp4.cu:
 *   FP4 E2M1 decode: lookup table FP4_DECODE[16] -> float
 *   FP4 E2M1 encode: float_to_fp4() via min-distance quantization
 *
 * The 2-cells-per-thread pattern avoids read-modify-write nibble races:
 * thread k owns byte k, processes cells 2k (lower nibble) and 2k+1 (upper).
 *
 * Key SASS instructions:
 *   SHF.R.U32   -- funnel shift right (extract bits)
 *   BFE         -- bit field extract
 *   BFI         -- bit field insert (pack without disturbing other bits)
 *   LOP3.LUT    -- 3-input logic for masking
 *   PRMT        -- permute bytes (for efficient byte-lane selection)
 */

// INT4 nibble extraction (both lanes from one byte)
extern "C" __global__ void __launch_bounds__(32)
probe_nibble_extract(int *out_lo, int *out_hi, const unsigned char *packed) {
    int i = threadIdx.x;
    unsigned char byte = packed[i];

    // Lower nibble: byte & 0xF
    int lo = byte & 0xF;
    // Upper nibble: (byte >> 4) & 0xF
    int hi = (byte >> 4) & 0xF;

    // Sign-extend INT4 [-8, 7] from unsigned [0, 15]
    int lo_signed = (lo >= 8) ? (lo - 16) : lo;
    int hi_signed = (hi >= 8) ? (hi - 16) : hi;

    out_lo[i] = lo_signed;
    out_hi[i] = hi_signed;
}

// INT4 nibble packing (two values into one byte)
extern "C" __global__ void __launch_bounds__(32)
probe_nibble_pack(unsigned char *packed, const int *lo_vals, const int *hi_vals) {
    int i = threadIdx.x;

    // Clamp to INT4 range [-8, 7] then pack
    int lo = lo_vals[i];
    int hi = hi_vals[i];

    // Saturating clamp
    lo = (lo < -8) ? -8 : ((lo > 7) ? 7 : lo);
    hi = (hi < -8) ? -8 : ((hi > 7) ? 7 : hi);

    // Pack: lower nibble = lo & 0xF, upper nibble = (hi & 0xF) << 4
    unsigned char byte = (unsigned char)(((hi & 0xF) << 4) | (lo & 0xF));
    packed[i] = byte;
}

// FP4 E2M1 lookup table decode (from kernels_fp4.cu pattern)
extern "C" __global__ void __launch_bounds__(32)
probe_fp4_decode(float *out, const unsigned char *packed) {
    // E2M1 format: 1 sign + 2 exponent + 1 mantissa
    // Values: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and negatives
    __shared__ float FP4_DECODE[16];
    if (threadIdx.x < 16) {
        const float table[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        FP4_DECODE[threadIdx.x] = table[threadIdx.x];
    }
    __syncthreads();

    int i = threadIdx.x;
    unsigned char byte = packed[i];

    // Decode lower nibble
    float lo = FP4_DECODE[byte & 0xF];
    // Decode upper nibble
    float hi = FP4_DECODE[(byte >> 4) & 0xF];

    out[i * 2 + 0] = lo;
    out[i * 2 + 1] = hi;
}

// Bulk nibble unpack: 19 directions, 2 cells per thread (D3Q19 pattern)
// This is the actual access pattern from kernels_int4.cu
extern "C" __global__ void __launch_bounds__(128)
probe_nibble_bulk_unpack(float *f_out, const unsigned char *packed_dist,
                         int n_cells) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid * 2 >= n_cells) return;

    int cell_a = tid * 2;       // Lower nibble cell
    int cell_b = tid * 2 + 1;   // Upper nibble cell

    float fa[19], fb[19];

    // Unpack 19 nibble pairs into 19 float pairs
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        // SoA nibble layout: packed_dist[d * (n_cells/2) + tid]
        int byte_idx = d * (n_cells / 2) + tid;
        unsigned char byte = packed_dist[byte_idx];

        int raw_a = byte & 0xF;
        int raw_b = (byte >> 4) & 0xF;

        // Sign-extend INT4
        int signed_a = (raw_a >= 8) ? (raw_a - 16) : raw_a;
        int signed_b = (raw_b >= 8) ? (raw_b - 16) : raw_b;

        // Scale to float (DIST_SCALE = 14 for INT4)
        fa[d] = (float)signed_a / 14.0f;
        fb[d] = (float)signed_b / 14.0f;
    }

    // Store (prevent dead-code elimination)
    float sum_a = 0.0f, sum_b = 0.0f;
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        sum_a += fa[d];
        sum_b += fb[d];
    }
    f_out[cell_a] = sum_a;
    if (cell_b < n_cells)
        f_out[cell_b] = sum_b;
}

// PRMT byte permute probe (used in efficient byte-lane selection)
extern "C" __global__ void __launch_bounds__(32)
probe_prmt_permute(unsigned int *out, const unsigned int *a,
                   const unsigned int *b) {
    int i = threadIdx.x;
    unsigned int va = a[i];
    unsigned int vb = b[i];

    // PRMT selects bytes from two 32-bit registers based on a selector
    unsigned int r;

    // Selector 0x3210: identity (bytes from 'a' in order)
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(va), "r"(vb), "r"(0x3210));
    out[i] = r;

    // Selector 0x7654: all bytes from 'b'
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(va), "r"(vb), "r"(0x7654));
    out[i + 32] = r;

    // Selector 0x0123: reverse bytes of 'a'
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(va), "r"(vb), "r"(0x0123));
    out[i + 64] = r;
}
