/*
 * SASS RE Probe: BCD (Binary Coded Decimal) Arithmetic
 * Isolates: Packed BCD add with carry correction, BCD multiply
 *
 * BCD stores one decimal digit (0-9) per 4-bit nibble.
 * Packed BCD: 2 decimal digits per byte (e.g., 99 decimal = 0x99)
 * Unpacked BCD: 1 decimal digit per byte (0x00-0x09)
 *
 * Ada SM 8.9: NO native BCD instructions.
 * x86 has DAA/DAS (decimal adjust after add/subtract) but GPU does not.
 * All BCD arithmetic must be emulated via integer ops:
 *   BCD ADD: binary add + carry correction (+6 when nibble > 9)
 *   BCD MUL: digit-by-digit multiply with carry propagation
 *
 * Expected cost: BCD add = ~6-8 SASS instructions per byte (IADD3 + BFE +
 * ISETP + SEL + LOP3 for carry correction). Much slower than binary.
 *
 * Historical note: BCD was used in early financial computing (COBOL, IBM S/360).
 * On GPU: no practical use case, but demonstrates the emulation cost.
 */

// Packed BCD add: add two 8-digit BCD numbers (32 bits each)
// Returns packed BCD result + carry
__device__ __forceinline__ unsigned int bcd_add32(unsigned int a, unsigned int b, int *carry) {
    // Binary add
    unsigned int sum = a + b;

    // Carry correction: for each nibble, if result > 9 or carry generated, add 6
    // This is the BCD "half-carry" correction
    unsigned int correction = 0;
    unsigned int carry_mask = ((a & 0x88888888u) + (b & 0x88888888u)) ^ (sum & 0x88888888u);

    // Check each nibble for overflow (>9 or binary carry from lower nibble)
    for (int nibble = 0; nibble < 8; nibble++) {
        unsigned int digit = (sum >> (nibble * 4)) & 0xF;
        if (digit > 9 || ((carry_mask >> (nibble * 4 + 3)) & 1)) {
            correction |= (6u << (nibble * 4));
        }
    }

    unsigned int result = sum + correction;
    *carry = (result < sum) ? 1 : 0;

    // Re-check after correction (cascaded carry)
    for (int nibble = 0; nibble < 8; nibble++) {
        if (((result >> (nibble * 4)) & 0xF) > 9) {
            result += (6u << (nibble * 4));
        }
    }

    return result;
}

// Packed BCD add chain (throughput measurement)
extern "C" __global__ void __launch_bounds__(32)
probe_bcd_add_chain(volatile unsigned int *vals, volatile long long *out) {
    unsigned int x = vals[0];  // BCD number: e.g., 0x12345678
    unsigned int y = vals[1];  // BCD increment: e.g., 0x00000001
    long long t0, t1;
    int carry;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 256; i++)
        x = bcd_add32(x, y, &carry);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    vals[3] = x;
    if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = 256; }
}

// BCD to binary conversion
__device__ __forceinline__ unsigned int bcd_to_binary(unsigned int bcd) {
    unsigned int result = 0;
    unsigned int multiplier = 1;
    for (int nibble = 0; nibble < 8; nibble++) {
        result += ((bcd >> (nibble * 4)) & 0xF) * multiplier;
        multiplier *= 10;
    }
    return result;
}

// Binary to BCD conversion
__device__ __forceinline__ unsigned int binary_to_bcd(unsigned int bin) {
    unsigned int bcd = 0;
    for (int nibble = 0; nibble < 8 && bin > 0; nibble++) {
        bcd |= ((bin % 10) << (nibble * 4));
        bin /= 10;
    }
    return bcd;
}

// BCD conversion throughput
extern "C" __global__ void __launch_bounds__(128)
probe_bcd_convert(unsigned int *out_bcd, unsigned int *out_bin,
                  const unsigned int *in_bcd, const unsigned int *in_bin, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    out_bin[i] = bcd_to_binary(in_bcd[i]);
    out_bcd[i] = binary_to_bcd(in_bin[i]);
}
