/*
 * SASS RE Probe: byte-lane predicate packing
 *
 * cuDNN library mining surfaced P2R.B1 / P2R.B2 / P2R.B3. This probe tries to
 * force the compiler to pack 16 live predicate results into separate byte lanes
 * of a single GPR so those byte-select spellings appear directly on sm_89.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_bytepack(uint32_t *out, const float *in) {
    int i = threadIdx.x;
    float v[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        v[k] = in[k * 32 + i];
    }

    int p0  = (v[0]  > 0.0f);
    int p1  = (v[1]  < 1.0f);
    int p2  = (v[2]  >= 0.5f);
    int p3  = (v[3]  != 0.0f);
    int p4  = (v[4]  > -1.0f);
    int p5  = (v[5]  <= 2.0f);
    int p6  = (v[6]  == 3.0f);
    int p7  = (v[7]  > 4.0f);
    int p8  = (v[8]  < -2.0f);
    int p9  = (v[9]  >= 1.5f);
    int p10 = (v[10] != 5.0f);
    int p11 = (v[11] > 6.0f);
    int p12 = (v[12] <= -3.0f);
    int p13 = (v[13] >= 7.0f);
    int p14 = (v[14] < 8.0f);
    int p15 = (v[15] != 9.0f);

    uint32_t packed = 0;
    packed |= (uint32_t)p0;
    packed |= (uint32_t)p1 << 1;
    packed |= (uint32_t)p2 << 2;
    packed |= (uint32_t)p3 << 3;
    packed |= (uint32_t)p4 << 8;
    packed |= (uint32_t)p5 << 9;
    packed |= (uint32_t)p6 << 10;
    packed |= (uint32_t)p7 << 11;
    packed |= (uint32_t)p8 << 16;
    packed |= (uint32_t)p9 << 17;
    packed |= (uint32_t)p10 << 18;
    packed |= (uint32_t)p11 << 19;
    packed |= (uint32_t)p12 << 24;
    packed |= (uint32_t)p13 << 25;
    packed |= (uint32_t)p14 << 26;
    packed |= (uint32_t)p15 << 27;

    // Keep the predicates live together and prevent trivial folding.
    packed ^= (packed >> 7);
    out[i] = packed;
}
