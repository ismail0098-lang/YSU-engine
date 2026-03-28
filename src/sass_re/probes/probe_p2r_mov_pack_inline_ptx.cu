/*
 * SASS RE Probe: incremental predicate-byte insertion
 *
 * cuDNN mining surfaced P2R.B1 / P2R.B2 / P2R.B3. Those spellings appear in
 * contexts where several predicate results are inserted into successive byte
 * lanes of the same live destination register. This probe mirrors that shape
 * directly in C/C++ so ptxas has a chance to lower the updates to byte-select
 * P2R forms instead of generic P2R plus shift/or glue.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_insert_b23(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t packed = seed_in[i];

    int a0 = in[i + 0 * 32];
    int a1 = in[i + 1 * 32];
    int a2 = in[i + 2 * 32];
    int a3 = in[i + 3 * 32];
    int b0 = in[i + 4 * 32];
    int b1 = in[i + 5 * 32];
    int b2 = in[i + 6 * 32];
    int b3 = in[i + 7 * 32];

    uint32_t nib2 = ((uint32_t)(a0 > 0))
                  | ((uint32_t)(a1 < 0) << 1)
                  | ((uint32_t)(a2 != 0) << 2)
                  | ((uint32_t)(a3 >= 7) << 3);
    packed = (packed & ~0x000f0000u) | (nib2 << 16);

    uint32_t nib3 = ((uint32_t)(b0 > 3))
                  | ((uint32_t)(b1 <= 1) << 1)
                  | ((uint32_t)(b2 == 9) << 2)
                  | ((uint32_t)(b3 != -5) << 3);
    packed = (packed & ~0x0f000000u) | (nib3 << 24);

    out[i] = packed ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_insert_b1(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t packed = seed_in[i];

    int a0 = in[i + 0 * 32];
    int a1 = in[i + 1 * 32];
    int a2 = in[i + 2 * 32];
    int a3 = in[i + 3 * 32];
    int a4 = in[i + 4 * 32];
    int a5 = in[i + 5 * 32];
    int a6 = in[i + 6 * 32];

    uint32_t mask7 = ((uint32_t)(a0 > 0))
                   | ((uint32_t)(a1 < 0) << 1)
                   | ((uint32_t)(a2 != 0) << 2)
                   | ((uint32_t)(a3 >= 3) << 3)
                   | ((uint32_t)(a4 <= 8) << 4)
                   | ((uint32_t)(a5 == -1) << 5)
                   | ((uint32_t)(a6 != 11) << 6);
    packed = (packed & ~0x00007f00u) | (mask7 << 8);

    out[i] = packed ^ 0x2468ace0u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_insert_b123(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t packed = seed_in[i];

    uint32_t b1 = ((uint32_t)(in[i + 0 * 32] > 0))
                | ((uint32_t)(in[i + 1 * 32] < 0) << 1)
                | ((uint32_t)(in[i + 2 * 32] != 0) << 2)
                | ((uint32_t)(in[i + 3 * 32] >= 3) << 3)
                | ((uint32_t)(in[i + 4 * 32] <= 8) << 4)
                | ((uint32_t)(in[i + 5 * 32] == -1) << 5)
                | ((uint32_t)(in[i + 6 * 32] != 11) << 6);
    packed = (packed & ~0x00007f00u) | (b1 << 8);

    uint32_t b2 = ((uint32_t)(in[i + 7 * 32] > 3))
                | ((uint32_t)(in[i + 8 * 32] <= 1) << 1)
                | ((uint32_t)(in[i + 9 * 32] == 9) << 2)
                | ((uint32_t)(in[i + 10 * 32] != -5) << 3);
    packed = (packed & ~0x000f0000u) | (b2 << 16);

    uint32_t b3 = ((uint32_t)(in[i + 11 * 32] > -7))
                | ((uint32_t)(in[i + 12 * 32] < 4) << 1)
                | ((uint32_t)(in[i + 13 * 32] != 2) << 2)
                | ((uint32_t)(in[i + 14 * 32] >= 6) << 3);
    packed = (packed & ~0x0f000000u) | (b3 << 24);

    out[i] = packed ^ (packed >> 3);
}
