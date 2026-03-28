/*
 * SASS RE Probe: byte-qualified predicate rehydrate exact follow-up
 *
 * Goal: keep a packed carrier live, then derive separate predicate groups from
 * three byte slices and use them in later control flow. This aims at the
 * byte-qualified R2P neighborhood seen in cuDNN-mined cubins.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_r2p_byte_rehydrate_exact(uint32_t *out, const uint32_t *in) {
    int lane = threadIdx.x;
    uint32_t carrier = in[lane] ^ 0x00ff0f0fu;

    uint32_t bank0 = (carrier >> 0) & 0x0fu;
    uint32_t bank1 = (carrier >> 8) & 0x0fu;
    uint32_t bank2 = (carrier >> 16) & 0x0fu;
    uint32_t bank3 = (carrier >> 24) & 0x0fu;

    bool p0 = (bank0 & 0x1u) != 0u;
    bool p1 = (bank1 & 0x2u) != 0u;
    bool p2 = (bank2 & 0x4u) != 0u;
    bool p3 = (bank3 & 0x8u) != 0u;

    uint32_t acc = carrier;
    if (p0) acc ^= 0x0000000fu;
    if (p1) acc += 0x000000f0u;
    if (p2) acc ^= 0x00000f00u;
    if (p3) acc += 0x0000f000u;
    if (p0 && p2) acc ^= 0x00f00000u;
    if (p1 || p3) acc += 0x0f000000u;

    out[lane] = acc;
}
