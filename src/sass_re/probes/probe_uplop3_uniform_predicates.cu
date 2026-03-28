/*
 * SASS RE Probe: uniform predicate fan-in
 *
 * cuDNN library mining surfaced UPLOP3.LUT. This probe keeps all control
 * inputs warp-uniform by sourcing them from kernel arguments only, then
 * combines many uniform predicates so the compiler has a reason to emit
 * uniform predicate logic instead of per-thread PLOP3.LUT.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(128)
probe_uplop3_uniform_predicates(uint32_t *out,
                                uint32_t a,
                                uint32_t b,
                                uint32_t c,
                                uint32_t d,
                                uint32_t e,
                                uint32_t f) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int p0 = ((a & 0x1u) != 0u);
    int p1 = ((b & 0x2u) == 0u);
    int p2 = (c > d);
    int p3 = (e != f);
    int p4 = ((a ^ c) < (b + d));
    int p5 = ((e | f) == 0u);
    int p6 = ((a + b + c) >= (d + e));
    int p7 = ((f & 0x8u) != 0u);

    uint32_t mask = 0;
    if (p0) mask |= 0x01u;
    if (p1) mask |= 0x02u;
    if (p2) mask |= 0x04u;
    if (p3) mask |= 0x08u;
    if (p4) mask |= 0x10u;
    if (p5) mask |= 0x20u;
    if (p6) mask |= 0x40u;
    if (p7) mask |= 0x80u;

    uint32_t select = 0;
    if ((p0 && p1) || (p2 && !p3) || (p4 && p5)) select ^= 0x11u;
    if ((p1 && p6) || (!p2 && p7) || (p0 && p4)) select ^= 0x22u;
    if ((p3 && p5) || (p6 && !p7) || (p2 && p4)) select ^= 0x44u;
    if ((p0 && p7) || (p1 && p2 && p3) || (p4 && !p6)) select ^= 0x88u;

    out[i] = mask ^ select ^ (uint32_t)i;
}
