/*
 * SASS RE Probe: strict predicate-byte packing via inline PTX
 *
 * Goal: give ptxas the cleanest possible route to P2R.B1 / P2R.B2 / P2R.B3.
 * Earlier C/C++ probes let the optimizer build the byte lanes through generic
 * shift/or glue. This probe creates predicates directly in inline PTX, packs
 * them with mov.b32, then inserts them into byte lanes 1/2/3 of a live
 * destination register.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b1_inline(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t seed = seed_in[i];
    uint32_t packed = 0;

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .b32 t, b0, b1, b2, b3;\n\t"
        "  setp.gt.s32 p0, %2, 0;\n\t"
        "  setp.lt.s32 p1, %3, 0;\n\t"
        "  setp.ne.s32 p2, %4, 0;\n\t"
        "  setp.ge.s32 p3, %5, 7;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 8;\n\t"
        "  and.b32 %0, %1, 0xffff00ff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "}\n\t"
        : "=&r"(packed)
        : "r"(seed),
          "r"(in[i + 0 * 32]),
          "r"(in[i + 1 * 32]),
          "r"(in[i + 2 * 32]),
          "r"(in[i + 3 * 32]));

    out[i] = packed ^ 0x13579bdfu;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b2_inline(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t seed = seed_in[i];
    uint32_t packed = 0;

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .b32 t, b0, b1, b2, b3;\n\t"
        "  setp.gt.s32 p0, %2, 3;\n\t"
        "  setp.le.s32 p1, %3, 1;\n\t"
        "  setp.eq.s32 p2, %4, 9;\n\t"
        "  setp.ne.s32 p3, %5, -5;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 16;\n\t"
        "  and.b32 %0, %1, 0xff00ffff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "}\n\t"
        : "=&r"(packed)
        : "r"(seed),
          "r"(in[i + 4 * 32]),
          "r"(in[i + 5 * 32]),
          "r"(in[i + 6 * 32]),
          "r"(in[i + 7 * 32]));

    out[i] = packed ^ 0x2468ace0u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b3_inline(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t seed = seed_in[i];
    uint32_t packed = 0;

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .b32 t, b0, b1, b2, b3;\n\t"
        "  setp.gt.s32 p0, %2, -7;\n\t"
        "  setp.lt.s32 p1, %3, 4;\n\t"
        "  setp.ne.s32 p2, %4, 2;\n\t"
        "  setp.ge.s32 p3, %5, 6;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 24;\n\t"
        "  and.b32 %0, %1, 0x00ffffff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "}\n\t"
        : "=&r"(packed)
        : "r"(seed),
          "r"(in[i + 8 * 32]),
          "r"(in[i + 9 * 32]),
          "r"(in[i + 10 * 32]),
          "r"(in[i + 11 * 32]));

    out[i] = packed ^ (packed >> 5);
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_b123_inline(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t packed = seed_in[i];

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .b32 t, b0, b1, b2, b3;\n\t"
        "  setp.gt.s32 p0, %1, 0;\n\t"
        "  setp.lt.s32 p1, %2, 0;\n\t"
        "  setp.ne.s32 p2, %3, 0;\n\t"
        "  setp.ge.s32 p3, %4, 3;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 8;\n\t"
        "  and.b32 %0, %0, 0xffff00ff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "  setp.gt.s32 p0, %5, 3;\n\t"
        "  setp.le.s32 p1, %6, 1;\n\t"
        "  setp.eq.s32 p2, %7, 9;\n\t"
        "  setp.ne.s32 p3, %8, -5;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 16;\n\t"
        "  and.b32 %0, %0, 0xff00ffff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "  setp.gt.s32 p0, %9, -7;\n\t"
        "  setp.lt.s32 p1, %10, 4;\n\t"
        "  setp.ne.s32 p2, %11, 2;\n\t"
        "  setp.ge.s32 p3, %12, 6;\n\t"
        "  selp.b32 b0, 0x1, 0x0, p0;\n\t"
        "  selp.b32 b1, 0x2, 0x0, p1;\n\t"
        "  selp.b32 b2, 0x4, 0x0, p2;\n\t"
        "  selp.b32 b3, 0x8, 0x0, p3;\n\t"
        "  or.b32 t, b0, b1;\n\t"
        "  or.b32 t, t, b2;\n\t"
        "  or.b32 t, t, b3;\n\t"
        "  shl.b32 t, t, 24;\n\t"
        "  and.b32 %0, %0, 0x00ffffff;\n\t"
        "  or.b32  %0, %0, t;\n\t"
        "}\n\t"
        : "+r"(packed)
        : "r"(in[i + 0 * 32]),
          "r"(in[i + 1 * 32]),
          "r"(in[i + 2 * 32]),
          "r"(in[i + 3 * 32]),
          "r"(in[i + 4 * 32]),
          "r"(in[i + 5 * 32]),
          "r"(in[i + 6 * 32]),
          "r"(in[i + 7 * 32]),
          "r"(in[i + 8 * 32]),
          "r"(in[i + 9 * 32]),
          "r"(in[i + 10 * 32]),
          "r"(in[i + 11 * 32]));

    out[i] = packed ^ (packed >> 3);
}
