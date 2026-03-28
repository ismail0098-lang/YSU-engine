/*
 * SASS RE Probe: predicate-derived byte vector packing
 *
 * Earlier P2R.B* attempts let ptxas synthesize the byte lanes through generic
 * integer glue. This probe instead creates byte-sized lanes directly from
 * predicates and uses mov.b32 vector packing before merging into a live
 * destination register.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_vector_b1(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t dst = seed_in[i];

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .u16 h0, h1, h2, h3;\n\t"
        "  .reg .b8  b0, b1, b2, b3;\n\t"
        "  .reg .b32 pack;\n\t"
        "  setp.gt.s32 p0, %1, 0;\n\t"
        "  setp.lt.s32 p1, %2, 0;\n\t"
        "  setp.ne.s32 p2, %3, 0;\n\t"
        "  setp.ge.s32 p3, %4, 7;\n\t"
        "  selp.u16 h0, 0x01, 0x00, p0;\n\t"
        "  selp.u16 h1, 0x02, 0x00, p1;\n\t"
        "  selp.u16 h2, 0x04, 0x00, p2;\n\t"
        "  selp.u16 h3, 0x08, 0x00, p3;\n\t"
        "  cvt.u8.u16 b0, h0;\n\t"
        "  cvt.u8.u16 b1, h1;\n\t"
        "  cvt.u8.u16 b2, h2;\n\t"
        "  cvt.u8.u16 b3, h3;\n\t"
        "  mov.b32 pack, {b0, b1, b2, b3};\n\t"
        "  and.b32 %0, %0, 0xffff00ff;\n\t"
        "  and.b32 pack, pack, 0x0000ff00;\n\t"
        "  or.b32  %0, %0, pack;\n\t"
        "}\n\t"
        : "+r"(dst)
        : "r"(in[i + 0 * 32]),
          "r"(in[i + 1 * 32]),
          "r"(in[i + 2 * 32]),
          "r"(in[i + 3 * 32]));

    out[i] = dst ^ 0x51a7d3c9u;
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_vector_b23(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t dst = seed_in[i];

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .u16 h0, h1, h2, h3;\n\t"
        "  .reg .b8  b0, b1, b2, b3;\n\t"
        "  .reg .b32 pack;\n\t"
        "  setp.gt.s32 p0, %1, -2;\n\t"
        "  setp.le.s32 p1, %2, 3;\n\t"
        "  setp.eq.s32 p2, %3, 9;\n\t"
        "  setp.ne.s32 p3, %4, -5;\n\t"
        "  selp.u16 h0, 0x01, 0x00, p0;\n\t"
        "  selp.u16 h1, 0x02, 0x00, p1;\n\t"
        "  selp.u16 h2, 0x04, 0x00, p2;\n\t"
        "  selp.u16 h3, 0x08, 0x00, p3;\n\t"
        "  cvt.u8.u16 b0, h0;\n\t"
        "  cvt.u8.u16 b1, h1;\n\t"
        "  cvt.u8.u16 b2, h2;\n\t"
        "  cvt.u8.u16 b3, h3;\n\t"
        "  mov.b32 pack, {b0, b1, b2, b3};\n\t"
        "  and.b32 %0, %0, 0x0000ffff;\n\t"
        "  and.b32 pack, pack, 0xffff0000;\n\t"
        "  or.b32  %0, %0, pack;\n\t"
        "}\n\t"
        : "+r"(dst)
        : "r"(in[i + 4 * 32]),
          "r"(in[i + 5 * 32]),
          "r"(in[i + 6 * 32]),
          "r"(in[i + 7 * 32]));

    out[i] = dst ^ (dst >> 11);
}

extern "C" __global__ void __launch_bounds__(32)
probe_p2r_vector_b123(uint32_t *out, const int *in, const uint32_t *seed_in) {
    int i = threadIdx.x;
    uint32_t dst = seed_in[i];

    asm volatile(
        "{\n\t"
        "  .reg .pred p0, p1, p2, p3;\n\t"
        "  .reg .u16 h0, h1, h2, h3;\n\t"
        "  .reg .b8  b0, b1, b2, b3;\n\t"
        "  .reg .b32 pack;\n\t"
        "  setp.gt.s32 p0, %1, 0;\n\t"
        "  setp.lt.s32 p1, %2, 0;\n\t"
        "  setp.ne.s32 p2, %3, 0;\n\t"
        "  setp.ge.s32 p3, %4, 7;\n\t"
        "  selp.u16 h0, 0x01, 0x00, p0;\n\t"
        "  selp.u16 h1, 0x02, 0x00, p1;\n\t"
        "  selp.u16 h2, 0x04, 0x00, p2;\n\t"
        "  selp.u16 h3, 0x08, 0x00, p3;\n\t"
        "  cvt.u8.u16 b0, h0;\n\t"
        "  cvt.u8.u16 b1, h1;\n\t"
        "  cvt.u8.u16 b2, h2;\n\t"
        "  cvt.u8.u16 b3, h3;\n\t"
        "  mov.b32 pack, {b0, b1, b2, b3};\n\t"
        "  and.b32 %0, %0, 0x000000ff;\n\t"
        "  and.b32 pack, pack, 0xffffff00;\n\t"
        "  or.b32  %0, %0, pack;\n\t"
        "}\n\t"
        : "+r"(dst)
        : "r"(in[i + 8 * 32]),
          "r"(in[i + 9 * 32]),
          "r"(in[i + 10 * 32]),
          "r"(in[i + 11 * 32]));

    out[i] = dst + 0x10203040u;
}
