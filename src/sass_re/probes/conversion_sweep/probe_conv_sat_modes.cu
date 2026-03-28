
// Saturating conversions (clamp to destination range)
extern "C" __global__ void __launch_bounds__(32)
conv_f2i_sat(int *out, const float *in) {
    int i=threadIdx.x;
    float v=in[i];
    int sat_s32;
    asm volatile("cvt.rni.sat.s32.f32 %0,%1;":"=r"(sat_s32):"f"(v)); // SAT clamp
    out[i*3]=sat_s32;
    unsigned sat_u32;
    asm volatile("cvt.rni.sat.u32.f32 %0,%1;":"=r"(sat_u32):"f"(v)); // Unsigned SAT
    out[i*3+1]=(int)sat_u32;
    unsigned short sat_s16_bits;
    asm volatile(
        "{ .reg .s16 t16;"
        "  cvt.rni.sat.s16.f32 t16, %1;"
        "  mov.b16 %0, t16;"
        "}"
        : "=h"(sat_s16_bits) : "f"(v)); // 16-bit SAT through a real 16-bit PTX reg
    out[i*3+2]=(int)(short)sat_s16_bits;
}
