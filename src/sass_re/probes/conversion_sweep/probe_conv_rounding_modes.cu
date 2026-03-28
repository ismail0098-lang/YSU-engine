
// All rounding modes for F2I conversion
extern "C" __global__ void __launch_bounds__(32)
conv_f2i_all_round(int *out, const float *in) {
    int i=threadIdx.x;
    float v=in[i];
    int rn,rz,rp,rm;
    asm volatile("cvt.rni.s32.f32 %0,%1;":"=r"(rn):"f"(v)); // Round nearest
    asm volatile("cvt.rzi.s32.f32 %0,%1;":"=r"(rz):"f"(v)); // Round toward zero
    asm volatile("cvt.rpi.s32.f32 %0,%1;":"=r"(rp):"f"(v)); // Round toward +inf
    asm volatile("cvt.rmi.s32.f32 %0,%1;":"=r"(rm):"f"(v)); // Round toward -inf
    out[i*4]=rn; out[i*4+1]=rz; out[i*4+2]=rp; out[i*4+3]=rm;
}
