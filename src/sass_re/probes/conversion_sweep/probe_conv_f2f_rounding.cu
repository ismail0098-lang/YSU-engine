
// F2F (float-to-float) with rounding mode control
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(32)
conv_f2f_round(unsigned short *out, const float *in) {
    int i=threadIdx.x;
    float v=in[i];
    unsigned short rn,rz,rp,rm;
    asm volatile("cvt.rn.f16.f32 %0,%1;":"=h"(rn):"f"(v)); // F2F.F16.F32 round-nearest
    asm volatile("cvt.rz.f16.f32 %0,%1;":"=h"(rz):"f"(v)); // round-zero
    asm volatile("cvt.rp.f16.f32 %0,%1;":"=h"(rp):"f"(v)); // round-positive
    asm volatile("cvt.rm.f16.f32 %0,%1;":"=h"(rm):"f"(v)); // round-minus
    out[i*4]=rn; out[i*4+1]=rz; out[i*4+2]=rp; out[i*4+3]=rm;
}
