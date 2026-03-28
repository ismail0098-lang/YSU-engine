
// LDG with ALL cache hint variants via PTX
extern "C" __global__ void __launch_bounds__(128)
dm_ldg_policies(float *out, const float *in_def, const float *in_nc,
                const float *in_cs, const float *in_lu, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v0=in_def[i];                                                  // LDG.E (default)
    float v1=__ldg(&in_nc[i]);                                           // LDG.E.CONSTANT
    float v2; asm volatile("ld.global.cs.f32 %0,[%1];":"=f"(v2):"l"(&in_cs[i])); // LDG.E.EF?
    float v3; asm volatile("ld.global.lu.f32 %0,[%1];":"=f"(v3):"l"(&in_lu[i])); // LDG.E.LU?
    out[i]=v0+v1+v2+v3;
}
