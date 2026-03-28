
// __ldg with vectorized widths (float2, float4, int4)
extern "C" __global__ void __launch_bounds__(128)
dm_ldg_readonly_vec(float *out, const float *in, const float2 *in2,
                     const float4 *in4, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v1=__ldg(&in[i]);         // LDG.E.CONSTANT (32-bit)
    float2 v2=__ldg(&in2[i]);       // LDG.E.64.CONSTANT?
    float4 v4=__ldg(&in4[i]);       // LDG.E.128.CONSTANT?
    out[i]=v1+v2.x+v2.y+v4.x+v4.y+v4.z+v4.w;
}
