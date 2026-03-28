
// STG with ALL cache policy variants via PTX
extern "C" __global__ void __launch_bounds__(128)
dm_stg_policies(float *out_wb, float *out_wt, float *out_cs, float *out_default,
                const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2.0f;
    out_default[i]=v;                                                    // STG.E (default)
    __stcs(out_cs+i,v);                                                  // STG.E.EF
    asm volatile("st.global.wb.f32 [%0],%1;"::"l"(&out_wb[i]),"f"(v)); // STG.E.WB?
    asm volatile("st.global.wt.f32 [%0],%1;"::"l"(&out_wt[i]),"f"(v)); // STG.E.WT?
}
