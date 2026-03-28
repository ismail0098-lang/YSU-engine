
// STG with all cache policy variants
extern "C" __global__ void __launch_bounds__(128)
stg_cache_default(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=in[i]*2.0f; // STG.E (default)
}

extern "C" __global__ void __launch_bounds__(128)
stg_cache_streaming(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2.0f;
    __stcs(out+i,v); // STG.E.EF (evict-first)
}

extern "C" __global__ void __launch_bounds__(128)
stg_cache_writeback(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2.0f;
    asm volatile("st.global.wb.f32 [%0], %1;"::"l"(&out[i]),"f"(v));
}

extern "C" __global__ void __launch_bounds__(128)
stg_cache_writethrough(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2.0f;
    asm volatile("st.global.wt.f32 [%0], %1;"::"l"(&out[i]),"f"(v));
}
