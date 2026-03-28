
// LDG with all cache hint variants
extern "C" __global__ void __launch_bounds__(128)
ldg_cache_default(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=in[i]; // LDG.E (default caching)
}

extern "C" __global__ void __launch_bounds__(128)
ldg_cache_readonly(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=__ldg(&in[i]); // LDG.E.CONSTANT (read-only cache)
}

extern "C" __global__ void __launch_bounds__(128)
ldg_cache_streaming(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.global.cs.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    out[i]=v; // LDG.E.CS? (streaming, evict-first on read)
}

extern "C" __global__ void __launch_bounds__(128)
ldg_cache_last_use(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.global.lu.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    out[i]=v; // LDG.E.LU? (last use, evict after read)
}
