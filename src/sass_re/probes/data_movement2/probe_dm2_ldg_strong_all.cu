
// LDG/STG with ALL scope + ordering variants via PTX
// Requires .relaxed/.acquire/.release ordering with scope qualifiers
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_relaxed_cta(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.relaxed.cta.global.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    asm volatile("st.relaxed.cta.global.f32 [%0], %1;"::"l"(&out[i]),"f"(v*2));
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_relaxed_gpu(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    asm volatile("st.relaxed.gpu.global.f32 [%0], %1;"::"l"(&out[i]),"f"(v*2));
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_relaxed_sys(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.relaxed.sys.global.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    asm volatile("st.relaxed.sys.global.f32 [%0], %1;"::"l"(&out[i]),"f"(v*2));
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_acquire_gpu(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.acquire.gpu.global.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    out[i]=v*2;
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_release_gpu(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2;
    asm volatile("st.release.gpu.global.f32 [%0], %1;"::"l"(&out[i]),"f"(v));
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_acquire_sys(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v;
    asm volatile("ld.acquire.sys.global.f32 %0, [%1];":"=f"(v):"l"(&in[i]));
    out[i]=v*2;
}
extern "C" __global__ void __launch_bounds__(128)
dm2_scope_release_sys(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i]*2;
    asm volatile("st.release.sys.global.f32 [%0], %1;"::"l"(&out[i]),"f"(v));
}
