
// All CCTL (cache control) variants via PTX
extern "C" __global__ void __launch_bounds__(128)
cctl_prefetch_l1(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    // Prefetch to L1
    if(i+128<n) asm volatile("prefetch.global.L1 [%0];"::"l"(&in[i+128]));
    if(i<n) out[i]=in[i]*2.0f;
}

extern "C" __global__ void __launch_bounds__(128)
cctl_prefetch_l2(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    // Prefetch to L2 only
    if(i+256<n) asm volatile("prefetch.global.L2 [%0];"::"l"(&in[i+256]));
    if(i<n) out[i]=in[i]*2.0f;
}

extern "C" __global__ void __launch_bounds__(128)
cctl_invalidate_l1(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n){
        out[i]=in[i]*2.0f;
        // Cache control: evict line after use
        // Note: 'discard' doesn't exist; use cache-streaming store instead
        __stcs(out+i, out[i]);
    }
}
