
// Misaligned vectorized loads (force non-aligned LDG patterns)
extern "C" __global__ void __launch_bounds__(128)
dm2_misaligned(float *out, const char *in, int offset, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Load float from non-4-byte-aligned address
    const char *p=in+offset+i*4;
    float v;
    asm volatile("ld.global.f32 %0, [%1];":"=f"(v):"l"(p));
    out[i]=v;
}
