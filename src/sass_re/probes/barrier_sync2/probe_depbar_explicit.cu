
// Explicit DEPBAR via PTX to control memory ordering without full barrier
extern "C" __global__ void __launch_bounds__(128)
depbar_explicit(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n) return;
    // Issue multiple loads
    float v0=in[i], v1=in[i+n], v2=in[i+2*n], v3=in[i+3*n];
    // Wait for first 2 loads to complete before using them
    asm volatile("bar.arrive 0, 128;" ::: "memory");
    float r0=v0+v1;
    // Wait for all loads
    asm volatile("bar.sync 0, 128;" ::: "memory");
    float r1=v2+v3;
    out[i]=r0+r1;
}
