
// Register reuse patterns (.reuse flag in SASS encoding)
// The compiler marks registers with .reuse when the same value
// is used by the next instruction, enabling register file bypass
extern "C" __global__ void __launch_bounds__(128)
dm_register_reuse(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float a=in[i], b=in[i+n];
    // Chain that reuses 'a' multiple times (should trigger .reuse)
    float r0=a*b;      // a used here
    float r1=a*r0;     // a reused from same register
    float r2=a+r1;     // a reused again
    float r3=a-r2;     // a reused again
    out[i]=r3;
}
