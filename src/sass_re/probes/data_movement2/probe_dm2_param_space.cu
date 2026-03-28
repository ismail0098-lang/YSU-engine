
// Parameter space load (kernel arguments via c[] constant bank)
extern "C" __global__ void __launch_bounds__(128)
dm2_param_load(float *out, float a, float b, float c, float d,
               float e, float f, float g, float h, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Many kernel parameters force multiple c[] loads
    out[i]=a*b+c*d+e*f+g*h;
}
