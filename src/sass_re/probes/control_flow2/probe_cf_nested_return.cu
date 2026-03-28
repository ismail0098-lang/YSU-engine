
// Early return from deeply nested control flow
__device__ float nested_compute(float v, int depth) {
    if(depth<=0) return v;
    if(v<0.0f) return -v; // Early return
    float r=nested_compute(v*0.9f,depth-1);
    if(r<0.01f) return 0.0f; // Another early return
    return r+v*0.1f;
}
extern "C" __global__ void __launch_bounds__(128)
cf_nested_return(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=nested_compute(in[i],4);
}
