
// Tail-call pattern: function calls itself as last operation
__device__ float tail_recurse(float v, int n) {
    if(n<=0) return v;
    return tail_recurse(v*0.99f+0.01f, n-1); // May optimize to loop
}
extern "C" __global__ void __launch_bounds__(128)
cf_tail_call(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=tail_recurse(in[i],10);
}
