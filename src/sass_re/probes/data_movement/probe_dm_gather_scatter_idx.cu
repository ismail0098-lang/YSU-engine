
// Gather/scatter with index computation patterns (address arithmetic)
extern "C" __global__ void __launch_bounds__(128)
dm_gather_linear(float *out, const float *in, int stride, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=in[i*stride]; // Linear gather with computed stride
}
extern "C" __global__ void __launch_bounds__(128)
dm_gather_modular(float *out, const float *in, int mod, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i]=in[(i*97+13)%mod]; // Pseudo-random gather (modular hash)
}
extern "C" __global__ void __launch_bounds__(128)
dm_scatter_random(float *out, const float *in, const int *idx, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[idx[i]]=in[i]; // Random scatter
}
