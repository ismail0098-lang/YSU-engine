
// Systematic coalescing test: stride 1, 2, 4, 8, 16, 32
extern "C" __global__ void __launch_bounds__(128)
dm_coalesce_s1(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n) out[i]=in[i]; // Stride 1 (perfect coalescing)
}
extern "C" __global__ void __launch_bounds__(128)
dm_coalesce_s2(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n/2) out[i*2]=in[i*2]; // Stride 2
}
extern "C" __global__ void __launch_bounds__(128)
dm_coalesce_s4(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n/4) out[i*4]=in[i*4]; // Stride 4
}
extern "C" __global__ void __launch_bounds__(128)
dm_coalesce_s32(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n/32) out[i*32]=in[i*32]; // Stride 32 (worst case)
}
