
// Shared memory broadcast patterns (all threads read same address)
extern "C" __global__ void __launch_bounds__(256)
dm_smem_broadcast_single(float *out, const float *in) {
    __shared__ float s;
    if(threadIdx.x==0) s=in[blockIdx.x];
    __syncthreads();
    // ALL threads read same smem address (broadcast, zero conflicts)
    out[threadIdx.x+blockIdx.x*256]=s*(float)(threadIdx.x+1);
}
extern "C" __global__ void __launch_bounds__(256)
dm_smem_broadcast_warp(float *out, const float *in) {
    __shared__ float s[8]; // 1 value per warp
    int warp=threadIdx.x/32;
    if(threadIdx.x%32==0) s[warp]=in[blockIdx.x*8+warp];
    __syncthreads();
    // Each warp reads its own broadcast value
    out[threadIdx.x+blockIdx.x*256]=s[warp]*(float)(threadIdx.x+1);
}
