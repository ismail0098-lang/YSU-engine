
#include <cuda_pipeline.h>
// All paths from global to shared: direct, __ldg, cp.async, via registers
extern "C" __global__ void __launch_bounds__(128)
dm_g2s_direct(float *out, const float *in) {
    __shared__ float s[128];
    s[threadIdx.x]=in[threadIdx.x+blockIdx.x*128]; // LDG -> STS
    __syncthreads();
    out[threadIdx.x+blockIdx.x*128]=s[127-threadIdx.x];
}
extern "C" __global__ void __launch_bounds__(128)
dm_g2s_ldg(float *out, const float *in) {
    __shared__ float s[128];
    s[threadIdx.x]=__ldg(&in[threadIdx.x+blockIdx.x*128]); // LDG.CONSTANT -> STS
    __syncthreads();
    out[threadIdx.x+blockIdx.x*128]=s[127-threadIdx.x];
}
extern "C" __global__ void __launch_bounds__(128)
dm_g2s_async(float *out, const float *in) {
    __shared__ float s[128];
    __pipeline_memcpy_async(&s[threadIdx.x],&in[threadIdx.x+blockIdx.x*128],4); // LDGSTS
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    out[threadIdx.x+blockIdx.x*128]=s[127-threadIdx.x];
}
