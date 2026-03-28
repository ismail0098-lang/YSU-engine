
#include <cuda_pipeline.h>
// Async copy with all sizes and commit/wait patterns
extern "C" __global__ void __launch_bounds__(128)
dm_async_sizes(float *out, const float *in) {
    __shared__ float s4[128];
    __shared__ float2 s8[64];
    __shared__ float4 s16[32];
    int tid=threadIdx.x;
    // 4-byte async copy
    __pipeline_memcpy_async(&s4[tid],&in[tid],4);
    // 8-byte async copy (if supported)
    if(tid<64) __pipeline_memcpy_async(&s8[tid],(const float2*)&in[tid*2],8);
    // 16-byte async copy
    if(tid<32) __pipeline_memcpy_async(&s16[tid],(const float4*)&in[tid*4],16);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    out[tid]=s4[tid];
    if(tid<64) out[128+tid]=s8[tid].x+s8[tid].y;
    if(tid<32) out[192+tid]=s16[tid].x+s16[tid].y+s16[tid].z+s16[tid].w;
}
