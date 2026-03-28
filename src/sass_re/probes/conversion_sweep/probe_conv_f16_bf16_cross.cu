
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Direct FP16 <-> BF16 conversion (cross-format)
extern "C" __global__ void __launch_bounds__(128)
conv_f16_to_bf16(__nv_bfloat16 *out, const __half *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // FP16 -> FP32 -> BF16 (two-step via FP32)
    float v=__half2float(in[i]);
    out[i]=__float2bfloat16(v);
}
extern "C" __global__ void __launch_bounds__(128)
conv_bf16_to_f16(__half *out, const __nv_bfloat16 *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=__bfloat162float(in[i]);
    out[i]=__float2half(v);
}
