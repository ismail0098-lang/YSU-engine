
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Full conversion chain: FP64 -> FP32 -> FP16 -> BF16 -> FP32 -> FP64
extern "C" __global__ void __launch_bounds__(128)
conv_full_chain(double *out, const double *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    double d=in[i];
    float f=(float)d;                    // F2F.F32.F64
    __half h=__float2half(f);            // F2FP.F16.F32
    __nv_bfloat16 bf=__float2bfloat16(__half2float(h)); // F16->F32->BF16
    float f2=__bfloat162float(bf);       // PRMT (BF16 decode)
    out[i]=(double)f2;                   // F2F.F64.F32
}
