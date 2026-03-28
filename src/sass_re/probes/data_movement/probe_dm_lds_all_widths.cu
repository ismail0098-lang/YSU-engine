
// LDS/STS with every width
extern "C" __global__ void __launch_bounds__(128)
dm_lds_widths(float *out, const float *in) {
    __shared__ char s8[128];
    __shared__ short s16[128];
    __shared__ float s32[128];
    __shared__ double s64[64];
    __shared__ float4 s128[32];
    int tid=threadIdx.x;
    // Store to smem at each width
    s8[tid]=(char)tid;
    s16[tid]=(short)tid;
    s32[tid]=in[tid+blockIdx.x*128];
    if(tid<64) s64[tid]=(double)tid;
    if(tid<32) s128[tid]=make_float4((float)tid,(float)tid,(float)tid,(float)tid);
    __syncthreads();
    // Load from smem at each width
    float sum=(float)s8[tid]+(float)s16[tid]+s32[127-tid];
    if(tid<64) sum+=(float)s64[tid];
    if(tid<32) sum+=s128[tid].x;
    out[tid+blockIdx.x*128]=sum;
}
