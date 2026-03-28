
// Compare decode throughput: NF4 vs FP4 E2M1 vs INT4 vs MXFP8
// All decode 1M elements, measure relative performance
__constant__ float NF4_CMP[16]={-1,-0.6962f,-0.5251f,-0.3949f,-0.2844f,-0.1848f,-0.09105f,0,
                                 0.07959f,0.1609f,0.2461f,0.3379f,0.4407f,0.5626f,0.7230f,1};
__constant__ float FP4_CMP[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};

extern "C" __global__ void __launch_bounds__(128)
decode_nf4_cmp(float *out, const unsigned char *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i*2]=NF4_CMP[in[i]&0xF];
    out[i*2+1]=NF4_CMP[(in[i]>>4)&0xF];
}

extern "C" __global__ void __launch_bounds__(128)
decode_fp4_cmp(float *out, const unsigned char *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    out[i*2]=FP4_CMP[in[i]&0xF];
    out[i*2+1]=FP4_CMP[(in[i]>>4)&0xF];
}

extern "C" __global__ void __launch_bounds__(128)
decode_int4_cmp(float *out, const unsigned char *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    int lo=in[i]&0xF; if(lo>=8)lo-=16;
    int hi=(in[i]>>4)&0xF; if(hi>=8)hi-=16;
    out[i*2]=(float)lo/14.0f;
    out[i*2+1]=(float)hi/14.0f;
}
