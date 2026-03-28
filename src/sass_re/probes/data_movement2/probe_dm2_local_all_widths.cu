
// Local memory (LMEM) access at all widths (spill patterns)
extern "C" __global__ void __launch_bounds__(32, 16) // Force spills
dm2_local_widths(float *out, const float *in) {
    int tid=threadIdx.x;
    // Force values to local memory by using too many registers
    float a[16];
    double d[4];
    int i[8];
    #pragma unroll
    for(int j=0;j<16;j++) a[j]=in[tid+j*32];
    #pragma unroll
    for(int j=0;j<4;j++) d[j]=(double)a[j*4];
    #pragma unroll
    for(int j=0;j<8;j++) i[j]=(int)(a[j*2]*1000.0f);
    float sum=0;
    #pragma unroll
    for(int j=0;j<16;j++) sum+=a[j];
    for(int j=0;j<4;j++) sum+=(float)d[j];
    for(int j=0;j<8;j++) sum+=(float)i[j];
    out[tid]=sum;
}
