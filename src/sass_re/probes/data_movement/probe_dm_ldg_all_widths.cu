
// LDG with every width: 8/16/32/64/128 bit loads
extern "C" __global__ void __launch_bounds__(128)
dm_ldg_widths(void *out, const void *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // 8-bit
    ((unsigned char*)out)[i]=((const unsigned char*)in)[i];
    // 16-bit
    ((unsigned short*)out)[n+i]=((const unsigned short*)in)[i];
    // 32-bit
    ((float*)out)[2*n+i]=((const float*)in)[i];
    // 64-bit
    if(i<n/2) ((double*)out)[3*n+i]=((const double*)in)[i];
    // 128-bit
    if(i<n/4) ((float4*)out)[4*n+i]=((const float4*)in)[i];
}
