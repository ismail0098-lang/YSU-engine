
// STG with every width: 8/16/32/64/128 bit stores
extern "C" __global__ void __launch_bounds__(128)
dm_stg_widths(void *out, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    ((unsigned char*)out)[i]=(unsigned char)i;           // STG.E.U8
    ((unsigned short*)out)[n+i]=(unsigned short)i;       // STG.E.U16
    ((int*)out)[2*n+i]=i;                                // STG.E
    if(i<n/2) ((long long*)out)[3*n+i]=(long long)i;    // STG.E.64
    if(i<n/4){float4 v=make_float4((float)i,(float)i,(float)i,(float)i);
              ((float4*)out)[4*n+i]=v;}                  // STG.E.128
}
