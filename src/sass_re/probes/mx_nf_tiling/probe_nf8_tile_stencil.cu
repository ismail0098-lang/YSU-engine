
// NF8 tiled stencil: decode from NF8, apply stencil in FP32, re-encode
__constant__ float NF8_STENCIL_LUT[256];
extern "C" __global__ void __launch_bounds__(128)
nf8_tile_stencil(unsigned char *out, const unsigned char *in, int nx) {
    __shared__ float s[130];
    int tid=threadIdx.x, gi=blockIdx.x*128+tid;
    // Decode NF8 to FP32 via LUT
    if(gi<nx) s[tid+1]=NF8_STENCIL_LUT[in[gi]];
    if(tid==0 && gi>0) s[0]=NF8_STENCIL_LUT[in[gi-1]];
    if(tid==127 && gi+1<nx) s[129]=NF8_STENCIL_LUT[in[gi+1]];
    __syncthreads();
    if(gi>0 && gi<nx-1){
        float result=0.25f*s[tid]+0.5f*s[tid+1]+0.25f*s[tid+2];
        // Simple re-encode: clamp and scale to [0,255]
        out[gi]=(unsigned char)max(0,min(255,(int)((result+3.0f)/6.0f*255.0f)));
    }
}
