
// MXFP4 with warp-cooperative scale extraction
// All 32 lanes in a warp share the same scale (perfect for MX block_size=32)
__constant__ float FP4_MX_LUT[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};

extern "C" __global__ void __launch_bounds__(128)
mxfp4_tile_warp(float *out, const unsigned char *packed_elems,
                const unsigned char *scales, int n_blocks) {
    int block_id=(threadIdx.x+blockIdx.x*blockDim.x)/32;
    int elem_in_block=(threadIdx.x+blockIdx.x*blockDim.x)%32;
    if(block_id>=n_blocks)return;
    
    // Scale broadcast: lane 0 loads, all lanes read via __shfl
    float scale;
    if(elem_in_block==0){
        unsigned int sb=((unsigned int)scales[block_id])<<23;
        scale=__int_as_float(sb);
    }
    scale=__shfl_sync(0xFFFFFFFF,scale,0); // Broadcast from lane 0
    
    // Decode nibble
    int byte_idx=block_id*16+elem_in_block/2;
    unsigned char b=packed_elems[byte_idx];
    unsigned char nibble=(elem_in_block&1)?((b>>4)&0xF):(b&0xF);
    
    out[block_id*32+elem_in_block]=FP4_MX_LUT[nibble]*scale;
}
