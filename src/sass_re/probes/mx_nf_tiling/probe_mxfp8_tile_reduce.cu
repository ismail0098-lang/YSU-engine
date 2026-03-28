
// MXFP8 block-level reduction: decode, sum, encode
extern "C" __global__ void __launch_bounds__(128)
mxfp8_tile_reduce(float *block_sums, const unsigned char *elems,
                   const unsigned char *scales, int n_blocks) {
    int block_id=(threadIdx.x+blockIdx.x*blockDim.x)/32;
    int elem_in_block=(threadIdx.x+blockIdx.x*blockDim.x)%32;
    if(block_id>=n_blocks)return;
    
    unsigned int scale_bits=((unsigned int)scales[block_id])<<23;
    float scale=__int_as_float(scale_bits);
    float val=(float)(int)(signed char)elems[block_id*32+elem_in_block]*scale;
    
    // Warp-level sum (block_size=32=warp_size -> no barrier needed!)
    float sum=val;
    sum+=__shfl_xor_sync(0xFFFFFFFF,sum,16);
    sum+=__shfl_xor_sync(0xFFFFFFFF,sum,8);
    sum+=__shfl_xor_sync(0xFFFFFFFF,sum,4);
    sum+=__shfl_xor_sync(0xFFFFFFFF,sum,2);
    sum+=__shfl_xor_sync(0xFFFFFFFF,sum,1);
    
    if(elem_in_block==0) block_sums[block_id]=sum;
}
