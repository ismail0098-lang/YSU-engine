
// MXFP6 E3M2: 6-bit float (Blackwell format, emulated on Ada)
// E3M2: 1 sign + 3 exponent + 2 mantissa = 64 distinct values
// Packed: 4 elements per 3 bytes (24 bits = 4 * 6 bits)
//
// This is a NOVEL format probe -- MXFP6 is defined by OCP but
// has NO hardware support on any current GPU (Blackwell SM 10.0 only).

// E3M2 decode table (64 entries)
__constant__ float FP6_E3M2_LUT[64]; // Would contain the 64 E3M2 values

extern "C" __global__ void __launch_bounds__(128)
mxfp6_tile_decode(float *out, const unsigned char *packed_3bytes,
                   const unsigned char *scales, int n_blocks) {
    int global_elem=(threadIdx.x+blockIdx.x*blockDim.x);
    int block_id=global_elem/32;
    int elem_in_block=global_elem%32;
    if(block_id>=n_blocks)return;
    
    // Scale
    unsigned int sb=((unsigned int)scales[block_id])<<23;
    float scale=__int_as_float(sb);
    
    // Extract 6-bit element from packed 3-byte groups
    // 4 elements per 3 bytes: [aaaaaabb|bbbbcccc|ccdddddd]
    int group=elem_in_block/4;
    int pos=elem_in_block%4;
    int byte_base=block_id*24+group*3; // 32 elems * 6 bits / 8 = 24 bytes
    unsigned char b0=packed_3bytes[byte_base];
    unsigned char b1=packed_3bytes[byte_base+1];
    unsigned char b2=packed_3bytes[byte_base+2];
    unsigned char val6;
    switch(pos){
        case 0: val6=(b0>>2)&0x3F; break;
        case 1: val6=((b0&0x3)<<4)|((b1>>4)&0xF); break;
        case 2: val6=((b1&0xF)<<2)|((b2>>6)&0x3); break;
        default: val6=b2&0x3F; break;
    }
    
    out[global_elem]=FP6_E3M2_LUT[val6]*scale;
}
