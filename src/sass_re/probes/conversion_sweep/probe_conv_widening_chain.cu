
// Widening chain: U8 -> U16 -> U32 -> U64 (each step widens)
extern "C" __global__ void __launch_bounds__(128)
conv_widen(unsigned long long *out, const unsigned char *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    unsigned char u8=in[i];
    unsigned short u16=(unsigned short)u8;        // I2I.U16.U8? or just MOV
    unsigned int u32=(unsigned int)u16;            // I2I.U32.U16?
    unsigned long long u64=(unsigned long long)u32; // I2I.U64.U32?
    out[i]=u64*u64; // Use result to prevent DCE
}
