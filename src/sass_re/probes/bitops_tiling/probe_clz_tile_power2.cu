
// Next power of 2 via CLZ
extern "C" __global__ void __launch_bounds__(128)
clz_next_pow2(unsigned *out, const unsigned *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    unsigned v=in[i];
    if(v<=1){out[i]=1;return;}
    int lz=__clz(v-1);
    if(lz==0){out[i]=0x80000000u;return;}
    out[i]=1u<<(32-lz);
}
