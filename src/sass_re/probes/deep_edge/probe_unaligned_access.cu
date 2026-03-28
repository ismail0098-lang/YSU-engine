
// Unaligned memory access patterns (may trigger different LDG variants)
extern "C" __global__ void __launch_bounds__(128)
unaligned_byte(float *out, const char *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Load float from unaligned byte address
    float v;
    memcpy(&v, &in[i], sizeof(float)); // Compiler may use LDG.E.U8 x4
    out[i]=v;
}

extern "C" __global__ void __launch_bounds__(128)
unaligned_short(float *out, const short *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Load 2 shorts as float
    float v;
    memcpy(&v, &in[i*2], sizeof(float)); // May use LDG.E.U16 x2
    out[i]=v;
}
