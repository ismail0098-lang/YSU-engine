
// Trap instruction (generates BPT or TRAP SASS)
extern "C" __global__ void __launch_bounds__(128)
cf_trap_on_nan(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i];
    // Check for NaN and trap if found
    if(__isnanf(v)){
        asm volatile("trap;"); // Should generate BPT.TRAP or similar
    }
    out[i]=v;
}
