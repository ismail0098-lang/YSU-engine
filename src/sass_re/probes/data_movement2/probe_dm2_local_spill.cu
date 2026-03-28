
// Force local memory spill by exceeding register budget
extern "C" __global__ void __launch_bounds__(128, 8) // 8 blocks/SM -> tight regs
dm2_force_spill(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // Allocate enough locals to force spilling
    float v[32];
    #pragma unroll
    for(int j=0;j<32;j++) v[j]=in[(i+j*n)%n];
    float sum=0;
    #pragma unroll
    for(int j=0;j<32;j++) sum+=v[j]*v[(j+1)%32];
    out[i]=sum;
}
