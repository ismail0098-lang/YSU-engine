
// Uniform branch: condition is same for all threads (BRA.U?)
extern "C" __global__ void __launch_bounds__(128)
cf_uniform_branch(float *out, const float *in, int mode, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i];
    // 'mode' is uniform (kernel arg) -- all threads take same branch
    if(mode==0) v=v*2.0f;       // Uniform branch (BRA.U or BRA.CONV?)
    else if(mode==1) v=v+1.0f;
    else if(mode==2) v=v*v;
    else v=-v;
    out[i]=v;
}
