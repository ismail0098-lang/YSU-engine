
// BAR.RED emulation: barrier + predicate reduction via __syncthreads_and/or/count
// Ada may or may not emit BAR.RED SASS (compiler may decompose to BAR.SYNC + VOTE)
extern "C" __global__ void __launch_bounds__(256)
bar_red_and(int *out, const float *in, float thresh, int n) {
    int gi=blockIdx.x*256+threadIdx.x;
    int pred=(gi<n)?(in[gi]>thresh):0;
    // __syncthreads_and: barrier + AND reduction
    int result=__syncthreads_and(pred);
    if(threadIdx.x==0) out[blockIdx.x]=result;
}

extern "C" __global__ void __launch_bounds__(256)
bar_red_or(int *out, const float *in, float thresh, int n) {
    int gi=blockIdx.x*256+threadIdx.x;
    int pred=(gi<n)?(in[gi]>thresh):0;
    int result=__syncthreads_or(pred);
    if(threadIdx.x==0) out[blockIdx.x]=result;
}

extern "C" __global__ void __launch_bounds__(256)
bar_red_count(int *out, const float *in, float thresh, int n) {
    int gi=blockIdx.x*256+threadIdx.x;
    int pred=(gi<n)?(in[gi]>thresh):0;
    int count=__syncthreads_count(pred);
    if(threadIdx.x==0) out[blockIdx.x]=count;
}
