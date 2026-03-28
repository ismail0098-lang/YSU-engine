
// Loop with continue (distinct SASS from break?)
extern "C" __global__ void __launch_bounds__(128)
cf_continue(float *out, const float *in, float skip_thresh, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float acc=0.0f;
    for(int j=0;j<32;j++){
        float v=in[(i+j)%n];
        if(v<skip_thresh) continue; // CONTINUE -- skip this iteration
        acc+=v;
    }
    out[i]=acc;
}
