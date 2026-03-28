
// MXINT8 tiled matmul: INT8 elements with per-block FP32 scales
// A[M,K] and B[K,N] each have per-32-element-block scales
extern "C" __global__ void __launch_bounds__(128)
mxint8_tile_dot(float *out, const signed char *a_elems, const float *a_scales,
                const signed char *b_elems, const float *b_scales,
                int n_dots, int K) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n_dots)return;
    
    float acc=0.0f;
    for(int k=0;k<K;k++){
        int a_block=k/32, b_block=k/32;
        float a_scale=a_scales[i*(K/32)+a_block];
        float b_scale=b_scales[i*(K/32)+b_block];
        float a_val=(float)a_elems[i*K+k]*a_scale;
        float b_val=(float)b_elems[i*K+k]*b_scale;
        acc+=a_val*b_val;
    }
    out[i]=acc;
}
