
// NF4 tiled with shared-memory LUT (faster than constant memory for broadcast)
extern "C" __global__ void __launch_bounds__(128)
nf4_tile_smem(float *out, const unsigned char *packed, int n_bytes) {
    __shared__ float lut[16];
    if(threadIdx.x<16){
        const float nf4[16]={-1.0f,-0.6962f,-0.5251f,-0.3949f,-0.2844f,-0.1848f,-0.09105f,0.0f,
                              0.07959f,0.1609f,0.2461f,0.3379f,0.4407f,0.5626f,0.7230f,1.0f};
        lut[threadIdx.x]=nf4[threadIdx.x];
    }
    __syncthreads();
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n_bytes)return;
    unsigned char b=packed[i];
    out[i*2]=lut[b&0xF];
    out[i*2+1]=lut[(b>>4)&0xF];
}
