
// All INT8 conversion paths
extern "C" __global__ void __launch_bounds__(128)
conv_i8_paths(float *fout, int *iout, const signed char *s8_in,
              const unsigned char *u8_in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    // S8 -> FP32 (I2F.S8 or I2FP.F32.S32 after widening)
    fout[i*4]=(float)s8_in[i];
    // U8 -> FP32 (I2F.U8?)
    fout[i*4+1]=(float)u8_in[i];
    // FP32 -> S8 (F2I + clamp)
    float v=fout[i*4]*0.5f;
    iout[i*2]=(int)max(-128.0f,min(127.0f,v));
    // FP32 -> U8
    iout[i*2+1]=(int)max(0.0f,min(255.0f,v+128.0f));
}
