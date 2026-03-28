
// All FP64 conversion variants
extern "C" __global__ void __launch_bounds__(128)
conv_f64_all(float *fout, int *iout, long long *llout,
             const double *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    double d=in[i];
    fout[i]=(float)d;              // F2F.F32.F64
    iout[i]=(int)d;                // F2I.S32.F64
    llout[i]=(long long)d;         // F2I.S64.F64
    unsigned u=(unsigned)fabs(d);  // F2I.U32.F64
    iout[i+n]=(int)u;
}
