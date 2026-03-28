
// 64-bit warp shuffle (requires 2 SHFL instructions)
extern "C" __global__ void __launch_bounds__(32)
dm_shfl_64(double *out, const double *in) {
    int tid=threadIdx.x;
    double val=in[tid];
    // Shuffle double via two 32-bit shuffles
    int lo=__double2loint(val), hi=__double2hiint(val);
    lo=__shfl_xor_sync(0xFFFFFFFF,lo,1);
    hi=__shfl_xor_sync(0xFFFFFFFF,hi,1);
    out[tid]=__hiloint2double(hi,lo);
}
