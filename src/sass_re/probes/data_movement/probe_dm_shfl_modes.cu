
// SHFL with CLAMP vs WRAP mode and variable lane
extern "C" __global__ void __launch_bounds__(32)
dm_shfl_clamp_wrap(float *out, const float *in) {
    int tid=threadIdx.x;
    float val=in[tid];
    // SHFL.DOWN with clamp (out-of-range lanes get own value)
    float down_clamp=__shfl_down_sync(0xFFFFFFFF,val,5);
    // SHFL with computed source lane (variable, not constant)
    int src_lane=(tid*7+3)%32; // Non-trivial computed source
    float indexed=__shfl_sync(0xFFFFFFFF,val,src_lane);
    // SHFL.UP with different delta
    float up1=__shfl_up_sync(0xFFFFFFFF,val,1);
    float up4=__shfl_up_sync(0xFFFFFFFF,val,4);
    float up16=__shfl_up_sync(0xFFFFFFFF,val,16);
    // XOR butterfly with different masks
    float xor1=__shfl_xor_sync(0xFFFFFFFF,val,1);
    float xor3=__shfl_xor_sync(0xFFFFFFFF,val,3);
    float xor7=__shfl_xor_sync(0xFFFFFFFF,val,7);
    float xor15=__shfl_xor_sync(0xFFFFFFFF,val,15);
    float xor31=__shfl_xor_sync(0xFFFFFFFF,val,31);
    out[tid]=down_clamp+indexed+up1+up4+up16+xor1+xor3+xor7+xor15+xor31;
}
